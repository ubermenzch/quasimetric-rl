#!/usr/bin/env python3
"""Queue runner for qrl-official experiments.

Task TSV columns:
    task_id    mode    env_name    seed    steps    extra_args

The runner periodically reconciles status files with actual GPU processes, so
stale RUNNING rows are requeued instead of being shown forever.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRANSIENT_FAILURE_PATTERNS = (
    "CUDA driver initialization failed",
    "CUDA unknown error",
    "Setting the available devices to be zero.",
)
CUDA_OOM_PATTERNS = (
    "CUDA out of memory",
    "torch.OutOfMemoryError",
)
TRANSIENT_LOG_TAIL_BYTES = 1024 * 1024
TASK_MANIFEST_NAME = ".qrl_task.json"


@dataclass
class Task:
    task_id: str
    mode: str
    env_name: str
    seed: str
    steps: str
    extra_args: str = ""


@dataclass
class ActiveJob:
    task: Task
    gpu: str
    proc: subprocess.Popen
    log_file: Path
    output_dir: Path


@dataclass
class GpuState:
    gpu: str
    mem_used_mb: int
    mem_total_mb: int
    util_pct: int


def normalized_task_definition(task: Task) -> dict[str, object]:
    return {
        "task_id": task.task_id,
        "mode": task.mode,
        "env_name": task.env_name,
        "seed": task.seed,
        "steps": task.steps,
        "extra_args": task.extra_args.split(),
    }


def task_fingerprint(task: Task) -> str:
    payload = json.dumps(
        normalized_task_definition(task),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


class QueueLock:
    def __init__(self, path: Path):
        self.path = path
        self.handle = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w")
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.handle.close()
            self.handle = None
            return False
        self.handle.write(f"pid={os.getpid()}\nstarted_at={timestamp()}\n")
        self.handle.flush()
        return True

    def release(self) -> None:
        if self.handle is None:
            return
        fcntl.flock(self.handle, fcntl.LOCK_UN)
        self.handle.close()
        self.handle = None


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def parse_config(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    config: dict[str, str] = {}
    for raw_line in path.read_text(errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        config[key.strip()] = os.path.expandvars(value)
    return config


def cfg(config: dict[str, str], key: str, default: str) -> str:
    return os.environ.get(key, config.get(key, default))


def as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def bool_arg_value(extra_args: list[str], key: str, default: bool) -> bool:
    value = default
    prefix = key + "="
    for arg in extra_args:
        if arg.startswith(prefix):
            value = as_bool(arg[len(prefix):])
    return value


def task_uses_goal_set_objective(task: Task) -> bool:
    """Return whether a task enables either learned or direct goal sets."""
    default = "_GSD_" in task.task_id or "_Direct-" in task.task_id
    return bool_arg_value(
        task.extra_args.split(),
        "agent.goal_set_distance.enabled",
        default,
    )


def as_int(value: str, default: int) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def as_float(value: str, default: float) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def resolve_path(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def resolve_executable(value: str) -> Path:
    """Resolve a configured executable without dereferencing virtualenv links."""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def default_nvidia_library_dir() -> str:
    candidates = [Path("/usr/local/nvidia/lib64"), Path("/usr/lib/nvidia")]
    candidates.extend(sorted(Path("/usr/lib").glob("nvidia-[0-9][0-9][0-9]")))
    for candidate in candidates:
        if candidate.is_dir():
            return str(candidate)
    return ""


def read_tasks(path: Path) -> list[Task]:
    if not path.exists():
        return []
    tasks: list[Task] = []
    with path.open(newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for parts in reader:
            if not parts or not "".join(parts).strip() or parts[0].lstrip().startswith("#"):
                continue
            parts.extend([""] * (6 - len(parts)))
            if len(parts) != 6:
                print(f"[{timestamp()}] WARNING: skipping malformed row: {parts}", flush=True)
                continue
            tasks.append(Task(*parts))
    return tasks


def status_file(status_dir: Path, task_id: str) -> Path:
    return status_dir / f"{task_id}.status"


def read_status(status_dir: Path, task_id: str) -> dict[str, str]:
    data: dict[str, str] = {}
    path = status_file(status_dir, task_id)
    if not path.exists():
        return data
    for line in path.read_text(errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            data[key] = value
    return data


def write_status(status_dir: Path, task: Task, state: str, extra: dict[str, str] | None = None) -> None:
    status_dir.mkdir(parents=True, exist_ok=True)
    previous = read_status(status_dir, task.task_id)
    extra = dict(extra or {})
    now = timestamp()
    for key in (
        "transient_failure_count",
        "submitted_at",
        "gpu_mem_peak_mb",
        "oom_retry_count",
        "oom_last_prelaunch_mem_mb",
        "oom_prelaunch_mem_limit_mb",
        "prelaunch_gpu_mem_used_mb",
    ):
        if previous.get(key) and key not in extra:
            extra[key] = previous[key]
    if not extra.get("submitted_at"):
        extra["submitted_at"] = previous.get("started_at") or previous.get("updated_at") or now
    if state == "RUNNING":
        same_attempt = (
            previous.get("state") == "RUNNING"
            and previous.get("pid")
            and previous.get("pid") == extra.get("pid", previous.get("pid"))
        )
        if not extra.get("started_at"):
            extra["started_at"] = previous.get("started_at") if same_attempt else now
        if not extra.get("gpu_started_at"):
            extra["gpu_started_at"] = (
                previous.get("gpu_started_at") or previous.get("started_at")
                if same_attempt
                else extra["started_at"]
            )
    elif previous.get("started_at") and "started_at" not in extra:
        extra["started_at"] = previous["started_at"]
    if state != "RUNNING" and previous.get("gpu_started_at") and "gpu_started_at" not in extra:
        extra["gpu_started_at"] = previous["gpu_started_at"]
    if state in {"DONE", "FAILED", "PAUSED"} and not extra.get("finished_at"):
        extra["finished_at"] = now
    if state != "RUNNING":
        extra.pop("gpu", None)
        extra.pop("pid", None)
    lines = [
        f"state={state}",
        f"updated_at={now}",
        f"task_id={task.task_id}",
        f"task_fingerprint={task_fingerprint(task)}",
        f"mode={task.mode}",
        f"env_name={task.env_name}",
        f"seed={task.seed}",
        f"steps={task.steps}",
    ]
    if task.extra_args:
        lines.append(f"extra_args={task.extra_args}")
    lines.extend(f"{key}={value}" for key, value in extra.items())
    if state != "RUNNING":
        lines.append("gpu=")
        lines.append("pid=")
    path = status_file(status_dir, task.task_id)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text("\n".join(lines) + "\n")
    tmp_path.replace(path)


STATUS_MANAGED_KEYS = {
    "state", "updated_at", "task_id", "task_fingerprint", "mode",
    "env_name", "seed", "steps", "extra_args",
}


def status_extra_fields(status: dict[str, str]) -> dict[str, str]:
    return {
        key: value
        for key, value in status.items()
        if key not in STATUS_MANAGED_KEYS
    }


def ensure_task_submission_statuses(
    tasks: list[Task],
    status_dir: Path,
    submission_fallback: str = "",
) -> None:
    """Persist when each task first becomes visible to the scheduler."""
    for task in tasks:
        status = read_status(status_dir, task.task_id)
        if status.get("submitted_at"):
            continue
        recorded_fingerprint = status.get("task_fingerprint")
        if recorded_fingerprint and recorded_fingerprint != task_fingerprint(task):
            continue
        if not status:
            write_status(
                status_dir,
                task,
                "PENDING",
                {"submitted_at": submission_fallback or timestamp()},
            )
            continue
        extra = status_extra_fields(status)
        candidates = [
            value
            for value in (
                submission_fallback,
                status.get("started_at", ""),
                status.get("updated_at", ""),
            )
            if value
        ]
        extra["submitted_at"] = min(candidates) if candidates else timestamp()
        write_status(status_dir, task, status.get("state", "PENDING"), extra)


def process_owner(pid: int | str) -> str:
    try:
        return Path(f"/proc/{pid}").owner()
    except Exception:
        return ""


def is_queue_user_process(pid: int | str) -> bool:
    # Ownership is fixed to the operating-system user that starts this runner.
    try:
        return Path(f"/proc/{pid}").stat().st_uid == os.geteuid()
    except OSError:
        return False


def pid_alive(pid: int | str) -> bool:
    text = str(pid)
    return text.isdigit() and Path(f"/proc/{text}").exists()


def termination_grace_seconds(config: dict[str, str]) -> float:
    return max(0.0, as_float(cfg(config, "DISALLOWED_GPU_TERMINATE_GRACE_SECONDS", "10"), 10.0))


def terminate_process(proc: subprocess.Popen, grace_seconds: float) -> str:
    if proc.poll() is not None:
        return f"already_exited:{proc.returncode}"
    try:
        proc.terminate()
    except ProcessLookupError:
        return "not_alive"
    except Exception as exc:
        return f"terminate_error:{exc}"
    try:
        proc.wait(timeout=grace_seconds)
        return f"terminated:{proc.returncode}"
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except ProcessLookupError:
            return "terminated_after_timeout"
        except Exception as exc:
            return f"kill_error:{exc}"
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            return "kill_timeout"
        return f"killed:{proc.returncode}"


def terminate_pid(pid: str, config: dict[str, str], grace_seconds: float) -> str:
    if not pid or not pid.isdigit():
        return "invalid_pid"
    if not pid_alive(pid):
        return "not_alive"
    if not is_queue_user_process(pid):
        return f"owner_mismatch:{process_owner(pid) or 'unknown'}"
    pid_int = int(pid)
    try:
        os.kill(pid_int, signal.SIGTERM)
    except ProcessLookupError:
        return "not_alive"
    except Exception as exc:
        return f"terminate_error:{exc}"
    deadline = time.time() + grace_seconds
    while time.time() < deadline:
        if not pid_alive(pid):
            return "terminated"
        time.sleep(0.2)
    if not pid_alive(pid):
        return "terminated"
    try:
        os.kill(pid_int, signal.SIGKILL)
    except ProcessLookupError:
        return "terminated_after_timeout"
    except Exception as exc:
        return f"kill_error:{exc}"
    deadline = time.time() + 5
    while time.time() < deadline:
        if not pid_alive(pid):
            return "killed"
        time.sleep(0.2)
    return "kill_timeout"


def gpu_uuid_to_index() -> dict[str, str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=10,
        )
    except Exception:
        return {}
    if result.returncode != 0:
        return {}
    mapping: dict[str, str] = {}
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2:
            mapping[parts[1]] = parts[0]
    return mapping


def gpu_compute_apps() -> list[dict[str, str]] | None:
    uuid_to_index = gpu_uuid_to_index()
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=10,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    apps: list[dict[str, str]] = []
    unmapped_gpu_uuids: set[str] = set()
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        gpu_uuid, pid, process_name, used_memory = parts[:4]
        gpu = uuid_to_index.get(gpu_uuid, "")
        if gpu_uuid and not gpu:
            unmapped_gpu_uuids.add(gpu_uuid)
        apps.append({
            "gpu": gpu,
            "gpu_uuid": gpu_uuid,
            "pid": pid,
            "process_name": process_name,
            "used_memory": used_memory,
        })
    if unmapped_gpu_uuids:
        print(
            f"[{timestamp()}] WARNING: cannot map GPU UUID(s) to index: "
            f"{','.join(sorted(unmapped_gpu_uuids))}; skipping GPU app reconciliation.",
            flush=True,
        )
        return None
    return apps


def parse_used_memory_mb(value: str) -> int | None:
    match = re.search(r"\d+(?:\.\d+)?", str(value).replace(",", ""))
    return int(round(float(match.group(0)))) if match else None


def update_running_gpu_memory_peaks(
    tasks: list[Task],
    status_dir: Path,
    apps: list[dict[str, str]] | None,
    dry_run: bool = False,
) -> None:
    """Persist the largest sampled per-process GPU allocation for each task."""
    if apps is None or dry_run:
        return
    memory_by_process: dict[tuple[str, str], int] = {}
    for app in apps:
        pid = app.get("pid", "")
        gpu = app.get("gpu", "")
        used_memory = parse_used_memory_mb(app.get("used_memory", ""))
        if not pid or not gpu or used_memory is None:
            continue
        key = (pid, gpu)
        memory_by_process[key] = memory_by_process.get(key, 0) + used_memory

    for task in tasks:
        status = read_status(status_dir, task.task_id)
        if status.get("state") != "RUNNING":
            continue
        current = memory_by_process.get((status.get("pid", ""), status.get("gpu", "")))
        if current is None:
            continue
        previous_peak = as_int(status.get("gpu_mem_peak_mb", "0"), 0)
        if current <= previous_peak:
            continue
        extra = status_extra_fields(status)
        extra["gpu_mem_peak_mb"] = str(current)
        write_status(status_dir, task, "RUNNING", extra)


def sample_gpu_memory_during_wait(
    config: dict[str, str],
    tasks: list[Task],
    status_dir: Path,
    wait_seconds: float,
    dry_run: bool,
) -> None:
    """Wait for the next allocation pass while sampling GPU memory peaks."""
    wait_seconds = max(0.0, wait_seconds)
    sample_seconds = max(
        0.1,
        as_float(cfg(config, "GPU_MEMORY_SAMPLE_SECONDS", "5"), 5.0),
    )
    deadline = time.monotonic() + wait_seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(sample_seconds, remaining))
        update_running_gpu_memory_peaks(
            tasks,
            status_dir,
            gpu_compute_apps(),
            dry_run=dry_run,
        )


def legal_running_gpu_pids(
    tasks: list[Task],
    status_dir: Path,
    allowed_gpus: list[str],
    apps: list[dict[str, str]] | None,
) -> set[str]:
    if apps is None:
        return set()
    allowed = set(allowed_gpus)
    gpu_by_pid = {
        app["pid"]: app["gpu"]
        for app in apps
        if app.get("pid", "").isdigit()
    }
    legal: set[str] = set()
    for task in tasks:
        status = read_status(status_dir, task.task_id)
        pid = status.get("pid", "").strip()
        expected_gpu = status.get("gpu", "").strip()
        if status.get("state") != "RUNNING" or not pid or expected_gpu not in allowed:
            continue
        if not is_queue_user_process(pid):
            continue
        if gpu_by_pid.get(pid) == expected_gpu:
            legal.add(pid)
    return legal


def kill_illegal_user_gpu_jobs(
    config: dict[str, str],
    apps: list[dict[str, str]] | None,
    legal_pids: set[str],
    dry_run: bool,
) -> bool:
    if apps is None:
        return False
    if not as_bool(cfg(config, "KILL_ILLEGAL_USER_GPU_JOBS", "1")):
        return False
    grace_seconds = termination_grace_seconds(config)
    killed_any = False
    for app in apps:
        gpu = app.get("gpu", "")
        pid = app.get("pid", "")
        if not pid or not pid.isdigit() or pid in legal_pids:
            continue
        if not gpu:
            print(
                f"[{timestamp()}] WARNING: skip killing process with unknown GPU mapping "
                f"pid={pid} process_name={app.get('process_name', '')} "
                f"gpu_uuid={app.get('gpu_uuid', '')}",
                flush=True,
            )
            continue
        if not is_queue_user_process(pid):
            continue
        details = (
            f"gpu={gpu} pid={pid} process_name={app.get('process_name', '')} "
            f"used_memory={app.get('used_memory', '')}"
        )
        if dry_run:
            print(f"[{timestamp()}] DRY_RUN would kill illegal user GPU process {details}", flush=True)
            continue
        result = terminate_pid(pid, config, grace_seconds)
        killed_any = True
        print(
            f"[{timestamp()}] KILL_ILLEGAL_USER_GPU_PROCESS {details} "
            f"terminate_result={result}",
            flush=True,
        )
    return killed_any



def query_gpu_state(gpu: str, apps: list[dict[str, str]] | None) -> GpuState | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                gpu,
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=10,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    line = result.stdout.strip().splitlines()[0].strip() if result.stdout.strip() else ""
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 3:
        return None
    try:
        return GpuState(
            gpu=gpu,
            mem_used_mb=int(float(parts[0])),
            mem_total_mb=int(float(parts[1])),
            util_pct=int(float(parts[2])),
        )
    except ValueError:
        return None


def query_gpu_states(gpus: list[str], apps: list[dict[str, str]] | None) -> dict[str, GpuState]:
    states: dict[str, GpuState] = {}
    for gpu in gpus:
        state = query_gpu_state(gpu, apps)
        if state is not None:
            states[gpu] = state
    return states


def results_root(config: dict[str, str]) -> Path:
    return resolve_path(cfg(config, "RESULTS_ROOT", "../qrl-assets/results/queue"))


def task_output_dir(config: dict[str, str], task: Task) -> Path:
    return results_root(config) / task.task_id


def task_manifest_path(output_dir: Path) -> Path:
    return output_dir / TASK_MANIFEST_NAME


def write_task_manifest(output_dir: Path, task: Task) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = task_manifest_path(output_dir)
    data = {
        "fingerprint": task_fingerprint(task),
        "task": normalized_task_definition(task),
        "updated_at": timestamp(),
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp_path.replace(path)


def task_manifest_fingerprint(output_dir: Path) -> str:
    path = task_manifest_path(output_dir)
    if not path.exists():
        return ""
    try:
        data = json.loads(path.read_text(errors="replace"))
    except (OSError, json.JSONDecodeError):
        return "invalid"
    fingerprint = data.get("fingerprint") if isinstance(data, dict) else None
    return fingerprint if isinstance(fingerprint, str) and fingerprint else "invalid"


def output_complete(output_dir: Path) -> bool:
    return (output_dir / "COMPLETE").exists()


def output_finished(output_dir: Path) -> bool:
    if output_complete(output_dir):
        return True
    return any(output_dir.glob("checkpoint_*_final.pth"))


def completion_evidence(output_dir: Path) -> str:
    if output_complete(output_dir):
        return "COMPLETE"
    final_ckpts = sorted(output_dir.glob("checkpoint_*_final.pth"))
    return final_ckpts[-1].name if final_ckpts else ""


def log_tail(log_file: Path, max_bytes: int = TRANSIENT_LOG_TAIL_BYTES) -> str:
    try:
        with log_file.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            return f.read().decode(errors="replace")
    except OSError:
        return ""


def transient_failure_reason_from_log(config: dict[str, str], log_file: Path) -> str:
    if not as_bool(cfg(config, "REQUEUE_TRANSIENT_FAILURES", "1")):
        return ""
    tail = log_tail(log_file)
    for pattern in TRANSIENT_FAILURE_PATTERNS:
        if pattern in tail:
            return pattern
    return ""


def requeue_transient_status(
    config: dict[str, str],
    task: Task,
    status_dir: Path,
    status: dict[str, str],
    *,
    dry_run: bool = False,
) -> bool:
    """Requeue a confirmed transient failure without adding an OOM limit."""
    log_file = Path(status.get("log_file", ""))
    reason = transient_failure_reason_from_log(config, log_file)
    if not reason:
        return False

    retry_count = as_int(status.get("transient_failure_count", "0"), 0) + 1
    max_retries = as_int(cfg(config, "MAX_TRANSIENT_RETRIES", "10"), 10)
    if retry_count > max_retries:
        return False

    extra = status_extra_fields(status)
    extra.pop("finished_at", None)
    extra.update({
        "exit_code": status.get("exit_code", "1"),
        "previous_gpu": status.get("gpu") or status.get("previous_gpu", ""),
        "previous_pid": status.get("pid") or status.get("previous_pid", ""),
        "error": "transient_failure_requeued",
        "requeue_reason": reason,
        "transient_failure_count": str(retry_count),
    })
    if dry_run:
        print(
            f"[{timestamp()}] DRY_RUN would requeue transient failure {task.task_id}: "
            f"retry={retry_count}/{max_retries} reason={reason}",
            flush=True,
        )
        return True
    write_status(status_dir, task, "PENDING", extra)
    print(
        f"[{timestamp()}] REQUEUE_TRANSIENT_FAILURE {task.task_id} "
        f"retry={retry_count}/{max_retries} reason={reason}",
        flush=True,
    )
    return True


def estimate_competing_gpu_memory_mb(log_text: str) -> int | None:
    """Estimate memory already in use before this process from a PyTorch OOM."""
    patterns = (
        r"total capacity of ([\d.]+) GiB of which ([\d.]+) GiB is free.*?"
        r"Including non-PyTorch memory, this process has ([\d.]+) GiB memory in use",
        r"total capacity of ([\d.]+) GiB of which ([\d.]+) GiB is free.*?"
        r"Process \d+ has ([\d.]+) GiB memory in use",
    )
    for pattern in patterns:
        matches = re.findall(pattern, log_text, flags=re.DOTALL)
        if not matches:
            continue
        total_gib, free_gib, process_gib = map(float, matches[-1])
        return max(0, int(round((total_gib - free_gib - process_gib) * 1024)))
    return None


def requeue_cuda_oom_status(
    config: dict[str, str],
    task: Task,
    status_dir: Path,
    status: dict[str, str],
    *,
    dry_run: bool = False,
) -> bool:
    """Turn a confirmed CUDA OOM into a stricter pending allocation."""
    if not as_bool(cfg(config, "REQUEUE_CUDA_OOM", "1")):
        return False
    log_file = Path(status.get("log_file", ""))
    tail = log_tail(log_file)
    if not any(pattern in tail for pattern in CUDA_OOM_PATTERNS):
        return False

    prelaunch_memory = as_int(status.get("prelaunch_gpu_mem_used_mb", "-1"), -1)
    if prelaunch_memory < 0:
        estimated_memory = estimate_competing_gpu_memory_mb(tail)
        prelaunch_memory = estimated_memory if estimated_memory is not None else -1
    if prelaunch_memory < 0:
        return False

    previous_limit = as_int(status.get("oom_prelaunch_mem_limit_mb", "-1"), -1)
    memory_limit = min(
        value for value in (previous_limit, prelaunch_memory) if value >= 0
    )
    retry_count = as_int(status.get("oom_retry_count", "0"), 0) + 1
    extra = status_extra_fields(status)
    extra.pop("finished_at", None)
    extra.update({
        "exit_code": status.get("exit_code", "1"),
        "previous_gpu": status.get("gpu") or status.get("previous_gpu", ""),
        "previous_pid": status.get("pid") or status.get("previous_pid", ""),
        "error": "cuda_oom_requeued",
        "requeue_reason": "cuda_out_of_memory",
        "oom_retry_count": str(retry_count),
        "oom_last_prelaunch_mem_mb": str(prelaunch_memory),
        "oom_prelaunch_mem_limit_mb": str(memory_limit),
    })
    if dry_run:
        print(
            f"[{timestamp()}] DRY_RUN would requeue CUDA OOM {task.task_id}: "
            f"retry={retry_count} require_prelaunch_mem_below={memory_limit}MB",
            flush=True,
        )
        return True
    write_status(status_dir, task, "PENDING", extra)
    print(
        f"[{timestamp()}] REQUEUE_CUDA_OOM {task.task_id} retry={retry_count} "
        f"last_prelaunch_mem={prelaunch_memory}MB "
        f"require_prelaunch_mem_below={memory_limit}MB",
        flush=True,
    )
    return True


def requeue_existing_cuda_oom_failures(
    config: dict[str, str],
    tasks: list[Task],
    status_dir: Path,
    dry_run: bool,
) -> None:
    for task in tasks:
        status = read_status(status_dir, task.task_id)
        if status.get("state") != "FAILED":
            continue
        if status.get("task_fingerprint") != task_fingerprint(task):
            continue
        requeue_cuda_oom_status(
            config,
            task,
            status_dir,
            status,
            dry_run=dry_run,
        )


def requeue_existing_transient_failures(
    config: dict[str, str],
    tasks: list[Task],
    status_dir: Path,
    dry_run: bool,
) -> None:
    for task in tasks:
        status = read_status(status_dir, task.task_id)
        if status.get("state") != "FAILED":
            continue
        if status.get("task_fingerprint") != task_fingerprint(task):
            continue
        requeue_transient_status(
            config,
            task,
            status_dir,
            status,
            dry_run=dry_run,
        )


def output_started(output_dir: Path) -> bool:
    if not output_dir.exists():
        return False
    try:
        return any(path.name != TASK_MANIFEST_NAME for path in output_dir.iterdir())
    except OSError:
        return True


def task_identity_issue(status: dict[str, str], output_dir: Path, task: Task) -> str:
    expected = task_fingerprint(task)
    if status:
        recorded = status.get("task_fingerprint", "")
        if not recorded:
            return "missing_status_task_fingerprint"
        if recorded != expected:
            return "status_task_definition_mismatch"
    if not output_started(output_dir):
        return ""
    recorded = task_manifest_fingerprint(output_dir)
    if not recorded:
        return "missing_output_task_manifest"
    if recorded == "invalid":
        return "invalid_output_task_manifest"
    if recorded != expected:
        return "output_task_definition_mismatch"
    return ""


def pause_for_identity_issue(
    status_dir: Path,
    task: Task,
    output_dir: Path,
    issue: str,
) -> None:
    write_status(status_dir, task, "PAUSED", {
        "error": issue,
        "output_dir": str(output_dir),
        "output_task_fingerprint": task_manifest_fingerprint(output_dir),
    })
    print(f"[{timestamp()}] PAUSE_TASK_IDENTITY {task.task_id} reason={issue}", flush=True)


def task_terminal(status: dict[str, str], retry_failed: bool) -> bool:
    state = status.get("state", "PENDING")
    return state == "DONE" or state == "PAUSED" or (state == "FAILED" and not retry_failed)


def sync_finished_outputs(config: dict[str, str], tasks: list[Task], status_dir: Path) -> None:
    for task in tasks:
        status = read_status(status_dir, task.task_id)
        output_dir = task_output_dir(config, task)
        issue = task_identity_issue(status, output_dir, task)
        if issue:
            # Do not relabel a process which may still be running under the old
            # definition. Once it exits, normal reconciliation makes it terminal.
            if status.get("state") == "RUNNING":
                continue
            if status.get("state") != "PAUSED" or status.get("error") != issue:
                pause_for_identity_issue(status_dir, task, output_dir, issue)
            continue
        if output_finished(output_dir) and status.get("state") != "DONE":
            write_status(status_dir, task, "DONE", {
                "exit_code": status.get("exit_code", "0"),
                "output_dir": str(output_dir),
                "completed_from_output": "1",
                "completion_evidence": completion_evidence(output_dir),
            })
            print(f"[{timestamp()}] MARK_DONE_FINISHED_OUTPUT {task.task_id}", flush=True)


def clear_non_running_transient_fields(tasks: list[Task], status_dir: Path, dry_run: bool) -> None:
    for task in tasks:
        status = read_status(status_dir, task.task_id)
        state = status.get("state", "")
        if state == "RUNNING" or not state:
            continue
        if not status.get("gpu") and not status.get("pid"):
            continue
        if dry_run:
            print(f"[{timestamp()}] DRY_RUN would clear gpu/pid for {task.task_id}", flush=True)
            continue
        write_status(status_dir, task, state, {
            "exit_code": status.get("exit_code", ""),
            "finished_at": status.get("finished_at", ""),
            "log_file": status.get("log_file", ""),
            "output_dir": status.get("output_dir", ""),
            "error": status.get("error", ""),
            "note": status.get("note", "cleared_non_running_transient_fields"),
        })


def terminate_disallowed_previous_pids(
    config: dict[str, str],
    tasks: list[Task],
    status_dir: Path,
    allowed_gpus: list[str],
    dry_run: bool,
) -> None:
    if not as_bool(cfg(config, "TERMINATE_DISALLOWED_GPU_JOBS", "1")):
        return
    allowed = set(allowed_gpus)
    grace_seconds = termination_grace_seconds(config)
    for task in tasks:
        status = read_status(status_dir, task.task_id)
        state = status.get("state", "")
        if state in {"", "RUNNING", "DONE"}:
            continue
        previous_gpu = status.get("previous_gpu", "")
        previous_pid = status.get("previous_pid", "")
        if not previous_gpu or previous_gpu in allowed or not previous_pid:
            continue
        if status.get("terminate_result"):
            continue
        extra = {
            "exit_code": status.get("exit_code", ""),
            "finished_at": status.get("finished_at", ""),
            "started_at": status.get("started_at", ""),
            "log_file": status.get("log_file", ""),
            "output_dir": status.get("output_dir", ""),
            "error": status.get("error", "disallowed_gpu_previous_pid"),
            "previous_gpu": previous_gpu,
            "actual_gpu": status.get("actual_gpu", ""),
            "previous_pid": previous_pid,
            "note": "terminated_disallowed_previous_pid",
        }
        if dry_run:
            print(f"[{timestamp()}] DRY_RUN would terminate disallowed previous pid {task.task_id}: {extra}", flush=True)
            continue
        extra["terminate_result"] = terminate_pid(previous_pid, config, grace_seconds)
        write_status(status_dir, task, state, extra)
        print(
            f"[{timestamp()}] TERMINATE_DISALLOWED_PREVIOUS_PID {task.task_id} "
            f"previous_gpu={previous_gpu} previous_pid={previous_pid} "
            f"terminate_result={extra['terminate_result']}",
            flush=True,
        )


def reconcile_running_statuses(
    config: dict[str, str],
    tasks: list[Task],
    status_dir: Path,
    allowed_gpus: list[str],
    apps: list[dict[str, str]] | None,
    dry_run: bool,
) -> dict[str, set[str]]:
    running_by_gpu: dict[str, set[str]] = {gpu: set() for gpu in allowed_gpus}
    if not as_bool(cfg(config, "STRICT_GPU_STATE_SYNC", "1")):
        for task in tasks:
            status = read_status(status_dir, task.task_id)
            if status.get("state") == "RUNNING":
                pid = status.get("pid", "")
                if pid and pid_alive(pid) and not is_queue_user_process(pid):
                    extra = {
                        "error": "running_pid_owner_mismatch",
                        "previous_gpu": status.get("gpu", ""),
                        "previous_pid": pid,
                        "actual_owner": process_owner(pid) or "unknown",
                        "log_file": status.get("log_file", ""),
                        "output_dir": status.get("output_dir", ""),
                    }
                    if not dry_run:
                        write_status(status_dir, task, "PAUSED", extra)
                    print(
                        f"[{timestamp()}] PAUSE_FOREIGN_RUNNING_PID {task.task_id} "
                        f"pid={pid} owner={extra['actual_owner']}",
                        flush=True,
                    )
                    continue
                gpu = status.get("gpu", "")
                running_by_gpu.setdefault(gpu, set()).add(task.task_id)
        return running_by_gpu

    clear_non_running_transient_fields(tasks, status_dir, dry_run)
    terminate_disallowed_previous_pids(config, tasks, status_dir, allowed_gpus, dry_run)
    if apps is None:
        print(f"[{timestamp()}] WARNING: cannot query GPU compute apps; skipping scheduling pass.", flush=True)
        return running_by_gpu

    allowed = set(allowed_gpus)
    terminate_disallowed = as_bool(cfg(config, "TERMINATE_DISALLOWED_GPU_JOBS", "1"))
    grace_seconds = termination_grace_seconds(config)
    gpu_by_pid_all = {
        app["pid"]: app["gpu"]
        for app in apps
        if app.get("pid", "").isdigit()
    }
    gpu_by_pid = {
        app["pid"]: app["gpu"]
        for app in apps
        if app.get("gpu") in allowed and app.get("pid", "").isdigit()
    }
    for task in tasks:
        status = read_status(status_dir, task.task_id)
        if status.get("state") != "RUNNING":
            continue
        pid = status.get("pid", "")
        expected_gpu = status.get("gpu", "")
        if pid and pid_alive(pid) and not is_queue_user_process(pid):
            extra = {
                "error": "running_pid_owner_mismatch",
                "previous_gpu": expected_gpu,
                "previous_pid": pid,
                "actual_owner": process_owner(pid) or "unknown",
                "log_file": status.get("log_file", ""),
                "output_dir": status.get("output_dir", ""),
            }
            if dry_run:
                print(f"[{timestamp()}] DRY_RUN would pause foreign RUNNING pid {task.task_id}: {extra}", flush=True)
                continue
            write_status(status_dir, task, "PAUSED", extra)
            print(
                f"[{timestamp()}] PAUSE_FOREIGN_RUNNING_PID {task.task_id} "
                f"pid={pid} owner={extra['actual_owner']}",
                flush=True,
            )
            continue
        actual_gpu_all = gpu_by_pid_all.get(pid, "")
        actual_gpu = gpu_by_pid.get(pid, "")
        if expected_gpu and expected_gpu not in allowed:
            extra = {
                "error": "disallowed_gpu_requeued",
                "previous_gpu": expected_gpu,
                "actual_gpu": actual_gpu_all,
                "previous_pid": pid,
                "log_file": status.get("log_file", ""),
                "output_dir": status.get("output_dir", ""),
            }
            if dry_run:
                print(f"[{timestamp()}] DRY_RUN would requeue disallowed-GPU RUNNING {task.task_id}: {extra}", flush=True)
                continue
            if terminate_disallowed:
                extra["terminate_result"] = terminate_pid(pid, config, grace_seconds)
            else:
                extra["terminate_result"] = "disabled"
            write_status(status_dir, task, "PENDING", extra)
            print(
                f"[{timestamp()}] REQUEUE_DISALLOWED_GPU {task.task_id} "
                f"previous_gpu={expected_gpu} actual_gpu={actual_gpu_all} previous_pid={pid} "
                f"terminate_result={extra['terminate_result']}",
                flush=True,
            )
            continue
        if actual_gpu_all and actual_gpu_all not in allowed:
            extra = {
                "error": "actual_disallowed_gpu_requeued",
                "previous_gpu": expected_gpu,
                "actual_gpu": actual_gpu_all,
                "previous_pid": pid,
                "log_file": status.get("log_file", ""),
                "output_dir": status.get("output_dir", ""),
            }
            if dry_run:
                print(f"[{timestamp()}] DRY_RUN would requeue actual-disallowed-GPU RUNNING {task.task_id}: {extra}", flush=True)
                continue
            if terminate_disallowed:
                extra["terminate_result"] = terminate_pid(pid, config, grace_seconds)
            else:
                extra["terminate_result"] = "disabled"
            write_status(status_dir, task, "PENDING", extra)
            print(
                f"[{timestamp()}] REQUEUE_ACTUAL_DISALLOWED_GPU {task.task_id} "
                f"previous_gpu={expected_gpu} actual_gpu={actual_gpu_all} previous_pid={pid} "
                f"terminate_result={extra['terminate_result']}",
                flush=True,
            )
            continue
        if expected_gpu in allowed and pid and pid_alive(pid):
            if not actual_gpu or actual_gpu == expected_gpu:
                running_by_gpu.setdefault(expected_gpu, set()).add(task.task_id)
                continue

        if requeue_cuda_oom_status(
            config,
            task,
            status_dir,
            status,
            dry_run=dry_run,
        ):
            continue

        extra = {
            "error": "stale_running_status",
            "previous_gpu": expected_gpu,
            "previous_pid": pid,
            "log_file": status.get("log_file", ""),
            "output_dir": status.get("output_dir", ""),
        }
        if dry_run:
            print(f"[{timestamp()}] DRY_RUN would requeue stale RUNNING {task.task_id}: {extra}", flush=True)
            continue
        write_status(status_dir, task, "PENDING", extra)
        print(
            f"[{timestamp()}] REQUEUE_STALE {task.task_id} previous_gpu={expected_gpu} previous_pid={pid}",
            flush=True,
        )
    return running_by_gpu


def evict_disallowed_active_jobs(
    config: dict[str, str],
    active: dict[str, ActiveJob],
    status_dir: Path,
    allowed_gpus: list[str],
    dry_run: bool,
) -> None:
    allowed = set(allowed_gpus)
    terminate_disallowed = as_bool(cfg(config, "TERMINATE_DISALLOWED_GPU_JOBS", "1"))
    grace_seconds = termination_grace_seconds(config)
    for task_id, job in list(active.items()):
        if job.gpu in allowed:
            continue
        extra = {
            "error": "disallowed_gpu_requeued",
            "previous_gpu": job.gpu,
            "previous_pid": str(job.proc.pid),
            "log_file": str(job.log_file),
            "output_dir": str(job.output_dir),
        }
        if dry_run:
            print(f"[{timestamp()}] DRY_RUN would evict active job on disallowed GPU {task_id}: {extra}", flush=True)
            continue
        if terminate_disallowed:
            extra["terminate_result"] = terminate_process(job.proc, grace_seconds)
        else:
            extra["terminate_result"] = "disabled"
        write_status(status_dir, job.task, "PENDING", extra)
        del active[task_id]
        print(
            f"[{timestamp()}] EVICT_ACTIVE_DISALLOWED_GPU {task_id} gpu={job.gpu} "
            f"pid={job.proc.pid} terminate_result={extra['terminate_result']}",
            flush=True,
        )


def external_gpu_busy(
    config: dict[str, str],
    gpu: str,
    apps: list[dict[str, str]] | None,
    running_queue_pids: set[str],
) -> bool:
    if as_bool(cfg(config, "IGNORE_EXTERNAL_GPU_USERS", "0")):
        return False
    if apps is None:
        return True
    for app in apps:
        pid = app.get("pid", "")
        if app.get("gpu") != gpu or not pid:
            continue
        if pid in running_queue_pids:
            continue
        if not is_queue_user_process(pid):
            return True
        if not as_bool(cfg(config, "ALLOW_UNTRACKED_OWN_PROCESSES", "0")):
            return True
    return False


def gpu_accepts_more_jobs(
    config: dict[str, str],
    gpu_state: GpuState | None,
    current_jobs: int,
    apps: list[dict[str, str]] | None,
    running_queue_pids: set[str],
    task: Task | None = None,
    task_status: dict[str, str] | None = None,
) -> bool:
    if gpu_state is None:
        return False
    # A non-positive value explicitly disables the per-GPU job-count limit.
    max_jobs = as_int(cfg(config, "MAX_JOBS_PER_GPU", "8"), 8)
    if max_jobs > 0 and current_jobs >= max_jobs:
        return False
    util_limit = as_float(cfg(config, "GPU_UTIL_LIMIT_PCT", "50"), 50.0)
    if util_limit > 0.0 and gpu_state.util_pct >= util_limit:
        return False
    mem_limit = as_float(cfg(config, "GPU_MEM_LIMIT_PCT", "90"), 90.0)
    mem_pct = 100.0 * gpu_state.mem_used_mb / max(gpu_state.mem_total_mb, 1)
    if mem_pct >= mem_limit:
        return False
    max_used = as_int(cfg(config, "GPU_MAX_USED_MB", "0"), 0)
    if max_used > 0 and gpu_state.mem_used_mb >= max_used:
        return False
    # Goal-set objectives construct a multi-candidate IQE batch. Do not launch
    # learned or direct variants on a GPU that is already substantially used.
    if task is not None and task_uses_goal_set_objective(task):
        gsd_max_used = as_int(cfg(config, "GSD_MAX_PRELAUNCH_MEM_MB", "4000"), 4000)
        if gsd_max_used > 0 and gpu_state.mem_used_mb >= gsd_max_used:
            return False
    if task_status is not None:
        oom_limit = as_int(task_status.get("oom_prelaunch_mem_limit_mb", "-1"), -1)
        if oom_limit >= 0 and gpu_state.mem_used_mb >= oom_limit:
            return False
    if external_gpu_busy(config, gpu_state.gpu, apps, running_queue_pids):
        return False
    return True


def gpu_schedule_key(gpu_state: GpuState, current_jobs: int) -> tuple[int, int, str]:
    return (current_jobs, gpu_state.mem_used_mb, gpu_state.gpu)


def refresh_gpu_state(gpu_states: dict[str, GpuState], gpu: str, apps: list[dict[str, str]] | None) -> None:
    state = query_gpu_state(gpu, apps)
    if state is not None:
        gpu_states[gpu] = state


def gpu_in_start_cooldown(config: dict[str, str], gpu: str, launched_at: dict[str, float]) -> bool:
    cooldown = max(0.0, as_float(cfg(config, "GPU_START_COOLDOWN_SECONDS", "0"), 0.0))
    if cooldown <= 0.0:
        return False
    last_launch = launched_at.get(gpu)
    return last_launch is not None and time.time() - last_launch < cooldown


def choose_pending(
    config: dict[str, str],
    tasks: list[Task],
    active_task_ids: set[str],
    status_dir: Path,
    retry_failed: bool,
    skip_task_ids: set[str] | None = None,
) -> Task | None:
    for task in tasks:
        if task.task_id in active_task_ids or (skip_task_ids and task.task_id in skip_task_ids):
            continue
        status = read_status(status_dir, task.task_id)
        state = status.get("state", "PENDING")
        if state not in {"PENDING", "FAILED"}:
            continue
        if state == "FAILED" and not retry_failed:
            continue
        # Completion evidence is authoritative across scheduler restarts. A
        # stale PENDING/FAILED state must not launch a second run for a task
        # that was already recorded as complete.
        if status.get("completion_evidence") and status.get("task_fingerprint") == task_fingerprint(task):
            write_status(status_dir, task, "DONE", {
                "exit_code": status.get("exit_code", "0"),
                "output_dir": status.get("output_dir", str(task_output_dir(config, task))),
                "completed_from_status": "1",
                "completion_evidence": status.get("completion_evidence", ""),
            })
            print(f"[{timestamp()}] SKIP_COMPLETED_STATUS {task.task_id}", flush=True)
            continue
        output_dir = task_output_dir(config, task)
        issue = task_identity_issue(status, output_dir, task)
        if issue:
            pause_for_identity_issue(status_dir, task, output_dir, issue)
            continue
        if output_finished(output_dir):
            write_status(status_dir, task, "DONE", {
                "exit_code": status.get("exit_code", "0"),
                "output_dir": str(output_dir),
                "completed_from_output": "1",
                "completion_evidence": completion_evidence(output_dir),
            })
            print(f"[{timestamp()}] SKIP_FINISHED_OUTPUT {task.task_id}", flush=True)
            continue
        if output_started(output_dir) and not as_bool(cfg(config, "ALLOW_EXISTING_INCOMPLETE_OUTPUT", "1")):
            write_status(status_dir, task, "PAUSED", {
                "error": "incomplete_output_exists",
                "output_dir": str(output_dir),
            })
            print(f"[{timestamp()}] SKIP_INCOMPLETE_OUTPUT {task.task_id}", flush=True)
            continue
        return task
    return None


def command_env(config: dict[str, str], gpu: str) -> dict[str, str]:
    qrl_dir = resolve_path(cfg(config, "QRL_OFFICIAL_DIR", "."))
    asset_root = resolve_path(cfg(config, "QRL_ASSET_ROOT", "../qrl-assets"))
    dataset_dir = resolve_path(
        cfg(config, "D4RL_DATASET_DIR", str(asset_root / "d4rl/datasets"))
    )
    mujoco_path = resolve_path(
        cfg(config, "MUJOCO_PY_MUJOCO_PATH", str(asset_root / "mujoco/mujoco210"))
    )
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": gpu,
        "EGL_DEVICE_ID": cfg(config, "EGL_DEVICE_ID", "0"),
        "GPUS": cfg(config, "RENDER_GPUS", "0"),
        "MUJOCO_GL": cfg(config, "MUJOCO_GL", "egl"),
        "PYOPENGL_PLATFORM": cfg(config, "PYOPENGL_PLATFORM", "egl"),
        "__GLX_VENDOR_LIBRARY_NAME": cfg(config, "__GLX_VENDOR_LIBRARY_NAME", "nvidia"),
        "NVIDIA_DRIVER_CAPABILITIES": cfg(config, "NVIDIA_DRIVER_CAPABILITIES", "compute,graphics,utility"),
        "D4RL_SUPPRESS_IMPORT_ERROR": cfg(config, "D4RL_SUPPRESS_IMPORT_ERROR", "1"),
        "QRL_ASSET_ROOT": str(asset_root),
        "D4RL_DATASET_DIR": str(dataset_dir),
        "OMP_NUM_THREADS": cfg(config, "OMP_NUM_THREADS", "12"),
        "HYDRA_FULL_ERROR": "1",
        "PYTHONPATH": str(qrl_dir) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
    })
    egl_vendor = cfg(config, "__EGL_VENDOR_LIBRARY_FILENAMES", "")
    if egl_vendor:
        env["__EGL_VENDOR_LIBRARY_FILENAMES"] = egl_vendor
    env["MUJOCO_PY_MUJOCO_PATH"] = str(mujoco_path)
    env["MUJOCO_PATH"] = str(resolve_path(cfg(config, "MUJOCO_PATH", str(mujoco_path))))
    graphics_prefix = resolve_path(
        cfg(config, "QRL_USER_GRAPHICS_PREFIX", str(asset_root / "micromamba/envs/graphics"))
    )
    env["QRL_USER_GRAPHICS_PREFIX"] = str(graphics_prefix)
    graphics_include = graphics_prefix / "include"
    if graphics_include.is_dir():
        env["CPATH"] = str(graphics_include) + (
            os.pathsep + env["CPATH"] if env.get("CPATH") else ""
        )
    graphics_patchelf = graphics_prefix / "bin/patchelf"
    if graphics_patchelf.is_file():
        env["PATH"] = str(graphics_patchelf.parent) + (
            os.pathsep + env["PATH"] if env.get("PATH") else ""
        )
    if "-DGLEW_NO_GLU" not in env.get("CFLAGS", "").split():
        env["CFLAGS"] = " ".join(
            part for part in ("-DGLEW_NO_GLU", env.get("CFLAGS", "")) if part
        )
    ld_paths = [str(mujoco_path / "bin")]
    driver_library_dir = cfg(config, "QRL_DRIVER_LIBRARY_DIR", "") or default_nvidia_library_dir()
    if driver_library_dir:
        ld_paths.append(driver_library_dir)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(ld_paths) + (
        os.pathsep + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
    )
    return env


def build_command(config: dict[str, str], task: Task, gpu: str) -> tuple[list[str], dict[str, str], Path, Path]:
    qrl_dir = resolve_path(cfg(config, "QRL_OFFICIAL_DIR", "."))
    python_bin = resolve_executable(cfg(config, "QRL_PYTHON_BIN", ".venv/bin/python"))
    output_dir = task_output_dir(config, task)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = command_env(config, gpu)
    extra = task.extra_args.split() if task.extra_args.strip() else []
    resume_enabled = bool_arg_value(
        extra,
        "resume_if_possible",
        as_bool(cfg(config, "RESUME_IF_POSSIBLE", "1")),
    )
    common = [
        f"seed={task.seed}",
        f"device.index={cfg(config, 'DEVICE_INDEX', '0')}",
        f"output_base_dir={results_root(config)}",
        f"output_folder={task.task_id}",
        f"overwrite_output={cfg(config, 'OVERWRITE_OUTPUT', 'False')}",
    ]
    if as_bool(cfg(config, "RESUME_IF_POSSIBLE", "1")):
        common.append("resume_if_possible=True")
    if task.mode == "online":
        save_replay_buffer = bool_arg_value(
            extra,
            "save_replay_buffer",
            as_bool(cfg(config, "ONLINE_SAVE_REPLAY_BUFFER", "False")),
        )
        if resume_enabled and not save_replay_buffer:
            raise ValueError(
                "online resume requires effective save_replay_buffer=True "
                "(set ONLINE_SAVE_REPLAY_BUFFER=True and do not override it); "
                "otherwise the replay buffer is empty after checkpoint restore"
            )
        common.append(f"save_replay_buffer={cfg(config, 'ONLINE_SAVE_REPLAY_BUFFER', 'False')}")
        common.append(f"save_final_replay_buffer={cfg(config, 'ONLINE_SAVE_FINAL_REPLAY_BUFFER', 'True')}")
    if task.mode == "online":
        cmd = [
            str(python_bin), "-m", "online.main",
            "env.kind=gcrl",
            f"env.name={task.env_name}",
            f"interaction.total_env_steps={task.steps}",
            *common,
            *extra,
        ]
    elif task.mode == "offline":
        cmd = [
            str(python_bin), "-m", "offline.main",
            "env.kind=d4rl",
            f"env.name={task.env_name}",
            f"total_optim_steps={task.steps}",
            *common,
            *extra,
        ]
    else:
        raise ValueError(f"unsupported mode: {task.mode}")
    return cmd, env, qrl_dir, output_dir


def launch_task(
    config: dict[str, str],
    task: Task,
    gpu: str,
    prelaunch_gpu_mem_used_mb: int,
) -> ActiveJob | None:
    log_dir = resolve_path(cfg(config, "LOG_DIR", "logs/qrl_queue"))
    status_dir = resolve_path(cfg(config, "STATUS_DIR", "runs/qrl_queue/status"))
    log_dir.mkdir(parents=True, exist_ok=True)
    output_dir = task_output_dir(config, task)
    status = read_status(status_dir, task.task_id)
    issue = task_identity_issue(status, output_dir, task)
    if issue:
        pause_for_identity_issue(status_dir, task, output_dir, issue)
        return None
    if output_finished(output_dir):
        write_status(status_dir, task, "DONE", {
            "exit_code": "0",
            "output_dir": str(output_dir),
            "completed_from_output": "1",
            "completion_evidence": completion_evidence(output_dir),
        })
        print(f"[{timestamp()}] SKIP_FINISHED_OUTPUT {task.task_id}", flush=True)
        return None
    cmd, env, cwd, output_dir = build_command(config, task, gpu)
    write_task_manifest(output_dir, task)
    log_file = log_dir / f"{task.task_id}_{time.strftime('%Y%m%d-%H%M%S')}.log"
    with log_file.open("w") as log:
        log.write(f"task_id: {task.task_id}\n")
        log.write(f"cwd: {cwd}\n")
        log.write(f"gpu: {gpu}\n")
        log.write("command: " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
    write_status(status_dir, task, "RUNNING", {
        "gpu": gpu,
        "pid": str(proc.pid),
        "log_file": str(log_file),
        "output_dir": str(output_dir),
        "prelaunch_gpu_mem_used_mb": str(prelaunch_gpu_mem_used_mb),
    })
    print(f"[{timestamp()}] START {task.task_id} mode={task.mode} env={task.env_name} seed={task.seed} gpu={gpu}", flush=True)
    return ActiveJob(task=task, gpu=gpu, proc=proc, log_file=log_file, output_dir=output_dir)


def mark_finished(config: dict[str, str], job: ActiveJob, exit_code: int) -> None:
    status_dir = resolve_path(cfg(config, "STATUS_DIR", "runs/qrl_queue/status"))
    finished = output_finished(job.output_dir)
    previous = read_status(status_dir, job.task.task_id)
    oom_status = dict(previous)
    oom_status["exit_code"] = str(exit_code)
    if not finished and exit_code != 0 and requeue_cuda_oom_status(
        config,
        job.task,
        status_dir,
        oom_status,
    ):
        return
    transient_status = dict(previous)
    transient_status.update({
        "exit_code": str(exit_code),
        "gpu": job.gpu,
        "pid": str(job.proc.pid),
        "log_file": str(job.log_file),
        "output_dir": str(job.output_dir),
        "completion_evidence": completion_evidence(job.output_dir),
    })
    if not finished and exit_code != 0 and requeue_transient_status(
        config,
        job.task,
        status_dir,
        transient_status,
    ):
        return

    state = "DONE" if finished else "FAILED"
    extra = {
        "exit_code": str(exit_code),
        "gpu": job.gpu,
        "pid": str(job.proc.pid),
        "log_file": str(job.log_file),
        "output_dir": str(job.output_dir),
        "completion_evidence": completion_evidence(job.output_dir),
    }
    if exit_code != 0 and not finished:
        extra["error"] = "nonzero_exit"
    elif exit_code == 0 and not finished:
        extra["error"] = "missing_completion_evidence"
    write_status(status_dir, job.task, state, extra)
    print(
        f"[{timestamp()}] {state} {job.task.task_id} exit_code={exit_code} "
        f"completion_evidence={extra['completion_evidence']}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qrl_queue.env")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--sync-only", action="store_true")
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    initial_config = parse_config(config_path)
    lock = QueueLock(resolve_path(cfg(initial_config, "LOCK_FILE", "runs/qrl_queue/qrl_queue.lock")))
    if not lock.acquire():
        print(f"[{timestamp()}] Another qrl queue runner is already active: {lock.path}", flush=True)
        return 2

    active: dict[str, ActiveJob] = {}
    gpu_launched_at: dict[str, float] = {}
    try:
        while True:
            config = parse_config(config_path)
            tasks_path = resolve_path(cfg(config, "TASKS_FILE", "configs/qrl_tasks.tsv"))
            tasks = read_tasks(tasks_path)
            status_dir = resolve_path(cfg(config, "STATUS_DIR", "runs/qrl_queue/status"))
            tasks_modified_at = time.strftime(
                "%Y-%m-%d %H:%M:%S",
                time.localtime(tasks_path.stat().st_mtime),
            ) if tasks_path.exists() else ""
            ensure_task_submission_statuses(
                tasks,
                status_dir,
                submission_fallback=tasks_modified_at,
            )
            allowed_gpus = [gpu for gpu in cfg(config, "GPU_IDS", "0 1 2 3").split() if gpu.strip()]
            retry_failed = as_bool(cfg(config, "RETRY_FAILED", "0"))
            dry_run = as_bool(cfg(config, "DRY_RUN", "0"))
            stop_on_failure = as_bool(cfg(config, "STOP_ON_FAILURE", "0"))
            requeue_existing_cuda_oom_failures(config, tasks, status_dir, dry_run)
            requeue_existing_transient_failures(config, tasks, status_dir, dry_run)

            if args.sync_only:
                sync_finished_outputs(config, tasks, status_dir)
                return 0

            evict_disallowed_active_jobs(config, active, status_dir, allowed_gpus, dry_run)

            for task_id, job in list(active.items()):
                exit_code = job.proc.poll()
                if exit_code is None:
                    continue
                mark_finished(config, job, exit_code)
                del active[task_id]
                finished_state = read_status(status_dir, job.task.task_id).get("state")
                if exit_code != 0 and stop_on_failure and finished_state == "FAILED":
                    return exit_code

            apps = gpu_compute_apps()
            update_running_gpu_memory_peaks(tasks, status_dir, apps, dry_run=dry_run)
            legal_pids = legal_running_gpu_pids(tasks, status_dir, allowed_gpus, apps)
            if kill_illegal_user_gpu_jobs(config, apps, legal_pids, dry_run):
                apps = gpu_compute_apps()
            gpu_states = query_gpu_states(allowed_gpus, apps)
            for gpu in allowed_gpus:
                if gpu not in gpu_states:
                    print(f"[{timestamp()}] WARNING: cannot query GPU {gpu}; treating as busy.", flush=True)

            sync_finished_outputs(config, tasks, status_dir)
            running_by_gpu = reconcile_running_statuses(
                config, tasks, status_dir, allowed_gpus, apps, dry_run)
            active_task_ids = {job.task.task_id for job in active.values()}
            running_queue_pids: set[str] = set()
            for task in tasks:
                status = read_status(status_dir, task.task_id)
                if status.get("state") == "RUNNING":
                    pid = status.get("pid", "").strip()
                    if pid and is_queue_user_process(pid):
                        active_task_ids.add(task.task_id)
                        running_queue_pids.add(pid)

            deferred_task_ids: set[str] = set()
            while True:
                candidates = []
                for gpu in allowed_gpus:
                    if gpu not in gpu_states or gpu_in_start_cooldown(config, gpu, gpu_launched_at):
                        continue
                    current_jobs = len(running_by_gpu.get(gpu, set()))
                    if gpu_accepts_more_jobs(config, gpu_states.get(gpu), current_jobs, apps, running_queue_pids):
                        candidates.append(gpu)
                if not candidates:
                    break
                task = choose_pending(
                    config, tasks, active_task_ids, status_dir, retry_failed, deferred_task_ids
                )
                if task is None:
                    break
                task_status = read_status(status_dir, task.task_id)
                candidates = [
                    gpu for gpu in candidates
                    if gpu_accepts_more_jobs(
                        config,
                        gpu_states.get(gpu),
                        len(running_by_gpu.get(gpu, set())),
                        apps,
                        running_queue_pids,
                        task,
                        task_status,
                    )
                ]
                if not candidates:
                    deferred_task_ids.add(task.task_id)
                    continue
                gpu = min(
                    candidates,
                    key=lambda candidate: gpu_schedule_key(
                        gpu_states[candidate],
                        len(running_by_gpu.get(candidate, set())),
                    ),
                )
                refresh_gpu_state(gpu_states, gpu, apps)
                if not gpu_accepts_more_jobs(
                    config,
                    gpu_states.get(gpu),
                    len(running_by_gpu.get(gpu, set())),
                    apps,
                    running_queue_pids,
                    task,
                    task_status,
                ):
                    deferred_task_ids.add(task.task_id)
                    continue
                if dry_run:
                    print(f"[{timestamp()}] DRY_RUN would launch {task.task_id} on GPU {gpu}", flush=True)
                    active_task_ids.add(task.task_id)
                    running_by_gpu.setdefault(gpu, set()).add(task.task_id)
                    gpu_launched_at[gpu] = time.time()
                    refresh_gpu_state(gpu_states, gpu, apps)
                    continue
                try:
                    job = launch_task(
                        config,
                        task,
                        gpu,
                        gpu_states[gpu].mem_used_mb,
                    )
                    if job is None:
                        active_task_ids.add(task.task_id)
                        continue
                    active[task.task_id] = job
                    active_task_ids.add(task.task_id)
                    running_by_gpu.setdefault(gpu, set()).add(task.task_id)
                    running_queue_pids.add(str(job.proc.pid))
                    gpu_launched_at[gpu] = time.time()
                    refresh_gpu_state(gpu_states, gpu, apps)
                except Exception as exc:
                    print(f"[{timestamp()}] FAILED_TO_LAUNCH {task.task_id}: {exc}", flush=True)
                    write_status(status_dir, task, "FAILED", {"error": f"launch_error:{exc}"})

            terminal_count = 0
            for task in tasks:
                if task_terminal(read_status(status_dir, task.task_id), retry_failed):
                    terminal_count += 1
            if tasks and terminal_count >= len(tasks) and not active:
                print(f"[{timestamp()}] All QRL tasks are terminal.", flush=True)
                return 0

            if args.once:
                return 0
            sample_gpu_memory_during_wait(
                config,
                tasks,
                status_dir,
                as_float(cfg(config, "POLL_SECONDS", "30"), 30.0),
                dry_run,
            )
    finally:
        lock.release()


if __name__ == "__main__":
    raise SystemExit(main())

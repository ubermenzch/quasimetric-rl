#!/usr/bin/env python3
"""Restart current online QRL jobs from their latest committed checkpoints.

Queue-managed jobs are terminated cleanly and left to the queue runner to
requeue. Jobs on GPUs outside the queue's configured GPU set are relaunched
directly with their original argv/environment and resume enabled.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/qrl_queue.env"
DEFAULT_STATUS_DIR = ROOT / "runs/qrl_queue/status"
DEFAULT_BACKUP_DIR = ROOT / "logs/qrl_queue"
CHECKPOINT_RE = re.compile(r"^checkpoint_env(\d+)_opt(\d+)(?:_[^.]*)?\.pth$")


@dataclass
class RestartTarget:
    pid: int
    start_time: str
    task_id: str
    gpu: str
    output_dir: Path
    checkpoint: Path
    checkpoint_env_steps: int
    argv: list[str]
    cwd: Path
    env: dict[str, str]
    queue_managed: bool


def parse_config(path: Path) -> dict[str, str]:
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


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def read_status(path: Path) -> dict[str, str]:
    status: dict[str, str] = {}
    for line in path.read_text(errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            status[key] = value
    return status


def process_argv(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return []
    return [part.decode(errors="replace") for part in raw.split(b"\0") if part]


def process_start_time(pid: int) -> str:
    try:
        suffix = Path(f"/proc/{pid}/stat").read_text().rpartition(") ")[2]
        return suffix.split()[19]
    except (OSError, IndexError):
        return ""


def process_cwd(pid: int) -> Path:
    return Path(os.readlink(f"/proc/{pid}/cwd"))


def process_env(pid: int) -> dict[str, str]:
    raw = Path(f"/proc/{pid}/environ").read_bytes()
    env: dict[str, str] = {}
    for entry in raw.split(b"\0"):
        if b"=" not in entry:
            continue
        key, value = entry.split(b"=", 1)
        env[key.decode(errors="replace")] = value.decode(errors="replace")
    return env


def online_training_argv(argv: list[str]) -> bool:
    return any(
        argv[index:index + 2] == ["-m", "online.main"]
        for index in range(max(0, len(argv) - 1))
    )


def arg_value(argv: list[str], name: str) -> str | None:
    prefix = name + "="
    values = [token[len(prefix):] for token in argv if token.startswith(prefix)]
    return values[-1] if values else None


def resume_argv(argv: list[str]) -> list[str]:
    result = [
        token for token in argv
        if not token.startswith("resume_if_possible=")
    ]
    result.append("resume_if_possible=True")
    return result


def latest_committed_checkpoint(output_dir: Path) -> tuple[Path, int]:
    candidates: list[tuple[int, int, Path]] = []
    for path in output_dir.glob("checkpoint_env*_opt*.pth"):
        match = CHECKPOINT_RE.fullmatch(path.name)
        if match:
            candidates.append((int(match.group(1)), int(match.group(2)), path))
    candidates.sort(reverse=True)
    issues: list[str] = []
    for env_steps, optim_steps, path in candidates:
        try:
            try:
                state = torch.load(
                    path, map_location="cpu", weights_only=False, mmap=True,
                )
            except TypeError:
                state = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            issues.append(f"{path.name}: unreadable ({exc})")
            continue
        issue = checkpoint_issue(state, env_steps, optim_steps)
        if issue is None:
            return path, env_steps
        issues.append(f"{path.name}: {issue}")
    detail = "; ".join(issues[:3]) or "no checkpoint files"
    raise RuntimeError(f"no committed checkpoint in {output_dir}: {detail}")


def checkpoint_issue(
    state: Mapping[str, object], env_steps: int, optim_steps: int,
) -> str | None:
    try:
        payload_cursor = (int(state["env_steps"]), int(state["optim_steps"]))
    except (KeyError, TypeError, ValueError):
        return "invalid payload cursor"
    if payload_cursor != (env_steps, optim_steps):
        return f"payload cursor {payload_cursor} does not match filename"
    if state.get("checkpoint_kind") != "online_committed":
        return f"checkpoint_kind={state.get('checkpoint_kind')!r}"
    validation = state.get("validation_summary")
    if not isinstance(validation, Mapping):
        return "missing validation_summary"
    try:
        validation_cursor = (
            int(validation["env_steps"]), int(validation["optim_steps"]),
        )
    except (KeyError, TypeError, ValueError):
        return "invalid validation cursor"
    if validation_cursor != payload_cursor:
        return f"validation cursor {validation_cursor} does not match payload"
    loop_state = state.get("loop_state")
    if not isinstance(loop_state, Mapping):
        return "missing loop_state"
    try:
        if int(loop_state["cycle_env_steps"]) != env_steps:
            return "checkpoint is not at a complete environment cycle"
        if int(loop_state["next_cycle_sample"]) <= 0:
            return "checkpoint has invalid next_cycle_sample"
    except (KeyError, TypeError, ValueError):
        return "invalid loop_state"
    for key in ("agent", "losses", "rng", "replay"):
        if key not in state:
            return f"missing {key}"
    return None


def verify_process(pid: int, expected_start: str, task_id: str) -> list[str]:
    if Path(f"/proc/{pid}").stat().st_uid != os.geteuid():
        raise RuntimeError(f"PID {pid} is not owned by the current user")
    actual_start = process_start_time(pid)
    if expected_start and actual_start != expected_start:
        raise RuntimeError(f"PID {pid} was reused; refusing to signal it")
    argv = process_argv(pid)
    if not online_training_argv(argv):
        raise RuntimeError(f"PID {pid} is not an online.main training process")
    if arg_value(argv, "output_folder") != task_id:
        raise RuntimeError(f"PID {pid} output_folder does not match {task_id}")
    return argv


def queue_targets(status_dir: Path) -> tuple[list[RestartTarget], list[str]]:
    targets: list[RestartTarget] = []
    skipped: list[str] = []
    for path in sorted(status_dir.glob("*.status")):
        status = read_status(path)
        if status.get("state") != "RUNNING" or status.get("mode") != "online":
            continue
        task_id = status.get("task_id", path.stem)
        try:
            pid = int(status["pid"])
            output_dir = Path(status["output_dir"])
            if (output_dir / "COMPLETE").exists():
                skipped.append(f"{task_id}: already complete")
                continue
            argv = verify_process(pid, "", task_id)
            checkpoint, checkpoint_steps = latest_committed_checkpoint(output_dir)
            targets.append(RestartTarget(
                pid=pid,
                start_time=process_start_time(pid),
                task_id=task_id,
                gpu=status.get("gpu", ""),
                output_dir=output_dir,
                checkpoint=checkpoint,
                checkpoint_env_steps=checkpoint_steps,
                argv=argv,
                cwd=ROOT,
                env={},
                queue_managed=True,
            ))
        except (KeyError, ValueError, OSError, RuntimeError) as exc:
            skipped.append(f"{task_id}: {exc}")
    return targets, skipped


def latest_affinity_backup(backup_dir: Path) -> Path:
    candidates = sorted(
        backup_dir.glob("cpu_affinity_backup_*.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise RuntimeError(f"no CPU-affinity backup found in {backup_dir}")
    return candidates[-1]


def standalone_targets(
    backup_path: Path, queue_gpus: set[str], queue_pids: set[int],
) -> tuple[list[RestartTarget], list[str]]:
    payload = json.loads(backup_path.read_text())
    targets: list[RestartTarget] = []
    skipped: list[str] = []
    for record in payload.get("processes", []):
        pid = int(record["pid"])
        gpus = {str(gpu) for gpu in record.get("gpus", [])}
        if pid in queue_pids or not gpus or not gpus.isdisjoint(queue_gpus):
            continue
        command = str(record.get("command", ""))
        task_match = re.search(r"(?:^| )output_folder=([^ ]+)", command)
        task_id = task_match.group(1) if task_match else f"pid-{pid}"
        try:
            argv = verify_process(pid, str(record.get("start_time", "")), task_id)
            base_dir = arg_value(argv, "output_base_dir")
            cwd = process_cwd(pid)
            output_base = Path(base_dir) if base_dir else ROOT / "online/results"
            if not output_base.is_absolute():
                output_base = cwd / output_base
            output_dir = output_base / task_id
            if (output_dir / "COMPLETE").exists():
                skipped.append(f"{task_id}: already complete")
                continue
            checkpoint, checkpoint_steps = latest_committed_checkpoint(output_dir)
            targets.append(RestartTarget(
                pid=pid,
                start_time=process_start_time(pid),
                task_id=task_id,
                gpu=",".join(sorted(gpus)),
                output_dir=output_dir,
                checkpoint=checkpoint,
                checkpoint_env_steps=checkpoint_steps,
                argv=argv,
                cwd=cwd,
                env=process_env(pid),
                queue_managed=False,
            ))
        except (ValueError, OSError, RuntimeError) as exc:
            skipped.append(f"{task_id}: {exc}")
    return targets, skipped


def process_still_matches(target: RestartTarget) -> bool:
    return process_start_time(target.pid) == target.start_time


def wait_for_exit(targets: list[RestartTarget], timeout: float) -> list[RestartTarget]:
    deadline = time.monotonic() + timeout
    pending = list(targets)
    while pending and time.monotonic() < deadline:
        pending = [target for target in pending if process_still_matches(target)]
        if pending:
            time.sleep(0.5)
    return pending


def relaunch_standalone(target: RestartTarget, threads: int) -> tuple[int, Path]:
    env = dict(target.env)
    value = str(threads)
    env.update({
        "QRL_CPU_THREADS_PER_TASK": value,
        "OMP_NUM_THREADS": value,
        "MKL_NUM_THREADS": value,
        "OPENBLAS_NUM_THREADS": value,
        "NUMEXPR_NUM_THREADS": value,
    })
    if not env.get("CUDA_VISIBLE_DEVICES") and target.gpu:
        env["CUDA_VISIBLE_DEVICES"] = target.gpu
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = target.output_dir / f"restart_stdout_{stamp}.log"
    argv = resume_argv(target.argv)
    with log_path.open("ab") as log:
        proc = subprocess.Popen(
            argv,
            cwd=target.cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return proc.pid, log_path


def write_manifest(records: list[dict[str, object]]) -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = DEFAULT_BACKUP_DIR / f"checkpoint_restart_{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "records": records,
    }, indent=2) + "\n")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--status-dir", type=Path, default=DEFAULT_STATUS_DIR)
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--exit-timeout", type=float, default=180.0)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.threads <= 0 or args.exit_timeout <= 0:
        raise SystemExit("--threads and --exit-timeout must be positive")
    config = parse_config(args.config)
    queue_gpus = set(config.get("GPU_IDS", "0 1 2 3").split())
    queue, skipped = queue_targets(args.status_dir)
    backup = args.backup or latest_affinity_backup(DEFAULT_BACKUP_DIR)
    standalone, standalone_skipped = standalone_targets(
        backup, queue_gpus, {target.pid for target in queue},
    )
    skipped.extend(standalone_skipped)
    targets = queue + standalone

    print(f"Validated {len(queue)} queue and {len(standalone)} standalone targets")
    for target in targets:
        kind = "queue" if target.queue_managed else "standalone"
        print(
            f"  {kind:10s} pid={target.pid:<8d} gpu={target.gpu:<3s} "
            f"checkpoint={target.checkpoint_env_steps:<7d} {target.task_id}"
        )
    for reason in skipped:
        print(f"  SKIP {reason}", file=sys.stderr)
    if not targets:
        print("No live resumable targets found", file=sys.stderr)
        return 1
    if not args.apply:
        print("Dry run only; rerun with --apply to restart these targets")
        return 0

    records: list[dict[str, object]] = []
    signaled: list[RestartTarget] = []
    for target in targets:
        if not process_still_matches(target):
            print(f"SKIP pid={target.pid}: process changed after validation", file=sys.stderr)
            continue
        try:
            os.kill(target.pid, signal.SIGTERM)
        except ProcessLookupError:
            print(f"SKIP pid={target.pid}: process exited before SIGTERM", file=sys.stderr)
            continue
        signaled.append(target)
        print(f"SIGTERM pid={target.pid} {target.task_id}")

    still_running = wait_for_exit(signaled, args.exit_timeout)
    still_running_pids = {target.pid for target in still_running}
    for target in still_running:
        print(
            f"ERROR pid={target.pid} did not exit; it was not force-killed or relaunched",
            file=sys.stderr,
        )

    relaunch_failed = False
    for target in signaled:
        record: dict[str, object] = {
            "task_id": target.task_id,
            "old_pid": target.pid,
            "gpu": target.gpu,
            "checkpoint": str(target.checkpoint),
            "checkpoint_env_steps": target.checkpoint_env_steps,
            "queue_managed": target.queue_managed,
        }
        if target.pid in still_running_pids:
            record["result"] = "exit_timeout"
        elif target.queue_managed:
            record["result"] = "signaled_for_queue_requeue"
        else:
            new_pid, log_path = relaunch_standalone(target, args.threads)
            time.sleep(1.0)
            if process_start_time(new_pid):
                record.update(
                    result="relaunched",
                    new_pid=new_pid,
                    log_file=str(log_path),
                )
                print(
                    f"RELAUNCHED old_pid={target.pid} new_pid={new_pid} "
                    f"gpu={target.gpu} {target.task_id}"
                )
            else:
                relaunch_failed = True
                record.update(result="relaunch_exited", log_file=str(log_path))
                print(
                    f"ERROR relaunch exited immediately: {target.task_id}; see {log_path}",
                    file=sys.stderr,
                )
        records.append(record)

    manifest = write_manifest(records)
    print(f"Restart manifest: {manifest}")
    print("Queue jobs will be relaunched by the running scheduler on its normal poll cycle")
    return 1 if still_running or relaunch_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

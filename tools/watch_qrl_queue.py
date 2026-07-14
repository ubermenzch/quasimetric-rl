#!/usr/bin/env python3
"""Watcher for qrl-official queue status."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from math import ceil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Task:
    task_id: str
    mode: str
    env_name: str
    seed: str
    steps: str
    extra_args: str = ""


@dataclass
class Progress:
    pct: float | None
    text: str
    current: float | None
    total: float | None


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


def cfg(config: dict[str, str], key: str, default: str) -> str:
    return os.environ.get(key, config.get(key, default))


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


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
            if len(parts) == 6:
                tasks.append(Task(*parts))
    return tasks


def read_status(status_dir: Path, task_id: str) -> dict[str, str]:
    data: dict[str, str] = {}
    path = status_dir / f"{task_id}.status"
    if not path.exists():
        return data
    for line in path.read_text(errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            data[key] = value
    return data


def pid_alive(pid: str) -> bool:
    return bool(pid and pid.isdigit() and Path(f"/proc/{pid}").exists())


def parse_timestamp(value: str) -> float | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").timestamp()
    except ValueError:
        return None


def parse_compact_timestamp(value: str) -> float | None:
    try:
        return datetime.strptime(value, "%Y%m%d-%H%M%S").timestamp()
    except ValueError:
        return None


def queue_start_timestamp(config: dict[str, str]) -> float | None:
    task_file = cfg(config, "TASKS_FILE", "")
    match = re.search(r"(\d{8}-\d{6})", task_file)
    return parse_compact_timestamp(match.group(1)) if match else None


def log_start_timestamp(path: Path) -> float | None:
    match = re.search(r"_(\d{8}-\d{6})\.log$", path.name)
    return parse_compact_timestamp(match.group(1)) if match else None


def elapsed_hours(status: dict[str, str]) -> float | None:
    state = status.get("state", "PENDING")
    if state == "PENDING":
        return None
    start = parse_timestamp(status.get("started_at", "")) or parse_timestamp(status.get("updated_at", ""))
    if start is None:
        return None
    end = time.time()
    if state in {"DONE", "FAILED", "PAUSED"}:
        end = parse_timestamp(status.get("finished_at", "")) or parse_timestamp(status.get("updated_at", "")) or end
    return max(0.0, (end - start) / 3600.0)


def same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return str(left) == str(right)


def task_log_files(log_dir: Path, task: Task, queue_started: float | None) -> list[Path]:
    paths = []
    for path in log_dir.glob(f"{task.task_id}_*.log"):
        started = log_start_timestamp(path)
        if queue_started is not None and started is not None and started < queue_started - 60:
            continue
        paths.append(path)
    return sorted(paths, key=lambda p: log_start_timestamp(p) or p.stat().st_mtime)


def attempt_elapsed_hours(path: Path, current_log: Path | None, running: bool) -> float | None:
    started = log_start_timestamp(path)
    if started is None:
        return None
    try:
        end = time.time() if running and current_log is not None and same_path(path, current_log) else path.stat().st_mtime
    except OSError:
        return None
    return max(0.0, (end - started) / 3600.0)


def history_elapsed_hours(
    task: Task,
    status: dict[str, str],
    log_dir: Path,
    queue_started: float | None,
) -> float | None:
    current_log = Path(status.get("log_file", "")) if status.get("log_file") else None
    running = status.get("state") == "RUNNING"
    total = 0.0
    found = False
    for path in task_log_files(log_dir, task, queue_started):
        elapsed = attempt_elapsed_hours(path, current_log, running)
        if elapsed is None:
            continue
        total += elapsed
        found = True
    return total if found else elapsed_hours(status)


def eta_hours(pct: float | None, elapsed_h: float | None) -> float | None:
    if pct is None or elapsed_h is None:
        return None
    if pct <= 0.0 or pct >= 100.0:
        return None
    return elapsed_h * (100.0 - pct) / pct


def current_run_start_progress(task: Task, status: dict[str, str]) -> float | None:
    log_file = Path(status.get("log_file", ""))
    if not log_file.exists():
        return None
    text = read_head(log_file)
    if task.mode == "online":
        matches = re.findall(r"Fast forward to env_steps=(\d+)", text)
        if matches:
            return float(matches[-1])
        matches = re.findall(r"checkpoint_env(\d+)_opt\d+", text)
        if matches:
            return float(matches[-1])
        return 0.0
    matches = re.findall(r"Fast forward to epoch=(\d+)", text)
    if matches:
        return float(matches[-1])
    return 0.0


def estimated_eta_hours(
    task: Task,
    status: dict[str, str],
    progress: Progress,
    run_elapsed_h: float | None,
    history_elapsed_h: float | None,
) -> float | None:
    if progress.current is None or progress.total is None or progress.total <= progress.current:
        return None
    if status.get("state") == "RUNNING" and run_elapsed_h is not None and run_elapsed_h > 0:
        started_progress = current_run_start_progress(task, status)
        if started_progress is not None and progress.current > started_progress:
            speed = (progress.current - started_progress) / run_elapsed_h
            if speed > 0:
                return max(0.0, (progress.total - progress.current) / speed)
    return eta_hours(progress.pct, history_elapsed_h)


def nvidia_smi(gpus: str) -> str:
    rows = []
    for gpu in gpus.split():
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "-i",
                    gpu,
                    "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=5,
            )
        except Exception:
            continue
        if result.returncode == 0 and result.stdout.strip():
            idx, name, used, total, util = [part.strip() for part in result.stdout.strip().split(",", 4)]
            rows.append(f"GPU{idx} {name} mem={used}/{total} MiB util={util}%")
    return "\n".join(rows) if rows else "nvidia-smi failed"


def latest_eval(output_dir: Path) -> dict[str, str]:
    eval_log = output_dir / "eval.log"
    if not eval_log.exists():
        return {}
    best = None
    for line in eval_log.read_text(errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if best is None or float(row.get("env_steps", -1)) >= float(best.get("env_steps", -1)):
            best = row
    if best is None:
        return {}
    out: dict[str, str] = {}
    for key in ["env_steps", "optim_steps", "succ_rate", "epi_return"]:
        if key in best:
            value = best[key]
            out[key] = f"{float(value):.4g}" if isinstance(value, (float, int)) else str(value)
    return out


def online_progress(output_dir: Path, task: Task, status: dict[str, str]) -> Progress:
    best = 0.0
    for path in output_dir.glob("checkpoint_env*_opt*.pth"):
        match = re.search(r"checkpoint_env(\d+)_opt(\d+)", path.name)
        if match:
            best = max(best, float(match.group(1)))
    log_file = Path(status.get("log_file", ""))
    if log_file.exists():
        matches = re.findall(r"(\d+(?:\.\d+)?)\s+env steps", read_tail(log_file))
        if matches:
            best = max(best, float(matches[-1]))
    latest = latest_eval(output_dir)
    if latest.get("env_steps"):
        try:
            best = max(best, float(latest["env_steps"]))
        except ValueError:
            pass
    total = float(task.steps) if str(task.steps).replace(".", "", 1).isdigit() else 0.0
    if total <= 0:
        return Progress(None, "", None, None)
    pct = min(100.0, 100.0 * best / total)
    return Progress(pct, f"{best:.3g}/{total:.3g}" if best else "", best, total)


def read_tail(path: Path, max_bytes: int = 5_000_000) -> str:
    if not path.exists():
        return ""
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
            return f.read().decode("utf-8", "ignore")
    except OSError:
        return ""


def read_head(path: Path, max_bytes: int = 1_000_000) -> str:
    if not path.exists():
        return ""
    try:
        with path.open("rb") as f:
            return f.read(max_bytes).decode("utf-8", "ignore")
    except OSError:
        return ""


def training_phase_count(schedule: str) -> int:
    if schedule == "critic_then_dynamics_then_actor":
        return 3
    if schedule == "critic_then_dynamics_then_goal_set_distance_then_actor":
        return 4
    return 1


def training_phases(schedule: str) -> list[str]:
    if schedule == "critic_then_dynamics_then_actor":
        return ["critic", "latent_dynamics", "actor"]
    if schedule == "critic_then_dynamics_then_goal_set_distance_then_actor":
        return ["critic", "latent_dynamics", "goal_set_distance", "actor"]
    return ["all"]


def task_training_schedule(task: Task, log_text: str) -> str:
    schedule = extra_arg_value(task.extra_args, "agent.training_schedule")
    if schedule:
        return schedule
    match = re.search(r"\btraining_schedule:\s*([A-Za-z0-9_]+)", log_text)
    if match:
        return match.group(1)
    match = re.search(r"Training plan:.*?\bschedule=([A-Za-z0-9_]+)", log_text)
    if match:
        return match.group(1)
    return "joint"


def task_training_phase_count(task: Task, log_text: str) -> int:
    match = re.search(r"Training plan:.*?\bnum_phases=(\d+)", log_text)
    if match:
        return max(1, int(match.group(1)))
    return training_phase_count(task_training_schedule(task, log_text))


def offline_training_plan_total_epochs(log_text: str) -> int:
    matches = re.findall(r"Training plan:.*?\btotal_epochs=(\d+)", log_text)
    return int(matches[-1]) if matches else 0


def offline_phase_progress(log_text: str) -> tuple[str, int, int, int] | None:
    matches = re.findall(
        r"\b([A-Za-z_]+) phase\s+(\d+)/(\d+)\s+epoch\s+(\d+)",
        log_text,
    )
    if not matches:
        return None
    phase_name, phase_step, total_steps, epoch = matches[-1]
    return phase_name, int(phase_step), int(total_steps), int(epoch)


def offline_progress_from_steps(task: Task, log_text: str) -> tuple[float | None, float | None]:
    phase_progress = offline_phase_progress(log_text)
    if phase_progress is None:
        return None, None
    phase_name, phase_step, total_steps, _epoch = phase_progress
    if total_steps <= 0:
        return None, None
    phase_names = training_phases(task_training_schedule(task, log_text))
    phase_index = phase_names.index(phase_name) if phase_name in phase_names else 0
    num_phases = max(task_training_phase_count(task, log_text), phase_index + 1)
    current = min(float(total_steps * num_phases), float(phase_index * total_steps + phase_step))
    total = float(total_steps * num_phases)
    return current, total


def offline_total_epochs(task: Task, log_text: str, best_epoch: int, state: str) -> int:
    total_epochs = offline_training_plan_total_epochs(log_text)
    if total_epochs > 0:
        return total_epochs

    phase_progress = offline_phase_progress(log_text)
    total_steps = phase_progress[2] if phase_progress is not None else 0
    batch_totals = []
    for _current, total in re.findall(r"\|\s*(\d+)/(\d+)\s*\[", log_text):
        batch_totals.append(int(total))
    num_batches = max(batch_totals) if batch_totals else 0
    if total_steps > 0 and num_batches > 0:
        return ceil(total_steps * task_training_phase_count(task, log_text) / num_batches)

    if state == "DONE" and best_epoch > 0:
        return best_epoch + 1
    return 0


def offline_progress(output_dir: Path, task: Task, status: dict[str, str]) -> Progress:
    best_epoch = 0
    total_epochs = 0
    log_file = Path(status.get("log_file", ""))
    text = ""
    if log_file.exists():
        text = read_tail(log_file)
        matches = re.findall(r"Train epoch\s+(\d+)/(\d+)", text)
        if matches:
            epoch, total = matches[-1]
            best_epoch = max(best_epoch, int(epoch))
            total_epochs = max(total_epochs, int(total))
    for path in output_dir.glob("checkpoint_*.pth"):
        match = re.search(r"checkpoint_(\d{5})_(\d{5})", path.name)
        if match:
            best_epoch = max(best_epoch, int(match.group(1)))
    if best_epoch <= 0:
        return Progress(None, "", None, None)
    if total_epochs <= 0:
        total_epochs = offline_total_epochs(task, text, best_epoch, status.get("state", "PENDING"))
    if total_epochs > 0:
        if status.get("state") == "DONE":
            pct = 100.0
            current = float(total_epochs)
            text_progress = f"{total_epochs}/{total_epochs}"
        else:
            current = float(best_epoch)
            step_current, step_total = offline_progress_from_steps(task, text)
            if step_current is not None and step_total is not None and step_total > 0:
                pct = min(100.0, 100.0 * step_current / step_total)
            else:
                pct = min(100.0, 100.0 * best_epoch / total_epochs)
            text_progress = f"{best_epoch}/{total_epochs}"
        return Progress(pct, text_progress, current, float(total_epochs))
    return Progress(None, str(best_epoch), float(best_epoch), None)


def fmt(value: object, width: int) -> str:
    text = "" if value is None else str(value)
    return text[:width].ljust(width)


def extra_arg_value(extra_args: str, key: str) -> str:
    prefix = key + "="
    for arg in extra_args.split():
        if arg.startswith(prefix):
            return arg[len(prefix):]
    return ""


def task_variant(task: Task) -> str:
    if "CDA_Tr" in task.task_id:
        return "CDA-Tr"
    if "_GSD_" in task.task_id:
        return "GSD"
    if "_Tmse_" in task.task_id:
        return "Tmse"
    if "_A-latent_" in task.task_id or "_Alatent_" in task.task_id:
        return "A-latent"
    if "_Sonly_" in task.task_id:
        return "Sonly"
    if "TsepTr" in task.task_id:
        return "TsepTr"
    if "_Tr_" in task.task_id:
        return "Tr"
    if "Tsep" in task.task_id:
        return "Tsep"
    if "_Base_" in task.task_id:
        return "Base"
    return "base"


def task_history_length(task: Task) -> str:
    value = extra_arg_value(task.extra_args, "agent.quasimetric_critic.model.latent_dynamics.history_length")
    if value:
        return value
    match = re.search(r"_h(\d+)_", task.task_id)
    if match:
        return match.group(1)
    return "0"


def render(config: dict[str, str]) -> None:
    tasks = read_tasks(resolve_path(cfg(config, "TASKS_FILE", "configs/qrl_tasks.tsv")))
    status_dir = resolve_path(cfg(config, "STATUS_DIR", "runs/qrl_queue/status"))
    output_root = resolve_path(cfg(config, "RESULTS_ROOT", "online/results_queue"))
    log_dir = resolve_path(cfg(config, "LOG_DIR", "logs/qrl_queue"))
    queue_started = queue_start_timestamp(config)
    gpus = cfg(config, "GPU_IDS", "0 1 2 3")
    print(time.strftime("%Y-%m-%d %H:%M:%S"))
    print(nvidia_smi(gpus))

    rows = []
    counts: dict[str, int] = {}
    total_elapsed = 0.0
    queue_eta = 0.0
    for idx, task in enumerate(tasks, start=1):
        status = read_status(status_dir, task.task_id)
        state = status.get("state", "PENDING")
        pid = status.get("pid", "")
        alive = pid_alive(pid)
        if state == "RUNNING" and not alive:
            state = "STALE"
        output_dir = Path(status.get("output_dir") or str(output_root / task.task_id))
        if task.mode == "online":
            progress = online_progress(output_dir, task, status)
        else:
            progress = offline_progress(output_dir, task, status)
        run_elapsed = elapsed_hours(status)
        history_elapsed = history_elapsed_hours(task, status, log_dir, queue_started)
        display_elapsed = history_elapsed if history_elapsed is not None else run_elapsed
        if display_elapsed is not None:
            total_elapsed += display_elapsed
        eta = estimated_eta_hours(task, status, progress, run_elapsed, history_elapsed)
        if state in {"RUNNING", "PENDING"} and eta is not None:
            queue_eta = max(queue_eta, eta)
        latest = latest_eval(output_dir)
        succ = latest.get("succ_rate", "")
        err = status.get("error", "")
        rows.append([
            idx,
            state,
            status.get("gpu", "") if state == "RUNNING" else "",
            pid if state == "RUNNING" else "",
            task.task_id,
            task_variant(task),
            task_history_length(task),
            task.mode,
            task.env_name,
            task.seed,
            f"{progress.pct:.1f}" if progress.pct is not None else "",
            f"{eta:.2f}" if eta is not None else "",
            f"{display_elapsed:.2f}" if display_elapsed is not None else "",
            progress.text,
            succ,
            err,
        ])
        counts[state] = counts.get(state, 0) + 1

    print(
        "Status:",
        " ".join(f"{k}={v}" for k, v in sorted(counts.items())),
        f"task_elapsed_h={total_elapsed:.2f}",
        f"queue_eta_h={queue_eta:.2f}",
    )
    headers = [
        "#", "state", "gpu", "pid", "task", "variant", "hist", "mode", "env", "seed",
        "%", "eta_h", "elapsed_h", "progress", "succ", "err",
    ]
    widths = [4, 8, 4, 8, 44, 8, 4, 7, 18, 6, 6, 7, 9, 17, 8, 18]
    print(" ".join(fmt(h, w) for h, w in zip(headers, widths)))
    print(" ".join("-" * w for w in widths))
    for row in rows:
        print(" ".join(fmt(v, w) for v, w in zip(row, widths)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qrl_queue.env")
    parser.add_argument("--watch", type=float, default=None)
    parser.add_argument(
        "--clear-screen",
        action="store_true",
        help="Clear the terminal before each refresh. Default is append-only output.",
    )
    args = parser.parse_args()
    config = parse_config(resolve_path(args.config))
    watch = args.watch
    if watch is None:
        watch = float(cfg(config, "WATCH_SECONDS", cfg(config, "POLL_SECONDS", "30")))
    while True:
        if args.clear_screen:
            print("\033[2J\033[H", end="")
        render(config)
        if watch <= 0:
            return 0
        time.sleep(watch)
        print()


if __name__ == "__main__":
    raise SystemExit(main())

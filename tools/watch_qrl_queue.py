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
from functools import lru_cache
from math import ceil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS_DELETED_MARKER = "CHECKPOINTS_DELETED"


@dataclass
class Task:
    task_id: str
    mode: str
    env_name: str
    seed: str
    steps: str
    params: str = ""
    extra_args: str = ""


@dataclass
class Progress:
    pct: float | None
    text: str
    current: float | None
    total: float | None


class RecentProgressHistory:
    """Persistent progress samples used for recent-window ETA estimates."""

    def __init__(self, path: Path):
        self.path = path
        self.tasks: dict[str, dict[str, Any]] = {}
        try:
            payload = json.loads(path.read_text())
            if payload.get("version") == 1 and isinstance(payload.get("tasks"), dict):
                self.tasks = payload["tasks"]
        except (OSError, json.JSONDecodeError, AttributeError):
            pass

    def observe(
        self,
        task_id: str,
        attempt: str,
        progress: float,
        now: float,
        window_progress: float,
    ) -> list[tuple[float, float]]:
        record = self.tasks.get(task_id)
        if (
            not isinstance(record, dict)
            or record.get("attempt") != attempt
            or not isinstance(record.get("samples"), list)
        ):
            record = {"attempt": attempt, "samples": []}
            self.tasks[task_id] = record

        samples = []
        for sample in record["samples"]:
            if not isinstance(sample, list) or len(sample) != 2:
                continue
            try:
                timestamp, value = float(sample[0]), float(sample[1])
            except (TypeError, ValueError):
                continue
            if timestamp <= now:
                samples.append((timestamp, value))
        samples.sort()
        compressed = []
        for sample in samples:
            if compressed and sample[1] == compressed[-1][1]:
                continue
            compressed.append(sample)
        samples = compressed
        if samples and progress < samples[-1][1]:
            samples = []
        if not samples or progress > samples[-1][1]:
            samples.append((now, progress))

        cutoff = progress - max(0.0, window_progress)
        first_recent = next(
            (index for index, sample in enumerate(samples) if sample[1] >= cutoff),
            len(samples) - 1,
        )
        # Keep one sample immediately before the cutoff so sparse progress
        # updates still provide a usable speed estimate.
        first_recent = max(0, first_recent - 1)
        recent = samples[first_recent:]
        record["samples"] = [[timestamp, value] for timestamp, value in recent]

        estimate_samples = list(recent)
        if estimate_samples and now > estimate_samples[-1][0]:
            estimate_samples.append((now, progress))
        return estimate_samples

    def retain(self, active_task_ids: set[str]) -> None:
        self.tasks = {
            task_id: record
            for task_id, record in self.tasks.items()
            if task_id in active_task_ids
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(
            self.path.suffix + f".{os.getpid()}.tmp"
        )
        tmp_path.write_text(json.dumps({"version": 1, "tasks": self.tasks}) + "\n")
        tmp_path.replace(self.path)


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
            if len(parts) == 6:
                tasks.append(Task(*parts[:5], extra_args=parts[5]))
            elif len(parts) == 7:
                tasks.append(Task(*parts[:5], params=parts[5], extra_args=parts[6]))
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
    recent_samples: list[tuple[float, float]] | None = None,
) -> float | None:
    if progress.current is None or progress.total is None or progress.total <= progress.current:
        return None
    if recent_samples and len(recent_samples) >= 2:
        started_at, started_progress = recent_samples[0]
        ended_at, ended_progress = recent_samples[-1]
        elapsed_seconds = ended_at - started_at
        progress_delta = ended_progress - started_progress
        if elapsed_seconds > 0 and progress_delta > 0:
            speed_per_hour = progress_delta * 3600.0 / elapsed_seconds
            return max(0.0, (progress.total - progress.current) / speed_per_hour)
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
    latest = None
    best_success = None
    for line in eval_log.read_text(errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if latest is None or (
                float(row.get("env_steps", -1)),
                float(row.get("optim_steps", -1)),
        ) >= (
                float(latest.get("env_steps", -1)),
                float(latest.get("optim_steps", -1)),
        ):
            latest = row
        if "succ_rate" in row and (
                best_success is None
                or float(row["succ_rate"]) > float(best_success)
        ):
            best_success = row["succ_rate"]
    if latest is None:
        return {}
    out: dict[str, str] = {}
    for key in ["env_steps", "optim_steps", "succ_rate", "epi_return"]:
        if key in latest:
            value = latest[key]
            out[key] = f"{float(value):.4g}" if isinstance(value, (float, int)) else str(value)
    if best_success is not None:
        out["best_succ_rate"] = f"{float(best_success):.4g}"
    return out


def latest_test(output_dir: Path) -> dict[str, str]:
    test_log = output_dir / "test.log"
    if not test_log.exists():
        return {}
    latest = None
    for line in test_log.read_text(errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("split", "test") == "test":
            latest = row
    if latest is None:
        return {}
    out: dict[str, str] = {}
    for key in ("env_steps", "optim_steps", "succ_rate", "epi_return"):
        if key in latest:
            value = latest[key]
            out[key] = f"{float(value):.4g}" if isinstance(value, (float, int)) else str(value)
    return out


def online_progress(
    output_dir: Path,
    task: Task,
    status: dict[str, str],
    latest: dict[str, str] | None = None,
) -> Progress:
    best = 0.0
    for path in output_dir.glob("checkpoint_env*_opt*.pth"):
        match = re.search(r"checkpoint_env(\d+)_opt(\d+)", path.name)
        if match:
            best = max(best, float(match.group(1)))
    log_file = Path(status.get("log_file", ""))
    if log_file.exists():
        log_progress = latest_number_before_marker(log_file, b"env steps")
        if log_progress is not None:
            best = max(best, log_progress)
    if latest is None:
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


def read_tail_bytes(path: Path, max_bytes: int = 5_000_000) -> bytes:
    if not path.exists():
        return b""
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
            return f.read()
    except OSError:
        return b""


def read_tail(path: Path, max_bytes: int = 5_000_000) -> str:
    return read_tail_bytes(path, max_bytes).decode("utf-8", "ignore")


def latest_number_before_marker(
    path: Path,
    marker: bytes,
    max_bytes: int = 5_000_000,
) -> float | None:
    """Return the last ASCII number separated by whitespace from a marker."""
    if not marker or max_bytes <= 0:
        return None
    initial_bytes = min(max_bytes, 256_000)
    data = read_tail_bytes(path, initial_bytes)
    result = latest_number_in_bytes(data, marker)
    if result is not None:
        return result
    try:
        needs_fallback = path.stat().st_size > initial_bytes and max_bytes > initial_bytes
    except OSError:
        needs_fallback = False
    if not needs_fallback:
        return None
    return latest_number_in_bytes(read_tail_bytes(path, max_bytes), marker)


def latest_number_in_bytes(data: bytes, marker: bytes) -> float | None:
    search_end = len(data)
    whitespace = b" \t\n\r\v\f"
    while search_end > 0:
        marker_start = data.rfind(marker, 0, search_end)
        if marker_start < 0:
            return None
        number_end = marker_start
        while number_end > 0 and data[number_end - 1] in whitespace:
            number_end -= 1
        if number_end == marker_start:
            search_end = marker_start
            continue
        number_start = number_end
        while number_start > 0 and (
            48 <= data[number_start - 1] <= 57 or data[number_start - 1] == 46
        ):
            number_start -= 1
        token = data[number_start:number_end]
        if (
            token
            and token[:1] != b"."
            and token[-1:] != b"."
            and token.count(b".") <= 1
        ):
            try:
                return float(token)
            except ValueError:
                pass
        search_end = marker_start
    return None


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


def compact_count(value: str) -> str:
    try:
        count = int(value)
    except (TypeError, ValueError):
        return value
    for scale, suffix in ((1_000_000, "m"), (1_000, "k")):
        if count >= scale and count % scale == 0:
            return f"{count // scale}{suffix}"
    return str(count)


@lru_cache(maxsize=None)
def model_size_preset_critic_count(family: str, level: str) -> str:
    if family == "base":
        family = "qrl"
    path = ROOT / "configs" / "model_size" / family / f"{level.lower()}.yaml"
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return ""
    for line in lines:
        match = re.fullmatch(r"num_critics:\s*(\d+)\s*(?:#.*)?", line)
        if match:
            return match.group(1)
    return ""


def task_critic_count(task: Task) -> str:
    value = extra_arg_value(task.extra_args, "agent.num_critics")
    if value:
        return value
    for group, family in (
        ("+cqrl_model_size", "go_qrl"),
        ("+go_qrl_model_size", "go_qrl"),
        ("+qrl_model_size", "qrl"),
        ("+base_model_size", "base"),
    ):
        level = extra_arg_value(task.extra_args, group)
        if level:
            value = model_size_preset_critic_count(family, level)
            if value:
                return value
    match = re.search(r"(?:^|_)(\d+)q(?:_|$)", task.task_id, re.IGNORECASE)
    if match:
        return match.group(1)
    return "2"


def task_checkpoint_interval(task: Task) -> str:
    value = extra_arg_value(task.extra_args, "save_steps")
    if not value:
        value = "50000" if task.mode == "online" else "10000"
    return compact_count(value)


def task_checkpoint_display(task: Task, output_dir: Path) -> str:
    if (output_dir / CHECKPOINTS_DELETED_MARKER).is_file():
        return "deleted"
    return task_checkpoint_interval(task)


def compact_parameter_count(count: int) -> str:
    for scale, suffix in ((1_000_000_000, "b"), (1_000_000, "m"), (1_000, "k")):
        if count >= scale:
            value = f"{count / scale:.1f}".rstrip("0").rstrip(".")
            return f"{value}{suffix}"
    return str(count)


def task_parameter_count(task: Task) -> str:
    value = task.params.strip().lower().replace(",", "")
    if not value:
        return ""
    try:
        return compact_parameter_count(int(value))
    except ValueError:
        return value


def task_dynamics_factorial_variant(task: Task) -> str:
    if "dynfac_" not in task.task_id:
        return ""

    id_match = re.search(
        r"dynfac_(?:(A\d{2})-)?(S[01])-(IQE|MSE|Hybrid)-(ReLU|Leaky)",
        task.task_id,
        re.IGNORECASE,
    )
    code = id_match.group(1).upper() if id_match and id_match.group(1) else ""

    separate = extra_arg_value(
        task.extra_args,
        "agent.quasimetric_critic.losses.separate_latent_dynamics",
    ).lower()
    separate_label = {"false": "S0", "true": "S1"}.get(separate, "")
    if not separate_label and id_match:
        separate_label = id_match.group(2).upper()

    distance = extra_arg_value(
        task.extra_args,
        "agent.quasimetric_critic.losses.latent_dynamics.distance",
    ).lower()
    distance_label = {
        "iqe": "IQE",
        "mse": "MSE",
        "iqe_mse": "Hybrid",
    }.get(distance, "")
    if not distance_label and id_match:
        distance_label = {
            "iqe": "IQE",
            "mse": "MSE",
            "hybrid": "Hybrid",
        }[id_match.group(3).lower()]

    activation = extra_arg_value(
        task.extra_args,
        "agent.quasimetric_critic.model.quasimetric_model.projector_activation",
    ).lower()
    activation_label = {
        "relu": "ReLU",
        "leaky_relu": "Leaky",
    }.get(activation, "")
    if not activation_label and id_match:
        activation_label = {
            "relu": "ReLU",
            "leaky": "Leaky",
        }[id_match.group(4).lower()]

    goal_variant = re.search(
        r"GO-QRL(?:\+|-)(?:Max|Min)\d+",
        task.task_id,
        re.IGNORECASE,
    )
    if goal_variant:
        prefix = re.sub(
            r"^GO-QRL-",
            "GO-QRL+",
            goal_variant.group(0),
            flags=re.IGNORECASE,
        )
    else:
        mode = extra_arg_value(
            task.extra_args, "agent.actor.losses.min_dist.latent_goal_mode"
        )
        steps = extra_arg_value(
            task.extra_args, "agent.actor.losses.min_dist.latent_goal_steps"
        )
        prefix = (
            f"GO-QRL+{mode.capitalize()}{steps}"
            if mode in {"max", "min"} and steps
            else "GO-QRL"
        )

    labels = [label for label in (separate_label, distance_label, activation_label) if label]
    suffix = "-".join(labels)
    if code:
        suffix = f"{code}:{suffix}" if suffix else code
    return f"{prefix}/{suffix}" if suffix else prefix


def task_variant(task: Task) -> str:
    algorithm = extra_arg_value(task.extra_args, "agent.algorithm")
    baseline_labels = {
        "td_infonce": "TD-InfoNCE",
        "crl": "CRL",
        "scaling_crl": "Scaling-CRL",
        "gcbc": "GCSL",
        "gcsl": "GCSL",
        "c_learning": "C-Learning",
    }
    if algorithm in baseline_labels:
        return baseline_labels[algorithm]

    dynamics_factorial_variant = task_dynamics_factorial_variant(task)
    if dynamics_factorial_variant:
        return dynamics_factorial_variant

    implementation = extra_arg_value(
        task.extra_args, "agent.goal_set_distance.losses.implementation"
    )
    if not implementation and "_Direct-" in task.task_id:
        implementation = "direct"
    if implementation in {"direct", "learned"}:
        aggregation = extra_arg_value(
            task.extra_args, "agent.goal_set_distance.losses.aggregation"
        ) or "hard_min"
        sampler = extra_arg_value(
            task.extra_args, "agent.goal_set_distance.losses.candidate_sampling"
        ) or "uniform_bounds"
        aggregation_label = {
            "hard_min": "HardMin",
            "lme_min": "LMEMin",
            "median": "Median",
            "hard_max": "HardMax",
            "lme_max": "LMEMax",
        }.get(aggregation, aggregation)
        sampler_label = {
            "uniform_bounds": "UniformBounds",
            "dataset_radius": "DatasetRadius",
        }.get(sampler, sampler)
        implementation_label = implementation.capitalize()
        return f"{implementation_label}/{aggregation_label}/{sampler_label}"

    encoder_kind = extra_arg_value(
        task.extra_args, "agent.quasimetric_critic.model.encoder.kind"
    )
    actor_input_mode = extra_arg_value(task.extra_args, "agent.actor.model.input_mode")
    latent_goal_mode = extra_arg_value(
        task.extra_args, "agent.actor.losses.min_dist.latent_goal_mode"
    )
    latent_goal_search = extra_arg_value(
        task.extra_args, "agent.actor.losses.min_dist.latent_goal_search"
    )
    latent_goal_optim = extra_arg_value(
        task.extra_args, "agent.actor.losses.min_dist.latent_goal_optim"
    )
    branch_normalization = extra_arg_value(
        task.extra_args, "agent.quasimetric_critic.model.encoder.branch_normalization"
    )
    dynamics_output_normalization = extra_arg_value(
        task.extra_args,
        "agent.quasimetric_critic.model.dynamics_output_normalization",
    )
    cqrl_model_size = extra_arg_value(task.extra_args, "+cqrl_model_size")
    named_cqrl_variant = re.search(
        r"(?:^|_)CQRL(?:\+|-)Inner(\d+)(?:_|-|\+|$)",
        task.task_id,
        re.IGNORECASE,
    )
    is_cqrl = bool(cqrl_model_size or named_cqrl_variant)
    named_latent_variant = re.search(
        r"_(GO-QRL(?:\+|-)(?:Max|Min)\d+|SplitLatent(?:Max|Min)\d+|SplitZero|LatentBase)(?:_|-|\+|$)",
        task.task_id,
    )
    if not encoder_kind and (
        is_cqrl
        or extra_arg_value(task.extra_args, "+go_qrl_model_size")
        or (
            named_latent_variant
            and named_latent_variant.group(1).startswith("GO-QRL")
        )
    ):
        encoder_kind = "split"
    if encoder_kind == "split":
        if latent_goal_mode in {"max", "min"}:
            latent_goal_steps = extra_arg_value(
                task.extra_args, "agent.actor.losses.min_dist.latent_goal_steps"
            )
            if is_cqrl:
                if not latent_goal_steps and named_cqrl_variant:
                    latent_goal_steps = named_cqrl_variant.group(1)
                modules = ["CQRL"]
                if latent_goal_steps:
                    modules.append(f"Inner{latent_goal_steps}")
            else:
                mode_label = (
                    "Inner0"
                    if latent_goal_steps == "0"
                    else f"{latent_goal_mode.capitalize()}{latent_goal_steps}"
                )
                modules = ["GO-QRL", mode_label]
            if latent_goal_search == "residual":
                modules.append("Res")
            if branch_normalization == "layernorm":
                modules.append(
                    "LNv1" if dynamics_output_normalization == "none" else "LN"
                )
            if latent_goal_optim == "rmsg":
                modules.append("RMSG")
            variant = "+".join(modules)
        elif is_cqrl:
            variant = "CQRL"
            if named_cqrl_variant:
                variant += f"+Inner{named_cqrl_variant.group(1)}"
        elif named_latent_variant and (
                named_latent_variant.group(1).startswith("SplitLatent")
                or named_latent_variant.group(1).startswith("GO-QRL")):
            variant = named_latent_variant.group(1)
            variant = re.sub(r'^SplitLatent(Max|Min)', r'GO-QRL+\1', variant)
            variant = re.sub(r'^GO-QRL-', 'GO-QRL+', variant)
        else:
            variant = "SplitZero"
        if latent_goal_search == "bounded_residual":
            radius = extra_arg_value(
                task.extra_args, "agent.actor.losses.min_dist.latent_goal_residual_radius"
            )
            try:
                radius = f"{float(radius):g}"
            except ValueError:
                pass
            variant += f"+BR{radius}" if radius else "+BR"
        return variant
    if actor_input_mode == "latent" or (
        named_latent_variant and named_latent_variant.group(1) == "LatentBase"
    ):
        return "LatentBase"
    if named_latent_variant:
        return named_latent_variant.group(1)
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
    if extra_arg_value(task.extra_args, "+qrl_model_size"):
        return "Base"
    return "base"


def task_environment_token_aliases(env_name: str) -> list[list[str]]:
    aliases: list[list[str]] = []
    hyphenated = re.split(r"[-_]", env_name.lower())
    if hyphenated and re.fullmatch(r"v\d+", hyphenated[-1]):
        hyphenated.pop()
    if hyphenated:
        aliases.append(hyphenated)

    camel_words = [
        word.lower()
        for word in re.findall(r"[A-Z]+(?=[A-Z][a-z]|$)|[A-Z]?[a-z]+|\d+", env_name)
    ]
    if len(camel_words) >= 2:
        aliases.append(["".join(camel_words[:2]), *camel_words[2:]])
        aliases.append(camel_words)
    return sorted(aliases, key=lambda parts: (len(parts), len("".join(parts))), reverse=True)


def task_display_name(task: Task) -> str:
    parts = task.task_id.split("_")
    lowered = [part.lower() for part in parts]
    for alias in task_environment_token_aliases(task.env_name):
        alias_len = len(alias)
        for index in range(len(parts) - alias_len + 1):
            if lowered[index:index + alias_len] == alias:
                del parts[index:index + alias_len]
                del lowered[index:index + alias_len]
                break
        else:
            continue
        break

    step_label = compact_count(task.steps).lower()
    variant_tokens = {
        "base", "latentbase", "splitzero", "gsd", "tmse", "a-latent", "alatent",
        "sonly", "tseptr", "tsep", "tr", "cda",
    }
    filtered = []
    for part in parts:
        lower = part.lower()
        if re.fullmatch(r"\d+q", lower):
            continue
        if re.fullmatch(r"bc\d+(?:\.\d+)?", lower):
            continue
        if lower == step_label or re.fullmatch(r"\d+(?:\.\d+)?[km]?ckpt", lower):
            continue
        if lower == f"s{task.seed}".lower() or lower == task.mode.lower():
            continue
        if re.fullmatch(r"h\d+", lower):
            continue
        if lower in variant_tokens:
            continue
        if re.fullmatch(r"splitlatent(?:max|min)\d+", lower):
            continue
        if re.fullmatch(
            r"go-qrl\+(?:max|min)\d+(?:\+(?:res|ln|rmsg))*-[sml]", lower
        ):
            continue
        if re.fullmatch(r"boundedresr\d+(?:\.\d+)?", lower):
            continue
        if lower.startswith("direct-") or lower.startswith("learned-"):
            continue
        filtered.append(part)
    return "_".join(filtered) or task.task_id


def display_status_timestamp(value: str) -> str:
    if re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", value):
        return value[5:]
    return value


def render(
    config: dict[str, str],
    recent_progress: RecentProgressHistory | None = None,
) -> None:
    tasks = read_tasks(resolve_path(cfg(config, "TASKS_FILE", "configs/qrl_tasks.tsv")))
    status_dir = resolve_path(cfg(config, "STATUS_DIR", "runs/qrl_queue/status"))
    output_root = resolve_path(cfg(config, "RESULTS_ROOT", "online/results_queue"))
    log_dir = resolve_path(cfg(config, "LOG_DIR", "logs/qrl_queue"))
    queue_started = queue_start_timestamp(config)
    eta_window_steps = max(
        0.0, float(cfg(config, "ETA_WINDOW_STEPS", "5000"))
    )
    gpus = cfg(config, "GPU_IDS", "0 1 2 3")
    print(time.strftime("%Y-%m-%d %H:%M:%S"))
    print(nvidia_smi(gpus))

    rows = []
    counts: dict[str, int] = {}
    total_elapsed = 0.0
    queue_eta = 0.0
    active_eta_task_ids: set[str] = set()
    for idx, task in enumerate(tasks, start=1):
        status = read_status(status_dir, task.task_id)
        state = status.get("state", "PENDING")
        pid = status.get("pid", "")
        alive = pid_alive(pid)
        if state == "RUNNING" and not alive:
            state = "STALE"
            status = {**status, "state": state}
        output_dir = Path(status.get("output_dir") or str(output_root / task.task_id))
        latest = latest_eval(output_dir)
        test = latest_test(output_dir)
        if task.mode == "online":
            progress = online_progress(output_dir, task, status, latest)
        else:
            progress = offline_progress(output_dir, task, status)
        run_elapsed = elapsed_hours(status)
        history_elapsed = history_elapsed_hours(task, status, log_dir, queue_started)
        display_elapsed = history_elapsed if history_elapsed is not None else run_elapsed
        if display_elapsed is not None:
            total_elapsed += display_elapsed
        recent_samples = None
        if (
            recent_progress is not None
            and state == "RUNNING"
            and progress.current is not None
        ):
            active_eta_task_ids.add(task.task_id)
            attempt = "|".join((
                status.get("pid", ""),
                status.get("started_at", ""),
                status.get("log_file", ""),
                str(progress.total),
            ))
            recent_samples = recent_progress.observe(
                task.task_id,
                attempt,
                progress.current,
                time.time(),
                eta_window_steps,
            )
        eta = estimated_eta_hours(
            task,
            status,
            progress,
            run_elapsed,
            history_elapsed,
            recent_samples=recent_samples,
        )
        if state in {"RUNNING", "PENDING"} and eta is not None:
            queue_eta = max(queue_eta, eta)
        val_last = latest.get("succ_rate", "")
        val_best = latest.get("best_succ_rate", "")
        test_succ = test.get("succ_rate", "")
        err = status.get("error", "")
        submitted_at = status.get("submitted_at") or status.get("started_at", "")
        gpu_started_at = status.get("gpu_started_at") or status.get("started_at", "")
        rows.append([
            idx,
            state,
            status.get("gpu", "") if state == "RUNNING" else "",
            pid if state == "RUNNING" else "",
            task_variant(task),
            task_parameter_count(task),
            task_critic_count(task),
            task.mode,
            task.env_name,
            task.seed,
            compact_count(task.steps),
            task_checkpoint_display(task, output_dir),
            display_status_timestamp(submitted_at),
            display_status_timestamp(gpu_started_at),
            status.get("gpu_mem_peak_mb", ""),
            status.get("oom_retry_count", ""),
            status.get("oom_prelaunch_mem_limit_mb", ""),
            f"{progress.pct:.1f}" if progress.pct is not None else "",
            f"{eta:.2f}" if eta is not None else "",
            f"{display_elapsed:.2f}" if display_elapsed is not None else "",
            progress.text,
            val_last,
            val_best,
            test_succ,
            err,
        ])
        counts[state] = counts.get(state, 0) + 1

    if recent_progress is not None:
        recent_progress.retain(active_eta_task_ids)
        recent_progress.save()

    print(
        "Status:",
        " ".join(f"{k}={v}" for k, v in sorted(counts.items())),
        f"task_elapsed_h={total_elapsed:.2f}",
        f"queue_eta_h={queue_eta:.2f}",
    )
    headers = [
        "#", "state", "gpu", "pid", "variant", "params", "critics", "mode", "env",
        "seed", "steps", "ckpt", "submitted", "gpu_start", "peak_mb", "oom_n", "mem_cap",
        "%", "eta_h", "elapsed_h", "progress", "val_last", "val_best", "test_succ",
        "err",
    ]
    widths = [
        4, 8, 4, 8, 34, 7, 7, 7, 22, 6, 6, 7, 14, 14, 8, 5, 8, 6, 7,
        9, 17, 9, 9, 9, 18,
    ]
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
    eta_sample_file = resolve_path(cfg(
        config,
        "ETA_SAMPLE_FILE",
        str(resolve_path(cfg(config, "STATUS_DIR", "runs/qrl_queue/status"))
            / "watch_eta_samples.json"),
    ))
    recent_progress = RecentProgressHistory(eta_sample_file)
    watch = args.watch
    if watch is None:
        watch = float(cfg(config, "WATCH_SECONDS", cfg(config, "POLL_SECONDS", "30")))
    while True:
        if args.clear_screen:
            print("\033[2J\033[H", end="")
        render(config, recent_progress)
        if watch <= 0:
            return 0
        time.sleep(watch)
        print()


if __name__ == "__main__":
    raise SystemExit(main())

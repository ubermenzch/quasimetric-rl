#!/usr/bin/env python3
"""Migrate the active AntNavigate ablations to compact checkpoint retention."""

from __future__ import annotations

import argparse
import gc
import shutil
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from online.main import (  # noqa: E402
    ONLINE_CHECKPOINT_KIND_COMMITTED,
    ensure_selected_best_agent_checkpoint,
    online_checkpoint_key,
    select_best_validation,
)
from tools.run_qrl_queue import (  # noqa: E402
    cfg,
    completion_evidence,
    output_finished,
    parse_config,
    pid_alive,
    read_status,
    read_tasks,
    resolve_path,
    status_file,
    task_output_dir,
    write_status,
    write_task_manifest,
)


TARGET_PREFIXES = (
    "ablation_GO-QRL+Max4-Plain-L_500k_50kckpt_val500_test1000_"
    "antnavigate_v4_l500k_online_s",
    "ablation_GO-QRL+Max1-ResidualLN-SiLU-L_500k_50kckpt_val500_test1000_"
    "antnavigate_v4_l500k_online_s",
)
TARGET_IDS = {
    f"{prefix}{seed}"
    for prefix in TARGET_PREFIXES
    for seed in range(1000, 1005)
}
COMPACT_ARGS = {
    "keep_only_latest_checkpoint": "true",
    "keep_only_best_and_final_checkpoints": "true",
    "save_replay_buffer": "true",
    "save_final_replay_buffer": "true",
    "resume_if_possible": "true",
}
SOURCE_TASK_FILES = (
    ROOT / "configs/go_qrl_plain_l_max4_antnavigate_5seed.tsv",
    ROOT / "configs/go_qrl_residual_l_max1_antnavigate_5seed.tsv",
)


class MigrationError(RuntimeError):
    pass


def update_extra_args(extra_args: str) -> str:
    tokens = extra_args.split()
    rewritten: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        key = token.split("=", 1)[0]
        if key not in COMPACT_ARGS:
            rewritten.append(token)
            continue
        if key not in seen:
            rewritten.append(f"{key}={COMPACT_ARGS[key]}")
            seen.add(key)
    for key, value in COMPACT_ARGS.items():
        if key not in seen:
            rewritten.append(f"{key}={value}")
    return " ".join(rewritten)


def rewrite_task_file(path: Path, expected_ids: set[str], stamp: str) -> Path:
    lines = path.read_text().splitlines(keepends=True)
    found: set[str] = set()
    rewritten: list[str] = []
    for raw_line in lines:
        stripped = raw_line.rstrip("\r\n")
        newline = raw_line[len(stripped):]
        if not stripped or stripped.startswith("#"):
            rewritten.append(raw_line)
            continue
        parts = stripped.split("\t", 6)
        if len(parts) == 7 and parts[0] in expected_ids:
            parts[6] = update_extra_args(parts[6])
            found.add(parts[0])
            rewritten.append("\t".join(parts) + newline)
        else:
            rewritten.append(raw_line)
    if found != expected_ids:
        missing = sorted(expected_ids - found)
        unexpected = sorted(found - expected_ids)
        raise MigrationError(
            f"{path}: target mismatch; missing={missing}, unexpected={unexpected}"
        )

    backup = path.with_name(f"{path.name}.before_compact_{stamp}")
    shutil.copy2(path, backup)
    temp = path.with_name(f"{path.name}.compact.tmp")
    temp.write_text("".join(rewritten))
    temp.replace(path)
    return backup


def checkpoint_sort_key(path: Path) -> tuple[int, int, int]:
    key = online_checkpoint_key(str(path))
    if key is None:
        raise MigrationError(f"Unrecognized online checkpoint: {path}")
    return (*key, int(path.name.endswith("_final.pth")))


def latest_full_checkpoint(output_dir: Path) -> Path:
    candidates = [
        path for path in output_dir.glob("checkpoint_*.pth")
        if online_checkpoint_key(str(path)) is not None
        and not path.name.endswith("_finalizing.pth")
    ]
    if not candidates:
        raise MigrationError(f"No resumable checkpoint in {output_dir}")
    return max(candidates, key=checkpoint_sort_key)


def load_validated_checkpoint(latest: Path) -> tuple[dict, dict]:
    state = torch.load(latest, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise MigrationError(f"Checkpoint payload is not a mapping: {latest}")
    kind = state.get("checkpoint_kind")
    if kind not in (None, ONLINE_CHECKPOINT_KIND_COMMITTED):
        raise MigrationError(f"Checkpoint is not committed: {latest} ({kind!r})")
    for key in ("agent", "losses", "rng", "replay", "validation_summary"):
        if key not in state:
            raise MigrationError(f"Checkpoint lacks {key}: {latest}")
    summaries = state.get("val_summaries")
    if not isinstance(summaries, list) or not summaries:
        raise MigrationError(f"Checkpoint has no validation history: {latest}")
    best = select_best_validation(summaries)
    return state, best


def validate_and_publish_best(output_dir: Path, latest: Path) -> Path:
    state, best = load_validated_checkpoint(latest)
    _, selected_path, _ = ensure_selected_best_agent_checkpoint(
        str(output_dir),
        validation_summary=best,
        committed_state=state,
    )
    del state
    gc.collect()
    return Path(selected_path)


def compact_checkpoint_files(
    output_dir: Path, latest: Path, selected_path: Path,
) -> tuple[int, int]:
    if not selected_path.is_file() or not latest.is_file():
        raise MigrationError(
            f"Refusing cleanup without selected best and latest: {output_dir}"
        )
    removed_files = 0
    removed_bytes = 0
    for pattern in ("checkpoint_*.pth", "agent_checkpoint_*.pth"):
        for path in sorted(output_dir.glob(pattern)):
            if path.resolve() == latest.resolve():
                continue
            size = path.stat().st_size
            path.unlink()
            removed_files += 1
            removed_bytes += size
    return removed_files, removed_bytes


def status_extra_for_migration(previous: dict[str, str], output_dir: Path) -> dict[str, str]:
    keep = (
        "submitted_at", "gpu_mem_peak_mb", "prelaunch_gpu_mem_used_mb",
        "transient_failure_count", "oom_retry_count",
    )
    extra = {key: previous[key] for key in keep if previous.get(key)}
    extra.update({
        "output_dir": str(output_dir),
        "error": "compact_checkpoint_policy_migration",
        "requeue_reason": "resume_with_best_and_final_retention",
        "previous_gpu": previous.get("gpu") or previous.get("previous_gpu", ""),
        "previous_pid": previous.get("pid") or previous.get("previous_pid", ""),
        "started_at": "",
        "gpu_started_at": "",
    })
    return extra


def migrate(config_path: Path, execute: bool) -> None:
    config = parse_config(config_path)
    tasks_path = resolve_path(cfg(config, "TASKS_FILE", "runs/qrl_queue/tasks.tsv"))
    status_dir = resolve_path(cfg(config, "STATUS_DIR", "runs/qrl_queue/status"))
    old_tasks = {task.task_id: task for task in read_tasks(tasks_path)}
    if not TARGET_IDS.issubset(old_tasks):
        raise MigrationError(
            f"Queue is missing targets: {sorted(TARGET_IDS - old_tasks.keys())}"
        )

    for task_id in sorted(TARGET_IDS):
        status = read_status(status_dir, task_id)
        pid = status.get("pid", "")
        if pid and pid_alive(pid):
            raise MigrationError(
                f"Task process is still alive; stop the queue first: {task_id} pid={pid}"
            )
        output_dir = task_output_dir(config, old_tasks[task_id])
        latest = latest_full_checkpoint(output_dir)
        state, best = load_validated_checkpoint(latest)
        print(
            f"PREFLIGHT {task_id}: state={status.get('state', 'PENDING')} "
            f"latest={latest.name} size={latest.stat().st_size} "
            f"best_env_steps={best.get('env_steps')}"
        )
        del state
        gc.collect()

    if not execute:
        print("Dry run only. Re-run with --yes after stopping the scheduler and jobs.")
        return

    stamp = time.strftime("%Y%m%d-%H%M%S")
    backups = [rewrite_task_file(tasks_path, TARGET_IDS, stamp)]
    for source_path, prefix in zip(SOURCE_TASK_FILES, TARGET_PREFIXES):
        expected = {f"{prefix}{seed}" for seed in range(1000, 1005)}
        backups.append(rewrite_task_file(source_path, expected, stamp))

    new_tasks = {task.task_id: task for task in read_tasks(tasks_path)}
    removed_files = 0
    removed_bytes = 0
    for task_id in sorted(TARGET_IDS):
        task = new_tasks[task_id]
        previous = read_status(status_dir, task_id)
        output_dir = task_output_dir(config, task)
        latest = latest_full_checkpoint(output_dir)
        selected_path = validate_and_publish_best(output_dir, latest)
        removed_count, reclaimed = compact_checkpoint_files(
            output_dir, latest, selected_path,
        )
        removed_files += removed_count
        removed_bytes += reclaimed

        status_path = status_file(status_dir, task_id)
        shutil.copy2(
            status_path,
            status_path.with_name(f"{status_path.name}.before_compact_{stamp}"),
        )
        write_task_manifest(output_dir, task)
        if output_finished(output_dir):
            extra = status_extra_for_migration(previous, output_dir)
            extra.update({
                "exit_code": previous.get("exit_code", "0"),
                "completion_evidence": completion_evidence(output_dir),
            })
            write_status(status_dir, task, "DONE", extra)
            state = "DONE"
        else:
            write_status(
                status_dir, task, "PENDING",
                status_extra_for_migration(previous, output_dir),
            )
            state = "PENDING"
        print(
            f"MIGRATED {task_id}: state={state} latest={latest.name} "
            f"selected={selected_path.name} removed={removed_count}"
        )

    print("Task-table backups:")
    for backup in backups:
        print(f"  {backup}")
    print(
        f"Migration complete: tasks={len(TARGET_IDS)}, removed_files={removed_files}, "
        f"reclaimed_bytes={removed_bytes}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/qrl_queue.env")
    parser.add_argument("--yes", action="store_true", help="Execute the migration")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        migrate(resolve_path(args.config), args.yes)
    except (MigrationError, OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

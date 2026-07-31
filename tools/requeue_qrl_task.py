#!/usr/bin/env python3
"""Safely return one queue task to PENDING by its stable task ID."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_qrl_queue import (
    TASK_MANIFEST_NAME,
    cfg,
    parse_config,
    read_status,
    read_tasks,
    resolve_path,
    task_identity_issue,
    task_output_dir,
    write_status,
    write_task_manifest,
)


class RequeueError(RuntimeError):
    pass


def real_output_files(output_dir: Path) -> list[Path]:
    if not output_dir.exists():
        return []
    metadata_names = {TASK_MANIFEST_NAME, f"{TASK_MANIFEST_NAME}.tmp"}
    try:
        return [path for path in output_dir.iterdir() if path.name not in metadata_names]
    except OSError as exc:
        raise RequeueError(f"cannot inspect output directory {output_dir}: {exc}") from exc


def requeue(config_path: Path, task_id: str, execute: bool) -> None:
    config = parse_config(config_path)
    tasks_file = resolve_path(cfg(config, "TASKS_FILE", "runs/qrl_queue/tasks.tsv"))
    matches = [task for task in read_tasks(tasks_file) if task.task_id == task_id]
    if len(matches) != 1:
        raise RequeueError(
            f"expected exactly one active task with task_id={task_id!r}, found {len(matches)}"
        )

    task = matches[0]
    status_dir = resolve_path(cfg(config, "STATUS_DIR", "runs/qrl_queue/status"))
    status = read_status(status_dir, task_id)
    state = status.get("state", "PENDING")
    if state in {"RUNNING", "DONE"}:
        raise RequeueError(f"refusing to requeue task in {state} state")

    output_dir = task_output_dir(config, task)
    issue = task_identity_issue(status, output_dir, task)
    real_outputs = real_output_files(output_dir)
    if issue and real_outputs:
        names = ", ".join(path.name for path in real_outputs[:5])
        raise RequeueError(
            f"refusing to repair {issue} because training outputs exist: {names}"
        )
    manifest_path = output_dir / TASK_MANIFEST_NAME
    manifest_temp_path = output_dir / f"{TASK_MANIFEST_NAME}.tmp"
    repair_manifest = issue in {
        "missing_output_task_manifest",
        "invalid_output_task_manifest",
    } or (manifest_temp_path.exists() and not manifest_path.exists())
    if issue and not repair_manifest:
        raise RequeueError(f"refusing to requeue task with identity issue: {issue}")

    print(f"task_id={task_id}")
    print(f"state={state} -> PENDING")
    print(f"output_dir={output_dir}")
    print(f"repair_manifest={str(repair_manifest).lower()}")
    if not execute:
        print("Dry run only. Re-run with --yes to requeue this task.")
        return

    try:
        if repair_manifest:
            write_task_manifest(output_dir, task)
        write_status(status_dir, task, "PENDING", {
            "error": "manual_requeue",
            "requeue_reason": "safe_task_identity_recovery",
            "previous_error": status.get("error", ""),
            "submitted_at": status.get("submitted_at", ""),
            "output_dir": str(output_dir),
        })
    except OSError as exc:
        raise RequeueError(f"cannot repair and requeue task: {exc}") from exc
    print("Requeued task as PENDING.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Safely requeue one QRL task using its stable task_id."
    )
    parser.add_argument("--config", default="configs/qrl_queue.env")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--yes", action="store_true", help="Execute the requeue")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        requeue(resolve_path(args.config), args.task_id, args.yes)
    except RequeueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

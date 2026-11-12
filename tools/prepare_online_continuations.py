#!/usr/bin/env python3
"""Mark completed online tasks for scheduler-managed continuation."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_qrl_queue import (  # noqa: E402
    QueueLock,
    cfg,
    latest_online_checkpoint_step,
    output_finished,
    parse_config,
    read_status,
    read_tasks,
    resolve_path,
    task_output_dir,
    task_fingerprint,
    task_manifest_fingerprint,
    write_status,
    write_task_manifest,
)


def update_task_steps(tasks_path: Path, task_ids: set[str], target_steps: int) -> None:
    lines = tasks_path.read_text().splitlines(keepends=True)
    seen: set[str] = set()
    updated: list[str] = []
    for raw in lines:
        newline = "\n" if raw.endswith("\n") else ""
        body = raw[:-1] if newline else raw
        if not body or body.lstrip().startswith("#"):
            updated.append(raw)
            continue
        fields = body.split("\t")
        if fields and fields[0] in task_ids:
            if len(fields) < 7:
                raise ValueError(f"Malformed task row for {fields[0]!r}")
            fields[4] = str(target_steps)
            seen.add(fields[0])
            body = "\t".join(fields)
        updated.append(body + newline)
    missing = task_ids - seen
    if missing:
        raise ValueError(
            "Task IDs were not found in the queue file: " + ", ".join(sorted(missing))
        )
    temp = tasks_path.with_suffix(tasks_path.suffix + ".tmp")
    temp.write_text("".join(updated))
    temp.replace(tasks_path)


def prepare(
        config_path: Path, task_ids: set[str], target_steps: int,
        *, allow_stale_lock: bool,
) -> None:
    if target_steps <= 0:
        raise ValueError("target steps must be positive")
    config = parse_config(config_path)
    tasks_path = resolve_path(cfg(config, "TASKS_FILE", "runs/qrl_queue/tasks.tsv"))
    status_dir = resolve_path(cfg(config, "STATUS_DIR", "runs/qrl_queue/status"))
    tasks = {task.task_id: task for task in read_tasks(tasks_path)}
    missing = task_ids - tasks.keys()
    if missing:
        raise ValueError(
            "Task IDs were not found in the queue file: " + ", ".join(sorted(missing))
        )

    continuations = []
    for task_id in sorted(task_ids):
        task = tasks[task_id]
        if task.mode != "online":
            raise ValueError(f"{task_id} is not an online task")
        status = read_status(status_dir, task_id)
        state = status.get("state")
        stale_paused_status = (
            state == "PAUSED"
            and status.get("error") == "status_task_definition_mismatch"
            and task_manifest_fingerprint(task_output_dir(config, task))
            == task_fingerprint(task)
        )
        if state != "DONE" and not stale_paused_status:
            raise ValueError(
                f"{task_id} must be DONE before continuation; got {state or 'MISSING'}"
            )
        output_dir = task_output_dir(config, task)
        latest_step = latest_online_checkpoint_step(output_dir)
        if latest_step is None:
            raise ValueError(f"{task_id} has no resumable online checkpoint")
        if latest_step >= target_steps:
            raise ValueError(
                f"{task_id} already has checkpoint at {latest_step}, target is {target_steps}"
            )
        if not output_finished(output_dir):
            raise ValueError(f"{task_id} does not have completed output evidence")
        if not os.access(output_dir, os.W_OK):
            raise PermissionError(
                f"Result directory is not writable: {output_dir}; "
                "run this preparation on the training host"
            )
        continuations.append(
            (task, replace(task, steps=str(target_steps)), latest_step, state or "MISSING")
        )

    lock_path = resolve_path(cfg(config, "LOCK_FILE", "runs/qrl_queue/qrl_queue.lock"))
    lock = QueueLock(lock_path)
    lock_acquired = lock.acquire()
    if not lock_acquired:
        # A failed flock is authoritative: another process holds this inode.
        # The PID stored inside the file is only diagnostic metadata and may
        # have been overwritten by an older, buggy scheduler invocation.
        raise RuntimeError(
            f"The queue scheduler is active; could not acquire {lock_path}. "
            "Stop it before preparing continuations."
        )
    try:
        now_task_ids = {task.task_id for task, _, _, _ in continuations}
        update_task_steps(tasks_path, now_task_ids, target_steps)
        for old_task, continuation, latest_step, previous_state in continuations:
            output_dir = task_output_dir(config, old_task)
            write_task_manifest(output_dir, continuation)
            remote_marker = output_dir / ".qrl_remote_sync.json"
            if remote_marker.exists():
                remote_marker.unlink()
            write_status(
                status_dir,
                continuation,
                "PENDING",
                {
                    "output_dir": str(output_dir),
                    "requeue_reason": "extend_completed_online_run",
                    "resume_from_env_steps": str(latest_step),
                    "target_env_steps": str(target_steps),
                    "previous_state": previous_state,
                    "started_at": "",
                    "submitted_at": "",
                },
            )
            print(
                f"{old_task.task_id}: {latest_step} -> {target_steps} env steps; state=PENDING"
            )
    finally:
        if lock_acquired:
            lock.release()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qrl_queue.env")
    parser.add_argument("--target-env-steps", type=int, required=True)
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--task-list")
    parser.add_argument(
        "--allow-stale-lock",
        action="store_true",
        help=(
            "Deprecated compatibility option. An actively held queue lock is "
            "never bypassed; an unlocked stale file needs no special option."
        ),
    )
    args = parser.parse_args()
    task_ids = set(args.task_id or [])
    if args.task_list:
        task_list = resolve_path(args.task_list)
        task_ids.update(
            line.strip()
            for line in task_list.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    if not task_ids:
        parser.error("provide --task-id or --task-list")
    prepare(
        resolve_path(args.config),
        task_ids,
        args.target_env_steps,
        allow_stale_lock=args.allow_stale_lock,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

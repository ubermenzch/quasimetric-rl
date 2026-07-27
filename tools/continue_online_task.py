#!/usr/bin/env python3
"""Continue a completed online queue task in its existing output directory."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quasimetric_rl.utils import full_checkpoint_key
from tools.run_qrl_queue import (
    QueueLock,
    Task,
    build_command,
    cfg,
    output_complete,
    parse_config,
    read_tasks,
    resolve_path,
    task_output_dir,
)


def latest_online_checkpoint_step(output_dir: Path) -> int | None:
    keys = [
        key
        for path in output_dir.glob('checkpoint_env*_opt*.pth')
        if (key := full_checkpoint_key(path)) is not None
    ]
    return max(keys)[0] if keys else None


def make_continuation_task(task: Task, total_env_steps: int) -> Task:
    if task.mode != 'online':
        raise ValueError(f'Only online tasks can be continued, got mode={task.mode!r}')
    if total_env_steps <= 0:
        raise ValueError('total_env_steps must be positive')
    forced_args = (
        'resume_if_possible=True '
        'save_replay_buffer=True '
        f'interaction.total_env_steps={total_env_steps}'
    )
    return replace(
        task,
        steps=str(total_env_steps),
        extra_args=' '.join(part for part in (task.extra_args, forced_args) if part),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            'Resume a completed online task from its latest full checkpoint. '
            'The requested step count is the new absolute total, not an increment.'
        ),
    )
    parser.add_argument('task_id')
    parser.add_argument('--total-env-steps', type=int, required=True)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--config', default='configs/qrl_queue.env')
    parser.add_argument('--tasks-file')
    parser.add_argument('--print-command', action='store_true')
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    config = parse_config(config_path)
    tasks_path = (
        resolve_path(args.tasks_file)
        if args.tasks_file
        else resolve_path(cfg(config, 'TASKS_FILE', 'runs/qrl_queue/tasks.tsv'))
    )
    matches = [task for task in read_tasks(tasks_path) if task.task_id == args.task_id]
    if len(matches) != 1:
        parser.error(
            f'Expected exactly one task named {args.task_id!r} in {tasks_path}, '
            f'found {len(matches)}'
        )
    task = matches[0]
    output_dir = task_output_dir(config, task)
    if not output_complete(output_dir):
        parser.error(f'Task output is not complete: {output_dir}')
    latest_step = latest_online_checkpoint_step(output_dir)
    if latest_step is None:
        parser.error(f'No resumable full checkpoint found in {output_dir}')
    if args.total_env_steps <= latest_step:
        parser.error(
            f'--total-env-steps must exceed latest checkpoint step {latest_step}'
        )

    continuation = make_continuation_task(task, args.total_env_steps)
    command, env, cwd, _ = build_command(config, continuation, args.gpu)
    print(f'Continue {task.task_id}: {latest_step} -> {args.total_env_steps} env steps')
    print(shlex.join(command), flush=True)
    if args.print_command:
        return 0

    lock = QueueLock(resolve_path(cfg(config, 'LOCK_FILE', 'runs/qrl_queue/qrl_queue.lock')))
    if not lock.acquire():
        parser.error(
            'The queue scheduler is active. Stop it before continuing a completed task.'
        )
    try:
        return subprocess.run(command, cwd=cwd, env=env, check=False).returncode
    finally:
        lock.release()


if __name__ == '__main__':
    raise SystemExit(main())

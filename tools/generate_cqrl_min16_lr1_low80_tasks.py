#!/usr/bin/env python3
"""Generate the Min16/LR1 counterpart of the current Min32 task set."""

from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path
import shlex
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_qrl_queue import Task, read_tasks


DEFAULT_SOURCE = ROOT / "configs/cqrl_min32_lr1_l_low80_1seed.tsv"
DEFAULT_OUTPUT = ROOT / "configs/cqrl_min16_lr1_l_low80_1seed.tsv"
SOURCE_VARIANT = "CQRL+Min32-L-InnerLR1"
TARGET_VARIANT = "CQRL+Min16-L-InnerLR1"


def replace_extra_args(extra_args: str) -> str:
    tokens = shlex.split(extra_args)
    expected = {
        "agent.actor.losses.min_dist.latent_goal_mode": "min",
        "agent.actor.losses.min_dist.latent_goal_lr": "1",
        "agent.actor.losses.min_dist.latent_goal_steps": "32",
    }
    seen: set[str] = set()
    rewritten = []
    for token in tokens:
        key, _, value = token.partition("=")
        if key in expected:
            if value != expected[key]:
                raise ValueError(
                    f"Expected {key}={expected[key]}, got {token}"
                )
            value = "16" if key.endswith("latent_goal_steps") else value
            token = f"{key}={value}"
            seen.add(key)
        rewritten.append(token)
    missing = set(expected) - seen
    if missing:
        raise ValueError(f"Missing Min32 inner-loop arguments: {sorted(missing)}")
    return " ".join(rewritten)


def transform_task(task: Task) -> Task:
    marker = f"_{SOURCE_VARIANT}_"
    if marker not in task.task_id:
        raise ValueError(f"Expected a Min32 task, got {task.task_id}")
    task_id = task.task_id.replace(marker, f"_{TARGET_VARIANT}_", 1)
    return replace(
        task,
        task_id=task_id,
        extra_args=replace_extra_args(task.extra_args),
    )


def generate_tasks(path: Path = DEFAULT_SOURCE) -> list[Task]:
    tasks = [transform_task(task) for task in read_tasks(path)]
    if len(tasks) != 15 or len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError(f"Expected 15 unique Min32 tasks, got {len(tasks)}")
    return tasks


def task_line(task: Task) -> str:
    return "\t".join((
        task.task_id,
        task.mode,
        task.env_name,
        task.seed,
        task.steps,
        task.params,
        task.extra_args,
    ))


def render_tasks(tasks: list[Task]) -> str:
    return "\n".join([
        "# CQRL-L Min16/LR1, paired with the CQRL-L Min32 low80 task set.",
        "# Same environments, seed, budgets, and settings; only inner steps=16.",
        *(task_line(task) for task in tasks),
    ]) + "\n"


def write_tasks(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{path.name}.", suffix=".tmp", dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def append_tasks(path: Path, tasks: list[Task]) -> int:
    existing_ids = {task.task_id for task in read_tasks(path)}
    additions = [task for task in tasks if task.task_id not in existing_ids]
    if not additions:
        return 0
    current = path.read_text() if path.exists() else ""
    separator = "" if not current or current.endswith("\n\n") else (
        "\n" if current.endswith("\n") else "\n\n"
    )
    block = "\n".join([
        "# CQRL-L Min16/LR1: paired with the Min32 low80 task set.",
        *(task_line(task) for task in additions),
    ]) + "\n"
    write_tasks(path, current + separator + block)
    return len(additions)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--append-to", type=Path, default=None)
    args = parser.parse_args()

    tasks = generate_tasks(args.source)
    write_tasks(args.output, render_tasks(tasks))
    print(f"Wrote {len(tasks)} tasks to {args.output}")
    if args.append_to is not None:
        added = append_tasks(args.append_to, tasks)
        print(f"Appended {added} new tasks to {args.append_to}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

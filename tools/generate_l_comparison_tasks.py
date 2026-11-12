#!/usr/bin/env python3
"""Generate L-scale five-algorithm tasks on five comparison environments."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import os
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.generate_iclr_main_table_tasks import (
    ALGORITHMS as MAIN_ALGORITHMS,
    CHECKPOINT_ARGS,
    ENVIRONMENTS as MAIN_ENVIRONMENTS,
    algorithm_args,
    parameter_count,
    task_version_tag,
)
from tools.generate_online_baseline_tasks import compact_parameter_count
from tools.generate_selected_success_l_min32_tasks import (
    atomic_write,
    completed_task_ids,
    extra_arg_map,
    task_line,
)
from tools.run_qrl_queue import Task, read_tasks


TRAINING_SEEDS = tuple(range(1000, 1005))
ALGORITHM_KEYS = ("qrl", "td_infonce", "gcsl", "c_learning", "crl")
COMPARISON_SPECS = (
    ("fetchpush", 200_000),
    ("fetchslide", 500_000),
    ("fetchpickandplace", 200_000),
    ("reacher_hard", 200_000),
    ("maze2d_large", 200_000),
)
DEFAULT_OUTPUT = ROOT / "configs/l_comparison_5alg_5env_5seed.tsv"


def comparison_environments():
    by_key = {environment.key: environment for environment in MAIN_ENVIRONMENTS}
    return tuple(
        replace(by_key[key], model_size="l", total_steps=total_steps)
        for key, total_steps in COMPARISON_SPECS
    )


def selected_algorithms():
    by_key = {algorithm.key: algorithm for algorithm in MAIN_ALGORITHMS}
    return tuple(by_key[key] for key in ALGORITHM_KEYS)


def generate_tasks() -> list[Task]:
    tasks = []
    for environment in comparison_environments():
        total_label = f"{environment.total_steps // 1000}k"
        checkpoint_label = f"{environment.save_steps // 1000}kckpt"
        for algorithm in selected_algorithms():
            params = compact_parameter_count(parameter_count(
                algorithm, environment,
            ))
            for seed in TRAINING_SEEDS:
                task_id = (
                    f"iclr_main_v1_{algorithm.label}-L_{total_label}_"
                    f"{checkpoint_label}_val{environment.validation_episodes}_"
                    f"test{environment.test_episodes}_"
                    f"{task_version_tag(algorithm, environment)}"
                    f"{environment.slug}_online_s{seed}"
                )
                extra_args = " ".join((
                    f"env.kind={environment.kind}",
                    *algorithm_args(algorithm, environment),
                    f"interaction.validation_seed={environment.validation_seed}",
                    f"interaction.test_seed={environment.test_seed}",
                    f"interaction.num_eval_episodes={environment.validation_episodes}",
                    f"interaction.num_test_episodes={environment.test_episodes}",
                    "eval_steps=null",
                    f"save_steps={environment.save_steps}",
                    *CHECKPOINT_ARGS,
                ))
                tasks.append(Task(
                    task_id=task_id,
                    mode="online",
                    env_name=environment.name,
                    seed=str(seed),
                    steps=str(environment.total_steps),
                    params=params,
                    extra_args=extra_args,
                ))
    validate_tasks(tasks)
    return tasks


def validate_tasks(tasks: Sequence[Task]) -> None:
    environments = {
        environment.name: environment for environment in comparison_environments()
    }
    expected = len(environments) * len(ALGORITHM_KEYS) * len(TRAINING_SEEDS)
    if len(tasks) != expected:
        raise ValueError(f"Expected {expected} tasks, got {len(tasks)}")
    duplicates = [
        task_id for task_id, count
        in Counter(task.task_id for task in tasks).items() if count > 1
    ]
    if duplicates:
        raise ValueError(f"Duplicate generated task IDs: {duplicates}")
    for task in tasks:
        environment = environments[task.env_name]
        args = extra_arg_map(task)
        if task.steps != str(environment.total_steps):
            raise ValueError(f"Invalid training budget: {task.task_id}")
        if args.get("save_steps") != str(environment.save_steps):
            raise ValueError(f"Invalid checkpoint interval: {task.task_id}")
        model_sizes = [
            value for key, value in args.items() if key.endswith("_model_size")
        ]
        if model_sizes != ["l"]:
            raise ValueError(f"Expected one L model preset: {task.task_id}")
        if not 21.0 <= float(task.params.removesuffix("m")) <= 23.0:
            raise ValueError(f"Expected L-scale parameter count: {task.task_id}")


def render_tasks(tasks: Sequence[Task]) -> str:
    return "\n".join((
        "# task_id\tmode\tenv_name\tseed\tsteps\tparams\textra_args",
        "# Five algorithms x five environments x five seeds at residual L scale.",
        "# Checkpoint intervals match the existing main-table protocols.",
        *(task_line(task) for task in tasks),
    )) + "\n"


def append_tasks(
        path: Path, tasks: Sequence[Task],
        status_dir: Path | None = None) -> int:
    if status_dir is None:
        status_dir = path.parent / "status"
    completed = completed_task_ids(status_dir)
    existing = {task.task_id: task for task in read_tasks(path)}
    additions = []
    for task in tasks:
        previous = existing.get(task.task_id)
        if previous is not None:
            if previous != task:
                raise ValueError(
                    f"Existing task ID has a different definition: {task.task_id}"
                )
            continue
        if task.task_id in completed:
            continue
        existing[task.task_id] = task
        additions.append(task)
    if not additions:
        return 0
    current = path.read_text() if path.exists() else ""
    separator = "" if not current or current.endswith("\n\n") else (
        "\n" if current.endswith("\n") else "\n\n"
    )
    block = "\n".join((
        "# L-scale five-algorithm comparison tasks; matched checkpoint intervals.",
        *(task_line(task) for task in additions),
    )) + "\n"
    atomic_write(path, current + separator + block)
    return len(additions)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--append-to", type=Path)
    args = parser.parse_args()
    tasks = generate_tasks()
    atomic_write(args.output, render_tasks(tasks))
    print(f"Wrote {len(tasks)} tasks to {args.output}")
    if args.append_to is not None:
        print(f"Added {append_tasks(args.append_to, tasks)} tasks to {args.append_to}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

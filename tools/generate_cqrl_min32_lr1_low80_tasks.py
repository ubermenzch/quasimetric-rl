#!/usr/bin/env python3
"""Generate CQRL-L Min32/LR1 tasks for seed-1000 test success in [10%, 80%)."""

from __future__ import annotations

import argparse
from collections import Counter
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


DEFAULT_SOURCE = ROOT / "runs/qrl_queue/tasks.tsv"
DEFAULT_OUTPUT = ROOT / "configs/cqrl_min32_lr1_l_low80_1seed.tsv"
SOURCE_SEED = "1000"
TARGET_SEED = "1001"
TARGET_VARIANT = "CQRL+Min32-L-InnerLR1"

# Final test success reported for CQRL/1-L seed 1000 on 2026-08-25/26.
TEST_SUCCESS = {
    "manipulator_insert_ball": 0.07,
    "manipulator_insert_peg": 0.07,
    "dog_fetch": 0.0,
    "stacker_stack_2": 0.02,
    "HandReach-v3": 0.0,
    "HandManipulateBlockRotateZ-v1": 0.185,
    "HandManipulateEggRotate-v1": 0.09,
    "HandManipulatePenRotate-v1": 0.05,
    "HandManipulateBlockRotateZ_BooleanTouchSensors-v1": 0.205,
    "HandManipulateBlockRotateZ_ContinuousTouchSensors-v1": 0.24,
    "HandManipulateEggRotate_BooleanTouchSensors-v1": 0.115,
    "HandManipulateEggRotate_ContinuousTouchSensors-v1": 0.105,
    "HandManipulatePenRotate_BooleanTouchSensors-v1": 0.15,
    "HandManipulatePenRotate_ContinuousTouchSensors-v1": 0.095,
    "AntMaze_UMaze-v5": 0.44,
    "AntMaze_BigMaze_DGR-v5": 0.0,
    "PandaSlide-v3": 0.525,
    "PandaPickAndPlace-v3": 0.475,
    "PandaStack-v3": 0.11,
    "PandaFlip-v3": 0.52,
    "PandaSlideJoints-v3": 0.185,
    "PandaPickAndPlaceJoints-v3": 0.445,
    "PandaStackJoints-v3": 0.07,
    "PandaFlipJoints-v3": 0.14,
    "point_mass_hard": 0.2,
}

MIN_TEST_SUCCESS = 0.10
MAX_TEST_SUCCESS = 0.80
SELECTED_ENVIRONMENTS = frozenset(
    environment
    for environment, success in TEST_SUCCESS.items()
    if MIN_TEST_SUCCESS <= success < MAX_TEST_SUCCESS
)

# Use the final budgets shown in the completed CQRL/1 runs, including extensions.
EXPECTED_STEPS = {
    "manipulator_insert_ball": 500_000,
    "manipulator_insert_peg": 500_000,
    "dog_fetch": 500_000,
    "stacker_stack_2": 500_000,
    "HandReach-v3": 100_000,
    "HandManipulateBlockRotateZ-v1": 200_000,
    "HandManipulateEggRotate-v1": 200_000,
    "HandManipulatePenRotate-v1": 100_000,
    "HandManipulateBlockRotateZ_BooleanTouchSensors-v1": 200_000,
    "HandManipulateBlockRotateZ_ContinuousTouchSensors-v1": 200_000,
    "HandManipulateEggRotate_BooleanTouchSensors-v1": 200_000,
    "HandManipulateEggRotate_ContinuousTouchSensors-v1": 200_000,
    "HandManipulatePenRotate_BooleanTouchSensors-v1": 200_000,
    "HandManipulatePenRotate_ContinuousTouchSensors-v1": 200_000,
    "AntMaze_UMaze-v5": 400_000,
    "AntMaze_BigMaze_DGR-v5": 500_000,
    "PandaSlide-v3": 400_000,
    "PandaPickAndPlace-v3": 400_000,
    "PandaStack-v3": 100_000,
    "PandaFlip-v3": 400_000,
    "PandaSlideJoints-v3": 200_000,
    "PandaPickAndPlaceJoints-v3": 400_000,
    "PandaStackJoints-v3": 100_000,
    "PandaFlipJoints-v3": 100_000,
    "point_mass_hard": 500_000,
}

TARGET_ARGS = {
    "agent.actor.losses.min_dist.latent_goal_mode": "min",
    "agent.actor.losses.min_dist.latent_goal_lr": "1",
    "agent.actor.losses.min_dist.latent_goal_steps": "32",
}


def extra_arg_map(extra_args: str) -> dict[str, str]:
    tokens = shlex.split(extra_args)
    pairs = [token.split("=", 1) for token in tokens]
    if any(len(pair) != 2 for pair in pairs):
        raise ValueError(f"Malformed extra arguments: {extra_args}")
    keys = [pair[0] for pair in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate extra-argument keys: {extra_args}")
    return dict(pairs)


def replace_extra_args(extra_args: str) -> str:
    tokens = shlex.split(extra_args)
    seen: set[str] = set()
    rewritten = []
    for token in tokens:
        key = token.split("=", 1)[0]
        if key in TARGET_ARGS:
            token = f"{key}={TARGET_ARGS[key]}"
            seen.add(key)
        rewritten.append(token)
    missing = set(TARGET_ARGS) - seen
    if missing:
        raise ValueError(f"Missing CQRL inner-loop arguments: {sorted(missing)}")
    return " ".join(rewritten)


def source_tasks(path: Path) -> list[Task]:
    selected = [
        task for task in read_tasks(path)
        if task.env_name in SELECTED_ENVIRONMENTS
        and task.seed == SOURCE_SEED
        and "_CQRL-1-L_" in task.task_id
    ]
    counts = Counter(task.env_name for task in selected)
    invalid = {name: count for name, count in counts.items() if count != 1}
    missing = set(SELECTED_ENVIRONMENTS) - set(counts)
    if invalid or missing:
        raise ValueError(
            f"Expected one CQRL/1-L source per environment; "
            f"invalid={invalid}, missing={sorted(missing)}"
        )
    return selected


def transform_task(task: Task) -> Task:
    expected_steps = EXPECTED_STEPS[task.env_name]
    if int(task.steps) != expected_steps:
        raise ValueError(
            f"{task.env_name} requires final budget {expected_steps}, "
            f"got {task.steps}"
        )
    args = extra_arg_map(task.extra_args)
    expected_source_args = {
        "agent.actor.losses.min_dist.latent_goal_mode": "max",
        "agent.actor.losses.min_dist.latent_goal_lr": "0.01",
        "agent.actor.losses.min_dist.latent_goal_steps": "1",
    }
    for key, value in expected_source_args.items():
        if args.get(key) != value:
            raise ValueError(
                f"{task.task_id} requires source {key}={value}, got {args.get(key)}"
            )
    task_id = task.task_id.replace("_CQRL-1-L_", f"_{TARGET_VARIANT}_", 1)
    if not task_id.endswith(f"_s{SOURCE_SEED}"):
        raise ValueError(f"Unexpected source task ID: {task.task_id}")
    task_id = task_id[: -len(SOURCE_SEED)] + TARGET_SEED
    return replace(
        task,
        task_id=task_id,
        seed=TARGET_SEED,
        extra_args=replace_extra_args(task.extra_args),
    )


def generate_tasks(path: Path = DEFAULT_SOURCE) -> list[Task]:
    if set(TEST_SUCCESS) != set(EXPECTED_STEPS):
        raise ValueError("Success and budget environment sets do not match")
    if any(
        not MIN_TEST_SUCCESS <= TEST_SUCCESS[environment] < MAX_TEST_SUCCESS
        for environment in SELECTED_ENVIRONMENTS
    ):
        raise ValueError("The selected task set is outside the [10%, 80%) range")
    tasks = [transform_task(task) for task in source_tasks(path)]
    if len(tasks) != 15 or len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError(f"Expected 15 unique tasks, got {len(tasks)}")
    for task in tasks:
        args = extra_arg_map(task.extra_args)
        for key, value in TARGET_ARGS.items():
            if args.get(key) != value:
                raise ValueError(f"{task.task_id} requires {key}={value}")
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
    lines = [
        "# CQRL-L Min32/LR1 on CQRL/1 seed-1000 tasks with 0.10 <= test_succ < 0.80.",
        "# One matching training seed (1001); final source-task budgets are preserved.",
        *(task_line(task) for task in tasks),
    ]
    return "\n".join(lines) + "\n"


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    original_mode = path.stat().st_mode if path.exists() else None
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{path.name}.", suffix=".tmp", dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if original_mode is not None:
            os.chmod(temporary, original_mode)
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
        "# CQRL-L Min32/LR1: CQRL/1 seed-1000 0.10 <= test_succ < 0.80.",
        *(task_line(task) for task in additions),
    ]) + "\n"
    atomic_write(path, current + separator + block)
    return len(additions)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--append-to", type=Path)
    args = parser.parse_args()

    tasks = generate_tasks(args.source)
    atomic_write(args.output, render_tasks(tasks))
    print(f"Wrote {len(tasks)} tasks to {args.output}")
    if args.append_to is not None:
        added = append_tasks(args.append_to, tasks)
        print(f"Appended {added} new tasks to {args.append_to}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

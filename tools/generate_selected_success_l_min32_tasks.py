#!/usr/bin/env python3
"""Generate the selected L-scale comparison and Min32 main-table tasks."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, replace
import os
from pathlib import Path
import shlex
import sys
import tempfile
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quasimetric_rl.model_size import cqrl_agent_parameter_count
from tools.generate_extended_online_tasks import (
    ALGORITHMS as EXTENDED_ALGORITHMS,
    CHECKPOINT_ARGS as EXTENDED_CHECKPOINT_ARGS,
    ENVIRONMENTS as EXTENDED_ENVIRONMENTS,
    QRL_COMMON_ARGS as EXTENDED_QRL_COMMON_ARGS,
    algorithm_args as extended_algorithm_args,
    gcsl_uses_factorized_actions,
    parameter_count as extended_parameter_count,
    step_label,
)
from tools.generate_iclr_main_table_tasks import (
    CHECKPOINT_ARGS as MAIN_CHECKPOINT_ARGS,
    ENVIRONMENTS as MAIN_ENVIRONMENTS,
    QRL_COMMON_ARGS as MAIN_QRL_COMMON_ARGS,
)
from tools.generate_online_baseline_tasks import compact_parameter_count
from tools.run_qrl_queue import Task, read_tasks


TRAINING_SEEDS = tuple(range(1000, 1005))
SELECTED_OUTPUT = ROOT / "configs/selected_mid_success_l_6alg_7env_5seed.tsv"
MAIN_MIN32_OUTPUT = ROOT / "configs/cqrl_min32_lr1_main7_5seed.tsv"
DEFAULT_QUEUE = ROOT / "runs/qrl_queue/tasks.tsv"

BASE_ALGORITHM_KEYS = (
    "td_infonce",
    "gcsl",
    "c_learning",
    "crl",
    "qrl",
)
MIN32_KEY = "cqrl_min32_lr1"
SELECTED_ALGORITHM_KEYS = (*BASE_ALGORITHM_KEYS, MIN32_KEY)

CQRL_MIN32_ARGS = (
    "agent.quasimetric_critic.losses.separate_latent_dynamics=false",
    "agent.quasimetric_critic.losses.latent_dynamics.distance=iqe_mse",
    "agent.quasimetric_critic.losses.latent_dynamics.mse_weight=1.0",
    "agent.quasimetric_critic.losses.latent_dynamics.iqe_weight=1.0",
    "agent.quasimetric_critic.model.encoder.branch_normalization=none",
    "agent.actor.losses.min_dist.latent_goal_mode=min",
    "agent.actor.losses.min_dist.latent_goal_keep_best=true",
    "agent.actor.losses.min_dist.latent_goal_lr=1",
    "agent.actor.losses.min_dist.latent_goal_search=direct",
    "agent.actor.losses.min_dist.latent_goal_optim=sgd",
    "agent.actor.losses.min_dist.latent_goal_steps=32",
)


@dataclass(frozen=True)
class SelectedProtocol:
    environment_name: str
    initial_steps: int
    total_steps: int
    save_steps: int
    validation_episodes: int
    test_episodes: int


# These are the strict 0.40 < test_succ < 0.90 CQRL/1 seed-1000 runs.
SELECTED_PROTOCOLS = (
    SelectedProtocol(
        "PointMaze_Medium-v3", 200_400, 200_400, 20_000, 100, 100,
    ),
    SelectedProtocol(
        "AntMaze_UMaze-v5", 200_200, 400_000, 20_000, 100, 100,
    ),
    SelectedProtocol(
        "PandaSlide-v3", 100_000, 400_000, 20_000, 200, 200,
    ),
    SelectedProtocol(
        "PandaPickAndPlace-v3", 100_000, 400_000, 20_000, 200, 200,
    ),
    SelectedProtocol(
        "PandaFlip-v3", 100_000, 400_000, 20_000, 200, 200,
    ),
    SelectedProtocol(
        "PandaPushJoints-v3", 100_000, 400_000, 20_000, 200, 200,
    ),
    SelectedProtocol(
        "PandaPickAndPlaceJoints-v3", 100_000, 400_000, 20_000, 200, 200,
    ),
)


LEGACY_MAIN_MIN32_SPECS = (
    ("fetchpush", "m", 100_000),
    ("fetchslide", "l", 500_000),
    ("fetchpickandplace", "m", 200_000),
    ("reacher_hard", "m", 200_000),
    ("maze2d_large", "m", 200_000),
    ("pusher_v4", "l", 500_000),
    ("antnavigate_v4", "l", 500_000),
)

PREVIOUS_MAIN_MIN32_SPECS = tuple(
    (environment_key, "l", total_steps)
    for environment_key, _model_size, total_steps in LEGACY_MAIN_MIN32_SPECS
)

MAIN_MIN32_SPECS = (
    ("fetchpush", "l", 200_000),
    *PREVIOUS_MAIN_MIN32_SPECS[1:],
)


def extra_arg_map(task: Task) -> dict[str, str]:
    tokens = shlex.split(task.extra_args)
    pairs = [token.split("=", 1) for token in tokens]
    if any(len(pair) != 2 for pair in pairs):
        raise ValueError(f"Malformed extra arguments: {task.extra_args}")
    keys = [pair[0] for pair in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate extra-argument keys: {task.task_id}")
    return dict(pairs)


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


def _extended_environment_map():
    return {environment.name: environment for environment in EXTENDED_ENVIRONMENTS}


def _extended_algorithm_map():
    return {algorithm.key: algorithm for algorithm in EXTENDED_ALGORITHMS}


def _selected_algorithm_label(algorithm_key: str) -> str:
    if algorithm_key == MIN32_KEY:
        return "CQRL+Min32-L-InnerLR1"
    return f"{_extended_algorithm_map()[algorithm_key].label}-L"


def _selected_algorithm_args(algorithm_key: str, environment) -> tuple[str, ...]:
    if algorithm_key == MIN32_KEY:
        return (
            "+cqrl_model_size=l",
            *EXTENDED_QRL_COMMON_ARGS,
            *CQRL_MIN32_ARGS,
        )
    algorithm = _extended_algorithm_map()[algorithm_key]
    return extended_algorithm_args(algorithm, environment, "l")


def _selected_parameter_count(algorithm_key: str, environment) -> int:
    if algorithm_key == MIN32_KEY:
        return cqrl_agent_parameter_count(
            environment.state_dim,
            environment.action_dim,
            environment.goal_dim,
            "l",
        )
    return extended_parameter_count(algorithm_key, environment, "l")


def generate_selected_tasks() -> list[Task]:
    environments = _extended_environment_map()
    tasks = []
    for protocol in SELECTED_PROTOCOLS:
        environment = environments[protocol.environment_name]
        initial_label = step_label(protocol.initial_steps)
        checkpoint_label = f"{protocol.save_steps // 1000}kckpt"
        for algorithm_key in SELECTED_ALGORITHM_KEYS:
            factorized_tag = ""
            if (
                algorithm_key == "gcsl"
                and gcsl_uses_factorized_actions(environment)
            ):
                factorized_tag = "factorized_"
            params = compact_parameter_count(
                _selected_parameter_count(algorithm_key, environment)
            )
            for seed in TRAINING_SEEDS:
                task_id = (
                    f"extended_v1_{_selected_algorithm_label(algorithm_key)}_"
                    f"{initial_label}_{checkpoint_label}_"
                    f"val{protocol.validation_episodes}_"
                    f"test{protocol.test_episodes}_{factorized_tag}"
                    f"{environment.slug}_online_s{seed}"
                )
                extra_args = " ".join((
                    f"env.kind={environment.kind}",
                    f"env.init_num_transitions={protocol.initial_steps}",
                    f"env.increment_num_transitions={protocol.initial_steps}",
                    *_selected_algorithm_args(algorithm_key, environment),
                    "interaction.validation_seed=1000",
                    "interaction.test_seed=2000000",
                    f"interaction.num_eval_episodes={protocol.validation_episodes}",
                    f"interaction.num_test_episodes={protocol.test_episodes}",
                    f"save_steps={protocol.save_steps}",
                    *EXTENDED_CHECKPOINT_ARGS,
                ))
                tasks.append(Task(
                    task_id=task_id,
                    mode="online",
                    env_name=environment.name,
                    seed=str(seed),
                    steps=str(protocol.total_steps),
                    params=params,
                    extra_args=extra_args,
                ))
    validate_selected_tasks(tasks)
    return tasks


def _main_min32_environments(specs=MAIN_MIN32_SPECS):
    by_key = {environment.key: environment for environment in MAIN_ENVIRONMENTS}
    return tuple(
        replace(
            by_key[key],
            model_size=model_size,
            total_steps=total_steps,
        )
        for key, model_size, total_steps in specs
    )


def generate_main_min32_tasks(specs=MAIN_MIN32_SPECS) -> list[Task]:
    tasks = []
    for environment in _main_min32_environments(specs):
        size_label = environment.model_size.upper()
        total_label = step_label(environment.total_steps)
        checkpoint_label = f"{environment.save_steps // 1000}kckpt"
        params = compact_parameter_count(cqrl_agent_parameter_count(
            environment.state_dim,
            environment.action_dim,
            environment.goal_dim,
            environment.model_size,
        ))
        for seed in TRAINING_SEEDS:
            task_id = (
                f"iclr_main_v1_CQRL+Min32-{size_label}-InnerLR1_"
                f"{total_label}_{checkpoint_label}_"
                f"val{environment.validation_episodes}_"
                f"test{environment.test_episodes}_{environment.slug}_"
                f"online_s{seed}"
            )
            extra_args = " ".join((
                f"env.kind={environment.kind}",
                f"+cqrl_model_size={environment.model_size}",
                *MAIN_QRL_COMMON_ARGS,
                *CQRL_MIN32_ARGS,
                f"interaction.validation_seed={environment.validation_seed}",
                f"interaction.test_seed={environment.test_seed}",
                f"interaction.num_eval_episodes={environment.validation_episodes}",
                f"interaction.num_test_episodes={environment.test_episodes}",
                "eval_steps=null",
                f"save_steps={environment.save_steps}",
                *MAIN_CHECKPOINT_ARGS,
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
    validate_main_min32_tasks(tasks, specs)
    return tasks


def _validate_unique(tasks: Sequence[Task], expected: int) -> None:
    if len(tasks) != expected:
        raise ValueError(f"Expected {expected} tasks, got {len(tasks)}")
    duplicates = [
        task_id
        for task_id, count in Counter(task.task_id for task in tasks).items()
        if count > 1
    ]
    if duplicates:
        raise ValueError(f"Duplicate generated task IDs: {duplicates}")


def _validate_min32_args(task: Task, model_size: str) -> None:
    args = extra_arg_map(task)
    required = {
        "+cqrl_model_size": model_size,
        "agent.actor.losses.min_dist.latent_goal_mode": "min",
        "agent.actor.losses.min_dist.latent_goal_steps": "32",
        "agent.actor.losses.min_dist.latent_goal_lr": "1",
        "agent.actor.losses.min_dist.latent_goal_keep_best": "true",
        "agent.actor.losses.min_dist.latent_goal_search": "direct",
        "agent.actor.losses.min_dist.latent_goal_optim": "sgd",
    }
    for key, value in required.items():
        if args.get(key) != value:
            raise ValueError(
                f"{task.task_id} requires {key}={value}, got {args.get(key)}"
            )


def validate_selected_tasks(tasks: Sequence[Task]) -> None:
    expected = (
        len(SELECTED_PROTOCOLS)
        * len(SELECTED_ALGORITHM_KEYS)
        * len(TRAINING_SEEDS)
    )
    _validate_unique(tasks, expected)
    protocols = {
        protocol.environment_name: protocol for protocol in SELECTED_PROTOCOLS
    }
    for task in tasks:
        protocol = protocols[task.env_name]
        args = extra_arg_map(task)
        if task.steps != str(protocol.total_steps):
            raise ValueError(f"Invalid total steps for {task.task_id}")
        if not 21.0 <= float(task.params.removesuffix("m")) <= 23.0:
            raise ValueError(f"Expected L-scale parameter count: {task.task_id}")
        if args.get("save_steps") != str(protocol.save_steps):
            raise ValueError(f"Invalid checkpoint interval for {task.task_id}")
        if "CQRL+Min32" in task.task_id:
            _validate_min32_args(task, "l")


def validate_main_min32_tasks(
        tasks: Sequence[Task], specs=MAIN_MIN32_SPECS) -> None:
    environments = {
        environment.name: environment
        for environment in _main_min32_environments(specs)
    }
    _validate_unique(tasks, len(environments) * len(TRAINING_SEEDS))
    for task in tasks:
        environment = environments[task.env_name]
        if task.steps != str(environment.total_steps):
            raise ValueError(f"Invalid total steps for {task.task_id}")
        _validate_min32_args(task, environment.model_size)


def render_tasks(tasks: Sequence[Task], description: str) -> str:
    lines = [
        "# task_id\tmode\tenv_name\tseed\tsteps\tparams\textra_args",
        f"# {description}",
        "# Training seeds: 1000-1004.",
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


def _new_tasks(existing: dict[str, Task], tasks: Sequence[Task]) -> list[Task]:
    additions = []
    for task in tasks:
        previous = existing.get(task.task_id)
        if previous is not None:
            if previous != task:
                raise ValueError(
                    f"Existing task ID has a different definition: {task.task_id}"
                )
            continue
        existing[task.task_id] = task
        additions.append(task)
    return additions


def completed_task_ids(status_dir: Path) -> set[str]:
    completed = set()
    if not status_dir.exists():
        return completed
    for path in status_dir.glob("*.status"):
        status = {}
        for line in path.read_text().splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                status[key] = value
        if status.get("state") == "DONE":
            completed.add(status.get("task_id", path.stem))
    return completed


def task_states(status_dir: Path) -> dict[str, str]:
    states = {}
    if not status_dir.exists():
        return states
    for path in status_dir.glob("*.status"):
        status = {}
        for line in path.read_text().splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                status[key] = value
        states[status.get("task_id", path.stem)] = status.get(
            "state", "UNKNOWN"
        )
    return states


def replace_pending_main_min32_tasks(
        path: Path, status_dir: Path | None = None) -> int:
    if status_dir is None:
        status_dir = path.parent / "status"
    legacy = {}
    for specs in (LEGACY_MAIN_MIN32_SPECS, PREVIOUS_MAIN_MIN32_SPECS):
        for task in generate_main_min32_tasks(specs):
            previous = legacy.setdefault(task.task_id, task)
            if previous != task:
                raise ValueError(
                    f"Legacy Min32 task ID has conflicting definitions: "
                    f"{task.task_id}"
                )
    current_ids = {
        task.task_id for task in generate_main_min32_tasks(MAIN_MIN32_SPECS)
    }
    obsolete_ids = set(legacy).difference(current_ids)
    states = task_states(status_dir)
    existing = {task.task_id: task for task in read_tasks(path)}
    removable = []
    for task_id in sorted(obsolete_ids.intersection(existing)):
        if existing[task_id] != legacy[task_id]:
            raise ValueError(
                f"Legacy Min32 task has a different definition: {task_id}"
            )
        state = states.get(task_id, "PENDING")
        if state != "PENDING":
            raise ValueError(
                f"Refusing to replace Min32 task in state {state}: {task_id}"
            )
        removable.append(task_id)
    if not removable:
        return 0
    removable_set = set(removable)
    lines = path.read_text().splitlines()
    retained = [
        line for line in lines
        if not line or line.startswith("#")
        or line.split("\t", 1)[0] not in removable_set
    ]
    content = "\n".join(retained) + "\n"
    content = content.replace(
        "# CQRL Min32/LR1 main seven environments, requested M/L budgets, five seeds.",
        "# CQRL Min32/LR1 main seven environments, residual L scale, five seeds.",
    )
    atomic_write(path, content)
    return len(removable)


def append_task_sets(
        path: Path, selected_tasks: Sequence[Task],
        main_min32_tasks: Sequence[Task],
        status_dir: Path | None = None) -> tuple[int, int]:
    existing = {task.task_id: task for task in read_tasks(path)}
    if status_dir is None:
        status_dir = path.parent / "status"
    completed = completed_task_ids(status_dir)
    selected_additions = _new_tasks(
        existing,
        [task for task in selected_tasks if task.task_id not in completed],
    )
    main_additions = _new_tasks(
        existing,
        [task for task in main_min32_tasks if task.task_id not in completed],
    )
    if not selected_additions and not main_additions:
        return 0, 0
    current = path.read_text() if path.exists() else ""
    separator = "" if not current or current.endswith("\n\n") else (
        "\n" if current.endswith("\n") else "\n\n"
    )
    blocks = []
    if selected_additions:
        blocks.append("\n".join((
            "# Selected 40%-90% CQRL/1 environments: six L-scale algorithms, five seeds.",
            *(task_line(task) for task in selected_additions),
        )))
    if main_additions:
        blocks.append("\n".join((
            "# CQRL Min32/LR1 main seven environments, requested M/L budgets, five seeds.",
            *(task_line(task) for task in main_additions),
        )))
    atomic_write(path, current + separator + "\n\n".join(blocks) + "\n")
    return len(selected_additions), len(main_additions)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-output", type=Path, default=SELECTED_OUTPUT)
    parser.add_argument("--main-min32-output", type=Path, default=MAIN_MIN32_OUTPUT)
    parser.add_argument("--append-to", type=Path)
    parser.add_argument("--replace-pending-main-min32", action="store_true")
    args = parser.parse_args()

    selected_tasks = generate_selected_tasks()
    main_min32_tasks = generate_main_min32_tasks()
    atomic_write(args.selected_output, render_tasks(
        selected_tasks,
        "Strict 40% < CQRL/1 test success < 90%; six L-scale algorithms.",
    ))
    atomic_write(args.main_min32_output, render_tasks(
        main_min32_tasks,
        "CQRL Min32/LR1 on the seven requested main-table environments.",
    ))
    print(f"Wrote {len(selected_tasks)} tasks to {args.selected_output}")
    print(f"Wrote {len(main_min32_tasks)} tasks to {args.main_min32_output}")
    if args.append_to is not None:
        if args.replace_pending_main_min32:
            replaced = replace_pending_main_min32_tasks(args.append_to)
            print(f"Removed {replaced} pending obsolete Min32 tasks")
        selected_added, main_added = append_task_sets(
            args.append_to, selected_tasks, main_min32_tasks,
        )
        print(
            f"Added {selected_added} selected L-scale tasks and "
            f"{main_added} main Min32 tasks to {args.append_to}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

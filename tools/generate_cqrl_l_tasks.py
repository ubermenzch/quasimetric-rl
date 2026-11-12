#!/usr/bin/env python3
"""Generate the six-environment CQRL/1-L experiment set."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quasimetric_rl.model_size import cqrl_agent_parameter_count
from tools.generate_extended_online_tasks import (
    CHECKPOINT_ARGS,
    CQRL_ARGS,
    QRL_COMMON_ARGS,
)
from tools.run_qrl_queue import Task, read_tasks


DEFAULT_OUTPUT = ROOT / 'configs/cqrl_inner1_l_6env_5seed.tsv'
TRAINING_SEEDS = tuple(range(1000, 1005))
MODEL_SIZE_LEVEL = 'l'
INNER_STEPS = 1


@dataclass(frozen=True)
class Environment:
    kind: str
    name: str
    slug: str
    state_dim: int
    action_dim: int
    goal_dim: int
    total_steps: int
    save_steps: int
    validation_episodes: int
    validation_seed: int
    test_episodes: int
    test_seed: int


# Budgets and evaluation ranges match the corresponding cells in the existing
# CQRL ablation table. Capacity is intentionally overridden to L for all six.
ENVIRONMENTS = (
    Environment(
        'gcrl', 'FetchPush', 'fetchpush', 25, 4, 3,
        100_000, 20_000, 1000, 1000, 1000, 2000,
    ),
    Environment(
        'gcrl', 'FetchSlide', 'fetchslide', 25, 4, 3,
        500_000, 50_000, 500, 1000, 1000, 1500,
    ),
    Environment(
        'gcrl', 'FetchPickAndPlace', 'fetchpickandplace', 25, 4, 3,
        200_000, 20_000, 1000, 1000, 1000, 2000,
    ),
    Environment(
        'dmc', 'reacher_hard', 'dmc_reacher_hard', 6, 2, 2,
        200_000, 20_000, 1000, 1000, 1000, 2000,
    ),
    Environment(
        'online_maze', 'maze2d-large', 'maze2d_large', 4, 2, 2,
        200_000, 20_000, 1000, 1000, 1000, 2000,
    ),
    Environment(
        'gym_mujoco', 'Pusher-v4', 'pusher_v4', 20, 7, 3,
        500_000, 50_000, 500, 1000, 1000, 1500,
    ),
)


def compact_parameter_count(count: int) -> str:
    return f'{count / 1_000_000:.1f}m'


def extra_arg_map(task: Task) -> dict[str, str]:
    tokens = shlex.split(task.extra_args)
    keys = [token.split('=', 1)[0] for token in tokens]
    if len(keys) != len(set(keys)):
        raise ValueError(f'Duplicate extra-argument keys in {task.task_id}')
    return dict(token.split('=', 1) for token in tokens)


def task_line(task: Task) -> str:
    return '\t'.join((
        task.task_id,
        task.mode,
        task.env_name,
        task.seed,
        task.steps,
        task.params,
        task.extra_args,
    ))


def generate_tasks() -> list[Task]:
    tasks = []
    for environment in ENVIRONMENTS:
        params = compact_parameter_count(cqrl_agent_parameter_count(
            environment.state_dim,
            environment.action_dim,
            environment.goal_dim,
            MODEL_SIZE_LEVEL,
        ))
        step_label = f'{environment.total_steps // 1000}k'
        checkpoint_label = f'{environment.save_steps // 1000}kckpt'
        for seed in TRAINING_SEEDS:
            task_id = (
                f'cqrl_l_v1_CQRL-1-L_{step_label}_{checkpoint_label}_'
                f'val{environment.validation_episodes}_'
                f'test{environment.test_episodes}_{environment.slug}_'
                f'online_s{seed}'
            )
            extra_args = ' '.join((
                f'env.kind={environment.kind}',
                f'env.init_num_transitions={environment.total_steps}',
                f'env.increment_num_transitions={environment.total_steps}',
                f'+cqrl_model_size={MODEL_SIZE_LEVEL}',
                *QRL_COMMON_ARGS,
                *CQRL_ARGS,
                f'agent.actor.losses.min_dist.latent_goal_steps={INNER_STEPS}',
                f'interaction.validation_seed={environment.validation_seed}',
                f'interaction.test_seed={environment.test_seed}',
                f'interaction.num_eval_episodes={environment.validation_episodes}',
                f'interaction.num_test_episodes={environment.test_episodes}',
                f'save_steps={environment.save_steps}',
                *CHECKPOINT_ARGS,
            ))
            tasks.append(Task(
                task_id=task_id,
                mode='online',
                env_name=environment.name,
                seed=str(seed),
                steps=str(environment.total_steps),
                params=params,
                extra_args=extra_args,
            ))
    validate_tasks(tasks)
    return tasks


def validate_tasks(tasks: list[Task]) -> None:
    expected_count = len(ENVIRONMENTS) * len(TRAINING_SEEDS)
    if len(tasks) != expected_count:
        raise ValueError(f'Expected {expected_count} tasks, got {len(tasks)}')
    duplicate_ids = [
        task_id for task_id, count in Counter(
            task.task_id for task in tasks
        ).items() if count > 1
    ]
    if duplicate_ids:
        raise ValueError(f'Duplicate generated task IDs: {duplicate_ids}')

    for environment in ENVIRONMENTS:
        group = [task for task in tasks if task.env_name == environment.name]
        if {int(task.seed) for task in group} != set(TRAINING_SEEDS):
            raise ValueError(f'Invalid seeds for {environment.name}')
        expected_params = compact_parameter_count(
            cqrl_agent_parameter_count(
                environment.state_dim,
                environment.action_dim,
                environment.goal_dim,
                MODEL_SIZE_LEVEL,
            )
        )
        required_args = {
            'env.kind': environment.kind,
            'env.init_num_transitions': str(environment.total_steps),
            'env.increment_num_transitions': str(environment.total_steps),
            '+cqrl_model_size': MODEL_SIZE_LEVEL,
            'agent.actor.losses.min_dist.latent_goal_mode': 'max',
            'agent.actor.losses.min_dist.latent_goal_steps': str(INNER_STEPS),
            'interaction.validation_seed': str(environment.validation_seed),
            'interaction.test_seed': str(environment.test_seed),
            'interaction.num_eval_episodes': str(environment.validation_episodes),
            'interaction.num_test_episodes': str(environment.test_episodes),
            'save_steps': str(environment.save_steps),
            'eval_steps': 'null',
            'keep_only_latest_checkpoint': 'true',
            'keep_only_best_and_final_checkpoints': 'true',
            'save_replay_buffer': 'true',
            'save_final_replay_buffer': 'true',
            'resume_if_possible': 'true',
        }
        for task in group:
            if (
                    task.mode != 'online'
                    or task.steps != str(environment.total_steps)
                    or task.params != expected_params):
                raise ValueError(f'Invalid metadata in {task.task_id}')
            args = extra_arg_map(task)
            for key, value in required_args.items():
                if args.get(key) != value:
                    raise ValueError(
                        f'{task.task_id} requires {key}={value}, '
                        f'got {args.get(key)}'
                    )


def render_tasks(tasks: list[Task]) -> str:
    lines = [
        '# task_id\tmode\tenv_name\tseed\tsteps\tparams\textra_args',
        '# CQRL/1-L: 6 environments x 5 training seeds.',
        '# L resolves to the residual CQRL architecture.',
        '# Budgets and evaluation ranges match the existing ablation-table cells.',
    ]
    lines.extend(task_line(task) for task in tasks)
    return '\n'.join(lines) + '\n'


def append_tasks(path: Path, tasks: list[Task]) -> int:
    existing_ids = {task.task_id for task in read_tasks(path)}
    additions = [task for task in tasks if task.task_id not in existing_ids]
    if not additions:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    current = path.read_text() if path.exists() else ''
    separator = '' if not current or current.endswith('\n\n') else (
        '\n' if current.endswith('\n') else '\n\n'
    )
    block = '\n'.join([
        '# CQRL/1-L six-environment experiment set.',
        *(task_line(task) for task in additions),
    ]) + '\n'
    original_mode = path.stat().st_mode if path.exists() else None
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f'{path.name}.', suffix='.tmp', dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, 'w') as handle:
            handle.write(current + separator + block)
            handle.flush()
            os.fsync(handle.fileno())
        if original_mode is not None:
            os.chmod(temporary, original_mode)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return len(additions)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--append-to', type=Path)
    args = parser.parse_args()

    tasks = generate_tasks()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_tasks(tasks))
    print(f'Wrote {len(tasks)} tasks to {args.output}')
    if args.append_to is not None:
        added = append_tasks(args.append_to, tasks)
        print(f'Appended {added} new tasks to {args.append_to}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

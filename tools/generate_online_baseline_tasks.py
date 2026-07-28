#!/usr/bin/env python3
"""Generate the four reference-baseline tasks for the seven validated envs."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_qrl_queue import Task


DEFAULT_OUTPUT = ROOT / 'configs/gcrl_baselines_4alg_7env_5seed_200k.tsv'
TOTAL_ENV_STEPS = 200_000
TRAINING_SEEDS = tuple(range(1000, 1005))

ENVIRONMENTS = (
    ('gcrl', 'FetchReach', 'fetchreach', 10, 4, 3),
    ('gcrl', 'FetchPush', 'fetchpush', 25, 4, 3),
    ('gcrl', 'FetchSlide', 'fetchslide', 25, 4, 3),
    ('gcrl', 'FetchPickAndPlace', 'fetchpickandplace', 25, 4, 3),
    ('dmc', 'reacher_easy', 'dmc_reacher_easy', 6, 2, 2),
    ('dmc', 'reacher_hard', 'dmc_reacher_hard', 6, 2, 2),
    ('gym_mujoco', 'Reacher-v4', 'reacher_v4', 8, 2, 2),
)

ALGORITHMS = (
    ('TD-InfoNCE', 'td_infonce'),
    ('CRL', 'crl'),
    ('GCBC', 'gcbc'),
    ('C-Learning', 'c_learning'),
)

COMMON_ARGS = (
    'interaction.exploration_eps=0',
    'interaction.validation_seed=1000',
    'interaction.test_seed=2000000',
    'eval_steps=null',
    'save_steps=20000',
    'keep_only_latest_checkpoint=false',
    'save_replay_buffer=true',
    'save_final_replay_buffer=true',
)


def evaluation_episode_count(env_kind: str, env_name: str) -> int:
    if env_kind == 'dmc':
        return 100
    if (env_kind, env_name) == ('gcrl', 'FetchReach'):
        return 1000
    return 200


def _mlp_parameters(input_dim: int, hidden_sizes, output_dim: int) -> int:
    sizes = (input_dim, *hidden_sizes, output_dim)
    return sum(
        (left + 1) * right
        for left, right in zip(sizes, sizes[1:])
    )


def baseline_parameter_count(
        algorithm: str, state_dim: int, action_dim: int,
        goal_dim: int) -> int:
    action_output_dim = 2 * action_dim
    if algorithm == 'td_infonce':
        hidden = (512, 512, 512, 512)
        representation_dim = 16
        return (
            _mlp_parameters(state_dim + goal_dim, hidden, action_output_dim)
            + 2 * _mlp_parameters(
                state_dim + action_dim + goal_dim, hidden, representation_dim,
            )
            + 2 * _mlp_parameters(goal_dim, hidden, representation_dim)
        )
    if algorithm == 'crl':
        hidden = (256, 256)
        representation_dim = 64
        return (
            _mlp_parameters(state_dim + goal_dim, hidden, action_output_dim)
            + _mlp_parameters(state_dim + action_dim, hidden, representation_dim)
            + _mlp_parameters(goal_dim, hidden, representation_dim)
            + 1  # adaptive log-alpha
        )
    if algorithm == 'gcbc':
        return _mlp_parameters(
            state_dim + goal_dim, (400, 300), action_output_dim,
        )
    if algorithm == 'c_learning':
        hidden = (256, 256)
        return (
            _mlp_parameters(state_dim + goal_dim, hidden, action_output_dim)
            + 2 * _mlp_parameters(
                state_dim + action_dim + goal_dim, hidden, 1,
            )
        )
    raise ValueError(f'Unknown algorithm: {algorithm!r}')


def compact_parameter_count(count: int) -> str:
    if count >= 1_000_000:
        return f'{count / 1_000_000:.1f}m'
    if count >= 1_000:
        return f'{count / 1_000:.1f}k'
    return str(count)


def generate_tasks() -> list[Task]:
    tasks = []
    for env_kind, env_name, env_slug, state_dim, action_dim, goal_dim in ENVIRONMENTS:
        evaluation_episodes = evaluation_episode_count(env_kind, env_name)
        for display_name, algorithm in ALGORITHMS:
            parameter_count = baseline_parameter_count(
                algorithm, state_dim, action_dim, goal_dim,
            )
            for seed in TRAINING_SEEDS:
                task_id = (
                    f'reference_{display_name}_200k_20kckpt_'
                    f'val{evaluation_episodes}_test{evaluation_episodes}_'
                    f'{env_slug}_online_s{seed}'
                )
                extra_args = ' '.join((
                    f'env.kind={env_kind}',
                    f'agent.algorithm={algorithm}',
                    f'interaction.num_eval_episodes={evaluation_episodes}',
                    f'interaction.num_test_episodes={evaluation_episodes}',
                    *COMMON_ARGS,
                ))
                tasks.append(Task(
                    task_id=task_id,
                    mode='online',
                    env_name=env_name,
                    seed=str(seed),
                    steps=str(TOTAL_ENV_STEPS),
                    params=compact_parameter_count(parameter_count),
                    extra_args=extra_args,
                ))
    validate_tasks(tasks)
    return tasks


def validate_tasks(tasks: list[Task]) -> None:
    expected = len(ENVIRONMENTS) * len(ALGORITHMS) * len(TRAINING_SEEDS)
    if len(tasks) != expected:
        raise ValueError(f'Expected {expected} tasks, got {len(tasks)}')
    counts = Counter(task.task_id for task in tasks)
    duplicates = [task_id for task_id, count in counts.items() if count > 1]
    if duplicates:
        raise ValueError(f'Duplicate generated task IDs: {duplicates}')


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


def render_tasks(tasks: list[Task]) -> str:
    lines = [
        '# task_id\tmode\tenv_name\tseed\tsteps\tparams\textra_args',
        '# Four reference baselines x seven validated environments x five seeds.',
        '# Checkpoint-bound validation every 20k; test runs only on the selected best model.',
    ]
    lines.extend(task_line(task) for task in tasks)
    return '\n'.join(lines) + '\n'


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    tasks = generate_tasks()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_tasks(tasks))
    print(f'Wrote {len(tasks)} tasks to {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

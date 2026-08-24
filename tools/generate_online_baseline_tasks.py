#!/usr/bin/env python3
"""Generate the four reference-baseline tasks for the seven validated envs."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quasimetric_rl.model_size import load_model_size_preset
from quasimetric_rl.modules.gcrl_baselines import (
    FETCH_MANIPULATION_ENVIRONMENTS,
    GCRLBaselinesConf,
    resolve_baseline_goal_dims,
)
from tools.run_qrl_queue import Task, read_tasks


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
    ('GCSL', 'gcsl'),
    ('C-Learning', 'c_learning'),
)
MODEL_SIZE_LEVEL = 'm'
MODEL_SIZE_FAMILIES = {
    'td_infonce': 'td_infonce',
    'crl': 'crl',
    'gcsl': 'gcsl',
    'c_learning': 'c_learning',
}

COMMON_ARGS = (
    'batch_size=256',
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
        goal_dim: int, *, model_size_level: str = MODEL_SIZE_LEVEL,
        gcsl_action_discretization: str = 'joint') -> int:
    family = MODEL_SIZE_FAMILIES.get(algorithm)
    if family is None:
        raise ValueError(f'Unknown algorithm: {algorithm!r}')
    preset = load_model_size_preset(family, model_size_level)
    config_name = 'gcbc' if algorithm == 'gcsl' else algorithm
    conf = getattr(preset.baselines, config_name)
    source_conf = getattr(GCRLBaselinesConf(), config_name)
    hidden = tuple(map(int, conf.hidden_sizes))
    action_output_dim = 2 * action_dim
    if algorithm == 'td_infonce':
        representation_dim = source_conf.representation_dim
        return (
            _mlp_parameters(state_dim + goal_dim, hidden, action_output_dim)
            + 2 * _mlp_parameters(
                state_dim + action_dim + goal_dim, hidden, representation_dim,
            )
            + 2 * _mlp_parameters(goal_dim, hidden, representation_dim)
        )
    if algorithm == 'crl':
        representation_dim = source_conf.representation_dim
        return (
            _mlp_parameters(state_dim + goal_dim, hidden, action_output_dim)
            + _mlp_parameters(state_dim + action_dim, hidden, representation_dim)
            + _mlp_parameters(goal_dim, hidden, representation_dim)
        )
    if algorithm == 'gcsl':
        if gcsl_action_discretization == 'joint':
            action_output_dim = source_conf.action_granularity ** action_dim
        elif gcsl_action_discretization == 'factorized':
            action_output_dim = source_conf.action_granularity * action_dim
        else:
            raise ValueError(
                'gcsl_action_discretization must be "joint" or '
                f'"factorized", got {gcsl_action_discretization!r}'
            )
        return _mlp_parameters(
            state_dim + goal_dim, hidden, action_output_dim,
        )
    if algorithm == 'c_learning':
        return (
            _mlp_parameters(state_dim + goal_dim, hidden, action_output_dim)
            + 2 * _mlp_parameters(
                state_dim + action_dim + goal_dim, hidden, 1,
            )
        )
    raise AssertionError(algorithm)


def compact_parameter_count(count: int) -> str:
    if count >= 1_000_000:
        return f'{count / 1_000_000:.1f}m'
    if count >= 1_000:
        return f'{count / 1_000:.1f}k'
    return str(count)


def generate_tasks(algorithms: Iterable[str] | None = None) -> list[Task]:
    selected = set(MODEL_SIZE_FAMILIES if algorithms is None else algorithms)
    unknown = selected.difference(MODEL_SIZE_FAMILIES)
    if unknown:
        raise ValueError(f'Unknown algorithms: {sorted(unknown)}')
    tasks = []
    for env_kind, env_name, env_slug, state_dim, action_dim, goal_dim in ENVIRONMENTS:
        evaluation_episodes = evaluation_episode_count(env_kind, env_name)
        for display_name, algorithm in ALGORITHMS:
            if algorithm not in selected:
                continue
            if algorithm == 'crl':
                task_version_tag = 'originalcrl2022_'
            elif (env_kind, env_name) in FETCH_MANIPULATION_ENVIRONMENTS:
                task_version_tag = 'goalreprv2_'
            else:
                task_version_tag = ''
            conditioning_goal_dim = len(resolve_baseline_goal_dims(
                algorithm,
                env_kind=env_kind,
                env_name=env_name,
                state_dim=state_dim,
                success_goal_dims=tuple(range(goal_dim)),
            ))
            parameter_count = baseline_parameter_count(
                algorithm, state_dim, action_dim, conditioning_goal_dim,
            )
            for seed in TRAINING_SEEDS:
                task_id = (
                    f'reference_{display_name}-M_200k_20kckpt_'
                    f'val{evaluation_episodes}_test{evaluation_episodes}_'
                    f'{task_version_tag}{env_slug}_online_s{seed}'
                )
                extra_args = ' '.join((
                    f'env.kind={env_kind}',
                    f'agent.algorithm={algorithm}',
                    f'+{MODEL_SIZE_FAMILIES[algorithm]}_model_size={MODEL_SIZE_LEVEL}',
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
    validate_tasks(tasks, selected)
    return tasks


def validate_tasks(tasks: list[Task], algorithms: Iterable[str] | None = None) -> None:
    selected = set(MODEL_SIZE_FAMILIES if algorithms is None else algorithms)
    expected = len(ENVIRONMENTS) * len(selected) * len(TRAINING_SEEDS)
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
        '# Reward-free M-level baselines x seven validated environments x five seeds.',
        '# Every trainable model contains 4.0M-4.5M parameters.',
        '# Checkpoint-bound validation every 20k; test runs only on the selected best model.',
    ]
    lines.extend(task_line(task) for task in tasks)
    return '\n'.join(lines) + '\n'


def append_tasks(path: Path, tasks: list[Task]) -> int:
    existing_ids = {task.task_id for task in read_tasks(path)}
    additions = [task for task in tasks if task.task_id not in existing_ids]
    if not additions:
        return 0
    current = path.read_text() if path.exists() else ''
    separator = '' if not current or current.endswith('\n\n') else (
        '\n' if current.endswith('\n') else '\n\n'
    )
    block = '\n'.join([
        '# Reward-free GCRL M-level baseline tasks.',
        '# Frozen target copies are excluded from the displayed trainable count.',
        *(task_line(task) for task in additions),
    ]) + '\n'
    path.write_text(current + separator + block)
    return len(additions)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    parser.add_argument('--append-to', type=Path)
    parser.add_argument(
        '--algorithms', nargs='+', choices=tuple(MODEL_SIZE_FAMILIES),
        default=tuple(MODEL_SIZE_FAMILIES),
    )
    args = parser.parse_args()
    tasks = generate_tasks(args.algorithms)
    output = args.output
    if output is None and args.append_to is None:
        output = DEFAULT_OUTPUT
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(render_tasks(tasks))
        print(f'Wrote {len(tasks)} tasks to {output}')
    if args.append_to is not None:
        added = append_tasks(args.append_to, tasks)
        print(f'Added {added} tasks to {args.append_to}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

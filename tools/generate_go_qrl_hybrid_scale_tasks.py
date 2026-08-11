#!/usr/bin/env python3
"""Generate and partition the 125-run GO-QRL Hybrid depth-scale sweep."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quasimetric_rl.model_size import go_qrl_agent_parameter_count
from tools.run_qrl_queue import Task, read_tasks


TOTAL_ENV_STEPS = 500_000
SAVE_STEPS = 50_000
TRAINING_SEEDS = tuple(range(1000, 1005))
VALIDATION_SEED = 1000
VALIDATION_EPISODES = 500
TEST_SEED = 1500
TEST_EPISODES = 1000


@dataclass(frozen=True)
class Environment:
    kind: str
    name: str
    slug: str
    state_dim: int
    action_dim: int
    goal_dim: int


@dataclass(frozen=True)
class ModelScale:
    level: str
    label: str


ENVIRONMENTS = (
    Environment(
        'dmc', 'manipulator_bring_ball', 'dmc_manipulator', 40, 5, 2,
    ),
    Environment('gcrl', 'FetchSlide', 'fetchslide', 25, 4, 3),
    Environment('dmc', 'swimmer6', 'dmc_swimmer6', 17, 5, 2),
    Environment('gym_mujoco', 'Pusher-v4', 'pusher_v4', 20, 7, 3),
    Environment(
        'gym_mujoco', 'AntNavigate-v4', 'antnavigate_v4', 29, 8, 2,
    ),
)

MODEL_SCALES = (
    ModelScale('m', 'M'),
    ModelScale('l', 'L'),
    ModelScale('xl', 'XL'),
    ModelScale('xxl', 'XXL'),
    ModelScale('xxxl', 'XXXL'),
)

# Each tuple is one indivisible five-seed group. The 13/6/6 group split gives
# 65/30/30 runs. Every partition covers all environments and all model scales.
PARTITION_GROUPS = {
    'server_2x': frozenset({
        ('swimmer6', 'm'),
        ('Pusher-v4', 'm'),
        ('AntNavigate-v4', 'm'),
        ('manipulator_bring_ball', 'l'),
        ('Pusher-v4', 'l'),
        ('AntNavigate-v4', 'l'),
        ('manipulator_bring_ball', 'xl'),
        ('FetchSlide', 'xl'),
        ('AntNavigate-v4', 'xl'),
        ('swimmer6', 'xxl'),
        ('FetchSlide', 'xxxl'),
        ('swimmer6', 'xxxl'),
        ('Pusher-v4', 'xxxl'),
    }),
    'local_1x': frozenset({
        ('manipulator_bring_ball', 'm'),
        ('FetchSlide', 'l'),
        ('swimmer6', 'xl'),
        ('manipulator_bring_ball', 'xxl'),
        ('Pusher-v4', 'xxl'),
        ('AntNavigate-v4', 'xxxl'),
    }),
    'server_1x': frozenset({
        ('FetchSlide', 'm'),
        ('swimmer6', 'l'),
        ('Pusher-v4', 'xl'),
        ('FetchSlide', 'xxl'),
        ('AntNavigate-v4', 'xxl'),
        ('manipulator_bring_ball', 'xxxl'),
    }),
}

DEFAULT_OUTPUTS = {
    'all': ROOT / 'configs/go_qrl_hybrid_scale_125_5env_5seed_500k.tsv',
    'server_2x': ROOT / 'configs/go_qrl_hybrid_scale_server_2x65.tsv',
    'local_1x': ROOT / 'configs/go_qrl_hybrid_scale_local_1x30.tsv',
    'server_1x': ROOT / 'configs/go_qrl_hybrid_scale_server_1x30.tsv',
}

COMMON_ARGS = (
    'batch_size=256',
    'agent.training_schedule=joint',
    'agent.quasimetric_critic.losses.separate_latent_dynamics=false',
    'agent.quasimetric_critic.losses.latent_dynamics.distance=iqe_mse',
    'agent.quasimetric_critic.losses.latent_dynamics.mse_weight=1.0',
    'agent.quasimetric_critic.losses.latent_dynamics.iqe_weight=1.0',
    'agent.actor.losses.min_dist.latent_goal_steps=4',
    'agent.actor.losses.min_dist.latent_goal_keep_best=true',
    'agent.actor.losses.min_dist.latent_goal_lr=0.01',
    'agent.actor.losses.min_dist.latent_goal_search=direct',
    'agent.quasimetric_critic.model.encoder.branch_normalization=none',
    'agent.actor.losses.min_dist.latent_goal_mode=max',
    'agent.actor.losses.min_dist.latent_goal_optim=sgd',
    'agent.actor.losses.min_dist.adaptive_entropy_regularizer=true',
    'agent.actor.losses.min_dist.entropy_mc_samples=100',
    'agent.actor.losses.min_dist.add_goal_as_future_state=true',
    'agent.actor.losses.behavior_cloning.weight=0',
    'agent.goal_set_distance.enabled=false',
    'interaction.exploration_eps=0',
    f'interaction.validation_seed={VALIDATION_SEED}',
    f'interaction.test_seed={TEST_SEED}',
    f'interaction.num_eval_episodes={VALIDATION_EPISODES}',
    f'interaction.num_test_episodes={TEST_EPISODES}',
    'eval_steps=null',
    f'save_steps={SAVE_STEPS}',
    'keep_only_latest_checkpoint=false',
    'save_replay_buffer=true',
    'save_final_replay_buffer=true',
    'resume_if_possible=true',
)


def compact_parameter_count(count: int) -> str:
    return f'{count / 1_000_000:.1f}m'


def task_line(task: Task) -> str:
    return '\t'.join((
        task.task_id, task.mode, task.env_name, task.seed, task.steps,
        task.params, task.extra_args,
    ))


def task_scale(task: Task) -> str:
    return task.task_id.split('-Hybrid-', 1)[1].split('_', 1)[0].lower()


def task_group(task: Task) -> tuple[str, str]:
    return task.env_name, task_scale(task)


def extra_arg_map(task: Task) -> dict[str, str]:
    return dict(token.split('=', 1) for token in shlex.split(task.extra_args))


def generate_all_tasks() -> list[Task]:
    tasks = []
    for scale in MODEL_SCALES:
        for environment in ENVIRONMENTS:
            params = compact_parameter_count(go_qrl_agent_parameter_count(
                environment.state_dim,
                environment.action_dim,
                environment.goal_dim,
                scale.level,
            ))
            for seed in TRAINING_SEEDS:
                task_id = (
                    f'scale_GO-QRL+Max4-Hybrid-{scale.label}_500k_50kckpt_'
                    f'val500_test1000_{environment.slug}_online_s{seed}'
                )
                extra_args = ' '.join((
                    f'env.kind={environment.kind}',
                    f'+go_qrl_model_size={scale.level}',
                    *COMMON_ARGS,
                ))
                tasks.append(Task(
                    task_id=task_id,
                    mode='online',
                    env_name=environment.name,
                    seed=str(seed),
                    steps=str(TOTAL_ENV_STEPS),
                    params=params,
                    extra_args=extra_args,
                ))
    validate_tasks(tasks)
    return tasks


def partition_tasks(tasks: list[Task], partition: str) -> list[Task]:
    if partition == 'all':
        return list(tasks)
    groups = PARTITION_GROUPS[partition]
    selected = [task for task in tasks if task_group(task) in groups]
    expected = {'server_2x': 65, 'local_1x': 30, 'server_1x': 30}
    if len(selected) != expected[partition]:
        raise ValueError(
            f'Expected {expected[partition]} tasks for {partition}, '
            f'got {len(selected)}'
        )
    return selected


def validate_tasks(tasks: list[Task]) -> None:
    if len(tasks) != 125:
        raise ValueError(f'Expected 125 scale-sweep tasks, got {len(tasks)}')
    counts = Counter(task.task_id for task in tasks)
    duplicates = [task_id for task_id, count in counts.items() if count > 1]
    if duplicates:
        raise ValueError(f'Duplicate generated task IDs: {duplicates}')

    expected_groups = {
        (environment.name, scale.level)
        for environment in ENVIRONMENTS
        for scale in MODEL_SCALES
    }
    actual_groups = {task_group(task) for task in tasks}
    if actual_groups != expected_groups:
        raise ValueError('Generated task groups do not cover the full matrix')
    for group in expected_groups:
        group_seeds = {
            int(task.seed) for task in tasks if task_group(task) == group
        }
        if group_seeds != set(TRAINING_SEEDS):
            raise ValueError(f'Invalid training seeds for {group}: {group_seeds}')

    required_args = {
        'batch_size': '256',
        'agent.quasimetric_critic.losses.separate_latent_dynamics': 'false',
        'agent.quasimetric_critic.losses.latent_dynamics.distance': 'iqe_mse',
        'interaction.validation_seed': str(VALIDATION_SEED),
        'interaction.test_seed': str(TEST_SEED),
        'interaction.num_eval_episodes': str(VALIDATION_EPISODES),
        'interaction.num_test_episodes': str(TEST_EPISODES),
        'eval_steps': 'null',
        'save_steps': str(SAVE_STEPS),
        'keep_only_latest_checkpoint': 'false',
    }
    for task in tasks:
        if task.mode != 'online' or int(task.steps) != TOTAL_ENV_STEPS:
            raise ValueError(f'Invalid mode/steps for {task.task_id}')
        args = extra_arg_map(task)
        for key, value in required_args.items():
            if args.get(key) != value:
                raise ValueError(
                    f'{task.task_id} requires {key}={value}, got {args.get(key)}'
                )
        validation_end = int(args['interaction.validation_seed']) + (
            int(args['interaction.num_eval_episodes']) - 1
        )
        test_end = int(args['interaction.test_seed']) + (
            int(args['interaction.num_test_episodes']) - 1
        )
        if (validation_end, test_end) != (1499, 2499):
            raise ValueError(f'Invalid evaluation seed ranges for {task.task_id}')


def validate_partitions(tasks: list[Task]) -> None:
    partitions = {
        name: partition_tasks(tasks, name) for name in PARTITION_GROUPS
    }
    ids = {
        name: {task.task_id for task in partition}
        for name, partition in partitions.items()
    }
    if any(
            ids[left] & ids[right]
            for left, right in (
                ('server_2x', 'local_1x'),
                ('server_2x', 'server_1x'),
                ('local_1x', 'server_1x'),
            )):
        raise ValueError('Scale-sweep partitions overlap')
    if set().union(*ids.values()) != {task.task_id for task in tasks}:
        raise ValueError('Scale-sweep partitions do not cover every task')
    for name, partition in partitions.items():
        if {task.env_name for task in partition} != {
                environment.name for environment in ENVIRONMENTS}:
            raise ValueError(f'{name} does not cover every environment')
        if {task_scale(task) for task in partition} != {
                scale.level for scale in MODEL_SCALES}:
            raise ValueError(f'{name} does not cover every model scale')


def render_tasks(tasks: list[Task], partition: str) -> str:
    lines = [
        '# task_id\tmode\tenv_name\tseed\tsteps\tparams\textra_args',
        f'# GO-QRL Max4 Hybrid model-scale sweep; partition: {partition}.',
        '# 500k steps; validation every 50k on seeds 1000-1499 (500 episodes).',
        '# Best validation checkpoint is tested on seeds 1500-2499 (1000 episodes).',
    ]
    lines.extend(task_line(task) for task in tasks)
    return '\n'.join(lines) + '\n'


def append_tasks(path: Path, tasks: list[Task], partition: str) -> int:
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
        f'# GO-QRL Max4 Hybrid model-scale sweep: {partition} assignment.',
        *(task_line(task) for task in additions),
    ]) + '\n'
    path.write_text(current + separator + block)
    return len(additions)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--partition', choices=tuple(DEFAULT_OUTPUTS), default='all',
    )
    parser.add_argument('--output', type=Path)
    parser.add_argument('--append-to', type=Path)
    parser.add_argument('--write-all-partitions', action='store_true')
    args = parser.parse_args()

    all_tasks = generate_all_tasks()
    validate_partitions(all_tasks)
    if args.write_all_partitions:
        for partition, output in DEFAULT_OUTPUTS.items():
            tasks = partition_tasks(all_tasks, partition)
            output.write_text(render_tasks(tasks, partition))
            print(f'Wrote {len(tasks)} tasks to {output}')
        return 0

    tasks = partition_tasks(all_tasks, args.partition)
    output = args.output
    if output is None and args.append_to is None:
        output = DEFAULT_OUTPUTS[args.partition]
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(render_tasks(tasks, args.partition))
        print(f'Wrote {len(tasks)} tasks to {output}')
    if args.append_to is not None:
        added = append_tasks(args.append_to, tasks, args.partition)
        print(f'Added {added} tasks to {args.append_to}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

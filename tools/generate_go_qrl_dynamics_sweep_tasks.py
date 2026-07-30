#!/usr/bin/env python3
"""Generate the three-seed GO-QRL dynamics factorial sweep."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_qrl_queue import Task, read_tasks


TRAINING_SEEDS = (1000, 1001, 1002)
MODEL_PARAMS = '4.4m'


@dataclass(frozen=True)
class Environment:
    kind: str
    name: str
    slug: str
    steps: int
    evaluation_episodes: int


@dataclass(frozen=True)
class Variant:
    code: str
    label: str
    separate: bool
    distance: str
    activation: str


ENVIRONMENTS = (
    Environment('gcrl', 'FetchPush', 'fetchpush', 200_000, 200),
    Environment('gcrl', 'FetchSlide', 'fetchslide', 200_000, 200),
    Environment(
        'gcrl', 'FetchPickAndPlace', 'fetchpickandplace', 200_000, 200,
    ),
    Environment('dmc', 'reacher_hard', 'dmc_reacher_hard', 200_000, 100),
    Environment('gym_mujoco', 'Pusher-v4', 'pusher_v4', 200_000, 200),
    Environment(
        'gym_mujoco', 'AntNavigate-v4', 'antnavigate_v4', 500_000, 100,
    ),
    Environment('online_maze', 'maze2d-large', 'maze2d_large', 500_000, 100),
    Environment('dmc', 'swimmer6', 'dmc_swimmer6', 500_000, 100),
    Environment(
        'dmc', 'manipulator_bring_ball', 'dmc_manipulator_bring_ball',
        500_000, 100,
    ),
)

VARIANTS = (
    Variant('A01', 'S0-IQE-ReLU', False, 'iqe', 'relu'),
    Variant('A02', 'S0-IQE-Leaky', False, 'iqe', 'leaky_relu'),
    Variant('A03', 'S0-MSE-ReLU', False, 'mse', 'relu'),
    Variant('A04', 'S0-MSE-Leaky', False, 'mse', 'leaky_relu'),
    Variant('A05', 'S0-Hybrid-ReLU', False, 'iqe_mse', 'relu'),
    Variant('A06', 'S0-Hybrid-Leaky', False, 'iqe_mse', 'leaky_relu'),
    Variant('A07', 'S1-IQE-ReLU', True, 'iqe', 'relu'),
    Variant('A08', 'S1-IQE-Leaky', True, 'iqe', 'leaky_relu'),
    Variant('A09', 'S1-MSE-ReLU', True, 'mse', 'relu'),
    Variant('A10', 'S1-MSE-Leaky', True, 'mse', 'leaky_relu'),
    Variant('A11', 'S1-Hybrid-ReLU', True, 'iqe_mse', 'relu'),
    Variant('A12', 'S1-Hybrid-Leaky', True, 'iqe_mse', 'leaky_relu'),
)

# A01 is the standard GO-QRL+Max4 configuration. These exact 200k controls
# already completed for seeds 1000-1002 and are reused by the sweep.
COMPLETED_CONTROL_ENVIRONMENTS = frozenset({
    'FetchPush', 'FetchSlide', 'FetchPickAndPlace', 'reacher_hard',
})

PARTITIONS = ('all', 'local', 'server_2', 'server_3')
DEFAULT_OUTPUTS = {
    partition: ROOT / f'configs/go_qrl_dynamics_sweep_3seed_{partition}.tsv'
    for partition in PARTITIONS
}

COMMON_ARGS = (
    '+go_qrl_model_size=m',
    'agent.training_schedule=joint',
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
    'interaction.validation_seed=1000',
    'interaction.test_seed=2000000',
    'eval_steps=null',
    'save_steps=20000',
    'keep_only_latest_checkpoint=false',
    'save_replay_buffer=true',
    'save_final_replay_buffer=true',
    'resume_if_possible=true',
)


def task_line(task: Task) -> str:
    return '\t'.join((
        task.task_id, task.mode, task.env_name, task.seed, task.steps,
        task.params, task.extra_args,
    ))


def task_variant_code(task: Task) -> str:
    marker = 'dynfac_'
    return task.task_id.split(marker, 1)[1].split('-', 1)[0]


def generate_all_tasks() -> list[Task]:
    tasks: list[Task] = []
    for environment in ENVIRONMENTS:
        for variant in VARIANTS:
            if (
                variant.code == 'A01'
                and environment.name in COMPLETED_CONTROL_ENVIRONMENTS
            ):
                continue
            for seed in TRAINING_SEEDS:
                step_label = f'{environment.steps // 1000}k'
                task_id = (
                    f'ablation_GO-QRL+Max4-dynfac_{variant.code}-{variant.label}-M_'
                    f'{step_label}_20kckpt_val{environment.evaluation_episodes}_'
                    f'test{environment.evaluation_episodes}_{environment.slug}_online_'
                    f's{seed}'
                )
                variant_args = [
                    'agent.quasimetric_critic.losses.'
                    f'separate_latent_dynamics={str(variant.separate).lower()}',
                    'agent.quasimetric_critic.losses.latent_dynamics.'
                    f'distance={variant.distance}',
                    'agent.quasimetric_critic.model.quasimetric_model.'
                    f'projector_activation={variant.activation}',
                ]
                if variant.distance == 'iqe_mse':
                    variant_args.extend((
                        'agent.quasimetric_critic.losses.latent_dynamics.'
                        'mse_weight=1.0',
                        'agent.quasimetric_critic.losses.latent_dynamics.'
                        'iqe_weight=1.0',
                    ))
                if variant.activation == 'leaky_relu':
                    variant_args.append(
                        'agent.quasimetric_critic.model.quasimetric_model.'
                        'projector_negative_slope=0.01'
                    )
                extra_args = ' '.join((
                    f'env.kind={environment.kind}',
                    *variant_args,
                    *COMMON_ARGS,
                    f'interaction.num_eval_episodes='
                    f'{environment.evaluation_episodes}',
                    f'interaction.num_test_episodes='
                    f'{environment.evaluation_episodes}',
                ))
                tasks.append(Task(
                    task_id=task_id,
                    mode='online',
                    env_name=environment.name,
                    seed=str(seed),
                    steps=str(environment.steps),
                    params=MODEL_PARAMS,
                    extra_args=extra_args,
                ))
    validate_all_tasks(tasks)
    return tasks


def partition_tasks(tasks: list[Task], partition: str) -> list[Task]:
    if partition == 'all':
        return list(tasks)

    selected: list[Task] = []
    for task in tasks:
        code = task_variant_code(task)
        if partition == 'local':
            include = code in {'A01', 'A02', 'A03', 'A04', 'A05', 'A06'} or (
                task.env_name == 'Pusher-v4' and code in {'A07', 'A10'}
            )
        elif partition == 'server_2':
            include = code in {'A07', 'A08', 'A09'} and not (
                task.env_name == 'Pusher-v4' and code == 'A07'
            )
        elif partition == 'server_3':
            include = code in {'A10', 'A11', 'A12'} and not (
                task.env_name == 'Pusher-v4' and code == 'A10'
            )
        else:
            raise ValueError(f'Unknown partition: {partition!r}')
        if include:
            selected.append(task)

    expected = {'local': 156, 'server_2': 78, 'server_3': 78}
    if len(selected) != expected[partition]:
        raise ValueError(
            f'Expected {expected[partition]} tasks for {partition}, '
            f'got {len(selected)}'
        )
    return selected


def task_workload(tasks: list[Task]) -> int:
    return sum(int(task.steps) for task in tasks)


def validate_all_tasks(tasks: list[Task]) -> None:
    if len(tasks) != 312:
        raise ValueError(f'Expected 312 new sweep tasks, got {len(tasks)}')
    counts = Counter(task.task_id for task in tasks)
    duplicates = [task_id for task_id, count in counts.items() if count > 1]
    if duplicates:
        raise ValueError(f'Duplicate generated task IDs: {duplicates}')


def validate_partitions(tasks: list[Task]) -> None:
    partitions = {
        name: partition_tasks(tasks, name)
        for name in ('local', 'server_2', 'server_3')
    }
    ids = {name: {task.task_id for task in part} for name, part in partitions.items()}
    if ids['local'] & ids['server_2'] or ids['local'] & ids['server_3']:
        raise ValueError('Local partition overlaps a remote partition')
    if ids['server_2'] & ids['server_3']:
        raise ValueError('Remote partitions overlap')
    if set().union(*ids.values()) != {task.task_id for task in tasks}:
        raise ValueError('Partitions do not cover the complete task matrix')
    expected_workloads = {
        'local': 52_800_000,
        'server_2': 26_400_000,
        'server_3': 26_400_000,
    }
    for name, expected in expected_workloads.items():
        actual = task_workload(partitions[name])
        if actual != expected:
            raise ValueError(
                f'Expected workload {expected} for {name}, got {actual}'
            )


def render_tasks(tasks: list[Task], partition: str) -> str:
    workload = task_workload(tasks) / 1_000_000
    lines = [
        '# task_id\tmode\tenv_name\tseed\tsteps\tparams\textra_args',
        '# GO-QRL+Max4 dynamics factorial: Separate(2) x loss(3) x '
        'projector activation(2).',
        '# Seeds 1000-1002; A01 controls already completed on FetchPush, '
        'FetchSlide, FetchPickAndPlace, and reacher_hard are omitted.',
        f'# Assignment: {partition}; tasks={len(tasks)}; workload={workload:.1f}M '
        'environment steps.',
    ]
    lines.extend(task_line(task) for task in tasks)
    return '\n'.join(lines) + '\n'


def append_tasks(path: Path, tasks: list[Task], partition: str) -> int:
    existing_ids = {task.task_id for task in read_tasks(path)}
    additions = [task for task in tasks if task.task_id not in existing_ids]
    if not additions:
        return 0
    current = path.read_text() if path.exists() else ''
    separator = '' if not current else ('\n' if current.endswith('\n') else '\n\n')
    block = '\n'.join((
        f'# GO-QRL dynamics factorial: {partition} assignment, 3 seeds.',
        *(task_line(task) for task in additions),
    )) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(current + separator + block)
    return len(additions)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--partition', choices=PARTITIONS, default='all')
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
            print(
                f'Wrote {len(tasks)} tasks ({task_workload(tasks) / 1e6:.1f}M '
                f'steps) to {output}'
            )
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

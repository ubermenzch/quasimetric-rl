#!/usr/bin/env python3
"""Generate legacy and full-table GO-QRL inner-step ablation matrices."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quasimetric_rl.model_size import go_qrl_agent_parameter_count
from tools.run_qrl_queue import Task, read_tasks


TOTAL_ENV_STEPS = 200_000
TRAINING_SEEDS = tuple(range(1000, 1005))
MODEL_SIZE_LEVEL = 'm'

ENVIRONMENTS = (
    ('gcrl', 'FetchReach', 'fetchreach', 10, 4, 3),
    ('gcrl', 'FetchPush', 'fetchpush', 25, 4, 3),
    ('gcrl', 'FetchSlide', 'fetchslide', 25, 4, 3),
    ('gcrl', 'FetchPickAndPlace', 'fetchpickandplace', 25, 4, 3),
    ('dmc', 'reacher_easy', 'dmc_reacher_easy', 6, 2, 2),
    ('dmc', 'reacher_hard', 'dmc_reacher_hard', 6, 2, 2),
    ('gym_mujoco', 'Reacher-v4', 'reacher_v4', 8, 2, 2),
)


@dataclass(frozen=True)
class TableEnvironment:
    """One column in the eight-environment ablation table."""

    key: str
    kind: str
    name: str
    slug: str
    state_dim: int
    action_dim: int
    goal_dim: int
    model_size: str
    total_steps: int
    save_steps: int
    validation_episodes: int
    validation_seed: int
    test_episodes: int
    test_seed: int


# This is the full factorial matrix used for the requested table.  The two
# reacher_hard rows are separate cells because their training budgets differ.
TABLE_ENVIRONMENTS = (
    TableEnvironment(
        'fetchpush_m100k', 'gcrl', 'FetchPush', 'fetchpush_m100k',
        25, 4, 3, 'm', 100_000, 20_000, 1000, 1000, 1000, 2000,
    ),
    TableEnvironment(
        'fetchslide_l500k', 'gcrl', 'FetchSlide', 'fetchslide_l500k',
        25, 4, 3, 'l', 500_000, 50_000, 500, 1000, 1000, 1500,
    ),
    TableEnvironment(
        'fetchpickandplace_m200k', 'gcrl', 'FetchPickAndPlace',
        'fetchpickandplace_m200k', 25, 4, 3, 'm', 200_000, 20_000,
        1000, 1000, 1000, 2000,
    ),
    TableEnvironment(
        'reacher_hard_m200k', 'dmc', 'reacher_hard',
        'dmc_reacher_hard_m200k', 6, 2, 2, 'm', 200_000, 20_000,
        1000, 1000, 1000, 2000,
    ),
    TableEnvironment(
        'reacher_hard_m100k', 'dmc', 'reacher_hard',
        'dmc_reacher_hard_m100k', 6, 2, 2, 'm', 100_000, 20_000,
        1000, 1000, 1000, 2000,
    ),
    TableEnvironment(
        'maze2d_large_m200k', 'online_maze', 'maze2d-large',
        'maze2d_large_m200k', 4, 2, 2, 'm', 200_000, 20_000,
        1000, 1000, 1000, 2000,
    ),
    TableEnvironment(
        'pusher_v4_l500k', 'gym_mujoco', 'Pusher-v4', 'pusher_v4_l500k',
        20, 7, 3, 'l', 500_000, 50_000, 500, 1000, 1000, 1500,
    ),
    TableEnvironment(
        'antnavigate_v4_l500k', 'gym_mujoco', 'AntNavigate-v4',
        'antnavigate_v4_l500k', 29, 8, 2, 'l', 500_000, 50_000,
        500, 1000, 1000, 1500,
    ),
)

VARIANTS = (
    ('Inner0', 'min', 0),
    ('Min1', 'min', 1),
    ('Min8', 'min', 8),
    ('Max1', 'max', 1),
    ('Max8', 'max', 8),
)

FULL_SWEEP_ENVIRONMENTS = frozenset({
    'FetchPush', 'reacher_hard', 'Reacher-v4',
})

PARTITION_GROUPS = {
    'local': frozenset({
        *[(env_name, variant) for env_name in ('FetchPush', 'reacher_hard')
          for variant, _mode, _steps in VARIANTS],
        ('FetchReach', 'Inner0'),
        ('FetchSlide', 'Inner0'),
        ('FetchPickAndPlace', 'Inner0'),
    }),
    'server_crl': frozenset({
        ('Reacher-v4', 'Inner0'),
        ('Reacher-v4', 'Min1'),
        ('Reacher-v4', 'Min8'),
    }),
    'server_c_learning': frozenset({
        ('Reacher-v4', 'Max1'),
        ('Reacher-v4', 'Max8'),
        ('reacher_easy', 'Inner0'),
    }),
}

DEFAULT_OUTPUTS = {
    'all': ROOT / 'configs/go_qrl_inner_steps_ablation_95_7env_5seed_200k.tsv',
    'local': ROOT / 'configs/go_qrl_inner_steps_ablation_local65.tsv',
    'server_crl': ROOT / 'configs/go_qrl_inner_steps_ablation_crl_server15.tsv',
    'server_c_learning': (
        ROOT / 'configs/go_qrl_inner_steps_ablation_c_learning_server15.tsv'
    ),
}

TABLE_DEFAULT_OUTPUTS = {
    'all': (
        ROOT / 'configs/go_qrl_inner_steps_ablation_table_8env_5variant_5seed.tsv'
    ),
    'local_2x': (
        ROOT / 'configs/go_qrl_inner_steps_ablation_table_local135.tsv'
    ),
    'remote_1x': (
        ROOT / 'configs/go_qrl_inner_steps_ablation_table_remote65.tsv'
    ),
}

# Five seeds belonging to one variant/environment cell stay together.  This
# assignment gives remote 5 L-500k cells and 8 M cells (19.5M environment
# steps); local receives the complementary 10 L and 17 M cells (38.0M steps).
TABLE_REMOTE_GROUPS = frozenset({
    ('Inner0', 'fetchslide_l500k'),
    ('Inner0', 'fetchpickandplace_m200k'),
    ('Inner0', 'maze2d_large_m200k'),
    ('Min1', 'fetchpush_m100k'),
    ('Min1', 'reacher_hard_m200k'),
    ('Min1', 'pusher_v4_l500k'),
    ('Min8', 'maze2d_large_m200k'),
    ('Min8', 'antnavigate_v4_l500k'),
    ('Max1', 'fetchslide_l500k'),
    ('Max1', 'fetchpickandplace_m200k'),
    ('Max1', 'reacher_hard_m100k'),
    ('Max8', 'reacher_hard_m200k'),
    ('Max8', 'pusher_v4_l500k'),
})

COMMON_ARGS = (
    '+go_qrl_model_size=m',
    'agent.quasimetric_critic.model.encoder.branch_normalization=none',
    'agent.actor.losses.min_dist.latent_goal_keep_best=true',
    'agent.actor.losses.min_dist.latent_goal_lr=0.01',
    'agent.actor.losses.min_dist.latent_goal_search=direct',
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
)

TABLE_FIXED_ARGS = tuple(
    argument for argument in (
        'batch_size=256',
        'agent.training_schedule=joint',
        'agent.quasimetric_critic.losses.separate_latent_dynamics=false',
        'agent.quasimetric_critic.losses.latent_dynamics.distance=iqe_mse',
        'agent.quasimetric_critic.losses.latent_dynamics.mse_weight=1.0',
        'agent.quasimetric_critic.losses.latent_dynamics.iqe_weight=1.0',
        *COMMON_ARGS[1:],
    )
    if argument.split('=', 1)[0] not in {
        'interaction.validation_seed',
        'interaction.test_seed',
        'eval_steps',
        'save_steps',
    }
)


def evaluation_episode_count(env_kind: str, env_name: str) -> int:
    if env_kind == 'dmc':
        return 100
    if (env_kind, env_name) == ('gcrl', 'FetchReach'):
        return 1000
    return 200


def compact_parameter_count(count: int) -> str:
    return f'{count / 1_000_000:.1f}m'


def generate_all_tasks() -> list[Task]:
    tasks = []
    for (
        env_kind, env_name, env_slug, state_dim, action_dim, goal_dim,
    ) in ENVIRONMENTS:
        variants = (
            VARIANTS if env_name in FULL_SWEEP_ENVIRONMENTS else VARIANTS[:1]
        )
        episodes = evaluation_episode_count(env_kind, env_name)
        params = compact_parameter_count(go_qrl_agent_parameter_count(
            state_dim, action_dim, goal_dim, MODEL_SIZE_LEVEL,
        ))
        for variant, mode, inner_steps in variants:
            for seed in TRAINING_SEEDS:
                task_id = (
                    f'ablation_GO-QRL+{variant}-M_200k_20kckpt_'
                    f'val{episodes}_test{episodes}_{env_slug}_online_s{seed}'
                )
                extra_args = ' '.join((
                    f'env.kind={env_kind}',
                    *COMMON_ARGS,
                    f'agent.actor.losses.min_dist.latent_goal_mode={mode}',
                    f'agent.actor.losses.min_dist.latent_goal_steps={inner_steps}',
                    f'interaction.num_eval_episodes={episodes}',
                    f'interaction.num_test_episodes={episodes}',
                ))
                tasks.append(Task(
                    task_id=task_id,
                    mode='online',
                    env_name=env_name,
                    seed=str(seed),
                    steps=str(TOTAL_ENV_STEPS),
                    params=params,
                    extra_args=extra_args,
                ))
    validate_tasks(tasks)
    return tasks


def table_task_variant(task: Task) -> str:
    prefix = 'ablation_GO-QRL+'
    if not task.task_id.startswith(prefix):
        raise ValueError(f'Not a GO-QRL ablation task: {task.task_id}')
    return task.task_id[len(prefix):].split('-', 1)[0]


def table_task_environment_key(task: Task) -> str:
    matches = [
        environment.key for environment in TABLE_ENVIRONMENTS
        if f'_{environment.slug}_online_s' in task.task_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f'Could not uniquely identify table environment for {task.task_id}'
        )
    return matches[0]


def table_task_group(task: Task) -> tuple[str, str]:
    return table_task_variant(task), table_task_environment_key(task)


def table_extra_arg_map(task: Task) -> dict[str, str]:
    tokens = task.extra_args.split()
    keys = [token.split('=', 1)[0] for token in tokens]
    if len(keys) != len(set(keys)):
        raise ValueError(f'Duplicate extra-argument keys in {task.task_id}')
    return dict(token.split('=', 1) for token in tokens)


def generate_table_tasks() -> list[Task]:
    """Generate the complete 5-variant by 8-environment table matrix."""

    tasks = []
    for variant, mode, inner_steps in VARIANTS:
        for environment in TABLE_ENVIRONMENTS:
            params = compact_parameter_count(go_qrl_agent_parameter_count(
                environment.state_dim,
                environment.action_dim,
                environment.goal_dim,
                environment.model_size,
            ))
            step_label = f'{environment.total_steps // 1000}k'
            save_label = f'{environment.save_steps // 1000}k'
            size_label = environment.model_size.upper()
            for seed in TRAINING_SEEDS:
                task_id = (
                    f'ablation_GO-QRL+{variant}-{size_label}_{step_label}_'
                    f'{save_label}ckpt_val{environment.validation_episodes}_'
                    f'test{environment.test_episodes}_{environment.slug}_'
                    f'online_s{seed}'
                )
                extra_args = ' '.join((
                    f'env.kind={environment.kind}',
                    f'+go_qrl_model_size={environment.model_size}',
                    *TABLE_FIXED_ARGS,
                    f'agent.actor.losses.min_dist.latent_goal_mode={mode}',
                    f'agent.actor.losses.min_dist.latent_goal_steps={inner_steps}',
                    f'interaction.validation_seed={environment.validation_seed}',
                    f'interaction.test_seed={environment.test_seed}',
                    f'interaction.num_eval_episodes='
                    f'{environment.validation_episodes}',
                    f'interaction.num_test_episodes={environment.test_episodes}',
                    'eval_steps=null',
                    f'save_steps={environment.save_steps}',
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
    validate_table_tasks(tasks)
    validate_table_partitions(tasks)
    return tasks


def validate_table_tasks(tasks: list[Task]) -> None:
    expected_count = len(VARIANTS) * len(TABLE_ENVIRONMENTS) * len(TRAINING_SEEDS)
    if len(tasks) != expected_count:
        raise ValueError(
            f'Expected {expected_count} table ablation tasks, got {len(tasks)}'
        )
    duplicate_ids = [
        task_id for task_id, count in Counter(
            task.task_id for task in tasks
        ).items() if count > 1
    ]
    if duplicate_ids:
        raise ValueError(f'Duplicate table task IDs: {duplicate_ids}')

    expected_variants = {variant for variant, _mode, _steps in VARIANTS}
    variant_counts = Counter(table_task_variant(task) for task in tasks)
    if set(variant_counts) != expected_variants:
        raise ValueError(
            f'Unexpected table variants: {sorted(variant_counts)}'
        )
    expected_per_variant = len(TABLE_ENVIRONMENTS) * len(TRAINING_SEEDS)
    if set(variant_counts.values()) != {expected_per_variant}:
        raise ValueError(f'Invalid table variant counts: {variant_counts}')

    for variant, mode, inner_steps in VARIANTS:
        variant_tasks = [
            task for task in tasks if table_task_variant(task) == variant
        ]
        for environment in TABLE_ENVIRONMENTS:
            group = [
                task for task in variant_tasks
                if task.env_name == environment.name
                and int(task.steps) == environment.total_steps
                and environment.slug in task.task_id
            ]
            if len(group) != len(TRAINING_SEEDS):
                raise ValueError(
                    f'Invalid seed count for {variant}/{environment.key}: '
                    f'{len(group)}'
                )
            expected_params = compact_parameter_count(
                go_qrl_agent_parameter_count(
                    environment.state_dim,
                    environment.action_dim,
                    environment.goal_dim,
                    environment.model_size,
                )
            )
            for task in group:
                if task.seed not in {str(seed) for seed in TRAINING_SEEDS}:
                    raise ValueError(f'Invalid seed in {task.task_id}')
                if task.params != expected_params or task.mode != 'online':
                    raise ValueError(f'Invalid metadata in {task.task_id}')
                args = table_extra_arg_map(task)
                required = {
                    'env.kind': environment.kind,
                    '+go_qrl_model_size': environment.model_size,
                    'agent.actor.losses.min_dist.latent_goal_mode': mode,
                    'agent.actor.losses.min_dist.latent_goal_steps': str(
                        inner_steps
                    ),
                    'interaction.validation_seed': str(
                        environment.validation_seed
                    ),
                    'interaction.test_seed': str(environment.test_seed),
                    'interaction.num_eval_episodes': str(
                        environment.validation_episodes
                    ),
                    'interaction.num_test_episodes': str(
                        environment.test_episodes
                    ),
                    'eval_steps': 'null',
                    'save_steps': str(environment.save_steps),
                }
                for key, value in required.items():
                    if args.get(key) != value:
                        raise ValueError(
                            f'{task.task_id} requires {key}={value}, '
                            f'got {args.get(key)}'
                        )


def partition_table_tasks(tasks: list[Task], partition: str) -> list[Task]:
    if partition == 'all':
        return list(tasks)
    groups = (
        TABLE_REMOTE_GROUPS
        if partition == 'remote_1x'
        else all_table_groups().difference(TABLE_REMOTE_GROUPS)
    )
    selected = [task for task in tasks if table_task_group(task) in groups]
    expected = {'local_2x': 135, 'remote_1x': 65}[partition]
    if len(selected) != expected:
        raise ValueError(
            f'Expected {expected} table tasks for {partition}, got {len(selected)}'
        )
    return selected


def all_table_groups() -> frozenset[tuple[str, str]]:
    return frozenset(
        (variant, environment.key)
        for variant, _mode, _steps in VARIANTS
        for environment in TABLE_ENVIRONMENTS
    )


def validate_table_partitions(tasks: list[Task]) -> None:
    all_groups = all_table_groups()
    if len(TABLE_REMOTE_GROUPS) != 13:
        raise ValueError('Expected 13 remote table groups')
    if not TABLE_REMOTE_GROUPS.issubset(all_groups):
        raise ValueError('Remote table partition contains unknown groups')
    local = partition_table_tasks(tasks, 'local_2x')
    remote = partition_table_tasks(tasks, 'remote_1x')
    local_ids = {task.task_id for task in local}
    remote_ids = {task.task_id for task in remote}
    if local_ids & remote_ids:
        raise ValueError('Table partitions overlap')
    if local_ids | remote_ids != {task.task_id for task in tasks}:
        raise ValueError('Table partitions do not cover the full matrix')

    environments = {environment.key: environment
                    for environment in TABLE_ENVIRONMENTS}
    remote_l_groups = sum(
        environments[environment_key].model_size == 'l'
        for _variant, environment_key in TABLE_REMOTE_GROUPS
    )
    if remote_l_groups != 5:
        raise ValueError(f'Expected 5 remote L groups, got {remote_l_groups}')
    local_groups = all_groups.difference(TABLE_REMOTE_GROUPS)
    expected_variants = {variant for variant, _mode, _steps in VARIANTS}
    expected_environments = {
        environment.key for environment in TABLE_ENVIRONMENTS
    }
    for partition, groups in (
        ('local_2x', local_groups),
        ('remote_1x', TABLE_REMOTE_GROUPS),
    ):
        if {variant for variant, _environment in groups} != expected_variants:
            raise ValueError(f'{partition} does not cover every variant')
        if {
            environment for _variant, environment in groups
        } != expected_environments:
            raise ValueError(f'{partition} does not cover every environment')


def render_table_tasks(tasks: list[Task], partition: str = 'all') -> str:
    lines = [
        '# task_id\tmode\tenv_name\tseed\tsteps\tparams\textra_args',
        '# GO-QRL inner-step ablation table: 5 variants x 8 environments x 5 seeds.',
        f'# Assignment partition: {partition}.',
        '# Columns: FetchPush M-100k; FetchSlide L-500k; FetchPickAndPlace M-200k;',
        '# reacher_hard M-200k; reacher_hard M-100k; maze2d-large M-200k;',
        '# Pusher-v4 L-500k; AntNavigate-v4 L-500k.',
        '# Each variant covers every column. Inner0 is min mode with zero updates.',
    ]
    lines.extend(task_line(task) for task in tasks)
    return '\n'.join(lines) + '\n'


def task_variant(task: Task) -> str:
    return task.task_id.split('ablation_GO-QRL+', 1)[1].split('-M_', 1)[0]


def partition_tasks(tasks: list[Task], partition: str) -> list[Task]:
    if partition == 'all':
        return list(tasks)
    groups = PARTITION_GROUPS[partition]
    selected = [
        task for task in tasks
        if (task.env_name, task_variant(task)) in groups
    ]
    expected = {'local': 65, 'server_crl': 15, 'server_c_learning': 15}
    if len(selected) != expected[partition]:
        raise ValueError(
            f'Expected {expected[partition]} tasks for {partition}, got {len(selected)}'
        )
    return selected


def validate_tasks(tasks: list[Task]) -> None:
    if len(tasks) != 95:
        raise ValueError(f'Expected 95 ablation tasks, got {len(tasks)}')
    counts = Counter(task.task_id for task in tasks)
    duplicates = [task_id for task_id, count in counts.items() if count > 1]
    if duplicates:
        raise ValueError(f'Duplicate generated task IDs: {duplicates}')


def task_line(task: Task) -> str:
    return '\t'.join((
        task.task_id, task.mode, task.env_name, task.seed, task.steps,
        task.params, task.extra_args,
    ))


def render_tasks(tasks: list[Task], partition: str) -> str:
    lines = [
        '# task_id\tmode\tenv_name\tseed\tsteps\tparams\textra_args',
        f'# GO-QRL inner-loop ablations; assignment partition: {partition}.',
        '# Inner0 covers all seven environments; 1/8-step Min/Max covers '
        'FetchPush, reacher_hard, and Reacher-v4.',
        '# Every task uses GO-QRL-M, 200k steps, 20k checkpoints, and best-val test.',
    ]
    lines.extend(task_line(task) for task in tasks)
    return '\n'.join(lines) + '\n'


def append_tasks(path: Path, tasks: list[Task], partition: str) -> int:
    existing_ids = {task.task_id for task in read_tasks(path)}
    additions = [task for task in tasks if task.task_id not in existing_ids]
    if not additions:
        return 0
    current = path.read_text() if path.exists() else ''
    separator = '' if not current or current.endswith('\n\n') else (
        '\n' if current.endswith('\n') else '\n\n'
    )
    block = '\n'.join([
        f'# GO-QRL inner-loop ablations: {partition} assignment.',
        *(task_line(task) for task in additions),
    ]) + '\n'
    path.write_text(current + separator + block)
    return len(additions)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--table', action='store_true',
        help='Generate the full 8-environment inner-step ablation table.',
    )
    parser.add_argument(
        '--partition',
        choices=tuple(dict.fromkeys((*DEFAULT_OUTPUTS, *TABLE_DEFAULT_OUTPUTS))),
        default='all',
    )
    parser.add_argument('--output', type=Path)
    parser.add_argument('--append-to', type=Path)
    parser.add_argument('--write-all-partitions', action='store_true')
    args = parser.parse_args()

    if args.table:
        if args.partition not in TABLE_DEFAULT_OUTPUTS:
            parser.error(
                f'--table does not support partition {args.partition!r}'
            )
        all_tasks = generate_table_tasks()
        if args.write_all_partitions:
            for partition, output in TABLE_DEFAULT_OUTPUTS.items():
                tasks = partition_table_tasks(all_tasks, partition)
                output.write_text(render_table_tasks(tasks, partition))
                print(f'Wrote {len(tasks)} tasks to {output}')
            return 0
        tasks = partition_table_tasks(all_tasks, args.partition)
        output = args.output
        if output is None and args.append_to is None:
            output = TABLE_DEFAULT_OUTPUTS[args.partition]
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(render_table_tasks(tasks, args.partition))
            print(f'Wrote {len(tasks)} tasks to {output}')
        if args.append_to is not None:
            added = append_tasks(args.append_to, tasks, args.partition)
            print(f'Added {added} tasks to {args.append_to}')
        return 0

    if args.partition not in DEFAULT_OUTPUTS:
        parser.error(
            f'Legacy ablations do not support partition {args.partition!r}'
        )
    all_tasks = generate_all_tasks()
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

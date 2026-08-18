#!/usr/bin/env python3
"""Generate the missing five-seed experiments for the ICLR main table."""

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

from quasimetric_rl.model_size import (
    go_qrl_agent_parameter_count,
    qrl_agent_parameter_count,
)
from quasimetric_rl.modules.gcrl_baselines import resolve_baseline_goal_dims
from tools.generate_online_baseline_tasks import (
    MODEL_SIZE_FAMILIES,
    baseline_parameter_count,
    compact_parameter_count,
)
from tools.run_qrl_queue import Task, read_tasks


TRAINING_SEEDS = tuple(range(1000, 1005))


@dataclass(frozen=True)
class Environment:
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


@dataclass(frozen=True)
class Algorithm:
    key: str
    label: str
    baseline: str | None = None
    goal_mode: str | None = None


ENVIRONMENTS = (
    Environment(
        'fetchpush', 'gcrl', 'FetchPush', 'fetchpush', 25, 4, 3,
        'm', 100_000, 20_000, 1000, 1000, 1000, 2000,
    ),
    Environment(
        'fetchslide', 'gcrl', 'FetchSlide', 'fetchslide', 25, 4, 3,
        'l', 500_000, 50_000, 500, 1000, 1000, 1500,
    ),
    Environment(
        'fetchpickandplace', 'gcrl', 'FetchPickAndPlace',
        'fetchpickandplace', 25, 4, 3,
        'm', 200_000, 20_000, 1000, 1000, 1000, 2000,
    ),
    Environment(
        'reacher_hard', 'dmc', 'reacher_hard', 'dmc_reacher_hard', 6, 2, 2,
        'm', 100_000, 20_000, 1000, 1000, 1000, 2000,
    ),
    Environment(
        'maze2d_large', 'online_maze', 'maze2d-large', 'maze2d_large',
        4, 2, 2, 'm', 200_000, 20_000, 1000, 1000, 1000, 2000,
    ),
    Environment(
        'pusher_v4', 'gym_mujoco', 'Pusher-v4', 'pusher_v4', 20, 7, 3,
        'l', 500_000, 50_000, 500, 1000, 1000, 1500,
    ),
    Environment(
        'antnavigate_v4', 'gym_mujoco', 'AntNavigate-v4', 'antnavigate_v4',
        29, 8, 2, 'l', 500_000, 50_000, 500, 1000, 1000, 1500,
    ),
)

ALGORITHMS = (
    Algorithm('qrl', 'QRL'),
    Algorithm('go_max4', 'GO-QRL+Max4-Hybrid', goal_mode='max'),
    Algorithm('go_min4', 'GO-QRL+Min4-Hybrid', goal_mode='min'),
    Algorithm('td_infonce', 'TD-InfoNCE', baseline='td_infonce'),
    Algorithm('gcsl', 'GCSL', baseline='gcsl'),
    Algorithm('c_learning', 'C-Learning', baseline='c_learning'),
    Algorithm('crl', 'CRL', baseline='crl'),
)

# These table cells already have five-seed results and must not be rerun.
COMPLETED_GROUPS = frozenset({
    ('qrl', 'fetchpickandplace'),
    ('go_max4', 'fetchslide'),
    ('go_max4', 'fetchpickandplace'),
    ('go_max4', 'pusher_v4'),
    ('go_max4', 'antnavigate_v4'),
    ('go_min4', 'fetchpickandplace'),
})

# A group is indivisible: all five training seeds run on the same server.
# There are 43 missing groups, so the closest whole-group 2:1 split is 29:14.
# L-500k groups are split exactly 12:6; M groups are split 17:8.
REMOTE_GROUPS = frozenset({
    ('qrl', 'fetchpush'),
    ('qrl', 'fetchslide'),
    ('go_max4', 'maze2d_large'),
    ('go_min4', 'reacher_hard'),
    ('go_min4', 'pusher_v4'),
    ('td_infonce', 'fetchpickandplace'),
    ('td_infonce', 'antnavigate_v4'),
    ('gcsl', 'fetchslide'),
    ('gcsl', 'maze2d_large'),
    ('c_learning', 'fetchpush'),
    ('c_learning', 'pusher_v4'),
    ('crl', 'fetchpickandplace'),
    ('crl', 'reacher_hard'),
    ('crl', 'antnavigate_v4'),
})

DEFAULT_OUTPUTS = {
    'all': ROOT / 'configs/iclr_main_table_missing_215.tsv',
    'local_2x': ROOT / 'configs/iclr_main_table_local_2x145.tsv',
    'remote_1x': ROOT / 'configs/iclr_main_table_remote_1x70.tsv',
}

CHECKPOINT_ARGS = (
    'keep_only_latest_checkpoint=true',
    'keep_only_best_and_final_checkpoints=true',
    'save_replay_buffer=true',
    'save_final_replay_buffer=true',
    'resume_if_possible=true',
)

QRL_COMMON_ARGS = (
    'batch_size=256',
    'agent.training_schedule=joint',
    'agent.actor.losses.min_dist.adaptive_entropy_regularizer=true',
    'agent.actor.losses.min_dist.entropy_mc_samples=100',
    'agent.actor.losses.min_dist.add_goal_as_future_state=true',
    'agent.actor.losses.behavior_cloning.weight=0',
    'agent.goal_set_distance.enabled=false',
    'interaction.exploration_eps=0',
)

GO_QRL_HYBRID_ARGS = (
    'agent.quasimetric_critic.losses.separate_latent_dynamics=false',
    'agent.quasimetric_critic.losses.latent_dynamics.distance=iqe_mse',
    'agent.quasimetric_critic.losses.latent_dynamics.mse_weight=1.0',
    'agent.quasimetric_critic.losses.latent_dynamics.iqe_weight=1.0',
    'agent.actor.losses.min_dist.latent_goal_steps=4',
    'agent.actor.losses.min_dist.latent_goal_keep_best=true',
    'agent.actor.losses.min_dist.latent_goal_lr=0.01',
    'agent.actor.losses.min_dist.latent_goal_search=direct',
    'agent.quasimetric_critic.model.encoder.branch_normalization=none',
    'agent.actor.losses.min_dist.latent_goal_optim=sgd',
)


def all_missing_groups() -> frozenset[tuple[str, str]]:
    return frozenset(
        (algorithm.key, environment.key)
        for algorithm in ALGORITHMS
        for environment in ENVIRONMENTS
        if (algorithm.key, environment.key) not in COMPLETED_GROUPS
    )


def local_groups() -> frozenset[tuple[str, str]]:
    return all_missing_groups().difference(REMOTE_GROUPS)


def task_group(task: Task) -> tuple[str, str]:
    prefix = 'iclr_main_v1_'
    if not task.task_id.startswith(prefix):
        raise ValueError(f'Not an ICLR main-table task: {task.task_id}')
    algorithm = next(
        candidate for candidate in ALGORITHMS
        if task.task_id.startswith(f'{prefix}{candidate.label}-')
    )
    environment = next(
        candidate for candidate in ENVIRONMENTS
        if candidate.name == task.env_name
    )
    return algorithm.key, environment.key


def extra_arg_map(task: Task) -> dict[str, str]:
    return dict(token.split('=', 1) for token in shlex.split(task.extra_args))


def baseline_model_size(algorithm: Algorithm, environment: Environment) -> str:
    if algorithm.baseline != 'gcsl' or environment.model_size != 'l':
        return environment.model_size
    if environment.key == 'pusher_v4':
        return 'l_pusher'
    if environment.key == 'antnavigate_v4':
        return 'l_antnavigate'
    return 'l'


def parameter_count(algorithm: Algorithm, environment: Environment) -> int:
    if algorithm.key == 'qrl':
        return qrl_agent_parameter_count(
            environment.state_dim, environment.action_dim,
            environment.model_size,
        )
    if algorithm.goal_mode is not None:
        return go_qrl_agent_parameter_count(
            environment.state_dim, environment.action_dim,
            environment.goal_dim, environment.model_size,
        )
    assert algorithm.baseline is not None
    conditioning_goal_dim = len(resolve_baseline_goal_dims(
        algorithm.baseline,
        env_kind=environment.kind,
        env_name=environment.name,
        state_dim=environment.state_dim,
        success_goal_dims=tuple(range(environment.goal_dim)),
    ))
    return baseline_parameter_count(
        algorithm.baseline,
        environment.state_dim,
        environment.action_dim,
        conditioning_goal_dim,
        model_size_level=baseline_model_size(algorithm, environment),
    )


def algorithm_args(
        algorithm: Algorithm, environment: Environment) -> tuple[str, ...]:
    if algorithm.key == 'qrl':
        return (f'+qrl_model_size={environment.model_size}', *QRL_COMMON_ARGS)
    if algorithm.goal_mode is not None:
        return (
            f'+go_qrl_model_size={environment.model_size}',
            *QRL_COMMON_ARGS,
            *GO_QRL_HYBRID_ARGS,
            f'agent.actor.losses.min_dist.latent_goal_mode={algorithm.goal_mode}',
        )
    assert algorithm.baseline is not None
    level = baseline_model_size(algorithm, environment)
    return (
        f'agent.algorithm={algorithm.baseline}',
        f'+{MODEL_SIZE_FAMILIES[algorithm.baseline]}_model_size={level}',
        'batch_size=256',
        'interaction.exploration_eps=0',
    )


def task_version_tag(algorithm: Algorithm, environment: Environment) -> str:
    if algorithm.baseline == 'crl':
        return 'originalcrl2022_'
    if (
        algorithm.baseline is not None
        and environment.key in {
            'fetchpush', 'fetchslide', 'fetchpickandplace',
        }
    ):
        return 'goalreprv2_'
    return ''


def generate_all_tasks() -> list[Task]:
    tasks: list[Task] = []
    for algorithm in ALGORITHMS:
        for environment in ENVIRONMENTS:
            group = algorithm.key, environment.key
            if group in COMPLETED_GROUPS:
                continue
            size_label = environment.model_size.upper()
            step_label = f'{environment.total_steps // 1000}k'
            save_label = f'{environment.save_steps // 1000}k'
            params = compact_parameter_count(parameter_count(
                algorithm, environment,
            ))
            for seed in TRAINING_SEEDS:
                task_id = (
                    f'iclr_main_v1_{algorithm.label}-{size_label}_{step_label}_'
                    f'{save_label}ckpt_val{environment.validation_episodes}_'
                    f'test{environment.test_episodes}_'
                    f'{task_version_tag(algorithm, environment)}'
                    f'{environment.slug}_online_s{seed}'
                )
                extra_args = ' '.join((
                    f'env.kind={environment.kind}',
                    *algorithm_args(algorithm, environment),
                    f'interaction.validation_seed={environment.validation_seed}',
                    f'interaction.test_seed={environment.test_seed}',
                    'interaction.num_eval_episodes='
                    f'{environment.validation_episodes}',
                    f'interaction.num_test_episodes={environment.test_episodes}',
                    'eval_steps=null',
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


def partition_tasks(tasks: list[Task], partition: str) -> list[Task]:
    if partition == 'all':
        return list(tasks)
    groups = local_groups() if partition == 'local_2x' else REMOTE_GROUPS
    selected = [task for task in tasks if task_group(task) in groups]
    expected = {'local_2x': 145, 'remote_1x': 70}[partition]
    if len(selected) != expected:
        raise ValueError(
            f'Expected {expected} tasks for {partition}, got {len(selected)}'
        )
    return selected


def validate_tasks(tasks: list[Task]) -> None:
    if len(tasks) != 215:
        raise ValueError(f'Expected 215 missing tasks, got {len(tasks)}')
    duplicate_ids = [
        task_id for task_id, count
        in Counter(task.task_id for task in tasks).items() if count > 1
    ]
    if duplicate_ids:
        raise ValueError(f'Duplicate generated task IDs: {duplicate_ids}')
    actual_groups = {task_group(task) for task in tasks}
    if actual_groups != all_missing_groups():
        raise ValueError('Generated tasks do not cover exactly the missing cells')
    for group in actual_groups:
        seeds = {
            int(task.seed) for task in tasks if task_group(task) == group
        }
        if seeds != set(TRAINING_SEEDS):
            raise ValueError(f'Invalid training seeds for {group}: {seeds}')

    environments = {environment.key: environment for environment in ENVIRONMENTS}
    for task in tasks:
        _algorithm_key, environment_key = task_group(task)
        environment = environments[environment_key]
        args = extra_arg_map(task)
        required = {
            'env.kind': environment.kind,
            'interaction.validation_seed': str(environment.validation_seed),
            'interaction.test_seed': str(environment.test_seed),
            'interaction.num_eval_episodes': str(
                environment.validation_episodes
            ),
            'interaction.num_test_episodes': str(environment.test_episodes),
            'eval_steps': 'null',
            'save_steps': str(environment.save_steps),
            'keep_only_latest_checkpoint': 'true',
            'keep_only_best_and_final_checkpoints': 'true',
            'save_replay_buffer': 'true',
            'save_final_replay_buffer': 'true',
            'resume_if_possible': 'true',
        }
        for key, value in required.items():
            if args.get(key) != value:
                raise ValueError(
                    f'{task.task_id} requires {key}={value}, '
                    f'got {args.get(key)}'
                )
        validation_end = environment.validation_seed + (
            environment.validation_episodes - 1
        )
        test_end = environment.test_seed + environment.test_episodes - 1
        expected_ends = (
            (1499, 2499)
            if environment.total_steps == 500_000 else (1999, 2999)
        )
        if (validation_end, test_end) != expected_ends:
            raise ValueError(f'Invalid evaluation ranges for {task.task_id}')
        if int(task.steps) != environment.total_steps or task.mode != 'online':
            raise ValueError(f'Invalid mode/steps for {task.task_id}')
        if 'queue.delete_checkpoints_after_completion' in args:
            raise ValueError(f'{task.task_id} would delete retained checkpoints')


def validate_partitions(tasks: list[Task]) -> None:
    local = partition_tasks(tasks, 'local_2x')
    remote = partition_tasks(tasks, 'remote_1x')
    local_ids = {task.task_id for task in local}
    remote_ids = {task.task_id for task in remote}
    if local_ids & remote_ids:
        raise ValueError('Local and remote task partitions overlap')
    if local_ids | remote_ids != {task.task_id for task in tasks}:
        raise ValueError('Local and remote partitions do not cover all tasks')
    if len(local_groups()) != 29 or len(REMOTE_GROUPS) != 14:
        raise ValueError('Expected a 29:14 whole-group split')
    environment_levels = {
        environment.key: environment.model_size for environment in ENVIRONMENTS
    }
    local_l = sum(
        environment_levels[environment] == 'l'
        for _algorithm, environment in local_groups()
    )
    remote_l = sum(
        environment_levels[environment] == 'l'
        for _algorithm, environment in REMOTE_GROUPS
    )
    if (local_l, remote_l) != (12, 6):
        raise ValueError(f'Expected a 12:6 L-group split, got {local_l}:{remote_l}')


def task_line(task: Task) -> str:
    return '\t'.join((
        task.task_id, task.mode, task.env_name, task.seed, task.steps,
        task.params, task.extra_args,
    ))


def render_tasks(tasks: list[Task], partition: str) -> str:
    lines = [
        '# task_id\tmode\tenv_name\tseed\tsteps\tparams\textra_args',
        f'# ICLR main-table missing experiments; partition: {partition}.',
        '# Five training seeds (1000-1004) remain together per table cell.',
        '# 100k/200k: val seeds 1000-1999, test seeds 2000-2999.',
        '# 500k: val seeds 1000-1499, test seeds 1500-2499.',
        '# Training keeps one rolling best agent and one resumable latest state;',
        '# completion retains the best agent and resumable final checkpoint.',
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
        f'# ICLR main-table missing experiments: {partition} assignment.',
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

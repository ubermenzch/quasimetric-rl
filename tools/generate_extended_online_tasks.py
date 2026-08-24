#!/usr/bin/env python3
"""Generate the complete extended online QRL/CQRL experiment matrix."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quasimetric_rl.data.base import GOAL_SET_DIMS_REGISTRY
from quasimetric_rl.data.online.dmc import TASK_SPECS as DMC_TASK_SPECS
from quasimetric_rl.data.online.gymnasium_robotics import (
    TASK_SPECS as GYMNASIUM_ROBOTICS_TASK_SPECS,
)
from quasimetric_rl.data.online.panda_gym import (
    TASK_SPECS as PANDA_TASK_SPECS,
)
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


DEFAULT_OUTPUT = ROOT / 'configs/extended_online_9alg_34env_default_5seed.tsv'
TRAINING_SEEDS = tuple(range(1000, 1005))
MODEL_SIZE_LEVELS = ('m', 'l')
DEFAULT_TRAINING_PROTOCOLS = {
    'short': dict(model_size='m', total_steps=100_000, save_steps=20_000),
    'medium': dict(model_size='m', total_steps=200_000, save_steps=20_000),
    'long': dict(model_size='l', total_steps=500_000, save_steps=50_000),
}

DMC_TASKS = (
    'point_mass_easy',
    'finger_turn_easy',
    'manipulator_insert_ball',
    'manipulator_insert_peg',
    'dog_fetch',
    'stacker_stack_2',
    'ball_in_cup_catch',
)

SHADOW_HAND_TASKS = (
    'HandReach-v2',
    'HandManipulateBlockRotateZ-v1',
    'HandManipulateEggRotate-v1',
    'HandManipulatePenRotate-v1',
)

SHADOW_TOUCH_TASKS = tuple(
    f'{task_name.rsplit("-", 1)[0]}_{touch_kind}-{task_name.rsplit("-", 1)[1]}'
    for task_name in SHADOW_HAND_TASKS
    if task_name.startswith('HandManipulate')
    for touch_kind in ('BooleanTouchSensors', 'ContinuousTouchSensors')
)

MAZE_TASKS = (
    'PointMaze_UMaze-v3',
    'PointMaze_Open-v3',
    'PointMaze_Medium-v3',
    'AntMaze_UMaze-v5',
    'AntMaze_BigMaze_DGR-v5',
)

PANDA_TASKS = (
    'PandaReach-v3',
    'PandaPush-v3',
    'PandaSlide-v3',
    'PandaPickAndPlace-v3',
    'PandaStack-v3',
    'PandaFlip-v3',
)

PANDA_JOINT_TASKS = tuple(
    task_name.replace('-v3', 'Joints-v3') for task_name in PANDA_TASKS
)


@dataclass(frozen=True)
class Environment:
    family: str
    kind: str
    name: str
    state_dim: int
    action_dim: int
    goal_dim: int
    horizon: int

    @property
    def slug(self) -> str:
        return re.sub(r'[^a-z0-9]+', '_', self.name.lower()).strip('_')


@dataclass(frozen=True)
class Algorithm:
    key: str
    label: str
    baseline: str | None = None
    inner_steps: int | None = None


ALGORITHMS = (
    Algorithm('qrl', 'QRL'),
    Algorithm('td_infonce', 'TD-InfoNCE', baseline='td_infonce'),
    Algorithm('gcsl', 'GCSL', baseline='gcsl'),
    Algorithm('c_learning', 'C-Learning', baseline='c_learning'),
    Algorithm('crl', 'CRL', baseline='crl'),
    Algorithm('cqrl_inner0', 'CQRL/Inner0', inner_steps=0),
    Algorithm('cqrl_inner1', 'CQRL/Inner1', inner_steps=1),
    Algorithm('cqrl_inner4', 'CQRL/Inner4', inner_steps=4),
    Algorithm('cqrl_inner8', 'CQRL/Inner8', inner_steps=8),
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

CQRL_ARGS = (
    'agent.quasimetric_critic.losses.separate_latent_dynamics=false',
    'agent.quasimetric_critic.losses.latent_dynamics.distance=iqe_mse',
    'agent.quasimetric_critic.losses.latent_dynamics.mse_weight=1.0',
    'agent.quasimetric_critic.losses.latent_dynamics.iqe_weight=1.0',
    'agent.quasimetric_critic.model.encoder.branch_normalization=none',
    'agent.actor.losses.min_dist.latent_goal_mode=max',
    'agent.actor.losses.min_dist.latent_goal_keep_best=true',
    'agent.actor.losses.min_dist.latent_goal_lr=0.01',
    'agent.actor.losses.min_dist.latent_goal_search=direct',
    'agent.actor.losses.min_dist.latent_goal_optim=sgd',
)

CHECKPOINT_ARGS = (
    'eval_steps=null',
    'keep_only_latest_checkpoint=true',
    'keep_only_best_and_final_checkpoints=true',
    'save_replay_buffer=true',
    'save_final_replay_buffer=true',
    'resume_if_possible=true',
)


def _environment(
        family: str, kind: str, name: str, spec: dict,
        *, state_key: str, action_key: str) -> Environment:
    goal_dims = tuple(spec['goal_dims'])
    state_dim = int(spec[state_key])
    action_dim = int(spec[action_key])
    horizon = int(spec.get('episode_length', 1000))
    if goal_dims != tuple(range(len(goal_dims))):
        raise ValueError(
            f'Extended environment {kind}/{name} must expose a goal prefix, '
            f'got {goal_dims}'
        )
    registered = tuple(GOAL_SET_DIMS_REGISTRY[kind, name])
    if registered != goal_dims:
        raise ValueError(
            f'{kind}/{name} registration disagrees with TASK_SPECS: '
            f'{registered} != {goal_dims}'
        )
    if not 0 < len(goal_dims) < state_dim:
        raise ValueError(
            f'{kind}/{name} requires a partial goal, got goal_dim='
            f'{len(goal_dims)}, state_dim={state_dim}'
        )
    return Environment(
        family=family,
        kind=kind,
        name=name,
        state_dim=state_dim,
        action_dim=action_dim,
        goal_dim=len(goal_dims),
        horizon=horizon,
    )


def build_environment_catalog() -> tuple[Environment, ...]:
    environments = []
    for name in DMC_TASKS:
        environments.append(_environment(
            'dmc', 'dmc', name, DMC_TASK_SPECS[name],
            state_key='state_dim', action_key='action_dim',
        ))
    for family, names in (
            ('shadow-hand', SHADOW_HAND_TASKS),
            ('shadow-touch', SHADOW_TOUCH_TASKS),
            ('maze', MAZE_TASKS)):
        for name in names:
            environments.append(_environment(
                family, 'gymnasium_robotics', name,
                GYMNASIUM_ROBOTICS_TASK_SPECS[name],
                state_key='state_dim', action_key='action_dim',
            ))
    for family, names in (
            ('panda', PANDA_TASKS),
            ('panda-joints', PANDA_JOINT_TASKS)):
        for name in names:
            environments.append(_environment(
                family, 'panda_gym', name, PANDA_TASK_SPECS[name],
                state_key='state_size', action_key='action_size',
            ))
    result = tuple(environments)
    if len(result) != 34 or len({(env.kind, env.name) for env in result}) != 34:
        raise ValueError(
            f'Expected 34 unique extended environments, got {len(result)}'
        )
    return result


ENVIRONMENTS = build_environment_catalog()
FAMILIES = tuple(dict.fromkeys(environment.family for environment in ENVIRONMENTS))
ALGORITHM_KEYS = tuple(algorithm.key for algorithm in ALGORITHMS)


def evaluation_episode_count(environment: Environment) -> int:
    return 200 if environment.horizon <= 100 else 100


def default_training_protocol(environment: Environment) -> dict:
    """Select the primary capacity and interaction budget from the horizon."""
    if environment.horizon <= 100:
        return DEFAULT_TRAINING_PROTOCOLS['short']
    if environment.horizon < 1000:
        return DEFAULT_TRAINING_PROTOCOLS['medium']
    return DEFAULT_TRAINING_PROTOCOLS['long']


def align_total_steps(environment: Environment, requested_steps: int) -> int:
    """Round a budget up so the fixed-length collector ends on an episode."""
    requested_steps = int(requested_steps)
    if requested_steps <= 0:
        raise ValueError('requested_steps must be positive')
    return (
        (requested_steps + environment.horizon - 1)
        // environment.horizon
        * environment.horizon
    )


def step_label(total_steps: int) -> str:
    if total_steps % 1000 == 0:
        return f'{total_steps // 1000}k'
    return f'{total_steps}steps'


def gcsl_uses_factorized_actions(environment: Environment) -> bool:
    """Avoid the exponential 3**action_dim head on high-DoF robots."""
    return environment.action_dim >= 7


@lru_cache(maxsize=None)
def parameter_count(
        algorithm_key: str, environment: Environment, model_size: str) -> int:
    algorithm = next(
        item for item in ALGORITHMS if item.key == algorithm_key
    )
    if algorithm.key == 'qrl':
        return qrl_agent_parameter_count(
            environment.state_dim, environment.action_dim, model_size,
        )
    if algorithm.inner_steps is not None:
        return go_qrl_agent_parameter_count(
            environment.state_dim,
            environment.action_dim,
            environment.goal_dim,
            model_size,
        )
    assert algorithm.baseline is not None
    goal_dims = resolve_baseline_goal_dims(
        algorithm.baseline,
        env_kind=environment.kind,
        env_name=environment.name,
        state_dim=environment.state_dim,
        success_goal_dims=tuple(range(environment.goal_dim)),
    )
    return baseline_parameter_count(
        algorithm.baseline,
        environment.state_dim,
        environment.action_dim,
        len(goal_dims),
        model_size_level=model_size,
        gcsl_action_discretization=(
            'factorized'
            if algorithm.baseline == 'gcsl'
            and gcsl_uses_factorized_actions(environment)
            else 'joint'
        ),
    )


def algorithm_args(
        algorithm: Algorithm, environment: Environment,
        model_size: str) -> tuple[str, ...]:
    if algorithm.key == 'qrl':
        return (f'+qrl_model_size={model_size}', *QRL_COMMON_ARGS)
    if algorithm.inner_steps is not None:
        return (
            f'+cqrl_model_size={model_size}',
            *QRL_COMMON_ARGS,
            *CQRL_ARGS,
            'agent.actor.losses.min_dist.latent_goal_steps='
            f'{algorithm.inner_steps}',
        )
    assert algorithm.baseline is not None
    args = [
        f'agent.algorithm={algorithm.baseline}',
        f'+{MODEL_SIZE_FAMILIES[algorithm.baseline]}_model_size={model_size}',
        'batch_size=256',
        'interaction.exploration_eps=0',
    ]
    if (
            algorithm.baseline == 'gcsl'
            and gcsl_uses_factorized_actions(environment)):
        args.append('agent.baselines.gcbc.action_discretization=factorized')
    return tuple(args)


def _select_algorithms(keys: Iterable[str] | None) -> tuple[Algorithm, ...]:
    selected = set(ALGORITHM_KEYS if keys is None else keys)
    unknown = selected.difference(ALGORITHM_KEYS)
    if unknown:
        raise ValueError(f'Unknown algorithms: {sorted(unknown)}')
    return tuple(algorithm for algorithm in ALGORITHMS if algorithm.key in selected)


def _select_environments(
        families: Iterable[str] | None,
        max_horizon: int | None = None,
) -> tuple[Environment, ...]:
    selected = set(FAMILIES if families is None else families)
    unknown = selected.difference(FAMILIES)
    if unknown:
        raise ValueError(f'Unknown environment families: {sorted(unknown)}')
    environments = tuple(
        env for env in ENVIRONMENTS if env.family in selected
    )
    if max_horizon is not None:
        max_horizon = int(max_horizon)
        if max_horizon <= 0:
            raise ValueError('max_horizon must be positive')
        environments = tuple(
            environment for environment in environments
            if environment.horizon <= max_horizon
        )
    if not environments:
        raise ValueError('Environment filters selected no tasks')
    return environments


def generate_tasks(
        *, algorithms: Iterable[str] | None = None,
        families: Iterable[str] | None = None,
        max_horizon: int | None = None,
        model_sizes: Iterable[str] | None = None,
        seeds: Sequence[int] = TRAINING_SEEDS) -> list[Task]:
    selected_algorithms = _select_algorithms(algorithms)
    selected_environments = _select_environments(families, max_horizon)
    selected_sizes = None
    if model_sizes is not None:
        selected_sizes = tuple(dict.fromkeys(
            str(size).lower() for size in model_sizes
        ))
        unknown_sizes = set(selected_sizes).difference(MODEL_SIZE_LEVELS)
        if unknown_sizes:
            raise ValueError(f'Unknown model sizes: {sorted(unknown_sizes)}')
        if not selected_sizes:
            raise ValueError('Explicit model sizes must be non-empty')
    selected_seeds = tuple(int(seed) for seed in seeds)
    if not selected_seeds or len(set(selected_seeds)) != len(selected_seeds):
        raise ValueError('Training seeds must be non-empty and unique')

    tasks = []
    for environment in selected_environments:
        episodes = evaluation_episode_count(environment)
        protocol = default_training_protocol(environment)
        environment_sizes = (
            (protocol['model_size'],)
            if selected_sizes is None else selected_sizes
        )
        total_steps = align_total_steps(
            environment, int(protocol['total_steps']),
        )
        save_steps = int(protocol['save_steps'])
        total_step_label = step_label(total_steps)
        checkpoint_label = f'{save_steps // 1000}kckpt'
        for algorithm in selected_algorithms:
            id_label = algorithm.label.replace('/', '-')
            for model_size in environment_sizes:
                count = compact_parameter_count(parameter_count(
                    algorithm.key, environment, model_size,
                ))
                gcsl_tag = (
                    'factorized_'
                    if algorithm.baseline == 'gcsl'
                    and gcsl_uses_factorized_actions(environment)
                    else ''
                )
                for seed in selected_seeds:
                    task_id = (
                        f'extended_v1_{id_label}-{model_size.upper()}_'
                        f'{total_step_label}_{checkpoint_label}_val{episodes}_'
                        f'test{episodes}_{gcsl_tag}{environment.slug}_'
                        f'online_s{seed}'
                    )
                    extra_args = ' '.join((
                        f'env.kind={environment.kind}',
                        f'env.init_num_transitions={total_steps}',
                        f'env.increment_num_transitions={total_steps}',
                        *algorithm_args(algorithm, environment, model_size),
                        'interaction.validation_seed=1000',
                        'interaction.test_seed=2000000',
                        f'interaction.num_eval_episodes={episodes}',
                        f'interaction.num_test_episodes={episodes}',
                        f'save_steps={save_steps}',
                        *CHECKPOINT_ARGS,
                    ))
                    tasks.append(Task(
                        task_id=task_id,
                        mode='online',
                        env_name=environment.name,
                        seed=str(seed),
                        steps=str(total_steps),
                        params=count,
                        extra_args=extra_args,
                    ))
    validate_tasks(
        tasks,
        algorithms=selected_algorithms,
        environments=selected_environments,
        model_sizes=selected_sizes,
        seeds=selected_seeds,
    )
    return tasks


def validate_tasks(
        tasks: Sequence[Task], *, algorithms: Sequence[Algorithm],
        environments: Sequence[Environment],
        model_sizes: Sequence[str] | None,
        seeds: Sequence[int]) -> None:
    sizes_by_environment = {
        environment.name: (
            (default_training_protocol(environment)['model_size'],)
            if model_sizes is None else tuple(model_sizes)
        )
        for environment in environments
    }
    expected = len(algorithms) * len(seeds) * sum(
        len(sizes) for sizes in sizes_by_environment.values()
    )
    if len(tasks) != expected:
        raise ValueError(f'Expected {expected} tasks, got {len(tasks)}')
    duplicate_ids = [
        task_id for task_id, count
        in Counter(task.task_id for task in tasks).items() if count > 1
    ]
    if duplicate_ids:
        raise ValueError(f'Duplicate generated task IDs: {duplicate_ids}')
    expected_cells = {
        (environment.name, algorithm.label, size, int(seed))
        for environment in environments
        for algorithm in algorithms
        for size in sizes_by_environment[environment.name]
        for seed in seeds
    }
    actual_cells = set()
    for task in tasks:
        algorithm = next(
            candidate for candidate in algorithms
            if f'_{candidate.label.replace("/", "-")}-' in task.task_id
        )
        size = next(
            candidate for candidate in sizes_by_environment[task.env_name]
            if f'-{candidate.upper()}_' in task.task_id
        )
        actual_cells.add((task.env_name, algorithm.label, size, int(task.seed)))
    if actual_cells != expected_cells:
        raise ValueError('Generated tasks do not cover the requested factorial matrix')


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


def render_tasks(
        tasks: Sequence[Task], *, partition: str = 'all') -> str:
    lines = [
        '# task_id\tmode\tenv_name\tseed\tsteps\tparams\textra_args',
        '# Extended online goal tasks generated from the requested environment, '
        'algorithm, capacity, and seed filters.',
        f'# Assignment partition: {partition}.',
        '# Default protocol: horizon <=100 uses M/100k; horizon 101--999 uses '
        'M/200k; horizon >=1000 uses L/500k. Total steps are rounded up to a '
        'complete episode.',
        '# GCSL uses independent per-action 3-bin heads when action_dim >= 7; '
        'task IDs mark those runs as factorized.',
    ]
    lines.extend(task_line(task) for task in tasks)
    return '\n'.join(lines) + '\n'


def _task_algorithm(task: Task, algorithms: Sequence[Algorithm]) -> Algorithm:
    matches = [
        algorithm for algorithm in algorithms
        if f'_{algorithm.label.replace("/", "-")}-' in task.task_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f'Could not identify one algorithm for {task.task_id}: {matches}'
        )
    return matches[0]


def partition_tasks(
        tasks: Sequence[Task], partition: str, *,
        algorithms: Sequence[Algorithm],
        environments: Sequence[Environment],
        seeds: Sequence[int]) -> list[Task]:
    if partition == 'all':
        return list(tasks)
    if partition not in ('local_2x', 'remote_1x'):
        raise ValueError(f'Unknown partition: {partition!r}')

    environment_indices = {
        environment.name: index
        for index, environment in enumerate(environments)
    }
    algorithm_indices = {
        algorithm.key: index for index, algorithm in enumerate(algorithms)
    }
    seed_indices = {int(seed): index for index, seed in enumerate(seeds)}
    selected = []
    for task in tasks:
        environment_index = environment_indices[task.env_name]
        algorithm = _task_algorithm(task, algorithms)
        algorithm_index = algorithm_indices[algorithm.key]
        seed_index = seed_indices[int(task.seed)]
        within_environment = algorithm_index * len(seeds) + seed_index
        is_local = (within_environment + environment_index) % 3 != 2
        if (partition == 'local_2x') == is_local:
            selected.append(task)
    return selected


def validate_two_to_one_partitions(
        tasks: Sequence[Task], *, algorithms: Sequence[Algorithm],
        environments: Sequence[Environment], seeds: Sequence[int]) -> None:
    local = partition_tasks(
        tasks, 'local_2x', algorithms=algorithms,
        environments=environments, seeds=seeds,
    )
    remote = partition_tasks(
        tasks, 'remote_1x', algorithms=algorithms,
        environments=environments, seeds=seeds,
    )
    all_ids = {task.task_id for task in tasks}
    local_ids = {task.task_id for task in local}
    remote_ids = {task.task_id for task in remote}
    if local_ids & remote_ids or local_ids | remote_ids != all_ids:
        raise ValueError('The 2:1 partitions must be disjoint and complete')
    if len(tasks) % 3 or len(local) * 3 != len(tasks) * 2:
        raise ValueError(
            f'Cannot form an exact 2:1 split from {len(tasks)} tasks'
        )

    dimensions = (
        ('environment', lambda task: task.env_name,
         {environment.name for environment in environments}),
        ('algorithm', lambda task: _task_algorithm(task, algorithms).key,
         {algorithm.key for algorithm in algorithms}),
        ('seed', lambda task: int(task.seed), {int(seed) for seed in seeds}),
    )
    for dimension, key, expected_values in dimensions:
        all_counts = Counter(key(task) for task in tasks)
        local_counts = Counter(key(task) for task in local)
        remote_counts = Counter(key(task) for task in remote)
        if set(all_counts) != expected_values:
            raise ValueError(f'Invalid {dimension} coverage in full matrix')
        for value, count in all_counts.items():
            if count % 3:
                raise ValueError(
                    f'{dimension}={value} has {count} tasks, not divisible by 3'
                )
            if (
                    local_counts[value] != 2 * count // 3
                    or remote_counts[value] != count // 3):
                raise ValueError(
                    f'Unbalanced {dimension}={value}: '
                    f'{local_counts[value]}:{remote_counts[value]}'
                )


def append_tasks(path: Path, tasks: Sequence[Task], partition: str) -> int:
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
        f'# Extended online goal tasks: {partition} assignment.',
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
    parser.add_argument(
        '--families', nargs='+', choices=FAMILIES, default=FAMILIES,
    )
    parser.add_argument(
        '--algorithms', nargs='+', choices=ALGORITHM_KEYS,
        default=ALGORITHM_KEYS,
    )
    parser.add_argument(
        '--model-sizes', nargs='+', choices=MODEL_SIZE_LEVELS, default=None,
        help=(
            'Override the horizon-based default capacity. One or both sizes '
            'may be requested; the environment-specific step budget is kept.'
        ),
    )
    parser.add_argument(
        '--max-horizon', type=int, default=None,
        help='Keep only environments with a native horizon at or below this value.',
    )
    parser.add_argument(
        '--partition', choices=('all', 'local_2x', 'remote_1x'), default='all',
    )
    parser.add_argument('--append-to', type=Path, default=None)
    parser.add_argument('--seeds', nargs='+', type=int, default=TRAINING_SEEDS)
    args = parser.parse_args()

    selected_algorithms = _select_algorithms(args.algorithms)
    selected_environments = _select_environments(
        args.families, args.max_horizon,
    )
    all_tasks = generate_tasks(
        algorithms=args.algorithms,
        families=args.families,
        max_horizon=args.max_horizon,
        model_sizes=args.model_sizes,
        seeds=args.seeds,
    )
    if args.partition != 'all':
        validate_two_to_one_partitions(
            all_tasks, algorithms=selected_algorithms,
            environments=selected_environments, seeds=args.seeds,
        )
    tasks = partition_tasks(
        all_tasks, args.partition, algorithms=selected_algorithms,
        environments=selected_environments, seeds=args.seeds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_tasks(tasks, partition=args.partition))
    print(f'Wrote {len(tasks)} tasks to {args.output}')
    if args.append_to is not None:
        added = append_tasks(args.append_to, tasks, args.partition)
        print(f'Appended {added} new tasks to {args.append_to}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

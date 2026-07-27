#!/usr/bin/env python3
"""Generate the QRL/GO-QRL online goal-environment task matrix."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quasimetric_rl.model_size import (
    go_qrl_agent_parameter_count,
    qrl_agent_parameter_count,
)
from tools.run_qrl_queue import Task, read_tasks


DEFAULT_OUTPUT = ROOT / 'configs/qrl_tasks_online_goal_envs_5variants_5seeds_200k.tsv'
TOTAL_ENV_STEPS = 200_000
TRAINING_SEEDS = tuple(range(1000, 1005))
MODEL_SIZE_LEVEL = 'm'

ENVIRONMENTS = (
    ('gcrl', 'FetchReach', 'fetchreach'),
    ('gcrl', 'FetchPush', 'fetchpush'),
    ('gcrl', 'FetchSlide', 'fetchslide'),
    ('gcrl', 'FetchPickAndPlace', 'fetchpickandplace'),
    ('gym_mujoco', 'Reacher-v4', 'reacher_v4'),
    ('gym_mujoco', 'Pusher-v4', 'pusher_v4'),
    ('gym_mujoco', 'AntNavigate-v4', 'antnavigate_v4'),
    ('dmc', 'reacher_easy', 'dmc_reacher_easy'),
    ('dmc', 'reacher_hard', 'dmc_reacher_hard'),
    ('dmc', 'manipulator_bring_ball', 'dmc_manipulator_bring_ball'),
    ('dmc', 'manipulator_bring_peg', 'dmc_manipulator_bring_peg'),
)

ENVIRONMENT_DIMS = {
    ('gcrl', 'FetchReach'): (10, 4, 3),
    ('gcrl', 'FetchPush'): (25, 4, 3),
    ('gcrl', 'FetchSlide'): (25, 4, 3),
    ('gcrl', 'FetchPickAndPlace'): (25, 4, 3),
    ('gym_mujoco', 'Reacher-v4'): (8, 2, 2),
    ('gym_mujoco', 'Pusher-v4'): (20, 7, 3),
    ('gym_mujoco', 'AntNavigate-v4'): (113, 8, 2),
    ('dmc', 'reacher_easy'): (6, 2, 2),
    ('dmc', 'reacher_hard'): (6, 2, 2),
    ('dmc', 'manipulator_bring_ball'): (40, 5, 2),
    ('dmc', 'manipulator_bring_peg'): (40, 5, 4),
}

LONG_HORIZON_ENVIRONMENTS = frozenset({
    ('gym_mujoco', 'AntNavigate-v4'),
    ('dmc', 'reacher_easy'),
    ('dmc', 'reacher_hard'),
    ('dmc', 'manipulator_bring_ball'),
    ('dmc', 'manipulator_bring_peg'),
})

GO_QRL_ARGS = (
    'agent.actor.losses.min_dist.latent_goal_steps=4',
    'agent.actor.losses.min_dist.latent_goal_keep_best=true',
    'agent.actor.losses.min_dist.latent_goal_lr=0.01',
    'agent.actor.losses.min_dist.latent_goal_search=direct',
)

COMMON_ARGS = (
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


def evaluation_episode_count(env_kind: str, env_name: str) -> int:
    if (env_kind, env_name) in LONG_HORIZON_ENVIRONMENTS:
        return 100
    if (env_kind, env_name) == ('gcrl', 'FetchReach'):
        return 1000
    return 200

VARIANTS = (
    ('Base', 'auto', ()),
    (
        'GO-QRL+Max4',
        'auto',
        GO_QRL_ARGS + (
            'agent.quasimetric_critic.model.encoder.branch_normalization=none',
            'agent.actor.losses.min_dist.latent_goal_mode=max',
            'agent.actor.losses.min_dist.latent_goal_optim=sgd',
        ),
    ),
    (
        'GO-QRL+Min4',
        'auto',
        GO_QRL_ARGS + (
            'agent.quasimetric_critic.model.encoder.branch_normalization=none',
            'agent.actor.losses.min_dist.latent_goal_mode=min',
            'agent.actor.losses.min_dist.latent_goal_optim=sgd',
        ),
    ),
    (
        'GO-QRL+Max4+LN+RMSG',
        'auto',
        GO_QRL_ARGS + (
            'agent.quasimetric_critic.model.encoder.branch_normalization=layernorm',
            'agent.actor.losses.min_dist.latent_goal_mode=max',
            'agent.actor.losses.min_dist.latent_goal_optim=rmsg',
        ),
    ),
    (
        'GO-QRL+Min4+LN+RMSG',
        'auto',
        GO_QRL_ARGS + (
            'agent.quasimetric_critic.model.encoder.branch_normalization=layernorm',
            'agent.actor.losses.min_dist.latent_goal_mode=min',
            'agent.actor.losses.min_dist.latent_goal_optim=rmsg',
        ),
    ),
)


def compact_parameter_count(count: int) -> str:
    if count >= 1_000_000:
        return f'{count / 1_000_000:.1f}m'
    if count >= 1_000:
        return f'{count / 1_000:.1f}k'
    return str(count)


def resolve_variant_for_environment(
        env_kind: str, env_name: str, variant: str,
        params: str, variant_args: tuple[str, ...],
) -> tuple[str, str, tuple[str, ...]]:
    state_dim, action_dim, goal_dim = ENVIRONMENT_DIMS[env_kind, env_name]
    level = MODEL_SIZE_LEVEL
    if variant == 'Base':
        parameter_count = qrl_agent_parameter_count(state_dim, action_dim, level)
        return (
            f'Base-{level.upper()}',
            compact_parameter_count(parameter_count),
            (f'+qrl_model_size={level}',),
        )
    if variant.startswith('GO-QRL+'):
        parameter_count = go_qrl_agent_parameter_count(
            state_dim, action_dim, goal_dim, level,
        )
        return (
            f'{variant}-{level.upper()}',
            compact_parameter_count(parameter_count),
            (f'+go_qrl_model_size={level}', *variant_args),
        )
    return variant, params, variant_args


def generate_tasks() -> list[Task]:
    tasks = []
    for env_kind, env_name, env_slug in ENVIRONMENTS:
        evaluation_episodes = evaluation_episode_count(env_kind, env_name)
        for variant, params, variant_args in VARIANTS:
            variant, params, variant_args = resolve_variant_for_environment(
                env_kind, env_name, variant, params, variant_args,
            )
            for seed in TRAINING_SEEDS:
                task_id = (
                    f'official_qrl_1q_{variant}_200k_20kckpt_'
                    f'val{evaluation_episodes}_test{evaluation_episodes}_'
                    f'{env_slug}_online_s{seed}'
                )
                extra_args = ' '.join((
                    f'env.kind={env_kind}',
                    *variant_args,
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
                    params=params,
                    extra_args=extra_args,
                ))
    validate_tasks(tasks)
    return tasks


def validate_tasks(tasks: list[Task]) -> None:
    expected = len(ENVIRONMENTS) * len(VARIANTS) * len(TRAINING_SEEDS)
    if len(tasks) != expected:
        raise ValueError(f'Expected {expected} tasks, got {len(tasks)}')
    ids = [task.task_id for task in tasks]
    duplicates = [task_id for task_id, count in Counter(ids).items() if count > 1]
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
        '# Five online schemes x five training seeds for eleven goal environments.',
        '# Every task uses the M model-size preset.',
        '# Checkpoints every 20k; validation and test use the same episode count.',
        '# FetchReach stays at 1000, other short-horizon environments use 200, and 1000-step environments use 100.',
    ]
    lines.extend(task_line(task) for task in tasks)
    return '\n'.join(lines) + '\n'


def append_tasks(path: Path, tasks: list[Task]) -> int:
    existing_tasks = read_tasks(path)
    existing_ids = {task.task_id for task in existing_tasks}
    additions = [task for task in tasks if task.task_id not in existing_ids]
    if not additions:
        return 0
    current = path.read_text() if path.exists() else ''
    separator = '' if not current or current.endswith('\n\n') else ('\n' if current.endswith('\n') else '\n\n')
    block = '\n'.join([
        '# Online goal environments: 5 schemes x 5 seeds x 11 environments.',
        '# Every task uses the M model-size preset.',
        '# 200k training steps; validation and test use the same episode count.',
        '# FetchReach stays at 1000, other short-horizon environments use 200, and 1000-step environments use 100.',
        *(task_line(task) for task in additions),
    ]) + '\n'
    path.write_text(current + separator + block)
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
        print(f'Added {added} tasks to {args.append_to}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

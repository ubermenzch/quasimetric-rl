#!/usr/bin/env python3
"""Generate and partition the 95-task GO-QRL inner-step ablation matrix."""

from __future__ import annotations

import argparse
from collections import Counter
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
        '--partition', choices=tuple(DEFAULT_OUTPUTS), default='all',
    )
    parser.add_argument('--output', type=Path)
    parser.add_argument('--append-to', type=Path)
    parser.add_argument('--write-all-partitions', action='store_true')
    args = parser.parse_args()

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

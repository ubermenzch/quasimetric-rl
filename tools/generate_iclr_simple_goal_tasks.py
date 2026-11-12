#!/usr/bin/env python3
"""Queue matched mask, goal-only, and real replay-target controls (27 runs)."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quasimetric_rl.model_size import (
    _configured_mlp_parameter_count, load_model_size_preset, qrl_agent_parameter_count,
)
from tools.generate_nqrl_search_ablation_tasks import generate_tasks as search_tasks
from tools.generate_selected_success_l_min32_tasks import atomic_write, extra_arg_map, task_line
from tools.run_qrl_queue import read_tasks, task_fingerprint

QUEUE = ROOT / 'runs/qrl_queue/tasks.tsv'
OUTPUT = ROOT / 'configs/iclr_rebuttal_simple_goals_3env_3seed.tsv'
SEEDS = (1000, 1001, 1002)
# Dimensions and success tolerances in the actual task coordinates. Reacher-H:
# target radius .015 plus finger radius .01, as in dm_control's native reward.
ENVIRONMENTS = {
    'FetchPickAndPlace': (25, 4, (3, 4, 5), 0.05),
    'maze2d-large': (4, 2, (0, 1), 0.5),
    'reacher_hard': (6, 2, (0, 1), 0.025),
}
VARIANTS = ('QRL-Mask', 'QRL-GoalOnly', 'AQRL-ReplayMin8')
LOSS_PREFIX = 'agent.actor.losses.min_dist.'


def input_control_parameters(name, mode):
    state_dim, action_dim, goal_dims, _ = ENVIRONMENTS[name]
    preset = load_model_size_preset('qrl', 'l_aqrl_backbone')
    actor = preset.actor.model
    args = dict(hidden_sizes=actor.arch, kind=actor.mlp_kind,
                residual_block_size=actor.residual_block_size)
    original = _configured_mlp_parameter_count(2 * state_dim, output_size=2 * action_dim, **args)
    new_input = 3 * state_dim if mode == 'masked' else state_dim + len(goal_dims)
    updated = _configured_mlp_parameter_count(new_input, output_size=2 * action_dim, **args)
    return qrl_agent_parameter_count(state_dim, action_dim, 'l_aqrl_backbone') + updated - original


def generate_tasks(existing):
    by_id = {task.task_id: task for task in existing}
    if len(by_id) != len(existing):
        raise ValueError('The queue contains duplicate task IDs')
    sources = {(task.env_name, int(task.seed)): by_id[task.task_id]
               for task in search_tasks()
               if task.env_name in ENVIRONMENTS and int(task.seed) in SEEDS
               and extra_arg_map(task)[LOSS_PREFIX + 'latent_goal_steps'] == '0'}
    tasks = []
    for variant in VARIANTS:
        for name, (_, _, goal_dims, radius) in ENVIRONMENTS.items():
            for seed in SEEDS:
                source = sources[name, seed]
                values = extra_arg_map(source)
                if (values['+cqrl_model_size'] != 'l'
                        or values['agent.quasimetric_critic.losses.latent_dynamics.distance'] != 'iqe_mse'
                        or float(values['agent.quasimetric_critic.losses.latent_dynamics.mse_weight']) != 1):
                    raise ValueError(f'Unexpected AQRL/0 source: {source.task_id}')
                values[LOSS_PREFIX + 'latent_goal_mode'] = 'none'
                values[LOSS_PREFIX + 'latent_goal_keep_best'] = 'false'
                params = source.params
                if variant.startswith('QRL-'):
                    del values['+cqrl_model_size']
                    values['+qrl_model_size'] = 'l_aqrl_backbone'
                    mode = 'masked' if variant == 'QRL-Mask' else 'goal_only'
                    values['agent.actor.model.input_mode'] = mode
                    values['agent.actor.model.goal_dims'] = '[' + ','.join(map(str, goal_dims)) + ']'
                    params = str(input_control_parameters(name, mode))
                else:
                    values[LOSS_PREFIX + 'replay_goal_candidates'] = '8'
                    values[LOSS_PREFIX + 'replay_goal_radius'] = str(radius)
                slug = name.lower().replace('-', '_')
                tasks.append(replace(
                    source,
                    task_id=(f'iclr_rebuttal_v1_{variant}-L_'
                             f'{source.steps}steps_{slug}_online_s{seed}'),
                    params=params,
                    extra_args=' '.join(f'{key}={value}' for key, value in values.items()),
                ))
    assert len(tasks) == len({task.task_id for task in tasks}) == 27
    return tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--append', action='store_true')
    args = parser.parse_args()
    before = QUEUE.read_text()
    existing = read_tasks(QUEUE)
    by_id = {task.task_id: task for task in existing}
    requested = generate_tasks(existing)
    from tools.set_pending_logs_only import preserve_storage_policy
    requested = preserve_storage_policy(requested, existing)
    additions = []
    for task in requested:
        if task.task_id in by_id:
            if task_fingerprint(task) != task_fingerprint(by_id[task.task_id]):
                raise ValueError(f'Existing task has different settings: {task.task_id}')
        else:
            additions.append(task)
    if args.append:
        if QUEUE.read_text() != before:
            raise RuntimeError('Task list changed; rerun before appending')
        if additions:
            backup = QUEUE.with_name('tasks.tsv.before_iclr_simple_goals_20261108')
            if not backup.exists():
                backup.write_text(before)
            atomic_write(QUEUE, before.rstrip('\n') + '\n\n'
                         '# Reviewer controls: Mask, GoalOnly, ReplayMin8; 3 tasks x 3 seeds each.\n'
                         + '\n'.join(task_line(task) for task in additions) + '\n')
        after = read_tasks(QUEUE)
        assert after[:len(existing)] == existing
        assert len(after) == len({task.task_id for task in after}) == len(existing) + len(additions)
        header = '# task_id\tmode\tenv_name\tseed\tsteps\tparams\textra_args\n'
        atomic_write(OUTPUT, header + '\n'.join(task_line(task) for task in requested) + '\n')
        for index, variant in enumerate(VARIANTS):
            group = requested[index * 9:(index + 1) * 9]
            atomic_write(ROOT / f'configs/iclr_rebuttal_{variant.lower()}_3env_3seed.tsv',
                         header + '\n'.join(task_line(task) for task in group) + '\n')
        indices = {task.task_id: index for index, task in enumerate(after, 1)}
        print('Task indices:', [indices[task.task_id] for task in requested])
    print(f'Requested: {len(requested)}; new: {len(additions)}; seeds: {SEEDS}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Rerun three main-table tasks with opt-in completion-displacement logging."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.analyze_nqrl_paper_results import ENVIRONMENTS, artifact_dir
from tools.generate_selected_success_l_min32_tasks import atomic_write, extra_arg_map, task_line
from tools.run_qrl_queue import (
    DELETE_CHECKPOINTS_AFTER_COMPLETION_ARG, read_tasks, task_fingerprint,
)

QUEUE = ROOT / 'runs/qrl_queue/tasks.tsv'
OUTPUT = ROOT / 'configs/iclr_rebuttal_displacement_3env_3seed.tsv'
ENV_NAMES = ('PointMaze_Medium-v3', 'FetchPickAndPlace', 'FetchPush')
SEEDS = (1000, 1001, 1002)
LOG_KEY = 'agent.actor.losses.min_dist.latent_goal_log_displacement'


def generate_tasks(existing):
    by_id = {task.task_id: task for task in existing}
    environments = {env.paper_name.replace('PointMaze Medium-v3', 'PointMaze_Medium-v3'): env
                    for env in ENVIRONMENTS}
    tasks = []
    for name in ENV_NAMES:
        environment = environments[name]
        for seed in SEEDS:
            source = by_id[artifact_dir(environment, 'NQRL', seed).name]
            values = extra_arg_map(source)
            assert values['+cqrl_model_size'] == 'l'
            assert values['agent.actor.losses.min_dist.latent_goal_mode'] == 'min'
            assert values['agent.actor.losses.min_dist.latent_goal_steps'] == '32'
            assert float(values['agent.actor.losses.min_dist.latent_goal_lr']) == 1
            assert values['agent.quasimetric_critic.losses.latent_dynamics.distance'] == 'iqe_mse'
            assert float(values['agent.quasimetric_critic.losses.latent_dynamics.mse_weight']) == 1
            assert int(source.steps) == environment.budget
            values[LOG_KEY] = 'true'
            values[DELETE_CHECKPOINTS_AFTER_COMPLETION_ARG] = 'false'
            tasks.append(replace(
                source,
                task_id=(f'iclr_rebuttal_v1_AQRL-Displacement-L-InnerLR1_'
                         f'{source.steps}steps_{environment.slug}_online_s{seed}'),
                extra_args=' '.join(f'{key}={value}' for key, value in values.items()),
            ))
    assert len(tasks) == len({task.task_id for task in tasks}) == 9
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
            assert task_fingerprint(task) == task_fingerprint(by_id[task.task_id])
        else:
            additions.append(task)
    if args.append:
        assert QUEUE.read_text() == before, 'Task list changed; rerun.'
        if additions:
            backup = QUEUE.with_name('tasks.tsv.before_iclr_displacement_20261108')
            if not backup.exists():
                backup.write_text(before)
            atomic_write(QUEUE, before.rstrip('\n') + '\n\n'
                         '# Q3 displacement diagnostics: three tasks x three seeds; K=32.\n'
                         + '\n'.join(task_line(task) for task in additions) + '\n')
        after = read_tasks(QUEUE)
        assert after[:len(existing)] == existing
        assert len(after) == len({task.task_id for task in after}) == len(existing) + len(additions)
        atomic_write(OUTPUT, '# task_id\tmode\tenv_name\tseed\tsteps\tparams\textra_args\n'
                     + '\n'.join(task_line(task) for task in requested) + '\n')
        indices = {task.task_id: i for i, task in enumerate(after, 1)}
        print('Task indices:', [indices[task.task_id] for task in requested])
    print(f'Requested: {len(requested)}; new: {len(additions)}; '
          f'tasks: {ENV_NAMES}; seeds: {SEEDS}; K: 32')


if __name__ == '__main__':
    main()

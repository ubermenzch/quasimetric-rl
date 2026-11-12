#!/usr/bin/env python3
"""Complete the main 14-environment NQRL/0, /8, /16 five-seed grid."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from tools.generate_selected_success_l_min32_tasks import (
    atomic_write, extra_arg_map, generate_main_min32_tasks,
    generate_selected_tasks, task_line,
)
from tools.run_qrl_queue import Task, read_status, read_tasks

QUEUE = ROOT / 'runs/qrl_queue/tasks.tsv'
STATUS = ROOT / 'runs/qrl_queue/status'
OUTPUT = ROOT / 'configs/nqrl_search_0_8_16_l_14env_5seed.tsv'
OPERATIONAL = {
    'resume_if_possible', 'keep_only_latest_checkpoint',
    'keep_only_best_and_final_checkpoints', 'save_replay_buffer',
    'save_final_replay_buffer',
}


def generate_tasks():
    templates = generate_main_min32_tasks() + [
        task for task in generate_selected_tasks()
        if '_CQRL+Min32-L-InnerLR1_' in task.task_id
    ]
    tasks = []
    for task in templates:
        for steps in (0, 8, 16):
            tasks.append(replace(
                task,
                task_id=task.task_id.replace('CQRL+Min32-', f'CQRL+Min{steps}-'),
                extra_args=task.extra_args.replace(
                    'latent_goal_steps=32', f'latent_goal_steps={steps}',
                ),
            ))
    assert len(tasks) == len({task.task_id for task in tasks}) == 210
    assert len({task.env_name for task in tasks}) == 14
    groups = Counter((t.env_name, extra_arg_map(t)[
        'agent.actor.losses.min_dist.latent_goal_steps']) for t in tasks)
    assert set(groups.values()) == {5}
    assert {t.seed for t in tasks} == {str(s) for s in range(1000, 1005)}
    return tasks


def signature(task):
    args = extra_arg_map(task)
    if '+go_qrl_model_size' in args:
        args['+cqrl_model_size'] = args.pop('+go_qrl_model_size')
    for key in OPERATIONAL:
        args.pop(key, None)
    if args.get('agent.actor.losses.min_dist.latent_goal_steps') == '0':
        args.pop('agent.actor.losses.min_dist.latent_goal_lr', None)
    return (task.mode, task.env_name, task.seed, task.steps, tuple(sorted(args.items())))


def completed_residual(status):
    if status.get('state') != 'DONE' or not status.get('output_dir'):
        return False
    output = Path(status['output_dir'])
    if not (output / 'COMPLETE').is_file() or not (output / 'config.yaml').is_file():
        return False
    config = yaml.safe_load((output / 'config.yaml').read_text())
    return config['agent']['actor']['model'].get('mlp_kind') == 'residual'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--append', action='store_true')
    args = parser.parse_args()
    requested = generate_tasks()
    before = QUEUE.read_text()
    existing = read_tasks(QUEUE)
    candidates = {t.task_id: t for t in existing}
    statuses = {}
    for path in STATUS.glob('*.status'):
        status = read_status(STATUS, path.stem)
        statuses[path.stem] = status
        if all(k in status for k in ('mode', 'env_name', 'seed', 'steps', 'extra_args')):
            candidates[path.stem] = Task(
                path.stem, status['mode'], status['env_name'], status['seed'],
                status['steps'], status.get('params', ''), status['extra_args'],
            )
    indexed = {}
    for task in candidates.values():
        indexed.setdefault(signature(task), []).append(task)
    existing_ids = {t.task_id for t in existing}
    additions, coverage, reused = [], [], []
    for task in requested:
        match = None
        reason = None
        for candidate in indexed.get(signature(task), []):
            status = statuses.get(candidate.task_id, {})
            if completed_residual(status):
                match, reason = candidate, 'completed'
                break
            if (candidate.task_id in existing_ids
                    and status.get('state', 'PENDING') in ('PENDING', 'RUNNING')
                    and '+cqrl_model_size=l' in candidate.extra_args):
                match, reason = candidate, 'queued_or_running'
                break
        if match:
            coverage.append(match)
            reused.append(dict(requested=task.task_id, reused=match.task_id, reason=reason))
        else:
            if task.task_id in candidates:
                raise ValueError(f'Unresolved existing task: {task.task_id}')
            additions.append(task)
            coverage.append(task)
    report = dict(total=len(requested), added=len(additions), reused=reused,
                  added_by_search_steps=dict(Counter(extra_arg_map(t)[
                      'agent.actor.losses.min_dist.latent_goal_steps'] for t in additions)))
    print(json.dumps(report, indent=2))
    if args.append:
        # Preserve every existing queue row and abort if another writer changed it.
        assert QUEUE.read_text() == before, 'Queue changed during audit; rerun.'
        atomic_write(OUTPUT, '# NQRL search ablation: 14 environments x 3 K x 5 seeds.\n'
                     '# Reuses completed equivalent runs; PointMaze uses 200400 steps.\n'
                     + '\n'.join(task_line(t) for t in coverage) + '\n')
        if additions:
            atomic_write(QUEUE, before.rstrip('\n') + '\n\n'
                         '# NQRL/0, /8, /16: L, main 14 environments, seeds 1000-1004.\n'
                         + '\n'.join(task_line(t) for t in additions) + '\n')
        atomic_write(OUTPUT.with_suffix('.audit.json'), json.dumps(report, indent=2) + '\n')
        after = read_tasks(QUEUE)
        assert len(after) == len(existing) + len(additions)
        assert len({t.task_id for t in after}) == len(after)
        assert {t.task_id for t in coverage} <= {t.task_id for t in after}


if __name__ == '__main__':
    main()

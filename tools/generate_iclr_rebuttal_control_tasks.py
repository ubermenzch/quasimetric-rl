#!/usr/bin/env python3
"""Add reviewer controls with their separately selected task sets.

Copy the actual QRL and AQRL/0 queue rows so evaluation, interaction budgets,
sampling, losses and optimization follow their corresponding controls.
Only QRL's backbone or AQRL/0's MSE weight changes, respectively.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from quasimetric_rl.model_size import qrl_agent_parameter_count
from tools.analyze_nqrl_paper_results import ENVIRONMENTS, artifact_dir
from tools.generate_extended_online_tasks import ENVIRONMENTS as DIMENSIONS
from tools.generate_iclr_main_table_tasks import ENVIRONMENTS as MAIN_DIMENSIONS
from tools.generate_nqrl_search_ablation_tasks import generate_tasks as search_tasks
from tools.generate_selected_success_l_min32_tasks import (
    atomic_write, extra_arg_map, task_line,
)
from tools.run_qrl_queue import Task, read_status, read_tasks

QUEUE = ROOT / 'runs/qrl_queue/tasks.tsv'
STATUS = ROOT / 'runs/qrl_queue/status'
OUTPUT = ROOT / 'configs/iclr_rebuttal_controls_6env_3seed.tsv'
QRL_OUTPUT = ROOT / 'configs/iclr_rebuttal_qrl_aqrl_backbone_6env_3seed.tsv'
NO_MSE_OUTPUT = ROOT / 'configs/iclr_rebuttal_aqrl0_no_mse_6env_3seed.tsv'
SEEDS = tuple(range(1000, 1003))
SELECTED_ENV_NAMES = (
    'AntMaze_UMaze-v5', 'FetchPickAndPlace', 'maze2d-large',
    'PandaFlip-v3', 'Pusher-v4', 'reacher_hard',
)
NO_MSE_ENV_NAMES = (
    'AntNavigate-v4', 'FetchPush', 'PandaPickAndPlaceJoints-v3',
    'PandaPushJoints-v3', 'PandaSlide-v3', 'Pusher-v4',
)
# The backbone comparison has already been extended to all main-table tasks.
# Restricting the no-MSE control must never prune that independent comparison.
BACKBONE_ALLOWED_ENV_NAMES = frozenset(
    {'AntMaze UMaze-v5': 'AntMaze_UMaze-v5',
     'PointMaze Medium-v3': 'PointMaze_Medium-v3'}.get(env.paper_name, env.paper_name)
    for env in ENVIRONMENTS
)
CONTROL_PREFIXES = (
    'iclr_rebuttal_v1_QRL-AQRLBackbone-L_',
    'iclr_rebuttal_v1_AQRL0-NoMSE-L_',
)
GROUP_SIZE = len(SELECTED_ENV_NAMES) * len(SEEDS)
QRL_PRESET = 'l_aqrl_backbone'
MSE_KEY = 'agent.quasimetric_critic.losses.latent_dynamics.mse_weight'
STEP_KEY = 'agent.actor.losses.min_dist.latent_goal_steps'

# Runtime output/checkpoint options do not change the training definition.
OPERATIONAL = {
    'resume_if_possible', 'keep_only_latest_checkpoint',
    'keep_only_best_and_final_checkpoints', 'save_replay_buffer',
    'save_final_replay_buffer', 'delete_checkpoints_after_completion',
    'logs_only', 'queue.logs_only', 'collect_rebuttal_diagnostics',
    'timing.enabled',
}


def replace_args(task: Task, changes: dict[str, str]) -> str:
    values = extra_arg_map(task)
    values.update(changes)
    return ' '.join(f'{key}={value}' for key, value in values.items())


def generate_tasks(existing: list[Task]) -> tuple[list[Task], list[dict]]:
    by_id = {task.task_id: task for task in existing}
    if len(by_id) != len(existing):
        raise ValueError('The task list contains duplicate task IDs')
    dimensions = {env.name: env for env in (*DIMENSIONS, *MAIN_DIMENSIONS)}
    no_search = [task for task in search_tasks()
                 if extra_arg_map(task)[STEP_KEY] == '0'
                 and task.env_name in NO_MSE_ENV_NAMES
                 and int(task.seed) in SEEDS]
    tasks, provenance = [], []
    for environment in ENVIRONMENTS:
        environment_name = {'AntMaze UMaze-v5': 'AntMaze_UMaze-v5'}.get(
            environment.paper_name, environment.paper_name,
        )
        if environment_name not in SELECTED_ENV_NAMES:
            continue
        for seed in SEEDS:
            source_id = artifact_dir(environment, 'QRL', seed).name
            source = by_id[source_id]
            values = extra_arg_map(source)
            if values.get('+qrl_model_size') != 'l':
                raise ValueError(f'Unexpected source preset: {source_id}')
            dim = dimensions[source.env_name]
            count = qrl_agent_parameter_count(dim.state_dim, dim.action_dim, QRL_PRESET)
            task = replace(
                source,
                task_id=(f'iclr_rebuttal_v1_QRL-AQRLBackbone-L_'
                         f'{source.steps}steps_{environment.slug}_online_s{seed}'),
                params=str(count),
                extra_args=replace_args(source, {'+qrl_model_size': QRL_PRESET}),
            )
            tasks.append(task)
            provenance.append(dict(task_id=task.task_id, variant='qrl_aqrl_backbone',
                                   source_task_id=source_id,
                                   override_changes={'+qrl_model_size': QRL_PRESET}))
    for template in no_search:
        source = by_id[template.task_id]
        values = extra_arg_map(source)
        if (values.get('+cqrl_model_size') != 'l'
                or values.get(STEP_KEY) != '0'
                or values.get('agent.quasimetric_critic.losses.latent_dynamics.distance')
                != 'iqe_mse' or float(values[MSE_KEY]) != 1):
            raise ValueError(f'Unexpected AQRL/0 control: {source.task_id}')
        task = replace(
            source,
            task_id=(f'iclr_rebuttal_v1_AQRL0-NoMSE-L_{source.steps}steps_'
                     f'{dimensions[source.env_name].slug}_online_s{source.seed}'),
            extra_args=replace_args(source, {MSE_KEY: '0.0'}),
        )
        tasks.append(task)
        provenance.append(dict(task_id=task.task_id, variant='aqrl0_no_mse',
                               source_task_id=source.task_id,
                               override_changes={MSE_KEY: '0.0'}))
    assert len(tasks) == len({task.task_id for task in tasks}) == 2 * GROUP_SIZE
    assert {task.env_name for task in tasks[:GROUP_SIZE]} == set(SELECTED_ENV_NAMES)
    assert {task.env_name for task in tasks[GROUP_SIZE:]} == set(NO_MSE_ENV_NAMES)
    assert set(Counter((row['variant'], task.env_name)
                       for task, row in zip(tasks, provenance)).values()) == {len(SEEDS)}
    return tasks, provenance


def signature(task: Task) -> tuple:
    values = extra_arg_map(task)
    if '+go_qrl_model_size' in values:
        values['+cqrl_model_size'] = values.pop('+go_qrl_model_size')
    if '+base_model_size' in values:
        values['+qrl_model_size'] = values.pop('+base_model_size')
    for key in OPERATIONAL:
        values.pop(key, None)
    return task.mode, task.env_name, task.seed, task.steps, tuple(sorted(values.items()))


def restrict_pending_controls(before: str, existing: list[Task], status_dir: Path):
    removed = [task for task in existing
               if (task.task_id.startswith(CONTROL_PREFIXES[0])
                   and task.env_name not in BACKBONE_ALLOWED_ENV_NAMES)
               or (task.task_id.startswith(CONTROL_PREFIXES[1])
                   and task.env_name not in NO_MSE_ENV_NAMES)]
    for task in removed:
        state = read_status(status_dir, task.task_id).get('state', 'PENDING')
        if state != 'PENDING':
            raise ValueError(f'Cannot remove a control that is not pending: {task.task_id}: {state}')
    removed_ids = {task.task_id for task in removed}
    lines = [line for line in before.splitlines(keepends=True)
             if line.split('\t', 1)[0] not in removed_ids]
    return ''.join(lines), removed


def append_tasks(queue: Path, before: str, additions: list[Task], *,
                 retained_text: str | None = None) -> None:
    if queue.read_text() != before:
        raise RuntimeError('Task list changed during the audit; rerun before appending')
    updated = before if retained_text is None else retained_text
    if additions:
        updated = (updated.rstrip('\n') + '\n\n'
                   '# ICLR rebuttal: QRL + AQRL backbone and AQRL/0 without MSE; '
                   'separate six-task sets; seeds 1000-1002 per control.\n'
                   + '\n'.join(task_line(task) for task in additions) + '\n')
    if updated != before:
        atomic_write(queue, updated)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--append', action='store_true')
    parser.add_argument('--prune-unselected', action='store_true',
                        help='Remove pending controls outside their variant-specific task sets')
    parser.add_argument('--queue', type=Path, default=QUEUE)
    parser.add_argument('--status-dir', type=Path, default=STATUS)
    args = parser.parse_args()
    before = args.queue.read_text()
    original_existing = read_tasks(args.queue)
    retained_text, removed = (restrict_pending_controls(before, original_existing, args.status_dir)
                              if args.prune_unselected else (before, []))
    removed_ids = {task.task_id for task in removed}
    existing = [task for task in original_existing if task.task_id not in removed_ids]
    requested, provenance = generate_tasks(existing)
    from tools.set_pending_logs_only import logs_only_task, preserve_storage_policy
    requested = preserve_storage_policy(requested, existing)
    existing_ids = {task.task_id for task in existing}
    requested = [task if task.task_id in existing_ids else logs_only_task(task)
                 for task in requested]
    candidates = {task.task_id: task for task in existing}
    statuses = {}
    status_paths = sorted(args.status_dir.glob('*.status'))
    for path in status_paths:
        status = read_status(args.status_dir, path.stem)
        statuses[path.stem] = status
        if all(key in status for key in ('mode', 'env_name', 'seed', 'steps', 'extra_args')):
            candidates.setdefault(path.stem, Task(
                path.stem, status['mode'], status['env_name'], status['seed'],
                status['steps'], status.get('params', ''), status['extra_args']))
    indexed = {}
    for task in candidates.values():
        indexed.setdefault(signature(task), []).append(task)
    existing_ids = {row.task_id for row in existing}
    additions, coverage, reused = [], [], []
    for task in requested:
        match = None
        for candidate in indexed.get(signature(task), []):
            status = statuses.get(candidate.task_id, {})
            state = status.get('state', 'PENDING')
            if state == 'DONE':
                output = Path(status.get('output_dir', ''))
                if (output / 'COMPLETE').is_file() and (output / 'config.yaml').is_file():
                    match = candidate
                    break
            elif candidate.task_id in existing_ids and state in ('PENDING', 'RUNNING'):
                match = candidate
                break
        if match:
            coverage.append(match)
            reused.append(dict(requested=task.task_id, reused=match.task_id,
                               state=statuses.get(match.task_id, {}).get('state', 'PENDING')))
        else:
            if task.task_id in candidates:
                raise ValueError(f'Existing task needs investigation: {task.task_id}')
            additions.append(task)
            coverage.append(task)
    report = dict(requested=len(requested), added=len(additions), reused=reused,
                  queue=str(args.queue.relative_to(ROOT)) if args.queue.is_relative_to(ROOT) else str(args.queue),
                  queue_sha256_before=hashlib.sha256(before.encode()).hexdigest(),
                  existing_tasks=len(original_existing),
                  removed_pending=len(removed), removed_task_ids=sorted(removed_ids),
                  retained_tasks=len(existing), status_records_scanned=len(status_paths),
                  seeds=list(SEEDS), variants=dict(Counter(row['variant'] for row in provenance)),
                  environments=dict(Counter(task.env_name for task in requested)),
                  provenance=provenance)
    if args.append:
        if removed:
            backup = args.queue.with_name('tasks.tsv.before_iclr_rebuttal_subset_20261107')
            if not backup.exists():
                backup.write_text(before)
        append_tasks(args.queue, before, additions, retained_text=retained_text)
        after = read_tasks(args.queue)
        assert after[:len(existing)] == existing
        assert len(after) == len(existing) + len(additions)
        assert len({task.task_id for task in after}) == len(after)
        report['queue_sha256_after'] = hashlib.sha256(args.queue.read_bytes()).hexdigest()
        report['task_indices'] = {task.task_id: i for i, task in enumerate(after, 1)
                                  if task.task_id in {row.task_id for row in coverage}}
        header = '# task_id\tmode\tenv_name\tseed\tsteps\tparams\textra_args\n'
        atomic_write(OUTPUT, header + '\n'.join(task_line(task) for task in coverage) + '\n')
        atomic_write(QRL_OUTPUT, header + '\n'.join(task_line(task) for task in coverage[:GROUP_SIZE]) + '\n')
        atomic_write(NO_MSE_OUTPUT, header + '\n'.join(task_line(task) for task in coverage[GROUP_SIZE:]) + '\n')
        atomic_write(OUTPUT.with_suffix('.audit.json'), json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ('provenance', 'task_indices')}, indent=2))


if __name__ == '__main__':
    main()

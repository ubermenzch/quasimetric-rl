#!/usr/bin/env python3
"""Report all state-based main-task obligations and existing validation results."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.generate_baseline_paper_validation_tasks import generate_tasks
from tools.run_qrl_queue import read_status, read_tasks


def last_metrics(path):
    if not path.exists():
        return None
    last = None
    with path.open() as handle:
        for line in handle:
            if line.strip():
                last = json.loads(line)
    if last is None:
        return None
    keys = (
        'env_steps', 'num_episodes', 'succ_rate', 'final_success_rate',
        'initial_goal_distance', 'final_goal_distance', 'minimum_goal_distance',
        'reference_final_goal_distance', 'reference_final_success_rate',
    )
    return {key: last[key] for key in keys if key in last}


def audit(queue_path, status_dir):
    queued = {task.task_id: task for task in read_tasks(queue_path)}
    protocols = json.loads((ROOT / 'configs/baseline_paper_validation_protocols.json').read_text())['tasks']
    rows = []
    for task in generate_tasks(suite='paper-main'):
        args = dict(token.split('=', 1) for token in shlex.split(task.extra_args))
        status = read_status(status_dir, task.task_id)
        state = status.get('state', 'PENDING') if task.task_id in queued else 'NOT_QUEUED'
        if task.task_id in queued and queued[task.task_id] != task:
            state = 'CONFIG_MISMATCH'
        row = dict(
            algorithm=args['agent.algorithm'], environment=task.env_name,
            task_id=task.task_id, seed=int(task.seed), steps=int(task.steps),
            status=state,
        )
        if task.env_name in protocols:
            row['protocol_fidelity'] = protocols[task.env_name]['fidelity']
            row['protocol_reference'] = 'configs/baseline_paper_validation_protocols.json'
        if status.get('output_dir'):
            directory = Path(status['output_dir'])
            row['output_dir'] = str(directory)
            row['last_validation'] = last_metrics(directory / 'eval.log')
            row['selected_test'] = last_metrics(directory / 'test.log')
            row['complete_marker'] = (directory / 'COMPLETE').exists()
            if state == 'DONE' and (not row['complete_marker'] or row['selected_test'] is None):
                row['status'] = 'INCOMPLETE_ARTIFACTS'
        rows.append(row)
    remaining_path = ROOT / 'configs/baseline_paper_validation_remaining.json'
    remaining = json.loads(remaining_path.read_text())
    rows.extend(dict(row, seed=remaining['training_seed']) for row in remaining['tasks'])
    pairs = [(row['algorithm'], row['environment']) for row in rows]
    if len(pairs) != len(set(pairs)):
        raise ValueError('Repeated algorithm/environment in paper coverage')
    planned = [row for row in rows if row['status'] != 'SKIPPED_USER_REQUEST']
    return dict(
        scope='Four baseline papers: online state-observation main task sets',
        counts=dict(Counter(row['status'] for row in rows)),
        total_tasks=len(rows),
        planned_tasks=len(planned),
        skipped_tasks=len(rows) - len(planned),
        reconstructed_protocol_tasks=sum(
            row.get('protocol_fidelity') == 'PAPER_PROTOCOL_RECONSTRUCTION' for row in planned),
        all_complete=all(row['status'] == 'DONE' for row in rows),
        all_planned_complete=all(row['status'] == 'DONE' for row in planned),
        tasks=rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, default=ROOT / 'runs/qrl_queue/tasks.tsv')
    parser.add_argument('--status-dir', type=Path, default=ROOT / 'runs/qrl_queue/status')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'analysis/baseline_paper_validation/coverage.json')
    args = parser.parse_args()
    report = audit(args.queue, args.status_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'tasks'}))
    print(f'Report: {args.output}')


if __name__ == '__main__':
    main()

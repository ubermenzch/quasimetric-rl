#!/usr/bin/env python3
"""Stop only the six superseded M-residual Max4 tasks, preserving all files."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.run_qrl_queue import (
    log_tail, read_status, read_tasks, status_extra_fields, write_status,
)


def checked_process(pid, task_id):
    path = Path(f'/proc/{pid}')
    try:
        argv = (path / 'cmdline').read_bytes().decode().split('\0')
        stat = (path / 'stat').read_text().rpartition(') ')[2].split()
        start = stat[19]
        owner = path.stat().st_uid
    except FileNotFoundError:
        return None
    if stat[0] == 'Z':
        return None
    if (owner != os.geteuid() or 'online.main' not in argv
            or f'output_folder={task_id}' not in argv
            or 'agent.actor.losses.min_dist.latent_goal_mode=max' not in argv
            or '+cqrl_model_size=m_residual' not in argv):
        raise RuntimeError(f'PID {pid} does not match the expected M-residual Max4 task {task_id}')
    return start


def interruption_evidence(task_id, status):
    """Accept only the final interruption record from this task's current attempt."""
    path = Path(status.get('log_file', ''))
    if not path.name.startswith(task_id + '_') or not path.is_file():
        return None
    lines = log_tail(path, max_bytes=16384).strip().splitlines()
    if not lines:
        return None
    match = re.fullmatch(
        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[WARNING\]\s+'
        r'Interrupted: received signal (?:2|15) at env_steps=\d+ optim_steps=\d+\. '
        r'Discarding progress after the latest committed checkpoint\.', lines[-1],
    )
    if not match:
        return None
    try:
        started = datetime.strptime(status['started_at'], '%Y-%m-%d %H:%M:%S')
        interrupted = datetime.strptime(match[1], '%Y-%m-%d %H:%M:%S')
    except (KeyError, ValueError):
        return None
    if interrupted < started:
        return None
    return {'log_file': str(path), 'record': lines[-1], 'interrupted_at': match[1]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='Send SIGTERM after checking every task identity.')
    args = parser.parse_args(argv)
    tasks = read_tasks(ROOT / 'configs/figure2_m_residual_antnavigate_6seed.tsv')
    assert len(tasks) == 6
    active_ids = {t.task_id for t in read_tasks(ROOT / 'runs/qrl_queue/tasks.tsv')}
    if any(t.task_id in active_ids for t in tasks):
        raise RuntimeError('Remove the superseded Max4 tasks from the active queue before stopping them.')
    targets = []
    unavailable = []
    already_exited = []
    statuses = {}
    evidence = {}
    for task in tasks:
        status = read_status(ROOT / 'runs/qrl_queue/status', task.task_id)
        statuses[task.task_id] = status
        pid = int(status.get('pid') or status.get('previous_pid') or 0)
        start = checked_process(pid, task.task_id) if pid > 1 else None
        if start is None:
            proof = interruption_evidence(task.task_id, status)
            if proof or status.get('state') in {'DONE', 'FAILED', 'PAUSED'}:
                already_exited.append(task.task_id)
                if proof:
                    evidence[task.task_id] = proof
                print(f'ALREADY_EXITED pid={pid} {task.task_id}')
            else:
                print(f'ABSENT_UNVERIFIED pid={pid} state={status.get("state")} {task.task_id}')
                unavailable.append(task.task_id)
            continue
        targets.append((pid, start, task.task_id))
        print(f'{"STOP" if args.apply else "WOULD_STOP"} pid={pid} {task.task_id}')
    if unavailable:
        print('Some PIDs are absent without current-attempt exit evidence. This can mean stale '
              'status or limited process visibility; no signals or status changes were made.', file=sys.stderr)
        return 2
    if not args.apply:
        return 0
    signaled = []
    for pid, start, task_id in targets:
        current_start = checked_process(pid, task_id)
        if current_start is None:
            continue
        if current_start != start:
            raise RuntimeError(f'PID {pid} changed before SIGTERM')
        try:
            os.kill(pid, signal.SIGTERM)
            signaled.append(task_id)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 30
    remaining = targets
    while remaining and time.monotonic() < deadline:
        remaining = [(pid, start, task_id) for pid, start, task_id in remaining
                     if checked_process(pid, task_id) == start]
        if remaining:
            time.sleep(.25)
    stopped_ids = set(already_exited) | {task_id for _, _, task_id in targets}
    stopped_ids -= {task_id for _, _, task_id in remaining}
    for task in tasks:
        previous = statuses[task.task_id]
        if task.task_id not in stopped_ids or previous.get('state') in {'DONE', 'FAILED', 'PAUSED'}:
            continue
        if read_status(ROOT / 'runs/qrl_queue/status', task.task_id) != previous:
            raise RuntimeError(f'Status changed concurrently for {task.task_id}; refusing to overwrite it')
        extra = status_extra_fields(previous)
        extra.update({'previous_pid': previous.get('pid') or previous.get('previous_pid', ''),
                      'previous_gpu': previous.get('gpu') or previous.get('previous_gpu', ''),
                      'previous_state': previous.get('state', ''),
                      'note': 'superseded_by_fresh_min4_and_min32',
                      'error': 'superseded_max4_stopped'})
        if task.task_id in evidence:
            extra['finished_at'] = evidence[task.task_id]['interrupted_at']
        write_status(ROOT / 'runs/qrl_queue/status', task, 'PAUSED', extra)
    report = {'signaled_tasks': signaled,
              'already_exited_tasks': already_exited,
              'interruption_evidence': evidence,
              'still_running': [task_id for _, _, task_id in remaining],
              'results_preserved': True}
    path = ROOT / 'runs/m_residual_min_variants_submission/stop_result.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 1 if remaining else 0


if __name__ == '__main__':
    raise SystemExit(main())

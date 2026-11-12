#!/usr/bin/env python3
"""Stop and reset the nine Move32 diagnostics, leaving them last in the queue."""
from __future__ import annotations

import argparse
import os
import shutil
import signal
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.qrl_queue_control import lock_is_held, start, stop
from tools.run_qrl_queue import (
    QueueLock, cfg, parse_config, read_status, read_tasks, resolve_path,
    task_fingerprint, task_manifest_fingerprint, task_output_dir, write_status,
)

PREFIX = 'iclr_rebuttal_v1_AQRL-Displacement-L-InnerLR1_'
ENVIRONMENTS = {'PointMaze_Medium-v3', 'FetchPickAndPlace', 'FetchPush'}
SEEDS = {'1000', '1001', '1002'}
DEFER_HEADER = b'# Deferred Move32 displacement diagnostics.'


def select_tasks(tasks):
    selected = [task for task in tasks if task.task_id.startswith(PREFIX)]
    pairs = {(task.env_name, task.seed) for task in selected}
    expected = {(env, seed) for env in ENVIRONMENTS for seed in SEEDS}
    if len(selected) != 9 or pairs != expected:
        raise RuntimeError('Expected exactly the nine specified Move32 tasks')
    return selected


def _read_process(pid):
    proc = Path('/proc') / str(pid)
    try:
        owner = proc.stat().st_uid
        cmd = (proc / 'cmdline').read_bytes().split(b'\0')
        stat = (proc / 'stat').read_text()
    except FileNotFoundError:
        return None
    state = stat[stat.rfind(')') + 2:][:1]
    if state in {'Z', 'X'}:
        return None
    return owner, cmd


def matching_processes(task_ids):
    wanted = {f'output_folder={task_id}'.encode(): task_id for task_id in task_ids}
    matches = {}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != os.geteuid():
                continue
            process = _read_process(entry.name)
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if process is None:
            continue
        _, cmd = process
        for arg in cmd:
            if arg in wanted:
                task_id = wanted[arg]
                if task_id in matches:
                    raise RuntimeError(f'Multiple processes for task: {task_id}')
                matches[task_id] = int(entry.name)
    return matches


def verify_running_statuses(records, matches):
    for task_id, (status, _, _) in records.items():
        pid = status.get('pid', '')
        if (status.get('state') == 'RUNNING' and pid.isdigit()
                and (Path('/proc') / pid).exists()
                and matches.get(task_id) != int(pid)):
            raise RuntimeError(f'Running task process could not be verified: {task_id}, PID {pid}')


def validate_records(tasks, status_dir, config):
    records = {}
    for task in tasks:
        status = read_status(status_dir, task.task_id)
        if status.get('task_fingerprint') != task_fingerprint(task):
            raise RuntimeError(f'Status does not match task: {task.task_id}')
        if status.get('state') == 'DONE':
            raise RuntimeError(f'Task already finished; preserving its result: {task.task_id}')
        expected = task_output_dir(config, task)
        output = Path(status.get('output_dir') or expected)
        if output.is_symlink() or output.resolve() != expected.resolve():
            raise RuntimeError(f'Unexpected output path: {output}')
        if output.exists() and task_manifest_fingerprint(output) != task_fingerprint(task):
            raise RuntimeError(f'Output manifest does not match task: {output}')
        log = Path(status['log_file']) if status.get('log_file') else None
        if log is not None:
            log_root = resolve_path(cfg(config, 'LOG_DIR', 'logs/qrl_queue'))
            if (log.is_symlink() or log.parent.resolve() != log_root.resolve()
                    or not log.name.startswith(task.task_id + '_') or log.suffix != '.log'):
                raise RuntimeError(f'Unexpected log path: {log}')
        records[task.task_id] = (status, output, log)
    return records


def terminate_tasks(matches, grace=15.0):
    for task_id, pid in matches.items():
        current = _read_process(pid)
        if current is None:
            continue
        if current[0] != os.geteuid() or f'output_folder={task_id}'.encode() not in current[1]:
            raise RuntimeError(f'Process no longer matches task: {task_id}, PID {pid}')
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline and matching_processes(matches):
        time.sleep(.2)
    remaining = matching_processes(matches)
    for task_id, pid in remaining.items():
        if matches.get(task_id) != pid:
            raise RuntimeError(f'Unexpected replacement process for {task_id}')
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and matching_processes(matches):
        time.sleep(.2)
    if matching_processes(matches):
        raise RuntimeError('Some Move32 processes did not stop; outputs were not deleted')


def move_to_end(queue, task_ids):
    original = queue.read_bytes()
    kept, deferred = [], []
    for line in original.splitlines(keepends=True):
        task_id = line.split(b'\t', 1)[0].decode(errors='replace').strip()
        if task_id in task_ids:
            deferred.append(line if line.endswith(b'\n') else line + b'\n')
        elif line.strip() == DEFER_HEADER:
            continue
        else:
            kept.append(line)
    if len(deferred) != len(task_ids):
        raise RuntimeError('Task list changed; refusing to update it')
    updated = b''.join(kept).rstrip(b'\n') + b'\n\n' + DEFER_HEADER + b'\n' + b''.join(deferred)
    if updated == original:
        return
    fd, tmp = tempfile.mkstemp(prefix='.queue-defer-', dir=queue.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(updated)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, queue.stat().st_mode & 0o777)
        if queue.read_bytes() != original:
            raise RuntimeError('Queue changed concurrently')
        os.replace(tmp, queue)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    config_path = ROOT / 'configs/qrl_queue.env'
    config = parse_config(config_path)
    queue = resolve_path(cfg(config, 'TASKS_FILE', 'runs/qrl_queue/tasks.tsv'))
    status_dir = resolve_path(cfg(config, 'STATUS_DIR', 'runs/qrl_queue/status'))
    lock_path = resolve_path(os.environ.get('QRL_QUEUE_LOCK_FILE') or cfg(config, 'LOCK_FILE', 'runs/qrl_queue/qrl_queue.lock'))
    tasks = select_tasks(read_tasks(queue))
    records = validate_records(tasks, status_dir, config)
    matches = matching_processes({task.task_id for task in tasks})
    verify_running_statuses(records, matches)
    print(f'Move32 tasks: {len(tasks)}; running processes: {len(matches)}', flush=True)
    if args.dry_run:
        return
    for task_id, pid in matches.items():
        try:
            os.kill(pid, 0)
        except OSError as exc:
            raise RuntimeError('Run this command in a normal server shell with access to the training processes') from exc
    for _, output, log in records.values():
        for path in (output.parent, log.parent if log else None):
            if path is not None and not os.access(path, os.W_OK | os.X_OK):
                raise RuntimeError(f'Cannot write to {path}; run in the normal server shell')

    scheduler_was_running = lock_is_held(lock_path)
    held_lock = QueueLock(lock_path)
    acquired = False
    try:
        if scheduler_was_running:
            stop(lock_path, with_jobs=False)
        if not held_lock.acquire():
            raise RuntimeError('Another scheduler acquired the queue lock')
        acquired = True
        tasks = select_tasks(read_tasks(queue))
        records = validate_records(tasks, status_dir, config)
        matches = matching_processes({task.task_id for task in tasks})
        verify_running_statuses(records, matches)
        terminate_tasks(matches)
        if matching_processes({task.task_id for task in tasks}):
            raise RuntimeError('A Move32 process remains active; outputs were not deleted')
        for task in tasks:
            status, output, log = records[task.task_id]
            if output.exists():
                if output.is_symlink() or task_manifest_fingerprint(output) != task_fingerprint(task):
                    raise RuntimeError(f'Output changed during cleanup: {output}')
                shutil.rmtree(output)
            if log is not None:
                log.unlink(missing_ok=True)
        move_to_end(queue, {task.task_id for task in tasks})
        for task in tasks:
            status, _, _ = records[task.task_id]
            (status_dir / f'{task.task_id}.status').unlink(missing_ok=True)
            write_status(status_dir, task, 'PENDING', {'submitted_at': status.get('submitted_at', '')})
        print('Stopped and reset 9 Move32 tasks; they are last in the queue.', flush=True)
    finally:
        if acquired:
            held_lock.release()
        if scheduler_was_running and not lock_is_held(lock_path):
            start(config_path, [])


if __name__ == '__main__':
    main()

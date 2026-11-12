#!/usr/bin/env python3
"""Stage or activate the paper-QRL goal-marker replacement jobs."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_qrl_queue as q, qrl_queue_control as control
from tools.analyze_nqrl_paper_results import ENVIRONMENTS, artifact_dir
from tools.generate_extended_online_tasks import ENVIRONMENTS as EXTENDED_DIMS
from tools.generate_iclr_main_table_tasks import ENVIRONMENTS as MAIN_DIMS
from tools.generate_selected_success_l_min32_tasks import extra_arg_map, task_line
from quasimetric_rl.model_size import qrl_agent_parameter_count

CANDIDATE_FILE = ROOT / 'configs/iclr_rebuttal_qrl_goalmarker_paper_5env_3seed.tsv'
PREFIX = 'iclr_rebuttal_v1_QRL-GoalMarker-L_'
OLD_ENVS = {'AntNavigate-v4', 'FetchPush', 'PandaPushJoints-v3'}
NEW_ENVS = {
    'PandaPickAndPlace-v3': 'PandaPickAndPlace-v3',
    'PointMaze_Medium-v3': 'PointMaze Medium-v3',
    'PandaSlide-v3': 'PandaSlide-v3',
    'Pusher-v4': 'Pusher-v4',
    'PandaFlip-v3': 'PandaFlip-v3',
}
SEEDS = {'1000', '1001', '1002'}
STAGED_REASON = 'awaiting_plain_qrl_marker_switch'
OPERATIONAL = {
    'interaction.qrl_goal_marker', 'interaction.num_eval_episodes',
    'interaction.num_test_episodes', 'save_replay_buffer',
    'save_final_replay_buffer', 'resume_if_possible', 'queue.logs_only',
    'logs_only', 'collect_rebuttal_diagnostics', 'timing.enabled',
}


def paths(config):
    task_path = q.resolve_path(q.cfg(config, 'TASKS_FILE', 'runs/qrl_queue/tasks.tsv'))
    status_dir = q.resolve_path(q.cfg(config, 'STATUS_DIR', 'runs/qrl_queue/status'))
    results = q.resolve_path(q.cfg(config, 'RESULTS_ROOT', '../qrl-assets/results/queue'))
    lock_path = q.resolve_path(os.environ.get('QRL_QUEUE_LOCK_FILE') or
                               q.cfg(config, 'LOCK_FILE', 'runs/qrl_queue/qrl_queue.lock'))
    return task_path, status_dir, results, lock_path


def validate(tasks, candidates):
    by_id = {t.task_id: t for t in tasks}
    if len(by_id) != len(tasks):
        raise RuntimeError('Queue contains duplicate task IDs.')
    expected = {(env, seed) for env in NEW_ENVS for seed in SEEDS}
    if len(candidates) != 15 or {(t.env_name, t.seed) for t in candidates} != expected:
        raise RuntimeError('Replacement file must contain exactly three seeds for each of the five environments.')
    if len({t.task_id for t in candidates}) != 15:
        raise RuntimeError('Replacement file contains duplicate task IDs.')
    paper = {e.paper_name: e for e in ENVIRONMENTS}
    dims = {e.name: e for e in (*EXTENDED_DIMS, *MAIN_DIMS)}
    for t in candidates:
        if not t.task_id.startswith(PREFIX):
            raise RuntimeError(f'Unexpected replacement task ID: {t.task_id}')
        source_id = artifact_dir(paper[NEW_ENVS[t.env_name]], 'QRL', int(t.seed)).name
        source = by_id.get(source_id)
        if source is None or (t.mode, t.env_name, t.seed, t.steps) != (
                source.mode, source.env_name, source.seed, source.steps):
            raise RuntimeError(f'Missing or mismatched paper-QRL source: {source_id}')
        a, b = extra_arg_map(source), extra_arg_map(t)
        if {k: v for k, v in a.items() if k not in OPERATIONAL} != {
                k: v for k, v in b.items() if k not in OPERATIONAL}:
            raise RuntimeError(f'Algorithm settings differ from paper QRL: {t.task_id}')
        if b.get('+qrl_model_size') != 'l' or b.get('interaction.qrl_goal_marker') != 'true':
            raise RuntimeError(f'Expected plain QRL-L and goal marker: {t.task_id}')
        for key in ('interaction.num_eval_episodes', 'interaction.num_test_episodes'):
            expected_count = '200' if a.get(key) in {'500', '1000'} else a.get(key)
            if b.get(key) != expected_count:
                raise RuntimeError(f'Unexpected evaluation count for {t.task_id}: {key}')
        for key, expected_value in {
                'save_replay_buffer': 'false', 'save_final_replay_buffer': 'false',
                'resume_if_possible': 'false', 'queue.logs_only': 'true',
                'logs_only': 'true'}.items():
            if b.get(key) != expected_value:
                raise RuntimeError(f'Unexpected storage setting {key} for {t.task_id}')
        d = dims[t.env_name]
        if int(t.params) != qrl_agent_parameter_count(d.state_dim + 1, d.action_dim, 'l'):
            raise RuntimeError(f'Unexpected parameter count: {t.task_id}')
        if t.task_id in by_id and q.task_fingerprint(by_id[t.task_id]) != q.task_fingerprint(t):
            raise RuntimeError(f'Existing task differs from replacement: {t.task_id}')
    old = [t for t in tasks if t.task_id.startswith(PREFIX)
           and t.env_name in OLD_ENVS and t.seed in SEEDS]
    if len(old) != 9 or {(t.env_name, t.seed) for t in old} != {
            (env, seed) for env in OLD_ENVS for seed in SEEDS}:
        raise RuntimeError('Expected exactly the nine previous GoalMarker tasks.')
    return old


def status_extras(status, reason):
    fixed = {'state', 'updated_at', 'task_id', 'task_fingerprint', 'mode', 'env_name',
             'seed', 'steps', 'params', 'extra_args', 'pid', 'gpu', 'finished_at'}
    extra = {key: value for key, value in status.items() if key not in fixed}
    extra['previous_pid'] = status.get('pid') or status.get('previous_pid', '')
    extra['previous_gpu'] = status.get('gpu') or status.get('previous_gpu', '')
    if reason:
        extra['error'] = reason
    else:
        extra.pop('error', None)
    return extra


def reorder_first_pending(task_path, status_dir, candidates):
    before = task_path.read_bytes()
    lines = before.decode('utf-8').splitlines(keepends=True)
    tasks = q.read_tasks(task_path)
    new_ids = {t.task_id for t in candidates}
    ids = [t.task_id for t in tasks]
    if len(ids) != len(set(ids)):
        raise RuntimeError('Queue contains duplicate task IDs.')
    def line_id(line):
        return line.rstrip('\r\n').split('\t', 1)[0]
    kept = [line for line in lines if line_id(line) not in new_ids]
    first_pending = next((t.task_id for t in tasks if t.task_id not in new_ids and
                          q.read_status(status_dir, t.task_id).get('state', 'PENDING') == 'PENDING'), None)
    insert_at = (next(i for i, line in enumerate(kept) if line_id(line) == first_pending)
                 if first_pending else len(kept))
    if insert_at == len(kept) and kept and not kept[-1].endswith(('\n', '\r')):
        kept[-1] += '\n'
    block = [task_line(t) + '\n' for t in candidates]
    content = ''.join(kept[:insert_at] + block + kept[insert_at:]).encode('utf-8')
    if task_path.read_bytes() != before:
        raise RuntimeError('Task list changed while preparing the switch.')
    tmp = task_path.with_name(task_path.name + '.goalmarker_switch.tmp')
    tmp.write_bytes(content)
    os.chmod(tmp, task_path.stat().st_mode & 0o777)
    os.replace(tmp, task_path)


def stage_locked(task_path, status_dir, results, candidates):
    tasks = q.read_tasks(task_path)
    old = validate(tasks, candidates)
    existing = {t.task_id for t in tasks}
    for t in candidates:
        status = q.read_status(status_dir, t.task_id)
        if status:
            if status.get('task_fingerprint') != q.task_fingerprint(t):
                raise RuntimeError(f'Conflicting status for {t.task_id}')
            if status.get('state') not in {'PAUSED', 'PENDING'}:
                raise RuntimeError(f'Replacement already started or finished: {t.task_id}')
            if status.get('state') == 'PAUSED' and status.get('error') != STAGED_REASON:
                raise RuntimeError(f'Replacement was paused for another reason: {t.task_id}')
        if (results / t.task_id).exists():
            raise RuntimeError(f'Replacement output directory already exists: {t.task_id}')
    for t in old:
        status = q.read_status(status_dir, t.task_id)
        if status and status.get('task_fingerprint') != q.task_fingerprint(t):
            raise RuntimeError(f'Conflicting old task status for {t.task_id}')
    for t in candidates:
        status = q.read_status(status_dir, t.task_id)
        q.write_status(status_dir, t, 'PAUSED', status_extras(status, STAGED_REASON))
    for t in old:
        status = q.read_status(status_dir, t.task_id)
        if status.get('state', 'PENDING') == 'PENDING':
            q.write_status(status_dir, t, 'PAUSED', status_extras(status, 'user_requested_pause_preserve_logs'))
    reorder_first_pending(task_path, status_dir, candidates)
    return old


def matching_processes(old):
    task_ids = {t.task_id for t in old}
    found = []
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            args = [a for a in (proc / 'cmdline').read_bytes().split(b'\0') if a]
            if b'-m' not in args or b'online.main' not in args:
                continue
            folder = next((a[len(b'output_folder='):].decode() for a in args
                           if a.startswith(b'output_folder=')), '')
            if folder in task_ids:
                stat = (proc / 'stat').read_text()
                if stat[stat.rfind(')') + 2] not in 'ZX':
                    found.append((int(proc.name), folder))
        except (OSError, UnicodeDecodeError, IndexError):
            continue
    return found


def ensure_recorded_pids(old, status_dir, processes):
    found = set(processes)
    for t in old:
        status = q.read_status(status_dir, t.task_id)
        if status.get('state') != 'RUNNING':
            continue
        pid_text = status.get('pid', '')
        if not pid_text.isdigit():
            raise RuntimeError(f'RUNNING task has no valid PID: {t.task_id}')
        pid = int(pid_text)
        proc = Path(f'/proc/{pid}')
        if not proc.exists() or (pid, t.task_id) in found:
            continue
        try:
            stat = (proc / 'stat').read_text()
            if stat[stat.rfind(')') + 2] in 'ZX':
                continue
        except (OSError, IndexError):
            pass
        raise RuntimeError(f'RUNNING task PID does not match its recorded command: {t.task_id}, PID {pid}')


def ensure_signalable(processes):
    for pid, task_id in processes:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            if Path(f'/proc/{pid}/cmdline').exists():
                raise RuntimeError(f'Cannot signal server PID {pid} ({task_id}) from this environment; run on the server terminal.')
        except PermissionError as exc:
            raise RuntimeError(f'No permission to signal PID {pid} ({task_id}).') from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/qrl_queue.env')
    parser.add_argument('--stage-only', action='store_true')
    args = parser.parse_args()
    config_path = q.resolve_path(args.config)
    config = q.parse_config(config_path)
    task_path, status_dir, results, lock_path = paths(config)
    candidates = q.read_tasks(CANDIDATE_FILE)
    old = validate(q.read_tasks(task_path), candidates)
    if not args.stage_only:
        processes = matching_processes(old)
        ensure_recorded_pids(old, status_dir, processes)
        ensure_signalable(processes)
        if control.lock_is_held(lock_path):
            control.stop(lock_path, with_jobs=False)
    lock = q.QueueLock(lock_path)
    if not lock.acquire():
        raise RuntimeError('Queue scheduler is active; stage-only made no changes.')
    success = False
    try:
        old = stage_locked(task_path, status_dir, results, candidates)
        if args.stage_only:
            print('Staged 15 replacement tasks as PAUSED and paused old pending tasks; running jobs were not signaled.')
            success = True
            return 0
        processes = matching_processes(old)
        ensure_recorded_pids(old, status_dir, processes)
        ensure_signalable(processes)
        for pid, task_id in processes:
            try:
                os.kill(pid, signal.SIGTERM)
                print(f'SIGTERM {task_id} pid={pid}', flush=True)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and matching_processes(old):
            time.sleep(0.25)
        remaining = matching_processes(old)
        if remaining:
            raise RuntimeError(f'Old GoalMarker jobs did not exit after 60 seconds: {remaining}. Scheduler remains stopped.')
        for t in old:
            status = q.read_status(status_dir, t.task_id)
            state = status.get('state', 'PENDING')
            if state in {'DONE', 'FAILED', 'EARLY_STOPPED'}:
                continue
            output = Path(status.get('output_dir') or results / t.task_id)
            if q.output_finished_for_task(output, t):
                q.write_status(status_dir, t, 'DONE', status_extras(status, None))
            else:
                q.write_status(status_dir, t, 'PAUSED', status_extras(status, 'user_requested_pause_preserve_logs'))
        for t in candidates:
            q.write_status(status_dir, t, 'PENDING', {'note': 'priority_paper_qrl_goalmarker'})
        reorder_first_pending(task_path, status_dir, candidates)
        pending = [t.task_id for t in q.read_tasks(task_path)
                   if q.read_status(status_dir, t.task_id).get('state', 'PENDING') == 'PENDING']
        if pending[:len(candidates)] != [t.task_id for t in candidates]:
            raise RuntimeError('Replacement tasks are not first among pending tasks; scheduler remains stopped.')
        success = True
    finally:
        lock.release()
        if not success and not args.stage_only:
            print('Scheduler remains stopped; existing logs were not deleted.', file=sys.stderr)
    control.start(config_path, [])
    print('Paused previous GoalMarker jobs; activated 15 paper-QRL GoalMarker tasks at the front of the pending queue.')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1)

#!/usr/bin/env python3
"""Disable checkpoint persistence for unstarted queue tasks only."""
from dataclasses import replace
import argparse
import fcntl
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.generate_selected_success_l_min32_tasks import atomic_write, extra_arg_map, task_line
from tools.run_qrl_queue import (
    TASK_LOGS_ONLY_ARG, parse_config, read_status, read_tasks, task_output_dir,
    task_uses_native_qrl, write_status,
)


def logs_only_task(task):
    values = extra_arg_map(task)
    values[TASK_LOGS_ONLY_ARG] = 'true'
    if not task_uses_native_qrl(task):
        values.update(logs_only='true', resume_if_possible='false',
                      save_replay_buffer='false', save_final_replay_buffer='false',
                      collect_rebuttal_diagnostics='true', **{'timing.enabled': 'true'})
        if task.task_id.startswith('iclr_rebuttal_v1_AQRLRepo-QRLOriginalConfig_'):
            values['interaction.original_qrl_evaluation'] = 'true'
        prefix = 'agent.actor.losses.min_dist.'
        if values.get(prefix + 'latent_goal_log_displacement') == 'true':
            values[prefix + 'latent_goal_probe_replay'] = 'true'
            values[prefix + 'latent_goal_probe_interval'] = '1000'
            values[prefix + 'latent_goal_reference_size'] = '2048'
            values[prefix + 'latent_goal_probe_queries'] = '32'
        if int(values.get(prefix + 'replay_goal_candidates', '0')) > 0:
            values[prefix + 'replay_goal_log_diagnostics'] = 'true'
    return replace(task, extra_args=' '.join(f'{key}={value}' for key, value in values.items()))


def preserve_storage_policy(requested, existing):
    """Retain saved storage and evaluation settings when regenerating tasks."""
    by_id = {task.task_id: task for task in existing}
    preserved = []
    for task in requested:
        current = by_id.get(task.task_id)
        if current is None:
            preserved.append(task)
            continue
        actual = extra_arg_map(current)
        if actual.get(TASK_LOGS_ONLY_ARG) == 'true':
            task = logs_only_task(task)
        values = extra_arg_map(task)
        for key in ('interaction.num_eval_episodes', 'interaction.num_test_episodes'):
            if key in actual:
                values[key] = actual[key]
        preserved.append(replace(task, extra_args=' '.join(
            f'{key}={value}' for key, value in values.items())))
    return preserved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    config = parse_config(ROOT / 'configs/qrl_queue.env')
    queue = ROOT / 'runs/qrl_queue/tasks.tsv'
    status_dir = ROOT / 'runs/qrl_queue/status'
    with (ROOT / 'runs/qrl_queue/qrl_queue.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = queue.read_text()
        tasks = read_tasks(queue)
        updates = {}
        for task in tasks:
            state = read_status(status_dir, task.task_id).get('state', 'PENDING')
            if state != 'PENDING':
                continue
            output = task_output_dir(config, task)
            if output.exists() and any(output.iterdir()):
                raise RuntimeError(f'Pending task already has output; inspect before changing it: {task.task_id}')
            updated = logs_only_task(task)
            if updated != task:
                updates[task.task_id] = updated
        print(f'Pending tasks to update: {len(updates)}; scheduler is stopped.')
        if not args.apply or not updates:
            return
        backup = ROOT / 'runs/qrl_queue/before_logs_only_20261108'
        backup.mkdir(exist_ok=True)
        if not (backup / 'tasks.tsv').exists():
            (backup / 'tasks.tsv').write_text(before)
        for task_id in updates:
            path = status_dir / f'{task_id}.status'
            if path.exists() and not (backup / path.name).exists():
                shutil.copy2(path, backup / path.name)
        text = ''.join(task_line(updates[line.split('\t', 1)[0]]) + '\n'
                       if line.split('\t', 1)[0] in updates else line
                       for line in before.splitlines(keepends=True))
        if queue.read_text() != before:
            raise RuntimeError('Queue changed during the update')
        atomic_write(queue, text)
        for task in updates.values():
            write_status(status_dir, task, 'PENDING')
        # Keep the corresponding task manifests consistent with the live queue.
        for manifest in (ROOT / 'configs').glob('iclr_rebuttal*.tsv'):
            original = manifest.read_text()
            modified = ''.join(task_line(updates[line.split('\t', 1)[0]]) + '\n'
                               if line.split('\t', 1)[0] in updates else line
                               for line in original.splitlines(keepends=True))
            if modified != original:
                if not (backup / manifest.name).exists():
                    (backup / manifest.name).write_text(original)
                atomic_write(manifest, modified)
        print(f'Updated {len(updates)} tasks; no scheduler was started.')


if __name__ == '__main__':
    main()

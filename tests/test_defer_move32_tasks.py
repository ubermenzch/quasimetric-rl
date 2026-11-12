import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import defer_move32_tasks as defer
from tools.run_qrl_queue import Task, read_status, read_tasks, task_fingerprint, write_status


class DeferMove32Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.results = self.root / 'results'
        self.logs = self.root / 'logs'
        self.status = self.root / 'status'
        for path in (self.results, self.logs, self.status):
            path.mkdir()
        self.queue = self.root / 'tasks.tsv'
        self.lock = self.root / 'queue.lock'
        self.tasks = [
            Task(f'{defer.PREFIX}{env}_{seed}', 'online', env, seed, '200000', '21m', 'logs_only=true')
            for env in sorted(defer.ENVIRONMENTS) for seed in sorted(defer.SEEDS)
        ]
        self.other = Task('unrelated_task', 'online', 'other', '1', '10', '1m', '')
        self.queue.write_text(''.join('\t'.join((t.task_id, t.mode, t.env_name, t.seed,
                                                 t.steps, t.params, t.extra_args)) + '\n'
                                      for t in self.tasks + [self.other]))
        self.matches = {}
        self.fingerprints = {}
        for index, task in enumerate(self.tasks):
            output = self.results / task.task_id
            output.mkdir()
            (output / 'training.log').write_text('progress')
            log = self.logs / (task.task_id + '_20261110-101000.log')
            log.write_text('progress')
            self.matches[task.task_id] = 900000 + index
            self.fingerprints[output.name] = task_fingerprint(task)
            write_status(self.status, task, 'RUNNING', {
                'pid': str(900000 + index), 'gpu': '1', 'output_dir': str(output),
                'log_file': str(log), 'submitted_at': '2026-11-08 00:00:00',
            })
        self.unrelated = self.results / 'unrelated_data'
        self.unrelated.mkdir()
        (self.unrelated / 'keep').write_text('keep')
        self.config = {'TASKS_FILE': str(self.queue), 'STATUS_DIR': str(self.status),
                       'LOCK_FILE': str(self.lock), 'LOG_DIR': str(self.logs)}

    def tearDown(self):
        self.temp.cleanup()

    def test_cleanup_resets_only_target_tasks_and_restarts_scheduler(self):
        fake_lock = mock.Mock()
        fake_lock.acquire.return_value = True
        with (mock.patch.object(defer, 'parse_config', return_value=self.config),
              mock.patch.object(defer, 'task_output_dir', side_effect=lambda _, t: self.results / t.task_id),
              mock.patch.object(defer, 'task_manifest_fingerprint', side_effect=lambda p: self.fingerprints[p.name]),
              mock.patch.object(defer, 'matching_processes', side_effect=[self.matches, self.matches, {}]),
              mock.patch.object(defer, 'terminate_tasks') as terminate,
              mock.patch.object(defer.os, 'kill'),
              mock.patch.object(defer.os, 'access', return_value=True),
              mock.patch.object(defer, 'QueueLock', return_value=fake_lock),
              mock.patch.object(defer, 'lock_is_held', side_effect=[True, False]),
              mock.patch.object(defer, 'stop') as stop,
              mock.patch.object(defer, 'start') as start,
              mock.patch.object(sys, 'argv', ['defer_move32_tasks.py'])):
            defer.main()
        stop.assert_called_once_with(self.lock, with_jobs=False)
        start.assert_called_once()
        terminate.assert_called_once_with(self.matches)
        fake_lock.release.assert_called_once()
        self.assertEqual([t.task_id for t in read_tasks(self.queue)][-9:], [t.task_id for t in self.tasks])
        self.assertTrue((self.unrelated / 'keep').exists())
        for task in self.tasks:
            self.assertFalse((self.results / task.task_id).exists())
            self.assertEqual(list(self.logs.glob(task.task_id + '_*.log')), [])
            status = read_status(self.status, task.task_id)
            self.assertEqual(status['state'], 'PENDING')
            self.assertEqual(status['pid'], '')
            self.assertEqual(status['submitted_at'], '2026-11-08 00:00:00')

    def test_rejects_unexpected_move32_task_set(self):
        with self.assertRaises(RuntimeError):
            defer.select_tasks(self.tasks[:-1])

    def test_signal_access_failure_stops_before_scheduler_or_data_changes(self):
        before = self.queue.read_bytes()
        with (mock.patch.object(defer, 'parse_config', return_value=self.config),
              mock.patch.object(defer, 'task_output_dir', side_effect=lambda _, t: self.results / t.task_id),
              mock.patch.object(defer, 'task_manifest_fingerprint', side_effect=lambda p: self.fingerprints[p.name]),
              mock.patch.object(defer, 'matching_processes', return_value=self.matches),
              mock.patch.object(defer.os, 'kill', side_effect=ProcessLookupError()),
              mock.patch.object(defer, 'stop') as stop,
              mock.patch.object(sys, 'argv', ['defer_move32_tasks.py']),
              self.assertRaises(RuntimeError)):
            defer.main()
        stop.assert_not_called()
        self.assertEqual(self.queue.read_bytes(), before)
        self.assertTrue(all((self.results / task.task_id).exists() for task in self.tasks))
        self.assertTrue(all(read_status(self.status, task.task_id)['state'] == 'RUNNING' for task in self.tasks))

    def test_rejects_status_for_unrelated_output(self):
        task = self.tasks[0]
        write_status(self.status, task, 'RUNNING', {
            'pid': '1', 'gpu': '1', 'output_dir': str(self.unrelated),
        })
        with (mock.patch.object(defer, 'task_output_dir', return_value=self.results / task.task_id),
              self.assertRaises(RuntimeError)):
            defer.validate_records([task], self.status, self.config)
        self.assertTrue((self.unrelated / 'keep').exists())


if __name__ == '__main__':
    unittest.main()

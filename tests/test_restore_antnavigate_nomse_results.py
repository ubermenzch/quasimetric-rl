import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools import restore_antnavigate_nomse_results as restore
from tools import run_qrl_queue as queue


class RestoreCompletedRunsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.old = self.root / 'old'
        self.new = self.root / 'new'
        self.status = self.root / 'status'
        self.old.mkdir()
        self.new.mkdir()
        self.tasks = []
        for index, (seed, success) in enumerate(restore.EXPECTED.items(), start=1):
            task = queue.Task(restore.PREFIX + seed, 'online', 'AntNavigate-v4',
                              seed, '500000', '21.7m', 'logs_only=true')
            self.tasks.append(task)
            source = self.old / task.task_id
            destination = self.new / task.task_id
            queue.write_task_manifest(source, task)
            queue.write_task_manifest(destination, task)
            test = {'split': 'test', 'completed_training_env_steps': 500000,
                    'num_episodes': 200, 'seed': 1500, 'succ_rate': success}
            (source / 'test.log').write_text(json.dumps(test) + '\n')
            (source / 'best_checkpoint.json').write_text(json.dumps({
                'validation': {}, 'test': test, 'selected_model': 'in_memory',
            }))
            (source / 'eval.log').write_text('original validation\n')
            (source / 'output.log').write_text('original completed run\n')
            (source / 'COMPLETE').touch()
            (destination / 'partial.log').write_text('duplicate training\n')
            queue.write_status(self.status, task, 'RUNNING', {
                'pid': str(50000 + index), 'gpu': str(index),
                'output_dir': str(destination), 'log_file': 'duplicate.log',
            })
        self.tasks_file = self.root / 'tasks.tsv'
        self.tasks_file.write_text(''.join(
            '\t'.join((t.task_id, t.mode, t.env_name, t.seed, t.steps,
                       t.params, t.extra_args)) + '\n' for t in self.tasks
        ))
        self.lock = self.root / 'queue.lock'
        self.config = self.root / 'queue.env'
        self.config.write_text(
            f'TASKS_FILE="{self.tasks_file}"\n'
            f'STATUS_DIR="{self.status}"\n'
            f'RESULTS_ROOT="{self.new}"\n'
            f'LOCK_FILE="{self.lock}"\n'
        )
        self.env_patch = mock.patch.dict(os.environ, {
            'TASKS_FILE': str(self.tasks_file), 'STATUS_DIR': str(self.status),
            'RESULTS_ROOT': str(self.new), 'LOCK_FILE': str(self.lock),
            'QRL_QUEUE_LOCK_FILE': str(self.lock),
        })
        self.env_patch.start()

    def tearDown(self):
        self.env_patch.stop()
        self.temp.cleanup()

    def test_restores_all_results_and_marks_done(self):
        with (mock.patch.object(restore.control, 'lock_is_held', return_value=True),
              mock.patch.object(restore.control, 'stop') as stop,
              mock.patch.object(restore.control, 'start', return_value=0) as start,
              mock.patch.object(restore, 'process_matches'),
              mock.patch.object(restore, 'terminate_duplicates') as terminate):
            restore.restore(self.config, self.old)
        stop.assert_called_once()
        start.assert_called_once()
        terminate.assert_called_once()
        self.assertEqual(len(terminate.call_args.args[0]), 3)
        for task in self.tasks:
            source, destination = self.old / task.task_id, self.new / task.task_id
            self.assertEqual(restore.tree_hashes(source), restore.tree_hashes(destination))
            self.assertFalse((destination / 'partial.log').exists())
            status = queue.read_status(self.status, task.task_id)
            self.assertEqual(status['state'], 'DONE')
            self.assertEqual(status['output_dir'], str(destination))
            self.assertEqual(status['completion_evidence'], 'COMPLETE')
            self.assertEqual(status['task_fingerprint'], queue.task_fingerprint(task))
        self.assertEqual(list(self.new.glob('.restore-ant-nomse-*')), [])

    def test_incomplete_original_aborts_before_stopping_scheduler(self):
        (self.old / self.tasks[1].task_id / 'COMPLETE').unlink()
        with (mock.patch.object(restore.control, 'stop') as stop,
              mock.patch.object(restore, 'process_matches'),
              self.assertRaises(ValueError)):
            restore.restore(self.config, self.old)
        stop.assert_not_called()
        self.assertTrue((self.new / self.tasks[0].task_id / 'partial.log').exists())

    def test_mismatched_process_is_never_signalled(self):
        with (mock.patch.object(restore.os, 'kill') as kill,
              self.assertRaises(ValueError)):
            restore.process_matches(str(os.getpid()), 'not-this-process')
        kill.assert_not_called()

    def test_only_the_three_checked_processes_are_signalled(self):
        with mock.patch.object(restore, 'process_matches'):
            plans, _, _ = restore.make_plan(queue.parse_config(self.config), self.old)
        with (mock.patch.object(restore, 'process_matches'),
              mock.patch.object(restore, '_is_running', return_value=False),
              mock.patch.object(restore.os, 'kill') as kill):
            restore.terminate_duplicates(plans)
        self.assertEqual(
            kill.call_args_list,
            [mock.call(50001, restore.signal.SIGTERM),
             mock.call(50002, restore.signal.SIGTERM),
             mock.call(50003, restore.signal.SIGTERM)],
        )

    def test_failed_stop_keeps_partial_results_and_scheduler_stopped(self):
        with (mock.patch.object(restore.control, 'lock_is_held', return_value=True),
              mock.patch.object(restore.control, 'stop'),
              mock.patch.object(restore.control, 'start') as start,
              mock.patch.object(restore, 'process_matches'),
              mock.patch.object(restore, '_is_running', return_value=True),
              mock.patch.object(restore, 'terminate_duplicates', side_effect=RuntimeError('failed')),
              self.assertRaises(RuntimeError)):
            restore.restore(self.config, self.old)
        start.assert_not_called()
        for task in self.tasks:
            self.assertTrue((self.new / task.task_id / 'partial.log').exists())
            self.assertEqual(queue.read_status(self.status, task.task_id)['state'], 'RUNNING')

    def test_install_failure_rolls_back_all_partial_results(self):
        real_replace = os.replace
        failed_task = self.tasks[1]

        def replace(source, destination):
            source, destination = Path(source), Path(destination)
            if source.name == failed_task.task_id and source.parent.name.startswith('.restore-ant-nomse-'):
                raise OSError('simulated install failure')
            return real_replace(source, destination)

        with (mock.patch.object(restore.control, 'lock_is_held', return_value=True),
              mock.patch.object(restore.control, 'stop'),
              mock.patch.object(restore.control, 'start') as start,
              mock.patch.object(restore, 'process_matches'),
              mock.patch.object(restore, '_is_running', return_value=True),
              mock.patch.object(restore, 'terminate_duplicates'),
              mock.patch.object(restore.os, 'replace', side_effect=replace),
              self.assertRaises(OSError)):
            restore.restore(self.config, self.old)
        start.assert_not_called()
        for task in self.tasks:
            self.assertTrue((self.new / task.task_id / 'partial.log').exists())
            self.assertFalse((self.new / task.task_id / 'COMPLETE').exists())
            self.assertEqual(queue.read_status(self.status, task.task_id)['state'], 'RUNNING')

    def test_scheduler_start_failure_keeps_restored_results(self):
        with (mock.patch.object(restore.control, 'lock_is_held', return_value=True),
              mock.patch.object(restore.control, 'stop'),
              mock.patch.object(restore.control, 'start', return_value=1),
              mock.patch.object(restore, 'process_matches'),
              mock.patch.object(restore, 'terminate_duplicates'),
              self.assertRaises(RuntimeError)):
            restore.restore(self.config, self.old)
        for task in self.tasks:
            self.assertTrue((self.new / task.task_id / 'COMPLETE').is_file())
            self.assertEqual(queue.read_status(self.status, task.task_id)['state'], 'DONE')


if __name__ == '__main__':
    unittest.main()

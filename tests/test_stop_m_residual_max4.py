from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from tools.run_qrl_queue import Task, read_status, write_status
from tools.stop_m_residual_max4 import checked_process, interruption_evidence, main


class StopSupersededMax4Test(unittest.TestCase):
    def test_process_identity_requires_correct_task_model_and_max_mode(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'stat').write_text('42 (python) ' + ' '.join(['S'] + ['0'] * 18 + ['123456']))
            argv = ['python', '-m', 'online.main', 'output_folder=expected',
                    '+cqrl_model_size=m_residual', 'agent.actor.losses.min_dist.latent_goal_mode=max']
            with mock.patch('tools.stop_m_residual_max4.Path', return_value=root):
                (root / 'cmdline').write_bytes('\0'.join(argv).encode())
                self.assertEqual(checked_process(42, 'expected'), '123456')
                with self.assertRaises(RuntimeError):
                    checked_process(42, 'another_task')
                for old, new in [
                    ('latent_goal_mode=max', 'latent_goal_mode=min'),
                    ('+cqrl_model_size=m_residual', '+cqrl_model_size=l'),
                    ('online.main', 'another.main'),
                ]:
                    (root / 'cmdline').write_bytes('\0'.join(argv).replace(old, new).encode())
                    with self.assertRaises(RuntimeError):
                        checked_process(42, 'expected')

    def test_absent_process_is_reported_without_signaling(self):
        with TemporaryDirectory() as directory:
            with mock.patch('tools.stop_m_residual_max4.Path', return_value=Path(directory) / 'absent'):
                self.assertIsNone(checked_process(42, 'expected'))

    def test_exited_zombie_is_not_signaled(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'stat').write_text('42 (python) ' + ' '.join(['Z'] + ['0'] * 18 + ['123456']))
            (root / 'cmdline').write_bytes(b'')
            with mock.patch('tools.stop_m_residual_max4.Path', return_value=root):
                self.assertIsNone(checked_process(42, 'expected'))

    def make_stopped_tasks(self, root):
        tasks = [Task(f'max4_task_{i}', 'online', 'AntNavigate-v4', str(i), '500000')
                 for i in range(6)]
        for i, task in enumerate(tasks):
            log = root / f'{task.task_id}_20260910-192434.log'
            log.write_text('2026-09-10 19:48:00 [WARNING]  Interrupted: received signal 15 '
                           'at env_steps=21000 optim_steps=11524. Discarding progress after '
                           'the latest committed checkpoint.\n')
            write_status(root / 'runs/qrl_queue/status', task, 'RUNNING', {
                'pid': str(100 + i), 'gpu': str(i), 'started_at': '2026-09-10 19:24:34',
                'log_file': str(log), 'output_dir': str(root / task.task_id),
            })
        return tasks

    def test_already_interrupted_tasks_reconcile_without_signals_and_repeat_safely(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            tasks = self.make_stopped_tasks(root)
            # The scheduler can already have requeued an interrupted job, moving
            # its PID/GPU into history before the stop helper runs.
            task = tasks[0]
            status_dir = root / 'runs/qrl_queue/status'
            status = read_status(status_dir, task.task_id)
            write_status(status_dir, task, 'PENDING', {
                'previous_pid': status['pid'], 'previous_gpu': status['gpu'],
                'log_file': status['log_file'], 'started_at': status['started_at'],
            })
            with mock.patch('tools.stop_m_residual_max4.ROOT', root), \
                 mock.patch('tools.stop_m_residual_max4.read_tasks', side_effect=[tasks, [], tasks, []]), \
                 mock.patch('tools.stop_m_residual_max4.checked_process', return_value=None), \
                 mock.patch('tools.stop_m_residual_max4.os.kill') as kill, \
                 mock.patch('builtins.print'):
                self.assertEqual(main(['--apply']), 0)
                self.assertEqual(main(['--apply']), 0)
                kill.assert_not_called()
            for task in tasks:
                status = read_status(root / 'runs/qrl_queue/status', task.task_id)
                self.assertEqual(status['state'], 'PAUSED')
                self.assertEqual(status['pid'], '')
                self.assertTrue(status['previous_pid'])
                self.assertEqual(status['finished_at'], '2026-09-10 19:48:00')
                self.assertTrue(Path(status['log_file']).is_file())

    def test_missing_evidence_does_not_signal_other_targets_or_change_status(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            tasks = self.make_stopped_tasks(root)
            status_dir = root / 'runs/qrl_queue/status'
            Path(read_status(status_dir, tasks[1].task_id)['log_file']).unlink()
            before = [read_status(status_dir, t.task_id) for t in tasks]
            with mock.patch('tools.stop_m_residual_max4.ROOT', root), \
                 mock.patch('tools.stop_m_residual_max4.read_tasks', side_effect=[tasks, []]), \
                 mock.patch('tools.stop_m_residual_max4.checked_process', side_effect=['123', None, None, None, None, None]), \
                 mock.patch('tools.stop_m_residual_max4.os.kill') as kill, \
                 mock.patch('builtins.print'):
                self.assertEqual(main(['--apply']), 2)
                kill.assert_not_called()
            self.assertEqual(before, [read_status(status_dir, t.task_id) for t in tasks])

    def test_interruption_before_current_attempt_or_followed_by_activity_is_not_exit_evidence(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            task = self.make_stopped_tasks(root)[0]
            status = read_status(root / 'runs/qrl_queue/status', task.task_id)
            self.assertIsNotNone(interruption_evidence(task.task_id, status))
            self.assertIsNone(interruption_evidence('another_task', status))
            later_attempt = dict(status, started_at='2026-09-10 19:49:00')
            self.assertIsNone(interruption_evidence(task.task_id, later_attempt))
            with Path(status['log_file']).open('a') as log:
                log.write('2026-09-10 19:49:00 [INFO ] Resuming training\n')
            self.assertIsNone(interruption_evidence(task.task_id, status))


if __name__ == '__main__':
    unittest.main()

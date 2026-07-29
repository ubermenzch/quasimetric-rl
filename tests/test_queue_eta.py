import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.watch_qrl_queue import Progress
from tools.watch_qrl_queue import RecentProgressHistory
from tools.watch_qrl_queue import Task
from tools.watch_qrl_queue import current_run_start_progress
from tools.watch_qrl_queue import estimated_eta_hours


class QueueEtaTest(unittest.TestCase):
    def setUp(self):
        self.task = Task(
            task_id="online_task",
            mode="online",
            env_name="FetchPush",
            seed="1000",
            steps="100000",
        )

    def test_periodic_checkpoints_do_not_reset_new_run_start(self):
        with TemporaryDirectory() as directory:
            log_file = Path(directory) / "online.log"
            log_file.write_text(
                "Checkpointed to checkpoint_env00035000_opt00025000.pth\n"
                "Checkpointed to checkpoint_env00040000_opt00030000.pth\n"
            )
            status = {"state": "RUNNING", "log_file": str(log_file)}

            self.assertEqual(current_run_start_progress(self.task, status), 0.0)
            eta = estimated_eta_hours(
                self.task,
                status,
                Progress(41.0, "4.1e+04/1e+05", 41000.0, 100000.0),
                run_elapsed_h=0.39,
                history_elapsed_h=0.39,
            )
            self.assertAlmostEqual(eta, 0.39 * 59000.0 / 41000.0)

    def test_resume_marker_remains_start_after_periodic_checkpoints(self):
        with TemporaryDirectory() as directory:
            log_file = Path(directory) / "online.log"
            log_file.write_text(
                "Fast forward to env_steps=35000 optim_steps=25000\n"
                "Checkpointed to checkpoint_env00040000_opt00030000.pth\n"
            )
            status = {"state": "RUNNING", "log_file": str(log_file)}

            self.assertEqual(current_run_start_progress(self.task, status), 35000.0)

    def test_recent_window_speed_takes_priority_over_run_average(self):
        eta = estimated_eta_hours(
            self.task,
            {"state": "RUNNING"},
            Progress(70.0, "7e+04/1e+05", 70000.0, 100000.0),
            run_elapsed_h=10.0,
            history_elapsed_h=10.0,
            recent_samples=[
                (1000.0, 40000.0),
                (2800.0, 70000.0),
            ],
        )

        self.assertAlmostEqual(eta, 0.5)

    def test_recent_history_persists_and_resets_for_new_attempt(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "eta.json"
            history = RecentProgressHistory(path)
            history.observe("task", "pid-1", 1000.0, 100.0, 1800.0)
            history.observe("task", "pid-1", 2000.0, 200.0, 1800.0)
            history.save()

            restored = RecentProgressHistory(path)
            samples = restored.observe(
                "task", "pid-1", 3000.0, 300.0, 1800.0
            )
            self.assertEqual(samples[0], (100.0, 1000.0))
            self.assertEqual(samples[-1], (300.0, 3000.0))

            samples = restored.observe(
                "task", "pid-2", 500.0, 400.0, 1800.0
            )
            self.assertEqual(samples, [(400.0, 500.0)])

    def test_recent_history_window_is_based_on_progress(self):
        with TemporaryDirectory() as directory:
            history = RecentProgressHistory(Path(directory) / "eta.json")
            history.observe("task", "pid-1", 0.0, 100.0, 1500.0)
            history.observe("task", "pid-1", 1000.0, 200.0, 1500.0)
            history.observe("task", "pid-1", 2000.0, 300.0, 1500.0)

            samples = history.observe(
                "task", "pid-1", 3000.0, 400.0, 1500.0
            )

            self.assertEqual(samples, [
                (200.0, 1000.0),
                (300.0, 2000.0),
                (400.0, 3000.0),
            ])

    def test_current_stalled_step_contributes_to_recent_elapsed_time(self):
        with TemporaryDirectory() as directory:
            history = RecentProgressHistory(Path(directory) / "eta.json")
            history.observe("task", "pid-1", 1000.0, 100.0, 10000.0)
            history.observe("task", "pid-1", 2000.0, 200.0, 10000.0)

            samples = history.observe(
                "task", "pid-1", 2000.0, 300.0, 10000.0
            )

            self.assertEqual(samples[-1], (300.0, 2000.0))
            eta = estimated_eta_hours(
                self.task,
                {"state": "RUNNING"},
                Progress(2.0, "2e+03/1e+05", 2000.0, 100000.0),
                run_elapsed_h=1.0,
                history_elapsed_h=1.0,
                recent_samples=samples,
            )
            self.assertAlmostEqual(eta, 98.0 * 200.0 / 3600.0)


if __name__ == "__main__":
    unittest.main()

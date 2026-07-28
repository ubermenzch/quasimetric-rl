import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.watch_qrl_queue import Progress
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


if __name__ == "__main__":
    unittest.main()

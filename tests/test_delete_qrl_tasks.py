import fcntl
import io
import json
import os
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from tools.delete_qrl_tasks import main
from tools.run_qrl_queue import Task
from tools.run_qrl_queue import write_status


def task_row(task_id: str, env_name: str, seed: str) -> str:
    return "\t".join((
        task_id,
        "online",
        env_name,
        seed,
        "200000",
        "2.1m",
        "save_steps=20000",
    )) + "\n"


class DeleteQrlTasksTest(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.tasks_file = self.root / "tasks.tsv"
        self.status_dir = self.root / "status"
        self.results_root = self.root / "results"
        self.log_dir = self.root / "logs"
        self.eta_file = self.status_dir / "watch_eta_samples.json"
        self.lock_file = self.root / "queue.lock"
        self.config = self.root / "queue.env"
        self.config.write_text(
            f'TASKS_FILE="{self.tasks_file}"\n'
            f'STATUS_DIR="{self.status_dir}"\n'
            f'RESULTS_ROOT="{self.results_root}"\n'
            f'LOG_DIR="{self.log_dir}"\n'
            f'ETA_SAMPLE_FILE="{self.eta_file}"\n'
            f'LOCK_FILE="{self.lock_file}"\n'
        )
        self.tasks_file.write_text(
            "# task queue\n"
            + task_row("task_alpha", "FetchPush", "1000")
            + task_row("task_beta", "FetchSlide", "1001")
        )
        self.tasks_archive = self.root / "tasks.tsv.before_cleanup"
        self.tasks_archive.write_text(self.tasks_file.read_text())
        self.status_dir.mkdir()
        self.results_root.mkdir()
        self.log_dir.mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def args(self, *values: str) -> list[str]:
        return ["--config", str(self.config), *values]

    def make_artifacts(self, task_id: str, state: str = "PAUSED") -> None:
        task = Task(
            task_id=task_id,
            mode="online",
            env_name="FetchPush" if task_id == "task_alpha" else "FetchSlide",
            seed="1000" if task_id == "task_alpha" else "1001",
            steps="200000",
            params="2.1m",
            extra_args="save_steps=20000",
        )
        output_dir = self.results_root / task_id
        output_dir.mkdir()
        (output_dir / "checkpoint.pth").write_text("checkpoint")
        log_file = self.log_dir / f"{task_id}_20260731-120000.log"
        log_file.write_text("log")
        status_extra = {"output_dir": str(output_dir), "log_file": str(log_file)}
        if state == "RUNNING":
            status_extra["pid"] = str(os.getpid())
        write_status(self.status_dir, task, state, status_extra)

    def test_exact_id_permanently_deletes_all_owned_artifacts(self):
        self.make_artifacts("task_alpha")
        self.make_artifacts("task_beta", state="DONE")
        (self.status_dir / "task_alpha.status.tmp").write_text("partial status")
        self.eta_file.write_text(json.dumps({
            "version": 1,
            "tasks": {"task_alpha": {"samples": []}, "task_beta": {"samples": []}},
        }))

        output = io.StringIO()
        with redirect_stdout(output):
            result = main(self.args(
                "--task-id", "task_alpha", "--yes", "--expect", "1"
            ))

        self.assertEqual(result, 0)
        self.assertIn("Permanently deleted 1 task(s).", output.getvalue())
        self.assertNotIn("task_alpha", self.tasks_file.read_text())
        self.assertIn("task_beta", self.tasks_file.read_text())
        self.assertNotIn("task_alpha", self.tasks_archive.read_text())
        self.assertIn("task_beta", self.tasks_archive.read_text())
        self.assertFalse((self.status_dir / "task_alpha.status").exists())
        self.assertFalse((self.status_dir / "task_alpha.status.tmp").exists())
        self.assertFalse((self.results_root / "task_alpha").exists())
        self.assertFalse(any(self.log_dir.glob("task_alpha_*.log")))
        self.assertNotIn("task_alpha", json.loads(self.eta_file.read_text())["tasks"])
        self.assertTrue((self.status_dir / "task_beta.status").exists())
        self.assertTrue((self.results_root / "task_beta").exists())

    def test_metadata_filters_preview_without_deleting(self):
        self.make_artifacts("task_alpha")
        self.make_artifacts("task_beta")

        output = io.StringIO()
        with redirect_stdout(output):
            result = main(self.args(
                "--task-id-glob", "task_*",
                "--state", "PAUSED",
                "--env-name", "FetchPush",
                "--seed", "1000",
            ))

        self.assertEqual(result, 0)
        self.assertIn("Matched tasks: 1", output.getvalue())
        self.assertIn("task_alpha", output.getvalue())
        self.assertNotIn("task_beta\t", output.getvalue())
        self.assertIn("task_alpha", self.tasks_file.read_text())
        self.assertTrue((self.results_root / "task_alpha").exists())

    def test_execute_requires_exact_expected_count(self):
        self.make_artifacts("task_alpha")

        error = io.StringIO()
        with redirect_stderr(error):
            result = main(self.args(
                "--task-id", "task_alpha", "--yes", "--expect", "2"
            ))

        self.assertEqual(result, 2)
        self.assertIn("selector matched 1", error.getvalue())
        self.assertIn("task_alpha", self.tasks_file.read_text())

    def test_no_selector_is_rejected(self):
        error = io.StringIO()
        with redirect_stderr(error):
            result = main(self.args())

        self.assertEqual(result, 2)
        self.assertIn("at least one stable task selector", error.getvalue())
        self.assertIn("task_alpha", self.tasks_file.read_text())

    def test_active_scheduler_lock_prevents_task_table_deletion(self):
        self.make_artifacts("task_alpha")
        self.lock_file.touch()

        with self.lock_file.open("a+") as lock_handle:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            error = io.StringIO()
            with redirect_stderr(error):
                result = main(self.args(
                    "--task-id", "task_alpha", "--yes", "--expect", "1"
                ))

        self.assertEqual(result, 2)
        self.assertIn("scheduler is active", error.getvalue())
        self.assertIn("task_alpha", self.tasks_file.read_text())
        self.assertTrue((self.results_root / "task_alpha").exists())

    def test_running_task_is_never_deleted(self):
        self.make_artifacts("task_alpha", state="RUNNING")

        error = io.StringIO()
        with redirect_stderr(error):
            result = main(self.args(
                "--task-id", "task_alpha", "--yes", "--expect", "1"
            ))

        self.assertEqual(result, 2)
        self.assertIn("refusing to delete RUNNING", error.getvalue())
        self.assertIn("task_alpha", self.tasks_file.read_text())
        self.assertTrue((self.results_root / "task_alpha").exists())

    def test_stale_running_task_can_be_deleted(self):
        self.make_artifacts("task_alpha", state="RUNNING")

        output = io.StringIO()
        with mock.patch("tools.delete_qrl_tasks.pid_alive", return_value=False):
            with redirect_stdout(output):
                result = main(self.args(
                    "--task-id", "task_alpha",
                    "--state", "STALE",
                    "--yes", "--expect", "1",
                ))

        self.assertEqual(result, 0)
        self.assertIn("state=STALE", output.getvalue())
        self.assertIn("Permanently deleted 1 task(s).", output.getvalue())
        self.assertNotIn("task_alpha", self.tasks_file.read_text())
        self.assertFalse((self.status_dir / "task_alpha.status").exists())
        self.assertFalse((self.results_root / "task_alpha").exists())

    def test_orphaned_status_can_be_deleted_by_stable_id_file(self):
        orphan_id = "task_orphan"
        task = Task(orphan_id, "online", "FetchPush", "1002", "200000")
        write_status(self.status_dir, task, "PAUSED")
        id_file = self.root / "ids.tsv"
        id_file.write_text(task_row(orphan_id, "FetchPush", "1002"))

        result = main(self.args(
            "--task-id-file", str(id_file),
            "--orphaned",
            "--yes",
            "--expect", "1",
        ))

        self.assertEqual(result, 0)
        self.assertFalse((self.status_dir / f"{orphan_id}.status").exists())


if __name__ == "__main__":
    unittest.main()

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.prepare_online_continuations import prepare
from tools.run_qrl_queue import QueueLock
from tools.run_qrl_queue import Task
from tools.run_qrl_queue import read_status
from tools.run_qrl_queue import write_status
from tools.run_qrl_queue import write_task_manifest


class PrepareOnlineContinuationsTest(unittest.TestCase):
    def make_completed_task(self, root: Path) -> tuple[Path, Path, Path, Task]:
        task = Task(
            "continuation_test",
            "online",
            "PandaPush-v3",
            "1000",
            "100000",
            "1m",
            "resume_if_possible=true",
        )
        tasks_path = root / "tasks.tsv"
        tasks_path.write_text(
            "\t".join(
                (
                    task.task_id,
                    task.mode,
                    task.env_name,
                    task.seed,
                    task.steps,
                    task.params,
                    task.extra_args,
                )
            )
            + "\n"
        )
        status_dir = root / "status"
        results_root = root / "results"
        output_dir = results_root / task.task_id
        output_dir.mkdir(parents=True)
        (output_dir / "checkpoint_env00100000_opt00090500_final.pth").touch()
        (output_dir / "COMPLETE").touch()
        write_task_manifest(output_dir, task)
        write_status(status_dir, task, "DONE", {"completion_evidence": "COMPLETE"})

        config_path = root / "queue.env"
        config_path.write_text(
            f'TASKS_FILE="{tasks_path}"\n'
            f'STATUS_DIR="{status_dir}"\n'
            f'RESULTS_ROOT="{results_root}"\n'
            f'LOCK_FILE="{root / "queue.lock"}"\n'
        )
        return config_path, tasks_path, output_dir, task

    def test_stale_pid_metadata_never_bypasses_held_lock(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config_path, tasks_path, output_dir, task = self.make_completed_task(root)
            status_path = root / "status" / f"{task.task_id}.status"
            manifest_path = output_dir / ".qrl_task.json"
            lock = QueueLock(root / "queue.lock")
            self.assertTrue(lock.acquire())
            (root / "queue.lock").write_text("pid=999999999\nstarted_at=stale\n")
            before = (
                tasks_path.read_bytes(),
                status_path.read_bytes(),
                manifest_path.read_bytes(),
            )

            try:
                with self.assertRaisesRegex(RuntimeError, "scheduler is active"):
                    prepare(
                        config_path,
                        {task.task_id},
                        200000,
                        allow_stale_lock=True,
                    )
            finally:
                lock.release()

            self.assertEqual(
                before,
                (
                    tasks_path.read_bytes(),
                    status_path.read_bytes(),
                    manifest_path.read_bytes(),
                ),
            )

    def test_unlocked_stale_file_needs_no_bypass(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config_path, tasks_path, _, task = self.make_completed_task(root)
            (root / "queue.lock").write_text("pid=999999999\nstarted_at=stale\n")

            prepare(
                config_path,
                {task.task_id},
                200000,
                allow_stale_lock=False,
            )

            self.assertEqual(tasks_path.read_text().split("\t")[4], "200000")
            status = read_status(root / "status", task.task_id)
            self.assertEqual(status["state"], "PENDING")
            self.assertEqual(status["resume_from_env_steps"], "100000")
            self.assertEqual(status["target_env_steps"], "200000")


if __name__ == "__main__":
    unittest.main()

import io
import json
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.requeue_qrl_task import main
from tools.run_qrl_queue import Task
from tools.run_qrl_queue import read_status
from tools.run_qrl_queue import task_fingerprint
from tools.run_qrl_queue import write_status


class RequeueQrlTaskTest(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.tasks_file = self.root / "tasks.tsv"
        self.status_dir = self.root / "status"
        self.results_root = self.root / "results"
        self.config = self.root / "queue.env"
        self.task = Task("task_alpha", "online", "reacher_hard", "1002", "200000")
        self.tasks_file.write_text(
            "task_alpha\tonline\treacher_hard\t1002\t200000\t4.4m\t\n"
        )
        self.config.write_text(
            f'TASKS_FILE="{self.tasks_file}"\n'
            f'STATUS_DIR="{self.status_dir}"\n'
            f'RESULTS_ROOT="{self.results_root}"\n'
        )
        self.output_dir = self.results_root / self.task.task_id
        self.output_dir.mkdir(parents=True)
        write_status(self.status_dir, self.task, "PAUSED", {
            "error": "missing_output_task_manifest",
            "output_dir": str(self.output_dir),
        })

    def tearDown(self):
        self.temporary.cleanup()

    def args(self, *values: str) -> list[str]:
        return ["--config", str(self.config), "--task-id", self.task.task_id, *values]

    def test_repairs_manifest_temp_and_requeues(self):
        (self.output_dir / ".qrl_task.json.tmp").write_text("")

        result = main(self.args("--yes"))

        self.assertEqual(result, 0)
        status = read_status(self.status_dir, self.task.task_id)
        self.assertEqual(status["state"], "PENDING")
        manifest = json.loads((self.output_dir / ".qrl_task.json").read_text())
        self.assertEqual(manifest["fingerprint"], task_fingerprint(self.task))
        self.assertFalse((self.output_dir / ".qrl_task.json.tmp").exists())

    def test_refuses_identity_repair_when_training_output_exists(self):
        (self.output_dir / "checkpoint.pth").write_text("checkpoint")
        error = io.StringIO()

        with redirect_stderr(error):
            result = main(self.args("--yes"))

        self.assertEqual(result, 2)
        self.assertIn("training outputs exist", error.getvalue())
        self.assertEqual(
            read_status(self.status_dir, self.task.task_id)["state"], "PAUSED"
        )

    def test_refuses_running_task(self):
        write_status(self.status_dir, self.task, "RUNNING", {"pid": "123"})
        error = io.StringIO()

        with redirect_stderr(error):
            result = main(self.args("--yes"))

        self.assertEqual(result, 2)
        self.assertIn("RUNNING", error.getvalue())


if __name__ == "__main__":
    unittest.main()

import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from tools.watch_qrl_queue import Task
from tools.watch_qrl_queue import latest_eval
from tools.watch_qrl_queue import latest_number_before_marker
from tools.watch_qrl_queue import online_progress
from tools.watch_qrl_queue import render


class QueueWatcherProgressTest(unittest.TestCase):
    def test_latest_eval_reports_last_and_best_checkpoint_success(self):
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            (output_dir / "eval.log").write_text(
                '{"env_steps": 20000, "optim_steps": 10, "succ_rate": 0.4}\n'
                '{"env_steps": 40000, "optim_steps": 20, "succ_rate": 0.8}\n'
                '{"env_steps": 60000, "optim_steps": 30, "succ_rate": 0.6}\n'
            )

            summary = latest_eval(output_dir)

        self.assertEqual(summary["env_steps"], "6e+04")
        self.assertEqual(summary["succ_rate"], "0.6")
        self.assertEqual(summary["best_succ_rate"], "0.8")

    def test_latest_number_before_marker_uses_last_valid_value(self):
        with TemporaryDirectory() as directory:
            log_file = Path(directory) / "online.log"
            log_file.write_bytes(
                b"100 env steps\n"
                b"invalid env steps\n"
                b"150.5\tenv steps\r"
                b"not-a-number env steps\n"
            )

            self.assertEqual(
                latest_number_before_marker(log_file, b"env steps"),
                150.5,
            )

    def test_online_progress_does_not_regex_scan_log_tail(self):
        task = Task(
            task_id="online_task",
            mode="online",
            env_name="FetchPush",
            seed="1000",
            steps="1000",
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir = root / "output"
            output_dir.mkdir()
            log_file = root / "online.log"
            log_file.write_bytes(
                b"100 env steps\n" + b"x" * 1_000_000 + b"\r250 env steps\n"
            )

            with mock.patch(
                "tools.watch_qrl_queue.re.findall",
                side_effect=AssertionError("online log tail was regex-scanned"),
            ):
                progress = online_progress(
                    output_dir,
                    task,
                    {"log_file": str(log_file)},
                    latest={},
                )

            self.assertEqual(progress.current, 250.0)
            self.assertEqual(progress.pct, 25.0)

    def test_progress_search_falls_back_beyond_fast_tail(self):
        with TemporaryDirectory() as directory:
            log_file = Path(directory) / "online.log"
            log_file.write_bytes(b"75 env steps\n" + b"x" * 300_000)

            self.assertEqual(
                latest_number_before_marker(log_file, b"env steps"),
                75.0,
            )

    def test_render_omits_task_column(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "TASKS_FILE": str(root / "missing.tsv"),
                "STATUS_DIR": str(root / "status"),
                "RESULTS_ROOT": str(root / "results"),
                "LOG_DIR": str(root / "logs"),
            }
            output = io.StringIO()
            with mock.patch(
                "tools.watch_qrl_queue.nvidia_smi", return_value="no gpus"
            ), redirect_stdout(output):
                render(config)

        header = next(
            line for line in output.getvalue().splitlines()
            if line.split()[:4] == ["#", "state", "gpu", "pid"]
        )
        self.assertNotIn("task", header.split())
        self.assertEqual(header.split()[4], "variant")
        self.assertIn("last_succ", header.split())
        self.assertIn("best_succ", header.split())


if __name__ == "__main__":
    unittest.main()

import io
import os
import subprocess
import unittest
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from tools import watch_qrl_queue as watcher


class QueueWatcherGpuTest(unittest.TestCase):
    def test_gpu_query_batches_space_and_comma_separated_ids(self):
        result = subprocess.CompletedProcess(
            [], 0,
            '1, "GPU, Model", 100, 24000, 12\n'
            '2, RTX 4090, 200, 24000, 24\n',
            "",
        )
        with mock.patch.object(watcher.subprocess, "run", return_value=result) as run:
            output = watcher.nvidia_smi("1, 2")

        run.assert_called_once()
        self.assertIn("1,2", run.call_args.args[0])
        self.assertEqual(run.call_args.kwargs["timeout"], 5)
        self.assertEqual(
            output,
            "GPU1 GPU, Model mem=100/24000 MiB util=12%\n"
            "GPU2 RTX 4090 mem=200/24000 MiB util=24%",
        )

    def test_gpu_data_is_preserved_with_nonzero_exit_status(self):
        result = subprocess.CompletedProcess(
            [], 255, "3, RTX 4090, 1109, 24564, 13\n", ""
        )
        with mock.patch.object(watcher.subprocess, "run", return_value=result):
            output = watcher.nvidia_smi("3")

        self.assertIn("GPU3 RTX 4090 mem=1109/24564 MiB util=13%", output)
        self.assertIn("exit=255", output)
        self.assertNotIn("failed", output)

    def test_gpu_failure_includes_driver_error(self):
        result = subprocess.CompletedProcess(
            [], 255, "", "Failed to initialize NVML: Driver/library version mismatch\n"
        )
        with mock.patch.object(watcher.subprocess, "run", return_value=result):
            output = watcher.nvidia_smi("0 1")

        self.assertIn("exit=255", output)
        self.assertIn("Driver/library version mismatch", output)

    def test_gpu_failure_from_stdout_is_not_parsed_as_csv(self):
        result = subprocess.CompletedProcess(
            [], 255, "NVIDIA-SMI couldn't communicate with the NVIDIA driver.\n", ""
        )
        with mock.patch.object(watcher.subprocess, "run", return_value=result):
            output = watcher.nvidia_smi("0")

        self.assertIn("failed", output)
        self.assertIn("couldn't communicate", output)

    def test_timeout_and_missing_executable_are_reported(self):
        for error, expected in (
            (subprocess.TimeoutExpired("nvidia-smi", 5), "timed out after 5 seconds"),
            (FileNotFoundError("nvidia-smi not found"), "nvidia-smi not found"),
        ):
            with self.subTest(error=error), mock.patch.object(
                watcher.subprocess, "run", side_effect=error
            ) as run:
                self.assertIn(expected, watcher.nvidia_smi("0 1 2 3"))
                run.assert_called_once()

    def test_no_configured_gpus_skips_query(self):
        with mock.patch.object(watcher.subprocess, "run") as run:
            self.assertEqual(watcher.nvidia_smi("  "), "No GPUs configured")
        run.assert_not_called()


class QueueWatcherLogsTest(unittest.TestCase):
    def test_history_counts_exact_task_attempts_and_filters_old_logs(self):
        task = watcher.Task("task_a", "online", "FetchPush", "1000", "1000")
        queue_started = datetime(2026, 11, 10, 8).timestamp()
        with TemporaryDirectory() as directory:
            log_dir = Path(directory)
            for name, elapsed in (
                ("task_a_20261109-080000.log", 10 * 3600),
                ("task_a_20261110-080000.log", 3600),
                ("task_a_20261110-100000.log", 2 * 3600),
                ("task_a_extra_20261110-080000.log", 10 * 3600),
            ):
                path = log_dir / name
                path.touch()
                ended = watcher.log_start_timestamp(path) + elapsed
                os.utime(path, (ended, ended))

            log_index = watcher.index_task_logs(log_dir)
            status = {
                "state": "RUNNING",
                "log_file": str(log_dir / "task_a_20261110-100000.log"),
            }
            with mock.patch.object(
                watcher.time, "time", return_value=queue_started + 3 * 3600
            ):
                # One completed hour plus the current attempt's first hour.
                self.assertEqual(
                    watcher.history_elapsed_hours(
                        task, status, log_dir, queue_started, log_index
                    ),
                    2.0,
                )
            self.assertEqual(
                [path.name for path in watcher.task_log_files(
                    log_dir, task, queue_started, log_index
                )],
                ["task_a_20261110-080000.log", "task_a_20261110-100000.log"],
            )
            self.assertEqual(
                watcher.history_elapsed_hours(
                    task, {**status, "state": "DONE"}, log_dir, queue_started,
                    log_index,
                ),
                3.0,
            )

    def test_missing_logs_fall_back_to_status_elapsed_time(self):
        task = watcher.Task("task", "online", "FetchPush", "1000", "1000")
        status = {
            "state": "DONE",
            "started_at": "2026-11-10 08:00:00",
            "finished_at": "2026-11-10 10:00:00",
        }
        with TemporaryDirectory() as directory:
            self.assertEqual(
                watcher.history_elapsed_hours(
                    task, status, Path(directory) / "missing", None
                ),
                2.0,
            )

    def test_render_scans_log_directory_once_and_finds_new_attempts_on_refresh(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            tasks_file = root / "tasks.tsv"
            tasks_file.write_text(
                "task_a\tonline\tFetchPush\t1000\t1000\t2.1m\t\n"
                "task_b\tonline\tFetchSlide\t1001\t1000\t2.1m\t\n"
            )
            log_dir = root / "logs"
            log_dir.mkdir()
            config = {
                "TASKS_FILE": str(tasks_file),
                "STATUS_DIR": str(root / "status"),
                "RESULTS_ROOT": str(root / "results"),
                "LOG_DIR": str(log_dir),
            }
            original_glob = Path.glob
            scans = []

            def count_log_scans(path, pattern):
                if path == log_dir:
                    scans.append(pattern)
                return original_glob(path, pattern)

            def render_output():
                output = io.StringIO()
                with mock.patch.object(Path, "glob", count_log_scans), mock.patch.object(
                    watcher, "nvidia_smi", return_value="no gpus"
                ), redirect_stdout(output):
                    watcher.render(config)
                return output.getvalue()

            self.assertIn("task_elapsed_h=0.00", render_output())
            self.assertEqual(scans, ["*.log"])
            path = log_dir / "task_a_20261110-080000.log"
            path.touch()
            ended = watcher.log_start_timestamp(path) + 3600
            os.utime(path, (ended, ended))
            self.assertIn("task_elapsed_h=1.00", render_output())
            self.assertEqual(scans, ["*.log", "*.log"])


    def test_render_keeps_elapsed_history_after_the_log_directory_changes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            old_logs = root / "old_logs"
            old_logs.mkdir()
            status_dir = root / "status"
            status_dir.mkdir()
            tasks_file = root / "tasks.tsv"
            tasks_file.write_text("task_a\tonline\tFetchPush\t1000\t1000\t2.1m\t\n")
            for name in ("task_a_20261110-080000.log", "task_a_20261110-100000.log"):
                path = old_logs / name
                path.touch()
                ended = watcher.log_start_timestamp(path) + 3600
                os.utime(path, (ended, ended))
            (status_dir / "task_a.status").write_text(
                "state=DONE\nstarted_at=2026-11-10 10:00:00\n"
                "finished_at=2026-11-10 11:00:00\n"
                f"log_file={old_logs / 'task_a_20261110-100000.log'}\n"
            )
            config = {
                "TASKS_FILE": str(tasks_file), "STATUS_DIR": str(status_dir),
                "RESULTS_ROOT": str(root / "results"), "LOG_DIR": str(root / "new_logs"),
            }
            with mock.patch.object(watcher, "nvidia_smi", return_value="no gpus"), redirect_stdout(
                io.StringIO()
            ) as output:
                watcher.render(config)
            self.assertIn("task_elapsed_h=2.00", output.getvalue())


class QueueWatcherInterruptTest(unittest.TestCase):
    def test_ctrl_c_during_render_or_wait_exits_cleanly(self):
        for phase in ("render", "sleep"):
            with self.subTest(phase=phase), mock.patch(
                "sys.argv", ["watch_qrl_queue.py", "--watch", "1"]
            ), mock.patch.object(watcher, "parse_config", return_value={}), mock.patch.object(
                watcher, "RecentProgressHistory"
            ), mock.patch.object(
                watcher, "render",
                side_effect=KeyboardInterrupt if phase == "render" else None,
            ), mock.patch.object(
                watcher.time, "sleep",
                side_effect=KeyboardInterrupt if phase == "sleep" else None,
            ), redirect_stdout(io.StringIO()) as output:
                self.assertEqual(watcher.main(), 130)
                self.assertIn("Stopped watching.", output.getvalue())


if __name__ == "__main__":
    unittest.main()

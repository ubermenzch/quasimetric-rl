import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from tools import run_qrl_queue as queue


class QueueGpuQueriesTest(unittest.TestCase):
    def test_valid_gpu_data_and_process_mapping_survive_exit_255(self):
        for exit_code in (0, 255):
            with self.subTest(exit_code=exit_code), mock.patch.object(
                queue.subprocess, "run", side_effect=[
                    subprocess.CompletedProcess([], exit_code, "3, GPU-abc\n", ""),
                    subprocess.CompletedProcess(
                        [], exit_code, 'GPU-abc, 410162, "python, worker", 1076\n', ""
                    ),
                    subprocess.CompletedProcess([], exit_code, "1109, 24564, 13\n", ""),
                ]
            ):
                apps = queue.gpu_compute_apps()
                self.assertEqual(apps, [{
                    "gpu": "3", "gpu_uuid": "GPU-abc", "pid": "410162",
                    "process_name": "python, worker", "used_memory": "1076",
                }])
                self.assertEqual(queue.query_gpu_states(["3"], apps), {
                    "3": queue.GpuState(
                        gpu="3", mem_used_mb=1109, mem_total_mb=24564, util_pct=13
                    ),
                })

    def test_invalid_uuid_query_prevents_process_reconciliation(self):
        for output in ("", "Failed to initialize NVML\n", "3, GPU-abc\nbad row\n"):
            with self.subTest(output=output), mock.patch.object(
                queue.subprocess, "run",
                return_value=subprocess.CompletedProcess([], 255, output, "error"),
            ) as run:
                self.assertIsNone(queue.gpu_compute_apps())
                run.assert_called_once()

    def test_empty_process_output_is_idle_only_after_successful_query(self):
        for exit_code, expected in ((0, []), (255, None)):
            with self.subTest(exit_code=exit_code), mock.patch.object(
                queue, "gpu_uuid_to_index", return_value={"GPU-abc": "3"}
            ), mock.patch.object(
                queue.subprocess, "run",
                return_value=subprocess.CompletedProcess([], exit_code, "", ""),
            ):
                self.assertEqual(queue.gpu_compute_apps(), expected)

    def test_invalid_or_incomplete_process_rows_are_not_treated_as_idle(self):
        for output in (
            "Failed to initialize NVML\n",
            "GPU-abc, 410162, python\n",
            "GPU-abc, invalid-pid, python, 1076\n",
            "GPU-abc, 410162, python, 1076\nbad row\n",
            "GPU-unknown, 410162, python, 1076\n",
        ):
            with self.subTest(output=output), mock.patch.object(
                queue, "gpu_uuid_to_index", return_value={"GPU-abc": "3"}
            ), mock.patch.object(
                queue.subprocess, "run",
                return_value=subprocess.CompletedProcess([], 255, output, ""),
            ):
                self.assertIsNone(queue.gpu_compute_apps())

    def test_invalid_gpu_state_is_unavailable_even_with_exit_zero(self):
        for output in (
            "", "Failed to initialize NVML\n", "1109, 24564\n",
            "1109, 24564, 13\nerror\n", "N/A, 24564, 13\n",
            "-1, 24564, 13\n", "1109, 0, 13\n", "1109, 24564, 101\n",
            "1109, 24564, inf\n", "30000, 24564, 13\n",
        ):
            with self.subTest(output=output), mock.patch.object(
                queue.subprocess, "run",
                return_value=subprocess.CompletedProcess([], 0, output, ""),
            ):
                self.assertIsNone(queue.query_gpu_state("3", []))

    def test_disabled_disk_threshold_allows_a_new_results_directory(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            tasks = root / "tasks.tsv"
            tasks.write_text("# empty queue\n")
            config = {
                "TASKS_FILE": str(tasks), "STATUS_DIR": str(root / "status"),
                "LOCK_FILE": str(root / "queue.lock"),
                "RESULTS_ROOT": str(root / "new" / "results"),
                "MIN_FREE_DISK_GIB": "0", "GPU_IDS": "", "DRY_RUN": "1",
            }
            with mock.patch("sys.argv", ["run_qrl_queue.py", "--once"]), mock.patch.object(
                queue, "parse_config", return_value=config
            ), mock.patch.object(queue, "gpu_compute_apps", return_value=[]), mock.patch.object(
                queue, "available_disk_gib", side_effect=AssertionError("disabled disk check ran")
            ):
                self.assertEqual(queue.main(), 0)
            self.assertFalse((root / "new").exists())

    def test_gpu_query_timeout_keeps_gpu_unavailable(self):
        with mock.patch.object(
            queue.subprocess, "run", side_effect=subprocess.TimeoutExpired("nvidia-smi", 10)
        ):
            self.assertIsNone(queue.query_gpu_state("3", []))
            self.assertIsNone(queue.gpu_compute_apps())


if __name__ == "__main__":
    unittest.main()

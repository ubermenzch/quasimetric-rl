import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from tools.run_qrl_queue import Task
from tools.run_qrl_queue import read_status
from tools.run_qrl_queue import remote_result_sync_eligible
from tools.run_qrl_queue import remote_result_sync_marker
from tools.run_qrl_queue import remote_results_sync_baseline
from tools.run_qrl_queue import sync_completed_task_result
from tools.run_qrl_queue import sync_finished_outputs
from tools.run_qrl_queue import write_status
from tools.run_qrl_queue import write_task_manifest


class QrlRemoteResultSyncTest(unittest.TestCase):
    def make_config(self, root: Path, baseline: Path) -> dict[str, str]:
        key = root / "transfer_key"
        known_hosts = root / "known_hosts"
        key.write_text("private")
        known_hosts.write_text("host key")
        return {
            "RESULTS_ROOT": str(root / "results"),
            "REMOTE_RESULTS_SYNC_ENABLED": "1",
            "REMOTE_RESULTS_SYNC_HOST": "10.82.1.225",
            "REMOTE_RESULTS_SYNC_PORT": "8899",
            "REMOTE_RESULTS_SYNC_USER": "zhangcheng",
            "REMOTE_RESULTS_SYNC_ROOT": "/data2/zhangcheng/results/queue",
            "REMOTE_RESULTS_SYNC_KEY": str(key),
            "REMOTE_RESULTS_SYNC_KNOWN_HOSTS": str(known_hosts),
            "REMOTE_RESULTS_SYNC_BASELINE_FILE": str(baseline),
            "REMOTE_RESULTS_SYNC_TIMEOUT_SECONDS": "60",
        }

    def test_baseline_and_eligibility_exclude_existing_results(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.txt"
            baseline.write_text("# existing\nold_task\n")
            task_ids = remote_results_sync_baseline({
                "REMOTE_RESULTS_SYNC_BASELINE_FILE": str(baseline),
            })
            output = root / "old_task"
            output.mkdir()
            (output / "COMPLETE").touch()
            status = {"state": "DONE"}

            self.assertEqual(task_ids, {"old_task"})
            self.assertFalse(remote_result_sync_eligible(
                Task("old_task", "online", "FetchPush", "1000", "100000"),
                status,
                output,
                task_ids,
            ))
            self.assertTrue(remote_result_sync_eligible(
                Task("new_task", "online", "FetchPush", "1000", "100000"),
                status,
                output,
                task_ids,
            ))
            remote_result_sync_marker(output).touch()
            self.assertFalse(remote_result_sync_eligible(
                Task("new_task", "online", "FetchPush", "1000", "100000"),
                status,
                output,
                task_ids,
            ))

    def test_successful_sync_verifies_complete_and_writes_marker(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.txt"
            baseline.write_text("")
            config = self.make_config(root, baseline)
            task = Task("new_task", "online", "FetchPush", "1000", "100000")
            output = root / "results" / task.task_id
            output.mkdir(parents=True)
            (output / "COMPLETE").touch()
            completed = subprocess.CompletedProcess([], 0, "", "")

            with mock.patch(
                "tools.run_qrl_queue.subprocess.run",
                side_effect=[completed, completed],
            ) as run:
                self.assertTrue(sync_completed_task_result(config, task, output))

            self.assertEqual(run.call_count, 2)
            rsync_command = run.call_args_list[0].args[0]
            self.assertEqual(rsync_command[0], "rsync")
            self.assertIn(
                "zhangcheng@10.82.1.225:"
                "/data2/zhangcheng/results/queue/new_task/",
                rsync_command,
            )
            marker = json.loads(remote_result_sync_marker(output).read_text())
            self.assertEqual(marker["task_id"], task.task_id)

    def test_failed_sync_keeps_result_unmarked(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.txt"
            baseline.write_text("")
            config = self.make_config(root, baseline)
            task = Task("new_task", "online", "FetchPush", "1000", "100000")
            output = root / "results" / task.task_id
            output.mkdir(parents=True)
            (output / "COMPLETE").touch()
            failed = subprocess.CompletedProcess([], 23, "", "network error")

            with mock.patch(
                "tools.run_qrl_queue.subprocess.run", return_value=failed,
            ):
                self.assertFalse(sync_completed_task_result(config, task, output))

            self.assertFalse(remote_result_sync_marker(output).exists())

    def test_remote_verification_timeout_keeps_result_unmarked(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.txt"
            baseline.write_text("")
            config = self.make_config(root, baseline)
            task = Task("new_task", "online", "FetchPush", "1000", "100000")
            output = root / "results" / task.task_id
            output.mkdir(parents=True)
            (output / "COMPLETE").touch()
            completed = subprocess.CompletedProcess([], 0, "", "")

            with mock.patch(
                "tools.run_qrl_queue.subprocess.run",
                side_effect=[completed, subprocess.TimeoutExpired("ssh", 30)],
            ):
                self.assertFalse(sync_completed_task_result(config, task, output))

            self.assertFalse(remote_result_sync_marker(output).exists())

    def test_finished_output_scan_syncs_new_task_but_skips_baseline(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.txt"
            baseline.write_text("old_task\n")
            config = self.make_config(root, baseline)
            status_dir = root / "status"
            tasks = [
                Task("old_task", "online", "FetchPush", "1000", "100000"),
                Task("new_task", "online", "FetchPush", "1000", "100000"),
            ]
            for task in tasks:
                output = root / "results" / task.task_id
                output.mkdir(parents=True)
                (output / "COMPLETE").touch()
                write_task_manifest(output, task)
                write_status(status_dir, task, "DONE", {
                    "output_dir": str(output),
                    "completion_evidence": "COMPLETE",
                })

            with mock.patch(
                "tools.run_qrl_queue.sync_completed_task_result",
                return_value=True,
            ) as sync, mock.patch(
                "tools.run_qrl_queue.cleanup_completed_task_checkpoints",
                return_value=False,
            ):
                pending = sync_finished_outputs(config, tasks, status_dir)

            sync.assert_called_once()
            self.assertFalse(pending)
            self.assertEqual(sync.call_args.args[1].task_id, "new_task")
            self.assertEqual(read_status(status_dir, "new_task")["state"], "DONE")

    def test_sync_limit_reports_pending_remote_results(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.txt"
            baseline.write_text("")
            config = self.make_config(root, baseline)
            config["REMOTE_RESULTS_SYNC_MAX_PER_PASS"] = "1"
            status_dir = root / "status"
            tasks = [
                Task(f"new_task_{index}", "online", "FetchPush", "1000", "100000")
                for index in range(2)
            ]
            for task in tasks:
                output = root / "results" / task.task_id
                output.mkdir(parents=True)
                (output / "COMPLETE").touch()
                write_task_manifest(output, task)
                write_status(status_dir, task, "DONE", {
                    "output_dir": str(output),
                    "completion_evidence": "COMPLETE",
                })

            with mock.patch(
                "tools.run_qrl_queue.sync_completed_task_result",
                return_value=True,
            ) as sync, mock.patch(
                "tools.run_qrl_queue.cleanup_completed_task_checkpoints",
                return_value=False,
            ):
                pending = sync_finished_outputs(config, tasks, status_dir)

            self.assertTrue(pending)
            self.assertEqual(sync.call_count, 1)


if __name__ == "__main__":
    unittest.main()

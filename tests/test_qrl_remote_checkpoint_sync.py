import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from tools.run_qrl_queue import Task
from tools.run_qrl_queue import checkpoint_sync_marker_is_current
from tools.run_qrl_queue import cleanup_completed_task_checkpoints
from tools.run_qrl_queue import read_status
from tools.run_qrl_queue import remote_checkpoint_sync_marker
from tools.run_qrl_queue import remote_result_sync_marker
from tools.run_qrl_queue import requeue_existing_disk_full_failures
from tools.run_qrl_queue import sync_task_checkpoints
from tools.run_qrl_queue import sync_task_checkpoints_to_remote
from tools.run_qrl_queue import sync_finished_outputs
from tools.run_qrl_queue import write_status
from tools.run_qrl_queue import write_task_manifest


class QrlRemoteCheckpointSyncTest(unittest.TestCase):
    def make_config(self, root: Path) -> dict[str, str]:
        key = root / "transfer_key"
        known_hosts = root / "known_hosts"
        key.write_text("private")
        known_hosts.write_text("host key")
        return {
            "RESULTS_ROOT": str(root / "results"),
            "REMOTE_CHECKPOINT_SYNC_ENABLED": "1",
            "REMOTE_CHECKPOINT_SYNC_MAX_PER_PASS": "10",
            "REMOTE_CHECKPOINT_DELETE_COMPLETED_LOCAL": "1",
            "REMOTE_RESULTS_SYNC_HOST": "10.82.1.225",
            "REMOTE_RESULTS_SYNC_PORT": "8899",
            "REMOTE_RESULTS_SYNC_USER": "zhangcheng",
            "REMOTE_RESULTS_SYNC_ROOT": "/data2/zhangcheng/results/queue",
            "REMOTE_RESULTS_SYNC_KEY": str(key),
            "REMOTE_RESULTS_SYNC_KNOWN_HOSTS": str(known_hosts),
            "REMOTE_RESULTS_SYNC_TIMEOUT_SECONDS": "60",
        }

    def make_output(self, root: Path, task: Task) -> Path:
        output = root / "results" / task.task_id
        output.mkdir(parents=True)
        write_task_manifest(output, task)
        return output

    def add_final_evaluation(self, output: Path) -> None:
        (output / "COMPLETE").touch()
        (output / "test.log").write_text('{"succ_rate": 0.7}\n')
        (output / "best_checkpoint.json").write_text(json.dumps({
            "validation": {"succ_rate": 0.8},
            "test": {"succ_rate": 0.7},
            "selected_model": "selected_best_agent.pth",
        }))

    def test_sync_copies_only_checkpoints_and_verifies_remote_sizes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.make_config(root)
            task = Task("task_alpha", "online", "FetchPush", "1000", "200000")
            output = self.make_output(root, task)
            files = {
                "checkpoint_env00020000_opt00010000.pth": "full checkpoint",
                "selected_best_agent.pth": "selected agent",
            }
            for name, payload in files.items():
                (output / name).write_text(payload)
            (output / "metrics.jsonl").write_text("keep local\n")
            remote_root = config["REMOTE_RESULTS_SYNC_ROOT"]
            stat_output = "".join(
                f"{remote_root}/{task.task_id}/{name}\t{len(payload)}\n"
                for name, payload in sorted(files.items())
            )
            completed = subprocess.CompletedProcess([], 0, "", "")
            verified = subprocess.CompletedProcess([], 0, stat_output, "")

            with mock.patch(
                "tools.run_qrl_queue.subprocess.run",
                side_effect=[completed, verified],
            ) as run:
                self.assertTrue(
                    sync_task_checkpoints_to_remote(config, task, output)
                )

            rsync_command = run.call_args_list[0].args[0]
            self.assertNotIn(str(output / "metrics.jsonl"), rsync_command)
            for name in files:
                self.assertIn(str(output / name), rsync_command)
            self.assertTrue(
                checkpoint_sync_marker_is_current(config, task, output)
            )

    def test_size_mismatch_does_not_write_verified_marker(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.make_config(root)
            task = Task("task_alpha", "online", "FetchPush", "1000", "200000")
            output = self.make_output(root, task)
            (output / "checkpoint_env00020000_opt00010000.pth").write_text("data")
            completed = subprocess.CompletedProcess([], 0, "", "")
            wrong_size = subprocess.CompletedProcess(
                [],
                0,
                "/data2/zhangcheng/results/queue/task_alpha/"
                "checkpoint_env00020000_opt00010000.pth\t3\n",
                "",
            )

            with mock.patch(
                "tools.run_qrl_queue.subprocess.run",
                side_effect=[completed, wrong_size],
            ):
                self.assertFalse(
                    sync_task_checkpoints_to_remote(config, task, output)
                )

            self.assertFalse(remote_checkpoint_sync_marker(output).exists())

    def test_completed_checkpoint_is_deleted_only_after_verified_copy(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.make_config(root)
            status_dir = root / "status"
            task = Task("task_alpha", "online", "FetchPush", "1000", "200000")
            output = self.make_output(root, task)
            checkpoint = output / "selected_best_agent.pth"
            checkpoint.write_text("selected")
            self.add_final_evaluation(output)
            write_status(status_dir, task, "DONE", {"output_dir": str(output)})

            with mock.patch(
                "tools.run_qrl_queue.sync_task_checkpoints_to_remote",
                return_value=True,
            ):
                pending = sync_task_checkpoints(config, [task], status_dir)

            self.assertFalse(pending)
            self.assertFalse(checkpoint.exists())
            self.assertTrue((output / "CHECKPOINTS_DELETED").is_file())
            self.assertTrue((output / "test.log").is_file())

    def test_wrong_full_result_marker_never_authorizes_local_deletion(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.make_config(root)
            status_dir = root / "status"
            task = Task("task_alpha", "online", "FetchPush", "1000", "200000")
            output = self.make_output(root, task)
            checkpoint = output / "selected_best_agent.pth"
            checkpoint.write_text("selected")
            self.add_final_evaluation(output)
            remote_result_sync_marker(output).write_text(json.dumps({
                "host": "wrong-host",
                "remote_path": "/wrong/path/task_alpha",
                "task_id": task.task_id,
            }))
            write_status(status_dir, task, "DONE", {"output_dir": str(output)})

            with mock.patch(
                "tools.run_qrl_queue.sync_task_checkpoints_to_remote",
                return_value=False,
            ) as sync:
                pending = sync_task_checkpoints(config, [task], status_dir)

            self.assertTrue(pending)
            sync.assert_called_once()
            self.assertTrue(checkpoint.exists())

    def test_explicit_preservation_overrides_global_cleanup_after_backup(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.make_config(root)
            status_dir = root / 'status'
            task = Task('preserved', 'online', 'AntNavigate-v4', '1000', '500000',
                        extra_args='queue.delete_checkpoints_after_completion=false')
            output = self.make_output(root, task)
            checkpoint = output / 'checkpoint_env00500000_opt00490000.pth'
            checkpoint.write_text('model, optimizer and replay')
            self.add_final_evaluation(output)
            write_status(status_dir, task, 'DONE', {'output_dir': str(output)})
            with mock.patch('tools.run_qrl_queue.sync_task_checkpoints_to_remote', return_value=True) as sync:
                pending = sync_task_checkpoints(config, [task], status_dir)
            sync.assert_called_once()
            self.assertFalse(pending)
            self.assertTrue(checkpoint.exists())
            self.assertFalse(cleanup_completed_task_checkpoints(task, output, force=True))
            self.assertTrue(checkpoint.exists())
            self.assertFalse((output / 'CHECKPOINTS_DELETED').exists())

    def test_legacy_cleanup_flag_waits_for_remote_checkpoint_verification(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.make_config(root)
            config["REMOTE_RESULTS_SYNC_ENABLED"] = "0"
            status_dir = root / "status"
            task = Task(
                "task_alpha",
                "online",
                "FetchPush",
                "1000",
                "200000",
                extra_args="queue.delete_checkpoints_after_completion=true",
            )
            output = self.make_output(root, task)
            checkpoint = output / "selected_best_agent.pth"
            checkpoint.write_text("selected")
            self.add_final_evaluation(output)
            write_status(status_dir, task, "DONE", {"output_dir": str(output)})

            sync_finished_outputs(config, [task], status_dir)

            self.assertTrue(checkpoint.exists())
            self.assertFalse((output / "CHECKPOINTS_DELETED").exists())

    def test_disk_full_failure_is_requeued_for_resume(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            status_dir = root / "status"
            task = Task("task_alpha", "online", "FetchPush", "1000", "200000")
            log_file = root / "task.log"
            log_file.write_text(
                "RuntimeError: PytorchStreamWriter failed writing file data/0\n"
            )
            write_status(status_dir, task, "FAILED", {
                "exit_code": "1",
                "log_file": str(log_file),
                "error": "nonzero_exit",
            })

            requeue_existing_disk_full_failures(
                {"REQUEUE_DISK_FULL": "1"},
                [task],
                status_dir,
                False,
            )

            status = read_status(status_dir, task.task_id)
            self.assertEqual(status["state"], "PENDING")
            self.assertEqual(status["error"], "disk_full_requeued")
            self.assertEqual(
                status["requeue_reason"],
                "PytorchStreamWriter failed writing file",
            )
            self.assertNotIn("finished_at", status)


if __name__ == "__main__":
    unittest.main()

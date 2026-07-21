import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.run_qrl_queue import GpuState
from tools.run_qrl_queue import Task as RunnerTask
from tools.run_qrl_queue import ensure_task_submission_statuses
from tools.run_qrl_queue import gpu_accepts_more_jobs
from tools.run_qrl_queue import read_status
from tools.run_qrl_queue import requeue_cuda_oom_status
from tools.run_qrl_queue import requeue_existing_transient_failures
from tools.run_qrl_queue import task_uses_goal_set_objective
from tools.run_qrl_queue import update_running_gpu_memory_peaks
from tools.run_qrl_queue import write_status
from tools.watch_qrl_queue import Task as WatcherTask
from tools.watch_qrl_queue import display_status_timestamp
from tools.watch_qrl_queue import task_variant


class QueueTaskClassificationTest(unittest.TestCase):
    def test_direct_variants_are_derived_from_task_arguments(self):
        task = WatcherTask(
            task_id="official_qrl_Direct-LMEMin-DatasetRadius_s1000",
            mode="offline",
            env_name="maze2d-umaze-v1",
            seed="1000",
            steps="40000",
            extra_args=(
                "agent.goal_set_distance.enabled=true "
                "agent.goal_set_distance.losses.implementation=direct "
                "agent.goal_set_distance.losses.aggregation=lme_min "
                "agent.goal_set_distance.losses.candidate_sampling=dataset_radius"
            ),
        )
        self.assertEqual(task_variant(task), "Direct/LMEMin/DatasetRadius")

    def test_existing_base_and_gsd_labels_are_preserved(self):
        common = dict(
            mode="offline",
            env_name="maze2d-umaze-v1",
            seed="1000",
            steps="40000",
        )
        self.assertEqual(
            task_variant(WatcherTask(task_id="run_Base_s1000", extra_args="", **common)),
            "Base",
        )
        self.assertEqual(
            task_variant(WatcherTask(task_id="run_GSD_s1000", extra_args="", **common)),
            "GSD",
        )

    def test_runner_detects_direct_goal_set_tasks_from_arguments(self):
        task = RunnerTask(
            task_id="arbitrary_name",
            mode="offline",
            env_name="maze2d-umaze-v1",
            seed="1000",
            steps="40000",
            extra_args=(
                "agent.goal_set_distance.enabled=true "
                "agent.goal_set_distance.losses.implementation=direct"
            ),
        )
        self.assertTrue(task_uses_goal_set_objective(task))
        task.extra_args = "agent.goal_set_distance.enabled=false"
        self.assertFalse(task_uses_goal_set_objective(task))

    def test_submission_gpu_start_and_memory_peak_are_persisted(self):
        task = RunnerTask(
            task_id="timed_task",
            mode="offline",
            env_name="maze2d-umaze-v1",
            seed="1000",
            steps="40000",
        )
        with TemporaryDirectory() as directory:
            status_dir = Path(directory)
            ensure_task_submission_statuses(
                [task],
                status_dir,
                submission_fallback="2026-07-16 15:28:47",
            )
            pending = read_status(status_dir, task.task_id)
            self.assertEqual(pending["state"], "PENDING")
            self.assertEqual(pending["submitted_at"], "2026-07-16 15:28:47")

            write_status(status_dir, task, "RUNNING", {
                "gpu": "2",
                "pid": "12345",
                "started_at": "2026-07-16 15:31:02",
                "gpu_started_at": "2026-07-16 15:31:02",
            })
            update_running_gpu_memory_peaks([task], status_dir, [{
                "gpu": "2",
                "pid": "12345",
                "used_memory": "12,345 MiB",
            }])
            running = read_status(status_dir, task.task_id)
            self.assertEqual(running["submitted_at"], pending["submitted_at"])
            self.assertEqual(running["gpu_started_at"], "2026-07-16 15:31:02")
            self.assertEqual(running["gpu_mem_peak_mb"], "12345")

            update_running_gpu_memory_peaks([task], status_dir, [{
                "gpu": "2",
                "pid": "12345",
                "used_memory": "12000",
            }])
            self.assertEqual(
                read_status(status_dir, task.task_id)["gpu_mem_peak_mb"],
                "12345",
            )

    def test_watcher_compacts_status_timestamps(self):
        self.assertEqual(
            display_status_timestamp("2026-07-16 15:31:02"),
            "07-16 15:31:02",
        )

    def test_cuda_oom_requeues_with_a_stricter_prelaunch_memory_limit(self):
        task = RunnerTask(
            task_id="oom_task",
            mode="offline",
            env_name="maze2d-umaze-v1",
            seed="1000",
            steps="40000",
        )
        oom_text = (
            "CUDA out of memory. GPU 0 has a total capacity of 23.52 GiB "
            "of which 1.07 GiB is free. Process 123 has 10.13 GiB memory "
            "in use. Including non-PyTorch memory, this process has "
            "12.29 GiB memory in use."
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            status_dir = root / "status"
            log_file = root / "oom.log"
            log_file.write_text(oom_text)
            write_status(status_dir, task, "FAILED", {
                "exit_code": "1",
                "log_file": str(log_file),
                "error": "nonzero_exit",
            })

            status = read_status(status_dir, task.task_id)
            self.assertTrue(requeue_cuda_oom_status(
                {"REQUEUE_CUDA_OOM": "1"}, task, status_dir, status
            ))
            first_retry = read_status(status_dir, task.task_id)
            self.assertEqual(first_retry["state"], "PENDING")
            self.assertEqual(first_retry["oom_retry_count"], "1")
            self.assertEqual(first_retry["oom_prelaunch_mem_limit_mb"], "10404")

            write_status(status_dir, task, "RUNNING", {
                "gpu": "3",
                "pid": "456",
                "log_file": str(log_file),
                "prelaunch_gpu_mem_used_mb": "3000",
            })
            status = read_status(status_dir, task.task_id)
            self.assertTrue(requeue_cuda_oom_status(
                {"REQUEUE_CUDA_OOM": "1"}, task, status_dir, status
            ))
            second_retry = read_status(status_dir, task.task_id)
            self.assertEqual(second_retry["oom_retry_count"], "2")
            self.assertEqual(second_retry["oom_prelaunch_mem_limit_mb"], "3000")

            scheduler_config = {
                "GPU_UTIL_LIMIT_PCT": "0",
                "GPU_MEM_LIMIT_PCT": "100",
                "MAX_JOBS_PER_GPU": "0",
                "GSD_MAX_PRELAUNCH_MEM_MB": "0",
                "IGNORE_EXTERNAL_GPU_USERS": "1",
            }
            common_args = (
                scheduler_config,
                0,
                [],
                set(),
                task,
                second_retry,
            )
            self.assertTrue(gpu_accepts_more_jobs(
                common_args[0],
                GpuState("0", 2999, 24000, 0),
                *common_args[1:],
            ))
            self.assertFalse(gpu_accepts_more_jobs(
                common_args[0],
                GpuState("0", 3000, 24000, 0),
                *common_args[1:],
            ))

    def test_existing_cuda_unknown_error_requeues_without_oom_limit(self):
        task = RunnerTask(
            task_id="cuda_unknown_task",
            mode="offline",
            env_name="maze2d-umaze-v1",
            seed="1001",
            steps="40000",
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            status_dir = root / "status"
            log_file = root / "cuda_unknown.log"
            log_file.write_text(
                "RuntimeError: CUDA unknown error - this may be due to an "
                "incorrectly set up environment. Setting the available "
                "devices to be zero.\n"
            )
            write_status(status_dir, task, "FAILED", {
                "exit_code": "1",
                "log_file": str(log_file),
                "prelaunch_gpu_mem_used_mb": "10397",
                "error": "nonzero_exit",
            })

            requeue_existing_transient_failures(
                {
                    "REQUEUE_TRANSIENT_FAILURES": "1",
                    "MAX_TRANSIENT_RETRIES": "10",
                },
                [task],
                status_dir,
                False,
            )

            pending = read_status(status_dir, task.task_id)
            self.assertEqual(pending["state"], "PENDING")
            self.assertEqual(pending["error"], "transient_failure_requeued")
            self.assertEqual(pending["requeue_reason"], "CUDA unknown error")
            self.assertEqual(pending["transient_failure_count"], "1")
            self.assertNotIn("finished_at", pending)
            self.assertNotIn("oom_retry_count", pending)
            self.assertNotIn("oom_prelaunch_mem_limit_mb", pending)


if __name__ == "__main__":
    unittest.main()

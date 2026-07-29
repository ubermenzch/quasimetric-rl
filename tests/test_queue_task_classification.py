import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from tools.run_qrl_queue import ActiveJob
from tools.run_qrl_queue import GpuState
from tools.run_qrl_queue import Task as RunnerTask
from tools.run_qrl_queue import ensure_task_submission_statuses
from tools.run_qrl_queue import gpu_accepts_more_jobs
from tools.run_qrl_queue import mark_finished
from tools.run_qrl_queue import read_status
from tools.run_qrl_queue import reconcile_running_statuses
from tools.run_qrl_queue import requeue_cuda_oom_status
from tools.run_qrl_queue import requeue_existing_transient_failures
from tools.run_qrl_queue import task_uses_goal_set_objective
from tools.run_qrl_queue import task_fingerprint
from tools.run_qrl_queue import update_running_gpu_memory_peaks
from tools.run_qrl_queue import write_status
from tools.watch_qrl_queue import Task as WatcherTask
from tools.watch_qrl_queue import compact_count
from tools.watch_qrl_queue import compact_parameter_count
from tools.watch_qrl_queue import display_status_timestamp
from tools.watch_qrl_queue import task_checkpoint_interval
from tools.watch_qrl_queue import task_critic_count
from tools.watch_qrl_queue import task_display_name
from tools.watch_qrl_queue import task_parameter_count
from tools.watch_qrl_queue import task_variant


class QueueTaskClassificationTest(unittest.TestCase):
    def test_dead_allowed_gpu_status_is_automatically_requeued(self):
        task = RunnerTask(
            task_id="stale_task",
            mode="online",
            env_name="FetchPush",
            seed="1000",
            steps="200000",
        )
        with TemporaryDirectory() as directory:
            status_dir = Path(directory)
            write_status(status_dir, task, "RUNNING", {
                "gpu": "3",
                "pid": "999999",
                "output_dir": str(Path(directory) / "output"),
            })
            with mock.patch('tools.run_qrl_queue.pid_alive', return_value=False):
                reconcile_running_statuses(
                    {
                        "STRICT_GPU_STATE_SYNC": "1",
                        "TERMINATE_DISALLOWED_GPU_JOBS": "0",
                        "REQUEUE_CUDA_OOM": "0",
                    },
                    [task],
                    status_dir,
                    ["0", "1", "2", "3", "4", "5"],
                    [],
                    False,
                )

            status = read_status(status_dir, task.task_id)
            self.assertEqual(status["state"], "PENDING")
            self.assertEqual(status["error"], "stale_running_status")
            self.assertEqual(status["previous_gpu"], "3")

    def test_dead_disallowed_gpu_status_is_automatically_requeued(self):
        task = RunnerTask(
            task_id="stale_task",
            mode="online",
            env_name="FetchPush",
            seed="1000",
            steps="200000",
        )
        with TemporaryDirectory() as directory:
            status_dir = Path(directory)
            write_status(status_dir, task, "RUNNING", {
                "gpu": "7",
                "pid": "999999",
                "output_dir": str(Path(directory) / "output"),
            })
            with mock.patch('tools.run_qrl_queue.pid_alive', return_value=False):
                running = reconcile_running_statuses(
                    {
                        "STRICT_GPU_STATE_SYNC": "1",
                        "TERMINATE_DISALLOWED_GPU_JOBS": "0",
                        "REQUEUE_CUDA_OOM": "0",
                    },
                    [task],
                    status_dir,
                    ["0", "1", "2", "3", "4", "5"],
                    [],
                    False,
                )

            self.assertEqual(running, {str(gpu): set() for gpu in range(6)})
            status = read_status(status_dir, task.task_id)
            self.assertEqual(status["state"], "PENDING")
            self.assertEqual(status["error"], "disallowed_gpu_requeued")
            self.assertEqual(status["previous_gpu"], "7")

    def test_interrupted_training_exit_is_requeued(self):
        task = RunnerTask(
            task_id="interrupted_task",
            mode="online",
            env_name="FetchPush",
            seed="1000",
            steps="200000",
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            status_dir = root / "status"
            output_dir = root / "output"
            output_dir.mkdir()
            proc = mock.Mock(pid=12345)
            job = ActiveJob(
                task=task,
                gpu="2",
                proc=proc,
                log_file=root / "task.log",
                output_dir=output_dir,
            )
            write_status(status_dir, task, "RUNNING", {
                "gpu": "2",
                "pid": str(proc.pid),
                "output_dir": str(output_dir),
            })

            mark_finished({"STATUS_DIR": str(status_dir)}, job, 130)

            status = read_status(status_dir, task.task_id)
            self.assertEqual(status["state"], "PENDING")
            self.assertEqual(status["error"], "training_interrupted_requeued")
            self.assertEqual(status["previous_gpu"], "2")

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

    def test_reward_free_baselines_have_algorithm_labels(self):
        common = dict(
            task_id="reference_baseline_s1000",
            mode="online",
            env_name="FetchReach",
            seed="1000",
            steps="200000",
        )
        expected = {
            "td_infonce": "TD-InfoNCE",
            "crl": "CRL",
            "gcbc": "GCBC",
            "gcsl": "GCSL/GCBC",
            "c_learning": "C-Learning",
        }
        for algorithm, label in expected.items():
            with self.subTest(algorithm=algorithm):
                task = WatcherTask(
                    extra_args=f"agent.algorithm={algorithm}", **common
                )
                self.assertEqual(task_variant(task), label)

    def test_base_task_properties_are_split_into_monitor_columns(self):
        task = WatcherTask(
            task_id=(
                "official_qrl_1q_Base_bc0_100k_5kckpt_"
                "maze2d_umaze_offlinegoals_s1000"
            ),
            mode="offline",
            env_name="maze2d-umaze-v1",
            seed="1000",
            steps="100000",
            extra_args=(
                "agent.num_critics=1 "
                "agent.quasimetric_critic.model.latent_dynamics.history_length=1 "
                "agent.actor.losses.behavior_cloning.weight=0 "
                "save_steps=5000"
            ),
        )
        self.assertEqual(task_display_name(task), "official_qrl_offlinegoals")
        self.assertEqual(task_variant(task), "Base")
        self.assertEqual(task_critic_count(task), "1")
        self.assertEqual(compact_count(task.steps), "100k")
        self.assertEqual(task_checkpoint_interval(task), "5k")

    def test_go_qrl_variant_is_derived_from_task_arguments(self):
        task = WatcherTask(
            task_id=(
                "official_qrl_1q_SplitLatentMax8_RMSBase1q_bc0_100k_"
                "maze2d_umaze_offlinegoals_s1001"
            ),
            mode="offline",
            env_name="maze2d-umaze-v1",
            seed="1001",
            steps="100000",
            extra_args=(
                "agent.num_critics=1 "
                "agent.quasimetric_critic.model.encoder.kind=split "
                "agent.actor.model.input_mode=split_latent "
                "agent.actor.losses.min_dist.latent_goal_mode=max "
                "agent.actor.losses.min_dist.latent_goal_steps=8 "
                "agent.actor.losses.behavior_cloning.weight=0 "
                "save_steps=10000"
            ),
        )
        self.assertEqual(task_display_name(task), "official_qrl_RMSBase1q_offlinegoals")
        self.assertEqual(task_variant(task), "GO-QRL+Max8")
        self.assertEqual(task_checkpoint_interval(task), "10k")

        task.extra_args += (
            " agent.actor.losses.min_dist.latent_goal_search=bounded_residual"
            " agent.actor.losses.min_dist.latent_goal_residual_radius=1.0"
        )
        self.assertEqual(task_variant(task), "GO-QRL+Max8+BR1")

    def test_zero_inner_steps_are_labeled_inner0(self):
        task = WatcherTask(
            task_id="ablation_GO-QRL+Inner0-M_fetchpush_s1000",
            mode="online",
            env_name="FetchPush",
            seed="1000",
            steps="200000",
            extra_args=(
                "+go_qrl_model_size=m "
                "agent.actor.losses.min_dist.latent_goal_mode=min "
                "agent.actor.losses.min_dist.latent_goal_steps=0"
            ),
        )
        self.assertEqual(task_variant(task), "GO-QRL+Inner0")

    def test_layernorm_go_qrl_variant_has_a_distinct_monitor_label(self):
        task = WatcherTask(
            task_id="official_qrl_1q_SplitLatentMin8_LayerNorm_M_s1000",
            mode="online",
            env_name="FetchReach",
            seed="1000",
            steps="200000",
            extra_args=(
                "agent.quasimetric_critic.model.encoder.kind=split "
                "agent.quasimetric_critic.model.encoder.branch_normalization=layernorm "
                "agent.actor.model.input_mode=split_latent "
                "agent.actor.losses.min_dist.latent_goal_mode=min "
                "agent.actor.losses.min_dist.latent_goal_steps=8 "
                "agent.actor.losses.min_dist.latent_goal_search=direct"
            ),
        )

        self.assertEqual(task_variant(task), "GO-QRL+Min8+LN")
        self.assertEqual(
            task_display_name(task),
            "official_qrl_LayerNorm_M",
        )

        task.extra_args += " agent.actor.losses.min_dist.latent_goal_optim=rmsg"
        self.assertEqual(task_variant(task), "GO-QRL+Min8+LN+RMSG")

    def test_versioned_layernorm_variant_is_composed_from_task_modules(self):
        task = WatcherTask(
            task_id=(
                "official_qrl_1q_GO-QRL+LNv1+RMSG-M_200k_"
                "fetchreach_online_s1000"
            ),
            mode="online",
            env_name="FetchReach",
            seed="1000",
            steps="200000",
            extra_args=(
                "+go_qrl_model_size=m "
                "agent.quasimetric_critic.model.encoder.branch_normalization=layernorm "
                "agent.quasimetric_critic.model.dynamics_output_normalization=none "
                "agent.actor.losses.min_dist.latent_goal_mode=max "
                "agent.actor.losses.min_dist.latent_goal_steps=4 "
                "agent.actor.losses.min_dist.latent_goal_optim=rmsg"
            ),
        )

        self.assertEqual(task_variant(task), "GO-QRL+Max4+LNv1+RMSG")

    def test_residual_go_qrl_variant_has_a_distinct_monitor_label(self):
        task = WatcherTask(
            task_id="official_qrl_1q_SplitLatentMax4_Residual_M_s1000",
            mode="online",
            env_name="FetchReach",
            seed="1000",
            steps="200000",
            extra_args=(
                "+go_qrl_model_size=m "
                "agent.quasimetric_critic.model.encoder.branch_normalization=none "
                "agent.actor.model.input_mode=split_latent "
                "agent.actor.losses.min_dist.latent_goal_mode=max "
                "agent.actor.losses.min_dist.latent_goal_steps=4 "
                "agent.actor.losses.min_dist.latent_goal_search=residual"
            ),
        )

        self.assertEqual(task_variant(task), "GO-QRL+Max4+Res")
        self.assertEqual(
            task_display_name(task),
            "official_qrl_Residual_M",
        )

    def test_basic_go_qrl_variant_uses_mode_and_inner_steps(self):
        task = WatcherTask(
            task_id="official_qrl_1q_SplitLatentMin8_M_s1000",
            mode="online",
            env_name="FetchReach",
            seed="1000",
            steps="200000",
            extra_args=(
                "+go_qrl_model_size=m "
                "agent.quasimetric_critic.model.encoder.branch_normalization=none "
                "agent.actor.model.input_mode=split_latent "
                "agent.actor.losses.min_dist.latent_goal_mode=min "
                "agent.actor.losses.min_dist.latent_goal_steps=8 "
                "agent.actor.losses.min_dist.latent_goal_search=direct"
            ),
        )

        self.assertEqual(task_variant(task), "GO-QRL+Min8")

    def test_parameter_count_uses_task_metadata_and_compact_units(self):
        self.assertEqual(compact_parameter_count(842_100), "842.1k")
        self.assertEqual(compact_parameter_count(2_112_257), "2.1m")
        self.assertEqual(compact_parameter_count(1_300_000_000), "1.3b")

        task = WatcherTask(
            task_id="run_Base_s1000",
            mode="offline",
            env_name="maze2d-umaze-v1",
            seed="1000",
            steps="40000",
            params="2112257",
        )
        self.assertEqual(task_parameter_count(task), "2.1m")
        task.params = "11.6M"
        self.assertEqual(task_parameter_count(task), "11.6m")

    def test_parameter_metadata_does_not_change_execution_fingerprint(self):
        task = RunnerTask(
            task_id="run_Base_s1000",
            mode="offline",
            env_name="maze2d-umaze-v1",
            seed="1000",
            steps="40000",
            params="2.1m",
            extra_args="agent.num_critics=1",
        )
        fingerprint = task_fingerprint(task)
        task.params = "2112257"
        self.assertEqual(task_fingerprint(task), fingerprint)

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

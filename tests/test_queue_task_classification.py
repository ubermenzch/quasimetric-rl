import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from tools.run_qrl_queue import ActiveJob
from tools.run_qrl_queue import build_command
from tools.run_qrl_queue import GpuState
from tools.run_qrl_queue import Task as RunnerTask
from tools.run_qrl_queue import command_env
from tools.run_qrl_queue import cleanup_completed_task_checkpoints
from tools.run_qrl_queue import ensure_task_submission_statuses
from tools.run_qrl_queue import gpu_accepts_more_jobs
from tools.run_qrl_queue import kill_illegal_user_gpu_jobs
from tools.run_qrl_queue import mark_finished
from tools.run_qrl_queue import output_finished
from tools.run_qrl_queue import output_started
from tools.run_qrl_queue import read_status
from tools.run_qrl_queue import reconcile_running_statuses
from tools.run_qrl_queue import requeue_cuda_oom_status
from tools.run_qrl_queue import requeue_existing_transient_failures
from tools.run_qrl_queue import task_uses_goal_set_objective
from tools.run_qrl_queue import task_fingerprint
from tools.run_qrl_queue import task_training_args
from tools.run_qrl_queue import sync_finished_outputs
from tools.run_qrl_queue import update_running_gpu_memory_peaks
from tools.run_qrl_queue import write_status
from tools.run_qrl_queue import write_task_manifest
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
    def test_manifest_temp_file_does_not_mark_output_as_started(self):
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            (output_dir / ".qrl_task.json.tmp").write_text("")

            self.assertFalse(output_started(output_dir))

            (output_dir / "checkpoint.pth").write_text("checkpoint")
            self.assertTrue(output_started(output_dir))

    def test_finalizing_checkpoint_does_not_mark_output_finished(self):
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            finalizing = output_dir / (
                "checkpoint_env00200000_opt00190000_finalizing.pth"
            )
            finalizing.touch()

            self.assertFalse(output_finished(output_dir))

            finalizing.rename(output_dir / (
                "checkpoint_env00200000_opt00190000_final.pth"
            ))
            self.assertTrue(output_finished(output_dir))

    def test_command_env_limits_all_cpu_thread_pools(self):
        env = command_env({"CPU_THREADS_PER_TASK": "4"}, "2")

        self.assertEqual(env["QRL_CPU_THREADS_PER_TASK"], "4")
        for variable in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            with self.subTest(variable=variable):
                self.assertEqual(env[variable], "4")

    def test_command_env_rejects_nonpositive_cpu_thread_limit(self):
        with self.assertRaisesRegex(
            ValueError, "CPU_THREADS_PER_TASK must be positive"
        ):
            command_env({"CPU_THREADS_PER_TASK": "0"}, "2")

    def test_illegal_gpu_cleanup_only_manages_allowed_gpus(self):
        apps = [
            {
                "gpu": "5",
                "pid": "5005",
                "process_name": ".venv/bin/python",
                "used_memory": "710",
            },
            {
                "gpu": "6",
                "pid": "6006",
                "process_name": ".venv/bin/python",
                "used_memory": "710",
            },
        ]
        config = {
            "KILL_ILLEGAL_USER_GPU_JOBS": "1",
            "DISALLOWED_GPU_TERMINATE_GRACE_SECONDS": "0",
        }

        with mock.patch(
            "tools.run_qrl_queue.is_queue_user_process", return_value=True
        ), mock.patch(
            "tools.run_qrl_queue.terminate_pid", return_value="terminated"
        ) as terminate_pid:
            killed_any = kill_illegal_user_gpu_jobs(
                config,
                apps,
                legal_pids=set(),
                allowed_gpus=[str(gpu) for gpu in range(6)],
                dry_run=False,
            )

        self.assertTrue(killed_any)
        terminate_pid.assert_called_once_with("5005", config, 0.0)

    def test_illegal_gpu_cleanup_does_nothing_outside_allowed_gpus(self):
        apps = [{
            "gpu": "7",
            "pid": "7007",
            "process_name": ".venv/bin/python",
            "used_memory": "710",
        }]

        with mock.patch(
            "tools.run_qrl_queue.is_queue_user_process"
        ) as is_queue_user_process, mock.patch(
            "tools.run_qrl_queue.terminate_pid"
        ) as terminate_pid:
            killed_any = kill_illegal_user_gpu_jobs(
                {"KILL_ILLEGAL_USER_GPU_JOBS": "1"},
                apps,
                legal_pids=set(),
                allowed_gpus=[str(gpu) for gpu in range(6)],
                dry_run=False,
            )

        self.assertFalse(killed_any)
        is_queue_user_process.assert_not_called()
        terminate_pid.assert_not_called()

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
            "gcbc": "GCSL",
            "gcsl": "GCSL",
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

    def test_model_size_presets_report_one_critic(self):
        common = dict(
            task_id="ablation_without_critic_metadata",
            mode="online",
            env_name="FetchPush",
            seed="1000",
            steps="200000",
        )
        for group in (
            "+go_qrl_model_size=m",
            "+qrl_model_size=m",
            "+base_model_size=m",
        ):
            with self.subTest(group=group):
                task = WatcherTask(extra_args=group, **common)
                self.assertEqual(task_critic_count(task), "1")

    def test_unknown_model_size_preset_does_not_invent_a_critic_count(self):
        task = WatcherTask(
            task_id="ablation_without_critic_metadata",
            mode="online",
            env_name="FetchPush",
            seed="1000",
            steps="200000",
            extra_args="+go_qrl_model_size=unknown",
        )

        self.assertEqual(task_critic_count(task), "2")

    def test_explicit_critic_count_overrides_model_size_preset(self):
        task = WatcherTask(
            task_id="ablation_without_critic_metadata",
            mode="online",
            env_name="FetchPush",
            seed="1000",
            steps="200000",
            extra_args="+go_qrl_model_size=m agent.num_critics=3",
        )

        self.assertEqual(task_critic_count(task), "3")

    def test_task_without_critic_metadata_uses_legacy_default(self):
        task = WatcherTask(
            task_id="legacy_task",
            mode="offline",
            env_name="maze2d-umaze-v1",
            seed="1000",
            steps="40000",
        )

        self.assertEqual(task_critic_count(task), "2")

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

    def test_dynamics_factorial_variants_are_distinct(self):
        cases = (
            ("A01", False, "iqe", "relu", "S0-IQE-ReLU"),
            ("A02", False, "iqe", "leaky_relu", "S0-IQE-Leaky"),
            ("A03", False, "mse", "relu", "S0-MSE-ReLU"),
            ("A04", False, "mse", "leaky_relu", "S0-MSE-Leaky"),
            ("A05", False, "iqe_mse", "relu", "S0-Hybrid-ReLU"),
            ("A06", False, "iqe_mse", "leaky_relu", "S0-Hybrid-Leaky"),
            ("A07", True, "iqe", "relu", "S1-IQE-ReLU"),
            ("A08", True, "iqe", "leaky_relu", "S1-IQE-Leaky"),
            ("A09", True, "mse", "relu", "S1-MSE-ReLU"),
            ("A10", True, "mse", "leaky_relu", "S1-MSE-Leaky"),
            ("A11", True, "iqe_mse", "relu", "S1-Hybrid-ReLU"),
            ("A12", True, "iqe_mse", "leaky_relu", "S1-Hybrid-Leaky"),
        )
        for code, separate, distance, activation, label in cases:
            with self.subTest(code=code):
                task = WatcherTask(
                    task_id=(
                        f"ablation_GO-QRL+Max4-dynfac_{code}-{label}-M_"
                        "200k_fetchpush_online_s1000"
                    ),
                    mode="online",
                    env_name="FetchPush",
                    seed="1000",
                    steps="200000",
                    extra_args=(
                        "+go_qrl_model_size=m "
                        "agent.actor.losses.min_dist.latent_goal_mode=max "
                        "agent.actor.losses.min_dist.latent_goal_steps=4 "
                        "agent.quasimetric_critic.losses."
                        f"separate_latent_dynamics={str(separate).lower()} "
                        "agent.quasimetric_critic.losses.latent_dynamics."
                        f"distance={distance} "
                        "agent.quasimetric_critic.model.quasimetric_model."
                        f"projector_activation={activation}"
                    ),
                )
                self.assertEqual(
                    task_variant(task), f"GO-QRL+Max4/{code}:{label}"
                )
                self.assertEqual(task_critic_count(task), "1")

    def test_legacy_dynamics_factorial_variant_has_no_ablation_code(self):
        task = WatcherTask(
            task_id=(
                "ablation_GO-QRL+Max4-dynfac_S1-Hybrid-Leaky-M_100k_"
                "antnavigate_v4_s1000"
            ),
            mode="online",
            env_name="AntNavigate-v4",
            seed="1000",
            steps="100000",
            extra_args=(
                "+go_qrl_model_size=m "
                "agent.quasimetric_critic.losses.separate_latent_dynamics=true "
                "agent.quasimetric_critic.losses.latent_dynamics.distance=iqe_mse "
                "agent.quasimetric_critic.model.quasimetric_model."
                "projector_activation=leaky_relu"
            ),
        )

        self.assertEqual(
            task_variant(task), "GO-QRL+Max4/S1-Hybrid-Leaky"
        )

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

    def test_checkpoint_cleanup_flag_is_queue_metadata(self):
        task = RunnerTask(
            task_id="run_Base_s1000",
            mode="offline",
            env_name="maze2d-umaze-v1",
            seed="1000",
            steps="40000",
            extra_args="agent.num_critics=1",
        )
        fingerprint = task_fingerprint(task)
        task.extra_args += " queue.delete_checkpoints_after_completion=true"

        self.assertEqual(task_fingerprint(task), fingerprint)
        self.assertEqual(task_training_args(task), ["agent.num_critics=1"])

    def test_checkpoint_cleanup_flag_is_not_passed_to_hydra(self):
        task = RunnerTask(
            task_id="run_Base_s1000",
            mode="offline",
            env_name="maze2d-umaze-v1",
            seed="1000",
            steps="40000",
            extra_args=(
                "agent.num_critics=1 "
                "queue.delete_checkpoints_after_completion=true"
            ),
        )
        with TemporaryDirectory() as directory:
            command, _, _, _ = build_command(
                {
                    "RESULTS_ROOT": directory,
                    "RESUME_IF_POSSIBLE": "0",
                },
                task,
                "0",
            )

        self.assertIn("agent.num_critics=1", command)
        self.assertFalse(any(arg.startswith("queue.") for arg in command))

    def test_completed_online_task_can_delete_checkpoints_automatically(self):
        task = RunnerTask(
            task_id="cleanup_task",
            mode="online",
            env_name="FetchPush",
            seed="1000",
            steps="200000",
            extra_args="queue.delete_checkpoints_after_completion=true",
        )
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            checkpoint_payloads = {
                "checkpoint_env00200000_opt00190000.pth": "full",
                "agent_checkpoint_env00200000_opt00190000.pth": "agent",
                "selected_best_agent.pth": "selected",
            }
            for name, payload in checkpoint_payloads.items():
                (output_dir / name).write_text(payload)
            (output_dir / "eval.log").write_text('{"succ_rate": 0.8}\n')
            (output_dir / "test.log").write_text('{"succ_rate": 0.7}\n')
            (output_dir / "best_checkpoint.json").write_text(json.dumps({
                "validation": {"succ_rate": 0.8},
                "test": {"succ_rate": 0.7},
                "selected_model": "selected_best_agent.pth",
            }))
            (output_dir / "COMPLETE").touch()

            cleaned = cleanup_completed_task_checkpoints(task, output_dir)

            self.assertTrue(cleaned)
            self.assertFalse(any(output_dir.glob("*.pth")))
            for name in ("eval.log", "test.log", "best_checkpoint.json", "COMPLETE"):
                self.assertTrue((output_dir / name).exists())
            marker = json.loads(
                (output_dir / "CHECKPOINTS_DELETED").read_text()
            )
            self.assertEqual(marker["task_id"], task.task_id)
            self.assertEqual(marker["checkpoint_count"], len(checkpoint_payloads))
            self.assertEqual(
                marker["checkpoint_bytes"],
                sum(len(payload) for payload in checkpoint_payloads.values()),
            )

    def test_checkpoint_cleanup_waits_for_final_evaluation(self):
        task = RunnerTask(
            task_id="cleanup_task",
            mode="online",
            env_name="FetchPush",
            seed="1000",
            steps="200000",
            extra_args="queue.delete_checkpoints_after_completion=true",
        )
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            checkpoint = output_dir / "checkpoint_env00200000_opt00190000.pth"
            checkpoint.write_text("checkpoint")
            (output_dir / "test.log").write_text('{"succ_rate": 0.7}\n')

            self.assertFalse(
                cleanup_completed_task_checkpoints(task, output_dir)
            )
            self.assertTrue(checkpoint.exists())
            self.assertFalse((output_dir / "CHECKPOINTS_DELETED").exists())

            (output_dir / "COMPLETE").touch()
            self.assertFalse(
                cleanup_completed_task_checkpoints(task, output_dir)
            )
            self.assertTrue(checkpoint.exists())

    def test_offline_complete_marker_alone_does_not_trigger_cleanup(self):
        task = RunnerTask(
            task_id="offline_cleanup_task",
            mode="offline",
            env_name="maze2d-umaze-v1",
            seed="1000",
            steps="40000",
            extra_args="queue.delete_checkpoints_after_completion=true",
        )
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            checkpoint = output_dir / "checkpoint_00001_00001_final.pth"
            checkpoint.write_text("checkpoint")
            (output_dir / "COMPLETE").touch()

            self.assertFalse(
                cleanup_completed_task_checkpoints(task, output_dir)
            )
            self.assertTrue(checkpoint.exists())
            self.assertFalse((output_dir / "CHECKPOINTS_DELETED").exists())

    def test_completed_task_accepts_cleanup_flag_added_later(self):
        task = RunnerTask(
            task_id="cleanup_task",
            mode="online",
            env_name="FetchPush",
            seed="1000",
            steps="200000",
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir = root / "results" / task.task_id
            status_dir = root / "status"
            output_dir.mkdir(parents=True)
            write_task_manifest(output_dir, task)
            write_status(
                status_dir,
                task,
                "DONE",
                {
                    "output_dir": str(output_dir),
                    "completion_evidence": "COMPLETE",
                },
            )
            checkpoint = output_dir / "selected_best_agent.pth"
            checkpoint.write_text("selected")
            (output_dir / "test.log").write_text('{"succ_rate": 0.7}\n')
            (output_dir / "best_checkpoint.json").write_text(json.dumps({
                "validation": {"succ_rate": 0.8},
                "test": {"succ_rate": 0.7},
                "selected_model": checkpoint.name,
            }))
            (output_dir / "COMPLETE").touch()
            original_fingerprint = task_fingerprint(task)
            task.extra_args = "queue.delete_checkpoints_after_completion=true"

            sync_finished_outputs(
                {"RESULTS_ROOT": str(root / "results")},
                [task],
                status_dir,
            )

            status = read_status(status_dir, task.task_id)
            self.assertEqual(status["state"], "DONE")
            self.assertEqual(status["task_fingerprint"], original_fingerprint)
            self.assertFalse(checkpoint.exists())
            self.assertTrue((output_dir / "CHECKPOINTS_DELETED").exists())

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

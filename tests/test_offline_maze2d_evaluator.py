import argparse
import io
import json
import multiprocessing as mp
import os
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import gym
import numpy as np
import torch

from tools.evaluate_offline_maze2d import (
    ALL_CHECKPOINT_NUM_EPISODES,
    CHILD_OUTPUT_LOG_ENV,
    EvaluationTask,
    ProcessEnvPool,
    SelectionCriterion,
    checkpoint_models_are_identical,
    evaluation_seed_range,
    main as evaluator_main,
    expand_result_dirs_for_training_seeds,
    find_evaluation_checkpoints,
    format_cpu_cores,
    make_evaluation_tasks,
    parse_gpu_ids,
    parse_selection_criteria,
    parse_training_seeds,
    resolve_num_episodes,
    rollout_episodes,
    select_checkpoint,
    select_best_summaries_per_run,
    split_cpu_cores,
    validate_result_dir_training_seeds,
)


def emit_native_child_output():
    os.write(1, b"native child stdout\n")
    os.write(2, b"native child stderr\n")


class CheckpointSelectionTest(unittest.TestCase):
    def test_selects_agent_steps_but_never_rolling_resume(self):
        with TemporaryDirectory() as temp_dir:
            result_dir = Path(temp_dir)
            agent_10k = result_dir / "agent_checkpoint_step00010000.pth"
            agent_20k = result_dir / "agent_checkpoint_step00020000.pth"
            rolling = result_dir / "checkpoint_resume_latest.pth"
            malformed = result_dir / "checkpoint_broken.pth"
            for path in (agent_10k, agent_20k, rolling, malformed):
                path.touch()

            self.assertEqual(select_checkpoint(result_dir, "10000"), agent_10k)
            self.assertEqual(select_checkpoint(result_dir, "latest"), agent_20k)

            final = result_dir / "checkpoint_00184_00104_final.pth"
            final.touch()
            self.assertEqual(select_checkpoint(result_dir, "final"), final)
            self.assertEqual(select_checkpoint(result_dir, "latest"), final)

    def test_ignores_malformed_and_rolling_checkpoints(self):
        with TemporaryDirectory() as temp_dir:
            result_dir = Path(temp_dir)
            (result_dir / "checkpoint_resume_latest.pth").touch()
            (result_dir / "checkpoint_broken.pth").touch()

            with self.assertRaisesRegex(FileNotFoundError, "No evaluation checkpoint"):
                select_checkpoint(result_dir, "latest")

    def test_finds_every_archival_checkpoint_but_not_rolling_or_malformed_files(self):
        with TemporaryDirectory() as temp_dir:
            result_dir = Path(temp_dir)
            expected_names = [
                "checkpoint_00001_00002.pth",
                "agent_checkpoint_step00010000.pth",
                "agent_checkpoint_step00020000.pth",
                "checkpoint_00003_00004_final.pth",
            ]
            ignored_names = ["checkpoint_resume_latest.pth", "checkpoint_broken.pth"]
            for name in expected_names + ignored_names:
                (result_dir / name).touch()

            checkpoints = find_evaluation_checkpoints(result_dir)

            self.assertEqual([path.name for path in checkpoints], expected_names)
            tasks = make_evaluation_tasks([result_dir], all_checkpoints=True)
            self.assertEqual([task.checkpoint for task in tasks], checkpoints)

    def test_all_checkpoint_mode_defaults_to_1000_but_accepts_an_override(self):
        self.assertEqual(
            resolve_num_episodes(None, all_checkpoints=True),
            ALL_CHECKPOINT_NUM_EPISODES,
        )
        self.assertEqual(resolve_num_episodes(None, all_checkpoints=False), 100)
        self.assertEqual(resolve_num_episodes(500, all_checkpoints=True), 500)

    def test_deduplicates_identical_last_agent_checkpoint_and_final(self):
        with TemporaryDirectory() as temp_dir:
            result_dir = Path(temp_dir)
            agent_10k = result_dir / "agent_checkpoint_step00010000.pth"
            agent_20k = result_dir / "agent_checkpoint_step00020000.pth"
            periodic_20k = result_dir / "checkpoint_00002_00003.pth"
            final = result_dir / "checkpoint_00003_00004_final.pth"
            torch.save(
                {
                    "optim_steps": 10_000,
                    "agent": {"actor.weight": torch.tensor([1.0])},
                },
                agent_10k,
            )
            final_state = {
                "optim_steps": 20_000,
                "agent": {"actor.weight": torch.tensor([2.0])},
            }
            torch.save(final_state, agent_20k)
            torch.save(final_state, periodic_20k)
            torch.save(final_state, final)

            self.assertTrue(checkpoint_models_are_identical(agent_20k, final))
            checkpoints = find_evaluation_checkpoints(
                result_dir,
                deduplicate_final=True,
            )

            self.assertEqual([path.name for path in checkpoints], [agent_10k.name, final.name])

    def test_keeps_same_step_checkpoints_when_agent_weights_differ(self):
        with TemporaryDirectory() as temp_dir:
            result_dir = Path(temp_dir)
            agent = result_dir / "agent_checkpoint_step00020000.pth"
            final = result_dir / "checkpoint_00003_00004_final.pth"
            torch.save(
                {
                    "optim_steps": 20_000,
                    "agent": {"actor.weight": torch.tensor([1.0])},
                },
                agent,
            )
            torch.save(
                {
                    "optim_steps": 20_000,
                    "agent": {"actor.weight": torch.tensor([2.0])},
                },
                final,
            )

            self.assertFalse(checkpoint_models_are_identical(agent, final))
            checkpoints = find_evaluation_checkpoints(
                result_dir,
                deduplicate_final=True,
            )

            self.assertEqual([path.name for path in checkpoints], [agent.name, final.name])

    def test_ignores_checkpoint_parts_not_used_for_policy_evaluation(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            left = root / "agent_checkpoint_step00020000.pth"
            right = root / "checkpoint_00003_00004_final.pth"
            shared = {
                "actor.weight": torch.tensor([1.0]),
                "critics.0.encoder.weight": torch.tensor([2.0]),
            }
            torch.save(
                {
                    "optim_steps": 20_000,
                    "agent": {
                        **shared,
                        "critics.0.latent_dynamics.weight": torch.tensor([3.0]),
                    },
                },
                left,
            )
            torch.save(
                {
                    "optim_steps": 20_000,
                    "agent": {
                        **shared,
                        "critics.0.latent_dynamics.weight": torch.tensor([4.0]),
                    },
                },
                right,
            )

            self.assertTrue(checkpoint_models_are_identical(left, right))


class BestPerRunSelectionTest(unittest.TestCase):
    def test_parses_ordered_selection_criteria(self):
        self.assertEqual(
            parse_selection_criteria(
                "success_rate:max, first_success_step_success_only_mean:min"
            ),
            [
                SelectionCriterion("success_rate", "max"),
                SelectionCriterion("first_success_step_success_only_mean", "min"),
            ],
        )
        for value in (
            "",
            "success_rate",
            "unknown:max",
            "success_rate:up",
            "success_rate:max,success_rate:min",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_selection_criteria(value)

    def test_selects_one_best_checkpoint_per_scheme_and_seed_run(self):
        criteria = parse_selection_criteria(
            "success_rate:max,first_success_step_success_only_mean:min"
        )
        summaries = [
            {
                "task_id": "max4-s1000-early",
                "run_id": "max4-s1000",
                "result_dir": "/max4/s1000",
                "seed": 1000,
                "checkpoint": "/max4/early.pth",
                "success_rate": 0.8,
                "first_success_step_success_only_mean": 10.0,
            },
            {
                "task_id": "max4-s1000-final",
                "run_id": "max4-s1000",
                "result_dir": "/max4/s1000",
                "seed": 1000,
                "checkpoint": "/max4/final.pth",
                "success_rate": 0.9,
                "first_success_step_success_only_mean": 30.0,
            },
            {
                "task_id": "min4-s1000-early",
                "run_id": "min4-s1000",
                "result_dir": "/min4/s1000",
                "seed": 1000,
                "checkpoint": "/min4/early.pth",
                "success_rate": 0.7,
                "first_success_step_success_only_mean": 15.0,
            },
            {
                "task_id": "min4-s1000-final",
                "run_id": "min4-s1000",
                "result_dir": "/min4/s1000",
                "seed": 1000,
                "checkpoint": "/min4/final.pth",
                "success_rate": 0.8,
                "first_success_step_success_only_mean": 25.0,
            },
            {
                "task_id": "max4-s1001-empty",
                "run_id": "max4-s1001",
                "result_dir": "/max4/s1001",
                "seed": 1001,
                "checkpoint": "/max4/s1001-empty.pth",
                "success_rate": None,
                "first_success_step_success_only_mean": None,
            },
            {
                "task_id": "max4-s1001-zero",
                "run_id": "max4-s1001",
                "result_dir": "/max4/s1001",
                "seed": 1001,
                "checkpoint": "/max4/s1001-zero.pth",
                "success_rate": 0.0,
                "first_success_step_success_only_mean": None,
            },
        ]

        selected = select_best_summaries_per_run(summaries, criteria)

        self.assertEqual(
            [row["task_id"] for row in selected],
            ["max4-s1000-final", "max4-s1001-zero", "min4-s1000-final"],
        )
        self.assertEqual([row["selection_candidate_count"] for row in selected], [2, 2, 2])
        self.assertEqual(
            [row["selection_run_id"] for row in selected],
            ["max4-s1000", "max4-s1001", "min4-s1000"],
        )

    def test_second_split_is_adjacent_and_non_overlapping(self):
        self.assertEqual(evaluation_seed_range(1000, 500, 0), (1000, 1499))
        self.assertEqual(evaluation_seed_range(1000, 500, 1), (1500, 1999))
        self.assertEqual(evaluation_seed_range(1000, 1000, 1), (2000, 2999))

    def test_two_stage_main_runs_test_split_and_writes_complete_results(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result_dir = root / "run_s7"
            result_dir.mkdir()
            args = argparse.Namespace(
                result_dirs=[str(result_dir)],
                checkpoint="final",
                all_checkpoints=True,
                num_episodes=2,
                select_best_per_run=True,
                selection_criteria=(
                    "success_rate:max,first_success_step_success_only_mean:min"
                ),
                num_envs=1,
                num_workers=0,
                max_episode_steps=0,
                seed=1000,
                training_seeds=None,
                device="cpu",
                gpus=None,
                action_mode="mode",
                goal_mode="target_zero",
                success_radius=0.5,
                out_dir=str(root / "output"),
                prefix="two_stage",
                log_file=None,
            )

            def make_summary(name, success_rate, first_success, split_name):
                checkpoint = result_dir / f"{name}.pth"
                return {
                    "task_id": name,
                    "run_id": "run_s7",
                    "result_dir": str(result_dir),
                    "env_name": "fake-maze",
                    "seed": 7,
                    "evaluation_split": split_name,
                    "episode_seed_start": 1000 if split_name == "selection" else 1002,
                    "episode_seed_end": 1001 if split_name == "selection" else 1003,
                    "checkpoint": str(checkpoint),
                    "checkpoint_name": checkpoint.name,
                    "return_mean": 0.0,
                    "normalized_score_x100_mean": None,
                    "success_rate": success_rate,
                    "time_at_goal_mean": 0.0,
                    "first_success_step_mean": first_success,
                    "first_success_step_success_only_mean": first_success,
                }

            selection_summaries = [
                make_summary("candidate_a", 0.5, 2.0, "selection"),
                make_summary("candidate_b", 0.75, 3.0, "selection"),
            ]
            test_summaries = [make_summary("candidate_b", 0.6, 2.5, "test")]
            pending_results = iter([selection_summaries, test_summaries])

            def run_tasks_side_effect(
                evaluation_tasks,
                evaluation_args,
                _gpu_ids,
                _details_path,
                progress_callback,
            ):
                progress_callback(
                    len(evaluation_tasks) * evaluation_args.num_episodes
                )
                return next(pending_results)

            frontend_stdout = io.StringIO()
            frontend_stderr = io.StringIO()
            with (
                redirect_stdout(frontend_stdout),
                redirect_stderr(frontend_stderr),
                patch(
                    "tools.evaluate_offline_maze2d.parse_args",
                    return_value=args,
                ),
                patch(
                    "tools.evaluate_offline_maze2d.make_evaluation_tasks",
                    return_value=[
                        EvaluationTask(result_dir, result_dir / "candidate_a.pth"),
                        EvaluationTask(result_dir, result_dir / "candidate_b.pth"),
                    ],
                ),
                patch(
                    "tools.evaluate_offline_maze2d.run_evaluation_tasks",
                    side_effect=run_tasks_side_effect,
                ) as run_tasks,
            ):
                evaluator_main()

            self.assertEqual(frontend_stdout.getvalue(), "")
            frontend_text = frontend_stderr.getvalue()
            self.assertIn("total evaluation [selection]", frontend_text)
            self.assertIn("6/6 episodes", frontend_text)
            self.assertNotIn("selection phase: episode seeds", frontend_text)
            self.assertEqual(run_tasks.call_count, 2)
            test_call = run_tasks.call_args_list[1]
            self.assertEqual(test_call.args[1].seed, 1002)
            self.assertEqual(len(test_call.args[0]), 1)
            self.assertEqual(
                test_call.args[0][0].checkpoint.name,
                "candidate_b.pth",
            )

            selected = json.loads(
                (root / "output/two_stage_selected_best.json").read_text()
            )
            self.assertEqual(selected[0]["task_id"], "candidate_b")
            complete = json.loads(
                (root / "output/two_stage_all_results.json").read_text()
            )
            self.assertEqual(complete["selection_episode_seed_range"], [1000, 1001])
            self.assertEqual(complete["test_episode_seed_range"], [1002, 1003])
            self.assertEqual(complete["num_selection_candidates"], 2)
            self.assertEqual(complete["num_selected_models"], 1)
            log_path = root / "output/two_stage.log"
            self.assertEqual(complete["output_files"]["log"], str(log_path))
            log_text = log_path.read_text()
            self.assertIn("total workload: 3 evaluation(s), 6 episode(s)", log_text)
            self.assertIn("selection phase: episode seeds 1000-1001", log_text)
            self.assertIn("test phase: episode seeds 1002-1003", log_text)


class ParallelResourceAllocationTest(unittest.TestCase):
    def test_parses_comma_or_space_separated_gpu_indices(self):
        self.assertEqual(parse_gpu_ids("0,2,cuda:5"), [0, 2, 5])
        self.assertEqual(parse_gpu_ids("1 3"), [1, 3])

    def test_rejects_invalid_or_duplicate_gpu_indices(self):
        for value in ("", "-1", "0,0", "gpu0"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_gpu_ids(value)

    def test_splits_cpu_cores_evenly_without_losing_affinity_ids(self):
        self.assertEqual(
            split_cpu_cores([2, 4, 6, 8, 10, 12, 14, 16], 3),
            [[2, 4, 6], [8, 10, 12], [14, 16]],
        )
        self.assertEqual(split_cpu_cores([2, 4], 3), [[2], [4], [2]])

    def test_formats_cpu_core_ranges(self):
        self.assertEqual(format_cpu_cores([0, 1, 2, 4, 7, 8]), "0-2,4,7-8")

    def test_spawned_native_stdout_and_stderr_are_redirected_to_log(self):
        with TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "child.log"
            previous = os.environ.get(CHILD_OUTPUT_LOG_ENV)
            os.environ[CHILD_OUTPUT_LOG_ENV] = str(log_path)
            try:
                process = mp.get_context("spawn").Process(
                    target=emit_native_child_output
                )
                process.start()
            finally:
                if previous is None:
                    os.environ.pop(CHILD_OUTPUT_LOG_ENV, None)
                else:
                    os.environ[CHILD_OUTPUT_LOG_ENV] = previous
            process.join(timeout=10)

            self.assertEqual(process.exitcode, 0)
            log_text = log_path.read_text()
            self.assertIn("native child stdout", log_text)
            self.assertIn("native child stderr", log_text)


class MultipleTrainingSeedsTest(unittest.TestCase):
    def test_parses_training_seeds_and_rejects_invalid_values(self):
        self.assertEqual(parse_training_seeds("1000, 1001 1003"), [1000, 1001, 1003])
        for value in ("", "-1", "1000,1000", "seed1000"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_training_seeds(value)

    def test_expands_placeholder_and_existing_seed_token(self):
        self.assertEqual(
            expand_result_dirs_for_training_seeds(
                ["runs/maze_s1000", "runs/ant_s{seed}_final"],
                [1001, 1003],
            ),
            [
                "runs/maze_s1001",
                "runs/maze_s1003",
                "runs/ant_s1001_final",
                "runs/ant_s1003_final",
            ],
        )
        self.assertEqual(
            expand_result_dirs_for_training_seeds(
                ["runs/model_s1000_20260722"],
                [1002],
            ),
            ["runs/model_s1002_20260722"],
        )

    def test_requires_a_seed_marker_in_result_directory(self):
        with self.assertRaisesRegex(ValueError, "no .* placeholder or sNNN"):
            expand_result_dirs_for_training_seeds(["runs/model"], [1000, 1001])

    def test_validates_training_seeds_from_configs(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result_dirs = [root / "run_s1000", root / "run_s1001"]
            for training_seed, result_dir in zip((1000, 1001), result_dirs):
                result_dir.mkdir()
                (result_dir / "config.yaml").write_text(f"seed: {training_seed}\n")

            validate_result_dir_training_seeds(result_dirs, [1000, 1001])
            with self.assertRaisesRegex(ValueError, "No result directory.*1002"):
                validate_result_dir_training_seeds(result_dirs, [1000, 1001, 1002])


class FakeMazeEnv:
    max_episode_steps = 4

    def __init__(self):
        self.action_space = gym.spaces.Box(
            low=np.array([-0.5, -0.5], dtype=np.float32),
            high=np.array([0.5, 0.5], dtype=np.float32),
            dtype=np.float32,
        )
        self._seed = 0
        self._step = 0
        self._position = 0.0
        self._target = 1.0

    def seed(self, seed):
        self._seed = int(seed)

    def reset(self):
        self._step = 0
        self._position = 0.0
        self._target = 1.0 + 0.25 * (self._seed % 3)
        return self._observation()

    def step(self, action):
        self._step += 1
        self._position += float(action[0])
        distance = abs(self._position - self._target)
        reward = float(distance <= 0.05)
        truncated = self._step >= self.max_episode_steps
        return self._observation(), reward, False, truncated, {}

    def get_target(self):
        return np.array([self._target, 0.0], dtype=np.float32)

    @staticmethod
    def get_normalized_score(episode_return):
        return episode_return / FakeMazeEnv.max_episode_steps

    def close(self):
        pass

    def _observation(self):
        return np.array([self._position, 0.0, 0.0, 0.0], dtype=np.float32)


class FakeDataset:
    kind = "fake"
    name = "fake-maze"

    @staticmethod
    def create_env():
        return FakeMazeEnv()


class FakeDistribution:
    def __init__(self, action):
        self.mode = action
        self.mean = action

    def sample(self):
        return self.mode


class FakeAgent:
    actor = object()

    def __init__(self):
        self.batch_sizes = []

    def act(self, obs, goal):
        self.batch_sizes.append(obs.shape[0])
        return FakeDistribution(goal[..., :2] - obs[..., :2])


def evaluator_args(num_envs, num_workers=0):
    return argparse.Namespace(
        num_episodes=9,
        num_envs=num_envs,
        num_workers=num_workers,
        seed=1000,
        goal_mode="target_zero",
        action_mode="mode",
        success_radius=0.05,
    )


def base_result():
    return {
        "task_id": "fake-task",
        "max_episode_steps": FakeMazeEnv.max_episode_steps,
    }


class BatchedRolloutTest(unittest.TestCase):
    def run_rollout(self, num_envs, progress_callback=None):
        details = io.StringIO()
        agent = FakeAgent()
        episodes, _elapsed = rollout_episodes(
            agent,
            FakeDataset(),
            evaluator_args(num_envs),
            torch.device("cpu"),
            details,
            base_result(),
            progress_callback,
        )
        detail_rows = [json.loads(line) for line in details.getvalue().splitlines()]
        return agent, episodes, detail_rows

    def test_batched_rollout_matches_serial_metrics_and_order(self):
        serial_agent, serial_episodes, serial_details = self.run_rollout(1)
        batched_agent, batched_episodes, batched_details = self.run_rollout(3)

        self.assertEqual(serial_episodes, batched_episodes)
        self.assertEqual(serial_details, batched_details)
        self.assertEqual(
            [row["episode_idx"] for row in batched_details],
            list(range(9)),
        )
        self.assertEqual(max(serial_agent.batch_sizes), 1)
        self.assertEqual(max(batched_agent.batch_sizes), 3)

    def test_reports_completed_episode_batches_to_global_progress(self):
        updates = []

        _agent, episodes, _details = self.run_rollout(3, updates.append)

        self.assertEqual(len(episodes), 9)
        self.assertEqual(sum(updates), 9)
        self.assertTrue(all(update > 0 for update in updates))


class ProcessEnvPoolTest(unittest.TestCase):
    def test_spawn_workers_reset_step_and_close(self):
        pool = ProcessEnvPool(num_envs=4, num_workers=2, env_factory=FakeMazeEnv)
        processes = list(pool.processes)
        try:
            resets = pool.reset({slot_id: 1000 + slot_id for slot_id in range(4)})
            self.assertEqual(set(resets), set(range(4)))
            actions = {
                slot_id: np.array([0.5, 0.0], dtype=np.float32)
                for slot_id in range(4)
            }
            steps = pool.step(actions)
            self.assertEqual(set(steps), set(range(4)))
            for obs, reward, terminated, truncated in steps.values():
                self.assertEqual(obs.shape, (4,))
                self.assertEqual(reward, 0.0)
                self.assertFalse(terminated)
                self.assertFalse(truncated)
        finally:
            pool.close()
        self.assertTrue(all(not process.is_alive() for process in processes))


if __name__ == "__main__":
    unittest.main()

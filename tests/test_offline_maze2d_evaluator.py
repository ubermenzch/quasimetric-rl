import argparse
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import gym
import numpy as np
import torch

from tools.evaluate_offline_maze2d import (
    ALL_CHECKPOINT_NUM_EPISODES,
    ProcessEnvPool,
    expand_result_dirs_for_training_seeds,
    find_evaluation_checkpoints,
    format_cpu_cores,
    make_evaluation_tasks,
    parse_gpu_ids,
    parse_training_seeds,
    resolve_num_episodes,
    rollout_episodes,
    select_checkpoint,
    split_cpu_cores,
    validate_result_dir_training_seeds,
)


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

    def test_all_checkpoint_mode_requires_1000_episodes(self):
        self.assertEqual(
            resolve_num_episodes(None, all_checkpoints=True),
            ALL_CHECKPOINT_NUM_EPISODES,
        )
        self.assertEqual(resolve_num_episodes(None, all_checkpoints=False), 100)
        with self.assertRaisesRegex(ValueError, "requires --num-episodes=1000"):
            resolve_num_episodes(999, all_checkpoints=True)


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
    def run_rollout(self, num_envs):
        details = io.StringIO()
        agent = FakeAgent()
        episodes, _elapsed = rollout_episodes(
            agent,
            FakeDataset(),
            evaluator_args(num_envs),
            torch.device("cpu"),
            details,
            base_result(),
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

import argparse
import io
import json
import unittest

import gym
import numpy as np
import torch

from tools.evaluate_offline_maze2d import ProcessEnvPool, rollout_episodes


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

import unittest
from types import SimpleNamespace

import gym
import numpy as np

from quasimetric_rl.data.online.gym_mujoco import GymMujocoGoalEnv


class FakePusherBackend(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Box(
            low=-2 * np.ones(7, dtype=np.float32),
            high=2 * np.ones(7, dtype=np.float32),
            dtype=np.float32,
        )
        self.data = SimpleNamespace(
            qpos=np.zeros(11, dtype=np.float64),
            qvel=np.zeros(11, dtype=np.float64),
        )
        self.body_coms = {
            'object': np.array([0.50, -0.05, -0.275], dtype=np.float64),
            'goal': np.array([0.45, -0.05, -0.323], dtype=np.float64),
            'tips_arm': np.array([0.30, -0.05, -0.275], dtype=np.float64),
        }

    def get_body_com(self, name):
        return self.body_coms[name]

    def reset(self, seed=None):
        del seed
        return np.zeros(23, dtype=np.float64)

    def step(self, action):
        del action
        return np.zeros(23, dtype=np.float64), 0.0, False, {}


class PusherGoalTest(unittest.TestCase):
    def setUp(self):
        self.backend = FakePusherBackend()
        self.env = GymMujocoGoalEnv(
            'Pusher-v4', backend_env=self.backend,
        )

    def test_semantic_goal_uses_object_center_height(self):
        packed = self.env.reset()

        np.testing.assert_allclose(
            packed['desired_goal'][:3],
            np.array([0.45, -0.05, -0.275], dtype=np.float32),
        )
        self.assertEqual(self.backend.body_coms['goal'][2], -0.323)

    def test_success_radius_is_five_centimeters_in_the_table_plane(self):
        self.env.reset()

        self.backend.body_coms['object'][0] = 0.499
        _, _, _, inside_info = self.env.step(np.zeros(7, dtype=np.float32))
        self.assertTrue(inside_info['is_success'])
        self.assertAlmostEqual(inside_info['goal_distance'], 0.049, places=6)

        self.backend.body_coms['object'][0] = 0.501
        _, _, _, outside_info = self.env.step(np.zeros(7, dtype=np.float32))
        self.assertFalse(outside_info['is_success'])
        self.assertAlmostEqual(outside_info['goal_distance'], 0.051, places=6)


if __name__ == '__main__':
    unittest.main()

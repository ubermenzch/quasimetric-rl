import unittest
from types import SimpleNamespace

import gym
import numpy as np

from quasimetric_rl.data.online.gym_mujoco import GymMujocoGoalEnv


class FakeAntBackend(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Box(
            low=-np.ones(8, dtype=np.float32),
            high=np.ones(8, dtype=np.float32),
            dtype=np.float32,
        )
        self.data = SimpleNamespace(
            qpos=np.arange(15, dtype=np.float64),
            qvel=np.arange(15, 29, dtype=np.float64),
        )

    def reset(self, seed=None):
        del seed
        return np.full(111, 99.0, dtype=np.float64)


class AntNavigateStateTest(unittest.TestCase):
    def test_state_contains_qpos_and_qvel_but_not_backend_contact_forces(self):
        env = GymMujocoGoalEnv(
            'AntNavigate-v4', backend_env=FakeAntBackend(),
        )

        packed = env.reset()
        expected = np.arange(29, dtype=np.float32)

        self.assertEqual(env.observation_space['observation'].shape, (29,))
        np.testing.assert_array_equal(packed['observation'], expected)
        np.testing.assert_array_equal(packed['achieved_goal'], expected)
        np.testing.assert_array_equal(
            packed['desired_goal'][2:], np.zeros(27, dtype=np.float32),
        )


if __name__ == '__main__':
    unittest.main()

import unittest
from unittest import mock

import numpy as np

from quasimetric_rl.data.d4rl import antmaze


class _FakeAntMazeEnv:
    max_episode_steps = 100

    def __init__(self, dataset):
        self._dataset = dataset

    def get_dataset(self):
        return self._dataset


class AntMazeDataTest(unittest.TestCase):
    def setUp(self):
        self.observations = np.arange(8 * 3, dtype=np.float32).reshape(8, 3)
        self.actions = np.arange(8 * 2, dtype=np.float32).reshape(8, 2)
        self.dataset = {
            'observations': self.observations,
            'actions': self.actions,
            'rewards': np.arange(8, dtype=np.float32),
            # AntMaze can mark multiple successful states without ending a rollout.
            'terminals': np.array([False, True, True, False, False, False, True, False]),
            'timeouts': np.array([False, False, False, True, False, False, False, True]),
        }

    def test_episode_slices_ignore_success_terminals(self):
        slices = list(antmaze.episode_slices(self.dataset, max_episode_steps=100))
        self.assertEqual([(slc.start, slc.stop) for slc in slices], [(0, 4), (4, 8)])

    def test_loader_uses_raw_observations_and_never_crosses_episode_boundary(self):
        env = _FakeAntMazeEnv(self.dataset)
        with mock.patch.object(antmaze, 'load_environment', return_value=env):
            episodes = list(antmaze.load_episodes_antmaze('fake-antmaze'))

        self.assertEqual(len(episodes), 2)
        np.testing.assert_array_equal(episodes[0].episode_lengths.numpy(), [3])
        np.testing.assert_array_equal(episodes[1].episode_lengths.numpy(), [3])
        np.testing.assert_array_equal(episodes[0].all_observations.numpy(), self.observations[:4])
        np.testing.assert_array_equal(episodes[1].all_observations.numpy(), self.observations[4:])
        np.testing.assert_array_equal(episodes[0].actions.numpy(), self.actions[:3])
        np.testing.assert_array_equal(episodes[1].actions.numpy(), self.actions[4:7])
        self.assertFalse(np.array_equal(
            episodes[0].all_observations[-1].numpy(),
            episodes[1].all_observations[0].numpy(),
        ))


if __name__ == '__main__':
    unittest.main()

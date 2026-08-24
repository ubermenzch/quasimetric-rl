import unittest
import copy

import gym
import numpy as np
import torch

from quasimetric_rl.data import BatchData, EnvSpec
from quasimetric_rl.modules.gcrl_baselines import GCRLBaselinesConf


class FactorizedGCSLTest(unittest.TestCase):
    def setUp(self):
        self.state_dim = 12
        self.action_dim = 20
        self.env_spec = EnvSpec(
            observation_space=gym.spaces.Box(
                -np.inf, np.inf, shape=(self.state_dim,), dtype=np.float32,
            ),
            observation_space_is_dict=True,
            action_space=gym.spaces.Box(
                -np.ones(self.action_dim, dtype=np.float32),
                np.ones(self.action_dim, dtype=np.float32),
                dtype=np.float32,
            ),
        )

    def make_agent(self):
        conf = copy.deepcopy(GCRLBaselinesConf())
        conf.gcbc.hidden_sizes = (16,)
        conf.gcbc.action_discretization = 'factorized'
        return conf.make('gcsl', env_spec=self.env_spec, goal_dims=(0, 1, 2))

    def test_factorized_policy_scales_linearly_with_action_dimension(self):
        agent, _ = self.make_agent()
        self.assertIsNone(agent.actor.action_table)
        self.assertEqual(tuple(agent.actor.action_bins.shape), (20, 3))
        self.assertEqual(agent.actor.backbone.output_size, 60)

        observation = torch.randn(5, self.state_dim)
        goal = torch.randn(5, 3)
        distribution = agent.actor(observation, goal)
        self.assertEqual(tuple(distribution.mean.shape), (5, self.action_dim))
        self.assertEqual(tuple(distribution.sample().shape), (5, self.action_dim))
        self.assertEqual(tuple(distribution.log_prob(torch.zeros(5, 20)).shape), (5,))
        self.assertTrue(torch.isfinite(distribution.log_prob(torch.zeros(5, 20))).all())

    def test_factorized_gcsl_loss_and_uniform_exploration_are_finite(self):
        agent, losses = self.make_agent()
        batch_size = 8
        batch = BatchData(
            observations=torch.randn(batch_size, self.state_dim),
            actions=torch.empty(batch_size, self.action_dim).uniform_(-1, 1),
            next_observations=torch.randn(batch_size, self.state_dim),
            future_observations=torch.randn(batch_size, self.state_dim),
            rewards=torch.zeros(batch_size),
            terminals=torch.zeros(batch_size, dtype=torch.bool),
            timeouts=torch.zeros(batch_size, dtype=torch.bool),
        )
        result = losses(agent, batch, optimize=False)
        self.assertTrue(torch.isfinite(result.loss))
        action = agent.actor.sample_uniform_action()
        self.assertEqual(tuple(action.shape), (self.action_dim,))
        self.assertTrue(torch.isfinite(action).all())


if __name__ == '__main__':
    unittest.main()

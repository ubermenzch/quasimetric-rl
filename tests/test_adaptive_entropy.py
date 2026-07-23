import unittest

import gym
import numpy as np
import torch

from quasimetric_rl.data.env_spec import EnvSpec
from quasimetric_rl.modules.actor.losses.min_dist import MinDistLoss


class AdaptiveEntropyLossTest(unittest.TestCase):
    @staticmethod
    def make_env_spec() -> EnvSpec:
        return EnvSpec(
            observation_space=gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(3,),
                dtype=np.float32,
            ),
            observation_space_is_dict=False,
            action_space=gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(2,),
                dtype=np.float32,
            ),
        )

    def make_loss(self, **kwargs) -> MinDistLoss:
        return MinDistLoss(
            env_spec=self.make_env_spec(),
            adaptive_entropy_regularizer=True,
            add_goal_as_future_state=False,
            **kwargs,
        )

    def test_default_target_matches_sac_action_dimension_heuristic(self):
        loss = self.make_loss(entropy_mc_samples=17)
        self.assertEqual(loss.target_entropy, -2.0)
        self.assertEqual(loss.entropy_mc_samples, 17)

    def test_explicit_target_entropy_is_used(self):
        loss = self.make_loss(target_entropy=-3.5)
        self.assertEqual(loss.target_entropy, -3.5)

    def test_temperature_increases_when_entropy_is_below_target(self):
        loss = self.make_loss()
        entropy = torch.tensor(-5.0, requires_grad=True)
        regularizer, alpha = loss.adaptive_entropy_loss(entropy)

        regularizer.backward()

        self.assertEqual(alpha.item(), 1.0)
        self.assertLess(entropy.grad.item(), 0.0)
        self.assertLess(loss.raw_entropy_weight.grad.item(), 0.0)

    def test_temperature_decreases_when_entropy_is_above_target(self):
        loss = self.make_loss()
        entropy = torch.tensor(0.0, requires_grad=True)
        regularizer, alpha = loss.adaptive_entropy_loss(entropy)

        regularizer.backward()

        self.assertEqual(alpha.item(), 1.0)
        self.assertLess(entropy.grad.item(), 0.0)
        self.assertGreater(loss.raw_entropy_weight.grad.item(), 0.0)


if __name__ == '__main__':
    unittest.main()

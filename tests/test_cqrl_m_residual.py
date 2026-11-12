import unittest

import gym
import numpy as np
from omegaconf import OmegaConf, SCMode
import torch

from online.dynamics_diagnostics import measure_critic
from quasimetric_rl.data import BatchData, EnvSpec
from quasimetric_rl.model_size import cqrl_agent_parameter_count, load_model_size_preset
from quasimetric_rl.modules import QRLConf
from quasimetric_rl.modules.quasimetric_critic.losses.latent_dynamics import LatentDynamicsLoss
from quasimetric_rl.modules.utils import ResidualMLP


class CQRLResidualMTest(unittest.TestCase):
    def test_actual_antnavigate_model_trains_with_both_dynamics_objectives(self):
        torch.set_num_threads(1)
        spec = EnvSpec(
            observation_space=gym.spaces.Box(-np.inf, np.inf, shape=(29,), dtype=np.float32),
            observation_space_is_dict=True,
            action_space=gym.spaces.Box(-1, 1, shape=(8,), dtype=np.float32),
        )
        for mse_weight in (0., 1.):
            with self.subTest(mse_weight=mse_weight):
                torch.manual_seed(1000)
                conf = OmegaConf.to_container(
                    OmegaConf.merge(OmegaConf.structured(QRLConf()), load_model_size_preset('cqrl', 'm_residual')),
                    structured_config_mode=SCMode.INSTANTIATE,
                )
                conf.quasimetric_critic.model.encoder.goal_dims = (0, 1)
                loss_conf = conf.quasimetric_critic.losses.latent_dynamics
                loss_conf.distance, loss_conf.mse_weight = 'iqe_mse', mse_weight
                conf.actor.losses.min_dist.latent_goal_mode = 'max'
                conf.actor.losses.min_dist.latent_goal_steps = 4
                agent, losses = conf.make(env_spec=spec, total_optim_steps=1)
                count = sum(p.numel() for p in agent.parameters() if p.requires_grad)
                self.assertEqual(count, 4_391_126)
                self.assertEqual(count, cqrl_agent_parameter_count(29, 8, 2, 'm_residual'))
                mlps = [m for m in agent.modules() if isinstance(m, ResidualMLP)]
                self.assertEqual(len(mlps), 5)
                for mlp in mlps:
                    self.assertEqual((mlp.depth, mlp.residual_block_size, len(mlp.blocks)), (2, 2, 1))
                    self.assertEqual(sum(isinstance(m, torch.nn.LayerNorm) for m in mlp.modules()), 3)
                    self.assertEqual(sum(isinstance(m, torch.nn.SiLU) for m in mlp.modules()), 3)
                    self.assertFalse(any(isinstance(m, torch.nn.ReLU) for m in mlp.modules()))
                data = BatchData(
                    observations=torch.randn(4, 29), actions=torch.rand(4, 8) * 1.6 - .8,
                    next_observations=torch.randn(4, 29), future_observations=torch.randn(4, 29),
                    rewards=torch.zeros(4), terminals=torch.zeros(4, dtype=torch.bool),
                    timeouts=torch.zeros(4, dtype=torch.bool),
                )
                result = losses(agent, data, optimize=True)
                self.assertTrue(torch.isfinite(result.loss))
                for mlp in mlps:
                    self.assertTrue(any(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in mlp.parameters()))
                    self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in mlp.parameters()))
                stats, _ = measure_critic(agent.critics[0], {
                    key: getattr(data, key) for key in ('observations', 'actions', 'next_observations')
                }, loss_conf)
                self.assertFalse(stats['nonfinite_tensors'])
                self.assertEqual(len(stats['projector_activations']), 6)
                effective = stats['gradients']['dynamics_parameters']['mse_effective_l2']
                self.assertEqual(effective > 0, mse_weight > 0)

    def test_small_residual_branch_matching_supports_registered_task_shapes(self):
        from tools.generate_extended_online_tasks import ENVIRONMENTS
        for env in ENVIRONMENTS:
            with self.subTest(environment=env.name):
                count = cqrl_agent_parameter_count(env.state_dim, env.action_dim, env.goal_dim, 'm_residual')
                self.assertGreater(count, 4_300_000)
                self.assertLess(count, 4_700_000)

    def test_zero_weight_does_not_propagate_nonfinite_disabled_error(self):
        for value in (float('inf'), float('nan')):
            loss = LatentDynamicsLoss(weight=.1, distance='iqe_mse', mse_weight=0, iqe_weight=1)
            disabled = torch.tensor(value, requires_grad=True)
            active = torch.tensor(3., requires_grad=True)
            result = loss._combine_distance_losses(mse=disabled, sq_dists=active)
            result.backward()
            self.assertEqual(result.item(), 3.)
            self.assertEqual(active.grad.item(), 1.)
            self.assertIsNone(disabled.grad)


if __name__ == '__main__':
    unittest.main()

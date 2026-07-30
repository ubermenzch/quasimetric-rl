import unittest

import gym
import numpy as np
import torch

from quasimetric_rl.data import BatchData, EnvSpec
from quasimetric_rl.modules.quasimetric_critic.losses import (
    CriticBatchInfo,
    QuasimetricCriticLosses,
)
from quasimetric_rl.modules.quasimetric_critic.losses.latent_dynamics import (
    LatentDynamicsLoss,
)
from quasimetric_rl.modules.quasimetric_critic.models import QuasimetricCritic
from quasimetric_rl.modules.quasimetric_critic.models.quasimetric_model import (
    QuasimetricModel,
)


def vector_env_spec() -> EnvSpec:
    return EnvSpec(
        observation_space=gym.spaces.Box(
            -np.inf, np.inf, shape=(5,), dtype=np.float32,
        ),
        observation_space_is_dict=False,
        action_space=gym.spaces.Box(
            -np.ones(2, dtype=np.float32),
            np.ones(2, dtype=np.float32),
            dtype=np.float32,
        ),
    )


def batch_data(batch_size: int = 8) -> BatchData:
    return BatchData(
        observations=torch.randn(batch_size, 5),
        actions=torch.empty(batch_size, 2).uniform_(-0.8, 0.8),
        next_observations=torch.randn(batch_size, 5),
        future_observations=torch.randn(batch_size, 5),
        rewards=torch.zeros(batch_size),
        terminals=torch.zeros(batch_size, dtype=torch.bool),
        timeouts=torch.zeros(batch_size, dtype=torch.bool),
    )


def make_critic(projector_activation: str = 'relu') -> QuasimetricCritic:
    conf = QuasimetricCritic.Conf()
    conf.encoder.arch = (8,)
    conf.encoder.latent_size = 4
    conf.quasimetric_model.projector_arch = (8,)
    conf.quasimetric_model.projector_activation = projector_activation
    conf.quasimetric_model.quasimetric_head_spec = 'iqe(dim=8,components=2)'
    conf.latent_dynamics.arch = (8,)
    return conf.make(env_spec=vector_env_spec())


def critic_batch_info(critic: QuasimetricCritic, data: BatchData) -> CriticBatchInfo:
    zx, zy = critic.encoder(
        torch.stack([data.observations, data.next_observations], dim=0)
    ).unbind(0)
    return CriticBatchInfo(
        critic=critic,
        zx=zx,
        zy=zy,
        px=critic.quasimetric_model.project(zx),
        py=critic.quasimetric_model.project(zy),
    )


class LatentDynamicsDistanceTest(unittest.TestCase):
    def test_supported_distance_modes_select_expected_values(self):
        mse = torch.tensor(2.0)
        sq_dists = torch.tensor(3.0)
        expected = {'iqe': 3.0, 'mse': 2.0, 'iqe_mse': 5.0}
        for distance, value in expected.items():
            with self.subTest(distance=distance):
                loss = LatentDynamicsLoss(weight=0.1, distance=distance)
                actual = loss._combine_distance_losses(
                    mse=mse,
                    sq_dists=sq_dists,
                )
                torch.testing.assert_close(actual, torch.tensor(value))

    def test_iqe_mse_applies_component_weights_before_outer_weight(self):
        critic = make_critic(projector_activation='leaky_relu')
        data = batch_data()
        info = critic_batch_info(critic, data)
        loss = LatentDynamicsLoss(
            weight=0.25,
            distance='iqe_mse',
            mse_weight=2.0,
            iqe_weight=3.0,
        )
        result = loss(data, info)
        expected = 0.25 * (
            2.0 * result.info['mse'] + 3.0 * result.info['sq_dists']
        )
        torch.testing.assert_close(result.loss, expected)


class SeparateLatentDynamicsTest(unittest.TestCase):
    def test_separate_step_only_backpropagates_to_latent_dynamics(self):
        critic = make_critic(projector_activation='leaky_relu')
        conf = QuasimetricCriticLosses.Conf()
        conf.separate_latent_dynamics = True
        conf.latent_dynamics.distance = 'iqe'
        losses = conf.make(critic, total_optim_steps=2)
        data = batch_data()
        info = critic_batch_info(critic, data)

        critic_optim_params = {
            id(parameter)
            for group in losses.critic_optim.param_groups
            for parameter in group['params']
        }
        dynamics_optim_params = {
            id(parameter)
            for group in losses.latent_dynamics_optim.param_groups
            for parameter in group['params']
        }
        self.assertFalse(critic_optim_params & dynamics_optim_params)
        self.assertEqual(
            dynamics_optim_params,
            {id(parameter) for parameter in critic.latent_dynamics.parameters()},
        )

        losses._forward_latent_dynamics_only(data, info, optimize=False)

        self.assertTrue(any(
            parameter.grad is not None
            and torch.count_nonzero(parameter.grad).item() > 0
            for parameter in critic.latent_dynamics.parameters()
        ))
        self.assertTrue(all(
            parameter.grad is None
            for parameter in critic.encoder.parameters()
        ))
        self.assertTrue(all(
            parameter.grad is None
            for parameter in critic.quasimetric_model.parameters()
        ))
        self.assertTrue(all(
            parameter.requires_grad
            for parameter in critic.quasimetric_model.parameters()
        ))


class ProjectorActivationTest(unittest.TestCase):
    def test_projector_defaults_to_relu(self):
        model = QuasimetricModel(
            input_size=4,
            projector_arch=(8,),
            quasimetric_head_spec='iqe(dim=8,components=2)',
        )
        self.assertIsInstance(model.projector.module[1], torch.nn.ReLU)

    def test_projector_supports_leaky_relu_with_configured_slope(self):
        model = QuasimetricModel(
            input_size=4,
            projector_arch=(8,),
            projector_activation='leaky_relu',
            projector_negative_slope=0.01,
            quasimetric_head_spec='iqe(dim=8,components=2)',
        )
        activation = model.projector.module[1]
        self.assertIsInstance(activation, torch.nn.LeakyReLU)
        self.assertEqual(activation.negative_slope, 0.01)


if __name__ == '__main__':
    unittest.main()

import unittest
from unittest import mock

import gym
import numpy as np
import torch

from online.trainer import TrainingOptimizationsConf
from quasimetric_rl.data.env_spec import EnvSpec
from quasimetric_rl.modules.actor.model import Actor
from quasimetric_rl.modules.optim import AdamWSpec
from quasimetric_rl.modules.quasimetric_critic.models.quasimetric_model import (
    QuasimetricModel,
)


class TrainingOptimizationsTest(unittest.TestCase):
    def test_defaults_preserve_legacy_execution(self):
        conf = TrainingOptimizationsConf()
        self.assertFalse(conf.enabled)
        self.assertFalse(conf.tf32)
        self.assertIsNone(conf.torch_amp_dtype)
        self.assertFalse(conf.fused_adamw)
        self.assertFalse(conf.compile_heavy_modules)
        self.assertEqual(conf.compile_mode, 'default')

    def test_gpu_optimizations_reject_cpu_execution(self):
        conf = TrainingOptimizationsConf(tf32=True)
        with self.assertRaisesRegex(RuntimeError, 'require a CUDA device'):
            conf.configure_backend(torch.device('cpu'))

    @mock.patch('quasimetric_rl.modules.optim.torch.optim.AdamW')
    def test_fused_adamw_is_forwarded_to_pytorch(self, adamw):
        parameter = torch.nn.Parameter(torch.ones(()))
        AdamWSpec.Conf(fused=True).make().create_optim([parameter])
        self.assertTrue(adamw.call_args.kwargs['fused'])

    def test_quasimetric_head_stays_fp32_under_bf16_autocast(self):
        model = QuasimetricModel(
            input_size=4,
            projector_arch=(8,),
            quasimetric_head_spec='iqe(dim=8,components=2)',
        )
        left = torch.randn(3, 4, requires_grad=True)
        right = torch.randn(3, 4, requires_grad=True)

        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            projected_left = model.project(left)
            projected_right = model.project(right)
            distance = model.forward_projected(
                projected_left, projected_right,
            )

        self.assertEqual(projected_left.dtype, torch.bfloat16)
        self.assertEqual(projected_right.dtype, torch.bfloat16)
        self.assertEqual(distance.dtype, torch.float32)
        distance.sum().backward()
        self.assertTrue(torch.isfinite(left.grad).all())
        self.assertTrue(torch.isfinite(right.grad).all())

    def test_action_distribution_stays_fp32_under_bf16_autocast(self):
        env_spec = EnvSpec(
            observation_space=gym.spaces.Box(
                -np.inf, np.inf, shape=(4,), dtype=np.float32,
            ),
            observation_space_is_dict=True,
            action_space=gym.spaces.Box(
                -1, 1, shape=(2,), dtype=np.float32,
            ),
        )
        actor = Actor(env_spec=env_spec, arch=(8,), input_mode='raw')
        observation = torch.randn(3, 4)
        goal = torch.randn(3, 4)

        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            distribution = actor(observation, goal)
            action = distribution.rsample()
            log_prob = distribution.log_prob(action)

        self.assertEqual(distribution.mean.dtype, torch.float32)
        self.assertEqual(action.dtype, torch.float32)
        self.assertEqual(log_prob.dtype, torch.float32)
        self.assertTrue(torch.isfinite(log_prob).all())


if __name__ == '__main__':
    unittest.main()

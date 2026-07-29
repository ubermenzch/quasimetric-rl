import unittest

import gym
import numpy as np
from omegaconf import OmegaConf, SCMode

from quasimetric_rl.data.env_spec import EnvSpec
from quasimetric_rl.model_size import (
    MODEL_SIZE_LEVELS,
    go_qrl_agent_parameter_count,
    load_model_size_preset,
    match_go_qrl_split_encoder,
    qrl_agent_parameter_count,
    select_qrl_model_size,
)
from quasimetric_rl.modules import QRLConf


class QRLModelSizePresetTest(unittest.TestCase):
    EXPECTED = {
        's': dict(
            qrl_name='QRL-S', go_name='GO-QRL-S', latent=128, width=512,
            qrl_params=2_112_261, go_params=2_206_469,
            goal_arch=(350, 396), non_goal_arch=(404, 350),
        ),
        'm': dict(
            qrl_name='QRL-M', go_name='GO-QRL-M', latent=256, width=768,
            qrl_params=4_150_533, go_params=4_439_301,
            goal_arch=(518, 609), non_goal_arch=(596, 543),
        ),
        'l': dict(
            qrl_name='QRL-L', go_name='GO-QRL-L', latent=512, width=1024,
            qrl_params=7_368_709, go_params=8_146_949,
            goal_arch=(768, 768), non_goal_arch=(768, 768),
        ),
    }

    def make_conf(self, family: str, level: str) -> QRLConf:
        merged = OmegaConf.merge(
            OmegaConf.structured(QRLConf()),
            load_model_size_preset(family, level),
        )
        return OmegaConf.to_container(
            merged, structured_config_mode=SCMode.INSTANTIATE,
        )

    def test_presets_have_expected_architectures_and_parameter_counts(self):
        env_spec = EnvSpec(
            observation_space=gym.spaces.Box(
                -np.inf, np.inf, shape=(4,), dtype=np.float32,
            ),
            observation_space_is_dict=True,
            action_space=gym.spaces.Box(
                -1, 1, shape=(2,), dtype=np.float32,
            ),
        )
        for level in MODEL_SIZE_LEVELS:
            with self.subTest(level=level):
                expected = self.EXPECTED[level]
                conf = self.make_conf('qrl', level)
                self.assertEqual(conf.model_size, expected['qrl_name'])
                encoder = conf.quasimetric_critic.model.encoder
                self.assertEqual(encoder.kind, 'standard')
                self.assertEqual(encoder.latent_size, expected['latent'])
                self.assertEqual(tuple(encoder.arch), (expected['width'],) * 2)
                self.assertEqual(conf.actor.model.input_mode, 'raw')
                agent, _ = conf.make(env_spec=env_spec, total_optim_steps=1)
                parameter_count = sum(
                    parameter.numel()
                    for parameter in agent.parameters()
                    if parameter.requires_grad
                )
                self.assertEqual(parameter_count, expected['qrl_params'])
                self.assertEqual(
                    qrl_agent_parameter_count(4, 2, level),
                    expected['qrl_params'],
                )

    def test_go_qrl_presets_match_qrl_encoder_budget(self):
        env_spec = EnvSpec(
            observation_space=gym.spaces.Box(
                -np.inf, np.inf, shape=(4,), dtype=np.float32,
            ),
            observation_space_is_dict=True,
            action_space=gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32),
        )
        for level in MODEL_SIZE_LEVELS:
            with self.subTest(level=level):
                expected = self.EXPECTED[level]
                conf = self.make_conf('go_qrl', level)
                conf.quasimetric_critic.model.encoder.goal_dims = (0, 1)
                self.assertEqual(conf.model_size, expected['go_name'])
                agent, _ = conf.make(env_spec=env_spec, total_optim_steps=1)
                encoder = agent.critics[0].encoder
                resolved = conf.quasimetric_critic.model.encoder
                self.assertEqual(tuple(resolved.goal_arch), expected['goal_arch'])
                self.assertEqual(
                    tuple(resolved.non_goal_arch),
                    expected['non_goal_arch'],
                )
                encoder_count = sum(parameter.numel() for parameter in encoder.parameters())
                plan = match_go_qrl_split_encoder(
                    4, 2, expected['latent'], (expected['width'],) * 2,
                )
                self.assertEqual(encoder_count, plan.qrl_encoder_parameters)
                parameter_count = sum(
                    parameter.numel()
                    for parameter in agent.parameters()
                    if parameter.requires_grad
                )
                self.assertEqual(parameter_count, expected['go_params'])
                self.assertEqual(
                    go_qrl_agent_parameter_count(4, 2, 2, level),
                    expected['go_params'],
                )

    def test_go_qrl_goal_branch_respects_parameter_floor(self):
        for level, latent, width in (
                ('s', 128, 512), ('m', 256, 768), ('l', 512, 1024)):
            for state_dim, goal_dim in ((113, 2), (40, 2), (40, 4)):
                with self.subTest(level=level, state_dim=state_dim, goal_dim=goal_dim):
                    plan = match_go_qrl_split_encoder(
                        state_dim, goal_dim, latent, (width, width),
                        min_goal_ratio=0.125,
                    )
                    self.assertTrue(plan.minimum_ratio_applied)
                    self.assertGreaterEqual(
                        plan.goal_encoder_parameters * 8,
                        plan.non_goal_encoder_parameters,
                    )
                    self.assertEqual(
                        plan.goal_encoder_parameters + plan.non_goal_encoder_parameters,
                        plan.qrl_encoder_parameters,
                    )

    def test_go_qrl_parameter_floor_holds_at_exact_dimension_boundary(self):
        for level, latent, width in (
                ('s', 128, 512), ('m', 256, 768), ('l', 512, 1024)):
            with self.subTest(level=level):
                plan = match_go_qrl_split_encoder(
                    state_dim=9,
                    goal_dim=1,
                    latent_size=latent,
                    reference_arch=(width, width),
                    min_goal_ratio=0.125,
                )
                self.assertFalse(plan.minimum_ratio_applied)
                self.assertGreaterEqual(
                    plan.goal_encoder_parameters * 8,
                    plan.non_goal_encoder_parameters,
                )

    def test_go_qrl_ceil_goal_latent_uses_effective_branch_ratio(self):
        cases = (
            # 128 * 16 / 37 = 55.35..., so the goal branch receives 56 dims.
            (37, 16, 56, False, 16 / 37),
            # 2:38 is below 1:8, so the effective goal share is 1 / (1 + 8).
            (40, 2, 15, True, 1 / 9),
            (113, 2, 15, True, 1 / 9),
        )
        for (
                state_dim, goal_dim, expected_goal_latent,
                expected_floor, target_share,
        ) in cases:
            with self.subTest(state_dim=state_dim, goal_dim=goal_dim):
                plan = match_go_qrl_split_encoder(
                    state_dim=state_dim,
                    goal_dim=goal_dim,
                    latent_size=128,
                    reference_arch=(512, 512),
                )
                self.assertEqual(plan.minimum_ratio_applied, expected_floor)
                self.assertEqual(plan.goal_latent_size, expected_goal_latent)
                self.assertEqual(
                    plan.non_goal_latent_size,
                    128 - expected_goal_latent,
                )
                self.assertEqual(
                    plan.goal_encoder_parameters + plan.non_goal_encoder_parameters,
                    plan.qrl_encoder_parameters,
                )
                actual_share = (
                    plan.goal_encoder_parameters / plan.qrl_encoder_parameters
                )
                self.assertLessEqual(
                    abs(actual_share - target_share) / target_share,
                    0.0025,
                )

    def test_selects_smallest_level_with_strict_half_latent_capacity(self):
        expected = {
            1: 's',
            63: 's',
            64: 'm',
            113: 'm',
            127: 'm',
            128: 'l',
            255: 'l',
        }
        for state_dim, level in expected.items():
            with self.subTest(state_dim=state_dim):
                self.assertEqual(select_qrl_model_size(state_dim), level)
        with self.assertRaisesRegex(ValueError, 'state_dim must be positive'):
            select_qrl_model_size(0)
        with self.assertRaisesRegex(ValueError, 'No QRL model-size level'):
            select_qrl_model_size(256)

    def test_reward_free_baseline_m_presets_merge_into_structured_config(self):
        expected = {
            'td_infonce': ('TD-InfoNCE-M', (512, 512, 512, 512), 16),
            'crl': ('CRL-M', (1152, 1152), 64),
            'gcbc': ('GCBC-M', (2304, 1728), None),
            'c_learning': ('C-Learning-M', (1184, 1184), None),
        }
        for family, (label, hidden_sizes, representation_dim) in expected.items():
            with self.subTest(family=family):
                merged = OmegaConf.merge(
                    OmegaConf.structured(QRLConf()),
                    load_model_size_preset(family, 'm'),
                )
                conf = OmegaConf.to_container(
                    merged, structured_config_mode=SCMode.INSTANTIATE,
                )
                baseline = getattr(conf.baselines, family)
                self.assertEqual(conf.model_size, label)
                self.assertEqual(tuple(baseline.hidden_sizes), hidden_sizes)
                if representation_dim is not None:
                    self.assertEqual(
                        baseline.representation_dim, representation_dim,
                    )

    def test_reward_free_m_presets_only_record_capacity_fields(self):
        allowed = {
            'td_infonce': {'hidden_sizes'},
            'crl': {'hidden_sizes'},
            'gcbc': {'hidden_sizes'},
            'c_learning': {'hidden_sizes'},
        }
        for family, expected_fields in allowed.items():
            with self.subTest(family=family):
                raw = OmegaConf.to_container(
                    load_model_size_preset(family, 'm'), resolve=True,
                )
                self.assertEqual(set(raw), {'model_size', 'baselines'})
                self.assertEqual(set(raw['baselines']), {family})
                self.assertEqual(
                    set(raw['baselines'][family]), expected_fields,
                )


if __name__ == '__main__':
    unittest.main()

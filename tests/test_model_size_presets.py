import unittest

import gym
import numpy as np
import torch
from omegaconf import OmegaConf, SCMode

from quasimetric_rl.data import BatchData
from quasimetric_rl.data.env_spec import EnvSpec
from quasimetric_rl.model_size import (
    QRL_MODEL_SIZE_LEVELS,
    SCALING_CRL_MODEL_SIZE_LEVELS,
    go_qrl_agent_parameter_count,
    load_model_size_preset,
    match_go_qrl_split_encoder,
    qrl_agent_parameter_count,
    scaling_crl_agent_parameter_count,
    select_qrl_model_size,
)
from quasimetric_rl.modules import QRLConf
from quasimetric_rl.modules.quasimetric_critic.models.quasimetric_model import (
    create_quasimetric_head_from_spec,
)
from quasimetric_rl.modules.utils import ResidualBlock, ResidualMLP


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
        # S/M retain their original two-layer plain MLPs.
        for level in ('s', 'm'):
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

    def test_qrl_l_is_plain_and_larger_tiers_follow_the_residual_scale(self):
        expected = {
            'xl': ('QRL-XL', 8, 4096, 128, 39_980_037),
            'xxl': ('QRL-XXL', 16, 8192, 256, 77_831_173),
            'xxxl': ('QRL-XXXL', 32, 16384, 512, 153_533_445),
        }
        self.assertEqual(
            QRL_MODEL_SIZE_LEVELS,
            ('s', 'm', 'l', 'xl', 'xxl', 'xxxl'),
        )

        conf = self.make_conf('qrl', 'l')
        encoder = conf.quasimetric_critic.model.encoder
        quasimetric = conf.quasimetric_critic.model.quasimetric_model
        dynamics = conf.quasimetric_critic.model.latent_dynamics
        actor = conf.actor.model
        self.assertEqual(conf.model_size, 'QRL-L')
        self.assertEqual(encoder.kind, 'standard')
        self.assertEqual(actor.input_mode, 'raw')
        self.assertEqual(encoder.latent_size, 512)
        self.assertEqual(tuple(encoder.arch), (1184,) * 4)
        self.assertEqual(tuple(quasimetric.projector_arch), (1184,) * 4)
        self.assertEqual(tuple(dynamics.arch), (1184,) * 4)
        self.assertEqual(tuple(actor.arch), (1184,) * 4)
        self.assertEqual(encoder.mlp_kind, 'plain')
        self.assertEqual(quasimetric.projector_mlp_kind, 'plain')
        self.assertEqual(dynamics.mlp_kind, 'plain')
        self.assertEqual(actor.mlp_kind, 'plain')
        self.assertEqual(quasimetric.projector_activation, 'relu')
        self.assertEqual(
            quasimetric.quasimetric_head_spec,
            'iqe(dim=2048,components=64)',
        )
        self.assertEqual(qrl_agent_parameter_count(4, 2, 'l'), 21_715_269)

        for level, (
                label, depth, head_dim, components, parameter_count,
        ) in expected.items():
            with self.subTest(level=level):
                conf = self.make_conf('qrl', level)
                encoder = conf.quasimetric_critic.model.encoder
                quasimetric = conf.quasimetric_critic.model.quasimetric_model
                dynamics = conf.quasimetric_critic.model.latent_dynamics
                actor = conf.actor.model

                self.assertEqual(conf.model_size, label)
                self.assertEqual(encoder.kind, 'standard')
                self.assertEqual(actor.input_mode, 'raw')
                self.assertEqual(encoder.latent_size, 512)
                self.assertEqual(tuple(encoder.arch), (1024,) * depth)
                self.assertEqual(tuple(quasimetric.projector_arch), (1024,) * depth)
                self.assertEqual(tuple(dynamics.arch), (1024,) * depth)
                self.assertEqual(tuple(actor.arch), (1024,) * depth)
                self.assertEqual(encoder.mlp_kind, 'residual')
                self.assertEqual(quasimetric.projector_mlp_kind, 'residual')
                self.assertEqual(dynamics.mlp_kind, 'residual')
                self.assertEqual(actor.mlp_kind, 'residual')
                self.assertEqual(quasimetric.projector_activation, 'silu')
                self.assertEqual(
                    quasimetric.quasimetric_head_spec,
                    f'iqe(dim={head_dim},components={components})',
                )
                self.assertEqual(head_dim // components, 32)
                self.assertEqual(
                    qrl_agent_parameter_count(4, 2, level),
                    parameter_count,
                )

    def test_qrl_l_actual_model_trains_and_matches_parameter_count(self):
        conf = self.make_conf('qrl', 'l')
        env_spec = EnvSpec(
            observation_space=gym.spaces.Box(
                -np.inf, np.inf, shape=(4,), dtype=np.float32,
            ),
            observation_space_is_dict=True,
            action_space=gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32),
        )
        agent, losses = conf.make(env_spec=env_spec, total_optim_steps=1)
        actual = sum(
            parameter.numel()
            for parameter in agent.parameters()
            if parameter.requires_grad
        )
        self.assertEqual(actual, qrl_agent_parameter_count(4, 2, 'l'))

        batch_size = 4
        data = BatchData(
            observations=torch.randn(batch_size, 4),
            actions=torch.empty(batch_size, 2).uniform_(-0.8, 0.8),
            next_observations=torch.randn(batch_size, 4),
            future_observations=torch.randn(batch_size, 4),
            rewards=torch.zeros(batch_size),
            terminals=torch.zeros(batch_size, dtype=torch.bool),
            timeouts=torch.zeros(batch_size, dtype=torch.bool),
        )
        result = losses(agent, data, optimize=True)
        self.assertTrue(torch.isfinite(result.loss))
        qrl_modules = (
            agent.critics[0].encoder,
            agent.critics[0].quasimetric_model.projector,
            agent.critics[0].latent_dynamics,
            agent.actor,
        )
        self.assertFalse(any(
            isinstance(
                child, (ResidualBlock, torch.nn.LayerNorm, torch.nn.SiLU),
            )
            for module in qrl_modules
            for child in module.modules()
        ))
        for module in qrl_modules:
            gradients = [parameter.grad for parameter in module.parameters()]
            self.assertTrue(any(gradient is not None for gradient in gradients))
            self.assertTrue(all(
                gradient is None or torch.isfinite(gradient).all()
                for gradient in gradients
            ))

    def test_go_qrl_presets_match_qrl_encoder_budget(self):
        env_spec = EnvSpec(
            observation_space=gym.spaces.Box(
                -np.inf, np.inf, shape=(4,), dtype=np.float32,
            ),
            observation_space_is_dict=True,
            action_space=gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32),
        )
        # S/M retain the original plain-MLP architecture and exact matching.
        for level in ('s', 'm'):
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

    def test_go_qrl_l_is_plain_and_larger_tiers_follow_the_residual_scale(self):
        expected = {
            'xl': ('GO-QRL-XL', 8, 4096, 128, 40_756_153, 731, 729),
            'xxl': ('GO-QRL-XXL', 16, 8192, 256, 78_613_803, 727, 727),
            'xxxl': ('GO-QRL-XXXL', 32, 16384, 512, 154_331_932, 725, 726),
        }

        conf = self.make_conf('go_qrl', 'l')
        encoder = conf.quasimetric_critic.model.encoder
        quasimetric = conf.quasimetric_critic.model.quasimetric_model
        dynamics = conf.quasimetric_critic.model.latent_dynamics
        actor = conf.actor.model
        self.assertEqual(conf.model_size, 'GO-QRL-L')
        self.assertEqual(encoder.kind, 'split')
        self.assertEqual(actor.input_mode, 'split_latent')
        self.assertEqual(encoder.latent_size, 512)
        self.assertEqual(tuple(encoder.arch), (1160,) * 4)
        self.assertEqual(tuple(quasimetric.projector_arch), (1160,) * 4)
        self.assertEqual(tuple(dynamics.arch), (1160,) * 4)
        self.assertEqual(tuple(actor.arch), (1160,) * 4)
        self.assertEqual(encoder.mlp_kind, 'plain')
        self.assertEqual(quasimetric.projector_mlp_kind, 'plain')
        self.assertEqual(dynamics.mlp_kind, 'plain')
        self.assertEqual(actor.mlp_kind, 'plain')
        self.assertEqual(quasimetric.projector_activation, 'relu')
        self.assertEqual(
            quasimetric.quasimetric_head_spec,
            'iqe(dim=2048,components=64)',
        )
        plan = encoder.resolve_go_qrl_branches(state_dim=4, goal_dim=2)
        self.assertEqual(tuple(encoder.goal_arch), (837,) * 4)
        self.assertEqual(tuple(encoder.non_goal_arch), (837,) * 4)
        self.assertLessEqual(
            abs(
                plan.goal_encoder_parameters
                + plan.non_goal_encoder_parameters
                - plan.qrl_encoder_parameters
            ) / plan.qrl_encoder_parameters,
            0.001,
        )
        self.assertEqual(
            go_qrl_agent_parameter_count(4, 2, 2, 'l'),
            21_824_679,
        )

        for level, (
                label, depth, head_dim, components, parameter_count,
                goal_width, non_goal_width,
        ) in expected.items():
            with self.subTest(level=level):
                conf = self.make_conf('go_qrl', level)
                encoder = conf.quasimetric_critic.model.encoder
                quasimetric = conf.quasimetric_critic.model.quasimetric_model
                dynamics = conf.quasimetric_critic.model.latent_dynamics
                actor = conf.actor.model

                self.assertEqual(conf.model_size, label)
                self.assertEqual(encoder.latent_size, 512)
                self.assertEqual(tuple(encoder.arch), (1024,) * depth)
                self.assertEqual(tuple(quasimetric.projector_arch), (1024,) * depth)
                self.assertEqual(tuple(dynamics.arch), (1024,) * depth)
                self.assertEqual(tuple(actor.arch), (1024,) * depth)
                self.assertEqual(encoder.mlp_kind, 'residual')
                self.assertEqual(quasimetric.projector_mlp_kind, 'residual')
                self.assertEqual(dynamics.mlp_kind, 'residual')
                self.assertEqual(actor.mlp_kind, 'residual')
                self.assertEqual(quasimetric.projector_activation, 'silu')
                self.assertEqual(
                    quasimetric.quasimetric_head_spec,
                    f'iqe(dim={head_dim},components={components})',
                )
                self.assertEqual(head_dim // components, 32)
                head = create_quasimetric_head_from_spec(
                    quasimetric.quasimetric_head_spec
                )
                self.assertEqual(head.input_size, head_dim)
                self.assertEqual(head.num_components, components)
                self.assertEqual(tuple(head.latent_2d_shape), (components, 32))
                x = torch.randn(2, head_dim, requires_grad=True)
                y = torch.randn(2, head_dim, requires_grad=True)
                distance = head(x, y)
                self.assertEqual(tuple(distance.shape), (2,))
                self.assertTrue(torch.isfinite(distance).all())
                distance.sum().backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                self.assertTrue(torch.isfinite(y.grad).all())

                plan = encoder.resolve_go_qrl_branches(state_dim=4, goal_dim=2)
                self.assertEqual(tuple(encoder.goal_arch), (goal_width,) * depth)
                self.assertEqual(
                    tuple(encoder.non_goal_arch), (non_goal_width,) * depth,
                )
                encoder_relative_error = abs(
                    plan.goal_encoder_parameters
                    + plan.non_goal_encoder_parameters
                    - plan.qrl_encoder_parameters
                ) / plan.qrl_encoder_parameters
                self.assertLessEqual(encoder_relative_error, 0.001)
                self.assertEqual(
                    go_qrl_agent_parameter_count(4, 2, 2, level),
                    parameter_count,
                )

    def test_go_qrl_l_actual_model_trains_and_matches_parameter_count(self):
        conf = self.make_conf('go_qrl', 'l')
        conf.quasimetric_critic.model.encoder.goal_dims = (0, 1)
        env_spec = EnvSpec(
            observation_space=gym.spaces.Box(
                -np.inf, np.inf, shape=(4,), dtype=np.float32,
            ),
            observation_space_is_dict=True,
            action_space=gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32),
        )
        agent, losses = conf.make(env_spec=env_spec, total_optim_steps=1)
        actual = sum(
            parameter.numel()
            for parameter in agent.parameters()
            if parameter.requires_grad
        )
        self.assertEqual(actual, go_qrl_agent_parameter_count(4, 2, 2, 'l'))

        batch_size = 4
        observations = torch.randn(batch_size, 4)
        data = BatchData(
            observations=observations,
            actions=torch.empty(batch_size, 2).uniform_(-0.8, 0.8),
            next_observations=torch.randn(batch_size, 4),
            future_observations=torch.randn(batch_size, 4),
            rewards=torch.zeros(batch_size),
            terminals=torch.zeros(batch_size, dtype=torch.bool),
            timeouts=torch.zeros(batch_size, dtype=torch.bool),
        )
        result = losses(agent, data, optimize=True)
        self.assertTrue(torch.isfinite(result.loss))
        go_qrl_modules = (
            agent.critics[0].encoder,
            agent.critics[0].quasimetric_model.projector,
            agent.critics[0].latent_dynamics,
            agent.actor,
        )
        self.assertFalse(any(
            isinstance(
                child, (ResidualBlock, torch.nn.LayerNorm, torch.nn.SiLU),
            )
            for module in go_qrl_modules
            for child in module.modules()
        ))
        for module in go_qrl_modules:
            gradients = [parameter.grad for parameter in module.parameters()]
            self.assertTrue(any(gradient is not None for gradient in gradients))
            self.assertTrue(all(
                gradient is None or torch.isfinite(gradient).all()
                for gradient in gradients
            ))

    def test_go_qrl_l_residual_backup_preserves_original_preset(self):
        conf = self.make_conf('go_qrl', 'l_residual')
        encoder = conf.quasimetric_critic.model.encoder
        quasimetric = conf.quasimetric_critic.model.quasimetric_model
        dynamics = conf.quasimetric_critic.model.latent_dynamics
        actor = conf.actor.model

        self.assertEqual(conf.model_size, 'GO-QRL-L-Residual')
        self.assertEqual(encoder.kind, 'split')
        self.assertEqual(actor.input_mode, 'split_latent')
        self.assertEqual(encoder.latent_size, 512)
        for arch in (
                encoder.arch,
                quasimetric.projector_arch,
                dynamics.arch,
                actor.arch,
        ):
            self.assertEqual(tuple(arch), (1024,) * 4)
        self.assertEqual(encoder.mlp_kind, 'residual')
        self.assertEqual(quasimetric.projector_mlp_kind, 'residual')
        self.assertEqual(dynamics.mlp_kind, 'residual')
        self.assertEqual(actor.mlp_kind, 'residual')
        self.assertEqual(quasimetric.projector_activation, 'silu')

        plan = encoder.resolve_go_qrl_branches(state_dim=4, goal_dim=2)
        self.assertEqual(tuple(encoder.goal_arch), (737,) * 4)
        self.assertEqual(tuple(encoder.non_goal_arch), (735,) * 4)
        self.assertLessEqual(
            abs(
                plan.goal_encoder_parameters
                + plan.non_goal_encoder_parameters
                - plan.qrl_encoder_parameters
            ) / plan.qrl_encoder_parameters,
            0.001,
        )
        self.assertEqual(
            go_qrl_agent_parameter_count(4, 2, 2, 'l_residual'),
            21_830_093,
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

    def test_residual_block_is_identity_when_its_dense_layers_are_zero(self):
        block = ResidualBlock(8, depth=4)
        with torch.no_grad():
            for module in block.modules():
                if isinstance(module, torch.nn.Linear):
                    module.weight.zero_()
                    module.bias.zero_()
        inputs = torch.randn(3, 8)
        torch.testing.assert_close(block(inputs), inputs)

    def test_residual_mlp_depth_and_parameter_count(self):
        for depth in (4, 8, 16, 32):
            with self.subTest(depth=depth):
                input_size, width, output_size = 7, 32, 5
                mlp = ResidualMLP(
                    input_size,
                    output_size,
                    hidden_sizes=(width,) * depth,
                    residual_block_size=4,
                )
                expected = (
                    (input_size + 1) * width + 2 * width
                    + depth * ((width + 1) * width + 2 * width)
                    + (width + 1) * output_size
                )
                self.assertEqual(len(mlp.blocks), depth // 4)
                self.assertIsInstance(mlp.output_layer, torch.nn.Linear)
                self.assertEqual(
                    sum(parameter.numel() for parameter in mlp.parameters()),
                    expected,
                )
                for block in mlp.blocks:
                    layers = tuple(block.module)
                    self.assertEqual(len(layers), 3 * mlp.residual_block_size)
                    for layer_index in range(mlp.residual_block_size):
                        linear, normalization, activation = layers[
                            3 * layer_index:3 * layer_index + 3
                        ]
                        self.assertIsInstance(linear, torch.nn.Linear)
                        self.assertIsInstance(normalization, torch.nn.LayerNorm)
                        self.assertIsInstance(activation, torch.nn.SiLU)
                self.assertEqual(
                    mlp(torch.randn(3, input_size)).shape,
                    (3, output_size),
                )

        with self.assertRaisesRegex(ValueError, 'must be divisible'):
            ResidualMLP(7, 5, hidden_sizes=(32,) * 6, residual_block_size=4)

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

    def test_reward_free_baseline_presets_merge_into_structured_config(self):
        expected = {
            'td_infonce': {
                'm': ('TD-InfoNCE-M', (512, 512, 512, 512), 16),
                'l': ('TD-InfoNCE-L', (1192, 1192, 1192, 1192), 16),
            },
            'crl': {
                'm': ('CRL-M', (1152, 1152), 64),
                'l': ('CRL-L', (1544, 1544, 1544, 1544), 64),
            },
            'gcsl': {
                'm': ('GCSL-M', (2304, 1728), None),
                'l': ('GCSL-L', (2680, 2680, 2680, 2680), None),
                'l_pusher': (
                    'GCSL-L-Pusher', (2336, 2344, 2344, 2344), None,
                ),
                'l_antnavigate': (
                    'GCSL-L-AntNavigate', (1808, 1808, 1800, 1800), None,
                ),
            },
            'c_learning': {
                'm': ('C-Learning-M', (1184, 1184), None),
                'l': ('C-Learning-L', (1544, 1544, 1544, 1544), None),
            },
        }
        for family, levels in expected.items():
            for level, (
                    label, hidden_sizes, representation_dim,
            ) in levels.items():
                with self.subTest(family=family, level=level):
                    merged = OmegaConf.merge(
                        OmegaConf.structured(QRLConf()),
                        load_model_size_preset(family, level),
                    )
                    conf = OmegaConf.to_container(
                        merged, structured_config_mode=SCMode.INSTANTIATE,
                    )
                    config_name = 'gcbc' if family == 'gcsl' else family
                    baseline = getattr(conf.baselines, config_name)
                    self.assertEqual(conf.model_size, label)
                    self.assertEqual(tuple(baseline.hidden_sizes), hidden_sizes)
                    if representation_dim is not None:
                        self.assertEqual(
                            baseline.representation_dim, representation_dim,
                        )

    def test_reward_free_presets_only_record_capacity_fields(self):
        allowed = {
            'td_infonce': {'hidden_sizes'},
            'crl': {'hidden_sizes'},
            'gcsl': {'hidden_sizes'},
            'c_learning': {'hidden_sizes'},
        }
        for family, expected_fields in allowed.items():
            levels = ('m', 'l')
            if family == 'gcsl':
                levels += ('l_pusher', 'l_antnavigate')
            for level in levels:
                with self.subTest(family=family, level=level):
                    raw = OmegaConf.to_container(
                        load_model_size_preset(family, level), resolve=True,
                    )
                    self.assertEqual(set(raw), {'model_size', 'baselines'})
                    config_name = 'gcbc' if family == 'gcsl' else family
                    self.assertEqual(set(raw['baselines']), {config_name})
                    self.assertEqual(
                        set(raw['baselines'][config_name]), expected_fields,
                    )

    def test_scaling_crl_presets_preserve_method_and_match_scale_budgets(self):
        expected = {
            'm': ('Scaling-CRL-M', 4, 595),
            'l': ('Scaling-CRL-L', 8, 944),
            'xl': ('Scaling-CRL-XL', 16, 916),
            'xxl': ('Scaling-CRL-XXL', 32, 901),
            'xxxl': ('Scaling-CRL-XXXL', 64, 894),
        }
        task_shapes = (
            (40, 5, 2),
            (25, 4, 3),
            (17, 5, 2),
            (20, 7, 3),
            (29, 8, 2),
        )
        nominal_targets = {
            'm': (4_397_835, 4_385_545, 4_381_707, 4_394_767, 4_396_305),
            'l': (21_670_317, 21_659_941, 21_651_644, 21_678_996, 21_668_891),
            'xl': (40_601_395, 40_580_499, 40_584_521, 40_600_566, 40_599_980),
            'xxl': (78_412_811, 78_449_907, 78_442_318, 78_459_798, 78_442_847),
            'xxxl': (154_100_303, 154_150_324, 154_116_725, 154_136_233, 154_098_910),
        }
        for level in SCALING_CRL_MODEL_SIZE_LEVELS:
            label, depth, width = expected[level]
            with self.subTest(level=level):
                raw = OmegaConf.to_container(
                    load_model_size_preset('scaling_crl', level), resolve=True,
                )
                self.assertEqual(set(raw), {'model_size', 'baselines'})
                self.assertEqual(raw['model_size'], label)
                self.assertEqual(set(raw['baselines']), {'scaling_crl'})
                self.assertEqual(
                    set(raw['baselines']['scaling_crl']), {'hidden_sizes'},
                )
                hidden = tuple(raw['baselines']['scaling_crl']['hidden_sizes'])
                self.assertEqual(hidden, (width,) * depth)

                for shape, target in zip(task_shapes, nominal_targets[level]):
                    actual = scaling_crl_agent_parameter_count(*shape, level)
                    self.assertLessEqual(abs(actual - target) / target, 0.003)

    def test_scaling_crl_parameter_formula_matches_instantiated_model(self):
        conf = self.make_conf('scaling_crl', 'm')
        conf.algorithm = 'scaling_crl'
        env_spec = EnvSpec(
            observation_space=gym.spaces.Box(
                -np.inf, np.inf, shape=(25,), dtype=np.float32,
            ),
            observation_space_is_dict=True,
            action_space=gym.spaces.Box(
                -1, 1, shape=(4,), dtype=np.float32,
            ),
        )
        agent, losses = conf.make(
            env_spec=env_spec,
            total_optim_steps=1,
            baseline_goal_dims=(3, 4, 5),
        )
        actual = sum(
            parameter.numel()
            for module in (agent, losses)
            for parameter in module.parameters()
            if parameter.requires_grad
        )
        self.assertEqual(
            actual, scaling_crl_agent_parameter_count(25, 4, 3, 'm'),
        )


if __name__ == '__main__':
    unittest.main()

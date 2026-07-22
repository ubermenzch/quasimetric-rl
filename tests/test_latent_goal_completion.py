import copy
import types
import unittest

import torch

from quasimetric_rl.data.env_spec.act_distn import BoxOutputLinearNormalization
from quasimetric_rl.modules import QRLConf
from quasimetric_rl.modules.actor import Actor
from quasimetric_rl.modules.actor.losses.min_dist import MinDistLoss
from quasimetric_rl.modules.quasimetric_critic.models.encoder import SplitEncoder
from tests.test_goal_set_distance import batch_data, vector_env_spec


class SquaredDistance(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.project_calls = 0

    def project(self, latent):
        self.project_calls += 1
        return latent

    def forward_projected(self, left, right):
        return (left - right).square().sum(dim=-1)

    def forward(self, left, right):
        return self.forward_projected(self.project(left), self.project(right))


class RecordingSquaredDistance(SquaredDistance):
    def __init__(self):
        super().__init__()
        self.right_inputs = []

    def forward_projected(self, left, right):
        self.right_inputs.append(right.detach().clone())
        return super().forward_projected(left, right)


def split_agent_conf(mode='none'):
    conf = copy.deepcopy(QRLConf())
    conf.num_critics = 1
    conf.actor.model.input_mode = 'split_latent'
    conf.actor.model.arch = (8,)
    conf.actor.losses.min_dist.adaptive_entropy_regularizer = False
    conf.actor.losses.min_dist.add_goal_as_future_state = False
    conf.actor.losses.min_dist.latent_goal_mode = mode
    conf.actor.losses.min_dist.latent_goal_steps = 8
    conf.actor.losses.min_dist.latent_goal_lr = 0.01
    conf.actor.losses.min_dist.latent_goal_keep_best = mode in ('min', 'max')
    encoder = conf.quasimetric_critic.model.encoder
    encoder.kind = 'split'
    encoder.latent_size = 4
    encoder.goal_dims = (0, 1)
    encoder.goal_arch = (8,)
    encoder.non_goal_arch = (8,)
    encoder.goal_latent_size = 2
    encoder.non_goal_latent_size = 2
    encoder.branch_normalization = 'rmsnorm'
    conf.quasimetric_critic.model.quasimetric_model.projector_arch = (8,)
    conf.quasimetric_critic.model.quasimetric_model.quasimetric_head_spec = 'l2(dim=4)'
    conf.quasimetric_critic.model.latent_dynamics.arch = (8,)
    return conf


LATENT_VARIANTS = (
    ('latent_base', 'standard', 'none'),
    ('split_zero', 'split', 'none'),
    ('split_latent_min8', 'split', 'min'),
    ('split_latent_max8', 'split', 'max'),
)


def latent_variant_conf(encoder_kind, mode):
    conf = split_agent_conf(mode)
    encoder_conf = conf.quasimetric_critic.model.encoder
    encoder_conf.kind = encoder_kind
    if encoder_kind == 'standard':
        conf.actor.model.input_mode = 'latent'
        encoder_conf.arch = (8,)
        encoder_conf.branch_normalization = 'none'
    return conf


class SplitEncoderTest(unittest.TestCase):
    def make_encoder(self, branch_normalization='none'):
        return SplitEncoder(
            env_spec=vector_env_spec(),
            goal_dims=(0, 1),
            goal_arch=(8,),
            non_goal_arch=(8,),
            goal_latent_size=2,
            non_goal_latent_size=2,
            branch_normalization=branch_normalization,
        )

    def test_branches_only_observe_their_own_coordinates(self):
        torch.manual_seed(3)
        encoder = self.make_encoder()
        reference = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        changed_goal = torch.tensor([[9.0, 8.0, 3.0, 4.0]])
        changed_non_goal = torch.tensor([[1.0, 2.0, 7.0, 6.0]])

        torch.testing.assert_close(
            encoder.encode_non_goal_part(reference),
            encoder.encode_non_goal_part(changed_goal),
        )
        torch.testing.assert_close(
            encoder.encode_goal_part(reference),
            encoder.encode_goal_part(changed_non_goal),
        )
        self.assertFalse(torch.equal(
            encoder.encode_goal_part(reference),
            encoder.encode_goal_part(changed_goal),
        ))
        self.assertFalse(torch.equal(
            encoder.encode_non_goal_part(reference),
            encoder.encode_non_goal_part(changed_non_goal),
        ))

    def test_actor_goal_has_an_exact_zero_non_goal_part(self):
        encoder = self.make_encoder()
        goal = torch.tensor([[1.0, 2.0, 99.0, -99.0]])
        encoded = encoder.encode_actor_goal(goal)
        goal_part, non_goal_part = encoder.split_latent(encoded)

        torch.testing.assert_close(goal_part, encoder.encode_goal_part(goal))
        torch.testing.assert_close(non_goal_part, torch.zeros_like(non_goal_part))
        self.assertEqual(tuple(encoded.shape), (1, 4))

    def test_rmsnorm_normalizes_each_branch_without_affine_parameters(self):
        encoder = self.make_encoder('rmsnorm')
        observations = torch.tensor([
            [1.0, 2.0, 3.0, 4.0],
            [-2.0, 0.5, 8.0, -1.0],
        ])
        goal_latent, non_goal_latent = encoder.split_latent(
            encoder(observations)
        )

        for latent in (goal_latent, non_goal_latent):
            rms = latent.square().mean(dim=-1).sqrt()
            torch.testing.assert_close(
                rms, torch.ones_like(rms), rtol=1e-5, atol=1e-5
            )
        self.assertEqual(
            list(encoder.goal_normalization.named_parameters()), []
        )
        self.assertEqual(
            list(encoder.non_goal_normalization.named_parameters()), []
        )


class LatentGoalAdamTest(unittest.TestCase):
    def make_loss_and_critic(self, mode):
        encoder = SplitEncoder(
            env_spec=vector_env_spec(),
            goal_dims=(0, 1),
            goal_arch=(),
            non_goal_arch=(),
            goal_latent_size=2,
            non_goal_latent_size=2,
        )
        critic = types.SimpleNamespace(
            encoder=encoder,
            quasimetric_model=SquaredDistance(),
        )
        loss = MinDistLoss(
            env_spec=vector_env_spec(),
            adaptive_entropy_regularizer=False,
            add_goal_as_future_state=False,
            latent_goal_mode=mode,
            latent_goal_steps=8,
            latent_goal_lr=0.01,
        )
        return loss, critic

    def optimize(self, mode, batch_size=1):
        loss, critic = self.make_loss_and_critic(mode)
        predicted = torch.tensor([[0.5, -0.25, 2.0, -3.0]]).expand(batch_size, -1)
        zero_goal = torch.tensor([[0.5, -0.25, 0.0, 0.0]]).expand(batch_size, -1)
        completed, diagnostics = loss._optimize_latent_goal(
            critic, predicted, zero_goal
        )
        return completed, diagnostics

    def test_min_decreases_and_max_increases_distance(self):
        min_goal, min_info = self.optimize('min')
        max_goal, max_info = self.optimize('max')

        self.assertGreater(min_info['latent_goal_improvement'].item(), 0)
        self.assertGreater(max_info['latent_goal_improvement'].item(), 0)
        self.assertGreater(min_goal[0, 2].item(), 0)
        self.assertLess(min_goal[0, 3].item(), 0)
        self.assertLess(max_goal[0, 2].item(), 0)
        self.assertGreater(max_goal[0, 3].item(), 0)

    def test_inner_adam_update_does_not_depend_on_batch_size(self):
        single, _ = self.optimize('min', batch_size=1)
        repeated, _ = self.optimize('min', batch_size=7)
        torch.testing.assert_close(
            repeated,
            single.expand_as(repeated),
            rtol=0,
            atol=0,
        )

    def test_frozen_prediction_is_projected_once_per_inner_loop(self):
        loss, critic = self.make_loss_and_critic('min')
        predicted = torch.tensor([[0.5, -0.25, 2.0, -3.0]])
        sampled_goal = torch.tensor([[0.5, -0.25, 1.0, -2.0]])

        loss._optimize_latent_goal(critic, predicted, sampled_goal)

        # One prediction, plus initial, each inner step, and final goal.
        self.assertEqual(
            critic.quasimetric_model.project_calls, loss.latent_goal_steps + 3
        )

    def test_inner_adam_starts_from_supplied_non_goal_latent(self):
        loss, critic = self.make_loss_and_critic('min')
        loss.latent_goal_steps = 1
        predicted = torch.tensor([[0.5, -0.25, 2.0, -3.0]])
        sampled_goal = torch.tensor([[0.5, -0.25, 1.0, -2.0]])

        completed, diagnostics = loss._optimize_latent_goal(
            critic, predicted, sampled_goal
        )

        torch.testing.assert_close(
            completed[..., 2:],
            torch.tensor([[1.01, -2.01]]),
            rtol=0,
            atol=1e-6,
        )
        torch.testing.assert_close(
            diagnostics['latent_goal_initial_norm'],
            torch.sqrt(torch.tensor(5.0)),
        )

    def test_rmsnorm_is_applied_to_every_inner_candidate(self):
        encoder = SplitEncoder(
            env_spec=vector_env_spec(),
            goal_dims=(0, 1),
            goal_arch=(),
            non_goal_arch=(),
            goal_latent_size=2,
            non_goal_latent_size=2,
            branch_normalization='rmsnorm',
        )
        distance = RecordingSquaredDistance()
        critic = types.SimpleNamespace(
            encoder=encoder,
            quasimetric_model=distance,
        )
        loss = MinDistLoss(
            env_spec=vector_env_spec(),
            adaptive_entropy_regularizer=False,
            add_goal_as_future_state=False,
            latent_goal_mode='max',
            latent_goal_steps=8,
            latent_goal_lr=0.01,
            latent_goal_keep_best=True,
        )
        sampled_goal = encoder(torch.tensor([[0.5, -0.25, 1.0, -2.0]]))
        predicted = torch.tensor([[0.5, -0.25, 2.0, -3.0]])

        completed, _diagnostics = loss._optimize_latent_goal(
            critic, predicted, sampled_goal
        )

        self.assertEqual(len(distance.right_inputs), 10)
        for candidate in distance.right_inputs:
            _goal, non_goal = encoder.split_latent(candidate)
            torch.testing.assert_close(
                non_goal.square().mean(dim=-1).sqrt(),
                torch.ones(1),
                rtol=1e-5,
                atol=1e-5,
            )
        _goal, final_non_goal = encoder.split_latent(completed)
        torch.testing.assert_close(
            final_non_goal.square().mean(dim=-1).sqrt(),
            torch.ones(1),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_bounded_residual_projects_every_inner_candidate(self):
        encoder = SplitEncoder(
            env_spec=vector_env_spec(),
            goal_dims=(0, 1),
            goal_arch=(),
            non_goal_arch=(),
            goal_latent_size=2,
            non_goal_latent_size=2,
        )
        distance = RecordingSquaredDistance()
        critic = types.SimpleNamespace(
            encoder=encoder,
            quasimetric_model=distance,
        )
        radius = 0.25
        loss = MinDistLoss(
            env_spec=vector_env_spec(),
            adaptive_entropy_regularizer=False,
            add_goal_as_future_state=False,
            latent_goal_mode='max',
            latent_goal_steps=8,
            latent_goal_lr=10.0,
            latent_goal_keep_best=True,
            latent_goal_search='bounded_residual',
            latent_goal_residual_radius=radius,
        )
        sampled_goal = torch.tensor([
            [0.5, -0.25, -3.0, -1.0],
            [0.5, -0.25, -1.0, 2.0],
            [0.5, -0.25, 2.0, 4.0],
            [0.5, -0.25, 6.0, 8.0],
        ])
        predicted = torch.zeros_like(sampled_goal)
        _goal, initial_non_goal = encoder.split_latent(sampled_goal)
        residual_scale = initial_non_goal.std(dim=0, unbiased=False)

        completed, diagnostics = loss._optimize_latent_goal(
            critic, predicted, sampled_goal
        )

        self.assertEqual(len(distance.right_inputs), 10)
        for candidate in [*distance.right_inputs, completed]:
            _goal, candidate_non_goal = encoder.split_latent(candidate)
            standardized_residual = (
                candidate_non_goal - initial_non_goal
            ) / residual_scale
            residual_rms = standardized_residual.square().mean(dim=-1).sqrt()
            self.assertLessEqual(residual_rms.max().item(), radius + 1e-6)
        torch.testing.assert_close(
            diagnostics['latent_goal_residual_scale_mean'],
            residual_scale.mean(),
        )
        self.assertLessEqual(
            diagnostics['latent_goal_last_residual_rms_max'].item(),
            radius + 1e-6,
        )
        self.assertLessEqual(
            diagnostics['latent_goal_final_residual_rms_max'].item(),
            radius + 1e-6,
        )
        self.assertGreaterEqual(
            diagnostics['latent_goal_final_inner_dist'].item() + 1e-6,
            diagnostics['latent_goal_initial_dist'].item(),
        )

    def test_bounded_residual_requires_latent_goal_optimization(self):
        with self.assertRaisesRegex(
                ValueError, 'requires latent_goal_mode=min or max'):
            MinDistLoss(
                env_spec=vector_env_spec(),
                adaptive_entropy_regularizer=False,
                add_goal_as_future_state=False,
                latent_goal_search='bounded_residual',
            )


class LatentGoalIntegrationTest(unittest.TestCase):
    def test_split_actor_requires_explicit_goal_latent_width(self):
        with self.assertRaisesRegex(ValueError, 'requires goal_latent_size'):
            Actor.Conf(input_mode='split_latent').make(
                env_spec=vector_env_spec(),
                latent_size=4,
            )

    def test_split_actor_receives_full_state_and_goal_branch_only(self):
        agent, _ = split_agent_conf().make(
            env_spec=vector_env_spec(), total_optim_steps=10
        )
        self.assertEqual(agent.actor.backbone.input_size, 6)

        data = batch_data()
        obs = data.observations
        goal = data.next_observations
        captured_inputs = []
        handle = agent.actor.backbone.register_forward_pre_hook(
            lambda _module, args: captured_inputs.append(args[0].detach().clone())
        )
        try:
            agent.act(obs, goal)
        finally:
            handle.remove()

        critic = agent.critics[0]
        expected = torch.cat([
            critic.encoder(obs),
            critic.encoder.encode_goal_part(goal),
        ], dim=-1)
        torch.testing.assert_close(captured_inputs[-1], expected)

    def test_split_actor_ignores_goal_non_goal_coordinates(self):
        agent, _ = split_agent_conf().make(
            env_spec=vector_env_spec(), total_optim_steps=10
        )
        obs = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        goal_a = torch.tensor([[0.8, 0.9, 1.0, 2.0]])
        goal_b = torch.tensor([[0.8, 0.9, -7.0, 11.0]])

        captured_inputs = []
        handle = agent.actor.backbone.register_forward_pre_hook(
            lambda _module, args: captured_inputs.append(args[0].detach().clone())
        )
        try:
            agent.act(obs, goal_a)
            agent.act(obs, goal_b)
        finally:
            handle.remove()
        torch.testing.assert_close(captured_inputs[0], captured_inputs[1], rtol=0, atol=0)

    def test_split_behavior_cloning_uses_goal_branch_only(self):
        conf = split_agent_conf()
        conf.actor.losses.behavior_cloning.weight = 1
        agent, losses = conf.make(
            env_spec=vector_env_spec(), total_optim_steps=10
        )
        data = batch_data()
        critic_infos = losses._make_critic_batch_infos(agent, data)

        captured_inputs = []
        handle = agent.actor.backbone.register_forward_pre_hook(
            lambda _module, args: captured_inputs.append(args[0].detach().clone())
        )
        try:
            result = losses.actor_loss.behavior_cloning(
                agent.actor, critic_infos, data
            )
        finally:
            handle.remove()

        critic = agent.critics[0]
        expected = torch.cat([
            critic.encoder(data.observations),
            critic.encoder.encode_goal_part(data.future_observations),
        ], dim=-1)
        self.assertTrue(torch.isfinite(result.loss))
        torch.testing.assert_close(captured_inputs[-1], expected)

    def test_split_minmax_starts_from_sampled_goal_non_goal_latent(self):
        for mode in ('min', 'max'):
            with self.subTest(mode=mode):
                agent, losses = split_agent_conf(mode).make(
                    env_spec=vector_env_spec(), total_optim_steps=10
                )
                data = batch_data()
                captured_inputs = []
                handle = agent.actor.backbone.register_forward_pre_hook(
                    lambda _module, args: captured_inputs.append(
                        args[0].detach().clone()
                    )
                )
                try:
                    result = losses(
                        agent, data, optimize=False, phase='actor'
                    )
                finally:
                    handle.remove()

                critic = agent.critics[0]
                sampled_goal = torch.roll(data.next_observations, 1, dims=0)
                expected_actor_input = torch.cat([
                    critic.encoder(data.observations),
                    critic.encoder.encode_goal_part(sampled_goal),
                ], dim=-1)
                initial_non_goal = critic.encoder.encode_non_goal_part(
                    sampled_goal
                )
                expected_norm = torch.linalg.vector_norm(
                    initial_non_goal, dim=-1
                ).mean()
                torch.testing.assert_close(
                    captured_inputs[-1], expected_actor_input
                )
                torch.testing.assert_close(
                    result.info['actor']['min_dist'][
                        'latent_goal_initial_norm_00'
                    ],
                    expected_norm,
                )
                info = result.info['actor']['min_dist']
                if mode == 'min':
                    self.assertLessEqual(
                        info['latent_goal_final_inner_dist_00'].item(),
                        info['latent_goal_initial_dist_00'].item() + 1e-6,
                    )
                    self.assertLessEqual(
                        info['latent_goal_final_inner_dist_00'].item(),
                        info['latent_goal_last_inner_dist_00'].item() + 1e-6,
                    )
                else:
                    self.assertGreaterEqual(
                        info['latent_goal_final_inner_dist_00'].item() + 1e-6,
                        info['latent_goal_initial_dist_00'].item(),
                    )
                    self.assertGreaterEqual(
                        info['latent_goal_final_inner_dist_00'].item() + 1e-6,
                        info['latent_goal_last_inner_dist_00'].item(),
                    )
                self.assertGreaterEqual(
                    info['latent_goal_best_step_00'].item(), 0
                )
                self.assertLessEqual(
                    info['latent_goal_best_step_max_00'].item(), 8
                )

    def test_bounded_residual_runs_end_to_end(self):
        conf = split_agent_conf('max')
        min_dist = conf.actor.losses.min_dist
        min_dist.latent_goal_search = 'bounded_residual'
        min_dist.latent_goal_residual_radius = 0.5
        min_dist.latent_goal_lr = 0.1
        conf.quasimetric_critic.model.encoder.branch_normalization = 'none'
        agent, losses = conf.make(
            env_spec=vector_env_spec(), total_optim_steps=10
        )

        result = losses(
            agent, batch_data(), optimize=False, phase='actor'
        )
        info = result.info['actor']['min_dist']

        self.assertTrue(torch.isfinite(result.loss))
        self.assertLessEqual(
            info['latent_goal_last_residual_rms_max_00'].item(), 0.5 + 1e-6
        )
        self.assertLessEqual(
            info['latent_goal_final_residual_rms_max_00'].item(), 0.5 + 1e-6
        )
        self.assertGreaterEqual(
            info['latent_goal_final_inner_dist_00'].item() + 1e-6,
            info['latent_goal_initial_dist_00'].item(),
        )

    def test_legacy_latent_minmax_keeps_zero_initialization(self):
        conf = split_agent_conf('min')
        conf.actor.model.input_mode = 'latent'
        conf.actor.losses.min_dist.latent_goal_keep_best = False
        conf.quasimetric_critic.model.encoder.branch_normalization = 'none'
        agent, losses = conf.make(
            env_spec=vector_env_spec(), total_optim_steps=10
        )
        result = losses(
            agent, batch_data(), optimize=False, phase='actor'
        )

        self.assertEqual(
            result.info['actor']['min_dist'][
                'latent_goal_initial_norm_00'
            ].item(),
            0,
        )

    def test_legacy_latent_actor_keeps_full_goal_width(self):
        conf = split_agent_conf()
        conf.actor.model.input_mode = 'latent'
        agent, _ = conf.make(env_spec=vector_env_spec(), total_optim_steps=10)
        self.assertEqual(agent.actor.backbone.input_size, 8)

    def test_all_latent_actor_variants_use_immutable_action_bounds(self):
        data = batch_data()
        goal = torch.roll(data.next_observations, 1, dims=0)

        for name, encoder_kind, mode in LATENT_VARIANTS:
            with self.subTest(variant=name):
                agent, _ = latent_variant_conf(encoder_kind, mode).make(
                    env_spec=vector_env_spec(), total_optim_steps=10
                )
                action_output = agent.actor.action_output

                self.assertIsInstance(
                    action_output, BoxOutputLinearNormalization
                )
                self.assertEqual(dict(action_output.named_buffers()), {})
                self.assertEqual(action_output._mean_values, (0.0, 0.0))
                self.assertEqual(action_output._half_len_values, (1.0, 1.0))

                torch.manual_seed(41)
                expected_action = agent.act(
                    data.observations, goal
                ).rsample()
                state = agent.state_dict()
                state['actor.action_output.mean'].fill_(99.0)
                state['actor.action_output.half_len'].zero_()
                with self.assertWarnsRegex(
                        RuntimeWarning,
                        'Ignoring checkpoint actor.action_output.half_len'):
                    agent.load_state_dict(state)
                torch.manual_seed(41)
                actual_action = agent.act(
                    data.observations, goal
                ).rsample()

                self.assertTrue(torch.equal(actual_action, expected_action))
                _, runtime_half_len = action_output._action_bounds_like(
                    torch.zeros(action_output.input_size)
                )
                self.assertEqual(runtime_half_len.tolist(), [1.0, 1.0])

    def test_all_latent_actor_variants_train_one_joint_step(self):
        for name, encoder_kind, mode in LATENT_VARIANTS:
            with self.subTest(variant=name):
                torch.manual_seed(7)
                agent, losses = latent_variant_conf(
                    encoder_kind, mode
                ).make(env_spec=vector_env_spec(), total_optim_steps=10)
                result = losses(
                    agent,
                    batch_data(),
                    optimize=True,
                    phase='all',
                )
                self.assertTrue(torch.isfinite(result.loss))

    def test_actor_backward_does_not_create_critic_gradients(self):
        torch.manual_seed(11)
        agent, losses = split_agent_conf('min').make(
            env_spec=vector_env_spec(), total_optim_steps=10
        )
        result = losses(
            agent,
            batch_data(),
            optimize=False,
            phase='actor',
        )

        self.assertTrue(torch.isfinite(result.loss))
        self.assertTrue(any(
            parameter.grad is not None
            for parameter in agent.actor.parameters()
        ))
        self.assertTrue(all(
            parameter.grad is None
            for parameter in agent.critics[0].parameters()
        ))
        info = result.info['actor']['min_dist']
        self.assertIn('latent_goal_initial_dist_00', info)
        self.assertIn('non_goal_latent_active_fraction_00', info)

    def test_actor_step_updates_actor_but_not_critic_or_encoder(self):
        torch.manual_seed(13)
        agent, losses = split_agent_conf('min').make(
            env_spec=vector_env_spec(), total_optim_steps=10
        )
        # Residual T is zero-initialized, so expose the action path as it would
        # be after T has taken training steps.
        final_dynamics_layer = agent.critics[0].latent_dynamics.module[-1]
        torch.nn.init.normal_(final_dynamics_layer.weight, std=0.05)
        actor_before = {
            name: parameter.detach().clone()
            for name, parameter in agent.actor.named_parameters()
        }
        critic_before = {
            name: parameter.detach().clone()
            for name, parameter in agent.critics[0].named_parameters()
        }

        losses(agent, batch_data(), optimize=True, phase='actor')

        self.assertTrue(any(
            not torch.equal(actor_before[name], parameter.detach())
            for name, parameter in agent.actor.named_parameters()
        ))
        for name, parameter in agent.critics[0].named_parameters():
            torch.testing.assert_close(
                parameter.detach(), critic_before[name], rtol=0, atol=0
            )
            self.assertIsNone(parameter.grad, name)

    def test_invalid_non_split_configuration_is_rejected(self):
        conf = split_agent_conf('min')
        conf.quasimetric_critic.model.encoder.kind = 'standard'
        with self.assertRaisesRegex(ValueError, 'encoder.kind=split'):
            conf.make(env_spec=vector_env_spec(), total_optim_steps=10)


if __name__ == '__main__':
    unittest.main()

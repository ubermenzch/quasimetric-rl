import copy
import types
import unittest

import torch

from quasimetric_rl.data.env_spec.act_distn import BoxOutputLinearNormalization
from quasimetric_rl.modules import QRLConf
from quasimetric_rl.modules.actor.losses.min_dist import MinDistLoss
from quasimetric_rl.modules.quasimetric_critic.models.encoder import SplitEncoder
from tests.test_goal_set_distance import batch_data, vector_env_spec


class SquaredDistance(torch.nn.Module):
    def forward(self, left, right):
        return (left - right).square().sum(dim=-1)


def split_agent_conf(mode='none'):
    conf = copy.deepcopy(QRLConf())
    conf.num_critics = 1
    conf.actor.model.input_mode = 'latent'
    conf.actor.model.arch = (8,)
    conf.actor.losses.min_dist.adaptive_entropy_regularizer = False
    conf.actor.losses.min_dist.add_goal_as_future_state = False
    conf.actor.losses.min_dist.latent_goal_mode = mode
    conf.actor.losses.min_dist.latent_goal_steps = 8
    conf.actor.losses.min_dist.latent_goal_lr = 0.01
    encoder = conf.quasimetric_critic.model.encoder
    encoder.kind = 'split'
    encoder.latent_size = 4
    encoder.goal_dims = (0, 1)
    encoder.goal_arch = (8,)
    encoder.non_goal_arch = (8,)
    encoder.goal_latent_size = 2
    encoder.non_goal_latent_size = 2
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
        encoder_conf.arch = (8,)
    return conf


class SplitEncoderTest(unittest.TestCase):
    def make_encoder(self):
        return SplitEncoder(
            env_spec=vector_env_spec(),
            goal_dims=(0, 1),
            goal_arch=(8,),
            non_goal_arch=(8,),
            goal_latent_size=2,
            non_goal_latent_size=2,
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


class LatentGoalIntegrationTest(unittest.TestCase):
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

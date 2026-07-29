import copy
from types import SimpleNamespace
import unittest

import gym
import numpy as np
import torch

from quasimetric_rl.data import BatchData, EnvSpec
from quasimetric_rl.data.online import ReplayBuffer
from quasimetric_rl.data.online.utils import get_empty_episodes
from quasimetric_rl.modules import QRLConf
from online.trainer import InteractionConf, Trainer


ENVIRONMENT_SHAPES = {
    ('gcrl', 'FetchReach'): (10, 4, (0, 1, 2)),
    ('gcrl', 'FetchPush'): (25, 4, (3, 4, 5)),
    ('gcrl', 'FetchSlide'): (25, 4, (3, 4, 5)),
    ('gcrl', 'FetchPickAndPlace'): (25, 4, (3, 4, 5)),
    ('dmc', 'reacher_easy'): (6, 2, (0, 1)),
    ('dmc', 'reacher_hard'): (6, 2, (0, 1)),
    ('gym_mujoco', 'Reacher-v4'): (8, 2, (0, 1)),
}


def make_env_spec(state_dim, action_dim):
    return EnvSpec(
        observation_space=gym.spaces.Box(
            -np.inf, np.inf, shape=(state_dim,), dtype=np.float32,
        ),
        observation_space_is_dict=True,
        action_space=gym.spaces.Box(
            -np.ones(action_dim, dtype=np.float32),
            np.ones(action_dim, dtype=np.float32),
            dtype=np.float32,
        ),
    )


def small_baseline_conf(algorithm):
    conf = copy.deepcopy(QRLConf())
    conf.algorithm = algorithm
    conf.baselines.td_infonce.hidden_sizes = (16, 16)
    conf.baselines.td_infonce.representation_dim = 8
    conf.baselines.crl.hidden_sizes = (16, 16)
    conf.baselines.crl.representation_dim = 8
    conf.baselines.gcbc.hidden_sizes = (16, 16)
    conf.baselines.c_learning.hidden_sizes = (16, 16)
    conf.baselines.td_infonce.batch_size = 8
    conf.baselines.crl.batch_size = 8
    conf.baselines.gcbc.batch_size = 8
    conf.baselines.c_learning.batch_size = 8
    return conf


def make_batch(state_dim, action_dim, batch_size=8):
    return BatchData(
        observations=torch.randn(batch_size, state_dim),
        actions=torch.empty(batch_size, action_dim).uniform_(-0.8, 0.8),
        next_observations=torch.randn(batch_size, state_dim),
        future_observations=torch.randn(batch_size, state_dim),
        rewards=torch.zeros(batch_size),
        terminals=torch.zeros(batch_size, dtype=torch.bool),
        timeouts=torch.zeros(batch_size, dtype=torch.bool),
    )


class GCRLBaselineShapeTest(unittest.TestCase):
    def test_all_algorithms_accept_all_seven_environment_shapes(self):
        for environment, (state_dim, action_dim, goal_dims) in ENVIRONMENT_SHAPES.items():
            env_spec = make_env_spec(state_dim, action_dim)
            batch = make_batch(state_dim, action_dim)
            for algorithm in ('td_infonce', 'crl', 'gcbc', 'c_learning'):
                with self.subTest(environment=environment, algorithm=algorithm):
                    conf = small_baseline_conf(algorithm)
                    agent, losses = conf.make(
                        env_spec=env_spec,
                        total_optim_steps=2,
                        goal_set_dims=goal_dims,
                    )
                    action = agent.act(
                        batch.observations, batch.future_observations,
                    ).mean
                    self.assertEqual(tuple(action.shape), (8, action_dim))
                    self.assertTrue(torch.isfinite(action).all())
                    result = losses(agent, batch, optimize=False)
                    self.assertTrue(torch.isfinite(result.loss))

    def test_td_target_networks_are_frozen(self):
        env_spec = make_env_spec(10, 4)
        for algorithm in ('td_infonce', 'c_learning'):
            with self.subTest(algorithm=algorithm):
                conf = small_baseline_conf(algorithm)
                agent, _ = conf.make(
                    env_spec=env_spec,
                    total_optim_steps=2,
                    goal_set_dims=(0, 1, 2),
                )
                self.assertIsNotNone(agent.target_critic)
                self.assertFalse(any(
                    parameter.requires_grad
                    for parameter in agent.target_critic.parameters()
                ))


class GCRLBaselineCheckpointTest(unittest.TestCase):
    def test_agent_and_optimizer_state_round_trip(self):
        env_spec = make_env_spec(10, 4)
        batch = make_batch(10, 4)
        for algorithm in ('td_infonce', 'crl', 'gcbc', 'c_learning'):
            with self.subTest(algorithm=algorithm):
                conf = small_baseline_conf(algorithm)
                agent, losses = conf.make(
                    env_spec=env_spec,
                    total_optim_steps=2,
                    goal_set_dims=(0, 1, 2),
                )
                losses(agent, batch, optimize=True)
                expected_action = agent.act(
                    batch.observations, batch.future_observations,
                ).mean.detach()
                agent_state = copy.deepcopy(agent.state_dict())
                losses_state = copy.deepcopy(losses.state_dict())

                restored_agent, restored_losses = conf.make(
                    env_spec=env_spec,
                    total_optim_steps=4,
                    goal_set_dims=(0, 1, 2),
                )
                restored_agent.load_state_dict(agent_state)
                restored_losses.load_state_dict(losses_state)
                restored_losses.set_scheduler_horizon(4)
                actual_action = restored_agent.act(
                    batch.observations, batch.future_observations,
                ).mean.detach()
                torch.testing.assert_close(actual_action, expected_action)
                result = restored_losses(restored_agent, batch, optimize=True)
                self.assertTrue(torch.isfinite(result.loss))


class GCSLReplaySamplingTest(unittest.TestCase):
    def test_uniform_pairs_stay_in_episode_and_are_strictly_future(self):
        state_dim = 2
        episode_length = 5
        num_episodes = 3
        env_spec = make_env_spec(state_dim, 1)
        replay = ReplayBuffer.__new__(ReplayBuffer)
        replay.env = SimpleNamespace(episode_length=episode_length)
        replay.env_spec = env_spec
        replay.num_episodes_realized = num_episodes
        replay.future_observation_discount = 0.99
        replay.transition_history_length = 0
        replay.max_episode_length = episode_length
        replay.raw_data = get_empty_episodes(
            env_spec, episode_length, num_episodes,
        )
        replay.indices_to_episode_indices = torch.repeat_interleave(
            torch.arange(num_episodes), episode_length,
        )
        replay.indices_to_episode_timesteps = torch.arange(
            episode_length,
        ).repeat(num_episodes)
        replay.training_episode_mask = torch.tensor([True, False, True])
        for episode in range(num_episodes):
            for timestep in range(episode_length + 1):
                index = episode * (episode_length + 1) + timestep
                replay.raw_data.all_observations[index] = torch.tensor([
                    float(episode), float(timestep),
                ])

        np.random.seed(7)
        batch = replay.sample_uniform_future_pairs(512)
        torch.testing.assert_close(
            batch.observations[:, 0], batch.future_observations[:, 0],
        )
        self.assertTrue(torch.all(
            batch.future_observations[:, 1] > batch.observations[:, 1]
        ))

        np.random.seed(7)
        training_batch = replay.sample_uniform_future_pairs(
            512, training_only=True,
        )
        self.assertNotIn(1.0, training_batch.observations[:, 0].unique().tolist())

        np.random.seed(7)
        latest_training_batch = replay.sample_uniform_future_pairs(
            512, training_only=True, max_episodes=1,
        )
        self.assertEqual(
            latest_training_batch.observations[:, 0].unique().tolist(), [2.0],
        )

        np.random.seed(7)
        recent_batch = replay.sample(512, max_transitions=3)
        self.assertEqual(recent_batch.observations[:, 0].unique().tolist(), [2.0])
        self.assertGreaterEqual(recent_batch.observations[:, 1].min().item(), 2.0)

    def test_default_action_discretization_matches_official_wrapper(self):
        env_spec = make_env_spec(10, 4)
        conf = small_baseline_conf('gcbc')
        agent, _ = conf.make(
            env_spec=env_spec,
            total_optim_steps=2,
            goal_set_dims=(0, 1, 2),
        )
        axes = [np.linspace(-1, 1, 3) for _ in range(4)]
        mesh = np.meshgrid(*axes)
        expected = np.array([axis.flat[:] for axis in mesh]).T

        self.assertEqual(tuple(agent.actor.action_table.shape), (81, 4))
        np.testing.assert_array_equal(
            agent.actor.action_table.cpu().numpy(), expected,
        )

        distribution = agent.actor(torch.zeros(2, 10), torch.zeros(2, 3))
        actions = agent.actor.action_table[torch.tensor([7, 51])]
        expected_nll = torch.nn.functional.cross_entropy(
            distribution.logits, torch.tensor([7, 51]), reduction='none',
        )
        torch.testing.assert_close(-distribution.log_prob(actions), expected_nll)


class CRLReplaySamplingTest(unittest.TestCase):
    def test_crl_discount_controls_geometric_future_sampling(self):
        replay = SimpleNamespace(
            episode_length=50,
            num_episodes_realized=0,
            transition_history_length=0,
            future_observation_discount=0.99,
            goal_set_dims=(0, 1, 2),
            env_spec=make_env_spec(10, 4),
        )
        conf = small_baseline_conf('crl')
        conf.baselines.crl.discount = 0.8
        interaction = InteractionConf(
            total_env_steps=50,
            num_prefill_episodes=0,
            num_samples_per_cycle=1,
            num_rollouts_per_cycle=1,
            num_eval_episodes=1,
            num_test_episodes=1,
            validation_seed=1000,
            test_seed=2000,
        )
        Trainer(
            agent_conf=conf,
            device=torch.device('cpu'),
            replay=replay,
            batch_size=8,
            interaction_conf=interaction,
        )
        self.assertEqual(replay.future_observation_discount, 0.8)


class ReferenceDefaultConfigTest(unittest.TestCase):
    def test_learning_defaults_match_reference_repositories(self):
        conf = QRLConf().baselines

        self.assertEqual(
            conf.td_infonce.reference_revision,
            '18f4e7e5872da9c3653f57661d01b4fbce85b50e',
        )
        self.assertEqual(conf.td_infonce.hidden_sizes, (512, 512, 512, 512))
        self.assertEqual(conf.td_infonce.representation_dim, 16)
        self.assertEqual(conf.td_infonce.actor_lr, 5e-5)
        self.assertEqual(conf.td_infonce.critic_lr, 3e-4)
        self.assertEqual(conf.td_infonce.discount, 0.99)
        self.assertEqual(conf.td_infonce.tau, 0.005)
        self.assertEqual(conf.td_infonce.min_replay_size, 10_000)
        self.assertEqual(conf.td_infonce.max_replay_size, 1_000_000)
        self.assertEqual(conf.td_infonce.batch_size, 256)
        self.assertEqual(conf.td_infonce.updates_per_env_step, 1.0)

        self.assertEqual(
            conf.crl.reference_revision,
            '7c53a0743a41423029abd17eb365c4822bf13687',
        )
        self.assertEqual(conf.crl.hidden_sizes, (256, 256))
        self.assertEqual(conf.crl.representation_dim, 64)
        self.assertEqual(conf.crl.actor_lr, 3e-4)
        self.assertEqual(conf.crl.critic_lr, 3e-4)
        self.assertEqual(conf.crl.alpha_lr, 3e-4)
        self.assertEqual(conf.crl.discount, 0.99)
        self.assertEqual(conf.crl.contrastive_loss, 'fwd_infonce')
        self.assertEqual(conf.crl.energy, 'norm')
        self.assertEqual(conf.crl.logsumexp_penalty, 0.1)
        self.assertIsNone(conf.crl.entropy_coefficient)
        self.assertEqual(conf.crl.min_replay_size, 1_000)
        self.assertEqual(conf.crl.max_replay_size, 10_000)
        self.assertEqual(conf.crl.batch_size, 256)
        self.assertEqual(
            conf.crl.updates_per_env_step,
            1001 / (256 * 62),
        )

        self.assertEqual(
            conf.gcbc.reference_revision,
            'cfae5609cee79e5a2228fb7653451023c41a64cb',
        )
        self.assertEqual(conf.gcbc.hidden_sizes, (400, 300))
        self.assertEqual(conf.gcbc.actor_lr, 5e-4)
        self.assertEqual(conf.gcbc.action_granularity, 3)
        self.assertEqual(conf.gcbc.start_policy_timesteps, 1_000)
        self.assertEqual(conf.gcbc.explore_timesteps, 10_000)
        self.assertEqual(conf.gcbc.validation_fraction, 0.2)
        self.assertEqual(conf.gcbc.replay_capacity_trajectories, 20_000)
        self.assertEqual(conf.gcbc.batch_size, 256)
        self.assertEqual(conf.gcbc.updates_per_env_step, 1.0)

        self.assertEqual(
            conf.c_learning.reference_revision,
            'ec7c3d346277b737bc2decffcd1b533d4b7ec105',
        )
        self.assertEqual(conf.c_learning.hidden_sizes, (256, 256))
        self.assertEqual(conf.c_learning.actor_lr, 3e-4)
        self.assertEqual(conf.c_learning.critic_lr, 3e-4)
        self.assertEqual(conf.c_learning.discount, 0.99)
        self.assertEqual(conf.c_learning.tau, 0.005)
        self.assertEqual(conf.c_learning.critic_loss_weight, 0.5)
        self.assertEqual(conf.c_learning.initial_collect_steps, 10_000)
        self.assertEqual(conf.c_learning.replay_buffer_capacity, 1_000_000)
        self.assertEqual(conf.c_learning.relabel_next_probability, 0.5)
        self.assertEqual(conf.c_learning.relabel_future_probability, 0.0)
        self.assertEqual(conf.c_learning.batch_size, 256)
        self.assertEqual(conf.c_learning.updates_per_env_step, 1.0)

    def test_trainer_uses_algorithm_specific_collection_defaults(self):
        cases = (
            ('td_infonce', 50, 200, 500, 10, 10_000),
            ('crl', 50, 20, 32, 10, 0),
            ('gcbc', 50, 21, 50, 1, 10_000),
            ('gcbc', 1000, 2, 1000, 1, 10_000),
            ('c_learning', 50, 200, 500, 10, 10_000),
        )
        for (
                algorithm, episode_length, prefill_episodes,
                samples_per_cycle, rollouts_per_cycle,
                random_policy_env_steps) in cases:
            with self.subTest(algorithm=algorithm, episode_length=episode_length):
                replay = SimpleNamespace(
                    episode_length=episode_length,
                    num_episodes_realized=0,
                    transition_history_length=0,
                    future_observation_discount=0.99,
                    goal_set_dims=(0, 1, 2),
                    env_spec=make_env_spec(10, 4),
                )
                trainer = Trainer(
                    agent_conf=small_baseline_conf(algorithm),
                    device=torch.device('cpu'),
                    replay=replay,
                    batch_size=8,
                    interaction_conf=InteractionConf(
                        total_env_steps=50_000,
                        num_eval_episodes=1,
                        num_test_episodes=1,
                        validation_seed=1000,
                        test_seed=2000,
                    ),
                )
                self.assertEqual(trainer.num_prefill_episodes, prefill_episodes)
                self.assertEqual(trainer.num_samples_per_cycle, samples_per_cycle)
                self.assertEqual(trainer.num_rollouts_per_cycle, rollouts_per_cycle)
                self.assertEqual(
                    trainer.random_policy_env_steps, random_policy_env_steps,
                )

    def test_reference_batch_size_cannot_be_overridden_by_shared_trainer(self):
        replay = SimpleNamespace(
            episode_length=50,
            num_episodes_realized=0,
            transition_history_length=0,
            future_observation_discount=0.99,
            goal_set_dims=(0, 1, 2),
            env_spec=make_env_spec(10, 4),
        )
        with self.assertRaisesRegex(ValueError, 'requires batch_size=256'):
            Trainer(
                agent_conf=QRLConf(algorithm='td_infonce'),
                device=torch.device('cpu'),
                replay=replay,
                batch_size=128,
                interaction_conf=InteractionConf(
                    total_env_steps=50_000,
                    num_eval_episodes=1,
                    num_test_episodes=1,
                    validation_seed=1000,
                    test_seed=2000,
                ),
            )


class ReferenceObjectiveTest(unittest.TestCase):
    def test_td_infonce_uses_reference_minimum_policy_std(self):
        env_spec = make_env_spec(10, 4)
        conf = small_baseline_conf('td_infonce')
        agent, _ = conf.make(
            env_spec=env_spec,
            total_optim_steps=2,
            goal_set_dims=(0, 1, 2),
        )
        for parameter in agent.actor.backbone.parameters():
            parameter.data.zero_()

        distribution = agent.actor(
            torch.zeros(3, 10), torch.zeros(3, 3),
        )
        expected_std = torch.full(
            (3, 4), float(np.log(2) + 1e-6),
        )
        torch.testing.assert_close(
            distribution._pre_tanh_distn.scale, expected_std,
        )

    def test_crl_uses_jaxgcrl_bounded_log_standard_deviation(self):
        env_spec = make_env_spec(10, 4)
        conf = small_baseline_conf('crl')
        agent, _ = conf.make(
            env_spec=env_spec,
            total_optim_steps=2,
            goal_set_dims=(0, 1, 2),
        )
        for parameter in agent.actor.backbone.parameters():
            parameter.data.zero_()

        distribution = agent.actor(
            torch.zeros(3, 10), torch.zeros(3, 3),
        )
        expected_std = torch.full((3, 4), float(np.exp(-1.5)))
        torch.testing.assert_close(
            distribution._pre_tanh_distn.scale, expected_std,
        )

    def test_crl_log_prob_uses_stable_pre_tanh_sample(self):
        env_spec = make_env_spec(10, 4)
        conf = small_baseline_conf('crl')
        agent, _ = conf.make(
            env_spec=env_spec,
            total_optim_steps=2,
            goal_set_dims=(0, 1, 2),
        )
        for parameter in agent.actor.backbone.parameters():
            parameter.data.zero_()
        agent.actor.backbone.module[-1].bias.data[:4] = 8.0
        distribution = agent.actor(
            torch.zeros(3, 10), torch.zeros(3, 3),
        )

        torch.manual_seed(23)
        pre_tanh_sample = distribution._pre_tanh_distn.rsample()
        expected_log_prob = (
            distribution._pre_tanh_distn.log_prob(pre_tanh_sample)
            - 2 * (
                np.log(2.0) - pre_tanh_sample
                - torch.nn.functional.softplus(-2 * pre_tanh_sample)
            )
            - distribution._affine_scale.abs().log()
        ).sum(dim=-1)
        expected_action = (
            distribution._affine_loc
            + distribution._affine_scale * torch.tanh(pre_tanh_sample)
        )

        torch.manual_seed(23)
        action, log_prob = distribution.rsample_with_log_prob()
        torch.testing.assert_close(action, expected_action)
        torch.testing.assert_close(log_prob, expected_log_prob)

    def test_crl_norm_energy_keeps_reference_epsilon_inside_sqrt(self):
        env_spec = make_env_spec(10, 4)
        conf = small_baseline_conf('crl')
        agent, _ = conf.make(
            env_spec=env_spec,
            total_optim_steps=2,
            goal_set_dims=(0, 1, 2),
        )
        zeros = torch.zeros(3, 8)
        expected = torch.full((3,), -1e-3)
        torch.testing.assert_close(agent.critic._energy(zeros, zeros), expected)

    def test_jaxgcrl_binary_nce_matches_reference_definition(self):
        env_spec = make_env_spec(10, 4)
        conf = small_baseline_conf('crl')
        conf.baselines.crl.contrastive_loss = 'binary_nce'
        conf.baselines.crl.logsumexp_penalty = 0.0
        agent, losses = conf.make(
            env_spec=env_spec,
            total_optim_steps=2,
            goal_set_dims=(0, 1, 2),
        )
        batch = make_batch(10, 4)
        future_goal = agent.extract_goal(batch.future_observations)
        logits = agent.critic.pairwise(
            batch.observations, batch.actions, future_goal,
        )
        expected = -torch.sigmoid(logits).mean()

        actual, _, _, _ = losses._crl_losses(agent, batch)
        torch.testing.assert_close(actual, expected)

    def test_c_learning_applies_reference_twin_critic_weight(self):
        env_spec = make_env_spec(10, 4)
        conf = small_baseline_conf('c_learning')
        agent, losses = conf.make(
            env_spec=env_spec,
            total_optim_steps=2,
            goal_set_dims=(0, 1, 2),
        )
        batch = make_batch(10, 4)

        torch.manual_seed(19)
        weighted_loss, _, _ = losses._c_learning_losses(agent, batch)
        losses.c_learning_conf.critic_loss_weight = 1.0
        torch.manual_seed(19)
        unweighted_loss, _, _ = losses._c_learning_losses(agent, batch)

        torch.testing.assert_close(weighted_loss * 2, unweighted_loss)

    def test_c_learning_default_policy_std_has_no_added_epsilon(self):
        env_spec = make_env_spec(10, 4)
        conf = small_baseline_conf('c_learning')
        agent, _ = conf.make(
            env_spec=env_spec,
            total_optim_steps=2,
            goal_set_dims=(0, 1, 2),
        )
        for parameter in agent.actor.backbone.parameters():
            parameter.data.zero_()

        distribution = agent.actor(
            torch.zeros(3, 10), torch.zeros(3, 3),
        )
        expected_std = torch.full((3, 4), float(np.log(2)))
        torch.testing.assert_close(
            distribution._pre_tanh_distn.scale, expected_std,
        )

    def test_c_learning_updates_critic_before_computing_actor_loss(self):
        env_spec = make_env_spec(10, 4)
        conf = small_baseline_conf('c_learning')
        agent, losses = conf.make(
            env_spec=env_spec,
            total_optim_steps=2,
            goal_set_dims=(0, 1, 2),
        )
        batch = make_batch(10, 4)
        critic_before = [
            parameter.detach().clone() for parameter in agent.critic.parameters()
        ]
        actor_saw_updated_critic = []
        original_actor_loss = losses._c_learning_actor_loss

        def recording_actor_loss(current_agent, data, goals):
            actor_saw_updated_critic.append(any(
                not torch.equal(before, after)
                for before, after in zip(
                    critic_before, current_agent.critic.parameters(),
                )
            ))
            return original_actor_loss(current_agent, data, goals)

        object.__setattr__(
            losses, '_c_learning_actor_loss', recording_actor_loss,
        )
        losses(agent, batch, optimize=True)

        self.assertEqual(actor_saw_updated_critic, [True])


if __name__ == '__main__':
    unittest.main()

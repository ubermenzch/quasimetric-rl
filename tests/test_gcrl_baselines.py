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


if __name__ == '__main__':
    unittest.main()

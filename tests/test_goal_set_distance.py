import copy
import types
import unittest

import gym
import numpy as np
import torch

from offline.trainer import ResumableRandomBatchIterator
from online.trainer import Trainer
from quasimetric_rl.data import BatchData, Dataset, EnvSpec
from quasimetric_rl.data.base import GOAL_SET_DIMS_REGISTRY
from quasimetric_rl.modules import QRLAgent, QRLConf
from quasimetric_rl.modules.actor.losses.min_dist import MinDistLoss
from quasimetric_rl.modules.quasimetric_critic import CriticBatchInfo


def vector_env_spec(observation_size=4, action_size=2):
    return EnvSpec(
        observation_space=gym.spaces.Box(
            low=-np.ones(observation_size, dtype=np.float32),
            high=np.ones(observation_size, dtype=np.float32),
            dtype=np.float32,
        ),
        observation_space_is_dict=False,
        action_space=gym.spaces.Box(
            low=-np.ones(action_size, dtype=np.float32),
            high=np.ones(action_size, dtype=np.float32),
            dtype=np.float32,
        ),
    )


def small_agent_conf(*, gsd_enabled, implementation='learned', aggregation='hard_min',
                     candidate_sampling='uniform_bounds'):
    conf = copy.deepcopy(QRLConf())
    conf.actor.model.arch = (8,)
    conf.quasimetric_critic.model.encoder.arch = (8,)
    conf.quasimetric_critic.model.encoder.latent_size = 4
    conf.quasimetric_critic.model.quasimetric_model.projector_arch = (8,)
    conf.quasimetric_critic.model.quasimetric_model.quasimetric_head_spec = 'l2(dim=4)'
    conf.quasimetric_critic.model.latent_dynamics.arch = (8,)
    conf.goal_set_distance.enabled = gsd_enabled
    conf.goal_set_distance.model.arch = (8,)
    conf.goal_set_distance.losses.implementation = implementation
    conf.goal_set_distance.losses.aggregation = aggregation
    conf.goal_set_distance.losses.candidate_sampling = candidate_sampling
    conf.goal_set_distance.losses.num_goal_samples = 4
    conf.goal_set_distance.losses.goal_dims = (0, 1)
    return conf


def batch_data(observation_size=4, batch_size=3):
    observations = torch.arange(
        batch_size * observation_size, dtype=torch.float32
    ).reshape(batch_size, observation_size)
    return BatchData(
        observations=observations,
        actions=torch.zeros(batch_size, 2),
        next_observations=observations + 1,
        future_observations=observations + 2,
        rewards=torch.zeros(batch_size),
        terminals=torch.zeros(batch_size, dtype=torch.bool),
        timeouts=torch.zeros(batch_size, dtype=torch.bool),
    )


class GoalSetDistanceCompatibilityTest(unittest.TestCase):
    @staticmethod
    def make_seeded_goal_set_loss(seed, *, implementation='learned', aggregation='hard_min'):
        _, losses = small_agent_conf(
            gsd_enabled=True,
            implementation=implementation,
            aggregation=aggregation,
        ).make(
            env_spec=vector_env_spec(),
            total_optim_steps=10,
        )
        loss = losses.goal_set_distance_loss
        loss.set_observation_bounds_provider(
            lambda *, device=None: (
                torch.full((4,), -2.0, device=device),
                torch.full((4,), 2.0, device=device),
            )
        )
        loss.set_candidate_seed(seed)
        return losses, loss

    def test_goal_dimensions_cover_all_requested_vector_environments(self):
        expected = {
            ('d4rl', 'maze2d-umaze-v1'): (0, 1),
            ('d4rl', 'maze2d-medium-v1'): (0, 1),
            ('d4rl', 'maze2d-large-v1'): (0, 1),
            ('gcrl', 'FetchReach'): (0, 1, 2),
            ('gcrl', 'FetchPush'): (3, 4, 5),
            ('gcrl', 'FetchSlide'): (3, 4, 5),
        }
        for env_key, goal_dims in expected.items():
            with self.subTest(env=env_key):
                self.assertEqual(GOAL_SET_DIMS_REGISTRY[env_key], goal_dims)

    def test_all_learned_and_direct_aggregations_train(self):
        aggregations = ('hard_min', 'lme_min', 'median', 'hard_max', 'lme_max')
        for implementation in ('learned', 'direct'):
            for aggregation in aggregations:
                with self.subTest(implementation=implementation, aggregation=aggregation):
                    agent, losses = small_agent_conf(
                        gsd_enabled=True,
                        implementation=implementation,
                        aggregation=aggregation,
                    ).make(env_spec=vector_env_spec(), total_optim_steps=10)
                    goal_set_loss = losses.goal_set_distance_loss
                    goal_set_loss.set_observation_bounds_provider(
                        lambda *, device=None: (
                            torch.full((4,), -2.0, device=device),
                            torch.full((4,), 2.0, device=device),
                        )
                    )
                    result = losses(agent, batch_data(), optimize=True)

                    self.assertTrue(torch.isfinite(result.loss))
                    self.assertEqual(goal_set_loss.candidate_step, 1)
                    if implementation == 'learned':
                        self.assertIsNotNone(agent.goal_set_distance)
                        self.assertIsNotNone(goal_set_loss.optim)
                        diagnostic_info = result.info['goal_set_distance']
                    else:
                        self.assertIsNone(agent.goal_set_distance)
                        self.assertIsNone(goal_set_loss.optim)
                        diagnostic_info = result.info['actor']['min_dist']
                    if aggregation == 'median':
                        self.assertNotIn('lme_hard_gap', diagnostic_info)
                    else:
                        self.assertIn('lme_hard_gap', diagnostic_info)
                        self.assertGreaterEqual(
                            diagnostic_info['lme_hard_gap'].item(), -1e-6
                        )

    def test_set_aggregation_values(self):
        distances = torch.tensor([[1.0, 2.0, 4.0, 8.0]])
        temperature = 0.7
        expected = {
            'hard_min': distances.min(dim=-1).values,
            'hard_max': distances.max(dim=-1).values,
            'median': torch.tensor([3.0]),
            'lme_min': -temperature * (
                torch.logsumexp(-distances / temperature, dim=-1)
                - np.log(distances.shape[-1])
            ),
            'lme_max': temperature * (
                torch.logsumexp(distances / temperature, dim=-1)
                - np.log(distances.shape[-1])
            ),
        }
        for aggregation, expected_value in expected.items():
            with self.subTest(aggregation=aggregation):
                _, loss = self.make_seeded_goal_set_loss(7)
                loss.aggregation = aggregation
                loss.lme_temperature = temperature
                torch.testing.assert_close(
                    loss.aggregate_distances(distances), expected_value
                )

    def test_lme_diagnostics_track_temperature_smoothing(self):
        distances = torch.tensor([[1.0, 2.0, 4.0, 8.0]])
        for aggregation in ('lme_min', 'lme_max'):
            with self.subTest(aggregation=aggregation):
                _, objective = self.make_seeded_goal_set_loss(7)
                objective.aggregation = aggregation
                objective.lme_temperature = 0.01
                cold = objective.aggregation_diagnostics(distances)
                objective.lme_temperature = 10.0
                hot = objective.aggregation_diagnostics(distances)

                self.assertGreater(
                    hot['lme_hard_gap'].item(), cold['lme_hard_gap'].item()
                )
                self.assertGreater(
                    hot['lme_weight_entropy_fraction'].item(),
                    cold['lme_weight_entropy_fraction'].item(),
                )
                self.assertGreater(
                    hot['lme_effective_candidates'].item(),
                    cold['lme_effective_candidates'].item(),
                )

    def test_direct_order_statistic_path_matches_full_aggregation(self):
        class SquaredDistance(torch.nn.Module):
            def forward(self, left, right):
                return (left - right).square().sum(dim=-1)

        critic = types.SimpleNamespace(quasimetric_model=SquaredDistance())
        candidate_latents = torch.tensor([
            [[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [7.0, 0.0]],
            [[0.0, 1.0], [0.0, 2.0], [0.0, 4.0], [0.0, 8.0]],
        ])
        for aggregation in ('hard_min', 'lme_min', 'median', 'hard_max', 'lme_max'):
            with self.subTest(aggregation=aggregation):
                _, losses = small_agent_conf(
                    gsd_enabled=True,
                    implementation='direct',
                    aggregation=aggregation,
                ).make(env_spec=vector_env_spec(), total_optim_steps=10)
                objective = losses.goal_set_distance_loss
                next_latent = torch.tensor(
                    [[0.25, 0.1], [0.1, 0.25]], requires_grad=True
                )
                actual = objective.direct_actor_distance(
                    critic, next_latent, candidate_latents
                )
                all_distances = critic.quasimetric_model(
                    next_latent[:, None, :].expand_as(candidate_latents),
                    candidate_latents,
                )
                expected = objective.aggregate_distances(all_distances)
                torch.testing.assert_close(actual, expected)
                gradient = torch.autograd.grad(actual.sum(), next_latent)[0]
                self.assertTrue(torch.isfinite(gradient).all())

    def test_dataset_radius_sampler_returns_only_matching_real_states(self):
        class ObservationDataset:
            _goal_condition_coordinates = Dataset._goal_condition_coordinates
            _goal_condition_grid = Dataset._goal_condition_grid
            sample_goal_conditioned_observations = Dataset.sample_goal_conditioned_observations

            def __init__(self, observations):
                self.raw_data = types.SimpleNamespace(all_observations=observations)

            @property
            def num_observations_available(self):
                return self.raw_data.all_observations.shape[0]

        observations = torch.tensor([
            [0.0, 0.0, 10.0, 11.0],
            [0.1, 0.1, 20.0, 21.0],
            [0.35, 0.35, 25.0, 26.0],
            [1.0, 1.0, 30.0, 31.0],
            [-1.0, -1.0, 40.0, 41.0],
        ])
        dataset = ObservationDataset(observations)
        raw_goals = observations[:2].clone()
        candidates_a, fallback_a = dataset.sample_goal_conditioned_observations(
            raw_goals,
            goal_dims=(0, 1),
            num_samples=32,
            radius=0.2,
            seed=123,
        )
        torch.rand(1000)
        candidates_b, fallback_b = dataset.sample_goal_conditioned_observations(
            raw_goals,
            goal_dims=(0, 1),
            num_samples=32,
            radius=0.2,
            seed=123,
        )

        torch.testing.assert_close(candidates_a, candidates_b, rtol=0, atol=0)
        self.assertEqual(fallback_a, fallback_b)
        self.assertEqual(fallback_a, 0.0)
        self.assertEqual(tuple(candidates_a.shape), (2, 32, 4))
        is_real_state = (
            candidates_a[..., None, :] == observations
        ).all(dim=-1).any(dim=-1)
        self.assertTrue(is_real_state.all())
        goal_distances = torch.linalg.vector_norm(
            candidates_a[..., :2] - raw_goals[:, None, :2], dim=-1
        )
        self.assertTrue((goal_distances <= 0.2).all())
        self.assertTrue(all(
            torch.unique(candidate_set, dim=0).shape[0] < candidate_set.shape[0]
            for candidate_set in candidates_a
        ))

        limited_candidates, limited_fallback = (
            dataset.sample_goal_conditioned_observations(
                raw_goals[:1],
                goal_dims=(0, 1),
                num_samples=32,
                radius=0.2,
                seed=123,
                max_attempts=1,
            )
        )
        self.assertGreater(limited_fallback, 0.0)
        limited_goal_distances = torch.linalg.vector_norm(
            limited_candidates[..., :2] - raw_goals[:1, None, :2], dim=-1
        )
        self.assertTrue((limited_goal_distances <= 0.2).all())

        _, losses = small_agent_conf(
            gsd_enabled=True,
            candidate_sampling='dataset_radius',
        ).make(env_spec=vector_env_spec(), total_optim_steps=10)
        objective = losses.goal_set_distance_loss
        objective.num_goal_samples = 16
        objective.goal_condition_radius = 0.2
        objective.set_candidate_state_provider(
            dataset.sample_goal_conditioned_observations
        )
        objective.set_candidate_seed(123)
        full_candidates, candidate_mask = (
            objective._sample_goal_condition_states_with_mask(raw_goals)
        )

        self.assertEqual(tuple(full_candidates.shape), (2, 16, 4))
        torch.testing.assert_close(full_candidates[:, 0], raw_goals, rtol=0, atol=0)
        self.assertTrue(candidate_mask.all())
        self.assertEqual(objective.last_candidate_count_mean, 16.0)
        self.assertEqual(objective.last_candidate_count_min, 16)
        self.assertEqual(objective.last_candidate_shortfall_fraction, 0.0)
        additional_candidates = full_candidates[:, 1:]
        is_real_state = (
            additional_candidates[..., None, :] == observations
        ).all(dim=-1).any(dim=-1)
        self.assertTrue(is_real_state.all())
        additional_goal_distances = torch.linalg.vector_norm(
            additional_candidates[..., :2] - raw_goals[:, None, :2], dim=-1
        )
        self.assertTrue((additional_goal_distances <= 0.2).all())
        self.assertTrue(all(
            torch.unique(candidate_set, dim=0).shape[0] < candidate_set.shape[0]
            for candidate_set in additional_candidates
        ))

    def test_dataset_grid_updates_incrementally(self):
        class GrowingObservationDataset:
            _goal_condition_coordinates = Dataset._goal_condition_coordinates
            _goal_condition_grid = Dataset._goal_condition_grid
            sample_goal_conditioned_observations = Dataset.sample_goal_conditioned_observations

            def __init__(self, observations):
                self.raw_data = types.SimpleNamespace(all_observations=observations)
                self.available = 2

            @property
            def num_observations_available(self):
                return self.available

        observations = torch.tensor([
            [0.0, 0.0, 10.0, 11.0],
            [0.1, 0.1, 20.0, 21.0],
            [2.0, 2.0, 30.0, 31.0],
            [2.1, 2.1, 40.0, 41.0],
            [99.0, 99.0, 99.0, 99.0],
        ])
        dataset = GrowingObservationDataset(observations)
        dataset.sample_goal_conditioned_observations(
            observations[:1], goal_dims=(0, 1), num_samples=8,
            radius=0.2, seed=1,
        )
        first_grid = dataset._goal_condition_grid_caches[((0, 1), 0.2)]
        self.assertEqual(first_grid['count'], 2)

        dataset.available = 4
        candidates, fallback = dataset.sample_goal_conditioned_observations(
            observations[3:4], goal_dims=(0, 1), num_samples=16,
            radius=0.2, seed=2,
        )
        updated_grid = dataset._goal_condition_grid_caches[((0, 1), 0.2)]
        self.assertIs(first_grid, updated_grid)
        self.assertEqual(updated_grid['count'], 4)
        self.assertEqual(fallback, 0.0)
        is_available_state = (
            candidates[..., None, :] == observations[:4]
        ).all(dim=-1).any(dim=-1)
        self.assertTrue(is_available_state.all())

    def test_enabling_gsd_does_not_change_base_initialization(self):
        env_spec = vector_env_spec()
        torch.manual_seed(1234)
        base_agent, _ = small_agent_conf(gsd_enabled=False).make(
            env_spec=env_spec,
            total_optim_steps=10,
        )
        torch.manual_seed(1234)
        gsd_agent, _ = small_agent_conf(gsd_enabled=True).make(
            env_spec=env_spec,
            total_optim_steps=10,
        )

        self.assertIsNone(base_agent.goal_set_distance)
        self.assertIsNotNone(gsd_agent.goal_set_distance)
        for base_param, gsd_param in zip(base_agent.actor.parameters(), gsd_agent.actor.parameters()):
            torch.testing.assert_close(base_param, gsd_param)
        for base_critic, gsd_critic in zip(base_agent.critics, gsd_agent.critics):
            for base_param, gsd_param in zip(base_critic.parameters(), gsd_critic.parameters()):
                torch.testing.assert_close(base_param, gsd_param)

    def test_agent_pads_only_gsd_goals(self):
        class CapturingActor(torch.nn.Module):
            input_mode = 'raw'

            def __init__(self):
                super().__init__()
                self.last_goal = None

            def forward(self, observation, goal):
                self.last_goal = goal
                return goal

        observation = torch.zeros(1, 6)
        goal = torch.arange(6, dtype=torch.float32).unsqueeze(0)

        base_actor = CapturingActor()
        QRLAgent(base_actor, []).act(observation, goal)
        torch.testing.assert_close(base_actor.last_goal, goal)

        gsd_actor = CapturingActor()
        QRLAgent(gsd_actor, [], goal_set_dims=(3, 4, 5)).act(observation, goal)
        torch.testing.assert_close(
            gsd_actor.last_goal,
            torch.tensor([[0.0, 0.0, 0.0, 3.0, 4.0, 5.0]]),
        )

    def test_gsd_actor_reencodes_both_latents_consistently(self):
        class Encoder(torch.nn.Module):
            def forward(self, value):
                return value * 2

        critic = types.SimpleNamespace(encoder=Encoder())
        data = batch_data()
        stale = torch.full_like(data.observations, -123.0)
        critic_info = CriticBatchInfo(
            critic=critic,
            zx=stale,
            zy=stale,
            px=stale,
            py=stale,
        )
        min_dist = MinDistLoss(
            env_spec=vector_env_spec(),
            adaptive_entropy_regularizer=False,
            add_goal_as_future_state=False,
        )

        class GoalSetLoss:
            @staticmethod
            def padded_goal_state(goal):
                padded = torch.zeros_like(goal)
                padded[..., :2] = goal[..., :2]
                return padded

        observations, goals, infos = min_dist.gather_obs_goal_pairs(
            [critic_info],
            data,
            goal_set_distance_loss=GoalSetLoss(),
        )
        torch.testing.assert_close(infos[0].zo, critic.encoder(observations))
        torch.testing.assert_close(infos[0].zg, critic.encoder(goals))
        self.assertFalse(torch.equal(infos[0].zo, stale))

    def test_gsd_head_update_does_not_modify_base_modules(self):
        agent, losses = small_agent_conf(gsd_enabled=True).make(
            env_spec=vector_env_spec(),
            total_optim_steps=10,
        )
        data = batch_data()
        losses.goal_set_distance_loss.set_observation_bounds_provider(
            lambda *, device=None: (
                torch.full((4,), -2.0, device=device),
                torch.full((4,), 2.0, device=device),
            )
        )
        critic_infos = losses._make_critic_batch_infos(agent, data)

        actor_before = [parameter.detach().clone() for parameter in agent.actor.parameters()]
        critic_before = [
            parameter.detach().clone()
            for critic in agent.critics
            for parameter in critic.parameters()
        ]
        gsd_before = [
            parameter.detach().clone()
            for parameter in agent.goal_set_distance.parameters()
        ]

        losses.goal_set_distance_loss(
            agent.goal_set_distance,
            critic_infos,
            data,
            optimize=True,
        )

        for before, after in zip(actor_before, agent.actor.parameters()):
            torch.testing.assert_close(before, after)
        for before, after in zip(
                critic_before,
                (parameter for critic in agent.critics for parameter in critic.parameters())):
            torch.testing.assert_close(before, after)
        self.assertTrue(any(
            not torch.equal(before, after)
            for before, after in zip(gsd_before, agent.goal_set_distance.parameters())
        ))

    def test_batch_and_candidate_streams_ignore_global_rng_consumption(self):
        class IndexedDataset:
            def __len__(self):
                return 24

            def __getitem__(self, indices):
                observations = indices.to(torch.float32)[:, None].expand(-1, 4)
                return BatchData(
                    observations=observations,
                    actions=torch.zeros(indices.numel(), 2),
                    next_observations=observations + 1,
                    future_observations=observations + 2,
                    rewards=torch.zeros(indices.numel()),
                    terminals=torch.zeros(indices.numel(), dtype=torch.bool),
                    timeouts=torch.zeros(indices.numel(), dtype=torch.bool),
                )

        iterator_a = ResumableRandomBatchIterator(
            IndexedDataset(), batch_size=6, drop_last=True, seed=17,
        )
        iterator_b = ResumableRandomBatchIterator(
            IndexedDataset(), batch_size=6, drop_last=True, seed=17,
        )
        _, loss_a = self.make_seeded_goal_set_loss(29)
        _, loss_b = self.make_seeded_goal_set_loss(
            29, implementation='direct', aggregation='hard_max'
        )

        for step in range(3):
            _, _, data_a = next(iter(iterator_a))
            iterator_a.advance_batch()
            torch.rand(1000 + step)
            _, _, data_b = next(iter(iterator_b))
            iterator_b.advance_batch()

            goal_a = torch.roll(data_a.next_observations, 1, dims=0)
            goal_b = torch.roll(data_b.next_observations, 1, dims=0)
            torch.testing.assert_close(goal_a, goal_b, rtol=0, atol=0)

            loss_a.load_candidate_rng_state_dict(dict(seed=29, step=step))
            loss_b.load_candidate_rng_state_dict(dict(seed=29, step=step))
            candidates_a = loss_a._sample_goal_condition_states(goal_a)
            torch.rand(2000 + step)
            candidates_b = loss_b._sample_goal_condition_states(goal_b)
            torch.testing.assert_close(candidates_a, candidates_b, rtol=0, atol=0)

    def test_candidate_rng_position_is_restored_with_loss_checkpoint(self):
        for implementation in ('learned', 'direct'):
            with self.subTest(implementation=implementation):
                losses_a, loss_a = self.make_seeded_goal_set_loss(
                    41, implementation=implementation
                )
                loss_a.load_candidate_rng_state_dict(dict(seed=41, step=7))
                checkpoint = copy.deepcopy(losses_a.state_dict())

                losses_b, loss_b = self.make_seeded_goal_set_loss(
                    999, implementation=implementation
                )
                losses_b.load_state_dict(checkpoint)

                self.assertEqual(loss_b.candidate_rng_state_dict(), dict(seed=41, step=7))
                raw_goals = batch_data().next_observations
                candidates_a = loss_a._sample_goal_condition_states(raw_goals)
                torch.rand(4096)
                candidates_b = loss_b._sample_goal_condition_states(raw_goals)
                torch.testing.assert_close(candidates_a, candidates_b, rtol=0, atol=0)

    def test_online_optimizer_step_count_includes_post_prefill_cycle(self):
        trainer = object.__new__(Trainer)
        trainer.replay = types.SimpleNamespace(
            num_episodes_realized=0,
            episode_length=50,
        )
        trainer.num_prefill_episodes = 200
        trainer.num_rollouts_per_cycle = 10
        trainer.num_samples_per_cycle = 500

        expected = {
            16_000: 6_500,
            22_000: 12_500,
            600_000: 590_500,
            650_000: 640_500,
            750_000: 740_500,
            800_000: 790_500,
            850_000: 840_500,
        }
        for env_steps, optim_steps in expected.items():
            with self.subTest(env_steps=env_steps):
                self.assertEqual(trainer.get_total_optim_steps(env_steps), optim_steps)


if __name__ == '__main__':
    unittest.main()

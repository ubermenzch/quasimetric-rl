import copy
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import gym
import numpy as np
import torch
from omegaconf import OmegaConf, SCMode

from quasimetric_rl.data.base import (
    BatchData,
    CREATE_ENV_REGISTRY,
    GOAL_SET_DIMS_REGISTRY,
)
from quasimetric_rl.data.env_spec import EnvSpec
from quasimetric_rl.model_size import (
    load_model_size_preset,
    select_qrl_model_size,
)
from quasimetric_rl.data.online.goal_env import (
    pack_goal_observation,
    unpack_reset_result,
    unpack_step_result,
    vector_goal_observation_space,
)
from quasimetric_rl.data.online.utils import get_empty_episode
from quasimetric_rl.modules import QRLConf
from online.main import (
    Conf as OnlineConf,
    ONLINE_CHECKPOINT_KIND_COMMITTED,
    ONLINE_CHECKPOINT_KIND_INTERRUPTED,
    SELECTED_BEST_AGENT_FILENAME,
    committed_online_checkpoint_issue,
    deployment_agent_checkpoint_state,
    ensure_selected_best_agent_checkpoint,
    load_latest_committed_online_checkpoint,
    maintain_selected_best_agent_checkpoint,
    publish_selected_best_agent_checkpoint,
    retain_only_online_best_and_final_checkpoints,
    resolve_online_resume_plan,
    resolve_split_encoder_goal_dims,
    selected_best_agent_checkpoint_issue,
    selected_best_validation_summary,
    select_best_validation,
    summarize_evaluation,
    validation_improves,
)
from online.trainer import (
    EvalEpisodeResult,
    InteractionConf,
    Trainer,
    add_gaussian_exploration,
    resolve_interaction_schedule,
)


ONLINE_GOAL_ENV_SPECS = {
    ('gcrl', 'FetchReach'): ((10,), (4,), (0, 1, 2)),
    ('gcrl', 'FetchPush'): ((25,), (4,), (3, 4, 5)),
    ('gcrl', 'FetchSlide'): ((25,), (4,), (3, 4, 5)),
    ('gcrl', 'FetchPickAndPlace'): ((25,), (4,), (3, 4, 5)),
    ('gym_mujoco', 'Reacher-v4'): ((8,), (2,), (0, 1)),
    ('gym_mujoco', 'Pusher-v4'): ((20,), (7,), (0, 1, 2)),
    ('gym_mujoco', 'AntNavigate-v4'): ((29,), (8,), (0, 1)),
    ('dmc', 'reacher_easy'): ((6,), (2,), (0, 1)),
    ('dmc', 'reacher_hard'): ((6,), (2,), (0, 1)),
    ('dmc', 'swimmer6'): ((17,), (5,), (0, 1)),
    ('dmc', 'swimmer15'): ((35,), (14,), (0, 1)),
    ('dmc', 'quadruped_fetch'): ((60,), (12,), (0, 1)),
    ('dmc', 'manipulator_bring_ball'): ((40,), (5,), (0, 1)),
    ('dmc', 'manipulator_bring_peg'): ((40,), (5,), (0, 1, 2, 3)),
    ('online_maze', 'maze2d-medium'): ((4,), (2,), (0, 1)),
    ('online_maze', 'maze2d-large'): ((4,), (2,), (0, 1)),
}


def small_agent_conf(*, split: bool, goal_dims):
    conf = copy.deepcopy(QRLConf())
    conf.num_critics = 1
    conf.actor.model.arch = (8,)
    conf.actor.losses.min_dist.adaptive_entropy_regularizer = False
    conf.actor.losses.min_dist.add_goal_as_future_state = True
    encoder = conf.quasimetric_critic.model.encoder
    encoder.latent_size = 4
    encoder.arch = (8,)
    conf.quasimetric_critic.model.quasimetric_model.projector_arch = (8,)
    conf.quasimetric_critic.model.quasimetric_model.quasimetric_head_spec = 'l2(dim=4)'
    conf.quasimetric_critic.model.latent_dynamics.arch = (8,)
    if split:
        encoder.kind = 'split'
        encoder.goal_dims = goal_dims
        encoder.goal_arch = (8,)
        encoder.non_goal_arch = (8,)
        encoder.goal_latent_size = 2
        encoder.non_goal_latent_size = 2
        conf.actor.model.input_mode = 'split_latent'
        conf.actor.losses.min_dist.latent_goal_mode = 'max'
        conf.actor.losses.min_dist.latent_goal_steps = 2
        conf.actor.losses.min_dist.latent_goal_keep_best = True
    return conf


class GoalEnvironmentHelpersTest(unittest.TestCase):
    def test_pack_goal_observation_pads_only_goal_dimensions(self):
        state = np.arange(6, dtype=np.float32)
        packed = pack_goal_observation(state, np.array([8.0, 9.0]), (1, 4))
        np.testing.assert_array_equal(packed['observation'], state)
        np.testing.assert_array_equal(packed['achieved_goal'], state)
        np.testing.assert_array_equal(
            packed['desired_goal'],
            np.array([0.0, 8.0, 0.0, 0.0, 9.0, 0.0], dtype=np.float32),
        )
        self.assertIsNot(packed['observation'], packed['achieved_goal'])

    def test_old_and_new_environment_results_are_normalized(self):
        observation = np.array([1.0], dtype=np.float32)
        self.assertEqual(unpack_reset_result(observation)[1], {})
        self.assertEqual(unpack_reset_result((observation, {'seed': 3}))[1], {'seed': 3})

        old = unpack_step_result((
            observation, 1.0, True, {'TimeLimit.truncated': True},
        ))
        self.assertEqual(old[1:4], (1.0, False, True))
        new = unpack_step_result((observation, 2.0, True, False, {'x': 1}))
        self.assertEqual(new[1:4], (2.0, True, False))


class OnlineGoalEnvironmentRegistrationTest(unittest.TestCase):
    def test_all_online_goal_environments_are_registered_with_goal_dimensions(self):
        for key, (_, _, goal_dims) in ONLINE_GOAL_ENV_SPECS.items():
            with self.subTest(environment=key):
                self.assertIn(key, CREATE_ENV_REGISTRY)
                self.assertEqual(GOAL_SET_DIMS_REGISTRY[key], goal_dims)

    def test_online_split_latent_automatically_uses_registered_goal_dimensions(self):
        for (kind, name), (_, _, goal_dims) in ONLINE_GOAL_ENV_SPECS.items():
            with self.subTest(environment=(kind, name)):
                cfg = copy.deepcopy(OnlineConf())
                cfg.env.kind = kind
                cfg.env.name = name
                cfg.agent.quasimetric_critic.model.encoder.kind = 'split'
                self.assertIsNone(
                    cfg.agent.quasimetric_critic.model.encoder.goal_dims
                )
                resolve_split_encoder_goal_dims(cfg)
                self.assertEqual(
                    cfg.agent.quasimetric_critic.model.encoder.goal_dims,
                    goal_dims,
                )

    def test_explicit_split_latent_goal_dimensions_are_preserved(self):
        cfg = copy.deepcopy(OnlineConf())
        cfg.env.kind = 'dmc'
        cfg.env.name = 'reacher_easy'
        encoder = cfg.agent.quasimetric_critic.model.encoder
        encoder.kind = 'split'
        encoder.goal_dims = (2, 3)
        resolve_split_encoder_goal_dims(cfg)
        self.assertEqual(encoder.goal_dims, (2, 3))

    def test_go_qrl_level_resolves_task_specific_encoder_branches(self):
        for (kind, name), (observation_shape, action_shape, goal_dims) in (
                ONLINE_GOAL_ENV_SPECS.items()):
            with self.subTest(environment=(kind, name)):
                level = select_qrl_model_size(observation_shape[0])
                cfg = copy.deepcopy(OnlineConf())
                cfg.env.kind = kind
                cfg.env.name = name
                merged_agent = OmegaConf.merge(
                    OmegaConf.structured(cfg.agent),
                    load_model_size_preset('go_qrl', level),
                )
                cfg.agent = OmegaConf.to_container(
                    merged_agent,
                    structured_config_mode=SCMode.INSTANTIATE,
                )
                resolve_split_encoder_goal_dims(cfg)

                env_spec = EnvSpec(
                    observation_space=gym.spaces.Box(
                        -np.inf, np.inf,
                        shape=observation_shape,
                        dtype=np.float32,
                    ),
                    observation_space_is_dict=True,
                    action_space=gym.spaces.Box(
                        -np.ones(action_shape, dtype=np.float32),
                        np.ones(action_shape, dtype=np.float32),
                        dtype=np.float32,
                    ),
                )
                encoder = cfg.agent.quasimetric_critic.model.encoder
                plan = encoder.resolve_split_parameterization(env_spec=env_spec)

                self.assertIsNotNone(plan)
                self.assertEqual(encoder.goal_dims, goal_dims)
                self.assertEqual(
                    encoder.goal_latent_size + encoder.non_goal_latent_size,
                    encoder.latent_size,
                )
                self.assertEqual(
                    plan.goal_encoder_parameters + plan.non_goal_encoder_parameters,
                    plan.qrl_encoder_parameters,
                )
                self.assertGreaterEqual(
                    plan.goal_encoder_parameters * 8,
                    plan.non_goal_encoder_parameters,
                )

    def test_base_and_split_latent_agents_accept_every_environment_shape(self):
        for key, (observation_shape, action_shape, goal_dims) in ONLINE_GOAL_ENV_SPECS.items():
            observation_space = vector_goal_observation_space(observation_shape[0])
            env_spec = EnvSpec(
                observation_space=observation_space['observation'],
                observation_space_is_dict=True,
                action_space=gym.spaces.Box(
                    low=-np.ones(action_shape, dtype=np.float32),
                    high=np.ones(action_shape, dtype=np.float32),
                    dtype=np.float32,
                ),
            )
            observation = torch.zeros(3, *observation_shape)
            goal = torch.zeros_like(observation)

            for split in (False, True):
                with self.subTest(environment=key, split=split):
                    conf = small_agent_conf(split=split, goal_dims=goal_dims)
                    agent, losses = conf.make(
                        env_spec=env_spec,
                        total_optim_steps=2,
                    )
                    action = agent.act(observation, goal).mean
                    self.assertEqual(tuple(action.shape), (3, *action_shape))
                    self.assertTrue(torch.isfinite(action).all())

                    batch = BatchData(
                        observations=observation,
                        actions=torch.zeros(3, *action_shape),
                        next_observations=observation + 0.01,
                        future_observations=observation + 0.02,
                        rewards=torch.zeros(3),
                        terminals=torch.zeros(3, dtype=torch.bool),
                        timeouts=torch.zeros(3, dtype=torch.bool),
                    )
                    result = losses(agent, batch, optimize=False)
                    self.assertTrue(torch.isfinite(result.loss))


class OnlineExplorationTest(unittest.TestCase):
    def test_noise_scales_and_clips_to_actual_action_bounds(self):
        action = torch.tensor([[1.9, -1.9]])
        action_space = gym.spaces.Box(
            low=np.array([-2.0, -2.0], dtype=np.float32),
            high=np.array([2.0, 2.0], dtype=np.float32),
            dtype=np.float32,
        )
        with mock.patch(
                'online.trainer.torch.randn_like',
                return_value=torch.tensor([[1.0, -1.0]])):
            explored = add_gaussian_exploration(action, action_space, 0.3)
        torch.testing.assert_close(explored, torch.tensor([[2.0, -2.0]]))

    def test_unit_action_bounds_keep_the_existing_noise_scale(self):
        action = torch.zeros(1, 2)
        action_space = gym.spaces.Box(
            low=-np.ones(2, dtype=np.float32),
            high=np.ones(2, dtype=np.float32),
            dtype=np.float32,
        )
        with mock.patch(
                'online.trainer.torch.randn_like',
                return_value=torch.tensor([[0.5, -0.5]])):
            explored = add_gaussian_exploration(action, action_space, 0.3)
        torch.testing.assert_close(explored, torch.tensor([[0.15, -0.15]]))


class OnlineInteractionScheduleTest(unittest.TestCase):
    def test_checkpoint_validation_and_test_defaults(self):
        cfg = copy.deepcopy(OnlineConf())
        self.assertEqual(cfg.save_steps, 20_000)
        self.assertIsNone(cfg.eval_steps)
        self.assertEqual(cfg.interaction.num_eval_episodes, 1000)
        self.assertEqual(cfg.interaction.num_test_episodes, 1000)
        self.assertNotEqual(
            cfg.interaction.validation_seed,
            cfg.interaction.test_seed,
        )

    def test_validation_episode_count_must_be_positive(self):
        with self.assertRaises(ValueError):
            InteractionConf(num_eval_episodes=0)

    def test_validation_and_test_seed_ranges_must_be_disjoint(self):
        conf = small_agent_conf(split=False, goal_dims=(0, 1))
        replay = mock.Mock()
        replay.episode_length = 50
        with self.assertRaisesRegex(ValueError, 'seed ranges must be disjoint'):
            Trainer(
                agent_conf=conf,
                device=torch.device('cpu'),
                replay=replay,
                batch_size=2,
                interaction_conf=InteractionConf(
                    num_eval_episodes=1000,
                    num_test_episodes=1000,
                    validation_seed=1000,
                    test_seed=1500,
                ),
            )

    def test_evaluation_reseeds_each_episode_with_a_contiguous_range(self):
        env_spec = EnvSpec(
            observation_space=gym.spaces.Box(
                -np.inf, np.inf, shape=(4,), dtype=np.float32,
            ),
            observation_space_is_dict=True,
            action_space=gym.spaces.Box(
                -1, 1, shape=(2,), dtype=np.float32,
            ),
        )
        fake_env = mock.Mock()
        fake_env.episode_length = 2
        trainer = Trainer.__new__(Trainer)
        trainer.eval_seed = 99
        trainer.num_eval_episodes = 3
        trainer.profiler = None
        trainer.make_evaluate_env = mock.Mock(return_value=fake_env)

        def rollout(**_kwargs):
            episode = get_empty_episode(env_spec, fake_env.episode_length)
            episode.rewards.zero_()
            episode.transition_infos['is_success'].zero_()
            return episode

        trainer.collect_rollout = mock.Mock(side_effect=rollout)

        trainer.evaluate(num_episodes=3, seed=1000)

        self.assertEqual(
            fake_env.seed.call_args_list,
            [mock.call(1000), mock.call(1001), mock.call(1002)],
        )
        fake_env.close.assert_called_once_with()

    def test_default_transition_budget_scales_with_episode_length(self):
        expected = {
            50: (200, 500, 10, 1000),
            100: (100, 500, 5, 1000),
            1000: (10, 1000, 1, 1000),
        }
        for episode_length, counts in expected.items():
            with self.subTest(episode_length=episode_length):
                schedule = resolve_interaction_schedule(
                    InteractionConf(), episode_length,
                )
                self.assertEqual(
                    (
                        schedule.num_prefill_episodes,
                        schedule.num_samples_per_cycle,
                        schedule.num_rollouts_per_cycle,
                        schedule.num_eval_episodes,
                    ),
                    counts,
                )

    def test_explicit_episode_counts_override_transition_budget(self):
        conf = InteractionConf(
            num_prefill_episodes=3,
            num_samples_per_cycle=4,
            num_rollouts_per_cycle=5,
            num_eval_episodes=6,
        )
        schedule = resolve_interaction_schedule(conf, 1000)
        self.assertEqual(
            (
                schedule.num_prefill_episodes,
                schedule.num_samples_per_cycle,
                schedule.num_rollouts_per_cycle,
                schedule.num_eval_episodes,
            ),
            (3, 4, 5, 6),
        )


class OnlineCheckpointSelectionTest(unittest.TestCase):
    @staticmethod
    def validation_summary(env_steps, success_count, *, hitting_time=10.0):
        return {
            'split': 'validation',
            'seed': 1000,
            'seed_end': 1999,
            'num_episodes': 1000,
            'env_steps': env_steps,
            'optim_steps': env_steps - 1000,
            'success_count': success_count,
            'succ_rate': success_count / 1000,
            'hitting_time': hitting_time,
            'epi_return': float(success_count),
            'checkpoint': f'checkpoint_env{env_steps:08d}_opt00000000.pth',
            'agent_checkpoint': f'agent_checkpoint_env{env_steps:08d}.pth',
        }

    def test_deployment_checkpoint_contains_only_agent_and_provenance(self):
        agent_state = {'actor.weight': torch.tensor([1.0])}
        validation_summary = {
            'env_steps': 200_000,
            'optim_steps': 190_000,
            'succ_rate': 0.8,
        }

        state = deployment_agent_checkpoint_state(
            agent_state, validation_summary,
        )

        self.assertEqual(set(state), {
            'checkpoint_kind',
            'env_steps',
            'optim_steps',
            'agent',
            'validation_summary',
        })
        self.assertIs(state['agent'], agent_state)
        self.assertNotIn('losses', state)
        self.assertNotIn('replay', state)
        self.assertNotIn('rng', state)

    def test_summary_uses_censored_one_based_hitting_time(self):
        rewards = torch.tensor([
            [0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ])
        successes = rewards.bool()
        result = EvalEpisodeResult.from_timestep_reward_is_success(
            rewards, successes,
        )
        summary = summarize_evaluation(
            result,
            split='validation',
            seed=10,
            env_steps=20_000,
            optim_steps=11_000,
            episode_length=4,
        )
        self.assertEqual(summary['success_count'], 2)
        self.assertAlmostEqual(summary['succ_rate'], 2 / 3)
        self.assertAlmostEqual(summary['hitting_time'], (2 + 5 + 1) / 3)

    def test_selection_prioritizes_success_then_speed_then_return(self):
        base = {
            'split': 'validation',
            'checkpoint': 'checkpoint.pth',
            'agent_checkpoint': 'agent.pth',
            'env_steps': 20_000,
            'optim_steps': 10_000,
        }
        candidates = [
            dict(base, success_count=899, hitting_time=1.0, epi_return=100.0),
            dict(base, success_count=900, hitting_time=12.0, epi_return=10.0),
            dict(base, success_count=900, hitting_time=11.0, epi_return=5.0),
            dict(base, success_count=900, hitting_time=11.0, epi_return=6.0),
        ]
        selected = select_best_validation(candidates)
        self.assertEqual(selected, candidates[-1])

    def test_rolling_best_replaces_only_on_improvement(self):
        incumbent = self.validation_summary(20_000, 700)
        worse = self.validation_summary(40_000, 699, hitting_time=1.0)
        better = self.validation_summary(60_000, 701, hitting_time=20.0)

        self.assertFalse(validation_improves(worse, incumbent))
        self.assertTrue(validation_improves(better, incumbent))
        self.assertTrue(validation_improves(incumbent, None))

    def test_rolling_best_file_is_unchanged_for_worse_validation(self):
        with TemporaryDirectory() as directory:
            incumbent = self.validation_summary(20_000, 700)
            selected, path, updated = maintain_selected_best_agent_checkpoint(
                directory,
                agent_state_dict={'actor.weight': torch.tensor([1.0])},
                candidate=incumbent,
                incumbent=None,
            )
            self.assertTrue(updated)
            original = Path(path).read_bytes()

            worse = self.validation_summary(40_000, 699, hitting_time=1.0)
            retained, retained_path, updated = (
                maintain_selected_best_agent_checkpoint(
                    directory,
                    agent_state_dict={'actor.weight': torch.tensor([2.0])},
                    candidate=worse,
                    incumbent=selected,
                )
            )

            self.assertFalse(updated)
            self.assertEqual(retained, selected)
            self.assertEqual(retained_path, path)
            self.assertEqual(Path(path).read_bytes(), original)

            better = self.validation_summary(60_000, 750)
            replaced, _, updated = maintain_selected_best_agent_checkpoint(
                directory,
                agent_state_dict={'actor.weight': torch.tensor([3.0])},
                candidate=better,
                incumbent=selected,
            )
            self.assertTrue(updated)
            self.assertEqual(replaced['env_steps'], 60_000)
            state = torch.load(path, map_location='cpu', weights_only=False)
            torch.testing.assert_close(
                state['agent']['actor.weight'], torch.tensor([3.0]),
            )

    def test_publish_selected_best_is_model_only_and_atomic_target(self):
        with TemporaryDirectory() as directory:
            summary = self.validation_summary(20_000, 700)
            agent_state = {'actor.weight': torch.tensor([2.0])}

            selected, path = publish_selected_best_agent_checkpoint(
                directory,
                agent_state_dict=agent_state,
                validation_summary=summary,
            )

            self.assertEqual(
                selected['agent_checkpoint'], SELECTED_BEST_AGENT_FILENAME,
            )
            self.assertEqual(
                selected['source_agent_checkpoint'],
                summary['agent_checkpoint'],
            )
            self.assertEqual(Path(path).name, SELECTED_BEST_AGENT_FILENAME)
            state = torch.load(path, map_location='cpu', weights_only=False)
            self.assertIsNone(
                selected_best_agent_checkpoint_issue(state, selected),
            )
            self.assertEqual(set(state), {
                'checkpoint_kind', 'env_steps', 'optim_steps', 'agent',
                'validation_summary',
            })
            self.assertFalse(Path(path + '.tmp').exists())

    def test_resume_repairs_best_from_newly_committed_checkpoint(self):
        with TemporaryDirectory() as directory:
            stale = selected_best_validation_summary(
                self.validation_summary(20_000, 700),
            )
            publish_selected_best_agent_checkpoint(
                directory,
                agent_state_dict={'actor.weight': torch.tensor([1.0])},
                validation_summary=stale,
            )
            committed_best = selected_best_validation_summary(
                self.validation_summary(40_000, 750),
            )

            selected, path, repaired = ensure_selected_best_agent_checkpoint(
                directory,
                validation_summary=committed_best,
                committed_state={
                    'agent': {'actor.weight': torch.tensor([2.0])},
                    'validation_summary': committed_best,
                },
            )

            self.assertTrue(repaired)
            self.assertEqual(selected, committed_best)
            state = torch.load(path, map_location='cpu', weights_only=False)
            torch.testing.assert_close(
                state['agent']['actor.weight'], torch.tensor([2.0]),
            )
            _, _, repaired_again = ensure_selected_best_agent_checkpoint(
                directory,
                validation_summary=committed_best,
            )
            self.assertFalse(repaired_again)

    def test_resume_repairs_non_mapping_best_payload(self):
        with TemporaryDirectory() as directory:
            best_path = Path(directory) / SELECTED_BEST_AGENT_FILENAME
            torch.save(torch.tensor([123.0]), best_path)
            committed_best = selected_best_validation_summary(
                self.validation_summary(40_000, 750),
            )

            _, path, repaired = ensure_selected_best_agent_checkpoint(
                directory,
                validation_summary=committed_best,
                committed_state={
                    'agent': {'actor.weight': torch.tensor([2.0])},
                    'validation_summary': committed_best,
                },
            )

            self.assertTrue(repaired)
            state = torch.load(path, map_location='cpu', weights_only=False)
            self.assertIsNone(
                selected_best_agent_checkpoint_issue(state, committed_best),
            )

    def test_best_and_final_retention_removes_other_online_checkpoints(self):
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            selected = output_dir / SELECTED_BEST_AGENT_FILENAME
            final = output_dir / (
                'checkpoint_env00500000_opt00490000_finalizing.pth'
            )
            removed_names = {
                'checkpoint_env00450000_opt00440000.pth',
                'checkpoint_env00450000_opt00440000_final.pth',
                'checkpoint_env00500000_opt00490000.pth',
                'agent_checkpoint_env00450000_opt00440000.pth',
                'agent_checkpoint_env00500000_opt00490000.pth',
                'checkpoint_resume_latest.pth',
            }
            for name in removed_names:
                (output_dir / name).touch()
            selected.touch()
            final.touch()
            metadata = output_dir / 'best_checkpoint.json'
            metadata.touch()

            removed = retain_only_online_best_and_final_checkpoints(
                str(output_dir),
                best_agent_checkpoint=str(selected),
                final_checkpoint=str(final),
            )

            self.assertEqual(set(removed), removed_names)
            self.assertTrue(selected.exists())
            self.assertTrue(final.exists())
            self.assertTrue(metadata.exists())

    def test_best_and_final_retention_validates_keep_files_before_cleanup(self):
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            old = output_dir / 'checkpoint_env00100000_opt00090000.pth'
            old.touch()
            selected = output_dir / SELECTED_BEST_AGENT_FILENAME
            selected.touch()

            with self.assertRaises(FileNotFoundError):
                retain_only_online_best_and_final_checkpoints(
                    str(output_dir),
                    best_agent_checkpoint=str(selected),
                    final_checkpoint=str(
                        output_dir /
                        'checkpoint_env00200000_opt00190000_final.pth'
                    ),
                )

            self.assertTrue(old.exists())


class OnlineResumePlanTest(unittest.TestCase):
    @staticmethod
    def checkpoint_state(env_steps, optim_steps, *, kind=None, cycle_sample=500):
        state = {
            'env_steps': env_steps,
            'optim_steps': optim_steps,
            'agent': {},
            'losses': {},
            'rng': {},
            'replay': {},
            'loop_state': {
                'next_cycle_sample': cycle_sample,
                'cycle_env_steps': env_steps,
            },
            'validation_summary': {
                'env_steps': env_steps,
                'optim_steps': optim_steps,
            },
        }
        if kind is not None:
            state['checkpoint_kind'] = kind
        return state

    def test_resume_selects_latest_committed_checkpoint(self):
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            committed = output_dir / 'checkpoint_env00140000_opt00130500.pth'
            interrupted = output_dir / 'checkpoint_env00154500_opt00144538_interrupted.pth'
            legacy_interrupt = output_dir / 'checkpoint_env00155000_opt00145000.pth'
            torch.save(self.checkpoint_state(
                140_000,
                130_500,
                kind=ONLINE_CHECKPOINT_KIND_COMMITTED,
            ), committed)
            torch.save(self.checkpoint_state(
                154_500,
                144_538,
                kind=ONLINE_CHECKPOINT_KIND_INTERRUPTED,
            ), interrupted)
            legacy_state = self.checkpoint_state(155_000, 145_000)
            legacy_state.pop('validation_summary')
            torch.save(legacy_state, legacy_interrupt)

            selected = load_latest_committed_online_checkpoint(
                str(output_dir), num_samples_per_cycle=500,
            )
            self.assertIsNotNone(selected)
            path, env_steps, optim_steps, _ = selected
            self.assertEqual(Path(path), committed)
            self.assertEqual((env_steps, optim_steps), (140_000, 130_500))

    def test_finalizing_checkpoint_is_resumable_during_final_test(self):
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            finalizing = output_dir / (
                'checkpoint_env02000000_opt01990000_finalizing.pth'
            )
            torch.save(self.checkpoint_state(
                2_000_000,
                1_990_000,
                kind=ONLINE_CHECKPOINT_KIND_COMMITTED,
            ), finalizing)

            selected = load_latest_committed_online_checkpoint(
                str(output_dir), num_samples_per_cycle=500,
            )

            self.assertIsNotNone(selected)
            path, env_steps, optim_steps, _ = selected
            self.assertEqual(Path(path), finalizing)
            self.assertEqual((env_steps, optim_steps), (2_000_000, 1_990_000))

    def test_legacy_validation_checkpoint_is_committed(self):
        state = self.checkpoint_state(120_000, 110_500)
        self.assertIsNone(committed_online_checkpoint_issue(
            state,
            expected_env_steps=120_000,
            expected_optim_steps=110_500,
            num_samples_per_cycle=500,
        ))

    def test_mid_cycle_checkpoint_is_rejected(self):
        state = self.checkpoint_state(
            120_000,
            110_238,
            kind=ONLINE_CHECKPOINT_KIND_COMMITTED,
            cycle_sample=237,
        )
        self.assertIn('next_cycle_sample', committed_online_checkpoint_issue(
            state,
            expected_env_steps=120_000,
            expected_optim_steps=110_238,
            num_samples_per_cycle=500,
        ))

    def test_only_uncommitted_checkpoints_restart_from_zero(self):
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            state = self.checkpoint_state(5_000, 1_234)
            state.pop('validation_summary')
            torch.save(
                state,
                output_dir / 'checkpoint_env00005000_opt00001234.pth',
            )
            self.assertIsNone(load_latest_committed_online_checkpoint(
                str(output_dir), num_samples_per_cycle=500,
            ))

    def test_extends_full_replay_from_200k_to_300k(self):
        plan = resolve_online_resume_plan(
            requested_total_env_steps=300_000,
            start_env_steps=200_000,
            loaded_replay=True,
            replay_env_steps_offset=0,
            replay_env_steps=200_000,
        )
        self.assertEqual(plan.env_steps_offset, 0)
        self.assertEqual(plan.local_total_env_steps, 300_000)
        self.assertEqual(plan.remaining_env_steps, 100_000)

    def test_chained_extension_preserves_replay_offset(self):
        plan = resolve_online_resume_plan(
            requested_total_env_steps=450_000,
            start_env_steps=300_000,
            loaded_replay=True,
            replay_env_steps_offset=200_000,
            replay_env_steps=100_000,
        )
        self.assertEqual(plan.env_steps_offset, 200_000)
        self.assertEqual(plan.local_total_env_steps, 250_000)
        self.assertEqual(plan.remaining_env_steps, 150_000)

    def test_rejects_target_below_latest_checkpoint(self):
        with self.assertRaisesRegex(ValueError, 'below the latest checkpoint'):
            resolve_online_resume_plan(
                requested_total_env_steps=199_999,
                start_env_steps=200_000,
                loaded_replay=True,
                replay_env_steps_offset=0,
                replay_env_steps=200_000,
            )

    def test_rejects_inconsistent_replay_cursor(self):
        with self.assertRaisesRegex(RuntimeError, 'replay cursor'):
            resolve_online_resume_plan(
                requested_total_env_steps=300_000,
                start_env_steps=200_000,
                loaded_replay=True,
                replay_env_steps_offset=0,
                replay_env_steps=190_000,
            )


if __name__ == '__main__':
    unittest.main()

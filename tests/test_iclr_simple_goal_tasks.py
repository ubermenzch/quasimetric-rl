import copy
from collections import Counter
import types
import unittest

import torch
from hydra import compose, initialize
from omegaconf import OmegaConf, SCMode

import online.main
from quasimetric_rl.data import Dataset
from quasimetric_rl.modules.actor import Actor
from quasimetric_rl.modules.actor.losses.min_dist import MinDistLoss
from tests.test_goal_set_distance import batch_data, small_agent_conf, vector_env_spec
from tests.test_latent_goal_completion import SquaredDistance
from quasimetric_rl.modules.quasimetric_critic.models.encoder import SplitEncoder
from tools.generate_iclr_simple_goal_tasks import (
    ENVIRONMENTS, LOSS_PREFIX, QUEUE, SEEDS, VARIANTS, generate_tasks,
)
from tools.generate_selected_success_l_min32_tasks import extra_arg_map
from tools.run_qrl_queue import read_tasks, task_training_args
from tools.watch_qrl_queue import task_variant


class SimpleGoalInputTest(unittest.TestCase):
    def test_training_and_inference_mask_only_goal_coordinates(self):
        for mode in ('masked', 'goal_only'):
            with self.subTest(mode=mode):
                conf = small_agent_conf(gsd_enabled=False)
                conf.actor.model.input_mode = mode
                conf.actor.model.goal_dims = (0, 1)
                agent, _ = conf.make(env_spec=vector_env_spec(), total_optim_steps=5)
                observed = []
                hook = agent.actor.backbone.register_forward_pre_hook(
                    lambda module, inputs: observed.append(inputs[0].clone()))
                state = torch.tensor([[1., 2., 3., 4.]])
                goal = torch.tensor([[5., 6., 7., 8.]])
                agent.actor(state, goal)
                goal[..., 2:] = float('nan')
                agent.act(state, goal)
                hook.remove()
                expected = ([1., 2., 3., 4., 5., 6., 0., 0., 1., 1., 0., 0.]
                            if mode == 'masked' else [1., 2., 3., 4., 5., 6.])
                for actual in observed:
                    torch.testing.assert_close(actual, torch.tensor([expected]))

    def test_full_goal_still_reaches_critic_and_action_gradients_survive(self):
        for mode in ('masked', 'goal_only'):
            with self.subTest(mode=mode):
                conf = small_agent_conf(gsd_enabled=False)
                conf.actor.model.input_mode = mode
                conf.actor.model.goal_dims = (0, 1)
                agent, losses = conf.make(env_spec=vector_env_spec(), total_optim_steps=5)
                data = batch_data()
                with torch.no_grad():
                    infos = losses._make_critic_batch_infos(agent, data)
                    _, full_goals = losses.actor_loss.min_dist.gather_raw_obs_goal_pairs(data)
                    expected = agent.critics[0].encoder(full_goals)
                targets = []
                hook = agent.critics[0].quasimetric_model.register_forward_pre_hook(
                    lambda module, inputs: targets.append(inputs[1].detach().clone()))
                result = losses.actor_loss.min_dist(agent.actor, infos, data)
                hook.remove()
                torch.testing.assert_close(targets[-1], expected)
                result.loss.backward()
                grads = [parameter.grad for parameter in agent.actor.parameters() if parameter.grad is not None]
                self.assertTrue(grads)
                self.assertTrue(all(torch.isfinite(grad).all() for grad in grads))
                self.assertTrue(any(grad.abs().sum() > 0 for grad in grads))

    def test_missing_or_invalid_coordinate_split_is_rejected(self):
        for dims in (None, (), (0, 0), (0, 4)):
            with self.assertRaises(ValueError):
                Actor.Conf(input_mode='goal_only', goal_dims=dims).make(env_spec=vector_env_spec())

    def test_new_input_controls_use_current_full_state_encoder_after_critic_step(self):
        conf = small_agent_conf(gsd_enabled=False)
        conf.actor.model.input_mode, conf.actor.model.goal_dims = 'masked', (0, 1)
        agent, losses = conf.make(env_spec=vector_env_spec(), total_optim_steps=5)
        data = batch_data()
        with torch.no_grad():
            infos = losses._make_critic_batch_infos(agent, data)
            for parameter in agent.critics[0].encoder.parameters():
                parameter.add_(.1)
            _, goals = losses.actor_loss.min_dist.gather_raw_obs_goal_pairs(data)
            expected = agent.critics[0].encoder(goals)
        targets = []
        hook = agent.critics[0].quasimetric_model.register_forward_pre_hook(
            lambda module, args: targets.append(args[1].detach().clone()))
        losses.actor_loss.min_dist(agent.actor, infos, data)
        hook.remove()
        torch.testing.assert_close(targets[-1], expected)


class ReplayGoalSelectionTest(unittest.TestCase):
    def make_loss_and_critic(self, count=5):
        encoder = SplitEncoder(env_spec=vector_env_spec(), goal_dims=(0, 1),
                               goal_arch=(), non_goal_arch=(),
                               goal_latent_size=2, non_goal_latent_size=2)
        # Identity branch maps make the selected raw state directly observable.
        encoder.goal_encoder = torch.nn.Identity()
        encoder.non_goal_encoder = torch.nn.Identity()
        critic = types.SimpleNamespace(encoder=encoder, quasimetric_model=SquaredDistance())
        loss = MinDistLoss(env_spec=vector_env_spec(), adaptive_entropy_regularizer=False,
                           add_goal_as_future_state=False, replay_goal_candidates=count,
                           replay_goal_radius=0.1)
        return loss, critic

    def test_selects_whole_real_state_and_rejects_wrong_goal(self):
        loss, critic = self.make_loss_and_critic()
        raw = torch.tensor([[0., 0., 10., 10.]])
        candidates = torch.tensor([[[0.02, 0., 2., 3.], [0., 0., 4., 4.],
                                    [1., 0., 2., 3.], [0., 0., 20., 20.]]])
        loss.set_replay_candidate_provider(lambda *args, **kwargs: (candidates, 0.), seed=7)
        predicted = torch.tensor([[1., 0., 2., 3.]], requires_grad=True)
        rng = torch.get_rng_state().clone()
        chosen, info = loss._select_replay_goal(critic, predicted, raw, critic.encoder(raw))
        torch.testing.assert_close(chosen, candidates[:, 0])
        self.assertFalse(chosen.requires_grad)
        torch.testing.assert_close(torch.get_rng_state(), rng)
        self.assertLessEqual(info['replay_selected_goal_offset_max'], 0.1)
        self.assertEqual(info['replay_candidate_rejected_fraction'].item(), 0.25)
        critic.quasimetric_model(predicted, chosen).mean().backward()
        self.assertTrue(torch.isfinite(predicted.grad).all())
        self.assertGreater(predicted.grad.abs().sum().item(), 0)

    def test_keeps_original_when_candidates_are_worse_and_resumes_rng_stream(self):
        loss, critic = self.make_loss_and_critic()
        raw = torch.tensor([[0., 0., 1., 2.]])
        seeds = []
        def provider(goals, **kwargs):
            seeds.append(kwargs['seed'])
            return goals[:, None, :].expand(-1, 4, -1).clone(), 1.
        loss.set_replay_candidate_provider(provider, seed=13)
        initial = critic.encoder(raw)
        chosen, info = loss._select_replay_goal(critic, initial, raw, initial)
        torch.testing.assert_close(chosen, initial)
        self.assertEqual(info['replay_candidate_fallback_fraction'].item(), 1.)
        saved = copy.deepcopy(loss.state_dict())
        loss._select_replay_goal(critic, initial, raw, initial)
        loss.load_state_dict(saved)
        loss._select_replay_goal(critic, initial, raw, initial)
        self.assertEqual(seeds, [13, 14, 14])

    def test_actual_replay_sampler_preserves_complete_states(self):
        class Bank:
            _goal_condition_coordinates = Dataset._goal_condition_coordinates
            _goal_condition_grid = Dataset._goal_condition_grid
            sample_goal_conditioned_observations = Dataset.sample_goal_conditioned_observations
            num_observations_available = 3
            raw_data = types.SimpleNamespace(all_observations=torch.tensor([
                [0., 0., 5., 6.], [0.03, 0.02, 1., 2.], [1., 1., -1., -2.],
            ]))
        loss, critic = self.make_loss_and_critic(count=32)
        bank = Bank()
        loss.set_replay_candidate_provider(bank.sample_goal_conditioned_observations, seed=5)
        raw = bank.raw_data.all_observations[:1]
        chosen, _ = loss._select_replay_goal(
            critic, torch.tensor([[0., 0., 1., 2.]]), raw, critic.encoder(raw))
        torch.testing.assert_close(chosen, bank.raw_data.all_observations[1:2])

    def test_actor_update_uses_full_initial_target_and_replay_gradients(self):
        conf = small_agent_conf(gsd_enabled=False)
        conf.num_critics = 1
        conf.actor.model.input_mode = 'split_latent'
        encoder = conf.quasimetric_critic.model.encoder
        encoder.kind, encoder.goal_dims = 'split', (0, 1)
        encoder.goal_arch = encoder.non_goal_arch = (8,)
        encoder.goal_latent_size = encoder.non_goal_latent_size = 2
        conf.quasimetric_critic.model.quasimetric_model.quasimetric_head_spec = 'iqe(dim=8,components=2)'
        conf.actor.losses.min_dist.replay_goal_candidates = 5
        conf.actor.losses.min_dist.replay_goal_radius = 0.1
        agent, losses = conf.make(env_spec=vector_env_spec(), total_optim_steps=5)
        def provider(goals, **kwargs):
            candidates = goals.unsqueeze(-2).expand(*goals.shape[:-1], 4, 4).clone()
            candidates[..., 2:] = 0
            return candidates, 0.
        losses.actor_loss.min_dist.set_replay_candidate_provider(provider, seed=3)
        data = batch_data()
        result = losses(agent, data, phase='actor')
        self.assertTrue(torch.isfinite(result.loss))
        grads = [p.grad for p in agent.actor.parameters() if p.grad is not None]
        self.assertTrue(any(grad.abs().sum() > 0 for grad in grads))


class SimpleGoalQueueTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = generate_tasks(read_tasks(QUEUE))

    def test_three_variants_three_tasks_three_seeds_and_matching_budgets(self):
        self.assertEqual(len(self.tasks), 27)
        self.assertEqual(Counter(task_variant(t) for t in self.tasks),
                         {'QRL+Mask': 9, 'QRL+GoalOnly': 9, 'AQRL+ReplayMin8': 9})
        for index, variant in enumerate(VARIANTS):
            tasks = self.tasks[index * 9:(index + 1) * 9]
            self.assertEqual(Counter(t.env_name for t in tasks), {name: 3 for name in ENVIRONMENTS})
            for name in ENVIRONMENTS:
                rows = [t for t in tasks if t.env_name == name]
                self.assertEqual({int(t.seed) for t in rows}, set(SEEDS))
                self.assertEqual({int(t.steps) for t in rows}, {200000})
            for task in tasks:
                args = extra_arg_map(task)
                self.assertEqual(args[LOSS_PREFIX + 'latent_goal_steps'], '0')
                self.assertEqual(args[LOSS_PREFIX + 'latent_goal_mode'], 'none')
                self.assertEqual(args['agent.quasimetric_critic.losses.latent_dynamics.distance'], 'iqe_mse')
                self.assertEqual(float(args['agent.quasimetric_critic.losses.latent_dynamics.mse_weight']), 1.)

    def test_hydra_composition_and_parameter_counts(self):
        with initialize(version_base=None, config_path=None):
            for task in self.tasks:
                if task.seed != '1000':
                    continue
                with self.subTest(task=task.task_id):
                    config = compose(config_name='config', overrides=[
                        f'env.name={task.env_name}', f'seed={task.seed}',
                        f'interaction.total_env_steps={task.steps}', *task_training_args(task),
                    ])
                    online.main.resolve_split_encoder_goal_dims(config)
                    conf = OmegaConf.to_container(config.agent, structured_config_mode=SCMode.INSTANTIATE)
                    state_dim, action_dim, _, _ = ENVIRONMENTS[task.env_name]
                    agent, _ = conf.make(env_spec=vector_env_spec(state_dim, action_dim), total_optim_steps=5)
                    if task_variant(task).startswith('QRL+'):
                        self.assertEqual(sum(p.numel() for p in agent.parameters()), int(task.params))
                        self.assertNotIsInstance(agent.critics[0].encoder, SplitEncoder)
                    else:
                        self.assertIsInstance(agent.critics[0].encoder, SplitEncoder)
                        self.assertEqual(conf.actor.losses.min_dist.replay_goal_candidates, 8)


if __name__ == '__main__':
    unittest.main()

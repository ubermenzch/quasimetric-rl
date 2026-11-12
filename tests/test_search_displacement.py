import unittest
from pathlib import Path

import torch

from quasimetric_rl.modules.actor.losses.min_dist import MinDistLoss
from tests.test_goal_set_distance import vector_env_spec
from tests import test_latent_goal_completion as completion_tests
from tools.generate_iclr_displacement_tasks import ENV_NAMES, LOG_KEY, SEEDS, generate_tasks
from tools.generate_selected_success_l_min32_tasks import extra_arg_map
from tools.run_qrl_queue import (
    DELETE_CHECKPOINTS_AFTER_COMPLETION_ARG, read_tasks,
    task_preserves_checkpoints_after_completion, task_training_args,
)


class SearchDisplacementTest(unittest.TestCase):
    def make_loss(self):
        loss, critic = completion_tests.LatentGoalAdamTest().make_loss_and_critic('min')
        loss.latent_goal_steps = 1
        loss.latent_goal_optim = 'sgd'
        loss.latent_goal_lr = 0.5
        loss.latent_goal_log_displacement = True
        return loss, critic

    def test_same_norm_vectors_have_nonzero_displacement(self):
        loss, critic = self.make_loss()
        initial = torch.tensor([[0.5, -0.25, 1., 0.]])
        predicted = torch.tensor([[0.5, -0.25, 0., 1.]])
        completed, info = loss._optimize_latent_goal(critic, predicted, initial)
        torch.testing.assert_close(completed, predicted)
        torch.testing.assert_close(info['latent_goal_initial_norm'], info['latent_goal_final_norm'])
        torch.testing.assert_close(info['latent_goal_final_displacement'], torch.tensor(2.).sqrt())
        torch.testing.assert_close(info['latent_goal_final_relative_displacement'], torch.tensor(2.).sqrt())
        torch.testing.assert_close(info['latent_goal_final_displacement_rms'], torch.tensor(1.))

    def test_retained_and_last_iterates_are_measured_separately(self):
        loss, critic = self.make_loss()
        loss.latent_goal_lr = 2.
        loss.latent_goal_keep_best = True
        initial = torch.tensor([[0.5, -0.25, 1., 0.]])
        predicted = torch.tensor([[0.5, -0.25, 0., 1.]])
        completed, info = loss._optimize_latent_goal(critic, predicted, initial)
        torch.testing.assert_close(completed, initial)
        self.assertEqual(info['latent_goal_final_displacement'].item(), 0.)
        torch.testing.assert_close(info['latent_goal_last_displacement'], torch.tensor(32.).sqrt())

    def test_logging_preserves_search_outputs_existing_metrics_and_rng(self):
        loss, critic = self.make_loss()
        loss.latent_goal_log_displacement = False
        initial = torch.tensor([[0.5, -0.25, 1., 0.], [0.5, -0.25, 0., 0.]])
        predicted = torch.tensor([[0.5, -0.25, 0., 1.], [0.5, -0.25, 1., 2.]])
        plain, previous = loss._optimize_latent_goal(critic, predicted, initial)
        loss.latent_goal_log_displacement = True
        rng = torch.get_rng_state().clone()
        completed, info = loss._optimize_latent_goal(critic, predicted, initial)
        torch.testing.assert_close(completed, plain, rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        for key, value in previous.items():
            torch.testing.assert_close(info[key], value, rtol=0, atol=0)
        for label in ('final', 'last'):
            self.assertTrue(torch.isfinite(info[f'latent_goal_{label}_relative_displacement']))
        self.assertEqual(info['latent_goal_initial_norm_near_zero_fraction'].item(), 0.5)

    def test_configuration_enables_logging(self):
        config = MinDistLoss.Conf(latent_goal_mode='min', latent_goal_log_displacement=True)
        self.assertTrue(config.make(vector_env_spec()).latent_goal_log_displacement)

    def test_task_grid_matches_main_table_and_preserves_checkpoints(self):
        existing = read_tasks(Path('runs/qrl_queue/tasks.tsv'))
        tasks = generate_tasks(existing)
        self.assertEqual(len(tasks), 9)
        self.assertEqual({t.env_name for t in tasks}, set(ENV_NAMES))
        self.assertEqual({int(t.seed) for t in tasks}, set(SEEDS))
        self.assertEqual({t.steps for t in tasks}, {'200000', '200400'})
        for task in tasks:
            self.assertEqual(int(task.steps), 200400 if task.env_name=='PointMaze_Medium-v3' else 200000)
            self.assertEqual(extra_arg_map(task)[LOG_KEY], 'true')
            self.assertTrue(task_preserves_checkpoints_after_completion(task))
            self.assertFalse(any(arg.startswith(DELETE_CHECKPOINTS_AFTER_COMPLETION_ARG)
                                 for arg in task_training_args(task)))


if __name__ == '__main__':
    unittest.main()

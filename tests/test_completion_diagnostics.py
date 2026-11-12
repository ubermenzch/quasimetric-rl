import copy
import json
from pathlib import Path
import tempfile
import types
import unittest

import numpy as np
import torch

from quasimetric_rl.data import Dataset
from quasimetric_rl.modules.actor.losses.min_dist import MinDistLoss
from tests.test_goal_set_distance import vector_env_spec
from tests.test_iclr_simple_goal_tasks import ReplayGoalSelectionTest
from tools.rebuttal_diagnostics import RunDiagnostics
from tools.run_qrl_queue import (
    Task, archive_logs_only_attempt, task_identity_issue, write_task_manifest,
)


class CompletionDiagnosticsTest(unittest.TestCase):
    def test_future_step_metadata_matches_the_observed_trajectory(self):
        class TinyDataset:
            __getitem__ = Dataset.__getitem__
            future_observation_discount = .9
            max_episode_length = 4
            transition_history_length = 0
            indices_to_episode_indices = torch.zeros(4, dtype=torch.long)
            indices_to_episode_timesteps = torch.arange(4)
            raw_data = types.SimpleNamespace(
                episode_lengths=torch.tensor([4]),
                all_observations=torch.arange(20.).reshape(5, 4),
                actions=torch.zeros(4, 2), rewards=torch.zeros(4),
                terminals=torch.zeros(4, dtype=torch.bool), timeouts=torch.zeros(4, dtype=torch.bool),
            )
            def get_observations(self, indices):
                return self.raw_data.all_observations[indices]
        sample = TinyDataset()[torch.tensor([0, 2, 3])]
        future_indices = sample.replay_timesteps + sample.future_steps
        torch.testing.assert_close(sample.future_observations[:, 0], future_indices.float() * 4)
        self.assertTrue((sample.future_steps >= 1).all())
        self.assertTrue((future_indices <= 4).all())

    def make_loss_and_critic(self):
        _, critic = ReplayGoalSelectionTest().make_loss_and_critic()
        loss = MinDistLoss(env_spec=vector_env_spec(), adaptive_entropy_regularizer=False,
                           add_goal_as_future_state=False, latent_goal_mode='min',
                           latent_goal_steps=32, latent_goal_lr=.1, latent_goal_optim='sgd',
                           latent_goal_keep_best=True, latent_goal_log_displacement=True,
                           latent_goal_probe_replay=True, latent_goal_probe_interval=2)
        return loss, critic

    def test_reference_sampling_is_independent_and_without_replacement(self):
        class Bank:
            sample_observation_reference = Dataset.sample_observation_reference
            raw_data = types.SimpleNamespace(all_observations=torch.arange(80.).reshape(20, 4))
            num_observations_available = 10
        before = torch.get_rng_state().clone()
        sample = Bank().sample_observation_reference(num_samples=8, seed=17)
        torch.testing.assert_close(torch.get_rng_state(), before)
        self.assertEqual(len(torch.unique(sample, dim=0)), 8)
        self.assertLess(sample.max(), 40)

    def test_excludes_self_matches_and_keeps_probes_available_for_scalar_logging(self):
        loss, critic = self.make_loss_and_critic()
        reference = torch.tensor([[0., 0., 1., 1.], [0., 0., 2., 1.]])
        raw = reference[:1]
        final = torch.tensor([[0., 0., 10., 10.]])
        loss.set_replay_reference_provider(lambda **kwargs: reference, seed=12)
        probes = []
        loss.set_diagnostic_sink(lambda *args: probes.append(args))
        first = loss._completion_support(critic, raw, raw, final)
        self.assertEqual(first['latent_goal_support_full_initial_l2'].item(), 1.)
        self.assertEqual(first['latent_goal_support_self_matches_excluded'].item(), 1)
        self.assertEqual(first['latent_goal_support_full_increased_fraction'].item(), 1.)
        second = loss._completion_support(critic, raw, raw, final)
        self.assertIs(first, second)
        self.assertEqual(second['latent_goal_support_probe_update'].item(), 1)
        third = loss._completion_support(critic, raw, raw, final)
        self.assertEqual(third['latent_goal_support_probe_update'].item(), 3)
        self.assertEqual(len(probes), 2)
        torch.testing.assert_close(probes[0][2]['reference_states'], reference)

    def test_full_search_trace_is_recorded_without_changing_search_or_rng(self):
        loss, critic = self.make_loss_and_critic()
        initial = torch.tensor([[0., 0., 1., 1.], [1., 1., 2., 3.]])
        predicted = torch.zeros_like(initial)
        before = torch.get_rng_state().clone()
        completed, stats = loss._optimize_latent_goal(critic, predicted, initial)
        trace = loss._last_search_probe
        self.assertEqual(trace['search_h'].shape, (33, 2, 2))
        self.assertEqual(trace['search_cost'].shape, (33, 2))
        torch.testing.assert_close(torch.get_rng_state(), before)
        loss.latent_goal_probe_replay = False
        plain, _ = loss._optimize_latent_goal(critic, predicted, initial)
        torch.testing.assert_close(plain, completed, rtol=0, atol=0)
        for step in (0, 1, 2, 4, 8, 16, 32):
            torch.testing.assert_close(stats[f'latent_goal_cost_step_{step:02d}'],
                                       trace['search_cost'][step].mean())

    def test_compressed_probe_retains_arrays_and_metadata_and_nonfinite_markers(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = RunDiagnostics(tmp, protocol={'seed': 1000})
            log.record_probe('completion', 1, {'h': torch.arange(6.).reshape(2, 3)}, env_steps=500)
            log.record_training(500, 1, {'cost': torch.tensor(float('nan'))})
            path = Path(tmp) / 'diagnostics/probes/completion_00000001.npz'
            with np.load(path, allow_pickle=False) as data:
                np.testing.assert_array_equal(data['h'], np.arange(6.).reshape(2, 3))
                self.assertEqual(json.loads(str(data['metadata_json']))['env_steps'], 500)
            entry = json.loads((Path(tmp) / 'diagnostics/training.jsonl').read_text())
            self.assertIsNone(entry['metrics']['cost'])

    def test_logs_only_restarts_archive_partial_attempts(self):
        task = Task('retry', 'online', 'FetchPush', '1000', '100', '4m',
                    'queue.native_qrl=true queue.logs_only=true')
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / task.task_id
            write_task_manifest(output, task)
            (output / 'eval.log').write_text('old attempt\n')
            self.assertEqual(task_identity_issue({}, output, task), '')
            archive = archive_logs_only_attempt(task, output, {'state': 'PENDING'})
            self.assertFalse(output.exists())
            self.assertEqual((archive / 'eval.log').read_text(), 'old attempt\n')


if __name__ == '__main__':
    unittest.main()

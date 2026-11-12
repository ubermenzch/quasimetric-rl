import copy
import csv
import json
from pathlib import Path
import random
import shlex
import tempfile
from types import SimpleNamespace
import unittest
import zipfile
from unittest.mock import Mock

import gym
import numpy as np
from omegaconf import OmegaConf
import torch

from online.dynamics_diagnostics import (
    DynamicsDiagnostics, DynamicsDiagnosticsConf, collect_fixed_probe,
    isolated_measurement, measure_critic, tensor_digest,
)
from quasimetric_rl import utils
from quasimetric_rl.data import EnvSpec
from quasimetric_rl.modules.quasimetric_critic.models import QuasimetricCritic
from quasimetric_rl.modules.quasimetric_critic.losses.latent_dynamics import LatentDynamicsLoss


def critic_and_data():
    conf = copy.deepcopy(QuasimetricCritic.Conf())
    conf.encoder.kind = 'split'
    conf.encoder.goal_dims = (0, 1)
    conf.encoder.arch = (8,)
    conf.encoder.goal_arch = (8,)
    conf.encoder.non_goal_arch = (8,)
    conf.encoder.latent_size = 4
    conf.encoder.goal_latent_size = 2
    conf.encoder.non_goal_latent_size = 2
    conf.quasimetric_model.projector_arch = (8,) * 4
    conf.quasimetric_model.projector_mlp_kind = 'residual'
    conf.quasimetric_model.projector_activation = 'silu'
    conf.quasimetric_model.quasimetric_head_spec = 'iqe(dim=8,components=2)'
    conf.latent_dynamics.arch = (8,)
    spec = EnvSpec(
        observation_space=gym.spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32),
        observation_space_is_dict=False,
        action_space=gym.spaces.Box(-1., 1., shape=(2,), dtype=np.float32),
    )
    critic = conf.make(env_spec=spec)
    data = {'observations': torch.randn(8, 5), 'actions': torch.randn(8, 2),
            'next_observations': torch.randn(8, 5)}
    return critic, data


def fake_replay(data):
    env = Mock()
    env.seed.side_effect = np.random.seed

    def rollout(actor, *, env):
        space = replay.env_spec.action_space
        actions = [actor(None, None, space) for _ in range(replay.episode_length)]
        return SimpleNamespace(all_observations=torch.tensor(np.random.randn(5, 5), dtype=torch.float32),
                               actions=torch.tensor(np.asarray(actions)))

    replay = SimpleNamespace(
        name='diagnostic_test', episode_length=4, num_transitions_realized=0,
        env_spec=SimpleNamespace(action_space=gym.spaces.Box(-1., 1., shape=(2,), dtype=np.float32)),
        create_env=lambda: env, collect_rollout=rollout,
        sample=lambda _: SimpleNamespace(**data),
    )
    return replay, env


class DynamicsDiagnosticsTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.critic, self.data = critic_and_data()
        self.loss = LatentDynamicsLoss(weight=.1, distance='iqe_mse', mse_weight=0, iqe_weight=1)

    def test_losses_gradients_and_branch_scaling_match_direct_calculation(self):
        stats, raw = measure_critic(self.critic, self.data, self.loss)
        pred, target = raw['predicted_latent'], raw['target_latent']
        self.assertAlmostEqual(stats['mse'], (pred - target).square().mean().item())
        self.assertAlmostEqual(stats['sq_dists'], raw['bidirectional_distances'].square().mean().item())
        torch.testing.assert_close(raw['mse_gradient_predicted_latent'], 2 * (pred - target) / pred.numel())
        for name, region in [('full', slice(None)), ('goal', slice(0, 2)), ('non_goal', slice(2, None))]:
            expected = (pred[:, region].double() - target[:, region].double()).square().mean().item()
            self.assertAlmostEqual(stats['branches'][name]['mse_per_transition']['mean'], expected)
        self.assertEqual(stats['gradients']['dynamics_parameters']['mse_effective_l2'], 0)
        self.assertGreater(stats['gradients']['dynamics_parameters']['mse_raw']['l2'], 0)
        self.assertTrue(stats['projector_activations'])

    def test_measurement_does_not_change_rng_gradients_modes_or_next_update(self):
        self.critic.train()
        self.critic.encoder.eval()
        for p in self.critic.parameters():
            p.grad = torch.randn_like(p)
        control, _ = critic_and_data()
        control.load_state_dict(self.critic.state_dict())
        control.train()
        control.encoder.eval()
        modes = [m.training for m in self.critic.modules()]
        grads = [p.grad.clone() for p in self.critic.parameters()]
        rng = utils.rng_state_dict()
        expected = (random.random(), np.random.rand(), torch.randn(3))
        utils.load_rng_state(rng)
        measure_critic(self.critic, self.data, self.loss)
        actual = (random.random(), np.random.rand(), torch.randn(3))
        self.assertEqual(actual[:2], expected[:2])
        torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
        self.assertEqual(modes, [m.training for m in self.critic.modules()])
        for p, grad in zip(self.critic.parameters(), grads):
            torch.testing.assert_close(p.grad, grad, rtol=0, atol=0)
        for model in (self.critic, control):
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            optimizer.zero_grad()
            x = model.encoder(self.data['observations'])
            y = model.encoder(self.data['next_observations'])
            prediction = model.predict_next_latent(x, self.data['actions'])
            model.quasimetric_model(prediction, y, bidirectional=True).square().mean().backward()
            optimizer.step()
        for p, expected_p in zip(self.critic.parameters(), control.parameters()):
            torch.testing.assert_close(p, expected_p, rtol=0, atol=0)

    def test_nonfinite_values_are_explicit_and_json_is_strict(self):
        self.data['observations'][0, 0] = float('nan')
        stats, raw = measure_critic(self.critic, self.data, self.loss)
        self.assertIsNone(stats['mse'])
        self.assertIn('observations', stats['nonfinite_tensors'])
        self.assertGreater(stats['prediction']['nonfinite_fraction'], 0)
        json.dumps(stats, allow_nan=False)
        self.assertTrue(torch.isnan(raw['observations']).any())

    def test_probe_is_shared_across_seeds_does_not_enter_replay_and_closes_env(self):
        replay, env = fake_replay(self.data)
        conf = DynamicsDiagnosticsConf(probe_episodes=2, samples_per_episode=3)
        rng = utils.rng_state_dict()
        first = collect_fixed_probe(replay, conf)
        self.assertEqual(replay.num_transitions_realized, 0)
        env.close.assert_called_once()
        torch.testing.assert_close(torch.get_rng_state(), rng['torch'])
        np.random.seed(456)
        torch.manual_seed(789)
        second = collect_fixed_probe(replay, conf)
        self.assertEqual(tensor_digest(first), tensor_digest(second))
        self.assertEqual(first['actions'].shape, (6, 2))

    def test_rng_restored_on_error(self):
        rng = utils.rng_state_dict()
        with self.assertRaisesRegex(RuntimeError, 'probe failed'):
            with isolated_measurement(self.critic):
                torch.rand(10)
                raise RuntimeError('probe failed')
        torch.testing.assert_close(torch.get_rng_state(), rng['torch'])

    def test_training_cycle_includes_every_update_and_counts_nonfinite(self):
        diagnostic = object.__new__(DynamicsDiagnostics)
        diagnostic.conf = DynamicsDiagnosticsConf(save_training_batches=True)
        diagnostic.training_values = {}
        diagnostic.training_start_optim = 11
        with tempfile.TemporaryDirectory() as directory:
            diagnostic.root = Path(directory)
            for value in (1., 3., float('nan')):
                diagnostic.observe_training_step({'critic_00': {'latent_dynamics': {
                    key: torch.tensor(value) for key in ('mse', 'sq_dists', 'dist_p2n', 'dist_n2p')
                }}})
            diagnostic.finish_training_cycle(1000, 13)
            result = json.loads((diagnostic.root / 'training_env00001000_opt00000013.json').read_text())
            self.assertEqual(result['first_optim_step'], 11)
            self.assertEqual(result['critics']['critic_00']['mse']['count'], 3)
            self.assertEqual(result['critics']['critic_00']['mse']['finite_count'], 2)
            self.assertEqual(result['critics']['critic_00']['mse']['mean'], 2.)
            raw = torch.load(diagnostic.root / 'training_env00001000_opt00000013.pth', weights_only=False)
            self.assertEqual(raw['first_optim_step'], 11)
            self.assertEqual(raw['metric_names'][0], 'mse')
            torch.testing.assert_close(raw['critics']['critic_00'][:2, 0], torch.tensor([1., 3.], dtype=torch.float64))
            self.assertTrue(torch.isnan(raw['critics']['critic_00'][2]).all())
            self.assertEqual(diagnostic.training_values, {})

    def test_checkpoint_snapshots_and_resume_discard_uncommitted_measurements(self):
        replay, _ = fake_replay(self.data)
        agent = torch.nn.Module()
        agent.critics = torch.nn.ModuleList([self.critic])
        trainer = SimpleNamespace(algorithm='qrl', agent=agent, replay=replay,
                                  losses=SimpleNamespace(critic_losses=[SimpleNamespace(latent_dynamics=self.loss)]))
        conf = DynamicsDiagnosticsConf(
            enabled=True, interval=5, probe_episodes=1, samples_per_episode=3,
            save_interval_models=True,
            source_manifests=('configs/figure2_m_residual_antnavigate_6seed.tsv',),
        )
        experiment_conf = OmegaConf.create({'agent': {'quasimetric_critic': {'model': {'latent_dynamics': {'kind': 'mlp'}}}}})
        with tempfile.TemporaryDirectory() as directory:
            diagnostic = DynamicsDiagnostics(conf, trainer, directory, start_env_steps=0,
                                             start_optim_steps=0, experiment_conf=experiment_conf)
            replay.num_transitions_realized = 10
            self.assertIsNone(diagnostic.record(4, 1))
            diagnostic.record(5, 1)
            saved_interval = torch.load(diagnostic.root / 'env00000005_opt00000001.pth', weights_only=False)
            self.assertIn('agent', saved_interval)
            self.assertFalse(saved_interval['summary']['checkpoint_measurement'])
            with zipfile.ZipFile(diagnostic.root / f'source_{diagnostic.provenance}.zip') as archive:
                for source in ('configs/model_size/cqrl/m_residual.yaml',
                               'configs/figure2_m_residual_antnavigate_6seed.tsv'):
                    self.assertEqual(archive.read(source), (Path(__file__).resolve().parents[1] / source).read_bytes())
            # The final validation can upgrade an ordinary sample to a checkpoint.
            diagnostic.record(5, 1, checkpoint=True)
            path = diagnostic.root / 'env00000005_opt00000001.pth'
            saved = torch.load(path, weights_only=False)
            self.assertIn('agent', saved)
            self.assertIn('replay', saved['panels'])
            diagnostic.record(10, 2)
            (diagnostic.root / 'training_env00000010_opt00000002.pth').touch()
            (diagnostic.root / 'env00000010_opt00000002.pth.tmp').touch()
            (diagnostic.root / 'validation_env00000010_opt00000002.pth').touch()
            resumed = DynamicsDiagnostics(conf, trainer, directory, start_env_steps=5,
                                          start_optim_steps=1, experiment_conf=experiment_conf)
            self.assertTrue(path.exists())
            self.assertFalse((diagnostic.root / 'env00000010_opt00000002.json').exists())
            self.assertFalse((diagnostic.root / 'training_env00000010_opt00000002.pth').exists())
            self.assertFalse((diagnostic.root / 'validation_env00000010_opt00000002.pth').exists())
            self.assertEqual(diagnostic.probe_hash, resumed.probe_hash)
            self.assertTrue((diagnostic.root / 'env00000000_opt00000000.pth').exists())

    def test_all_five_tasks_only_change_mse_and_enable_diagnostics(self):
        root = Path(__file__).resolve().parents[1]
        def rows(path):
            with path.open() as handle:
                return list(csv.reader((line for line in handle if not line.startswith('#') and line.strip()), delimiter='\t'))
        controls = {row[3]: row for row in rows(root / 'configs/cqrl_min32_lr1_main7_5seed.tsv')
                    if row[2] == 'AntNavigate-v4'}
        tasks = rows(root / 'configs/figure2_iqe_only_l_antnavigate_5seed.tsv')
        self.assertEqual({row[3] for row in tasks}, {str(seed) for seed in range(1000, 1005)})
        for row in tasks:
            self.assertEqual(row[1:6], controls[row[3]][1:6])
            actual = dict(arg.split('=', 1) for arg in shlex.split(row[6]))
            expected = dict(arg.split('=', 1) for arg in shlex.split(controls[row[3]][6]))
            self.assertEqual(actual.pop('dynamics_diagnostics.enabled'), 'true')
            for key in list(actual):
                if key.startswith('dynamics_diagnostics.'):
                    del actual[key]
            expected['agent.quasimetric_critic.losses.latent_dynamics.mse_weight'] = '0.0'
            self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()

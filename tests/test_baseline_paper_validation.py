import copy
from collections import Counter
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import gym
import numpy as np
import torch

from online.main import summarize_evaluation
from online.trainer import EvalEpisodeResult, InteractionConf
from quasimetric_rl.data.online.paper_validation import PaperGoalEnv, SPECS
from quasimetric_rl.data.online.paper_validation.reference import METAWORLD_REVISION, require_metaworld
from tools.check_baseline_paper_validation import task_config
from tools.generate_baseline_paper_validation_tasks import (
    EXPERIMENTS, append_verified, generate_tasks, render_tasks,
)
from tools.run_qrl_queue import build_command, read_tasks


class FakeOriginalEnv:
    def __init__(self):
        self.action_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)
        self.resets = 0

    def seed(self, seed):
        return [seed]

    def reset(self):
        self.resets += 1
        return dict(observation=np.array([.1, .2, .3, .4]),
                    desired_goal=np.full(4, self.resets / 10))

    def step(self, action):
        return dict(observation=np.array([.1, .2, .3, .4]),
                    desired_goal=np.full(4, .9)), np.bool_(False), False, {}

    def close(self):
        pass


class PaperValidationTest(unittest.TestCase):
    def test_twelve_unique_tasks_one_seed_excludes_qrl(self):
        tasks = generate_tasks()
        self.assertEqual(len(tasks), 12)
        self.assertEqual(len({t.task_id for t in tasks}), 12)
        self.assertEqual({t.seed for t in tasks}, {'1000'})
        self.assertEqual(Counter(e.algorithm for e in EXPERIMENTS),
                         dict(td_infonce=3, crl=3, gcsl=3, c_learning=3))

    def test_configs_parse_without_changing_baseline_defaults(self):
        from quasimetric_rl.modules import QRLConf
        for task, experiment in zip(generate_tasks(), EXPERIMENTS):
            cfg = task_config(task)
            self.assertEqual(cfg.agent.baselines, QRLConf().baselines)
            self.assertEqual(cfg.agent.algorithm, experiment.algorithm)
            self.assertEqual(cfg.interaction.total_env_steps, experiment.steps)
            self.assertNotIn('model_size=', task.extra_args)
            if experiment.kind == 'paper_validation':
                self.assertEqual(tuple(cfg.interaction.baseline_goal_dims), SPECS[experiment.name][3])
            else:
                self.assertIsNone(cfg.interaction.baseline_goal_dims)

    def test_main_suite_preserves_existing_runs_and_adds_missing_ready_tasks(self):
        from quasimetric_rl.modules import QRLConf
        original = generate_tasks()
        tasks = generate_tasks(suite='paper-main')
        self.assertEqual(tasks[:len(original)], original)
        self.assertEqual(len({task.task_id for task in tasks}), 21)
        self.assertEqual({task.env_name for task in tasks[len(original):14]},
                         {'FetchReach', 'CLearningSawyerDrawer'})
        for task in tasks[len(original):14]:
            cfg = task_config(task)
            self.assertEqual(cfg.agent.baselines, QRLConf().baselines)
            self.assertEqual(task.seed, '1000')
        drawer = task_config(tasks[13])
        self.assertEqual(tuple(drawer.interaction.baseline_goal_dims), tuple(range(6)))
        self.assertEqual(drawer.interaction.env_steps_per_cycle, 151)

    def test_seven_additions_use_task_specific_protocols(self):
        tasks = generate_tasks(suite='paper-main')[14:]
        self.assertEqual({t.env_name for t in tasks}, {
            'CRLSawyerPush', 'CRLSawyerBin', 'GCSLLunarLander', 'GCSLClaw',
            'CLearningJacoReach', 'CLearningFinger', 'CLearningHand',
        })
        configs = {t.env_name: task_config(t) for t in tasks}
        self.assertEqual(configs['GCSLClaw'].agent.baselines.gcbc.action_discretization, 'factorized')
        for name, discount, odds in (('JacoReach', .9, 2), ('Finger', .99, 50), ('Hand', .3, 2)):
            cfg = configs['CLearning' + name]
            self.assertEqual(cfg.agent.baselines.c_learning.discount, discount)
            self.assertEqual(cfg.agent.baselines.c_learning.odds_clip, odds)
        for task in tasks:
            cfg = configs[task.env_name]
            self.assertEqual(task.seed, '1000')
            self.assertNotIn('model_size=', task.extra_args)
            self.assertEqual(tuple(cfg.interaction.baseline_goal_dims), SPECS[task.env_name][3])

    def test_claw_distance_wraps_at_pi_for_numpy_and_torch(self):
        env = PaperGoalEnv.__new__(PaperGoalEnv)
        env.name = 'GCSLClaw'
        angles = np.array([np.pi - .02, 0., -np.pi + .02])
        goals = np.array([-np.pi + .02, np.pi, np.pi - .02])
        states = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
        targets = np.stack([np.cos(goals), np.sin(goals)], axis=-1)
        np.testing.assert_allclose(env.goal_distance(states, targets), [.04, np.pi, .04])
        torch.testing.assert_close(env.goal_distance(torch.tensor(states), torch.tensor(targets)),
                                   torch.tensor([.04, np.pi, .04], dtype=torch.float64))

    def test_coverage_counts_reconstructions_and_skipped_pen_separately(self):
        from tools.audit_baseline_paper_validation import audit
        tasks = generate_tasks(suite='paper-main')
        with tempfile.TemporaryDirectory() as tmp:
            queue = Path(tmp) / 'tasks.tsv'
            queue.write_text(render_tasks(tasks, suite='paper-main'))
            report = audit(queue, Path(tmp) / 'status')
        self.assertEqual(report['total_tasks'], 22)
        self.assertEqual(report['planned_tasks'], 21)
        self.assertEqual(report['skipped_tasks'], 1)
        self.assertEqual(report['reconstructed_protocol_tasks'], 3)
        self.assertEqual(report['counts'], {'PENDING': 21, 'SKIPPED_USER_REQUEST': 1})
        self.assertFalse(report['all_planned_complete'])

    def test_quick_task_preserves_crl_defaults_and_original_suite(self):
        from quasimetric_rl.modules import QRLConf
        original = generate_tasks()
        tasks = generate_tasks(suite='crl-quick')
        self.assertEqual(len(tasks), 1)
        task = tasks[0]
        self.assertNotIn(task.task_id, {t.task_id for t in original})
        cfg = task_config(task)
        self.assertEqual(cfg.agent.algorithm, 'crl')
        self.assertEqual(cfg.agent.baselines, QRLConf().baselines)
        self.assertEqual(cfg.env.name, 'CRLPointSmall')
        self.assertEqual(cfg.interaction.total_env_steps, 100_000)
        self.assertEqual(cfg.interaction.env_steps_per_cycle, 50)
        self.assertEqual(cfg.save_steps, 5_000)
        self.assertEqual(cfg.interaction.num_eval_episodes, 50)
        self.assertEqual(cfg.interaction.num_test_episodes, 200)
        self.assertEqual(tuple(cfg.interaction.baseline_goal_dims), (0, 1))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'tasks.tsv'
            path.write_text(render_tasks(tasks, suite='crl-quick'))
            self.assertEqual(read_tasks(path), tasks)
        self.assertEqual(generate_tasks(), original)

    def test_point_diagnostic_success_is_stricter_than_upstream_reward(self):
        base = FakeOriginalEnv()
        base.reset = lambda: np.array([0., 0., 1., 0.], dtype=np.float32)
        base.step = lambda action: (base.reset(), 1., False, {})
        with patch('quasimetric_rl.data.online.paper_validation.make_base', return_value=base):
            env = PaperGoalEnv('CRLPointSmall')
            obs = env.reset()
            np.testing.assert_array_equal(obs['observation'], [0., 0.])
            np.testing.assert_array_equal(obs['desired_goal'], [1., 0.])
            for step in range(50):
                _, reward, done, info = env.step(np.zeros(2, dtype=np.float32))
                self.assertEqual(reward, 1.)
                self.assertFalse(info['is_success'])
                self.assertEqual(info['goal_distance'], 1.)
                self.assertEqual(done, step == 49)
            base.reset = lambda: np.array([.9, 0., 1., 0.], dtype=np.float32)
            env.reset()
            self.assertTrue(env.step(np.zeros(2, dtype=np.float32))[3]['is_success'])

    def test_manifest_round_trip(self):
        tasks = generate_tasks()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'tasks.tsv'
            path.write_text(render_tasks(tasks))
            self.assertEqual(read_tasks(path), tasks)

    def test_scheduler_accepts_online_resume_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            for task in generate_tasks():
                command, _, _, _ = build_command({
                    'RESULTS_ROOT': tmp,
                    'RESUME_IF_POSSIBLE': '1',
                    'ONLINE_SAVE_REPLAY_BUFFER': 'True',
                }, task, '0')
                self.assertIn('save_replay_buffer=true', command)
                self.assertIn(f'env.name={task.env_name}', command)

    def test_append_is_idempotent_and_does_not_enqueue_blocked_tasks(self):
        tasks = generate_tasks()
        report = dict(tasks=[dict(task_id=t.task_id, status='PASS' if i < 9 else 'BLOCKED')
                             for i, t in enumerate(tasks)])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'tasks.tsv'
            path.write_text('# unrelated user queue content\n')
            self.assertEqual(append_verified(path, tasks, report), 9)
            first = path.read_bytes()
            self.assertTrue(first.startswith(b'# unrelated user queue content\n'))
            self.assertEqual(append_verified(path, tasks, report), 0)
            self.assertEqual(first, path.read_bytes())
            self.assertEqual(read_tasks(path), tasks[:9])
            changed = copy.deepcopy(tasks)
            changed[0].steps = '99'
            with self.assertRaises(ValueError):
                append_verified(path, changed, report)

    def test_newly_available_tasks_append_without_replacing_existing_tasks(self):
        tasks = generate_tasks()
        report = dict(tasks=[dict(task_id=t.task_id, status='PASS') for t in tasks])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'tasks.tsv'
            path.write_text(render_tasks(tasks[:9]))
            before = path.read_bytes()
            self.assertEqual(append_verified(path, tasks, report), 3)
            self.assertTrue(path.read_bytes().startswith(before))
            self.assertEqual(read_tasks(path), tasks)
            self.assertEqual(append_verified(path, tasks, report), 0)

    def test_metaworld_loader_skips_legacy_registration_but_loads_task_module(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / 'metaworld'
            for relative in ('', 'envs', 'envs/mujoco'):
                directory = package / relative
                directory.mkdir(parents=True, exist_ok=True)
                (directory / '__init__.py').write_text('raise RuntimeError("legacy registration")\n')
            task_module = package / 'envs/mujoco/sawyer_xyz.py'
            task_module.write_text('ORIGINAL_TASK_MODULE = True\n')
            with patch.dict(os.environ, QRL_C_LEARNING_METAWORLD=str(root)), \
                    patch.dict(sys.modules), patch.object(sys, 'path', sys.path.copy()), \
                    patch('quasimetric_rl.data.online.paper_validation.reference.subprocess.check_output',
                          return_value=METAWORLD_REVISION + '\n'):
                for name in list(sys.modules):
                    if name == 'metaworld' or name.startswith('metaworld.'):
                        del sys.modules[name]
                self.assertEqual(require_metaworld(), root)
                from metaworld.envs.mujoco import sawyer_xyz
                self.assertTrue(sawyer_xyz.ORIGINAL_TASK_MODULE)
                self.assertEqual(Path(sawyer_xyz.__file__), task_module)

    def test_gcsl_separate_goal_reset_and_fixed_horizon(self):
        with patch('quasimetric_rl.data.online.paper_validation.make_base', return_value=FakeOriginalEnv()):
            env = PaperGoalEnv('GCSLSawyerPush')
            obs = env.reset()
            self.assertEqual(env.base.resets, 2)
            np.testing.assert_allclose(obs['desired_goal'], .1)
            for step in range(50):
                obs, reward, done, info = env.step(np.zeros(2))
                np.testing.assert_allclose(obs['desired_goal'], .1)
                self.assertIsInstance(reward, float)
                self.assertEqual(done, step == 49)
                self.assertAlmostEqual(info['goal_distance'], np.hypot(.2, .3), places=6)

    def test_success_coordinates_differ_from_conditioning_coordinates(self):
        self.assertEqual(SPECS['GCSLSawyerPush'][3], (0, 1, 2, 3))
        self.assertEqual(SPECS['GCSLSawyerPush'][4], (2, 3))
        self.assertEqual(SPECS['GCSLSawyerDoor'][3], (3,))
        self.assertEqual(SPECS['CRLAntUMaze'][3], (0, 1))
        self.assertIsNone(InteractionConf().baseline_goal_dims)

    def test_metrics_distinguish_any_success_final_success_and_pre_action_final(self):
        result = EvalEpisodeResult.from_timestep_reward_is_success(
            torch.zeros(2, 2), torch.tensor([[True, False], [False, False]]),
        )
        result.timestep_goal_distance = torch.tensor([[1., .01, 2.], [2., 1., 3.]])
        result.reference_final_goal_distance = torch.tensor([.01, 1.])
        result.reference_final_is_success = torch.tensor([True, False])
        summary = summarize_evaluation(result, split='test', seed=1, env_steps=0,
                                       optim_steps=0, episode_length=2)
        self.assertEqual(summary['succ_rate'], .5)
        self.assertEqual(summary['final_success_rate'], 0)
        self.assertEqual(summary['final_goal_distance'], 2.5)
        self.assertAlmostEqual(summary['reference_final_goal_distance'], .505)
        self.assertEqual(summary['reference_final_success_rate'], .5)


if __name__ == '__main__':
    unittest.main()

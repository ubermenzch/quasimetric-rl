"""Real simulator checks for the seven adapters; source tools/qrl_env.sh first."""

import io
import unittest

import gym
import numpy as np
import torch

from quasimetric_rl.data.online.paper_validation import PaperGoalEnv, SPECS
from tools.check_baseline_paper_validation import task_config
from tools.generate_baseline_paper_validation_tasks import generate_tasks


class RemainingPaperEnvironmentTest(unittest.TestCase):
    def test_gcsl_native_goal_sampling_and_distance(self):
        for name in ('GCSLLunarLander', 'GCSLClaw'):
            with self.subTest(name=name):
                env = PaperGoalEnv(name)
                try:
                    env.seed(123)
                    obs = env.reset()
                    goal = env._sampled_goal.copy()
                    np.testing.assert_allclose(obs['desired_goal'][list(env.conditioning_dims)],
                                               env.base.extract_goal(goal))
                    for step in range(env.episode_length):
                        obs, _, done, info = env.step(env.action_space.sample())
                        expected = env.base.goal_distance(obs['observation'], goal)
                        self.assertAlmostEqual(info['goal_distance'], float(expected), places=5)
                        self.assertEqual(done, step == env.episode_length - 1)
                        np.testing.assert_array_equal(goal, env._sampled_goal)
                finally:
                    env.close()

    def test_dmc_first_reset_supplies_goal_second_supplies_initial_state(self):
        for name in ('CLearningJacoReach', 'CLearningFinger'):
            with self.subTest(name=name):
                env = PaperGoalEnv(name)
                try:
                    env.seed(321)
                    expected_goal = env.base.state(env.base.env.reset())
                    expected_start = env.base.state(env.base.env.reset())
                    env.seed(321)
                    obs = env.reset()
                    dims = list(env.conditioning_dims)
                    np.testing.assert_allclose(obs['observation'], expected_start, atol=1e-6)
                    np.testing.assert_allclose(obs['desired_goal'][dims], expected_goal[dims], atol=1e-6)
                    non_goal = sorted(set(range(len(expected_goal))) - set(dims))
                    np.testing.assert_array_equal(obs['desired_goal'][non_goal], 0)
                    for step in range(env.episode_length):
                        obs, _, done, _ = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
                        self.assertEqual(done, step == env.episode_length - 1)
                        np.testing.assert_allclose(obs['desired_goal'][dims], expected_goal[dims], atol=1e-6)
                    self.assertEqual(env.protocol_metadata()['protocol']['fidelity'],
                                     'PAPER_PROTOCOL_RECONSTRUCTION')
                finally:
                    env.close()

    def test_hand_model_removal_and_fifty_action_goal(self):
        env = PaperGoalEnv('CLearningHand')
        try:
            model = env.base.env.model
            self.assertNotIn('door', model.body_names)
            self.assertNotIn('frame', model.body_names)
            self.assertNotIn('latch', model.joint_names)
            self.assertNotIn('door_hinge', model.joint_names)
            self.assertEqual((model.nq, model.nu), (28, 28))
            env.seed(444)
            expected_initial = env.base.env.reset().copy()
            for _ in range(50):
                expected_goal, _, _, _ = env.base.env.step(env.action_space.sample())
            env.seed(444)
            obs = env.reset()
            np.testing.assert_allclose(obs['desired_goal'], expected_goal, atol=1e-6)
            np.testing.assert_allclose(obs['observation'], expected_initial, atol=1e-6)
            self.assertGreater(np.linalg.norm(expected_goal - expected_initial), .01)
            self.assertTrue(env.protocol_metadata()['model']['modified_xml_sha256'])
        finally:
            env.close()

    def test_lunar_replay_roundtrip_preserves_scalar_integer_actions(self):
        task = next(t for t in generate_tasks(suite='paper-main') if t.env_name == 'GCSLLunarLander')
        cfg = task_config(task)
        cfg.env.init_num_transitions = 100
        cfg.env.increment_num_transitions = 100
        replay = cfg.env.make()
        restored = cfg.env.make()
        try:
            self.assertIsInstance(replay.env_spec.action_space, gym.spaces.Discrete)
            rollout = replay.collect_rollout(lambda state, goal, space: space.sample())
            replay.add_rollout(rollout, training=True)
            stream = io.BytesIO()
            torch.save(replay.state_dict(), stream)
            stream.seek(0)
            saved = torch.load(stream, weights_only=False)
            restored.load_state_dict(saved)
            self.assertEqual(saved['actions'].shape, (50,))
            self.assertEqual(saved['actions'].dtype, torch.int64)
            torch.testing.assert_close(restored.state_dict()['actions'], saved['actions'])
            self.assertEqual(restored.num_episodes_realized, 1)
        finally:
            replay.env.close()
            restored.env.close()

    def test_crl_metaworld_keeps_existing_c_learning_classes(self):
        import sys
        old = PaperGoalEnv('CLearningSawyerReach')
        try:
            original_module = sys.modules['metaworld.envs.mujoco.sawyer_xyz']
            for name in ('CRLSawyerPush', 'CRLSawyerBin'):
                env = PaperGoalEnv(name)
                try:
                    env.seed(1000)
                    env.reset()
                    obs, _, _, info = env.step(env.action_space.sample())
                    self.assertEqual(obs['observation'].shape, (SPECS[name][1],))
                    self.assertTrue(np.isfinite(info['goal_distance']))
                    self.assertIs(sys.modules['metaworld.envs.mujoco.sawyer_xyz'], original_module)
                    old.reset()
                    self.assertTrue(np.isfinite(old.step(old.action_space.sample())[1]))
                finally:
                    env.close()
        finally:
            old.close()


if __name__ == '__main__':
    unittest.main()

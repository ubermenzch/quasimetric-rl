from dataclasses import replace
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

import yaml

from tools.generate_qrl_original_config_tasks import (
    ORIGINAL_REPO, ORIGINAL_REVISION, SEEDS, generate_tasks, parameter_count,
)
from tools.run_qrl_queue import (
    TASK_CODE_REPO_ARG, TASK_CODE_REVISION_ARG, TASK_NATIVE_QRL_ARG,
    build_command, normalized_task_definition, task_fingerprint,
    task_identity_issue, task_training_args, write_task_manifest,
)
from tools.watch_qrl_queue import task_critic_count, task_variant
from tools.generate_qrl_paper_validation_tasks import (
    SETTINGS, generate_tasks as validation_tasks,
)


class QRLOriginalConfigTasksTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = generate_tasks()

    def test_two_repositories_have_five_matching_training_seeds(self):
        self.assertEqual(len({task.task_id for task in self.tasks}), 10)
        for group in (self.tasks[:5], self.tasks[5:]):
            self.assertEqual({int(task.seed) for task in group}, set(SEEDS))
            self.assertEqual({task.env_name for task in group}, {'FetchPush'})
            self.assertEqual({int(task.steps) for task in group}, {1000000})
            self.assertEqual({task.params for task in group}, {str(parameter_count())})
            for task in group:
                values = dict(arg.split('=', 1) for arg in shlex.split(task.extra_args))
                self.assertEqual(values['agent.num_critics'], '2')
                self.assertEqual(values['interaction.exploration_eps'], '0.3')
                self.assertEqual(values['interaction.num_prefill_episodes'], '200')
                self.assertEqual(values['interaction.num_samples_per_cycle'], '500')
                self.assertEqual(values['interaction.num_rollouts_per_cycle'], '10')
                self.assertEqual(values['interaction.num_eval_episodes'], '50')
                self.assertEqual(values['agent.actor.losses.min_dist.adaptive_entropy_regularizer'], 'true')
                self.assertEqual(values['agent.actor.losses.min_dist.add_goal_as_future_state'], 'true')

    def test_repository_identity_is_fingerprinted_but_not_passed_to_hydra(self):
        task = self.tasks[5]
        self.assertFalse(any(arg.startswith('queue.') for arg in task_training_args(task)))
        definition = normalized_task_definition(task)
        self.assertEqual(definition['execution'][TASK_CODE_REVISION_ARG], ORIGINAL_REVISION)
        changed = replace(task, extra_args=task.extra_args.replace(ORIGINAL_REVISION, 'f' * 40))
        self.assertNotEqual(task_fingerprint(task), task_fingerprint(changed))

    def test_monitor_labels_distinguish_code_repositories(self):
        self.assertEqual(task_variant(self.tasks[0]), 'QRL-OrigCfg/AQRL')
        self.assertEqual(task_variant(self.tasks[5]), 'QRL-OrigSrc')
        for task in self.tasks:
            self.assertEqual(task_critic_count(task), '2')

    def test_partial_original_run_is_not_silently_restarted(self):
        task = self.tasks[5]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            write_task_manifest(out, task)
            self.assertEqual(task_identity_issue({}, out, task), '')
            (out / 'config.yaml').write_text('seed: 1000\n')
            self.assertIn('cannot_resume', task_identity_issue({}, out, task))
            (out / 'COMPLETE').touch()
            self.assertEqual(task_identity_issue({}, out, task), '')

    @unittest.skipUnless((ORIGINAL_REPO / 'online/main.py').is_file(), 'original QRL checkout not present')
    def test_original_command_uses_pinned_source_without_aqrl_arguments(self):
        with tempfile.TemporaryDirectory() as tmp:
            cmd, env, cwd, _ = build_command({
                'RESULTS_ROOT': tmp, 'RESUME_IF_POSSIBLE': '1',
                'ONLINE_SAVE_REPLAY_BUFFER': 'True',
            }, self.tasks[5], '0')
            self.assertEqual(cwd, ORIGINAL_REPO)
            self.assertEqual(cmd[1:3], ['-m', 'online.main'])
            self.assertEqual(env['PYTHONPATH'].split(os.pathsep)[0], str(ORIGINAL_REPO))
            self.assertNotIn(str(ORIGINAL_REPO.parents[1]), env['PYTHONPATH'].split(os.pathsep))
            self.assertFalse(any(arg.startswith(('queue.', 'resume_if_possible=',
                                                'save_replay_buffer=', 'save_final_replay_buffer=',
                                                '+qrl_model_size=')) for arg in cmd))
            result = subprocess.run([*cmd, '--cfg', 'job'], cwd=cwd, env=env,
                                    capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stderr)
            config = yaml.safe_load(result.stdout)
            self.assertEqual(config['base_git_dir'], str(ORIGINAL_REPO))
            self.assertEqual(config['git_commit'], ORIGINAL_REVISION)
            self.assertEqual(config['agent']['num_critics'], 2)
            self.assertEqual(config['agent']['actor']['model']['arch'], [512, 512])
            self.assertEqual(config['agent']['quasimetric_critic']['model']['encoder']['latent_size'], 128)

    @unittest.skipUnless((ORIGINAL_REPO / 'online/main.py').is_file(), 'original QRL checkout not present')
    def test_aqrl_repository_restores_the_original_training_hyperparameters(self):
        with tempfile.TemporaryDirectory() as tmp:
            cmd, env, cwd, _ = build_command({'RESULTS_ROOT': tmp}, self.tasks[0], '0')
            self.assertNotEqual(cwd, ORIGINAL_REPO)
            result = subprocess.run([*cmd, '--cfg', 'job'], cwd=cwd, env=env,
                                    capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stderr)
            config = yaml.safe_load(result.stdout)
            agent = config['agent']
            self.assertEqual(agent['num_critics'], 2)
            self.assertEqual(agent['actor']['model']['arch'], [512, 512])
            self.assertEqual(agent['actor']['model']['input_mode'], 'raw')
            self.assertEqual(agent['quasimetric_critic']['model']['encoder']['kind'], 'standard')
            self.assertEqual(agent['quasimetric_critic']['model']['encoder']['latent_size'], 128)
            self.assertEqual(agent['quasimetric_critic']['losses']['latent_dynamics']['distance'], 'iqe')
            self.assertFalse(agent['quasimetric_critic']['losses']['separate_latent_dynamics'])
            self.assertEqual(config['interaction']['exploration_eps'], 0.3)


class QRLPaperValidationTasksTest(unittest.TestCase):
    def test_published_online_settings_and_reuse(self):
        tasks = validation_tasks()
        self.assertEqual(tuple(task.env_name for task in tasks), SETTINGS)
        self.assertEqual({task.seed for task in tasks}, {'1000'})
        self.assertEqual({task.steps for task in tasks}, {'1000000'})
        self.assertEqual(tasks[2], generate_tasks()[0])
        for task in tasks:
            tokens = shlex.split(task.extra_args)
            values = dict(token.split('=', 1) for token in tokens)
            self.assertEqual(len(tokens), len(values))
            self.assertEqual(values[TASK_CODE_REPO_ARG], '.')
            self.assertEqual(values['agent.num_critics'], '2')
            self.assertEqual(values['interaction.exploration_eps'], '0.3')
            if task.env_name.endswith('Image'):
                self.assertEqual(values['agent.quasimetric_critic.model.encoder.arch'], '[512]')
                self.assertEqual(values['agent.quasimetric_critic.losses.latent_dynamics.weight'], '10')
                self.assertEqual(values['interaction.num_samples_per_cycle'], '125')
            else:
                self.assertEqual(values['agent.quasimetric_critic.losses.latent_dynamics.weight'], '0.1')
                self.assertEqual(values['interaction.num_samples_per_cycle'], '500')

    def test_image_and_state_parameter_counts_match_instantiated_models(self):
        import gym
        import numpy as np
        import torch
        from quasimetric_rl.data import EnvSpec
        from quasimetric_rl.modules import QRLConf

        torch.set_num_threads(1)
        for image_based, state_dim in ((False, 10), (False, 25), (True, 1024)):
            with self.subTest(image_based=image_based, state_dim=state_dim):
                observation = (
                    gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8)
                    if image_based else gym.spaces.Box(-np.inf, np.inf, (state_dim,), dtype=np.float32)
                )
                spec = EnvSpec(observation_space=observation, observation_space_is_dict=True,
                               action_space=gym.spaces.Box(-1, 1, (4,), dtype=np.float32))
                config = QRLConf()
                if image_based:
                    config.quasimetric_critic.model.encoder.arch = (512,)
                agent, _ = config.make(env_spec=spec, total_optim_steps=10)
                actual = sum(p.numel() for p in agent.parameters() if p.requires_grad)
                self.assertEqual(actual, parameter_count(state_dim=state_dim, image_based=image_based))


if __name__ == '__main__':
    unittest.main()

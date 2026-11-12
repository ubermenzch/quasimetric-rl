import shlex
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path

import gym
import numpy as np
from hydra import compose, initialize
from omegaconf import OmegaConf, SCMode

import online.main  # Register the same Hydra schema used by the scheduler.
from quasimetric_rl.data.env_spec import EnvSpec
from quasimetric_rl.model_size import load_model_size_preset, qrl_agent_parameter_count
from quasimetric_rl.modules import QRLConf
from quasimetric_rl.modules.utils import ResidualMLP
from tools.generate_iclr_rebuttal_control_tasks import (
    CONTROL_PREFIXES, GROUP_SIZE, MSE_KEY, QRL_PRESET, QUEUE, SEEDS,
    SELECTED_ENV_NAMES, NO_MSE_ENV_NAMES, append_tasks, generate_tasks, restrict_pending_controls,
)
from tools.generate_selected_success_l_min32_tasks import extra_arg_map, task_line
from tools.run_qrl_queue import read_tasks
from tools.watch_qrl_queue import task_variant


class ICLRRebuttalControlTasksTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.existing = read_tasks(QUEUE)
        cls.by_id = {task.task_id: task for task in cls.existing}
        cls.tasks, cls.provenance = generate_tasks(cls.existing)

    def test_controls_use_their_own_tasks_budgets_and_three_seeds(self):
        backbone_budgets = {
            'FetchPickAndPlace': 200000,
            'reacher_hard': 200000, 'maze2d-large': 200000,
            'Pusher-v4': 500000, 'AntMaze_UMaze-v5': 400000,
            'PandaFlip-v3': 400000,
        }
        no_mse_budgets = {
            'AntNavigate-v4': 500000, 'FetchPush': 200000,
            'PandaPickAndPlaceJoints-v3': 400000, 'PandaPushJoints-v3': 400000,
            'PandaSlide-v3': 400000, 'Pusher-v4': 500000,
        }
        self.assertEqual(set(backbone_budgets), set(SELECTED_ENV_NAMES))
        self.assertEqual(set(no_mse_budgets), set(NO_MSE_ENV_NAMES))
        self.assertEqual(len({task.task_id for task in self.tasks}), 36)
        for start, budgets in ((0, backbone_budgets), (GROUP_SIZE, no_mse_budgets)):
            group = self.tasks[start:start + GROUP_SIZE]
            self.assertEqual(Counter(task.env_name for task in group),
                             {name: 3 for name in budgets})
            for name, budget in budgets.items():
                rows = [task for task in group if task.env_name == name]
                self.assertEqual({int(task.seed) for task in rows}, set(SEEDS))
                self.assertEqual({int(task.steps) for task in rows}, {budget})

    def test_only_requested_training_knob_changes(self):
        for task, entry in zip(self.tasks, self.provenance):
            source = self.by_id[entry['source_task_id']]
            before, after = extra_arg_map(source), extra_arg_map(task)
            changes = {key: value for key, value in after.items()
                       if before.get(key) != value}
            self.assertEqual(changes, entry['override_changes'])
            self.assertEqual(set(before), set(after))
            for field in ('mode', 'env_name', 'seed', 'steps'):
                self.assertEqual(getattr(source, field), getattr(task, field))

    def test_monitor_distinguishes_the_two_controls(self):
        self.assertEqual(task_variant(self.tasks[0]), 'QRL+AQRLBackbone')
        self.assertEqual(task_variant(self.tasks[GROUP_SIZE]), 'AQRL/0-NoMSE')

    def test_pruning_preserves_other_tasks_and_rejects_started_controls(self):
        selected = self.tasks[GROUP_SIZE]
        excluded = replace(selected, task_id=CONTROL_PREFIXES[1] + 'excluded_fetchpick',
                           env_name='FetchPickAndPlace')
        unrelated = replace(excluded, task_id='unrelated_fetchpush_control')
        backbone = replace(self.tasks[0], task_id=CONTROL_PREFIXES[0] + 'expanded_fetchpush',
                           env_name='FetchPush')
        tasks = [selected, excluded, unrelated, backbone]
        before = '# unchanged header\n' + '\n'.join(task_line(task) for task in tasks) + '\n'
        with tempfile.TemporaryDirectory() as tmp:
            status = Path(tmp)
            after, removed = restrict_pending_controls(before, tasks, status)
            self.assertEqual(removed, [excluded])
            self.assertEqual(after, '# unchanged header\n' + task_line(selected) + '\n'
                             + task_line(unrelated) + '\n' + task_line(backbone) + '\n')
            (status / f'{excluded.task_id}.status').write_text('state=RUNNING\n')
            with self.assertRaisesRegex(ValueError, 'not pending'):
                restrict_pending_controls(before, tasks, status)

    def test_hydra_resolves_every_environment_for_both_variants(self):
        with initialize(version_base=None, config_path=None):
            for task in self.tasks:
                if task.seed != '1000':
                    continue
                with self.subTest(task=task.task_id):
                    c = compose(config_name='config', overrides=[
                        f'env.name={task.env_name}', f'seed={task.seed}',
                        f'interaction.total_env_steps={task.steps}',
                        *shlex.split(task.extra_args),
                    ])
                    self.assertEqual(c.agent.algorithm, 'qrl')
                    self.assertEqual(c.agent.num_critics, 1)
                    self.assertTrue(c.agent.actor.losses.min_dist.add_goal_as_future_state)
                    self.assertFalse(c.agent.quasimetric_critic.losses.separate_latent_dynamics)
                    self.assertEqual(c.agent.quasimetric_critic.losses.latent_dynamics.weight, 0.1)
                    self.assertEqual(c.agent.actor.losses.actor_optim.lr, 3e-5)
                    self.assertEqual(c.agent.quasimetric_critic.losses.critic_optim.lr, 1e-4)
                    if 'AQRLBackbone' in task.task_id:
                        self.assertEqual(c.agent.quasimetric_critic.model.encoder.kind, 'standard')
                        self.assertEqual(c.agent.actor.model.input_mode, 'raw')
                        self.assertEqual(c.agent.actor.losses.min_dist.latent_goal_mode, 'none')
                        self.assertEqual(c.agent.quasimetric_critic.losses.latent_dynamics.distance, 'iqe')
                    else:
                        self.assertEqual(c.agent.quasimetric_critic.model.encoder.kind, 'split')
                        self.assertEqual(c.agent.actor.model.input_mode, 'split_latent')
                        self.assertEqual(c.agent.actor.losses.min_dist.latent_goal_steps, 0)
                        self.assertEqual(c.agent.quasimetric_critic.losses.latent_dynamics.mse_weight, 0)

    def test_backbone_uses_aqrl_residual_modules_and_correct_parameter_count(self):
        preset = load_model_size_preset('qrl', QRL_PRESET)
        original = load_model_size_preset('qrl', 'l')
        self.assertEqual(original.quasimetric_critic.model.encoder.mlp_kind
                         if 'mlp_kind' in original.quasimetric_critic.model.encoder else 'plain', 'plain')
        aqrl = load_model_size_preset('cqrl', 'l')
        self.assertEqual(preset.actor.model.arch, aqrl.actor.model.arch)
        self.assertEqual(preset.quasimetric_critic.model.quasimetric_model,
                         aqrl.quasimetric_critic.model.quasimetric_model)
        conf = OmegaConf.to_container(OmegaConf.merge(
            OmegaConf.structured(QRLConf()), preset,
        ), structured_config_mode=SCMode.INSTANTIATE)
        env = EnvSpec(
            observation_space=gym.spaces.Box(-np.inf, np.inf, (29,), dtype=np.float32),
            observation_space_is_dict=True,
            action_space=gym.spaces.Box(-1, 1, (8,), dtype=np.float32),
        )
        agent, _ = conf.make(env_spec=env, total_optim_steps=1)
        counts = sum(parameter.numel() for parameter in agent.parameters()
                     if parameter.requires_grad)
        self.assertEqual(counts, qrl_agent_parameter_count(29, 8, QRL_PRESET))
        self.assertGreaterEqual(sum(isinstance(module, ResidualMLP)
                                   for module in agent.modules()), 4)

    def test_append_preserves_existing_rows_and_detects_concurrent_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            queue = Path(tmp) / 'tasks.tsv'
            original = '# existing\n' + task_line(self.existing[0]) + '\n'
            queue.write_text(original)
            append_tasks(queue, original, self.tasks[:1])
            self.assertTrue(queue.read_text().startswith(original))
            self.assertEqual(read_tasks(queue), [self.existing[0], self.tasks[0]])
            with self.assertRaises(RuntimeError):
                append_tasks(queue, original, self.tasks[1:2])


if __name__ == '__main__':
    unittest.main()

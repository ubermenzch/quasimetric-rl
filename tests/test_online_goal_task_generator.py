import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.continue_online_task import (
    latest_online_checkpoint_step,
    make_continuation_task,
)
from tools.generate_online_goal_env_tasks import (
    ENVIRONMENT_DIMS,
    ENVIRONMENTS,
    MODEL_SIZE_LEVEL,
    TRAINING_SEEDS,
    VARIANTS,
    evaluation_episode_count,
    generate_tasks,
)


class OnlineGoalTaskGeneratorTest(unittest.TestCase):
    def test_generates_complete_unique_matrix(self):
        tasks = generate_tasks()
        self.assertEqual(len(tasks), 275)
        self.assertEqual(len({task.task_id for task in tasks}), 275)
        self.assertEqual(
            Counter(task.env_name for task in tasks),
            {name: 25 for _, name, _ in ENVIRONMENTS},
        )
        self.assertEqual(
            Counter(task.seed for task in tasks),
            {str(seed): 55 for seed in TRAINING_SEEDS},
        )
        for variant, _, _ in VARIANTS:
            if variant == 'Base':
                continue
            self.assertEqual(
                sum(f'_{variant}-' in task.task_id for task in tasks),
                55,
            )
        self.assertEqual(
            sum('-M_' in task.task_id for task in tasks), 275,
        )

    def test_all_tasks_use_requested_training_and_evaluation_settings(self):
        expected_episodes = {
            'FetchReach': 1000,
            'FetchPush': 200,
            'FetchSlide': 200,
            'FetchPickAndPlace': 200,
            'Reacher-v4': 200,
            'Pusher-v4': 200,
            'AntNavigate-v4': 100,
            'reacher_easy': 100,
            'reacher_hard': 100,
            'manipulator_bring_ball': 100,
            'manipulator_bring_peg': 100,
        }
        for task in generate_tasks():
            with self.subTest(task=task.task_id):
                self.assertEqual(task.mode, 'online')
                self.assertEqual(task.steps, '200000')
                self.assertRegex(task.params, r'^\d+\.\d+m$')
                args = task.extra_args.split()
                env_kind = next(
                    arg.split('=', 1)[1]
                    for arg in args if arg.startswith('env.kind=')
                )
                evaluation_episodes = evaluation_episode_count(
                    env_kind, task.env_name
                )
                self.assertEqual(
                    evaluation_episodes, expected_episodes[task.env_name]
                )
                self.assertIn('interaction.exploration_eps=0', args)
                self.assertIn(
                    f'interaction.num_eval_episodes={evaluation_episodes}', args
                )
                self.assertIn(
                    f'interaction.num_test_episodes={evaluation_episodes}', args
                )
                self.assertIn(
                    f'val{evaluation_episodes}_test{evaluation_episodes}_',
                    task.task_id,
                )
                self.assertIn('interaction.validation_seed=1000', args)
                self.assertIn('interaction.test_seed=2000000', args)
                self.assertIn('eval_steps=null', args)
                self.assertIn('save_steps=20000', args)
                self.assertIn('keep_only_latest_checkpoint=false', args)
                self.assertIn('save_replay_buffer=true', args)
                self.assertIn('save_final_replay_buffer=true', args)

        self.assertEqual(
            Counter(
                evaluation_episode_count(
                    next(
                        arg.split('=', 1)[1]
                        for arg in task.extra_args.split()
                        if arg.startswith('env.kind=')
                    ),
                    task.env_name,
                )
                for task in generate_tasks()
            ),
            {1000: 25, 200: 125, 100: 125},
        )

    def test_split_variants_have_expected_completion_method(self):
        tasks = generate_tasks()
        for task in tasks:
            if '_GO-QRL+' not in task.task_id:
                continue
            args = task.extra_args.split()
            self.assertIn('agent.actor.losses.min_dist.latent_goal_steps=4', args)
            self.assertIn('agent.actor.losses.min_dist.latent_goal_keep_best=true', args)
            self.assertIn('agent.actor.losses.min_dist.latent_goal_lr=0.01', args)
            self.assertIn('agent.actor.losses.min_dist.latent_goal_search=direct', args)
            self.assertNotIn('agent.actor.losses.min_dist.latent_goal_optim=adam', args)
            self.assertNotIn('agent.actor.losses.min_dist.latent_goal_search=residual', args)
            if '+LN+RMSG-' in task.task_id:
                self.assertIn(
                    'agent.quasimetric_critic.model.encoder.branch_normalization=layernorm',
                    args,
                )
                self.assertIn('agent.actor.losses.min_dist.latent_goal_optim=rmsg', args)
            else:
                self.assertIn(
                    'agent.quasimetric_critic.model.encoder.branch_normalization=none',
                    args,
                )
                self.assertIn('agent.actor.losses.min_dist.latent_goal_optim=sgd', args)

    def test_all_variants_use_m_model_size(self):
        self.assertEqual(MODEL_SIZE_LEVEL, 'm')
        for task in generate_tasks():
            is_go_qrl = '_GO-QRL+' in task.task_id
            if is_go_qrl:
                self.assertIn('-M_', task.task_id)
                self.assertIn('+go_qrl_model_size=m', task.extra_args.split())
            else:
                self.assertIn('_Base-M_', task.task_id)
                self.assertIn('+qrl_model_size=m', task.extra_args.split())


class ContinueOnlineTaskTest(unittest.TestCase):
    def test_continuation_keeps_identity_and_sets_new_absolute_target(self):
        original = generate_tasks()[0]
        continued = make_continuation_task(original, 300_000)
        self.assertEqual(continued.task_id, original.task_id)
        self.assertEqual(continued.steps, '300000')
        self.assertIn('resume_if_possible=True', continued.extra_args.split())
        self.assertIn('save_replay_buffer=True', continued.extra_args.split())
        self.assertEqual(
            continued.extra_args.split()[-1],
            'interaction.total_env_steps=300000',
        )

    def test_latest_checkpoint_uses_environment_step(self):
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            (output_dir / 'checkpoint_env00200000_opt00190000.pth').touch()
            (output_dir / 'checkpoint_env00300000_opt00290000.pth').touch()
            (output_dir / 'agent_checkpoint_env00400000_opt00390000.pth').touch()
            self.assertEqual(latest_online_checkpoint_step(output_dir), 300_000)

    def test_rejects_offline_continuation(self):
        task = generate_tasks()[0]
        task.mode = 'offline'
        with self.assertRaisesRegex(ValueError, 'Only online'):
            make_continuation_task(task, 300_000)


if __name__ == '__main__':
    unittest.main()

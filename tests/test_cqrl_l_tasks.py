import tempfile
import unittest
from collections import Counter
from pathlib import Path

from quasimetric_rl.model_size import load_model_size_preset
from tools.generate_cqrl_l_tasks import (
    ENVIRONMENTS,
    MODEL_SIZE_LEVEL,
    TRAINING_SEEDS,
    append_tasks,
    extra_arg_map,
    generate_tasks,
)
from tools.run_qrl_queue import read_tasks


class CQRLLTaskGeneratorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = generate_tasks()

    def test_generates_six_environments_with_five_seeds(self):
        self.assertEqual(len(self.tasks), 30)
        self.assertEqual(len({task.task_id for task in self.tasks}), 30)
        self.assertEqual(
            Counter(task.env_name for task in self.tasks),
            {environment.name: 5 for environment in ENVIRONMENTS},
        )
        for environment in ENVIRONMENTS:
            group = [
                task for task in self.tasks
                if task.env_name == environment.name
            ]
            self.assertEqual(
                {int(task.seed) for task in group}, set(TRAINING_SEEDS),
            )

    def test_all_tasks_use_requested_l_budget_and_inner1_protocol(self):
        environments = {environment.name: environment for environment in ENVIRONMENTS}
        for task in self.tasks:
            environment = environments[task.env_name]
            args = extra_arg_map(task)
            self.assertEqual(task.mode, 'online')
            self.assertEqual(task.steps, str(environment.total_steps))
            self.assertEqual(args['+cqrl_model_size'], MODEL_SIZE_LEVEL)
            self.assertNotIn('+go_qrl_model_size', args)
            self.assertEqual(
                args['agent.actor.losses.min_dist.latent_goal_mode'], 'max',
            )
            self.assertEqual(
                args['agent.actor.losses.min_dist.latent_goal_steps'], '1',
            )
            self.assertEqual(args['save_steps'], str(environment.save_steps))
            self.assertEqual(
                args['interaction.num_eval_episodes'],
                str(environment.validation_episodes),
            )
            self.assertEqual(
                args['interaction.num_test_episodes'],
                str(environment.test_episodes),
            )
            self.assertEqual(args['keep_only_latest_checkpoint'], 'true')
            self.assertEqual(
                args['keep_only_best_and_final_checkpoints'], 'true',
            )
            self.assertEqual(args['resume_if_possible'], 'true')

    def test_l_preset_is_the_residual_architecture(self):
        preset = load_model_size_preset('cqrl', MODEL_SIZE_LEVEL)
        self.assertEqual(preset.model_size, 'CQRL-L')
        self.assertEqual(
            preset.quasimetric_critic.model.encoder.mlp_kind, 'residual',
        )
        self.assertEqual(
            preset.quasimetric_critic.model.quasimetric_model.projector_mlp_kind,
            'residual',
        )
        self.assertEqual(
            preset.quasimetric_critic.model.latent_dynamics.mlp_kind,
            'residual',
        )
        self.assertEqual(preset.actor.model.mlp_kind, 'residual')

    def test_queue_append_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = Path(directory) / 'tasks.tsv'
            queue.write_text(
                '# task_id\tmode\tenv_name\tseed\tsteps\tparams\textra_args\n'
            )
            self.assertEqual(append_tasks(queue, self.tasks), 30)
            self.assertEqual(append_tasks(queue, self.tasks), 0)
            parsed = read_tasks(queue)
            self.assertEqual(len(parsed), 30)
            self.assertEqual(
                {task.task_id for task in parsed},
                {task.task_id for task in self.tasks},
            )


if __name__ == '__main__':
    unittest.main()

import tempfile
import unittest
from collections import Counter
from pathlib import Path

from tools.generate_go_qrl_hybrid_scale_tasks import (
    ENVIRONMENTS,
    GPU_OPTIMIZATION_ARGS,
    GPU_OPTIMIZATION_TAG,
    GPU_OPTIMIZED_LEVELS,
    MODEL_SCALES,
    PARTITION_GROUPS,
    TRAINING_SEEDS,
    append_tasks,
    extra_arg_map,
    generate_all_tasks,
    partition_tasks,
    task_group,
    task_scale,
    validate_partitions,
)


class GOQRLHybridScaleTaskGeneratorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = generate_all_tasks()

    def test_full_matrix_and_protocol(self):
        self.assertEqual(len(self.tasks), 125)
        self.assertEqual(len({task.task_id for task in self.tasks}), 125)
        self.assertEqual(
            Counter(task.env_name for task in self.tasks),
            {environment.name: 25 for environment in ENVIRONMENTS},
        )
        self.assertEqual(
            Counter(task_scale(task) for task in self.tasks),
            {scale.level: 25 for scale in MODEL_SCALES},
        )
        group_counts = Counter(task_group(task) for task in self.tasks)
        self.assertEqual(set(group_counts.values()), {5})

        for task in self.tasks:
            args = extra_arg_map(task)
            self.assertEqual(task.steps, '500000')
            self.assertIn(int(task.seed), TRAINING_SEEDS)
            self.assertEqual(args['batch_size'], '256')
            self.assertEqual(args['save_steps'], '50000')
            self.assertEqual(args['eval_steps'], 'null')
            self.assertEqual(args['interaction.validation_seed'], '1000')
            self.assertEqual(args['interaction.num_eval_episodes'], '500')
            self.assertEqual(args['interaction.test_seed'], '1500')
            self.assertEqual(args['interaction.num_test_episodes'], '1000')
            self.assertEqual(
                args['agent.quasimetric_critic.losses.latent_dynamics.distance'],
                'iqe_mse',
            )
            self.assertNotIn(
                'agent.quasimetric_critic.model.quasimetric_model.'
                'projector_activation',
                args,
            )
            scale = task_scale(task)
            if scale in GPU_OPTIMIZED_LEVELS:
                self.assertIn(f'_{GPU_OPTIMIZATION_TAG}_', task.task_id)
                for optimization_arg in GPU_OPTIMIZATION_ARGS:
                    key, value = optimization_arg.split('=', 1)
                    self.assertEqual(args[key], value)
            else:
                self.assertNotIn(f'_{GPU_OPTIMIZATION_TAG}_', task.task_id)
                for optimization_arg in GPU_OPTIMIZATION_ARGS:
                    key = optimization_arg.split('=', 1)[0]
                    self.assertNotIn(key, args)

    def test_partitions_are_disjoint_complete_and_keep_seed_groups(self):
        validate_partitions(self.tasks)
        expected_sizes = {'server_2x': 65, 'local_1x': 30, 'server_1x': 30}
        partition_ids = {}
        for name, size in expected_sizes.items():
            partition = partition_tasks(self.tasks, name)
            self.assertEqual(len(partition), size)
            self.assertEqual(
                {task_group(task) for task in partition},
                set(PARTITION_GROUPS[name]),
            )
            for group in PARTITION_GROUPS[name]:
                self.assertEqual(
                    {int(task.seed) for task in partition if task_group(task) == group},
                    set(TRAINING_SEEDS),
                )
            partition_ids[name] = {task.task_id for task in partition}
        self.assertEqual(
            set().union(*partition_ids.values()),
            {task.task_id for task in self.tasks},
        )

    def test_queue_append_is_idempotent(self):
        local_tasks = partition_tasks(self.tasks, 'local_1x')
        with tempfile.TemporaryDirectory() as directory:
            queue = Path(directory) / 'tasks.tsv'
            queue.write_text(
                '# task_id\tmode\tenv_name\tseed\tsteps\tparams\textra_args\n'
            )
            self.assertEqual(append_tasks(queue, local_tasks, 'local_1x'), 30)
            self.assertEqual(append_tasks(queue, local_tasks, 'local_1x'), 0)
            task_lines = [
                line for line in queue.read_text().splitlines()
                if line and not line.startswith('#')
            ]
            self.assertEqual(len(task_lines), 30)


if __name__ == '__main__':
    unittest.main()

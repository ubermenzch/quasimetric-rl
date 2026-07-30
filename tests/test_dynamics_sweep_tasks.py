import unittest
from collections import Counter, defaultdict

from tools.generate_go_qrl_dynamics_sweep_tasks import (
    COMPLETED_CONTROL_ENVIRONMENTS,
    ENVIRONMENTS,
    TRAINING_SEEDS,
    VARIANTS,
    generate_all_tasks,
    partition_tasks,
    task_variant_code,
    task_workload,
)


def task_args(extra_args: str) -> dict[str, str]:
    return dict(argument.split('=', 1) for argument in extra_args.split())


class DynamicsSweepTasksTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = generate_all_tasks()

    def test_complete_matrix_with_completed_controls_omitted(self):
        self.assertEqual(len(self.tasks), 312)
        environment_counts = Counter(task.env_name for task in self.tasks)
        for environment in ENVIRONMENTS:
            expected = (
                11 * len(TRAINING_SEEDS)
                if environment.name in COMPLETED_CONTROL_ENVIRONMENTS
                else 12 * len(TRAINING_SEEDS)
            )
            self.assertEqual(environment_counts[environment.name], expected)

        grouped_seeds = defaultdict(set)
        for task in self.tasks:
            grouped_seeds[(task.env_name, task_variant_code(task))].add(task.seed)
        expected_seeds = {str(seed) for seed in TRAINING_SEEDS}
        self.assertTrue(grouped_seeds)
        self.assertTrue(all(seeds == expected_seeds for seeds in grouped_seeds.values()))

        for environment in COMPLETED_CONTROL_ENVIRONMENTS:
            self.assertNotIn((environment, 'A01'), grouped_seeds)

    def test_variants_encode_the_expected_factorial(self):
        expected = {
            (
                str(variant.separate).lower(),
                variant.distance,
                variant.activation,
            )
            for variant in VARIANTS
        }
        actual = set()
        for task in self.tasks:
            args = task_args(task.extra_args)
            actual.add((
                args[
                    'agent.quasimetric_critic.losses.separate_latent_dynamics'
                ],
                args[
                    'agent.quasimetric_critic.losses.latent_dynamics.distance'
                ],
                args[
                    'agent.quasimetric_critic.model.quasimetric_model.'
                    'projector_activation'
                ],
            ))
            self.assertEqual(args['save_steps'], '20000')
            self.assertEqual(args['save_replay_buffer'], 'true')
            self.assertEqual(args['resume_if_possible'], 'true')
        self.assertEqual(actual, expected)

    def test_partitions_are_disjoint_balanced_and_complete(self):
        expected = {
            'local': (156, 52_800_000),
            'server_2': (78, 26_400_000),
            'server_3': (78, 26_400_000),
        }
        partition_ids = {}
        for partition, (count, workload) in expected.items():
            tasks = partition_tasks(self.tasks, partition)
            self.assertEqual(len(tasks), count)
            self.assertEqual(task_workload(tasks), workload)
            partition_ids[partition] = {task.task_id for task in tasks}
        self.assertFalse(partition_ids['local'] & partition_ids['server_2'])
        self.assertFalse(partition_ids['local'] & partition_ids['server_3'])
        self.assertFalse(partition_ids['server_2'] & partition_ids['server_3'])
        self.assertEqual(
            set().union(*partition_ids.values()),
            {task.task_id for task in self.tasks},
        )


if __name__ == '__main__':
    unittest.main()

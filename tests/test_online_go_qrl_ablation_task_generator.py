import unittest
from collections import Counter

from tools.generate_online_go_qrl_ablation_tasks import (
    COMMON_ARGS,
    ENVIRONMENTS,
    PARTITION_GROUPS,
    TRAINING_SEEDS,
    generate_all_tasks,
    partition_tasks,
    task_variant,
)


class OnlineGOQRLAblationTaskGeneratorTest(unittest.TestCase):
    def test_generates_the_95_task_matrix(self):
        tasks = generate_all_tasks()
        self.assertEqual(len(tasks), 95)
        self.assertEqual(len({task.task_id for task in tasks}), 95)
        variant_counts = Counter(task_variant(task) for task in tasks)
        self.assertEqual(variant_counts['Inner0'], 35)
        for variant in ('Min1', 'Min8', 'Max1', 'Max8'):
            self.assertEqual(variant_counts[variant], 15)

    def test_partitions_are_disjoint_and_cover_all_tasks(self):
        all_tasks = generate_all_tasks()
        partitions = {
            name: partition_tasks(all_tasks, name)
            for name in PARTITION_GROUPS
        }
        self.assertEqual(
            {name: len(tasks) for name, tasks in partitions.items()},
            {'local': 65, 'server_crl': 15, 'server_c_learning': 15},
        )
        ids = [task.task_id for tasks in partitions.values() for task in tasks]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {task.task_id for task in all_tasks})

    def test_all_tasks_use_m_preset_and_checkpoint_protocol(self):
        for task in generate_all_tasks():
            args = task.extra_args.split()
            for argument in COMMON_ARGS:
                self.assertIn(argument, args)
            self.assertIn(task.seed, {str(seed) for seed in TRAINING_SEEDS})
            self.assertIn(task.env_name, {env[1] for env in ENVIRONMENTS})
            self.assertEqual(task.steps, '200000')
            self.assertEqual(task.params, '4.4m')

    def test_inner0_is_a_real_zero_step_initial_completion(self):
        inner0 = [
            task for task in generate_all_tasks()
            if task_variant(task) == 'Inner0'
        ]
        self.assertEqual(len(inner0), 35)
        for task in inner0:
            args = task.extra_args.split()
            self.assertIn(
                'agent.actor.losses.min_dist.latent_goal_steps=0', args,
            )
            self.assertIn(
                'agent.actor.losses.min_dist.latent_goal_mode=min', args,
            )


if __name__ == '__main__':
    unittest.main()

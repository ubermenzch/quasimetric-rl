import unittest
from collections import Counter

from tools.generate_online_go_qrl_ablation_tasks import (
    COMMON_ARGS,
    ENVIRONMENTS,
    PARTITION_GROUPS,
    TABLE_ENVIRONMENTS,
    TABLE_REMOTE_GROUPS,
    TRAINING_SEEDS,
    VARIANTS,
    generate_all_tasks,
    generate_table_tasks,
    partition_table_tasks,
    partition_tasks,
    table_task_group,
    table_task_variant,
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

    def test_table_matrix_covers_all_eight_columns_for_every_variant(self):
        tasks = generate_table_tasks()
        self.assertEqual(
            len(tasks), len(VARIANTS) * len(TABLE_ENVIRONMENTS) * len(TRAINING_SEEDS),
        )
        self.assertEqual(len({task.task_id for task in tasks}), len(tasks))
        for variant, mode, inner_steps in VARIANTS:
            variant_tasks = [
                task for task in tasks if table_task_variant(task) == variant
            ]
            self.assertEqual(
                len(variant_tasks), len(TABLE_ENVIRONMENTS) * len(TRAINING_SEEDS),
            )
            for environment in TABLE_ENVIRONMENTS:
                group = [
                    task for task in variant_tasks
                    if task.env_name == environment.name
                    and task.steps == str(environment.total_steps)
                    and environment.slug in task.task_id
                ]
                self.assertEqual(len(group), len(TRAINING_SEEDS))
                for task in group:
                    args = task.extra_args.split()
                    self.assertIn(
                        f'+go_qrl_model_size={environment.model_size}', args,
                    )
                    self.assertIn(
                        'agent.actor.losses.min_dist.latent_goal_mode='
                        f'{mode}', args,
                    )
                    self.assertIn(
                        'agent.actor.losses.min_dist.latent_goal_steps='
                        f'{inner_steps}', args,
                    )

    def test_table_keeps_distinct_reacher_budgets_and_model_scales(self):
        tasks = generate_table_tasks()
        reacher = [task for task in tasks if task.env_name == 'reacher_hard']
        self.assertEqual(len(reacher), len(VARIANTS) * 2 * len(TRAINING_SEEDS))
        self.assertEqual(
            {task.steps for task in reacher}, {'100000', '200000'},
        )
        self.assertTrue(any(
            '+go_qrl_model_size=l' in task.extra_args
            for task in tasks if task.env_name == 'FetchSlide'
        ))
        self.assertTrue(any(
            '+go_qrl_model_size=l' in task.extra_args
            for task in tasks if task.env_name == 'Pusher-v4'
        ))
        self.assertTrue(any(
            '+go_qrl_model_size=l' in task.extra_args
            for task in tasks if task.env_name == 'AntNavigate-v4'
        ))

    def test_table_partitions_are_disjoint_balanced_and_complete(self):
        all_tasks = generate_table_tasks()
        local = partition_table_tasks(all_tasks, 'local_2x')
        remote = partition_table_tasks(all_tasks, 'remote_1x')
        self.assertEqual((len(local), len(remote)), (135, 65))
        local_ids = {task.task_id for task in local}
        remote_ids = {task.task_id for task in remote}
        self.assertFalse(local_ids & remote_ids)
        self.assertEqual(
            local_ids | remote_ids,
            {task.task_id for task in all_tasks},
        )
        self.assertEqual(
            {table_task_group(task) for task in remote},
            TABLE_REMOTE_GROUPS,
        )
        self.assertEqual(
            sum(int(task.steps) for task in local), 38_000_000,
        )
        self.assertEqual(
            sum(int(task.steps) for task in remote), 19_500_000,
        )
        for tasks in (local, remote):
            self.assertEqual(
                {table_task_variant(task) for task in tasks},
                {variant for variant, _mode, _steps in VARIANTS},
            )
            self.assertEqual(
                {(task.env_name, task.steps) for task in tasks},
                {(environment.name, str(environment.total_steps))
                 for environment in TABLE_ENVIRONMENTS},
            )


if __name__ == '__main__':
    unittest.main()

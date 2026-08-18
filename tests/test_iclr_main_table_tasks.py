import tempfile
from pathlib import Path
import unittest

from tools.generate_iclr_main_table_tasks import (
    ALGORITHMS,
    COMPLETED_GROUPS,
    ENVIRONMENTS,
    REMOTE_GROUPS,
    TRAINING_SEEDS,
    all_missing_groups,
    append_tasks,
    baseline_model_size,
    extra_arg_map,
    generate_all_tasks,
    local_groups,
    partition_tasks,
    task_group,
    validate_partitions,
)
from tools.run_qrl_queue import read_tasks


class ICLRMainTableTasksTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = generate_all_tasks()

    def test_generates_only_missing_five_seed_groups(self):
        self.assertEqual(len(self.tasks), 215)
        self.assertEqual(len(all_missing_groups()), 43)
        self.assertTrue(COMPLETED_GROUPS.isdisjoint(all_missing_groups()))
        self.assertEqual(len({task.task_id for task in self.tasks}), 215)
        for group in all_missing_groups():
            seeds = {
                int(task.seed)
                for task in self.tasks if task_group(task) == group
            }
            self.assertEqual(seeds, set(TRAINING_SEEDS))

    def test_partitions_are_disjoint_complete_and_balanced(self):
        validate_partitions(self.tasks)
        local = partition_tasks(self.tasks, 'local_2x')
        remote = partition_tasks(self.tasks, 'remote_1x')
        self.assertEqual((len(local), len(remote)), (145, 70))
        self.assertEqual((len(local_groups()), len(REMOTE_GROUPS)), (29, 14))
        local_ids = {task.task_id for task in local}
        remote_ids = {task.task_id for task in remote}
        self.assertFalse(local_ids & remote_ids)
        self.assertEqual(local_ids | remote_ids, {
            task.task_id for task in self.tasks
        })

    def test_evaluation_ranges_and_compact_checkpoint_policy(self):
        environments = {environment.key: environment for environment in ENVIRONMENTS}
        for task in self.tasks:
            _algorithm, environment_key = task_group(task)
            environment = environments[environment_key]
            args = extra_arg_map(task)
            if environment.total_steps == 500_000:
                expected = ('1000', '500', '1500', '1000', '50000')
            else:
                expected = ('1000', '1000', '2000', '1000', '20000')
            actual = (
                args['interaction.validation_seed'],
                args['interaction.num_eval_episodes'],
                args['interaction.test_seed'],
                args['interaction.num_test_episodes'],
                args['save_steps'],
            )
            self.assertEqual(actual, expected)
            self.assertEqual(args['keep_only_latest_checkpoint'], 'true')
            self.assertEqual(
                args['keep_only_best_and_final_checkpoints'], 'true',
            )
            self.assertEqual(args['save_replay_buffer'], 'true')
            self.assertEqual(args['save_final_replay_buffer'], 'true')
            self.assertEqual(args['resume_if_possible'], 'true')
            self.assertNotIn('queue.delete_checkpoints_after_completion', args)

    def test_uses_environment_specific_gcsl_l_presets(self):
        gcsl = next(
            algorithm for algorithm in ALGORITHMS if algorithm.key == 'gcsl'
        )
        levels = {
            environment.key: baseline_model_size(gcsl, environment)
            for environment in ENVIRONMENTS
            if environment.model_size == 'l'
        }
        self.assertEqual(levels, {
            'fetchslide': 'l',
            'pusher_v4': 'l_pusher',
            'antnavigate_v4': 'l_antnavigate',
        })
        gcsl_tasks = {
            task.env_name: extra_arg_map(task)
            for task in self.tasks
            if task_group(task)[0] == 'gcsl' and task.seed == '1000'
        }
        self.assertEqual(gcsl_tasks['Pusher-v4']['+gcsl_model_size'], 'l_pusher')
        self.assertEqual(
            gcsl_tasks['AntNavigate-v4']['+gcsl_model_size'],
            'l_antnavigate',
        )

    def test_append_is_idempotent(self):
        local = partition_tasks(self.tasks, 'local_2x')
        with tempfile.TemporaryDirectory() as directory:
            queue = Path(directory) / 'tasks.tsv'
            self.assertEqual(append_tasks(queue, local, 'local_2x'), 145)
            self.assertEqual(append_tasks(queue, local, 'local_2x'), 0)
            loaded = read_tasks(queue)
            self.assertEqual(len(loaded), 145)
            self.assertEqual(
                {task.task_id for task in loaded},
                {task.task_id for task in local},
            )


if __name__ == '__main__':
    unittest.main()

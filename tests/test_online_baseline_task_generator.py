import unittest
from collections import Counter

from tools.generate_online_baseline_tasks import (
    ALGORITHMS,
    ENVIRONMENTS,
    TRAINING_SEEDS,
    baseline_parameter_count,
    evaluation_episode_count,
    generate_tasks,
)


class OnlineBaselineTaskGeneratorTest(unittest.TestCase):
    def test_generates_four_by_seven_by_five_matrix(self):
        tasks = generate_tasks()
        self.assertEqual(
            len(tasks), len(ALGORITHMS) * len(ENVIRONMENTS) * len(TRAINING_SEEDS),
        )
        self.assertEqual(len({task.task_id for task in tasks}), len(tasks))
        self.assertEqual(
            Counter(task.env_name for task in tasks),
            {environment[1]: 20 for environment in ENVIRONMENTS},
        )
        self.assertEqual(
            Counter(task.seed for task in tasks),
            {str(seed): 28 for seed in TRAINING_SEEDS},
        )

    def test_tasks_use_checkpoint_bound_validation_and_test(self):
        for task in generate_tasks():
            args = task.extra_args.split()
            env_kind = next(
                value.split('=', 1)[1]
                for value in args if value.startswith('env.kind=')
            )
            episodes = evaluation_episode_count(env_kind, task.env_name)
            self.assertIn(f'interaction.num_eval_episodes={episodes}', args)
            self.assertIn(f'interaction.num_test_episodes={episodes}', args)
            self.assertIn('save_steps=20000', args)
            self.assertIn('eval_steps=null', args)
            self.assertIn('keep_only_latest_checkpoint=false', args)
            self.assertIn('save_replay_buffer=true', args)
            self.assertIn('save_final_replay_buffer=true', args)
            self.assertEqual(
                sum(value.startswith('agent.algorithm=') for value in args), 1,
            )

    def test_parameter_counts_are_positive_for_every_task_shape(self):
        for _, algorithm in ALGORITHMS:
            for _, _, _, state_dim, action_dim, goal_dim in ENVIRONMENTS:
                self.assertGreater(
                    baseline_parameter_count(
                        algorithm, state_dim, action_dim, goal_dim,
                    ),
                    0,
                )


if __name__ == '__main__':
    unittest.main()

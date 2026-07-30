import unittest
from pathlib import Path

from tools.run_qrl_queue import read_tasks


ROOT = Path(__file__).resolve().parents[1]
TASKS_PATH = (
    ROOT / 'configs/go_qrl_max4_dynamics_ablation_antnavigate_12x100k.tsv'
)


EXPECTED_COMBINATIONS = {
    (separate, distance, activation)
    for separate in ('false', 'true')
    for distance in ('iqe', 'mse', 'iqe_mse')
    for activation in ('relu', 'leaky_relu')
}


def task_args(extra_args: str) -> dict[str, str]:
    return dict(argument.split('=', 1) for argument in extra_args.split())


class DynamicsAblationTasksTest(unittest.TestCase):
    def test_matrix_is_the_complete_twelve_task_factorial(self):
        tasks = read_tasks(TASKS_PATH)
        self.assertEqual(len(tasks), 12)
        self.assertEqual(
            {
                (
                    task_args(task.extra_args)[
                        'agent.quasimetric_critic.losses.separate_latent_dynamics'
                    ],
                    task_args(task.extra_args)[
                        'agent.quasimetric_critic.losses.latent_dynamics.distance'
                    ],
                    task_args(task.extra_args)[
                        'agent.quasimetric_critic.model.quasimetric_model.projector_activation'
                    ],
                )
                for task in tasks
            },
            EXPECTED_COMBINATIONS,
        )

        for task in tasks:
            with self.subTest(task=task.task_id):
                args = task_args(task.extra_args)
                self.assertEqual(task.mode, 'online')
                self.assertEqual(task.env_name, 'AntNavigate-v4')
                self.assertEqual(task.seed, '1000')
                self.assertEqual(task.steps, '100000')
                self.assertEqual(args['agent.training_schedule'], 'joint')
                self.assertEqual(
                    args['agent.quasimetric_critic.model.encoder.branch_normalization'],
                    'none',
                )
                self.assertEqual(args['agent.actor.losses.min_dist.latent_goal_mode'], 'max')
                self.assertEqual(args['agent.actor.losses.min_dist.latent_goal_steps'], '4')
                self.assertEqual(args['agent.actor.losses.min_dist.latent_goal_optim'], 'sgd')
                self.assertEqual(args['save_steps'], '20000')
                self.assertEqual(args['resume_if_possible'], 'false')
                if args[
                    'agent.quasimetric_critic.losses.latent_dynamics.distance'
                ] == 'iqe_mse':
                    self.assertEqual(
                        args[
                            'agent.quasimetric_critic.losses.latent_dynamics.mse_weight'
                        ],
                        '1.0',
                    )
                    self.assertEqual(
                        args[
                            'agent.quasimetric_critic.losses.latent_dynamics.iqe_weight'
                        ],
                        '1.0',
                    )


if __name__ == '__main__':
    unittest.main()

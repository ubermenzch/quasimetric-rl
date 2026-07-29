import unittest
from collections import Counter

import gym
import numpy as np
from omegaconf import OmegaConf, SCMode

from quasimetric_rl.data.env_spec import EnvSpec
from quasimetric_rl.model_size import load_model_size_preset
from quasimetric_rl.modules import QRLConf
from tools.generate_online_baseline_tasks import (
    ALGORITHMS,
    ENVIRONMENTS,
    MODEL_SIZE_FAMILIES,
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
            self.assertIn('batch_size=256', args)
            self.assertEqual(
                sum(value.startswith('agent.algorithm=') for value in args), 1,
            )
            algorithm = next(
                value.split('=', 1)[1]
                for value in args if value.startswith('agent.algorithm=')
            )
            self.assertIn(
                f'+{MODEL_SIZE_FAMILIES[algorithm]}_model_size=m', args,
            )
            self.assertIn('-M_', task.task_id)

    def test_parameter_counts_are_in_m_budget_for_every_task_shape(self):
        for _, algorithm in ALGORITHMS:
            for _, _, _, state_dim, action_dim, goal_dim in ENVIRONMENTS:
                count = baseline_parameter_count(
                    algorithm, state_dim, action_dim, goal_dim,
                )
                self.assertGreaterEqual(count, 3_990_000)
                self.assertLessEqual(count, 4_500_000)

    def test_can_generate_one_algorithm_for_server_assignment(self):
        tasks = generate_tasks(('crl',))
        self.assertEqual(len(tasks), len(ENVIRONMENTS) * len(TRAINING_SEEDS))
        self.assertTrue(all('agent.algorithm=crl' in task.extra_args for task in tasks))

    def test_count_formula_matches_instantiated_trainable_modules(self):
        _kind, _name, _slug, state_dim, action_dim, goal_dim = ENVIRONMENTS[0]
        env_spec = EnvSpec(
            observation_space=gym.spaces.Box(
                -np.inf, np.inf, shape=(state_dim,), dtype=np.float32,
            ),
            observation_space_is_dict=True,
            action_space=gym.spaces.Box(
                -1, 1, shape=(action_dim,), dtype=np.float32,
            ),
        )
        for _display_name, algorithm in ALGORITHMS:
            with self.subTest(algorithm=algorithm):
                merged = OmegaConf.merge(
                    OmegaConf.structured(QRLConf()),
                    load_model_size_preset(algorithm, 'm'),
                )
                conf = OmegaConf.to_container(
                    merged, structured_config_mode=SCMode.INSTANTIATE,
                )
                conf.algorithm = algorithm
                agent, losses = conf.make(
                    env_spec=env_spec,
                    total_optim_steps=1,
                    goal_set_dims=tuple(range(goal_dim)),
                )
                actual = sum(
                    parameter.numel()
                    for module in (agent, losses)
                    for parameter in module.parameters()
                    if parameter.requires_grad
                )
                self.assertEqual(
                    actual,
                    baseline_parameter_count(
                        algorithm, state_dim, action_dim, goal_dim,
                    ),
                )


if __name__ == '__main__':
    unittest.main()

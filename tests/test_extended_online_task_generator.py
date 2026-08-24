import unittest
from collections import Counter
from pathlib import Path
import tempfile

from quasimetric_rl.data.base import GOAL_SET_DIMS_REGISTRY
from quasimetric_rl.model_size import load_model_size_preset
from tools.generate_extended_online_tasks import (
    ALGORITHMS,
    CQRL_ARGS,
    DMC_TASKS,
    ENVIRONMENTS,
    FAMILIES,
    MAZE_TASKS,
    PANDA_JOINT_TASKS,
    PANDA_TASKS,
    SHADOW_HAND_TASKS,
    SHADOW_TOUCH_TASKS,
    _select_algorithms,
    _select_environments,
    align_total_steps,
    algorithm_args,
    append_tasks,
    default_training_protocol,
    gcsl_uses_factorized_actions,
    generate_tasks,
    partition_tasks,
    validate_two_to_one_partitions,
)


class ExtendedOnlineTaskGeneratorTest(unittest.TestCase):
    def test_catalog_has_every_requested_and_derived_environment(self):
        self.assertEqual(len(ENVIRONMENTS), 34)
        self.assertEqual(
            Counter(environment.family for environment in ENVIRONMENTS),
            {
                'dmc': 7,
                'shadow-hand': 4,
                'shadow-touch': 6,
                'maze': 5,
                'panda': 6,
                'panda-joints': 6,
            },
        )
        self.assertEqual(
            len(DMC_TASKS) + len(SHADOW_HAND_TASKS)
            + len(SHADOW_TOUCH_TASKS) + len(MAZE_TASKS)
            + len(PANDA_TASKS) + len(PANDA_JOINT_TASKS),
            34,
        )
        self.assertEqual(
            FAMILIES,
            ('dmc', 'shadow-hand', 'shadow-touch', 'maze', 'panda', 'panda-joints'),
        )
        related_ant_mazes = {
            'AntMaze_BigMaze-v5',
            'AntMaze_HardestMaze-v5',
            'AntMaze_BigMaze_DG-v5',
            'AntMaze_HardestMaze_DG-v5',
            'AntMaze_BigMaze_DGR-v5',
            'AntMaze_HardestMaze_DGR-v5',
        }
        self.assertEqual(
            related_ant_mazes.intersection(MAZE_TASKS),
            {'AntMaze_BigMaze_DGR-v5'},
        )

    def test_every_environment_has_registered_partial_goal_contract(self):
        for environment in ENVIRONMENTS:
            with self.subTest(environment=environment.name):
                self.assertEqual(
                    GOAL_SET_DIMS_REGISTRY[environment.kind, environment.name],
                    tuple(range(environment.goal_dim)),
                )
                self.assertGreater(environment.state_dim, environment.goal_dim)
                self.assertGreater(environment.action_dim, 0)
                self.assertGreater(environment.horizon, 0)

    def test_catalog_prunes_same_environment_difficulty_variants(self):
        excluded = {
            'point_mass_hard',
            'finger_turn_hard',
            'stacker_stack_4',
            'HandManipulateBlockRotateParallel-v1',
            'HandManipulateBlockRotateXYZ-v1',
            'PointMaze_Large-v3',
        }
        self.assertTrue(excluded.isdisjoint(
            environment.name for environment in ENVIRONMENTS
        ))

    def test_filtered_matrix_is_complete_and_unique(self):
        tasks = generate_tasks(
            families=('dmc',), model_sizes=('m', 'l'), seeds=(1000, 1001),
        )
        expected = len(DMC_TASKS) * len(ALGORITHMS) * 2 * 2
        self.assertEqual(len(tasks), expected)
        self.assertEqual(len({task.task_id for task in tasks}), expected)
        self.assertEqual(
            Counter(task.env_name for task in tasks),
            {name: len(ALGORITHMS) * 2 * 2 for name in DMC_TASKS},
        )
        self.assertEqual(
            Counter(task.seed for task in tasks),
            {'1000': len(DMC_TASKS) * len(ALGORITHMS) * 2,
             '1001': len(DMC_TASKS) * len(ALGORITHMS) * 2},
        )

    def test_shadow_matrix_contains_no_semantic_alias_rows(self):
        tasks = generate_tasks(
            families=('shadow-hand',), model_sizes=('m',), seeds=(1000,),
        )
        self.assertEqual(len(tasks), len(SHADOW_HAND_TASKS) * len(ALGORITHMS))
        self.assertFalse(any(
            'semantic_alias_' in task.task_id for task in tasks
        ))

    def test_explicit_m_and_l_sizes_keep_the_environment_budget(self):
        tasks = generate_tasks(
            families=('panda',), model_sizes=('m', 'l'), seeds=(1000,),
        )
        labels = {algorithm.label.replace('/', '-') for algorithm in ALGORITHMS}
        for label in labels:
            self.assertEqual(
                sum(f'_{label}-' in task.task_id for task in tasks),
                len(PANDA_TASKS) * 2,
            )
        for task in tasks:
            args = task.extra_args.split()
            self.assertTrue(
                '-M_' in task.task_id or '-L_' in task.task_id,
                f'Missing model-size label in {task.task_id}',
            )
            self.assertEqual(task.steps, '100000')
            self.assertIn('save_steps=20000', args)
            self.assertIn('env.init_num_transitions=100000', args)
            self.assertIn('env.increment_num_transitions=100000', args)

    def test_default_protocol_selects_one_size_and_budget_per_horizon(self):
        tasks = generate_tasks(seeds=(1000,))
        self.assertEqual(len(tasks), len(ENVIRONMENTS) * len(ALGORITHMS))
        environments = {environment.name: environment for environment in ENVIRONMENTS}
        for task in tasks:
            environment = environments[task.env_name]
            protocol = default_training_protocol(environment)
            expected_steps = align_total_steps(
                environment, protocol['total_steps'],
            )
            self.assertEqual(int(task.steps), expected_steps)
            self.assertIn(
                f'-{protocol["model_size"].upper()}_', task.task_id,
            )
            self.assertIn(
                f'save_steps={protocol["save_steps"]}', task.extra_args.split(),
            )

        short = next(env for env in ENVIRONMENTS if env.name == 'PandaStack-v3')
        medium = next(env for env in ENVIRONMENTS if env.name == 'PointMaze_UMaze-v3')
        long = next(env for env in ENVIRONMENTS if env.name == 'dog_fetch')
        self.assertEqual(
            default_training_protocol(short),
            {'model_size': 'm', 'total_steps': 100_000, 'save_steps': 20_000},
        )
        self.assertEqual(
            default_training_protocol(medium),
            {'model_size': 'm', 'total_steps': 200_000, 'save_steps': 20_000},
        )
        self.assertEqual(
            default_training_protocol(long),
            {'model_size': 'l', 'total_steps': 500_000, 'save_steps': 50_000},
        )

    def test_total_steps_end_on_complete_fixed_length_episodes(self):
        tasks = generate_tasks(
            families=('maze',), model_sizes=('m', 'l'), seeds=(1000,),
        )
        environments = {environment.name: environment for environment in ENVIRONMENTS}
        for task in tasks:
            environment = environments[task.env_name]
            total_steps = int(task.steps)
            args = task.extra_args.split()
            self.assertEqual(total_steps % environment.horizon, 0)
            self.assertIn(f'env.init_num_transitions={total_steps}', args)
            self.assertIn(f'env.increment_num_transitions={total_steps}', args)

        point = environments['PointMaze_UMaze-v3']
        ant = environments['AntMaze_UMaze-v5']
        self.assertEqual(align_total_steps(point, 200_000), 200_100)
        self.assertEqual(align_total_steps(point, 500_000), 500_100)
        self.assertEqual(align_total_steps(ant, 200_000), 200_200)
        self.assertEqual(align_total_steps(ant, 500_000), 500_500)

    def test_cqrl_uses_public_preset_and_inner_step_ablation(self):
        environment = next(env for env in ENVIRONMENTS if env.name == 'dog_fetch')
        for inner_steps in (0, 1, 4, 8):
            algorithm = next(
                item for item in ALGORITHMS if item.inner_steps == inner_steps
            )
            args = algorithm_args(algorithm, environment, 'm')
            self.assertIn('+cqrl_model_size=m', args)
            self.assertIn(
                f'agent.actor.losses.min_dist.latent_goal_steps={inner_steps}',
                args,
            )
            for required in CQRL_ARGS:
                self.assertIn(required, args)
            self.assertNotIn('+go_qrl_model_size=m', args)

    def test_factorized_gcsl_is_only_used_for_high_dof_actions(self):
        low_dof = next(env for env in ENVIRONMENTS if env.name == 'point_mass_easy')
        high_dof = next(env for env in ENVIRONMENTS if env.name == 'dog_fetch')
        gcsl = next(algorithm for algorithm in ALGORITHMS if algorithm.key == 'gcsl')
        self.assertFalse(gcsl_uses_factorized_actions(low_dof))
        self.assertTrue(gcsl_uses_factorized_actions(high_dof))
        self.assertNotIn(
            'agent.baselines.gcbc.action_discretization=factorized',
            algorithm_args(gcsl, low_dof, 'm'),
        )
        self.assertIn(
            'agent.baselines.gcbc.action_discretization=factorized',
            algorithm_args(gcsl, high_dof, 'm'),
        )

    def test_representative_dimensions_and_horizons(self):
        environments = {environment.name: environment for environment in ENVIRONMENTS}
        expected = {
            'dog_fetch': (210, 38, 3, 1000),
            'HandReach-v2': (63, 20, 15, 50),
            'HandManipulateBlockRotateZ_BooleanTouchSensors-v1': (
                158, 20, 9, 100,
            ),
            'AntMaze_UMaze-v5': (107, 8, 2, 700),
            'PointMaze_Medium-v3': (4, 2, 2, 600),
            'PandaStackJoints-v3': (49, 8, 6, 100),
        }
        for name, dimensions in expected.items():
            environment = environments[name]
            self.assertEqual(
                (
                    environment.state_dim,
                    environment.action_dim,
                    environment.goal_dim,
                    environment.horizon,
                ),
                dimensions,
            )

    def test_cqrl_model_size_alias_preserves_legacy_preset(self):
        cqrl = load_model_size_preset('cqrl', 'm')
        legacy = load_model_size_preset('go_qrl', 'm')
        self.assertEqual(cqrl.model_size, 'CQRL-M')
        self.assertEqual(legacy.model_size, 'GO-QRL-M')
        self.assertEqual(
            cqrl.quasimetric_critic.model.encoder,
            legacy.quasimetric_critic.model.encoder,
        )

    def test_short_horizon_matrix_has_balanced_two_to_one_partitions(self):
        algorithm_keys = (
            'qrl', 'cqrl_inner1', 'td_infonce',
            'gcsl', 'c_learning', 'crl',
        )
        seeds = tuple(range(1000, 1005))
        algorithms = _select_algorithms(algorithm_keys)
        environments = _select_environments(None, max_horizon=300)
        tasks = generate_tasks(
            algorithms=algorithm_keys, max_horizon=300, seeds=seeds,
        )

        self.assertEqual(len(environments), 24)
        self.assertEqual(len(tasks), 720)
        self.assertEqual(
            Counter(int(task.steps) for task in tasks),
            {100_000: 660, 200_100: 60},
        )
        validate_two_to_one_partitions(
            tasks, algorithms=algorithms,
            environments=environments, seeds=seeds,
        )
        local = partition_tasks(
            tasks, 'local_2x', algorithms=algorithms,
            environments=environments, seeds=seeds,
        )
        remote = partition_tasks(
            tasks, 'remote_1x', algorithms=algorithms,
            environments=environments, seeds=seeds,
        )
        self.assertEqual((len(local), len(remote)), (480, 240))
        self.assertEqual(
            Counter(task.env_name for task in local),
            {environment.name: 20 for environment in environments},
        )
        self.assertEqual(
            Counter(task.seed for task in local),
            {str(seed): 96 for seed in seeds},
        )
        self.assertEqual(
            Counter(
                next(
                    algorithm.key for algorithm in algorithms
                    if f'_{algorithm.label.replace("/", "-")}-'
                    in task.task_id
                )
                for task in local
            ),
            {algorithm.key: 80 for algorithm in algorithms},
        )

    def test_partition_append_is_idempotent(self):
        algorithms = _select_algorithms(('qrl', 'cqrl_inner1', 'td_infonce'))
        environments = _select_environments(('maze',), max_horizon=300)
        seeds = (1000, 1001, 1002)
        tasks = generate_tasks(
            algorithms=tuple(algorithm.key for algorithm in algorithms),
            families=('maze',), max_horizon=300, seeds=seeds,
        )
        local = partition_tasks(
            tasks, 'local_2x', algorithms=algorithms,
            environments=environments, seeds=seeds,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'tasks.tsv')
            self.assertEqual(append_tasks(path, local, 'local_2x'), len(local))
            self.assertEqual(append_tasks(path, local, 'local_2x'), 0)
            task_lines = [
                line for line in path.read_text().splitlines()
                if line and not line.startswith('#')
            ]
            self.assertEqual(len(task_lines), len(local))


if __name__ == '__main__':
    unittest.main()

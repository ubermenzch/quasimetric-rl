import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import gym
import numpy as np

from quasimetric_rl.data.base import (
    CREATE_ENV_REGISTRY,
    GOAL_SET_DIMS_REGISTRY,
)
from quasimetric_rl.data.online import gymnasium_robotics as robotics


REQUESTED_TASKS = {
    'HandReach-v2',
    'HandManipulateBlockRotateZ-v1',
    'HandManipulateEggRotate-v1',
    'HandManipulatePenRotate-v1',
    'PointMaze_UMaze-v3',
    'PointMaze_Open-v3',
    'PointMaze_Medium-v3',
    'AntMaze_UMaze-v5',
    'AntMaze_BigMaze_DGR-v5',
}
NON_TOUCH_MANIPULATION_TASKS = {
    name for name in REQUESTED_TASKS if name.startswith('HandManipulate')
}
TOUCH_TASKS = {
    f'{name[:-3]}_{touch_kind}-v1'
    for name in NON_TOUCH_MANIPULATION_TASKS
    for touch_kind in ('BooleanTouchSensors', 'ContinuousTouchSensors')
}
REQUESTED_TASKS |= TOUCH_TASKS


class FakeGoalEnv:
    def __init__(
            self, observation, achieved, desired, *, old_api=False,
            truncate_at=None, terminate_at=None, info_key='is_success',
            action_dim=2):
        self.action_space = gym.spaces.Box(
            low=-np.ones(action_dim, dtype=np.float32),
            high=np.ones(action_dim, dtype=np.float32),
            dtype=np.float32,
        )
        self.observation = np.asarray(observation, dtype=np.float32)
        self.achieved = np.asarray(achieved, dtype=np.float32)
        self.desired = np.asarray(desired, dtype=np.float32)
        self.old_api = old_api
        self.truncate_at = truncate_at
        self.terminate_at = terminate_at
        self.info_key = info_key
        self.steps = 0
        self.seeds = []
        self.closed = False

    @property
    def unwrapped(self):
        return self

    def _observation(self):
        return {
            'observation': self.observation.copy(),
            'achieved_goal': self.achieved.copy(),
            'desired_goal': self.desired.copy(),
        }

    def seed(self, seed):
        self.seeds.append(int(seed))

    def reset(self, **kwargs):
        if self.old_api and kwargs:
            raise TypeError('legacy reset accepts no keyword arguments')
        if 'seed' in kwargs:
            self.seeds.append(int(kwargs['seed']))
        self.steps = 0
        result = self._observation()
        if self.old_api:
            return result
        return result, {'reset': True}

    def step(self, action):
        del action
        self.steps += 1
        terminated = self.steps == self.terminate_at
        truncated = self.steps == self.truncate_at
        info = {self.info_key: True}
        if self.old_api:
            info['TimeLimit.truncated'] = truncated
            return self._observation(), 2.5, terminated or truncated, info
        return self._observation(), 2.5, terminated, truncated, info

    def close(self):
        self.closed = True


def native_observation(spec, achieved, *, values=None):
    if values is None:
        values = np.arange(spec['observation_dim'], dtype=np.float32)
    observation = np.asarray(values, dtype=np.float32).copy()
    indices = tuple(spec['native_goal_indices'])
    if indices:
        achieved = np.asarray(achieved, dtype=np.float32)
        repeated = (
            achieved[3:]
            if spec['goal_encoding'] == 'rotation_matrix'
            else achieved
        )
        observation[list(indices)] = repeated
    return observation


class TaskSpecificationTest(unittest.TestCase):
    def test_all_requested_tasks_are_specified_and_registered(self):
        self.assertEqual(len(REQUESTED_TASKS), 15)
        self.assertEqual(set(robotics.TASK_SPECS), REQUESTED_TASKS)
        self.assertEqual(set(robotics.GOAL_DIMS), REQUESTED_TASKS)
        for name, spec in robotics.TASK_SPECS.items():
            with self.subTest(name=name):
                self.assertEqual(
                    robotics.GOAL_DIMS[name],
                    tuple(range(len(robotics.GOAL_DIMS[name]))),
                )
                self.assertGreater(spec['episode_length'], 0)
                self.assertEqual(
                    spec['state_dim'],
                    spec['observation_dim']
                    - len(spec['native_goal_indices'])
                    + len(spec['goal_dims']),
                )
                self.assertIn(
                    ('gymnasium_robotics', name), CREATE_ENV_REGISTRY,
                )
                self.assertEqual(
                    GOAL_SET_DIMS_REGISTRY[('gymnasium_robotics', name)],
                    spec['goal_dims'],
                )

    def test_maze_specs_are_continuing_fixed_goal_tasks(self):
        for name, spec in robotics.TASK_SPECS.items():
            if 'Maze' not in name:
                continue
            with self.subTest(name=name):
                self.assertTrue(spec['backend_kwargs']['continuing_task'])
                self.assertFalse(spec['backend_kwargs']['reset_target'])
                if name.startswith('AntMaze'):
                    self.assertFalse(
                        spec['backend_kwargs']['terminate_when_unhealthy']
                    )

    def test_official_family_dimensions_and_horizons_are_pinned(self):
        reach = robotics.TASK_SPECS['HandReach-v2']
        self.assertEqual(
            (reach['state_dim'], reach['action_dim'], reach['episode_length']),
            (63, 20, 50),
        )
        for name in NON_TOUCH_MANIPULATION_TASKS:
            spec = robotics.TASK_SPECS[name]
            self.assertEqual(spec['state_dim'], 66)
            self.assertEqual(spec['action_dim'], 20)
            self.assertEqual(spec['episode_length'], 100)
        for name in TOUCH_TASKS:
            spec = robotics.TASK_SPECS[name]
            self.assertEqual(spec['state_dim'], 158)
            self.assertEqual(spec['action_dim'], 20)
            self.assertEqual(spec['episode_length'], 100)
        for name in (
                'PointMaze_UMaze-v3', 'PointMaze_Open-v3',
                'PointMaze_Medium-v3'):
            spec = robotics.TASK_SPECS[name]
            self.assertEqual((spec['state_dim'], spec['action_dim']), (4, 2))
        self.assertEqual(
            robotics.TASK_SPECS['AntMaze_UMaze-v5']['episode_length'], 700,
        )
        for name, spec in robotics.TASK_SPECS.items():
            if name.startswith('AntMaze'):
                self.assertEqual((spec['state_dim'], spec['action_dim']), (107, 8))

    def test_touch_variants_preserve_parent_goal_geometry(self):
        self.assertEqual(len(TOUCH_TASKS), 6)
        for parent_name in NON_TOUCH_MANIPULATION_TASKS:
            parent_spec = robotics.TASK_SPECS[parent_name]
            stem = parent_name[:-3]
            for touch_kind in (
                    'BooleanTouchSensors', 'ContinuousTouchSensors'):
                name = f'{stem}_{touch_kind}-v1'
                with self.subTest(name=name):
                    spec = robotics.TASK_SPECS[name]
                    self.assertEqual(spec['goal_dims'], parent_spec['goal_dims'])
                    self.assertEqual(
                        spec['goal_encoding'], parent_spec['goal_encoding'],
                    )
                    self.assertEqual(
                        spec['native_goal_indices'],
                        parent_spec['native_goal_indices'],
                    )
                    self.assertEqual(spec['raw_goal_size'], 7)
                    self.assertEqual(
                        spec['observation_dim'],
                        parent_spec['observation_dim'] + 92,
                    )
                    self.assertEqual(
                        spec['state_dim'], parent_spec['state_dim'] + 92,
                    )
                    self.assertEqual(spec['action_dim'], 20)

    def test_native_goal_indices_match_official_observation_layouts(self):
        self.assertEqual(
            robotics.TASK_SPECS['HandReach-v2']['native_goal_indices'],
            tuple(range(48, 63)),
        )
        for name in NON_TOUCH_MANIPULATION_TASKS:
            spec = robotics.TASK_SPECS[name]
            self.assertEqual(spec['native_goal_indices'], tuple(range(57, 61)))
        for name in (
                'PointMaze_UMaze-v3', 'PointMaze_Open-v3',
                'PointMaze_Medium-v3'):
            self.assertEqual(
                robotics.TASK_SPECS[name]['native_goal_indices'], (0, 1),
            )
        for name, spec in robotics.TASK_SPECS.items():
            if name.startswith('AntMaze'):
                self.assertEqual(spec['native_goal_indices'], ())

    def test_harder_or_redundant_variants_are_not_registered(self):
        excluded = {
            'HandManipulateBlock-v1',
            'HandManipulateBlockRotateParallel-v1',
            'HandManipulateBlockRotateXYZ-v1',
            'HandManipulateBlockFull-v1',
            'HandManipulateEgg-v1',
            'HandManipulateEggFull-v1',
            'HandManipulatePen-v1',
            'HandManipulatePenFull-v1',
            'PointMaze_Large-v3',
        }
        self.assertTrue(excluded.isdisjoint(robotics.TASK_SPECS))

    def test_backend_dependencies_are_imported_only_when_creating_an_env(self):
        # Optional simulator modules are local to _make_backend, not eager
        # globals that would break the repository's base installation.
        self.assertNotIn('gymnasium', robotics.__dict__)
        self.assertNotIn('gymnasium_robotics', robotics.__dict__)

    def test_runtime_dimensions_must_match_pinned_metadata(self):
        wrong_observation = FakeGoalEnv(
            observation=np.zeros(3), achieved=np.zeros(2), desired=np.ones(2),
        )
        with self.assertRaisesRegex(ValueError, 'native observation shape'):
            robotics.GymnasiumRoboticsGoalEnv(
                'PointMaze_UMaze-v3', backend_env=wrong_observation,
            )

        wrong_action = FakeGoalEnv(
            observation=np.zeros(61),
            achieved=np.r_[np.zeros(3), np.array([1, 0, 0, 0])],
            desired=np.r_[np.zeros(3), np.array([1, 0, 0, 0])],
        )
        with self.assertRaisesRegex(ValueError, 'action shape'):
            robotics.GymnasiumRoboticsGoalEnv(
                'HandManipulateBlockRotateZ-v1', backend_env=wrong_action,
            )

    def test_backend_factory_passes_fixed_goal_and_horizon_arguments(self):
        fake_backend = object()
        fake_gymnasium = SimpleNamespace(
            register_envs=mock.Mock(), make=mock.Mock(return_value=fake_backend),
        )
        fake_plugin = SimpleNamespace()
        with mock.patch.dict(sys.modules, {
            'gymnasium': fake_gymnasium,
            'gymnasium_robotics': fake_plugin,
        }):
            backend, backend_id = robotics._make_backend(
                'AntMaze_BigMaze_DGR-v5'
            )

        self.assertIs(backend, fake_backend)
        self.assertEqual(backend_id, 'AntMaze_BigMaze_DGR-v5')
        fake_gymnasium.register_envs.assert_called_once_with(fake_plugin)
        fake_gymnasium.make.assert_called_once_with(
            'AntMaze_BigMaze_DGR-v5',
            continuing_task=True,
            reset_target=False,
            terminate_when_unhealthy=False,
            include_cfrc_ext_in_observation=True,
            max_episode_steps=1000,
        )

    def test_hand_reach_v2_does_not_silently_change_protocol(self):
        fake_gymnasium = SimpleNamespace(
            register_envs=mock.Mock(),
            make=mock.Mock(side_effect=LookupError('v2 is not registered')),
        )
        with mock.patch.dict(sys.modules, {
            'gymnasium': fake_gymnasium,
            'gymnasium_robotics': SimpleNamespace(),
        }):
            with self.assertRaisesRegex(RuntimeError, 'HandReach-v2'):
                robotics._make_backend('HandReach-v2')
        fake_gymnasium.make.assert_called_once_with(
            'HandReach-v2', max_episode_steps=50,
        )

    def test_new_ant_maze_names_fall_back_to_legacy_registry_ids(self):
        fake_backend = object()

        def make(backend_id, **kwargs):
            del kwargs
            if backend_id == 'AntMaze_BigMaze_DGR-v5':
                raise LookupError('new documentation ID is not registered')
            return fake_backend

        fake_gymnasium = SimpleNamespace(
            register_envs=mock.Mock(), make=mock.Mock(side_effect=make),
        )
        with mock.patch.dict(sys.modules, {
            'gymnasium': fake_gymnasium,
            'gymnasium_robotics': SimpleNamespace(),
        }):
            backend, backend_id = robotics._make_backend(
                'AntMaze_BigMaze_DGR-v5'
            )
        self.assertIs(backend, fake_backend)
        self.assertEqual(backend_id, 'AntMaze_Medium_Diverse_GR-v5')


class GoalMappingTest(unittest.TestCase):
    def test_every_task_produces_same_shaped_qrl_goal_observations(self):
        for name, spec in robotics.TASK_SPECS.items():
            raw_goal = np.zeros(spec['raw_goal_size'], dtype=np.float32)
            if spec['raw_goal_size'] == 7:
                raw_goal[3] = 1.0
            backend = FakeGoalEnv(
                observation=native_observation(spec, raw_goal),
                achieved=raw_goal,
                desired=raw_goal,
                action_dim=spec['action_dim'],
            )
            with self.subTest(name=name):
                env = robotics.GymnasiumRoboticsGoalEnv(
                    name, backend_env=backend,
                )
                observation = env.reset()
                expected_size = spec['state_dim']
                self.assertEqual(
                    observation['observation'].shape, (expected_size,),
                )
                self.assertEqual(
                    observation['achieved_goal'].shape, (expected_size,),
                )
                self.assertEqual(
                    observation['desired_goal'].shape, (expected_size,),
                )
                np.testing.assert_allclose(
                    observation['observation'][list(spec['goal_dims'])],
                    observation['desired_goal'][list(spec['goal_dims'])],
                )

    def test_short_native_goal_is_exposed_as_full_state_prefix(self):
        backend = FakeGoalEnv(
            observation=[1, 2, 12, 13],
            achieved=[1, 2],
            desired=[3, 4],
            info_key='success',
        )
        env = robotics.GymnasiumRoboticsGoalEnv(
            'PointMaze_UMaze-v3', backend_env=backend,
        )
        observation = env.reset(seed=17)

        np.testing.assert_array_equal(
            observation['observation'],
            np.array([1, 2, 12, 13], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            observation['achieved_goal'], observation['observation'],
        )
        np.testing.assert_array_equal(
            observation['desired_goal'],
            np.array([3, 4, 0, 0], dtype=np.float32),
        )
        self.assertEqual(env.goal_dims, (0, 1))
        self.assertIn(17, backend.seeds)

        next_observation, reward, terminal, info = env.step(np.zeros(2))
        self.assertEqual(next_observation['observation'].shape, (4,))
        self.assertEqual(reward, 1.0)
        self.assertFalse(terminal)
        self.assertTrue(info['is_success'])
        self.assertAlmostEqual(info['goal_distance'], np.sqrt(8))
        self.assertEqual(info['backend_reward'], 2.5)

    def test_quaternion_signs_have_identical_block_goal_coordinates(self):
        quaternion = np.array([np.sqrt(0.5), 0, 0, np.sqrt(0.5)])
        raw_achieved = np.r_[np.zeros(3), quaternion]
        spec = robotics.TASK_SPECS['HandManipulateBlockRotateZ-v1']
        backend = FakeGoalEnv(
            observation=native_observation(spec, raw_achieved),
            achieved=raw_achieved,
            desired=np.r_[np.zeros(3), -quaternion],
            action_dim=20,
        )
        env = robotics.GymnasiumRoboticsGoalEnv(
            'HandManipulateBlockRotateZ-v1', backend_env=backend,
        )
        observation = env.reset()

        goal_dims = list(env.goal_dims)
        np.testing.assert_allclose(
            observation['observation'][goal_dims],
            observation['desired_goal'][goal_dims],
            atol=1e-6,
        )
        self.assertEqual(len(env.goal_dims), 9)
        _, _, _, info = env.step(np.zeros(20))
        self.assertAlmostEqual(info['goal_distance'], 0.0, places=6)

    def test_egg_rotation_keeps_the_full_all_axis_orientation(self):
        identity = np.array([1, 0, 0, 0], dtype=np.float32)
        z_quarter_turn = np.array([
            np.sqrt(0.5), 0, 0, np.sqrt(0.5),
        ], dtype=np.float32)
        raw_achieved = np.r_[np.zeros(3), identity]
        spec = robotics.TASK_SPECS['HandManipulateEggRotate-v1']
        backend = FakeGoalEnv(
            observation=native_observation(spec, raw_achieved),
            achieved=raw_achieved,
            desired=np.r_[np.zeros(3), z_quarter_turn],
            action_dim=20,
        )
        env = robotics.GymnasiumRoboticsGoalEnv(
            'HandManipulateEggRotate-v1', backend_env=backend,
        )
        observation = env.reset()
        self.assertFalse(np.allclose(
            observation['observation'][:9],
            observation['desired_goal'][:9],
            atol=1e-6,
        ))
        self.assertEqual(env.goal_dims, tuple(range(9)))
        _, _, _, info = env.step(np.zeros(20))
        self.assertAlmostEqual(
            info['goal_distance'], np.pi / 2, places=6,
        )

    def test_rotation_only_retains_position_and_appended_touch_sensors(self):
        raw_goal = np.array([1, 2, 3, 1, 0, 0, 0], dtype=np.float32)
        name = 'HandManipulateEggRotate_ContinuousTouchSensors-v1'
        spec = robotics.TASK_SPECS[name]
        native = native_observation(spec, raw_goal)
        native[54:57] = raw_goal[:3]
        backend = FakeGoalEnv(
            observation=native,
            achieved=raw_goal,
            desired=raw_goal,
            action_dim=20,
        )
        env = robotics.GymnasiumRoboticsGoalEnv(name, backend_env=backend)
        state = env.reset()['observation']

        np.testing.assert_array_equal(state[63:66], raw_goal[:3])
        np.testing.assert_array_equal(state[66:], native[61:])
        self.assertEqual(state.shape, (158,))

    def test_inconsistent_native_goal_coordinates_are_rejected(self):
        backend = FakeGoalEnv(
            observation=[9, 9, 0, 0],
            achieved=[1, 2],
            desired=[3, 4],
        )
        with self.assertRaisesRegex(ValueError, 'do not match achieved_goal'):
            robotics.GymnasiumRoboticsGoalEnv(
                'PointMaze_UMaze-v3', backend_env=backend,
            )

    def test_goal_change_within_episode_is_rejected(self):
        backend = FakeGoalEnv(
            observation=np.zeros(4), achieved=np.zeros(2), desired=np.ones(2),
        )
        env = robotics.GymnasiumRoboticsGoalEnv(
            'PointMaze_Open-v3', backend_env=backend,
        )
        env.reset()
        backend.desired[:] = 2
        with self.assertRaisesRegex(RuntimeError, 'changed the desired goal'):
            env.step(np.zeros(2))


class FixedHorizonCompatibilityTest(unittest.TestCase):
    def test_new_api_timeout_occurs_exactly_at_fixed_horizon(self):
        backend = FakeGoalEnv(
            observation=np.zeros(4), achieved=np.zeros(2), desired=np.ones(2),
            truncate_at=2,
        )
        env = robotics.GymnasiumRoboticsGoalEnv(
            'PointMaze_Open-v3', backend_env=backend,
        )
        env.episode_length = 2
        env.reset()

        _, _, terminal, first_info = env.step(np.zeros(2))
        self.assertFalse(terminal)
        self.assertNotIn('TimeLimit.truncated', first_info)
        _, _, terminal, second_info = env.step(np.zeros(2))
        self.assertFalse(terminal)
        self.assertTrue(second_info['TimeLimit.truncated'])
        with self.assertRaisesRegex(RuntimeError, 'after the fixed episode horizon'):
            env.step(np.zeros(2))

    def test_legacy_reset_and_step_api_are_supported(self):
        backend = FakeGoalEnv(
            observation=np.zeros(4), achieved=np.zeros(2), desired=np.ones(2),
            old_api=True, truncate_at=2,
        )
        env = robotics.GymnasiumRoboticsGoalEnv(
            'PointMaze_Open-v3', backend_env=backend,
        )
        env.episode_length = 2
        env.seed(23)
        observation = env.reset()
        self.assertEqual(observation['observation'].shape, (4,))
        self.assertIn(23, backend.seeds)
        env.step(np.zeros(2))
        _, _, terminal, info = env.step(np.zeros(2))
        self.assertFalse(terminal)
        self.assertTrue(info['TimeLimit.truncated'])

    def test_early_backend_termination_is_rejected(self):
        backend = FakeGoalEnv(
            observation=np.zeros(4), achieved=np.zeros(2), desired=np.ones(2),
            terminate_at=1,
        )
        env = robotics.GymnasiumRoboticsGoalEnv(
            'PointMaze_Open-v3', backend_env=backend,
        )
        env.reset()
        with self.assertRaisesRegex(RuntimeError, 'continuing task'):
            env.step(np.zeros(2))

    def test_close_is_forwarded(self):
        backend = FakeGoalEnv(
            observation=np.zeros(4), achieved=np.zeros(2), desired=np.ones(2),
        )
        env = robotics.GymnasiumRoboticsGoalEnv(
            'PointMaze_Open-v3', backend_env=backend,
        )
        env.close()
        self.assertTrue(backend.closed)


if __name__ == '__main__':
    unittest.main()

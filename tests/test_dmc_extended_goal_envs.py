import unittest
from types import SimpleNamespace

import numpy as np

from quasimetric_rl.data.base import (
    CREATE_ENV_REGISTRY,
    GOAL_SET_DIMS_REGISTRY,
)
from quasimetric_rl.data.online.dmc import DMCGoalEnv, TASK_SPECS


EXTENDED_DMC_SPECS = {
    'point_mass_easy': (4, 2, (0, 1)),
    'finger_turn_easy': (9, 2, (0, 1)),
    'manipulator_insert_ball': (40, 5, (0, 1)),
    'manipulator_insert_peg': (40, 5, (0, 1, 2, 3)),
    'dog_fetch': (210, 38, (0, 1, 2)),
    'stacker_stack_2': (49, 5, (0, 1)),
    'ball_in_cup_catch': (8, 2, (0, 1)),
}


class _NamedArray:
    def __init__(self, values):
        self._values = {
            name: np.asarray(value, dtype=np.float64)
            for name, value in values.items()
        }

    def __getitem__(self, key):
        if isinstance(key, tuple):
            name, index = key
            return self._values[name][index]
        return self._values[key]


class _TimeStep:
    def __init__(self, observation, reward=None, last=False):
        self.observation = observation
        self.reward = reward
        self._last = last

    def last(self):
        return self._last


class _ActionSpec:
    def __init__(self, size):
        self.minimum = -np.ones(size, dtype=np.float64)
        self.maximum = np.ones(size, dtype=np.float64)
        self.dtype = np.dtype(np.float64)


class _FakePhysics:
    def __init__(self, *, qpos=(), qvel=(), act=(), ball_qpos_address=0):
        self.data = SimpleNamespace(
            qpos=np.asarray(qpos, dtype=np.float64),
            qvel=np.asarray(qvel, dtype=np.float64),
            act=np.asarray(act, dtype=np.float64),
        )
        self.named = SimpleNamespace(
            data=SimpleNamespace(
                geom_xpos=_NamedArray({
                    'target': [0.0, 0.0, 0.0],
                    'ball': [0.05, 0.0, 0.0],
                }),
                site_xpos=_NamedArray({'target': [0.0, 0.0, 0.0]}),
            ),
            model=SimpleNamespace(
                geom_size=_NamedArray({'target': [0.1, 0.0, 0.0]}),
                site_size=_NamedArray({'target': [0.1, 0.0, 0.0]}),
                jnt_qposadr=_NamedArray({
                    'ball_root': ball_qpos_address,
                }),
            ),
        )
        self.signed_finger_distance = -0.01
        self.ball_target_delta = np.array([-0.01, 0.01])
        self.ball_is_in_cup = True

    def dist_to_target(self):
        return self.signed_finger_distance

    def ball_to_target(self):
        return self.ball_target_delta

    def in_target(self):
        return self.ball_is_in_cup


class _FakeBackend:
    def __init__(self, observation, action_size, physics, reward=1.0):
        self._observation = observation
        self._action_spec = _ActionSpec(action_size)
        self._reward = reward
        self._steps = 0
        self.physics = physics

    def action_spec(self):
        return self._action_spec

    def reset(self):
        self._steps = 0
        return _TimeStep(self._observation)

    def step(self, action):
        self._steps += 1
        assert np.asarray(action).dtype == self._action_spec.dtype
        return _TimeStep(
            self._observation,
            reward=self._reward,
            last=self._steps == 1000,
        )


def _manipulator_observation():
    return {
        'arm_pos': np.zeros((8, 2)),
        'arm_vel': np.zeros(8),
        'touch': np.zeros(5),
        'hand_pos': np.zeros(4),
        'object_pos': np.array([0.01, 0.02, 1.0, 0.0]),
        'object_vel': np.zeros(3),
        'target_pos': np.array([0.01, 0.02, 1.0, 0.0]),
    }


def _stacker_observation(n_boxes):
    boxes = np.zeros((n_boxes, 4))
    boxes[:, :2] = np.arange(n_boxes)[:, None] + 0.5
    boxes[-1, :2] = [0.01, 0.02]
    return {
        'arm_pos': np.zeros((8, 2)),
        'arm_vel': np.zeros(8),
        'touch': np.zeros(5),
        'hand_pos': np.zeros(4),
        'box_pos': boxes,
        'box_vel': np.zeros(3 * n_boxes),
        'target_pos': np.array([0.0, 0.0]),
    }


def _fake_backend(name, seed):
    del seed
    domain = TASK_SPECS[name]['domain']
    if domain == 'point_mass':
        return _FakeBackend(
            {'position': np.array([0.05, 0.0]), 'velocity': np.zeros(2)},
            2,
            _FakePhysics(),
        )
    if domain == 'finger':
        return _FakeBackend(
            {
                'position': np.array([0.5, 0.6, 0.01, 0.02]),
                'velocity': np.zeros(3),
                'touch': np.zeros(2),
                'target_position': np.array([0.01, 0.02]),
            },
            2,
            _FakePhysics(),
        )
    if domain == 'manipulator':
        return _FakeBackend(
            _manipulator_observation(), 5, _FakePhysics(),
        )
    if domain == 'dog':
        ball_qpos_address = 11
        qpos = np.arange(87, dtype=np.float64) / 100.0
        qpos[ball_qpos_address:ball_qpos_address + 7] = [
            0.05, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        ]
        return _FakeBackend(
            {},
            38,
            _FakePhysics(
                qpos=qpos,
                qvel=np.arange(85, dtype=np.float64) / 10.0,
                act=np.arange(38, dtype=np.float64) / 5.0,
                ball_qpos_address=ball_qpos_address,
            ),
        )
    if domain == 'stacker':
        return _FakeBackend(
            _stacker_observation(TASK_SPECS[name]['n_boxes']),
            5,
            _FakePhysics(),
            reward=0.96,
        )
    if domain == 'ball_in_cup':
        return _FakeBackend(
            {'position': np.zeros(4), 'velocity': np.zeros(4)},
            2,
            _FakePhysics(),
        )
    raise AssertionError(domain)


class ExtendedDMCGoalEnvTest(unittest.TestCase):
    def make_env(self, name):
        return DMCGoalEnv(
            name,
            environment_factory=lambda seed: _fake_backend(name, seed),
        )

    def test_all_tasks_are_registered_with_expected_suite_names(self):
        expected_suite_tasks = {
            'point_mass_easy': ('point_mass', 'easy'),
            'finger_turn_easy': ('finger', 'turn_easy'),
            'manipulator_insert_ball': ('manipulator', 'insert_ball'),
            'manipulator_insert_peg': ('manipulator', 'insert_peg'),
            'dog_fetch': ('dog', 'fetch'),
            'stacker_stack_2': ('stacker', 'stack_2'),
            'ball_in_cup_catch': ('ball_in_cup', 'catch'),
        }
        for name, suite_name in expected_suite_tasks.items():
            with self.subTest(name=name):
                self.assertEqual(
                    (TASK_SPECS[name]['domain'], TASK_SPECS[name]['task']),
                    suite_name,
                )
                self.assertIn(('dmc', name), CREATE_ENV_REGISTRY)
                self.assertEqual(
                    GOAL_SET_DIMS_REGISTRY[('dmc', name)],
                    EXTENDED_DMC_SPECS[name][2],
                )

    def test_goal_dict_shapes_actions_and_success(self):
        for name, (state_size, action_size, goal_dims) in (
                EXTENDED_DMC_SPECS.items()):
            with self.subTest(name=name):
                env = self.make_env(name)
                observation = env.reset()
                self.assertEqual(env.observation_space['observation'].shape,
                                 (state_size,))
                self.assertEqual(env.action_space.shape, (action_size,))
                self.assertEqual(env.goal_dims, goal_dims)
                for key in ('observation', 'achieved_goal', 'desired_goal'):
                    self.assertEqual(observation[key].shape, (state_size,))
                    self.assertEqual(observation[key].dtype, np.float32)
                non_goal_dims = np.setdiff1d(np.arange(state_size), goal_dims)
                np.testing.assert_array_equal(
                    observation['desired_goal'][non_goal_dims], 0.0,
                )

                next_observation, _, done, info = env.step(
                    np.zeros(action_size, dtype=np.float32),
                )
                self.assertFalse(done)
                self.assertTrue(info['is_success'])
                np.testing.assert_array_equal(
                    next_observation['desired_goal'],
                    observation['desired_goal'],
                )

    def test_stacker_uses_the_nearest_box_as_achieved_goal(self):
        env = self.make_env('stacker_stack_2')
        observation = env.reset()
        np.testing.assert_allclose(
            observation['observation'][:2], [0.01, 0.02],
        )
        np.testing.assert_allclose(observation['desired_goal'][:2], [0.0, 0.0])

    def test_dog_removes_only_named_ball_translation_from_qpos(self):
        env = self.make_env('dog_fetch')
        observation = env.reset()['observation']
        physics = env._env.physics
        qpos = physics.data.qpos
        address = int(physics.named.model.jnt_qposadr['ball_root'])
        expected = np.concatenate([
            physics.named.data.geom_xpos['ball'],
            qpos[:address],
            qpos[address + 3:],
            physics.data.qvel,
            physics.data.act,
        ]).astype(np.float32)
        np.testing.assert_array_equal(observation, expected)
        np.testing.assert_array_equal(
            observation[3 + address:3 + address + 4],
            qpos[address + 3:address + 7],
        )

    def test_dog_rejects_inconsistent_qpos_and_geom_ball_positions(self):
        backend = _fake_backend('dog_fetch', 0)
        address = int(backend.physics.named.model.jnt_qposadr['ball_root'])
        backend.physics.data.qpos[address] += 0.01
        with self.assertRaisesRegex(RuntimeError, 'qpos XYZ disagrees'):
            DMCGoalEnv('dog_fetch', environment_factory=lambda seed: backend)

    def test_ball_in_cup_uses_a_fixed_zero_goal_in_the_cup_frame(self):
        env = self.make_env('ball_in_cup_catch')
        observation = env.reset()
        np.testing.assert_allclose(observation['observation'][:2], [0.01, -0.01])
        np.testing.assert_array_equal(observation['desired_goal'], 0.0)

    def test_seed_rebuilds_the_backend_with_the_requested_seed(self):
        seeds = []

        def factory(seed):
            seeds.append(seed)
            return _fake_backend('point_mass_easy', seed)

        env = DMCGoalEnv('point_mass_easy', environment_factory=factory)
        env.seed(17)
        env.reset()
        self.assertEqual(seeds[-1], 17)

    def test_fixed_horizon_is_reported_as_truncation(self):
        env = self.make_env('point_mass_easy')
        env.reset()
        info = None
        for _ in range(1000):
            _, _, done, info = env.step(np.zeros(2, dtype=np.float32))
            self.assertFalse(done)
        self.assertTrue(info['TimeLimit.truncated'])


if __name__ == '__main__':
    unittest.main()

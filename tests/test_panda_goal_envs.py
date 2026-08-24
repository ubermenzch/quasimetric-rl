import unittest

import gym
import numpy as np

from quasimetric_rl.data.base import CREATE_ENV_REGISTRY
from quasimetric_rl.data.online.panda_gym import (
    PANDA_ARM_JOINT_COUNT,
    PANDA_FULL_JOINT_COUNT,
    PandaGoalEnv,
    TASK_SPECS,
    _create_box_with_texture_fallback,
)


EXPECTED_TASKS = {
    'PandaReach-v3': (3, 50, 'ee', 3),
    'PandaPush-v3': (3, 50, 'ee', 3),
    'PandaSlide-v3': (3, 50, 'ee', 3),
    'PandaPickAndPlace-v3': (3, 50, 'ee', 4),
    'PandaStack-v3': (6, 100, 'ee', 4),
    'PandaFlip-v3': (4, 50, 'ee', 4),
    'PandaReachJoints-v3': (3, 50, 'joints', 7),
    'PandaPushJoints-v3': (3, 50, 'joints', 7),
    'PandaSlideJoints-v3': (3, 50, 'joints', 7),
    'PandaPickAndPlaceJoints-v3': (3, 50, 'joints', 8),
    'PandaStackJoints-v3': (6, 100, 'joints', 8),
    'PandaFlipJoints-v3': (4, 50, 'joints', 8),
}

EXPECTED_STATE_SIZES = {
    'PandaReach-v3': 6,
    'PandaPush-v3': 18,
    'PandaSlide-v3': 18,
    'PandaPickAndPlace-v3': 19,
    'PandaStack-v3': 31,
    'PandaFlip-v3': 20,
    'PandaReachJoints-v3': 20,
    'PandaPushJoints-v3': 32,
    'PandaSlideJoints-v3': 32,
    'PandaPickAndPlaceJoints-v3': 37,
    'PandaStackJoints-v3': 49,
    'PandaFlipJoints-v3': 38,
}

EXPECTED_NATIVE_GOAL_INDICES = {
    'Reach': (0, 1, 2),
    'Push': (6, 7, 8),
    'Slide': (6, 7, 8),
    'PickAndPlace': (7, 8, 9),
    'Stack': (7, 8, 9, 19, 20, 21),
    'Flip': (10, 11, 12, 13),
}


class FakeTask:
    def is_success(self, achieved, desired):
        return np.linalg.norm(achieved - desired) < 1e-6


class FakeDirectTask(FakeTask):
    def __init__(self, backend):
        self.backend = backend

    def get_goal(self):
        return self.backend.desired.copy()

    def compute_reward(self, achieved, desired, info):
        del achieved, desired, info
        return -1.0


class FakeDirectRobot:
    def __init__(self, backend):
        self.backend = backend

    def set_action(self, action):
        self.backend.actions.append(np.asarray(action).copy())


class FakeDirectSim:
    def __init__(self, backend):
        self.backend = backend

    def step(self):
        self.backend._advance()


class FakePandaRobot:
    def __init__(self):
        self.joint_indices = np.array([0, 1, 2, 3, 4, 5, 6, 9, 10])
        self.fingers_indices = np.array([9, 10])
        self.positions = np.arange(9, dtype=np.float32) + 0.25
        self.velocities = -np.arange(9, dtype=np.float32) - 0.5
        self._joint_offsets = {
            int(joint): offset
            for offset, joint in enumerate(self.joint_indices)
        }

    def get_joint_angle(self, joint):
        return self.positions[self._joint_offsets[int(joint)]]

    def get_joint_velocity(self, joint):
        return self.velocities[self._joint_offsets[int(joint)]]


class FakePandaSim:
    def __init__(self):
        self.joint_indices = (0, 1, 2, 3, 4, 5, 6, 9, 10)
        self.positions = np.arange(9, dtype=np.float32) + 10.0
        self.velocities = np.arange(9, dtype=np.float32) + 20.0
        self._joint_offsets = {
            joint: offset for offset, joint in enumerate(self.joint_indices)
        }
        self.calls = []

    def get_joint_angle(self, body, joint):
        self.calls.append(('position', body, joint))
        return self.positions[self._joint_offsets[int(joint)]]

    def get_joint_velocity(self, body, joint):
        self.calls.append(('velocity', body, joint))
        return self.velocities[self._joint_offsets[int(joint)]]


class FakeSimOnlyPandaRobot:
    body_name = 'panda'
    joint_indices = np.array([0, 1, 2, 3, 4, 5, 6, 9, 10])
    fingers_indices = np.array([9, 10])

    def __init__(self):
        self.sim = FakePandaSim()


class FakeTexturePhysicsClient:
    def __init__(self):
        self.loaded = []
        self.changed = []

    def loadTexture(self, path):
        self.loaded.append(path)
        return 17

    def changeVisualShape(self, *args, **kwargs):
        self.changed.append((args, kwargs))


class FakeTextureSim:
    def __init__(self):
        self.physics_client = FakeTexturePhysicsClient()
        self._bodies_idx = {'object': 23}


class FakePandaBackend:
    def __init__(
            self, *, goal_size, observation_size, action_size,
            native_goal_indices,
            success_on_step=None, truncate_on_step=None, include_success=True,
            angular=False, robot=None, direct_core=False):
        self.goal_size = goal_size
        self.observation_size = observation_size
        self.action_space = gym.spaces.Box(
            low=-np.ones(action_size, dtype=np.float32),
            high=np.ones(action_size, dtype=np.float32),
            dtype=np.float32,
        )
        self.native_goal_indices = tuple(native_goal_indices)
        self.success_on_step = success_on_step
        self.truncate_on_step = truncate_on_step
        self.include_success = include_success
        self.angular = angular
        self.task = FakeTask()
        self.robot = robot
        self.unwrapped = self
        self.reset_seeds = []
        self.actions = []
        self.steps = 0
        self.closed = False
        self.corrupt_native_goal = False
        self.direct_core = bool(direct_core)
        self.desired = np.linspace(
            0.1, 0.1 * goal_size, goal_size, dtype=np.float32,
        )
        if angular:
            self.desired = np.zeros(goal_size, dtype=np.float32)
            self.desired[-1] = 1.0
        self.achieved = np.zeros(goal_size, dtype=np.float32)
        if angular:
            self.achieved[-1] = 1.0
        if self.direct_core:
            self.task = FakeDirectTask(self)
            self.robot = FakeDirectRobot(self)
            self.sim = FakeDirectSim(self)

    def _observation(self):
        observation = np.arange(
            self.observation_size, dtype=np.float32,
        ) + 10.0
        observation[list(self.native_goal_indices)] = self.achieved
        if self.corrupt_native_goal:
            observation[self.native_goal_indices[0]] += 1.0
        return {
            'observation': observation,
            'achieved_goal': self.achieved.copy(),
            'desired_goal': self.desired.copy(),
        }

    def _get_obs(self):
        return self._observation()

    def reset(self, *, seed=None, options=None):
        self.reset_seeds.append(seed)
        self.steps = 0
        self.achieved = np.zeros(self.goal_size, dtype=np.float32)
        if self.angular:
            self.achieved[-1] = 1.0
        return self._observation(), {'reset_seed': seed}

    def _advance(self):
        self.steps += 1
        is_success = (
            self.success_on_step is not None
            and self.steps >= self.success_on_step
        )
        if is_success:
            self.achieved = self.desired.copy()
            if self.angular:
                self.achieved *= -1.0
        return is_success

    def step(self, action):
        if self.direct_core:
            raise AssertionError('adapter must not call RobotTaskEnv.step')
        self.actions.append(np.asarray(action).copy())
        is_success = self._advance()
        info = {}
        if self.include_success:
            info['is_success'] = is_success
        return (
            self._observation(),
            -1.0,
            is_success,
            self.steps == self.truncate_on_step,
            info,
        )

    def render(self):
        return np.zeros((2, 3, 3), dtype=np.uint8)

    def close(self):
        self.closed = True


class FakeTerminatingOuterEnv:
    """Mimic a Gymnasium wrapper that cannot be stepped after termination."""

    def __init__(self, core):
        self.unwrapped = core
        self.action_space = core.action_space
        self.reset_calls = 0
        self.step_calls = 0
        self._terminated = False

    def reset(self, **kwargs):
        self.reset_calls += 1
        self._terminated = False
        return self.unwrapped.reset(**kwargs)

    def step(self, action):
        self.step_calls += 1
        if self._terminated:
            raise RuntimeError('outer env stepped after termination')
        result = self.unwrapped.step(action)
        self._terminated = bool(result[2] or result[3])
        return result

    def close(self):
        return self.unwrapped.close()


def make_env(name, *, wrapped=False, **backend_kwargs):
    spec = TASK_SPECS[name]
    backend_kwargs.setdefault(
        'observation_size', spec['native_observation_size'],
    )
    backend_kwargs.setdefault('action_size', spec['action_size'])
    if spec['control_type'] == 'joints':
        backend_kwargs.setdefault('robot', FakePandaRobot())
    backend = FakePandaBackend(
        goal_size=len(spec['goal_dims']),
        native_goal_indices=spec['native_goal_indices'],
        angular=spec['angular_goal'],
        **backend_kwargs,
    )
    backend_env = FakeTerminatingOuterEnv(backend) if wrapped else backend
    return PandaGoalEnv(name, backend_env=backend_env), backend


class PandaGoalTaskRegistrationTest(unittest.TestCase):
    def test_all_end_effector_and_joint_control_tasks_are_registered(self):
        self.assertEqual(set(TASK_SPECS), set(EXPECTED_TASKS))
        for name, (
                goal_size, horizon, control_type,
                action_size) in EXPECTED_TASKS.items():
            with self.subTest(name=name):
                spec = TASK_SPECS[name]
                self.assertEqual(spec['backend_id'], name)
                self.assertEqual(spec['goal_dims'], tuple(range(goal_size)))
                self.assertEqual(spec['episode_length'], horizon)
                self.assertEqual(spec['control_type'], control_type)
                expected_joint_count = 0
                if control_type == 'joints':
                    expected_joint_count = (
                        PANDA_ARM_JOINT_COUNT
                        if action_size == 7 else PANDA_FULL_JOINT_COUNT
                    )
                self.assertEqual(spec['joint_count'], expected_joint_count)
                self.assertEqual(
                    spec['joint_state_size'],
                    2 * expected_joint_count,
                )
                self.assertEqual(spec['action_size'], action_size)
                self.assertEqual(spec['state_size'], EXPECTED_STATE_SIZES[name])
                self.assertIn(('panda_gym', name), CREATE_ENV_REGISTRY)

    def test_native_goal_indices_match_panda_gym_observation_layouts(self):
        for name, spec in TASK_SPECS.items():
            task = name.removeprefix('Panda').removesuffix('Joints-v3')
            if task == name.removeprefix('Panda'):
                task = task.removesuffix('-v3')
            with self.subTest(name=name):
                self.assertEqual(
                    spec['native_goal_indices'],
                    EXPECTED_NATIVE_GOAL_INDICES[task],
                )

    def test_action_dimensions_match_panda_control_modes(self):
        for name, (goal_size, _, _, action_size) in EXPECTED_TASKS.items():
            with self.subTest(name=name):
                env, _ = make_env(name, action_size=action_size)
                self.assertEqual(env.action_space.shape, (action_size,))
                self.assertEqual(
                    env.observation_space['observation'].shape,
                    (EXPECTED_STATE_SIZES[name],),
                )
                self.assertEqual(env.goal_dims, tuple(range(goal_size)))
                env.close()

    def test_missing_flip_texture_uses_pybullet_fallback(self):
        calls = []

        def original(sim, *args, **kwargs):
            calls.append((sim, args, kwargs))

        sim = FakeTextureSim()
        create_box = _create_box_with_texture_fallback(
            original, '/textures/colors16.png',
        )
        create_box(
            sim, body_name='object', texture='colored_cube.png', mass=1.0,
        )

        self.assertIs(calls[0][0], sim)
        self.assertIsNone(calls[0][2]['texture'])
        self.assertEqual(sim.physics_client.loaded, ['/textures/colors16.png'])
        self.assertEqual(
            sim.physics_client.changed,
            [((23, -1), {'textureUniqueId': 17})],
        )


class PandaGoalObservationTest(unittest.TestCase):
    def test_native_goal_is_a_prefix_of_the_same_shaped_qrl_state(self):
        env, backend = make_env('PandaPickAndPlace-v3')
        packed = env.reset()

        state = packed['observation']
        self.assertEqual(state.shape, (EXPECTED_STATE_SIZES[env.name],))
        for value in packed.values():
            self.assertEqual(value.shape, state.shape)
            self.assertEqual(value.dtype, np.float32)
        np.testing.assert_array_equal(
            state[:3], backend.achieved,
        )
        np.testing.assert_array_equal(
            packed['achieved_goal'], state,
        )
        np.testing.assert_array_equal(
            packed['desired_goal'][:3], backend.desired,
        )
        native = backend._observation()['observation']
        np.testing.assert_array_equal(
            state[3:],
            np.delete(native, TASK_SPECS[env.name]['native_goal_indices']),
        )
        np.testing.assert_array_equal(
            packed['desired_goal'][3:],
            np.zeros(state.size - 3, dtype=np.float32),
        )
        self.assertTrue(env.observation_space.contains(packed))

    def test_mismatched_native_goal_coordinates_are_rejected(self):
        env, backend = make_env('PandaStack-v3')
        backend.corrupt_native_goal = True
        with self.assertRaisesRegex(
                ValueError, 'native observation coordinates .* achieved_goal'):
            env.reset()

    def test_seed_is_forwarded_on_the_next_reset(self):
        env, backend = make_env('PandaReach-v3')
        env.seed(2718)
        env.reset()
        self.assertEqual(backend.reset_seeds[-1], 2718)

        env.reset(seed=3141)
        self.assertEqual(backend.reset_seeds[-1], 3141)

    def test_changed_goal_within_episode_is_rejected(self):
        env, backend = make_env('PandaPush-v3')
        env.reset()
        backend.desired += 1.0
        with self.assertRaisesRegex(RuntimeError, 'changed the desired goal'):
            env.step(np.zeros(env.action_space.shape, dtype=np.float32))

    def test_joint_control_appends_required_positions_and_velocities(self):
        for name, (goal_size, _, control_type, _) in EXPECTED_TASKS.items():
            if control_type != 'joints':
                continue
            with self.subTest(name=name):
                env, backend = make_env(name)
                packed = env.reset()
                state = packed['observation']
                joint_count = TASK_SPECS[name]['joint_count']
                self.assertEqual(
                    state.size,
                    backend.observation_size + 2 * joint_count,
                )
                np.testing.assert_array_equal(
                    state[-2 * joint_count:-joint_count],
                    backend.robot.positions[:joint_count],
                )
                np.testing.assert_array_equal(
                    state[-joint_count:], backend.robot.velocities[:joint_count],
                )
                self.assertEqual(env.goal_dims, tuple(range(goal_size)))
                np.testing.assert_array_equal(
                    packed['desired_goal'][goal_size:],
                    np.zeros(state.size - goal_size, dtype=np.float32),
                )

    def test_locked_gripper_uses_seven_joints_and_controllable_uses_nine(self):
        self.assertEqual(TASK_SPECS['PandaReachJoints-v3']['joint_count'], 7)
        self.assertEqual(
            TASK_SPECS['PandaPickAndPlaceJoints-v3']['joint_count'], 9,
        )

    def test_joint_control_falls_back_to_simulator_joint_api(self):
        robot = FakeSimOnlyPandaRobot()
        env, _ = make_env('PandaReachJoints-v3', robot=robot)
        state = env.reset()['observation']
        np.testing.assert_array_equal(state[-14:-7], robot.sim.positions[:7])
        np.testing.assert_array_equal(state[-7:], robot.sim.velocities[:7])
        self.assertEqual(len(robot.sim.calls), 28)

        robot = FakeSimOnlyPandaRobot()
        env, _ = make_env('PandaPickAndPlaceJoints-v3', robot=robot)
        state = env.reset()['observation']
        np.testing.assert_array_equal(state[-18:-9], robot.sim.positions)
        np.testing.assert_array_equal(state[-9:], robot.sim.velocities)
        self.assertEqual(len(robot.sim.calls), 36)

    def test_joint_control_fails_clearly_without_joint_state_api(self):
        with self.assertRaisesRegex(
                RuntimeError,
                'requires positions and velocities for 7 Panda joints'):
            make_env('PandaReachJoints-v3', robot=None)
        with self.assertRaisesRegex(
                RuntimeError,
                'requires positions and velocities for 9 Panda joints'):
            make_env('PandaPickAndPlaceJoints-v3', robot=None)

    def test_end_effector_control_does_not_require_joint_state_api(self):
        env, backend = make_env('PandaReach-v3', robot=None)
        state = env.reset()['observation']
        self.assertEqual(state.size, backend.observation_size)


class PandaGoalTransitionTest(unittest.TestCase):
    def test_native_success_does_not_end_fixed_length_episode(self):
        env, backend = make_env(
            'PandaReach-v3', success_on_step=1, action_size=3,
        )
        env.reset()
        packed, reward, done, info = env.step(
            np.zeros(env.action_space.shape, dtype=np.float64),
        )

        self.assertFalse(done)
        self.assertEqual(reward, 1.0)
        self.assertTrue(info['is_success'])
        self.assertNotIn('TimeLimit.truncated', info)
        self.assertAlmostEqual(info['goal_distance'], 0.0)
        self.assertEqual(info['backend_reward'], -1.0)
        self.assertEqual(backend.actions[-1].dtype, np.float32)
        np.testing.assert_array_equal(
            packed['observation'][:3], backend.desired,
        )

    def test_success_then_another_step_bypasses_terminated_outer_wrapper(self):
        env, backend = make_env(
            'PandaReach-v3', success_on_step=1, wrapped=True,
            direct_core=True,
        )
        outer = env._env
        env.reset()
        _, first_reward, first_done, first_info = env.step(
            np.zeros(3, dtype=np.float32)
        )
        _, second_reward, second_done, second_info = env.step(
            np.zeros(3, dtype=np.float32)
        )

        self.assertEqual(backend.steps, 2)
        self.assertEqual(outer.reset_calls, 0)
        self.assertEqual(outer.step_calls, 0)
        self.assertEqual(len(backend.actions), 2)
        self.assertEqual((first_reward, second_reward), (1.0, 1.0))
        self.assertFalse(first_done)
        self.assertFalse(second_done)
        self.assertTrue(first_info['is_success'])
        self.assertTrue(second_info['is_success'])

    def test_registered_horizon_is_reported_as_timeout(self):
        env, _ = make_env('PandaReach-v3', action_size=3)
        env.reset()
        for _ in range(env.episode_length - 1):
            _, _, _, info = env.step(np.zeros(3, dtype=np.float32))
            self.assertNotIn('TimeLimit.truncated', info)
        _, _, done, info = env.step(np.zeros(3, dtype=np.float32))
        self.assertFalse(done)
        self.assertTrue(info['TimeLimit.truncated'])
        with self.assertRaisesRegex(RuntimeError, 'after the fixed episode horizon'):
            env.step(np.zeros(3, dtype=np.float32))

    def test_stack_uses_its_native_one_hundred_step_horizon(self):
        env, _ = make_env('PandaStack-v3')
        self.assertEqual(env.episode_length, 100)

    def test_early_backend_truncation_is_rejected(self):
        env, _ = make_env('PandaReach-v3', truncate_on_step=1)
        env.reset()
        with self.assertRaisesRegex(RuntimeError, 'truncated after 1 steps'):
            env.step(np.zeros(env.action_space.shape, dtype=np.float32))

    def test_success_falls_back_to_native_task_evaluator(self):
        env, backend = make_env(
            'PandaReach-v3', success_on_step=1, include_success=False,
        )
        env.reset()
        _, reward, _, info = env.step(
            np.zeros(env.action_space.shape, dtype=np.float32),
        )
        self.assertTrue(info['is_success'])
        self.assertEqual(reward, 1.0)

    def test_flip_distance_treats_opposite_quaternion_signs_as_equal(self):
        env, _ = make_env(
            'PandaFlip-v3', success_on_step=1,
        )
        env.reset()
        packed, _, _, info = env.step(
            np.zeros(env.action_space.shape, dtype=np.float32),
        )
        self.assertTrue(info['is_success'])
        self.assertAlmostEqual(info['goal_distance'], 0.0)
        np.testing.assert_array_equal(
            packed['observation'][:4], packed['desired_goal'][:4],
        )

    def test_close_and_render_delegate_to_backend(self):
        env, backend = make_env('PandaSlide-v3')
        self.assertEqual(env.render().shape, (2, 3, 3))
        env.close()
        self.assertTrue(backend.closed)


if __name__ == '__main__':
    unittest.main()

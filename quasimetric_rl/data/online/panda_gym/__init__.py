from __future__ import annotations

import logging
from pathlib import Path
import threading
from typing import Any, Dict, Optional, Tuple

import gym
import numpy as np

from ..goal_env import (
    pack_goal_observation,
    unpack_reset_result,
    unpack_step_result,
    vector_goal_observation_space,
)
from ..memory import register_online_env


_TASKS = {
    'Reach': dict(
        goal_size=3, episode_length=50, angular_goal=False,
        native_observation_size=6, ee_action_size=3, joints_action_size=7,
        native_goal_indices=(0, 1, 2), block_gripper=True,
    ),
    'Push': dict(
        goal_size=3, episode_length=50, angular_goal=False,
        native_observation_size=18, ee_action_size=3, joints_action_size=7,
        native_goal_indices=(6, 7, 8), block_gripper=True,
    ),
    'Slide': dict(
        goal_size=3, episode_length=50, angular_goal=False,
        native_observation_size=18, ee_action_size=3, joints_action_size=7,
        native_goal_indices=(6, 7, 8), block_gripper=True,
    ),
    'PickAndPlace': dict(
        goal_size=3, episode_length=50, angular_goal=False,
        native_observation_size=19, ee_action_size=4, joints_action_size=8,
        native_goal_indices=(7, 8, 9), block_gripper=False,
    ),
    'Stack': dict(
        goal_size=6, episode_length=100, angular_goal=False,
        native_observation_size=31, ee_action_size=4, joints_action_size=8,
        native_goal_indices=(7, 8, 9, 19, 20, 21), block_gripper=False,
    ),
    'Flip': dict(
        goal_size=4, episode_length=50, angular_goal=True,
        native_observation_size=20, ee_action_size=4, joints_action_size=8,
        native_goal_indices=(10, 11, 12, 13), block_gripper=False,
    ),
}

PANDA_ARM_JOINT_COUNT = 7
PANDA_FULL_JOINT_COUNT = 9
_PANDA_BACKEND_CREATION_LOCK = threading.Lock()


def _task_spec(task: str, *, joints: bool) -> Dict[str, Any]:
    base = _TASKS[task]
    suffix = 'Joints' if joints else ''
    native_goal_indices = tuple(base['native_goal_indices'])
    if (
            len(native_goal_indices) != base['goal_size']
            or len(set(native_goal_indices)) != len(native_goal_indices)
            or min(native_goal_indices) < 0
            or max(native_goal_indices) >= base['native_observation_size']):
        raise ValueError(
            f'Invalid native goal indices for Panda{task}: '
            f'{native_goal_indices}'
        )
    joint_count = 0
    if joints:
        joint_count = (
            PANDA_ARM_JOINT_COUNT
            if base['block_gripper'] else PANDA_FULL_JOINT_COUNT
        )
    joint_state_size = 2 * joint_count
    return dict(
        backend_id=f'Panda{task}{suffix}-v3',
        goal_dims=tuple(range(base['goal_size'])),
        native_goal_indices=native_goal_indices,
        episode_length=base['episode_length'],
        angular_goal=base['angular_goal'],
        block_gripper=base['block_gripper'],
        control_type='joints' if joints else 'ee',
        native_observation_size=base['native_observation_size'],
        joint_count=joint_count,
        joint_state_size=joint_state_size,
        # Removing the native goal copy and prepending the canonical copy leaves
        # the base state size unchanged.
        state_size=base['native_observation_size'] + joint_state_size,
        action_size=(
            base['joints_action_size'] if joints else base['ee_action_size']
        ),
    )


TASK_SPECS = {
    f'Panda{task}{"Joints" if joints else ""}-v3': _task_spec(
        task, joints=joints,
    )
    for task in _TASKS
    for joints in (False, True)
}

GOAL_DIMS = {
    task_name: tuple(task_spec['goal_dims'])
    for task_name, task_spec in TASK_SPECS.items()
}


def _create_box_with_texture_fallback(original_create_box, fallback_texture):
    fallback_texture = str(fallback_texture)

    def create_box(sim, *args, **kwargs):
        if kwargs.get('texture') != 'colored_cube.png':
            return original_create_box(sim, *args, **kwargs)

        kwargs = dict(kwargs)
        kwargs['texture'] = None
        result = original_create_box(sim, *args, **kwargs)
        body_name = kwargs.get('body_name', args[0] if args else None)
        if body_name is None:
            raise RuntimeError('Panda-Gym texture fallback requires body_name')
        texture_uid = sim.physics_client.loadTexture(fallback_texture)
        sim.physics_client.changeVisualShape(
            sim._bodies_idx[body_name], -1, textureUniqueId=texture_uid,
        )
        return result

    return create_box


def _make_backend(name: str):
    try:
        import gymnasium
        import panda_gym  # registers Panda-Gym environments
        import panda_gym.assets as panda_assets
    except ImportError as exc:
        raise ImportError(
            'Panda-Gym environments require the optional packages '
            '`gymnasium` and `panda-gym`. Install a Panda-Gym version that '
            'provides the v3 environment IDs.'
        ) from exc

    backend_id = TASK_SPECS[name]['backend_id']
    colored_cube = Path(
        panda_assets.get_data_path(), 'colored_cube.png',
    )
    if 'Flip' not in name or colored_cube.is_file():
        return gymnasium.make(backend_id)

    # panda-gym 3.0.7's wheel omits the Flip texture. PyBullet's packaged
    # multi-color texture preserves the visual orientation cue without
    # changing any collision, dynamics, state, goal, or reward semantics.
    try:
        import pybullet_data
        from panda_gym.pybullet import PyBullet
    except ImportError as exc:
        raise RuntimeError(
            'PandaFlip requires Panda-Gym colored_cube.png or PyBullet data'
        ) from exc
    fallback_texture = Path(pybullet_data.getDataPath(), 'colors16.png')
    if not fallback_texture.is_file():
        raise RuntimeError(
            f'PandaFlip texture fallback is missing: {fallback_texture}'
        )

    with _PANDA_BACKEND_CREATION_LOCK:
        original_create_box = PyBullet.create_box
        PyBullet.create_box = _create_box_with_texture_fallback(
            original_create_box, fallback_texture,
        )
        try:
            logging.warning(
                'Panda-Gym package lacks colored_cube.png; using %s for %s',
                fallback_texture, backend_id,
            )
            return gymnasium.make(backend_id)
        finally:
            PyBullet.create_box = original_create_box


def _to_float32_vector(value: Any, *, field: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError(
            f'Panda-Gym {field} must be a vector, got shape {array.shape}'
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f'Panda-Gym {field} contains non-finite values')
    return array


def _goal_distance(
        achieved: np.ndarray, desired: np.ndarray, *, angular: bool) -> float:
    if angular:
        # PandaFlip uses 1 - <q1, q2>^2, which is invariant to q and -q.
        return float(1.0 - np.clip(np.dot(achieved, desired), -1.0, 1.0) ** 2)
    return float(np.linalg.norm(achieved - desired))


def _canonicalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError('PandaFlip returned an invalid zero quaternion')
    canonical = np.asarray(quaternion, dtype=np.float32) / norm
    anchor = float(canonical[-1])
    if abs(anchor) <= 1e-7:
        nonzero = np.flatnonzero(np.abs(canonical[:-1]) > 1e-7)
        anchor = float(canonical[nonzero[0]]) if nonzero.size else 0.0
    if anchor < 0:
        canonical = -canonical
    return canonical.astype(np.float32)


def _joint_vector(value: Any, *, count: int, field: str) -> np.ndarray:
    vector = _to_float32_vector(value, field=field)
    if vector.size < count:
        raise ValueError(
            f'{field} must contain at least {count} Panda joints, '
            f'got shape {vector.shape}'
        )
    return vector[:count].copy()


def _read_value(source: Any, name: str) -> Any:
    value = getattr(source, name)
    return value() if callable(value) else value


class PandaGoalEnv(gym.Env):
    """Adapt Panda-Gym GoalEnv observations to QRL's fixed-horizon API."""

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, name: str, *, backend_env=None):
        if name not in TASK_SPECS:
            raise ValueError(f'Unknown Panda-Gym goal task: {name!r}')
        self.name = name
        self.task_spec = TASK_SPECS[name]
        self.backend_id = self.task_spec['backend_id']
        self.goal_dims = tuple(self.task_spec['goal_dims'])
        self.episode_length = int(self.task_spec['episode_length'])
        self._env = _make_backend(name) if backend_env is None else backend_env
        self._core_env = getattr(self._env, 'unwrapped', self._env)

        action_space = self._core_env.action_space
        if not hasattr(action_space, 'low') or not hasattr(action_space, 'high'):
            raise TypeError('Panda-Gym online training requires a Box action space')
        self.action_space = gym.spaces.Box(
            low=np.asarray(action_space.low, dtype=np.float32),
            high=np.asarray(action_space.high, dtype=np.float32),
            dtype=np.float32,
        )
        expected_action_shape = (int(self.task_spec['action_size']),)
        if self.action_space.shape != expected_action_shape:
            raise ValueError(
                f'{self.backend_id} expected action shape '
                f'{expected_action_shape}, got {self.action_space.shape}'
            )
        self.reward_mode = 'positive'
        self._pending_seed: Optional[int] = int(
            np.random.randint(0, 2 ** 31 - 1)
        )
        if hasattr(self.action_space, 'seed'):
            self.action_space.seed(self._pending_seed)

        raw_observation, _ = self._backend_reset()
        state, _, desired = self._state_and_goals(raw_observation)
        self.observation_space = vector_goal_observation_space(state.size)
        self._goal_values = desired
        self._elapsed_steps = 0

    def _backend_reset(self, *, options=None):
        seed = self._pending_seed
        self._pending_seed = None
        kwargs = {}
        if seed is not None:
            kwargs['seed'] = seed
        if options is not None:
            kwargs['options'] = options
        try:
            result = self._core_env.reset(**kwargs)
        except TypeError:
            if seed is not None and hasattr(self._core_env, 'seed'):
                self._core_env.seed(seed)
            if options is not None:
                try:
                    result = self._core_env.reset(options=options)
                except TypeError:
                    result = self._core_env.reset()
            else:
                result = self._core_env.reset()
        return unpack_reset_result(result)

    def _state_and_goals(
            self, raw_observation: Any,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not isinstance(raw_observation, dict):
            raise TypeError(
                'Panda-Gym must return a GoalEnv observation dictionary'
            )
        missing = {
            'observation', 'achieved_goal', 'desired_goal',
        }.difference(raw_observation)
        if missing:
            raise KeyError(
                f'Panda-Gym observation is missing fields: {sorted(missing)}'
            )

        observation = _to_float32_vector(
            raw_observation['observation'], field='observation',
        )
        expected_observation_size = int(
            self.task_spec['native_observation_size']
        )
        if observation.shape != (expected_observation_size,):
            raise ValueError(
                f'{self.backend_id} expected a native observation of size '
                f'{expected_observation_size}, got shape {observation.shape}'
            )
        raw_achieved = _to_float32_vector(
            raw_observation['achieved_goal'], field='achieved_goal',
        )
        desired = _to_float32_vector(
            raw_observation['desired_goal'], field='desired_goal',
        )
        expected_goal_size = len(self.goal_dims)
        if raw_achieved.shape != (expected_goal_size,):
            raise ValueError(
                f'{self.backend_id} expected an achieved goal of size '
                f'{expected_goal_size}, got shape {raw_achieved.shape}'
            )
        if desired.shape != raw_achieved.shape:
            raise ValueError(
                f'{self.backend_id} achieved and desired goal shapes differ: '
                f'{raw_achieved.shape} != {desired.shape}'
            )
        native_goal_indices = tuple(self.task_spec['native_goal_indices'])
        native_achieved = observation[list(native_goal_indices)]
        if self.task_spec['angular_goal']:
            achieved = _canonicalize_quaternion(raw_achieved)
            comparable_native_achieved = _canonicalize_quaternion(
                native_achieved
            )
            desired = _canonicalize_quaternion(desired)
        else:
            achieved = raw_achieved
            comparable_native_achieved = native_achieved
        if not np.allclose(
                comparable_native_achieved, achieved, rtol=0, atol=1e-6):
            raise ValueError(
                f'{self.backend_id} native observation coordinates '
                f'{native_goal_indices} do not match achieved_goal'
            )

        # Move the task coordinates to a canonical prefix without retaining a
        # duplicate in the non-goal branch.
        non_goal_observation = np.delete(observation, native_goal_indices)
        state_parts = [achieved, non_goal_observation]
        if self.task_spec['control_type'] == 'joints':
            joint_positions, joint_velocities = self._panda_joint_state()
            state_parts.extend((joint_positions, joint_velocities))
        state = np.concatenate(state_parts).astype(np.float32)
        if state.shape != (int(self.task_spec['state_size']),):
            raise RuntimeError(
                f'{self.backend_id} constructed state shape {state.shape}; '
                f'expected {(self.task_spec["state_size"],)}'
            )
        return state, achieved, desired

    @property
    def backend(self):
        return self._core_env

    def _joint_indices(self, robot: Any, count: int) -> Tuple[int, ...]:
        if hasattr(robot, 'joint_indices'):
            indices = np.asarray(robot.joint_indices).reshape(-1)
            if indices.size >= count:
                return tuple(
                    int(value)
                    for value in indices[:count]
                )
        if count == PANDA_ARM_JOINT_COUNT:
            for name in ('arm_joint_indices', 'arm_joint_ids'):
                if hasattr(robot, name):
                    indices = np.asarray(getattr(robot, name)).reshape(-1)
                    if indices.size >= count:
                        return tuple(int(value) for value in indices[:count])
            # Panda-Gym's seven arm joints use PyBullet indices 0 through 6.
            return tuple(range(PANDA_ARM_JOINT_COUNT))
        arm_indices = tuple(range(PANDA_ARM_JOINT_COUNT))
        fingers = getattr(robot, 'fingers_indices', (9, 10))
        finger_indices = tuple(int(value) for value in np.asarray(fingers).reshape(-1))
        if len(finger_indices) < PANDA_FULL_JOINT_COUNT - PANDA_ARM_JOINT_COUNT:
            raise ValueError('Panda robot did not expose both gripper joint indices')
        return arm_indices + finger_indices[:2]

    def _panda_joint_state(self) -> Tuple[np.ndarray, np.ndarray]:
        backend = self.backend
        robot = getattr(backend, 'robot', None)
        joint_count = int(self.task_spec['joint_count'])
        sources = []
        if robot is not None:
            sources.append(('robot', robot))
        if backend is not robot:
            sources.append(('backend', backend))
        errors = []

        # Support backends that expose complete joint vectors as properties or
        # zero-argument methods.
        bulk_names = (
            ('joint_positions', 'joint_velocities'),
            ('get_joint_positions', 'get_joint_velocities'),
            ('get_joint_angles', 'get_joint_velocities'),
        )
        for source_name, source in sources:
            for position_name, velocity_name in bulk_names:
                if not (
                        hasattr(source, position_name)
                        and hasattr(source, velocity_name)):
                    continue
                path = f'{source_name}.{position_name}/{velocity_name}'
                try:
                    return (
                        _joint_vector(
                            _read_value(source, position_name),
                            count=joint_count,
                            field=f'{path} positions',
                        ),
                        _joint_vector(
                            _read_value(source, velocity_name),
                            count=joint_count,
                            field=f'{path} velocities',
                        ),
                    )
                except Exception as exc:
                    errors.append(f'{path}: {exc}')

        indices = self._joint_indices(
            robot if robot is not None else backend, joint_count,
        )
        for source_name, source in sources:
            angle_fn = getattr(source, 'get_joint_angle', None)
            velocity_fn = getattr(source, 'get_joint_velocity', None)
            if not callable(angle_fn) or not callable(velocity_fn):
                continue
            path = f'{source_name}.get_joint_angle/get_joint_velocity'
            try:
                positions = [angle_fn(joint=index) for index in indices]
                velocities = [velocity_fn(joint=index) for index in indices]
            except TypeError:
                try:
                    positions = [angle_fn(index) for index in indices]
                    velocities = [velocity_fn(index) for index in indices]
                except Exception as exc:
                    errors.append(f'{path}: {exc}')
                    continue
            except Exception as exc:
                errors.append(f'{path}: {exc}')
                continue
            return (
                _joint_vector(
                    positions, count=joint_count, field=f'{path} positions',
                ),
                _joint_vector(
                    velocities, count=joint_count, field=f'{path} velocities',
                ),
            )

        if robot is not None:
            sim = getattr(robot, 'sim', None)
            body_name = getattr(robot, 'body_name', 'panda')
            if sim is not None:
                angle_fn = getattr(sim, 'get_joint_angle', None)
                velocity_fn = getattr(sim, 'get_joint_velocity', None)
                path = 'robot.sim.get_joint_angle/get_joint_velocity'
                if callable(angle_fn) and callable(velocity_fn):
                    try:
                        positions = [
                            angle_fn(body_name, index) for index in indices
                        ]
                        velocities = [
                            velocity_fn(body_name, index) for index in indices
                        ]
                        return (
                            _joint_vector(
                                positions, count=joint_count,
                                field=f'{path} positions',
                            ),
                            _joint_vector(
                                velocities, count=joint_count,
                                field=f'{path} velocities',
                            ),
                        )
                    except Exception as exc:
                        errors.append(f'{path}: {exc}')

                physics_client = getattr(sim, 'physics_client', None)
                body_indices = getattr(sim, '_bodies_idx', None)
                get_joint_states = getattr(
                    physics_client, 'getJointStates', None,
                )
                path = 'robot.sim.physics_client.getJointStates'
                if callable(get_joint_states) and body_indices is not None:
                    try:
                        body_index = body_indices[body_name]
                        joint_states = get_joint_states(body_index, indices)
                        positions = [state[0] for state in joint_states]
                        velocities = [state[1] for state in joint_states]
                        return (
                            _joint_vector(
                                positions, count=joint_count,
                                field=f'{path} positions',
                            ),
                            _joint_vector(
                                velocities, count=joint_count,
                                field=f'{path} velocities',
                            ),
                        )
                    except Exception as exc:
                        errors.append(f'{path}: {exc}')

        details = '; '.join(errors) if errors else 'no compatible API was found'
        raise RuntimeError(
            f'{self.backend_id} joints-control state requires positions and '
            f'velocities for {joint_count} Panda joints, but '
            f'they could not be read from the backend or robot API ({details})'
        )

    def _pack(self, state: np.ndarray):
        return pack_goal_observation(state, self._goal_values, self.goal_dims)

    def seed(self, seed: Optional[int] = None):
        if seed is None:
            seed = int(np.random.randint(0, 2 ** 31 - 1))
        self._pending_seed = int(seed)
        if hasattr(self.action_space, 'seed'):
            self.action_space.seed(int(seed))
        return [int(seed)]

    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            self.seed(seed)
        raw_observation, _ = self._backend_reset(options=options)
        self._elapsed_steps = 0
        state, _, self._goal_values = self._state_and_goals(raw_observation)
        return self._pack(state)

    def _is_success(
            self, achieved: np.ndarray, desired: np.ndarray,
            backend_info: Dict[str, Any]) -> bool:
        if 'is_success' in backend_info:
            return bool(np.asarray(backend_info['is_success']).item())

        task = getattr(self.backend, 'task', None)
        if task is None or not hasattr(task, 'is_success'):
            raise RuntimeError(
                f'{self.backend_id} did not expose `info["is_success"]` or '
                '`unwrapped.task.is_success`; success cannot be evaluated'
            )
        return bool(np.asarray(task.is_success(achieved, desired)).item())

    def _step_core(self, action: np.ndarray):
        """Advance Panda-Gym as a continuing process without a terminal latch."""
        backend = self.backend
        robot = getattr(backend, 'robot', None)
        sim = getattr(backend, 'sim', None)
        task = getattr(backend, 'task', None)
        get_observation = getattr(backend, '_get_obs', None)
        set_action = getattr(robot, 'set_action', None)
        sim_step = getattr(sim, 'step', None)
        get_goal = getattr(task, 'get_goal', None)
        is_success_fn = getattr(task, 'is_success', None)
        compute_reward = getattr(task, 'compute_reward', None)
        direct_core_api = all(callable(value) for value in (
            get_observation,
            set_action,
            sim_step,
            get_goal,
            is_success_fn,
            compute_reward,
        ))
        if not direct_core_api:
            # Injected and legacy test backends may only expose Gymnasium step.
            return unpack_step_result(backend.step(action))

        set_action(action)
        sim_step()
        raw_observation = get_observation()
        raw_achieved = _to_float32_vector(
            raw_observation['achieved_goal'], field='achieved_goal',
        )
        raw_desired = _to_float32_vector(get_goal(), field='desired_goal')
        success_value = np.asarray(
            is_success_fn(raw_achieved, raw_desired),
        )
        if success_value.size != 1:
            raise ValueError('Panda-Gym task.is_success must return a scalar')
        is_success = bool(success_value.item())
        info = {'is_success': is_success}
        reward_value = np.asarray(
            compute_reward(raw_achieved, raw_desired, info),
        )
        if reward_value.size != 1:
            raise ValueError('Panda-Gym task.compute_reward must return a scalar')
        return raw_observation, float(reward_value.item()), is_success, False, info

    def step(self, action):
        if self._elapsed_steps >= self.episode_length:
            raise RuntimeError('step() called after the fixed episode horizon')
        backend_action_space = self.backend.action_space
        action_dtype = getattr(backend_action_space, 'dtype', np.float32)
        raw_observation, backend_reward, terminated, truncated, info = (
            self._step_core(np.asarray(action, dtype=action_dtype))
        )
        self._elapsed_steps += 1
        state, achieved, desired = self._state_and_goals(raw_observation)
        if not np.allclose(desired, self._goal_values, rtol=0, atol=1e-6):
            raise RuntimeError(
                f'{self.backend_id} changed the desired goal within an episode'
            )

        is_success = self._is_success(achieved, desired, info)
        distance = _goal_distance(
            achieved,
            desired,
            angular=bool(self.task_spec['angular_goal']),
        )
        if self.reward_mode == 'dense':
            reward = float(np.exp(-distance))
        elif self.reward_mode == 'negative':
            reward = float(is_success) - 1.0
        else:
            reward = float(is_success)

        # Panda success is diagnostic while this adapter owns the fixed horizon.
        if bool(terminated) != is_success:
            raise RuntimeError(
                f'{self.backend_id} core termination disagrees with success: '
                f'terminated={terminated}, is_success={is_success}'
            )
        timeout = bool(truncated) or self._elapsed_steps == self.episode_length
        if truncated and self._elapsed_steps != self.episode_length:
            raise RuntimeError(
                f'{self.backend_id} truncated after {self._elapsed_steps} steps; '
                f'expected {self.episode_length}'
            )

        info.update(
            is_success=is_success,
            goal_distance=distance,
            backend_id=self.backend_id,
            backend_reward=float(backend_reward),
        )
        if timeout:
            info['TimeLimit.truncated'] = True
        else:
            info.pop('TimeLimit.truncated', None)
        return self._pack(state), reward, False, info

    def render(self, *args, **kwargs):
        try:
            return self.backend.render(*args, **kwargs)
        except TypeError:
            return self.backend.render()

    def close(self):
        close = getattr(self._env, 'close', None)
        if close is not None:
            return close()
        return None


def create_env_from_spec(name: str):
    return PandaGoalEnv(name)


for task_name, task_spec in TASK_SPECS.items():
    register_online_env(
        'panda_gym',
        task_name,
        create_env_fn=lambda task_name=task_name: create_env_from_spec(task_name),
        episode_length=task_spec['episode_length'],
        goal_dims=task_spec['goal_dims'],
    )


__all__ = [
    'GOAL_DIMS',
    'PANDA_ARM_JOINT_COUNT',
    'PANDA_FULL_JOINT_COUNT',
    'PandaGoalEnv',
    'TASK_SPECS',
    'create_env_from_spec',
]

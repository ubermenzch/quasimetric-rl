from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import gym
import numpy as np

from ..goal_env import (
    pack_goal_observation,
    unpack_reset_result,
    unpack_step_result,
    vector_goal_observation_space,
)
from ..memory import register_online_env


_HAND_REACH_NATIVE_GOAL_INDICES = tuple(range(48, 63))
_HAND_POSE_NATIVE_GOAL_INDICES = tuple(range(54, 61))
_HAND_ROTATION_NATIVE_GOAL_INDICES = tuple(range(57, 61))
_POINT_MAZE_NATIVE_GOAL_INDICES = (0, 1)


def _spec(
        *, episode_length: int, raw_goal_size: int, goal_encoding: str,
        observation_dim: int, action_dim: int,
        native_goal_indices: Sequence[int],
        backend_ids: Optional[Sequence[str]] = None,
        backend_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    encoded_sizes = {
        'vector': raw_goal_size,
        'rotation_matrix': 9,
        'pose_rotation_matrix': 12,
    }
    goal_size = encoded_sizes[goal_encoding]
    native_goal_indices = tuple(int(index) for index in native_goal_indices)
    if native_goal_indices != tuple(sorted(set(native_goal_indices))):
        raise ValueError('native_goal_indices must be sorted and unique')
    if any(index < 0 or index >= observation_dim for index in native_goal_indices):
        raise ValueError(
            f'native_goal_indices {native_goal_indices} exceed observation '
            f'size {observation_dim}'
        )
    repeated_goal_size = 4 if goal_encoding == 'rotation_matrix' else raw_goal_size
    if native_goal_indices and len(native_goal_indices) != repeated_goal_size:
        raise ValueError(
            f'{goal_encoding} repeats {repeated_goal_size} raw goal values, '
            f'got {len(native_goal_indices)} native indices'
        )
    return dict(
        episode_length=int(episode_length),
        raw_goal_size=int(raw_goal_size),
        goal_encoding=goal_encoding,
        goal_dims=tuple(range(goal_size)),
        observation_dim=int(observation_dim),
        native_goal_indices=native_goal_indices,
        state_dim=int(observation_dim - len(native_goal_indices) + goal_size),
        action_dim=int(action_dim),
        backend_ids=None if backend_ids is None else tuple(backend_ids),
        backend_kwargs={} if backend_kwargs is None else dict(backend_kwargs),
    )


TASK_SPECS = {
    # Keep the requested v2 protocol exact. Gymnasium-Robotics 1.3.2 added a
    # corrected v3 task, but silently substituting it would mix environment
    # versions under one experiment name.
    'HandReach-v2': _spec(
        episode_length=50,
        raw_goal_size=15,
        goal_encoding='vector',
        observation_dim=63,
        action_dim=20,
        native_goal_indices=_HAND_REACH_NATIVE_GOAL_INDICES,
    ),
    'HandManipulateBlockRotateZ-v1': _spec(
        episode_length=100, raw_goal_size=7,
        goal_encoding='rotation_matrix',
        observation_dim=61, action_dim=20,
        native_goal_indices=_HAND_ROTATION_NATIVE_GOAL_INDICES,
    ),
    # Egg and pen use the same all-axis quaternion angle goal geometry as the
    # block environments; they are not reduced to a symmetry-axis goal.
    'HandManipulateEggRotate-v1': _spec(
        episode_length=100, raw_goal_size=7,
        goal_encoding='rotation_matrix',
        observation_dim=61, action_dim=20,
        native_goal_indices=_HAND_ROTATION_NATIVE_GOAL_INDICES,
    ),
    'HandManipulatePenRotate-v1': _spec(
        episode_length=100, raw_goal_size=7,
        goal_encoding='rotation_matrix',
        observation_dim=61, action_dim=20,
        native_goal_indices=_HAND_ROTATION_NATIVE_GOAL_INDICES,
    ),
    'PointMaze_UMaze-v3': _spec(
        episode_length=300, raw_goal_size=2, goal_encoding='vector',
        observation_dim=4, action_dim=2,
        native_goal_indices=_POINT_MAZE_NATIVE_GOAL_INDICES,
        backend_kwargs=dict(continuing_task=True, reset_target=False),
    ),
    'PointMaze_Open-v3': _spec(
        episode_length=300, raw_goal_size=2, goal_encoding='vector',
        observation_dim=4, action_dim=2,
        native_goal_indices=_POINT_MAZE_NATIVE_GOAL_INDICES,
        backend_kwargs=dict(continuing_task=True, reset_target=False),
    ),
    'PointMaze_Medium-v3': _spec(
        episode_length=600, raw_goal_size=2, goal_encoding='vector',
        observation_dim=4, action_dim=2,
        native_goal_indices=_POINT_MAZE_NATIVE_GOAL_INDICES,
        backend_kwargs=dict(continuing_task=True, reset_target=False),
    ),
    'AntMaze_UMaze-v5': _spec(
        episode_length=700, raw_goal_size=2, goal_encoding='vector',
        observation_dim=105, action_dim=8,
        native_goal_indices=(),
        backend_kwargs=dict(
            continuing_task=True, reset_target=False,
            terminate_when_unhealthy=False,
            include_cfrc_ext_in_observation=True,
        ),
    ),
    'AntMaze_BigMaze_DGR-v5': _spec(
        episode_length=1000, raw_goal_size=2, goal_encoding='vector',
        observation_dim=105, action_dim=8,
        native_goal_indices=(),
        backend_ids=(
            'AntMaze_BigMaze_DGR-v5', 'AntMaze_Medium_Diverse_GR-v5',
        ),
        backend_kwargs=dict(
            continuing_task=True, reset_target=False,
            terminate_when_unhealthy=False,
            include_cfrc_ext_in_observation=True,
        ),
    ),
}

# Retained manipulation tasks have official Boolean and continuous 92-channel touch
# sensor variants. Their task goal is unchanged, so touch readings remain in
# the raw observation (the non-goal branch) and goal_dims match the parent task.
for _task_name, _task_specification in tuple(TASK_SPECS.items()):
    if not _task_name.startswith('HandManipulate'):
        continue
    _stem, _version = _task_name.rsplit('-', 1)
    for _touch_kind in ('BooleanTouchSensors', 'ContinuousTouchSensors'):
        _touch_name = f'{_stem}_{_touch_kind}-{_version}'
        TASK_SPECS[_touch_name] = dict(_task_specification)
        TASK_SPECS[_touch_name]['observation_dim'] += 92
        TASK_SPECS[_touch_name]['state_dim'] += 92

GOAL_DIMS = {
    task_name: tuple(task_spec['goal_dims'])
    for task_name, task_spec in TASK_SPECS.items()
}


def _make_backend(name: str):
    try:
        import gymnasium
        import gymnasium_robotics
    except ImportError as exc:
        raise ImportError(
            'Gymnasium-Robotics online environments require the optional '
            'packages `gymnasium` and `gymnasium-robotics`.'
        ) from exc

    # Gymnasium >= 1.0 uses this explicit plugin-registration hook. Older
    # Gymnasium-Robotics releases register on import, so no eager dependency is
    # needed merely to import this adapter module.
    register_envs = getattr(gymnasium, 'register_envs', None)
    if register_envs is not None:
        register_envs(gymnasium_robotics)
    elif hasattr(gymnasium_robotics, 'register_robotics_envs'):
        gymnasium_robotics.register_robotics_envs()

    spec = TASK_SPECS[name]
    backend_ids = spec['backend_ids'] or (name,)
    kwargs = dict(spec['backend_kwargs'])
    kwargs['max_episode_steps'] = spec['episode_length']
    failures = []
    for backend_id in backend_ids:
        try:
            return gymnasium.make(backend_id, **kwargs), backend_id
        except Exception as exc:
            failures.append(f'{backend_id}: {exc}')
    raise RuntimeError(
        f'Could not create Gymnasium-Robotics task {name!r}. Tried: '
        + '; '.join(failures)
    )


def _vector(value: Any, *, field: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError(f'{field} must be a vector, got shape {array.shape}')
    if not np.all(np.isfinite(array)):
        raise ValueError(f'{field} contains non-finite values')
    return array


def _canonical_quaternion(quaternion: np.ndarray) -> np.ndarray:
    """Normalize a MuJoCo wxyz quaternion and choose a deterministic sign."""
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape != (4,):
        raise ValueError(f'Expected a quaternion of shape (4,), got {quaternion.shape}')
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError('Goal quaternion must have finite, non-zero norm')
    quaternion = quaternion / norm
    for component in quaternion:
        if abs(component) > 1e-12:
            if component < 0:
                quaternion = -quaternion
            break
    return quaternion


def _quaternion_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Return a sign-invariant rotation matrix for a MuJoCo wxyz quaternion."""
    w, x, y, z = _canonical_quaternion(quaternion)
    return np.array([
        1 - 2 * (y * y + z * z),
        2 * (x * y - z * w),
        2 * (x * z + y * w),
        2 * (x * y + z * w),
        1 - 2 * (x * x + z * z),
        2 * (y * z - x * w),
        2 * (x * z - y * w),
        2 * (y * z + x * w),
        1 - 2 * (x * x + y * y),
    ], dtype=np.float32)


def _encode_goal(raw_goal: np.ndarray, encoding: str) -> np.ndarray:
    raw_goal = _vector(raw_goal, field='goal')
    if encoding == 'vector':
        return raw_goal.copy()

    if raw_goal.shape != (7,):
        raise ValueError(
            f'{encoding} requires a 7D position/quaternion goal, '
            f'got {raw_goal.shape}'
        )
    position = raw_goal[:3]
    rotation = _quaternion_matrix(raw_goal[3:]).reshape(3, 3)
    if encoding == 'rotation_matrix':
        return rotation.reshape(-1)
    if encoding == 'pose_rotation_matrix':
        return np.concatenate((position, rotation.reshape(-1))).astype(np.float32)

    raise ValueError(f'Unknown goal encoding: {encoding!r}')


def _raw_goal_values_repeated_in_observation(
        raw_goal: np.ndarray, encoding: str) -> np.ndarray:
    if encoding == 'rotation_matrix':
        return raw_goal[3:]
    return raw_goal


def _rotation_angle(matrix_a: np.ndarray, matrix_b: np.ndarray) -> float:
    relative_trace = float(np.trace(matrix_a.T @ matrix_b))
    cosine = np.clip((relative_trace - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cosine))


def _encoded_goal_distance(
        achieved: np.ndarray, desired: np.ndarray, encoding: str) -> float:
    if encoding == 'vector':
        return float(np.linalg.norm(achieved - desired))
    if encoding == 'rotation_matrix':
        return _rotation_angle(achieved.reshape(3, 3), desired.reshape(3, 3))
    if encoding == 'pose_rotation_matrix':
        position_distance = float(np.linalg.norm(achieved[:3] - desired[:3]))
        rotation_distance = _rotation_angle(
            achieved[3:].reshape(3, 3), desired[3:].reshape(3, 3),
        )
        return float(np.hypot(position_distance, rotation_distance))
    raise ValueError(f'Unknown goal encoding: {encoding!r}')


class GymnasiumRoboticsGoalEnv(gym.Env):
    """Adapt Gymnasium-Robotics GoalEnv tasks to QRL's online API.

    Native GoalEnv goals are shorter than observations. The adapter removes
    achieved-goal coordinates already present in the native observation, then
    prepends one task-equivalent encoding. This exposes stable coordinates at
    ``goal_dims`` without duplicating state. Rotation matrices avoid the q/-q
    ambiguity of quaternion coordinates.
    """

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, name: str, *, backend_env=None):
        if name not in TASK_SPECS:
            raise ValueError(f'Unknown Gymnasium-Robotics goal task: {name!r}')
        self.name = name
        self.task_spec = TASK_SPECS[name]
        self.goal_dims = tuple(self.task_spec['goal_dims'])
        self.episode_length = int(self.task_spec['episode_length'])
        if backend_env is None:
            self._env, self.backend_id = _make_backend(name)
        else:
            self._env, self.backend_id = backend_env, 'injected-test-backend'

        action_space = self._env.action_space
        if not hasattr(action_space, 'low') or not hasattr(action_space, 'high'):
            raise TypeError(
                'Gymnasium-Robotics online training requires a Box action space'
            )
        expected_action_shape = (int(self.task_spec['action_dim']),)
        if tuple(action_space.shape) != expected_action_shape:
            raise ValueError(
                f'{self.backend_id} expected action shape '
                f'{expected_action_shape}, got {action_space.shape}'
            )
        self.action_space = gym.spaces.Box(
            low=np.asarray(action_space.low, dtype=np.float32),
            high=np.asarray(action_space.high, dtype=np.float32),
            dtype=np.float32,
        )
        self.reward_mode = 'positive'
        self._pending_seed: Optional[int] = int(
            np.random.randint(0, 2 ** 31 - 1)
        )
        if hasattr(self.action_space, 'seed'):
            self.action_space.seed(self._pending_seed)

        raw_observation, _ = self._backend_reset()
        state, _, _, desired = self._state_and_goals(raw_observation)
        self.observation_space = vector_goal_observation_space(state.size)
        self._goal_values = desired
        self._elapsed_steps = 0

    @property
    def backend(self):
        return getattr(self._env, 'unwrapped', self._env)

    def _backend_reset(self, *, options=None):
        seed = self._pending_seed
        self._pending_seed = None
        kwargs = {}
        if seed is not None:
            kwargs['seed'] = seed
        if options is not None:
            kwargs['options'] = options
        try:
            result = self._env.reset(**kwargs)
        except TypeError:
            if seed is not None and hasattr(self._env, 'seed'):
                self._env.seed(seed)
            if options is not None:
                try:
                    result = self._env.reset(options=options)
                except TypeError:
                    result = self._env.reset()
            else:
                result = self._env.reset()
        return unpack_reset_result(result)

    def _state_and_goals(
            self, raw_observation: Any,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not isinstance(raw_observation, dict):
            raise TypeError(
                'Gymnasium-Robotics must return a GoalEnv observation dictionary'
            )
        missing = {
            'observation', 'achieved_goal', 'desired_goal',
        }.difference(raw_observation)
        if missing:
            raise KeyError(
                'Gymnasium-Robotics observation is missing fields: '
                f'{sorted(missing)}'
            )

        observation = _vector(raw_observation['observation'], field='observation')
        expected_observation_shape = (
            int(self.task_spec['observation_dim']),
        )
        if observation.shape != expected_observation_shape:
            raise ValueError(
                f'{self.name} expected native observation shape '
                f'{expected_observation_shape}, got {observation.shape}'
            )
        raw_achieved = _vector(
            raw_observation['achieved_goal'], field='achieved_goal',
        )
        raw_desired = _vector(
            raw_observation['desired_goal'], field='desired_goal',
        )
        expected_size = int(self.task_spec['raw_goal_size'])
        if raw_achieved.shape != (expected_size,):
            raise ValueError(
                f'{self.name} expected a raw achieved goal of size '
                f'{expected_size}, got shape {raw_achieved.shape}'
            )
        if raw_desired.shape != raw_achieved.shape:
            raise ValueError(
                f'{self.name} achieved and desired goal shapes differ: '
                f'{raw_achieved.shape} != {raw_desired.shape}'
            )

        encoding = self.task_spec['goal_encoding']
        achieved = _encode_goal(raw_achieved, encoding)
        desired = _encode_goal(raw_desired, encoding)
        native_goal_indices = tuple(self.task_spec['native_goal_indices'])
        if native_goal_indices:
            repeated_goal = _raw_goal_values_repeated_in_observation(
                raw_achieved, encoding,
            )
            native_values = observation[list(native_goal_indices)]
            if not np.allclose(
                    native_values, repeated_goal, rtol=1e-5, atol=1e-6):
                raise ValueError(
                    f'{self.name} native observation coordinates at '
                    f'{native_goal_indices} do not match achieved_goal'
                )
            observation = np.delete(observation, native_goal_indices)
        state = np.concatenate((achieved, observation)).astype(np.float32)
        if state.shape != (int(self.task_spec['state_dim']),):
            raise AssertionError(
                f'{self.name} constructed unexpected state shape {state.shape}'
            )
        return state, raw_achieved, achieved, desired

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
        state, _, _, self._goal_values = self._state_and_goals(raw_observation)
        return self._pack(state)

    def _is_success(
            self, raw_achieved: np.ndarray, raw_desired: np.ndarray,
            info: Dict[str, Any]) -> bool:
        for key in ('is_success', 'success'):
            if key in info:
                value = np.asarray(info[key])
                if value.size != 1:
                    raise ValueError(f'info[{key!r}] must be scalar')
                return bool(value.item())
        native_success = getattr(self.backend, '_is_success', None)
        if native_success is not None:
            value = np.asarray(native_success(raw_achieved, raw_desired))
            if value.size == 1:
                return bool(value.item())
        raise RuntimeError(
            f'{self.backend_id} did not expose a scalar success indicator'
        )

    def step(self, action):
        if self._elapsed_steps >= self.episode_length:
            raise RuntimeError('step() called after the fixed episode horizon')
        action_dtype = getattr(self._env.action_space, 'dtype', np.float32)
        result = self._env.step(np.asarray(action, dtype=action_dtype))
        raw_observation, backend_reward, terminated, truncated, info = (
            unpack_step_result(result)
        )
        self._elapsed_steps += 1
        state, raw_achieved, achieved, desired = self._state_and_goals(
            raw_observation
        )
        if not np.allclose(desired, self._goal_values, rtol=0, atol=1e-6):
            raise RuntimeError(
                f'{self.backend_id} changed the desired goal within an episode; '
                'fixed-goal QRL rollouts require reset_target=False'
            )

        is_success = self._is_success(raw_achieved, raw_observation['desired_goal'], info)
        if 'goal_distance' in info:
            value = np.asarray(info['goal_distance'])
            if value.size != 1:
                raise ValueError('info[\'goal_distance\'] must be scalar')
            distance = float(value.item())
        else:
            distance = _encoded_goal_distance(
                achieved, desired, self.task_spec['goal_encoding'],
            )

        if self.reward_mode == 'dense':
            reward = float(np.exp(-distance))
        elif self.reward_mode == 'negative':
            reward = float(is_success) - 1.0
        elif self.reward_mode == 'positive':
            reward = float(is_success)
        else:
            raise ValueError(f'Unknown reward mode: {self.reward_mode!r}')

        if terminated and self._elapsed_steps < self.episode_length:
            raise RuntimeError(
                f'{self.backend_id} terminated after {self._elapsed_steps} steps; '
                'the backend must be configured as a continuing task'
            )
        if truncated and self._elapsed_steps != self.episode_length:
            raise RuntimeError(
                f'{self.backend_id} truncated after {self._elapsed_steps} steps; '
                f'expected {self.episode_length}'
            )
        timeout = self._elapsed_steps == self.episode_length
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
            return self._env.render(*args, **kwargs)
        except TypeError:
            return self._env.render()

    def close(self):
        close = getattr(self._env, 'close', None)
        if close is not None:
            return close()
        return None


def create_env_from_spec(name: str):
    return GymnasiumRoboticsGoalEnv(name)


for task_name, task_spec in TASK_SPECS.items():
    register_online_env(
        'gymnasium_robotics',
        task_name,
        create_env_fn=lambda task_name=task_name: create_env_from_spec(task_name),
        episode_length=task_spec['episode_length'],
        goal_dims=task_spec['goal_dims'],
    )


__all__ = [
    'GOAL_DIMS',
    'GymnasiumRoboticsGoalEnv',
    'TASK_SPECS',
    'create_env_from_spec',
]

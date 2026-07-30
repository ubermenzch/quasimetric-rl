from __future__ import annotations

from typing import Optional

import logging

import gym
import numpy as np

from ..goal_env import (
    pack_goal_observation,
    unpack_reset_result,
    unpack_step_result,
    vector_goal_observation_space,
)
from ..memory import register_online_env


TASK_SPECS = {
    'Reacher-v4': dict(legacy_id='Reacher-v2', episode_length=50, goal_dims=(0, 1)),
    'Pusher-v4': dict(legacy_id='Pusher-v2', episode_length=100, goal_dims=(0, 1, 2)),
    'AntNavigate-v4': dict(legacy_id='Ant-v3', episode_length=1000, goal_dims=(0, 1)),
}


def _make_backend(name: str):
    spec = TASK_SPECS[name]
    modern_kwargs = {}
    legacy_kwargs = {}
    modern_id = name
    if name == 'AntNavigate-v4':
        modern_id = 'Ant-v4'
        modern_kwargs = dict(
            exclude_current_positions_from_observation=True,
            terminate_when_unhealthy=False,
        )
        legacy_kwargs = modern_kwargs

    try:
        import gymnasium
    except ImportError:
        gymnasium = None
    if gymnasium is not None:
        try:
            return gymnasium.make(modern_id, **modern_kwargs), modern_id
        except Exception as exc:
            logging.warning(
                'Could not create Gymnasium %s (%s); falling back to Gym %s',
                modern_id, exc, spec['legacy_id'],
            )

    return gym.make(spec['legacy_id'], **legacy_kwargs), spec['legacy_id']


class GymMujocoGoalEnv(gym.Env):
    """Goal-conditioned facade for modern or legacy Gym MuJoCo tasks."""

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, name: str, *, backend_env=None):
        if name not in TASK_SPECS:
            raise ValueError(f'Unknown Gym MuJoCo goal task: {name!r}')
        self.name = name
        self.episode_length = int(TASK_SPECS[name]['episode_length'])
        self.goal_dims = tuple(TASK_SPECS[name]['goal_dims'])
        if backend_env is None:
            self._env, self.backend_id = _make_backend(name)
        else:
            self._env, self.backend_id = backend_env, 'injected-test-backend'

        self.action_space = gym.spaces.Box(
            low=np.asarray(self._env.action_space.low, dtype=np.float32),
            high=np.asarray(self._env.action_space.high, dtype=np.float32),
            dtype=np.float32,
        )
        self.reward_mode = 'positive'
        initial_seed = int(np.random.randint(0, 2 ** 31 - 1))
        self._pending_seed: Optional[int] = initial_seed
        self._goal_rng = np.random.RandomState(initial_seed)
        if hasattr(self.action_space, 'seed'):
            self.action_space.seed(initial_seed)
        self._elapsed_steps = 0

        raw_observation, _ = self._backend_reset()
        state = self._state(raw_observation)
        self.observation_space = vector_goal_observation_space(state.size)
        self._goal_values = self._sample_goal()

    @property
    def backend(self):
        return self._env.unwrapped

    def _backend_reset(self):
        if self._pending_seed is not None:
            seed = self._pending_seed
            self._pending_seed = None
            try:
                result = self._env.reset(seed=seed)
            except TypeError:
                if hasattr(self._env, 'seed'):
                    self._env.seed(seed)
                result = self._env.reset()
        else:
            result = self._env.reset()
        return unpack_reset_result(result)

    def _data(self):
        backend = self.backend
        if hasattr(backend, 'data'):
            return backend.data
        return backend.sim.data

    def _body_com(self, name: str) -> np.ndarray:
        return np.asarray(self.backend.get_body_com(name), dtype=np.float64)

    def _state(self, raw_observation) -> np.ndarray:
        data = self._data()
        if self.name == 'Reacher-v4':
            theta = np.asarray(data.qpos[:2])
            return np.concatenate([
                self._body_com('fingertip')[:2],
                np.cos(theta),
                np.sin(theta),
                np.asarray(data.qvel[:2]),
            ]).astype(np.float32)
        if self.name == 'Pusher-v4':
            return np.concatenate([
                self._body_com('object'),
                np.asarray(data.qpos[:7]),
                np.asarray(data.qvel[:7]),
                self._body_com('tips_arm'),
            ]).astype(np.float32)
        if self.name == 'AntNavigate-v4':
            # Gym Ant may append 84 cfrc_ext values that this backend never computes.
            return np.concatenate([
                np.asarray(data.qpos),
                np.asarray(data.qvel),
            ]).astype(np.float32)
        raise AssertionError(self.name)

    def _sample_goal(self) -> np.ndarray:
        if self.name == 'Reacher-v4':
            return self._body_com('target')[:2].astype(np.float32)
        if self.name == 'Pusher-v4':
            goal = self._body_com('goal').copy()
            # The visual marker lies on the table, while the task goal is the
            # center of the object when it is positioned over that marker.
            goal[2] = self._body_com('object')[2]
            return goal.astype(np.float32)
        if self.name == 'AntNavigate-v4':
            angle = self._goal_rng.uniform(-np.pi, np.pi)
            radius = self._goal_rng.uniform(1.0, 5.0)
            origin = np.asarray(self._data().qpos[:2])
            return (origin + radius * np.array([
                np.cos(angle), np.sin(angle),
            ])).astype(np.float32)
        raise AssertionError(self.name)

    def _success_radius(self) -> float:
        return 0.5 if self.name == 'AntNavigate-v4' else 0.05

    def _pack(self, state: np.ndarray):
        return pack_goal_observation(state, self._goal_values, self.goal_dims)

    def seed(self, seed: Optional[int] = None):
        if seed is None:
            seed = int(np.random.randint(0, 2 ** 31 - 1))
        self._pending_seed = int(seed)
        self._goal_rng.seed(int(seed))
        if hasattr(self.action_space, 'seed'):
            self.action_space.seed(int(seed))
        return [int(seed)]

    def reset(self):
        raw_observation, _ = self._backend_reset()
        self._elapsed_steps = 0
        self._goal_values = self._sample_goal()
        return self._pack(self._state(raw_observation))

    def step(self, action):
        result = self._env.step(np.asarray(action, dtype=self._env.action_space.dtype))
        raw_observation, _, terminated, truncated, info = unpack_step_result(result)
        self._elapsed_steps += 1
        if terminated and self._elapsed_steps < self.episode_length:
            raise RuntimeError(
                f'{self.backend_id} terminated after {self._elapsed_steps} steps; '
                'QRL online environments must use a fixed horizon'
            )

        state = self._state(raw_observation)
        distance = float(np.linalg.norm(
            state[list(self.goal_dims)] - self._goal_values
        ))
        is_success = distance < self._success_radius()
        if self.reward_mode == 'dense':
            reward = float(np.exp(-distance / self._success_radius() * np.log(2)))
        elif self.reward_mode == 'negative':
            reward = float(is_success) - 1.0
        else:
            reward = float(is_success)

        timeout = truncated or self._elapsed_steps == self.episode_length
        info.update(
            is_success=bool(is_success),
            goal_distance=distance,
            backend_id=self.backend_id,
        )
        if timeout:
            info['TimeLimit.truncated'] = True
        return self._pack(state), reward, False, info

    def close(self):
        return self._env.close()


def create_env_from_spec(name: str):
    return GymMujocoGoalEnv(name)


for task_name, task_spec in TASK_SPECS.items():
    register_online_env(
        'gym_mujoco', task_name,
        create_env_fn=lambda task_name=task_name: create_env_from_spec(task_name),
        episode_length=task_spec['episode_length'],
    )


__all__ = ['GymMujocoGoalEnv', 'TASK_SPECS', 'create_env_from_spec']

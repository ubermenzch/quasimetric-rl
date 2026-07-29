from __future__ import annotations

from typing import Optional

import gym
import numpy as np

from ..goal_env import pack_goal_observation, vector_goal_observation_space
from ..memory import register_online_env


TASK_SPECS = {
    'maze2d-medium': dict(backend_id='maze2d-medium-v1', episode_length=600),
    'maze2d-large': dict(backend_id='maze2d-large-v1', episode_length=800),
}

GOAL_DIMS = (0, 1)
SUCCESS_RADIUS = 0.5


def _make_backend(backend_id: str):
    import d4rl  # noqa: F401  # Register the legacy Maze2D environments.

    return gym.make(backend_id).unwrapped


class OnlineMazeGoalEnv(gym.Env):
    """Expose D4RL Maze2D as a fixed-horizon online goal environment."""

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, name: str, *, backend_env=None):
        if name not in TASK_SPECS:
            raise ValueError(f'Unknown online Maze2D task: {name!r}')
        self.name = name
        spec = TASK_SPECS[name]
        self.backend_id = spec['backend_id']
        self.episode_length = int(spec['episode_length'])
        self.goal_dims = GOAL_DIMS
        self._env = (
            _make_backend(self.backend_id)
            if backend_env is None
            else backend_env
        )
        self.action_space = gym.spaces.Box(
            low=np.asarray(self._env.action_space.low, dtype=np.float32),
            high=np.asarray(self._env.action_space.high, dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = vector_goal_observation_space(4)
        initial_seed = int(np.random.randint(0, 2 ** 31 - 1))
        self._pending_seed: Optional[int] = initial_seed
        if hasattr(self.action_space, 'seed'):
            self.action_space.seed(initial_seed)
        self._elapsed_steps = 0
        self._goal_values = np.zeros(2, dtype=np.float32)

    def seed(self, seed: Optional[int] = None):
        if seed is None:
            seed = int(np.random.randint(0, 2 ** 31 - 1))
        self._pending_seed = int(seed)
        if hasattr(self.action_space, 'seed'):
            self.action_space.seed(int(seed))
        return [int(seed)]

    def _state(self, raw_observation) -> np.ndarray:
        state = np.asarray(raw_observation, dtype=np.float32).reshape(-1)
        if state.shape != (4,):
            raise RuntimeError(
                f'{self.backend_id} returned state shape {state.shape}, expected (4,)'
            )
        return state

    def _sample_nontrivial_goal(self, state: np.ndarray) -> np.ndarray:
        for _ in range(100):
            self._env.set_target()
            goal = np.asarray(self._env.get_target(), dtype=np.float32)
            if np.linalg.norm(state[list(self.goal_dims)] - goal) > SUCCESS_RADIUS:
                if hasattr(self._env, 'set_marker'):
                    self._env.set_marker()
                return goal
        raise RuntimeError(f'{self.backend_id} could not sample a nontrivial goal')

    def _pack(self, state: np.ndarray):
        return pack_goal_observation(state, self._goal_values, self.goal_dims)

    def reset(self):
        if self._pending_seed is not None:
            if hasattr(self._env, 'seed'):
                self._env.seed(self._pending_seed)
            self._pending_seed = None
        state = self._state(self._env.reset())
        self._goal_values = self._sample_nontrivial_goal(state)
        self._elapsed_steps = 0
        return self._pack(state)

    def step(self, action):
        raw_observation, _, _, backend_info = self._env.step(
            np.asarray(action, dtype=self._env.action_space.dtype)
        )
        self._elapsed_steps += 1
        state = self._state(raw_observation)
        distance = float(np.linalg.norm(
            state[list(self.goal_dims)] - self._goal_values
        ))
        is_success = distance <= SUCCESS_RADIUS
        timeout = self._elapsed_steps == self.episode_length
        info = dict(backend_info)
        info.update(is_success=bool(is_success), goal_distance=distance)
        if timeout:
            info['TimeLimit.truncated'] = True
        return self._pack(state), float(is_success), False, info

    def close(self):
        return self._env.close()


def create_env_from_spec(name: str):
    return OnlineMazeGoalEnv(name)


for task_name, task_spec in TASK_SPECS.items():
    register_online_env(
        'online_maze', task_name,
        create_env_fn=lambda task_name=task_name: create_env_from_spec(task_name),
        episode_length=task_spec['episode_length'],
    )


__all__ = ['OnlineMazeGoalEnv', 'TASK_SPECS', 'create_env_from_spec']

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import gym
import numpy as np


def vector_goal_observation_space(size: int) -> gym.spaces.Dict:
    """Create the same-shaped state/achieved-goal/desired-goal space QRL uses."""
    state_space = gym.spaces.Box(
        low=np.full(size, -np.inf, dtype=np.float32),
        high=np.full(size, np.inf, dtype=np.float32),
        dtype=np.float32,
    )
    return gym.spaces.Dict({
        'observation': state_space,
        'achieved_goal': state_space,
        'desired_goal': state_space,
    })


def pack_goal_observation(
        state: np.ndarray, goal_values: np.ndarray,
        goal_dims: Sequence[int]) -> Dict[str, np.ndarray]:
    state = np.asarray(state, dtype=np.float32)
    goal_values = np.asarray(goal_values, dtype=np.float32)
    goal_dims = tuple(goal_dims)
    if goal_values.shape != (len(goal_dims),):
        raise ValueError(
            f'Expected {len(goal_dims)} goal values, got shape {goal_values.shape}'
        )
    desired_goal = np.zeros_like(state)
    desired_goal[list(goal_dims)] = goal_values
    return {
        'observation': state,
        'achieved_goal': state.copy(),
        'desired_goal': desired_goal,
    }


def unpack_reset_result(result: Any) -> Tuple[Any, Dict[str, Any]]:
    """Normalize Gym and Gymnasium reset results without importing Gymnasium."""
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        return result
    return result, {}


def unpack_step_result(result: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
    """Normalize old Gym and Gymnasium step results."""
    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
        return observation, float(reward), bool(terminated), bool(truncated), dict(info)
    if len(result) == 4:
        observation, reward, done, info = result
        info = dict(info)
        truncated = bool(info.get('TimeLimit.truncated', False))
        return observation, float(reward), bool(done and not truncated), truncated, info
    raise ValueError(f'Expected a 4- or 5-element step result, got {len(result)}')


__all__ = [
    'pack_goal_observation',
    'unpack_reset_result',
    'unpack_step_result',
    'vector_goal_observation_space',
]

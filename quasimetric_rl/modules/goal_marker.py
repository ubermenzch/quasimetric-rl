"""Goal-set markers for a QRL Appendix A inspired online baseline.

Each desired coordinate vector identifies a separate set marker.  Real states
have indicator zero; markers retain the desired coordinates and have indicator
one.  Only observed successful transitions create synthetic local edges.
"""

from typing import Tuple

import attrs
import gym.spaces
import numpy as np
import torch

from ..data import BatchData, EnvSpec


def append_real_indicator(state: torch.Tensor) -> torch.Tensor:
    return torch.cat((state, state.new_zeros(*state.shape[:-1], 1)), dim=-1)


def make_goal_marker(augmented: torch.Tensor, goal_dims: Tuple[int, ...]) -> torch.Tensor:
    marker = torch.zeros_like(augmented)
    marker[..., list(goal_dims)] = augmented[..., list(goal_dims)]
    marker[..., -1] = 1
    return marker


def augmented_env_spec(spec: EnvSpec) -> EnvSpec:
    space = spec.observation_space
    if not isinstance(space, gym.spaces.Box) or len(space.shape) != 1:
        raise ValueError('QRL goal markers require vector Box observations')
    low = np.concatenate((space.low, np.zeros(1, dtype=space.dtype)))
    high = np.concatenate((space.high, np.ones(1, dtype=space.dtype)))
    return attrs.evolve(spec, observation_space=gym.spaces.Box(low=low, high=high, dtype=space.dtype))


def augment_replay_batch(data: BatchData, replay, goal_dims: Tuple[int, ...]) -> BatchData:
    if data.replay_episode_indices is None or data.replay_timesteps is None:
        raise ValueError('QRL goal markers require replay episode and timestep indices')
    if data.history_observations is not None:
        raise ValueError('QRL goal markers do not support transition-history models')
    episodes = data.replay_episode_indices.long()
    timesteps = data.replay_timesteps.long()
    length = replay.episode_length
    transition_indices = episodes * length + timesteps
    observation_indices = episodes * (length + 1) + timesteps + 1
    successful = replay.raw_data.transition_infos['is_success'][transition_indices].bool()
    desired = replay.raw_data.observation_infos['desired_goals'][observation_indices]
    sources = append_real_indicator(data.next_observations[successful])
    targets = make_goal_marker(append_real_indicator(desired[successful]), goal_dims)
    return attrs.evolve(
        data,
        observations=append_real_indicator(data.observations),
        next_observations=append_real_indicator(data.next_observations),
        future_observations=append_real_indicator(data.future_observations),
        goal_marker_sources=sources,
        goal_marker_targets=targets,
    )

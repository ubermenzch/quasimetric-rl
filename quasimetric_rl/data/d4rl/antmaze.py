from __future__ import annotations
from typing import *

import functools

import numpy as np

from ..base import EpisodeData, register_offline_env
from . import load_environment


ANTMAZE_NAMES = (
    'antmaze-umaze-v2',
    'antmaze-umaze-diverse-v2',
    'antmaze-medium-play-v2',
    'antmaze-medium-diverse-v2',
    'antmaze-large-play-v2',
    'antmaze-large-diverse-v2',
)


def episode_slices(dataset: Mapping[str, np.ndarray], max_episode_steps: int) -> Iterator[slice]:
    length = int(np.asarray(dataset['rewards']).shape[0])
    timeouts = np.asarray(dataset.get('timeouts', np.zeros(length, dtype=np.bool_)), dtype=np.bool_)
    use_timeouts = 'timeouts' in dataset
    start = 0
    episode_step = 0
    for idx in range(length):
        if use_timeouts:
            final_timestep = bool(timeouts[idx])
        else:
            final_timestep = episode_step == max_episode_steps - 1
        # AntMaze terminals mark successful states, but the recorded trajectory
        # continues. Only time limits (or the dataset end) delimit episodes.
        if final_timestep or idx == length - 1:
            if idx + 1 > start:
                yield slice(start, idx + 1)
            start = idx + 1
            episode_step = 0
        else:
            episode_step += 1


def load_episodes_antmaze(name: str) -> Iterator[EpisodeData]:
    env = load_environment(name)
    dataset = env.get_dataset()
    observations = np.asarray(dataset['observations'], dtype=np.float32)
    actions = np.asarray(dataset['actions'], dtype=np.float32)
    rewards = np.asarray(dataset['rewards'], dtype=np.float32)
    terminals = np.asarray(dataset.get('terminals', np.zeros_like(rewards, dtype=np.bool_)), dtype=np.bool_)
    timeouts = np.asarray(dataset.get('timeouts', np.zeros_like(rewards, dtype=np.bool_)), dtype=np.bool_)

    for slc in episode_slices(dataset, env.max_episode_steps):
        # Raw D4RL AntMaze files contain observations but no next_observations.
        # Adjacent rows within one episode form valid transitions; the final row
        # has no successor and must not be paired with the next episode's reset.
        if slc.stop - slc.start < 2:
            continue
        transition_slc = slice(slc.start, slc.stop - 1)
        next_observation_slc = slice(slc.start + 1, slc.stop)
        yield EpisodeData.from_simple_trajectory(
            observations=observations[transition_slc],
            actions=actions[transition_slc],
            next_observations=observations[next_observation_slc],
            rewards=rewards[transition_slc],
            terminals=terminals[transition_slc],
            timeouts=timeouts[transition_slc],
        )


for antmaze_name in ANTMAZE_NAMES:
    register_offline_env(
        'd4rl', antmaze_name,
        create_env_fn=functools.partial(load_environment, antmaze_name),
        load_episodes_fn=functools.partial(load_episodes_antmaze, antmaze_name),
    )

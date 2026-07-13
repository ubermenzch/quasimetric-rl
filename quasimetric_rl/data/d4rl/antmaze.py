from __future__ import annotations
from typing import *

import functools

import gym
import gym.spaces
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


class AntMazeGoalObsWrapper(gym.ObservationWrapper):
    """Expose AntMaze observations as [ant_state, goal_state]."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        observation_dim = int(np.prod(env.observation_space.shape))
        self.observation_space = gym.spaces.Box(
            low=-np.inf * np.ones(observation_dim * 2, dtype=np.float32),
            high=np.inf * np.ones(observation_dim * 2, dtype=np.float32),
            dtype=np.float32,
        )

    def observation(self, observation):
        goal_observation = np.zeros_like(observation, dtype=np.float32)
        goal_observation[:2] = np.asarray(self.env.target_goal, dtype=np.float32)
        return np.concatenate([
            np.asarray(observation, dtype=np.float32),
            goal_observation,
        ]).astype(np.float32, copy=False)

    @property
    def max_episode_steps(self):
        return self.env.max_episode_steps

    @property
    def name(self):
        return getattr(self.env, 'name', None)

    @name.setter
    def name(self, value):
        self.env.name = value

    def get_dataset(self):
        return self.env.get_dataset()

    def get_normalized_score(self, score):
        return self.env.get_normalized_score(score)


def make_antmaze_observation(observation: np.ndarray, goal_xy: np.ndarray) -> np.ndarray:
    observation = np.asarray(observation, dtype=np.float32)
    goal_xy = np.asarray(goal_xy, dtype=np.float32)
    goal_observation = np.zeros_like(observation, dtype=np.float32)
    goal_observation[..., :goal_xy.shape[-1]] = goal_xy
    return np.concatenate([observation, goal_observation], axis=-1).astype(np.float32, copy=False)


def load_antmaze_environment(name: str) -> AntMazeGoalObsWrapper:
    env = load_environment(name)
    return AntMazeGoalObsWrapper(env)


def with_next_observations(dataset: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    dataset = dict(dataset)
    if 'next_observations' in dataset:
        return dataset
    observations = np.asarray(dataset['observations'])
    next_observations = np.concatenate([observations[1:], observations[-1:]], axis=0)
    dataset['next_observations'] = next_observations
    return dataset


def episode_slices(dataset: Mapping[str, np.ndarray], max_episode_steps: int) -> Iterator[slice]:
    length = int(np.asarray(dataset['rewards']).shape[0])
    terminals = np.asarray(dataset.get('terminals', np.zeros(length, dtype=np.bool_)), dtype=np.bool_)
    timeouts = np.asarray(dataset.get('timeouts', np.zeros(length, dtype=np.bool_)), dtype=np.bool_)
    use_timeouts = 'timeouts' in dataset
    start = 0
    episode_step = 0
    for idx in range(length):
        if use_timeouts:
            final_timestep = bool(timeouts[idx])
        else:
            final_timestep = episode_step == max_episode_steps - 1
        if bool(terminals[idx]) or final_timestep or idx == length - 1:
            if idx + 1 > start:
                yield slice(start, idx + 1)
            start = idx + 1
            episode_step = 0
        else:
            episode_step += 1


def load_episodes_antmaze(name: str) -> Iterator[EpisodeData]:
    env = load_environment(name)
    dataset = with_next_observations(env.get_dataset())
    observations = np.asarray(dataset['observations'], dtype=np.float32)
    next_observations = np.asarray(dataset['next_observations'], dtype=np.float32)
    actions = np.asarray(dataset['actions'], dtype=np.float32)
    rewards = np.asarray(dataset['rewards'], dtype=np.float32)
    terminals = np.asarray(dataset.get('terminals', np.zeros_like(rewards, dtype=np.bool_)), dtype=np.bool_)
    timeouts = np.asarray(dataset.get('timeouts', np.zeros_like(rewards, dtype=np.bool_)), dtype=np.bool_)

    if 'infos/goal' not in dataset:
        raise KeyError(f"{name} dataset is missing 'infos/goal'")
    goals = np.asarray(dataset['infos/goal'], dtype=np.float32)

    for slc in episode_slices(dataset, env.max_episode_steps):
        yield EpisodeData.from_simple_trajectory(
            observations=make_antmaze_observation(observations[slc], goals[slc]),
            actions=actions[slc],
            next_observations=make_antmaze_observation(next_observations[slc], goals[slc]),
            rewards=rewards[slc],
            terminals=terminals[slc],
            timeouts=timeouts[slc],
        )


for antmaze_name in ANTMAZE_NAMES:
    register_offline_env(
        'd4rl', antmaze_name,
        create_env_fn=functools.partial(load_antmaze_environment, antmaze_name),
        load_episodes_fn=functools.partial(load_episodes_antmaze, antmaze_name),
    )

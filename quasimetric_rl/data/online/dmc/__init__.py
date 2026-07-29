from __future__ import annotations

from typing import Optional

import gym
import numpy as np

from ..goal_env import pack_goal_observation, vector_goal_observation_space
from ..memory import register_online_env


TASK_SPECS = {
    'reacher_easy': dict(domain='reacher', task='easy', goal_dims=(0, 1)),
    'reacher_hard': dict(domain='reacher', task='hard', goal_dims=(0, 1)),
    'swimmer6': dict(domain='swimmer', task='swimmer6', goal_dims=(0, 1)),
    'swimmer15': dict(domain='swimmer', task='swimmer15', goal_dims=(0, 1)),
    'quadruped_fetch': dict(
        domain='quadruped', task='fetch', goal_dims=(0, 1),
    ),
    'manipulator_bring_ball': dict(
        domain='manipulator', task='bring_ball', goal_dims=(0, 1),
    ),
    'manipulator_bring_peg': dict(
        domain='manipulator', task='bring_peg', goal_dims=(0, 1, 2, 3),
    ),
}

DMC_EPISODE_LENGTH = 1000


class DMCGoalEnv(gym.Env):
    """Expose selected dm_control tasks through QRL's goal-conditioned API."""

    metadata = {'render.modes': ['rgb_array']}

    def __init__(self, name: str, *, environment_factory=None):
        if name not in TASK_SPECS:
            raise ValueError(f'Unknown dm_control goal task: {name!r}')
        self.name = name
        self.task_spec = TASK_SPECS[name]
        self.goal_dims = tuple(self.task_spec['goal_dims'])
        self.episode_length = DMC_EPISODE_LENGTH
        self._environment_factory = environment_factory
        self._pending_seed: Optional[int] = None
        self._seed = int(np.random.randint(0, 2 ** 31 - 1))
        self._make_environment(self._seed)

    def _make_environment(self, seed: int):
        if self._environment_factory is None:
            from dm_control import suite
            self._env = suite.load(
                self.task_spec['domain'], self.task_spec['task'],
                task_kwargs={'random': seed},
            )
        else:
            self._env = self._environment_factory(seed)

        action_spec = self._env.action_spec()
        self.action_space = gym.spaces.Box(
            low=np.asarray(action_spec.minimum, dtype=np.float32),
            high=np.asarray(action_spec.maximum, dtype=np.float32),
            dtype=np.float32,
        )
        if hasattr(self.action_space, 'seed'):
            self.action_space.seed(seed)
        timestep = self._env.reset()
        state, goal_values = self._state_and_goal(timestep.observation)
        self.observation_space = vector_goal_observation_space(state.size)
        self._goal_values = goal_values
        self._elapsed_steps = 0
        return timestep

    def _state_and_goal(self, observation):
        domain = self.task_spec['domain']
        if domain == 'reacher':
            target = np.asarray(
                self._env.physics.named.data.geom_xpos['target', :2],
                dtype=np.float32,
            )
            finger = np.asarray(
                self._env.physics.named.data.geom_xpos['finger', :2],
                dtype=np.float32,
            )
            state = np.concatenate([
                finger,
                np.asarray(observation['position']).reshape(-1),
                np.asarray(observation['velocity']).reshape(-1),
            ])
            return state.astype(np.float32), target

        if domain == 'swimmer':
            physics = self._env.physics
            data = physics.data
            target = np.asarray(
                physics.named.data.geom_xpos['target', :2], dtype=np.float32,
            )
            nose = np.asarray(
                physics.named.data.geom_xpos['nose', :2], dtype=np.float32,
            )
            root_angle = float(data.qpos[2])
            state = np.concatenate([
                nose,
                np.array([np.cos(root_angle), np.sin(root_angle)]),
                np.asarray(data.qpos[3:]),
                np.asarray(data.qvel),
            ])
            return state.astype(np.float32), target

        if domain == 'quadruped':
            physics = self._env.physics
            target = np.asarray(
                physics.named.data.site_xpos['target', :2], dtype=np.float32,
            )
            ball = np.asarray(
                physics.named.data.xpos['ball', :2], dtype=np.float32,
            )
            state = np.concatenate([
                ball,
                np.asarray(physics.data.qpos),
                np.asarray(physics.data.qvel),
            ])
            return state.astype(np.float32), target

        object_pose = np.asarray(observation['object_pos']).reshape(-1)
        target_pose = np.asarray(observation['target_pos']).reshape(-1)
        state_parts = [object_pose]
        for key in (
                'arm_pos', 'arm_vel', 'touch', 'hand_pos', 'object_vel'):
            state_parts.append(np.asarray(observation[key]).reshape(-1))
        return (
            np.concatenate(state_parts).astype(np.float32),
            target_pose[list(self.goal_dims)].astype(np.float32),
        )

    def _pack(self, state):
        return pack_goal_observation(state, self._goal_values, self.goal_dims)

    def seed(self, seed: Optional[int] = None):
        if seed is None:
            seed = int(np.random.randint(0, 2 ** 31 - 1))
        self._pending_seed = int(seed)
        if hasattr(self.action_space, 'seed'):
            self.action_space.seed(int(seed))
        return [int(seed)]

    def reset(self):
        if self._pending_seed is not None:
            self._seed = self._pending_seed
            self._pending_seed = None
            timestep = self._make_environment(self._seed)
        else:
            timestep = self._env.reset()
        self._elapsed_steps = 0
        state, self._goal_values = self._state_and_goal(timestep.observation)
        return self._pack(state)

    def step(self, action):
        action_spec = self._env.action_spec()
        timestep = self._env.step(np.asarray(action, dtype=action_spec.dtype))
        self._elapsed_steps += 1
        state, goal_values = self._state_and_goal(timestep.observation)
        if not np.allclose(goal_values, self._goal_values, rtol=0, atol=1e-6):
            raise RuntimeError('dm_control changed the desired goal within an episode')
        reward = 0.0 if timestep.reward is None else float(timestep.reward)
        distance = float(np.linalg.norm(
            state[list(self.goal_dims)] - self._goal_values
        ))
        domain = self.task_spec['domain']
        if domain == 'swimmer':
            success_radius = float(
                self._env.physics.named.model.geom_size['target', 0]
            )
            is_success = distance <= success_radius
        elif domain == 'quadruped':
            success_radius = float(
                self._env.physics.named.model.site_size['target', 0]
            )
            is_success = distance <= success_radius
        else:
            is_success = reward >= 1.0 - 1e-7
        timeout = bool(timestep.last()) or self._elapsed_steps == self.episode_length
        if timeout and self._elapsed_steps != self.episode_length:
            raise RuntimeError(
                f'dm_control {self.name} ended after {self._elapsed_steps} steps; '
                f'expected {self.episode_length}'
            )
        info = {
            'is_success': bool(is_success),
            'goal_distance': distance,
        }
        if timeout:
            info['TimeLimit.truncated'] = True
        return self._pack(state), reward, False, info

    def close(self):
        close = getattr(self._env, 'close', None)
        if close is not None:
            close()


def create_env_from_spec(name: str):
    return DMCGoalEnv(name)


for task_name in TASK_SPECS:
    register_online_env(
        'dmc', task_name,
        create_env_fn=lambda task_name=task_name: create_env_from_spec(task_name),
        episode_length=DMC_EPISODE_LENGTH,
    )


__all__ = ['DMCGoalEnv', 'TASK_SPECS', 'create_env_from_spec']

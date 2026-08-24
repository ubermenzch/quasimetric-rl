from __future__ import annotations

from typing import Optional

import gym
import numpy as np

from ..goal_env import pack_goal_observation, vector_goal_observation_space
from ..memory import register_online_env


TASK_SPECS = {
    'reacher_easy': dict(domain='reacher', task='easy', goal_dims=(0, 1)),
    'reacher_hard': dict(domain='reacher', task='hard', goal_dims=(0, 1)),
    'point_mass_easy': dict(
        domain='point_mass', task='easy', goal_dims=(0, 1),
        state_dim=4, action_dim=2,
    ),
    'finger_turn_easy': dict(
        domain='finger', task='turn_easy', goal_dims=(0, 1),
        state_dim=9, action_dim=2,
    ),
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
    'manipulator_insert_ball': dict(
        domain='manipulator', task='insert_ball', goal_dims=(0, 1),
        state_dim=40, action_dim=5,
    ),
    'manipulator_insert_peg': dict(
        domain='manipulator', task='insert_peg', goal_dims=(0, 1, 2, 3),
        state_dim=40, action_dim=5,
    ),
    'dog_fetch': dict(
        domain='dog', task='fetch', goal_dims=(0, 1, 2),
        state_dim=210, action_dim=38,
    ),
    'stacker_stack_2': dict(
        domain='stacker', task='stack_2', goal_dims=(0, 1), n_boxes=2,
        state_dim=49, action_dim=5,
    ),
    'ball_in_cup_catch': dict(
        domain='ball_in_cup', task='catch', goal_dims=(0, 1),
        state_dim=8, action_dim=2,
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
        expected_state_dim = self.task_spec.get('state_dim')
        expected_action_dim = self.task_spec.get('action_dim')
        if expected_state_dim is not None and state.shape != (expected_state_dim,):
            raise RuntimeError(
                f'{self.name} state contract changed: expected '
                f'{expected_state_dim}, got {state.shape}'
            )
        if (
                expected_action_dim is not None
                and self.action_space.shape != (expected_action_dim,)):
            raise RuntimeError(
                f'{self.name} action contract changed: expected '
                f'{expected_action_dim}, got {self.action_space.shape}'
            )
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

        if domain == 'point_mass':
            target = np.asarray(
                self._env.physics.named.data.geom_xpos['target', :2],
                dtype=np.float32,
            )
            state = np.concatenate([
                np.asarray(observation['position']).reshape(-1),
                np.asarray(observation['velocity']).reshape(-1),
            ])
            return state.astype(np.float32), target

        if domain == 'finger':
            position = np.asarray(observation['position']).reshape(-1)
            # bounded_position stores the spinner tip (x, z) last. Move it to
            # the front so the task coordinates agree with goal_dims.
            state = np.concatenate([
                position[-2:],
                position[:-2],
                np.asarray(observation['velocity']).reshape(-1),
                np.asarray(observation['touch']).reshape(-1),
            ])
            target = np.asarray(
                observation['target_position'], dtype=np.float32,
            ).reshape(-1)
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

        if domain == 'manipulator':
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

        if domain == 'dog':
            physics = self._env.physics
            ball = np.asarray(
                physics.named.data.geom_xpos['ball'], dtype=np.float32,
            ).reshape(-1)
            target = np.asarray(
                physics.named.data.geom_xpos['target'], dtype=np.float32,
            ).reshape(-1)
            qpos = np.asarray(physics.data.qpos).reshape(-1)
            ball_qpos_address = int(
                physics.named.model.jnt_qposadr['ball_root']
            )
            if not 0 <= ball_qpos_address <= qpos.size - 7:
                raise RuntimeError(
                    'dog_fetch ball_root must address a seven-coordinate '
                    f'free joint, got qpos address {ball_qpos_address} for '
                    f'{qpos.size} coordinates'
                )
            ball_qpos = qpos[ball_qpos_address:ball_qpos_address + 3]
            if not np.allclose(ball_qpos, ball, rtol=0, atol=1e-6):
                raise RuntimeError(
                    'dog_fetch ball_root qpos XYZ disagrees with ball geom '
                    f'position: {ball_qpos} != {ball}'
                )
            qpos_without_ball_xyz = np.concatenate([
                qpos[:ball_qpos_address],
                qpos[ball_qpos_address + 3:],
            ])
            # The goal prefix already contains the ball translation. Remove
            # that duplicate from qpos while retaining its quaternion, all
            # other qpos, and the activation dynamics needed for Markov state.
            state = np.concatenate([
                ball,
                qpos_without_ball_xyz,
                np.asarray(physics.data.qvel).reshape(-1),
                np.asarray(physics.data.act).reshape(-1),
            ])
            return state.astype(np.float32), target

        if domain == 'stacker':
            boxes = np.asarray(observation['box_pos']).reshape(
                self.task_spec['n_boxes'], -1,
            )
            target = np.asarray(observation['target_pos']).reshape(-1)
            closest = int(np.argmin(np.linalg.norm(
                boxes[:, :2] - target[None, :], axis=1,
            )))
            # The native task accepts any box at the target. The nearest box
            # supplies the achieved goal while all box states remain visible.
            state_parts = [boxes[closest, :2]]
            for key in (
                    'arm_pos', 'arm_vel', 'touch', 'hand_pos',
                    'box_pos', 'box_vel'):
                state_parts.append(np.asarray(observation[key]).reshape(-1))
            return (
                np.concatenate(state_parts).astype(np.float32),
                target.astype(np.float32),
            )

        if domain == 'ball_in_cup':
            physics = self._env.physics
            position = np.asarray(observation['position']).reshape(-1)
            ball_from_target = -np.asarray(
                physics.ball_to_target(), dtype=np.float32,
            ).reshape(-1)
            # The cup target moves with the controlled cup. Expressing the
            # task in the cup frame gives a fixed zero goal. Cup position and
            # all velocities retain a Markov state without duplicating ball
            # position coordinates.
            state = np.concatenate([
                ball_from_target,
                position[:2],
                np.asarray(observation['velocity']).reshape(-1),
            ])
            return state.astype(np.float32), np.zeros(2, dtype=np.float32)

        raise RuntimeError(f'Unsupported dm_control domain: {domain!r}')

    def _is_success(self, reward: float, distance: float) -> bool:
        domain = self.task_spec['domain']
        physics = self._env.physics
        if domain == 'point_mass':
            radius = float(physics.named.model.geom_size['target', 0])
            return distance <= radius
        if domain == 'finger':
            return bool(physics.dist_to_target() <= 0)
        if domain == 'swimmer':
            radius = float(physics.named.model.geom_size['target', 0])
            return distance <= radius
        if domain == 'quadruped':
            radius = float(physics.named.model.site_size['target', 0])
            return distance <= radius
        if domain == 'dog':
            radius = float(physics.named.model.geom_size['target', 0])
            return distance <= radius
        if domain == 'stacker':
            # dm_control exposes only a shaped reward for stacker. A 0.95
            # threshold is an explicit binary evaluation convention.
            return reward >= 0.95
        if domain == 'ball_in_cup':
            return bool(physics.in_target())
        return reward >= 1.0 - 1e-7

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
        is_success = self._is_success(reward, distance)
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
        goal_dims=TASK_SPECS[task_name]['goal_dims'],
    )


__all__ = ['DMCGoalEnv', 'TASK_SPECS', 'create_env_from_spec']

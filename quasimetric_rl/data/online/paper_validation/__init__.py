"""Original-paper tasks, isolated from the unified NQRL comparison suite."""

from __future__ import annotations

import functools
import importlib.metadata
import json
from pathlib import Path

import gym
import numpy as np

from ..goal_env import pack_goal_observation, vector_goal_observation_space
from ..memory import register_online_env
from .reference import (
    crl_sawyer_env, gcsl_dependencies, gcsl_native_env, load_module,
    make_ant_maze, repository, require_metaworld,
)


# (horizon, observation size, action size, conditioning coordinates,
#  evaluation coordinates, diagnostic success threshold).
SPECS = {
    'GCSLFourRooms': (50, 2, 2, (0, 1), (0, 1), .08),
    'GCSLSawyerPush': (50, 4, 2, (0, 1, 2, 3), (2, 3), .05),
    'GCSLSawyerDoor': (50, 4, 3, (3,), (3,), .05),
    # For native Discrete actions the action-size entry is the category count.
    'GCSLLunarLander': (50, 8, 4, (0, 1, 4, 6, 7), (0, 1), .08),
    'GCSLClaw': (50, 11, 9, (9, 10), (9, 10), .1),
    'CRLAntUMaze': (700, 29, 8, (0, 1), (0, 1), .5),
    'CRLSawyerPush': (150, 7, 4, tuple(range(7)), (3, 4, 5), .05),
    'CRLSawyerBin': (150, 7, 4, tuple(range(7)), (4, 5, 6), .05),
    # Upstream point_Small dynamics/reward, with a tighter diagnostic threshold.
    'CRLPointSmall': (50, 2, 2, (0, 1), (0, 1), .25),
    'CLearningSawyerReach': (50, 6, 4, (0, 1, 2), (0, 1, 2), .05),
    'CLearningSawyerPush': (151, 6, 4, tuple(range(6)), (3, 4, 5), .05),
    'CLearningSawyerWindow': (151, 6, 4, tuple(range(6)), (3, 4, 5), .05),
    'CLearningSawyerDrawer': (151, 6, 4, tuple(range(6)), (3, 4, 5), .05),
    'CLearningJacoReach': (250, 45, 9, tuple(range(3, 15)) + (27, 28, 29),
                          tuple(range(3, 15)) + (27, 28, 29), .05),
    'CLearningFinger': (1000, 12, 2, (2, 3), (2, 3), .01),
    'CLearningHand': (50, 30, 28, tuple(range(30)), tuple(range(30)), .05),
}


def make_base(name):
    if name in ('CLearningJacoReach', 'CLearningFinger', 'CLearningHand'):
        from .c_learning import make_reconstructed_env
        return make_reconstructed_env(name)
    if name in ('GCSLLunarLander', 'GCSLClaw'):
        return gcsl_native_env(name)
    if name in ('CRLSawyerPush', 'CRLSawyerBin'):
        return crl_sawyer_env(name)
    if name.startswith('GCSL'):
        source = gcsl_dependencies()
        if name == 'GCSLFourRooms':
            from room_world.pointmass import PMEnv
            return PMEnv(
                room_type='rooms', potential_type='none', shaped=True,
                max_path_length=50, use_state_images=False, use_goal_images=False,
                start_config=(np.array([-.33, -.33]), np.array([-.27, -.27])),
            )
        if name == 'GCSLSawyerPush':
            module = load_module(
                '_qrl_gcsl_sawyer_push', source / 'gcsl/envs/sawyer_push.py',
                skip_imports=('matplotlib', 'gcsl'),
                skip_definitions=('SawyerPushGoalEnv', 'main'),
            )
            return module.SawyerPushAndReachXYEnv(**module.push_configs['fixed_start'])
        from multiworld.envs.mujoco.sawyer_xyz.sawyer_door_hook import SawyerDoorHookEnv
        return SawyerDoorHookEnv(
            goal_low=(0,), goal_high=(.83,), hand_low=(-.1, .45, .1),
            hand_high=(.05, .65, .25), max_angle=.83,
            xml_path='sawyer_xyz/sawyer_door_pull_hook.xml',
            reward_type='angle_diff_and_hand_distance', reset_free=False,
        )
    source = repository('google-research-gcrl')
    if name == 'CRLPointSmall':
        module = load_module('_qrl_crl_point', source / 'contrastive_rl/point_env.py')
        return module.PointEnv(walls='Small')
    if name == 'CRLAntUMaze':
        module = load_module('_qrl_crl_ant', source / 'contrastive_rl/ant_env.py')
        return make_ant_maze(module, map_name='umaze', non_zero_reset=True)
    require_metaworld()
    module = load_module(
        '_qrl_c_learning_sawyer', source / 'c_learning/c_learning_envs.py',
        skip_imports=('gin', 'tf_agents'), strip_gin=True,
    )
    if name == 'CLearningSawyerReach':
        return module.SawyerReach()
    if name == 'CLearningSawyerPush':
        class ConfiguredPush(module.SawyerPush):
            def reset(self):
                return super().reset(arm_goal_type='goal', fix_z=True)
        return ConfiguredPush(random_init=True, goal_low=(-.1, .5, .05))
    if name == 'CLearningSawyerWindow':
        class ConfiguredWindow(module.SawyerWindow):
            def reset(self):
                return super().reset(arm_goal_type='goal')
        return ConfiguredWindow()
    if name == 'CLearningSawyerDrawer':
        class ConfiguredDrawer(module.SawyerDrawer):
            def reset(self):
                return super().reset(arm_goal_type='goal')
        return ConfiguredDrawer()
    raise ValueError(name)


class PaperGoalEnv(gym.Env):
    """Adapt goal representation and API, retaining upstream dynamics/rewards."""

    is_paper_validation = True

    def __init__(self, name):
        self.name = name
        self.episode_length, size, _, self.conditioning_dims, self.distance_dims, self.threshold = SPECS[name]
        self.base = make_base(name)
        self.action_space = self.base.action_space
        self.observation_space = vector_goal_observation_space(size)
        self.is_image_based = False
        self.steps = 0

    def seed(self, seed=None):
        # Upstream reset/goal samplers use both global NumPy and Gym RNGs.
        np.random.seed(seed)
        self.action_space.seed(seed)
        if self.name in ('GCSLLunarLander', 'GCSLClaw'):
            return self.base.inner_env.seed(seed)
        return self.base.seed(seed)

    def goal_distance(self, state, goal):
        """Support NumPy or Torch batches, including the original Claw metric."""
        if self.name == 'GCSLClaw':
            import torch
            atan2 = torch.atan2 if isinstance(state, torch.Tensor) else np.arctan2
            distance = abs(atan2(state[..., -1], state[..., -2])
                           - atan2(goal[..., -1], goal[..., -2])) % (2 * np.pi)
            return (torch.minimum(distance, 2 * np.pi - distance)
                    if isinstance(distance, torch.Tensor)
                    else np.minimum(distance, 2 * np.pi - distance))
        difference = (state - goal)[..., list(self.distance_dims)]
        return (difference ** 2).sum(-1) ** .5

    def _pack(self, obs):
        if isinstance(obs, dict):
            state = np.asarray(obs['observation'], dtype=np.float32)
            goal = np.asarray(obs['desired_goal'], dtype=np.float32)
            if self.name in ('GCSLLunarLander', 'GCSLClaw'):
                goal = goal[list(self.conditioning_dims)]
            return pack_goal_observation(state, goal, self.conditioning_dims)
        state, goal = np.split(np.asarray(obs, dtype=np.float32), 2)
        return dict(observation=state, achieved_goal=state.copy(), desired_goal=goal)

    def reset(self):
        self.steps = 0
        if self.name in ('GCSLLunarLander', 'GCSLClaw'):
            self._sampled_goal = np.asarray(self.base.sample_goal()).copy()
            obs = dict(observation=self.base.reset(), desired_goal=self._sampled_goal)
        elif self.name.startswith('GCSL'):
            # GCSL samples a goal with one reset, then resets again for rollout.
            goal_obs = self.base.reset()
            goal = np.asarray(goal_obs['desired_goal']).copy()
            obs = dict(self.base.reset())
            self._sampled_goal = goal
            obs['desired_goal'] = goal
        else:
            obs = self.base.reset()
        return self._pack(obs)

    def step(self, action):
        obs, reward, _, info = self.base.step(action)
        if self.name in ('GCSLLunarLander', 'GCSLClaw'):
            obs = dict(observation=obs, desired_goal=self._sampled_goal)
        if isinstance(obs, dict):
            obs = dict(obs, desired_goal=self._sampled_goal)
        packed = self._pack(obs)
        self.steps += 1
        distance = self.goal_distance(packed['observation'], packed['desired_goal'])
        info = dict(info, is_success=bool(distance < self.threshold), goal_distance=float(distance))
        if self.name == 'CRLAntUMaze':
            info['is_success'] = bool(reward)
        info['TimeLimit.truncated'] = self.steps >= self.episode_length
        return packed, float(reward), info['TimeLimit.truncated'], info

    def close(self):
        if self.name in ('GCSLLunarLander', 'GCSLClaw'):
            self.base.inner_env.close()
            if hasattr(self.base, 'inner_env_render'):
                self.base.inner_env_render.close()
        else:
            self.base.close()

    def protocol_metadata(self):
        path = Path(__file__).resolve().parents[4] / 'configs/baseline_paper_validation_protocols.json'
        protocol = json.loads(path.read_text())['tasks'].get(self.name)
        if protocol is None:
            return None
        result = dict(environment=self.name, protocol=protocol)
        packages = ('gym', 'numpy', 'mujoco-py', 'dm-control', 'mujoco', 'd4rl',
                    'robel', 'box2d-py', 'transforms3d')
        result['installed_versions'] = {}
        for package in packages:
            try:
                result['installed_versions'][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        if self.name == 'CLearningHand':
            result['model'] = self.base.model_provenance
        return result


for _name, _spec in SPECS.items():
    register_online_env(
        'paper_validation', _name, create_env_fn=functools.partial(PaperGoalEnv, _name),
        episode_length=_spec[0], goal_dims=_spec[4],
    )

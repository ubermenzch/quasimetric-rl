"""Appendix H.3 reconstructions where the public release has no task wrapper.

Unspecified choices are recorded in baseline_paper_validation_protocols.json.
These are local validation tasks, not recovered original implementations.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import gym
import numpy as np


class ResetGoalDMC(gym.Env):
    def __init__(self, name):
        from . import SPECS
        self.name = name
        self.goal_dims = SPECS[name][3]
        if name == 'CLearningJacoReach':
            from dm_control import manipulation
            self.env = manipulation.load('reach_site_features')
        else:
            from dm_control import suite
            self.env = suite.load('finger', 'turn_easy')
        action = self.env.action_spec()
        self.action_space = gym.spaces.Box(
            np.broadcast_to(action.minimum, action.shape).astype(np.float32),
            np.broadcast_to(action.maximum, action.shape).astype(np.float32),
        )
        self.observation_keys = tuple(self.env.observation_spec())
        size = sum(int(np.prod(v.shape)) for v in self.env.observation_spec().values())
        if size != SPECS[name][1]:
            raise ValueError(f'{name} observation layout changed: {self.env.observation_spec()}')
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (2 * size,), dtype=np.float32)
        # Verify semantic coordinates as well as the total observation size.
        offsets, offset = {}, 0
        for key, spec in self.env.observation_spec().items():
            offsets[key] = tuple(range(offset, offset + int(np.prod(spec.shape))))
            offset += int(np.prod(spec.shape))
        expected = (offsets['jaco_arm/joints_pos'] + offsets['jaco_arm/jaco_hand/joints_pos']
                    if name == 'CLearningJacoReach' else offsets['position'][2:4])
        if expected != self.goal_dims:
            raise ValueError(f'{name} goal coordinates changed: {offsets}')

    def seed(self, seed=None):
        rng = self.env.random_state if self.name == 'CLearningJacoReach' else self.env.task.random
        rng.seed(seed)
        self.action_space.seed(seed)
        return [seed]

    def state(self, timestep):
        return np.concatenate([np.asarray(timestep.observation[key]).reshape(-1)
                               for key in self.observation_keys]).astype(np.float32)

    def pack(self, state):
        return np.concatenate([state, self.goal])

    def reset(self):
        sampled = self.state(self.env.reset())
        self.goal = np.zeros_like(sampled)
        self.goal[list(self.goal_dims)] = sampled[list(self.goal_dims)]
        return self.pack(self.state(self.env.reset()))

    def step(self, action):
        timestep = self.env.step(np.asarray(action, dtype=self.env.action_spec().dtype))
        state = self.state(timestep)
        distance = np.linalg.norm((state - self.goal)[list(self.goal_dims)])
        # The native reach/turn reward concerns a different goal and is not used.
        return self.pack(state), -float(distance), timestep.last(), {}

    def close(self):
        self.env.close()


def hand_model_xml(path):
    """Expand D4RL's includes and remove the physical door, with no asset edits."""
    source_hashes = {}

    def expand(source):
        source = source.resolve()
        source_hashes[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
        root = ET.parse(source).getroot()
        for compiler in root.iter('compiler'):
            for key in ('meshdir', 'texturedir'):
                if key in compiler.attrib:
                    compiler.set(key, str((source.parent / compiler.get(key)).resolve()))
        for parent in root.iter():
            for child in list(parent):
                if child.tag == 'include':
                    included = expand(source.parent / child.attrib['file'])
                    index = list(parent).index(child)
                    parent.remove(child)
                    for item in reversed(list(included)):
                        parent.insert(index, item)
        return root

    root = expand(Path(path))
    world = root.find('worldbody')
    frame = world.find("body[@name='frame']")
    if frame is None or frame.find(".//joint[@name='door_hinge']") is None:
        raise ValueError('Expected the original D4RL door frame and hinge')
    world.remove(frame)
    root.set('model', 'C-Learning-Hand-H3-reconstruction-v1')
    xml = ET.tostring(root, encoding='unicode')
    return xml, dict(source_xml_sha256=source_hashes,
                     modified_xml_sha256=hashlib.sha256(xml.encode()).hexdigest())


def make_hand():
    import d4rl.hand_manipulation_suite
    import mujoco_py
    from mjrl.envs.mujoco_env import MujocoEnv

    source = Path(d4rl.hand_manipulation_suite.__file__).parent / 'assets/DAPG_door.xml'
    xml, provenance = hand_model_xml(source)

    class Hand(MujocoEnv):
        def __init__(self):
            sim = mujoco_py.MjSim(mujoco_py.load_model_from_xml(xml))
            self.grasp_sid = sim.model.site_name2id('S_grasp')
            self.act_mid = sim.model.actuator_ctrlrange.mean(axis=1)
            self.act_rng = np.diff(sim.model.actuator_ctrlrange, axis=1).ravel() / 2
            for first, last, gain in (('A_WRJ1', 'A_WRJ0', 10), ('A_FFJ3', 'A_THJ0', 1)):
                ids = slice(sim.model.actuator_name2id(first), sim.model.actuator_name2id(last) + 1)
                sim.model.actuator_gainprm[ids, :3] = [gain, 0, 0]
                sim.model.actuator_biasprm[ids, :3] = [0, -gain, 0]
            super().__init__(sim=sim, frame_skip=5)
            if self.model.nq != 28 or self.model.nu != 28 or self.obs_dim != 30:
                raise ValueError('D4RL hand model/observation layout changed')
            self.action_space = gym.spaces.Box(-1, 1, (28,), dtype=np.float32)
            self.model_provenance = provenance

        def get_obs(self):
            # Preserve the original Door observation's hand-only coordinates.
            return np.concatenate([self.data.qpos[1:], self.data.site_xpos[self.grasp_sid]])

        def reset_model(self):
            self.set_state(self.init_qpos.copy(), self.init_qvel.copy())
            return self.get_obs()

        def step(self, action):
            self.do_simulation(self.act_mid + np.clip(action, -1, 1) * self.act_rng, self.frame_skip)
            return self.get_obs(), 0., False, {}

    return RandomActionGoalHand(Hand())


class RandomActionGoalHand(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (60,), dtype=np.float32)

    def reset(self):
        self.env.reset()
        for _ in range(50):
            sampled, _, _, _ = self.env.step(self.action_space.sample())
        self.goal = sampled.copy()
        return np.concatenate([self.env.reset(), self.goal]).astype(np.float32)

    def step(self, action):
        state, _, done, info = self.env.step(action)
        return (np.concatenate([state, self.goal]).astype(np.float32),
                -float(np.linalg.norm(state - self.goal)), done, info)


def make_reconstructed_env(name):
    return make_hand() if name == 'CLearningHand' else ResetGoalDMC(name)

#!/usr/bin/env python3
"""Verify that goal-set candidates are paired independently of global CUDA RNG."""

import argparse
import copy
import sys
from pathlib import Path

import gym
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quasimetric_rl.data import EnvSpec
from quasimetric_rl.modules import QRLConf


def make_env_spec() -> EnvSpec:
    return EnvSpec(
        observation_space=gym.spaces.Box(
            low=-np.ones(4, dtype=np.float32),
            high=np.ones(4, dtype=np.float32),
            dtype=np.float32,
        ),
        observation_space_is_dict=False,
        action_space=gym.spaces.Box(
            low=-np.ones(2, dtype=np.float32),
            high=np.ones(2, dtype=np.float32),
            dtype=np.float32,
        ),
    )


def make_goal_set_loss(seed: int, num_candidates: int):
    conf = copy.deepcopy(QRLConf())
    conf.actor.model.arch = (8,)
    conf.quasimetric_critic.model.encoder.arch = (8,)
    conf.quasimetric_critic.model.encoder.latent_size = 4
    conf.quasimetric_critic.model.quasimetric_model.projector_arch = (8,)
    conf.quasimetric_critic.model.quasimetric_model.quasimetric_head_spec = 'l2(dim=4)'
    conf.quasimetric_critic.model.latent_dynamics.arch = (8,)
    conf.goal_set_distance.enabled = True
    conf.goal_set_distance.model.arch = (8,)
    conf.goal_set_distance.losses.goal_dims = (0, 1)
    conf.goal_set_distance.losses.num_goal_samples = num_candidates

    _, losses = conf.make(env_spec=make_env_spec(), total_optim_steps=10)
    loss = losses.goal_set_distance_loss
    loss.set_observation_bounds_provider(
        lambda *, device=None: (
            torch.full((4,), -2.0, device=device),
            torch.full((4,), 2.0, device=device),
        )
    )
    loss.set_candidate_seed(seed)
    return loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=29)
    parser.add_argument('--step', type=int, default=123)
    parser.add_argument('--num-candidates', type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError(
            'CUDA is unavailable. Check the PyTorch CUDA build, NVIDIA driver, '
            'and CUDA_VISIBLE_DEVICES.'
        )

    device = torch.device(args.device)
    loss_a = make_goal_set_loss(args.seed, args.num_candidates)
    loss_b = make_goal_set_loss(args.seed, args.num_candidates)
    candidate_state = dict(seed=args.seed, step=args.step)
    loss_a.load_candidate_rng_state_dict(candidate_state)
    loss_b.load_candidate_rng_state_dict(candidate_state)

    raw_goal_a = torch.tensor([
        [0.1, 0.2, 10.0, 20.0],
        [0.3, 0.4, 30.0, 40.0],
        [0.5, 0.6, 50.0, 60.0],
    ], device=device)
    raw_goal_b = raw_goal_a.clone()

    padded_goal_a = loss_a.padded_goal_state(raw_goal_a)
    candidates_a = loss_a._sample_goal_condition_states(raw_goal_a)

    # Simulate unrelated stochastic work performed by another algorithm.
    torch.rand(1_000_000, device=device)

    padded_goal_b = loss_b.padded_goal_state(raw_goal_b)
    candidates_b = loss_b._sample_goal_condition_states(raw_goal_b)
    expected_goal_dims = raw_goal_a[:, None, :2].expand_as(candidates_a[..., :2])

    checks = {
        'raw goal identical': torch.equal(raw_goal_a, raw_goal_b),
        'padded goal identical': torch.equal(padded_goal_a, padded_goal_b),
        'candidate set identical': torch.equal(candidates_a, candidates_b),
        'candidate goal dimensions preserved': torch.equal(
            candidates_a[..., :2], expected_goal_dims
        ),
    }
    for name, passed in checks.items():
        print(f'{name}: {passed}')

    max_difference = (candidates_a - candidates_b).abs().max().item()
    print(f'candidate shape: {tuple(candidates_a.shape)}')
    print(f'maximum difference: {max_difference}')
    print(f'GPU: {torch.cuda.get_device_name(device)}')

    if not all(checks.values()):
        raise AssertionError('CUDA candidate pairing check failed')
    print('PASS')


if __name__ == '__main__':
    main()

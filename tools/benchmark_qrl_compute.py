#!/usr/bin/env python3
"""Benchmark representative QRL training steps without data-loading overhead."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import gym
import numpy as np
import torch

import quasimetric_rl
from quasimetric_rl.data import BatchData, EnvSpec
from quasimetric_rl.modules import QRLConf


VARIANTS = (
    '1q_base', '2q_base', 'split_none', 'split_max8',
    'split_layernorm_max4', 'split_layernorm_min4',
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--warmup-steps', type=int, default=10)
    parser.add_argument('--measure-steps', type=int, default=30)
    parser.add_argument('--seed', type=int, default=20260722)
    return parser.parse_args()


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


def configure_agent(variant: str) -> QRLConf:
    conf = QRLConf()
    conf.num_critics = 2 if variant == '2q_base' else 1
    conf.quasimetric_critic.model.latent_dynamics.history_length = 1
    assert conf.actor is not None
    conf.actor.losses.min_dist.adaptive_entropy_regularizer = False
    conf.actor.losses.min_dist.add_goal_as_future_state = False
    conf.actor.losses.behavior_cloning.weight = 0

    if not variant.startswith('split_'):
        return conf

    encoder = conf.quasimetric_critic.model.encoder
    encoder.kind = 'split'
    encoder.goal_dims = (0, 1)
    encoder.latent_size = 128
    encoder.goal_arch = (369, 378)
    encoder.non_goal_arch = (369, 379)
    encoder.goal_latent_size = 64
    encoder.non_goal_latent_size = 64
    encoder.branch_normalization = (
        'layernorm' if variant.startswith('split_layernorm_') else 'rmsnorm'
    )
    conf.quasimetric_critic.model.quasimetric_model.projector_arch = (512,)
    conf.quasimetric_critic.model.latent_dynamics.arch = (512, 512)
    conf.actor.model.arch = (506, 512)
    conf.actor.model.input_mode = 'split_latent'
    if variant in ('split_max8', 'split_layernorm_max4', 'split_layernorm_min4'):
        min_dist = conf.actor.losses.min_dist
        min_dist.latent_goal_mode = (
            'min' if variant == 'split_layernorm_min4' else 'max'
        )
        min_dist.latent_goal_steps = (
            4 if variant.startswith('split_layernorm_') else 8
        )
        min_dist.latent_goal_keep_best = True
        min_dist.latent_goal_optim = 'adam'
        min_dist.latent_goal_lr = 0.01
    return conf


def make_batch(batch_size: int, device: torch.device) -> BatchData:
    observations = torch.rand(batch_size, 4, device=device).mul_(2).sub_(1)
    next_observations = torch.rand(batch_size, 4, device=device).mul_(2).sub_(1)
    return BatchData(
        observations=observations,
        actions=torch.rand(batch_size, 2, device=device).mul_(2).sub_(1),
        next_observations=next_observations,
        future_observations=torch.rand(batch_size, 4, device=device).mul_(2).sub_(1),
        rewards=torch.zeros(batch_size, device=device),
        terminals=torch.zeros(batch_size, dtype=torch.bool, device=device),
        timeouts=torch.zeros(batch_size, dtype=torch.bool, device=device),
    )


def git_output(*args: str) -> str:
    try:
        return subprocess.check_output(
            ('git', *args), text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ''


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.warmup_steps < 0 or args.measure_steps <= 0:
        raise ValueError('batch-size and measure-steps must be positive; warmup-steps must be non-negative')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but torch.cuda.is_available() is false')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    conf = configure_agent(args.variant)
    total_steps = args.warmup_steps + args.measure_steps + 1
    agent, losses = conf.make(
        env_spec=make_env_spec(), total_optim_steps=total_steps
    )
    agent.to(device)
    losses.to(device)
    data = make_batch(args.batch_size, device)

    for _ in range(args.warmup_steps):
        result = losses(agent, data, optimize=True, phase='all')
        del result

    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        start_event.record()
    else:
        wall_start = time.perf_counter()

    for _ in range(args.measure_steps):
        result = losses(agent, data, optimize=True, phase='all')
        del result

    if device.type == 'cuda':
        end_event.record()
        end_event.synchronize()
        elapsed_s = start_event.elapsed_time(end_event) / 1000
        synchronized_wall_s = time.perf_counter() - wall_start
        peak_allocated_mb = torch.cuda.max_memory_allocated(device) / 2**20
        peak_reserved_mb = torch.cuda.max_memory_reserved(device) / 2**20
        gpu_name = torch.cuda.get_device_name(device)
    else:
        elapsed_s = time.perf_counter() - wall_start
        synchronized_wall_s = elapsed_s
        peak_allocated_mb = None
        peak_reserved_mb = None
        gpu_name = None

    repo_root = Path(quasimetric_rl.__file__).resolve().parent.parent
    agent_parameters = sum(
        parameter.numel()
        for parameter in agent.parameters()
        if parameter.requires_grad
    )
    loss_parameters = sum(
        parameter.numel()
        for parameter in losses.parameters()
        if parameter.requires_grad
    )
    row = {
        'variant': args.variant,
        'batch_size': args.batch_size,
        'warmup_steps': args.warmup_steps,
        'measure_steps': args.measure_steps,
        'seed': args.seed,
        'device': str(device),
        'gpu_name': gpu_name,
        'elapsed_s': elapsed_s,
        'synchronized_wall_s': synchronized_wall_s,
        'ms_per_step': elapsed_s * 1000 / args.measure_steps,
        'steps_per_second': args.measure_steps / elapsed_s,
        'samples_per_second': args.measure_steps * args.batch_size / elapsed_s,
        'peak_allocated_mb': peak_allocated_mb,
        'peak_reserved_mb': peak_reserved_mb,
        'agent_trainable_parameters': agent_parameters,
        'loss_trainable_parameters': loss_parameters,
        'trainable_parameters': agent_parameters + loss_parameters,
        'torch_version': torch.__version__,
        'hostname': platform.node(),
        'pid': os.getpid(),
        'import_root': str(repo_root),
        'git_commit': git_output('-C', str(repo_root), 'rev-parse', 'HEAD'),
        'git_dirty': bool(git_output('-C', str(repo_root), 'status', '--short')),
    }
    print(json.dumps(row, sort_keys=True))


if __name__ == '__main__':
    main()

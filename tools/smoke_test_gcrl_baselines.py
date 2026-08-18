#!/usr/bin/env python3
"""Run one real replay update for every GCRL baseline/environment pair."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quasimetric_rl.data.online import ReplayBuffer
from quasimetric_rl.modules import QRLConf
from quasimetric_rl.modules.gcrl_baselines import resolve_baseline_goal_dims


ENVIRONMENTS = (
    ('gcrl', 'FetchReach', 50),
    ('gcrl', 'FetchPush', 50),
    ('gcrl', 'FetchSlide', 50),
    ('gcrl', 'FetchPickAndPlace', 50),
    ('dmc', 'reacher_easy', 1000),
    ('dmc', 'reacher_hard', 1000),
    ('gym_mujoco', 'Reacher-v4', 50),
)
ALGORITHMS = ('td_infonce', 'crl', 'scaling_crl', 'gcsl', 'c_learning')


def small_conf(algorithm: str) -> QRLConf:
    conf = copy.deepcopy(QRLConf())
    conf.algorithm = algorithm
    conf.baselines.td_infonce.hidden_sizes = (16, 16)
    conf.baselines.td_infonce.representation_dim = 8
    conf.baselines.crl.hidden_sizes = (16, 16)
    conf.baselines.crl.representation_dim = 8
    conf.baselines.scaling_crl.hidden_sizes = (16, 16, 16, 16)
    conf.baselines.scaling_crl.representation_dim = 8
    conf.baselines.gcbc.hidden_sizes = (16, 16)
    conf.baselines.c_learning.hidden_sizes = (16, 16)
    return conf


def smoke_environment(kind: str, name: str, horizon: int, batch_size: int) -> None:
    replay = ReplayBuffer.Conf(
        kind=kind,
        name=name,
        init_num_transitions=horizon,
        increment_num_transitions=horizon,
    ).make()
    collect_env = replay.create_env()
    try:
        collect_env.seed(12345)
        rollout = replay.collect_rollout(
            lambda _observation, _goal, space: space.sample(),
            env=collect_env,
        )
        replay.add_rollout(rollout)

        for algorithm in ALGORITHMS:
            torch.manual_seed(12345)
            conf = small_conf(algorithm)
            baseline_goal_dims = resolve_baseline_goal_dims(
                algorithm,
                env_kind=kind,
                env_name=name,
                state_dim=replay.env_spec.observation_shape.numel(),
                success_goal_dims=replay.goal_set_dims,
            )
            agent, losses = conf.make(
                env_spec=replay.env_spec,
                total_optim_steps=1,
                baseline_goal_dims=baseline_goal_dims,
            )
            if algorithm == 'gcsl':
                batch = replay.sample_uniform_future_pairs(batch_size)
            else:
                batch = replay.sample(batch_size)
            result = losses(agent, batch, optimize=True)
            action = agent.act(
                batch.observations,
                batch.future_observations,
            ).mean
            if not torch.isfinite(result.loss):
                raise RuntimeError(f'{kind}/{name}/{algorithm}: non-finite loss')
            if not torch.isfinite(action).all():
                raise RuntimeError(f'{kind}/{name}/{algorithm}: non-finite action')
            print(
                f'PASS: {kind}/{name}/{algorithm}, '
                f'loss={result.loss.item():.6f}, action_shape={tuple(action.shape)}'
            )
    finally:
        collect_env.close()
        replay.env.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument(
        '--environment',
        choices=('all', *(name for _, name, _ in ENVIRONMENTS)),
        default='all',
    )
    args = parser.parse_args()
    if args.batch_size < 2:
        parser.error('--batch-size must be at least 2 for C-Learning')

    np.random.seed(12345)
    selected = (
        ENVIRONMENTS
        if args.environment == 'all'
        else tuple(env for env in ENVIRONMENTS if env[1] == args.environment)
    )
    for environment in selected:
        smoke_environment(*environment, batch_size=args.batch_size)
    print(
        f'All {len(selected) * len(ALGORITHMS)} real-environment '
        'baseline updates passed.'
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

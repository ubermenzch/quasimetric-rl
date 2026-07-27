#!/usr/bin/env python3
"""Create and step every non-Fetch online simulator environment."""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--suite',
        choices=('all', 'dmc', 'gym-mujoco'),
        default='all',
        help='Limit verification to one simulator family.',
    )
    return parser.parse_args()


def package_version(name: str, expected: str) -> str:
    try:
        actual = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(f'Required simulator package is missing: {name}') from exc
    if actual != expected:
        raise RuntimeError(
            f'{name} version mismatch: expected {expected}, got {actual}'
        )
    return actual


def validate_observation(name: str, observation: Any) -> tuple[int, ...]:
    expected_keys = {'observation', 'achieved_goal', 'desired_goal'}
    if not isinstance(observation, dict) or set(observation) != expected_keys:
        actual = set(observation) if isinstance(observation, dict) else type(observation)
        raise RuntimeError(f'{name} returned invalid observation keys: {actual}')
    shapes = {
        tuple(np.asarray(observation[key]).shape)
        for key in expected_keys
    }
    if len(shapes) != 1:
        raise RuntimeError(f'{name} returned mismatched observation shapes: {shapes}')
    return next(iter(shapes))


def smoke_environment(
        family: str, name: str, factory: Callable[[str], Any],
        *, expected_backend: str | None = None) -> None:
    env = factory(name)
    try:
        if expected_backend is not None:
            backend = getattr(env, 'backend_id', None)
            if backend != expected_backend:
                raise RuntimeError(
                    f'{name} expected local backend {expected_backend}, '
                    f'got {backend!r}'
                )
        env.seed(12345)
        observation = env.reset()
        validate_observation(name, observation)
        next_observation, reward, done, info = env.step(
            env.action_space.sample()
        )
        observation_shape = validate_observation(name, next_observation)
        if not np.isfinite(reward):
            raise RuntimeError(f'{name} returned non-finite reward: {reward}')
        if done:
            raise RuntimeError(f'{name} terminated during the first step')
        if 'is_success' not in info or 'goal_distance' not in info:
            raise RuntimeError(f'{name} did not return goal metrics: {info}')
        backend_suffix = (
            f', backend={expected_backend}' if expected_backend is not None else ''
        )
        print(
            f'PASS: {family}/{name}, observation={observation_shape}'
            f'{backend_suffix}'
        )
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    if args.suite in ('all', 'dmc'):
        print(
            f'dm-control={package_version("dm-control", "1.0.3")}, '
            f'mujoco={package_version("mujoco", "2.3.6")}'
        )
        from quasimetric_rl.data.online.dmc import (
            TASK_SPECS as dmc_tasks,
            create_env_from_spec as create_dmc_env,
        )
        for name in dmc_tasks:
            smoke_environment('dmc', name, create_dmc_env)

    if args.suite in ('all', 'gym-mujoco'):
        print(
            f'gym={package_version("gym", "0.18.0")}, '
            f'mujoco-py={package_version("mujoco-py", "2.1.2.14")}'
        )
        from quasimetric_rl.data.online.gym_mujoco import (
            TASK_SPECS as gym_tasks,
            create_env_from_spec as create_gym_env,
        )
        expected_backends = {
            'Reacher-v4': 'Reacher-v2',
            'Pusher-v4': 'Pusher-v2',
            'AntNavigate-v4': 'Ant-v3',
        }
        for name in gym_tasks:
            smoke_environment(
                'gym_mujoco', name, create_gym_env,
                expected_backend=expected_backends[name],
            )

    print('Online simulator verification passed.')


if __name__ == '__main__':
    main()

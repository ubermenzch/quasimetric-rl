#!/usr/bin/env python3
"""Validate and benchmark stable adaptive entropy on a CUDA device."""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import statistics
import sys
from pathlib import Path
from typing import Any, Optional

import gym
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quasimetric_rl.data import BatchData, EnvSpec
from quasimetric_rl.data.env_spec.act_distn import BoxOutputLinearNormalization
from quasimetric_rl.data.env_spec.act_distn.utils import (
    stable_tanh_log_abs_det_jacobian,
)
from quasimetric_rl.modules import QRLConf
from quasimetric_rl.modules.actor.losses.min_dist import MinDistLoss


def comma_separated_positive_ints(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(','))
    except ValueError as exc:
        raise argparse.ArgumentTypeError('expected comma-separated integers') from exc
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError('all sample counts must be positive')
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument(
        '--samples',
        type=comma_separated_positive_ints,
        default=(1, 8, 16, 32, 100),
        help='Entropy-only Monte Carlo sample counts.',
    )
    parser.add_argument('--warmup-steps', type=int, default=10)
    parser.add_argument('--measure-steps', type=int, default=30)
    parser.add_argument('--seed', type=int, default=20260723)
    parser.add_argument(
        '--full-actor',
        action='store_true',
        help='Also benchmark Base-sized AntMaze actor updates.',
    )
    parser.add_argument(
        '--full-actor-samples',
        type=comma_separated_positive_ints,
        default=(1, 32),
    )
    parser.add_argument('--full-actor-warmup-steps', type=int, default=2)
    parser.add_argument('--full-actor-measure-steps', type=int, default=5)
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Optional JSON output path.',
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        'batch-size': args.batch_size,
        'measure-steps': args.measure_steps,
        'full-actor-measure-steps': args.full_actor_measure_steps,
    }
    nonnegative = {
        'warmup-steps': args.warmup_steps,
        'full-actor-warmup-steps': args.full_actor_warmup_steps,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f'{name} must be positive, got {value}')
    for name, value in nonnegative.items():
        if value < 0:
            raise ValueError(f'{name} must be non-negative, got {value}')


def make_env_spec() -> EnvSpec:
    return EnvSpec(
        observation_space=gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(29,),
            dtype=np.float32,
        ),
        observation_space_is_dict=False,
        action_space=gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(8,),
            dtype=np.float32,
        ),
    )


def make_batch(batch_size: int, device: torch.device) -> BatchData:
    observations = torch.randn(batch_size, 29, device=device)
    next_observations = observations + 0.05 * torch.randn_like(observations)
    return BatchData(
        observations=observations,
        actions=torch.empty(batch_size, 8, device=device).uniform_(-1.0, 1.0),
        next_observations=next_observations,
        future_observations=torch.randn(batch_size, 29, device=device),
        rewards=torch.zeros(batch_size, device=device),
        terminals=torch.zeros(batch_size, dtype=torch.bool, device=device),
        timeouts=torch.zeros(batch_size, dtype=torch.bool, device=device),
    )


def synchronize(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def cuda_elapsed_ms(device: torch.device, function) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    function()
    end.record()
    synchronize(device)
    return float(start.elapsed_time(end))


def tensor_max_abs(value: torch.Tensor) -> float:
    return float(value.detach().abs().max().cpu())


def run_correctness_checks(device: torch.device, seed: int) -> dict[str, float]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    moderate_x = torch.linspace(-3.0, 3.0, 4097, device=device)
    stable = stable_tanh_log_abs_det_jacobian(moderate_x)
    direct = torch.log1p(-torch.tanh(moderate_x).square())
    formula_error = tensor_max_abs(stable - direct)
    if formula_error > 2e-5:
        raise AssertionError(f'stable Jacobian formula error is {formula_error:g}')

    extreme_x = torch.tensor([10.0], device=device, requires_grad=True)
    extreme_value = stable_tanh_log_abs_det_jacobian(extreme_x).sum()
    extreme_value.backward()
    expected_gradient = -2.0 * torch.tanh(extreme_x.detach())
    extreme_gradient_error = tensor_max_abs(extreme_x.grad - expected_gradient)
    if not torch.isfinite(extreme_value) or extreme_gradient_error > 1e-6:
        raise AssertionError('extreme tanh Jacobian value or gradient is invalid')

    converter = BoxOutputLinearNormalization(make_env_spec().action_space).to(device)
    target_std = torch.tensor(0.5 - 1e-4, device=device)
    raw_std = torch.log(torch.expm1(target_std))
    feature = torch.empty(64, 16, device=device)
    feature[:, :8] = 10.0
    feature[:, 8:] = raw_std
    feature.requires_grad_()
    entropy = converter(feature).entropy(num_samples=4096).mean()
    entropy.backward()
    saturated_loc_gradient = float(feature.grad[:, :8].mean().cpu())
    if not torch.isfinite(entropy) or saturated_loc_gradient >= -0.02:
        raise AssertionError(
            'stable action entropy did not retain a gradient for saturated means'
        )

    min_dist = MinDistLoss(
        env_spec=make_env_spec(),
        adaptive_entropy_regularizer=True,
        target_entropy=-8.0,
        entropy_mc_samples=32,
        add_goal_as_future_state=False,
    ).to(device)
    low_entropy = torch.tensor(-20.0, device=device, requires_grad=True)
    low_loss, _ = min_dist.adaptive_entropy_loss(low_entropy)
    low_loss.backward()
    low_entropy_alpha_gradient = float(min_dist.raw_entropy_weight.grad.cpu())
    if low_entropy_alpha_gradient >= 0:
        raise AssertionError('alpha would not increase below the target entropy')

    min_dist.raw_entropy_weight.grad = None
    high_entropy = torch.tensor(0.0, device=device, requires_grad=True)
    high_loss, _ = min_dist.adaptive_entropy_loss(high_entropy)
    high_loss.backward()
    high_entropy_alpha_gradient = float(min_dist.raw_entropy_weight.grad.cpu())
    if high_entropy_alpha_gradient <= 0:
        raise AssertionError('alpha would not decrease above the target entropy')

    return {
        'jacobian_formula_max_abs_error': formula_error,
        'extreme_x_value': float(extreme_value.detach().cpu()),
        'extreme_x_gradient': float(extreme_x.grad.detach().cpu()),
        'extreme_x_gradient_abs_error': extreme_gradient_error,
        'saturated_policy_entropy': float(entropy.detach().cpu()),
        'saturated_loc_gradient_mean': saturated_loc_gradient,
        'low_entropy_alpha_gradient': low_entropy_alpha_gradient,
        'high_entropy_alpha_gradient': high_entropy_alpha_gradient,
    }


def run_end_to_end_smoke(
        device: torch.device, seed: int, entropy_samples: int) -> dict[str, float]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    conf = QRLConf()
    conf.num_critics = 1
    assert conf.actor is not None
    conf.actor.model.arch = (64, 64)
    conf.actor.losses.min_dist.adaptive_entropy_regularizer = True
    conf.actor.losses.min_dist.target_entropy = -8.0
    conf.actor.losses.min_dist.entropy_mc_samples = entropy_samples
    conf.actor.losses.min_dist.add_goal_as_future_state = False
    conf.actor.losses.behavior_cloning.weight = 0.0
    critic = conf.quasimetric_critic.model
    critic.encoder.arch = (64, 64)
    critic.encoder.latent_size = 32
    critic.quasimetric_model.projector_arch = (64,)
    critic.quasimetric_model.quasimetric_head_spec = 'l2(dim=32)'
    critic.latent_dynamics.arch = (64, 64)
    critic.latent_dynamics.history_length = 1

    agent, losses = conf.make(env_spec=make_env_spec(), total_optim_steps=10)
    agent.to(device)
    losses.to(device)
    data = make_batch(256, device)
    actor_before = [parameter.detach().clone() for parameter in agent.actor.parameters()]
    alpha_before = float(
        losses.actor_loss.min_dist.raw_entropy_weight.detach().exp().cpu()
    )
    result = losses(agent, data, optimize=True, phase='actor')
    synchronize(device)
    if not torch.isfinite(result.loss):
        raise AssertionError('end-to-end actor loss is non-finite')
    actor_max_update = max(
        float((after.detach() - before).abs().max().cpu())
        for before, after in zip(actor_before, agent.actor.parameters())
    )
    alpha_after = float(
        losses.actor_loss.min_dist.raw_entropy_weight.detach().exp().cpu()
    )
    if actor_max_update == 0:
        raise AssertionError('end-to-end actor parameters did not update')
    if alpha_after == alpha_before:
        raise AssertionError('end-to-end entropy temperature did not update')

    min_dist_info = result.info['actor']['min_dist']
    return {
        'loss': float(result.loss.detach().cpu()),
        'entropy': float(min_dist_info['entropy'].cpu()),
        'target_entropy': float(min_dist_info['target_entropy']),
        'entropy_gap': float(min_dist_info['entropy_gap'].cpu()),
        'alpha_before': alpha_before,
        'alpha_after': alpha_after,
        'actor_max_parameter_update': actor_max_update,
    }


def entropy_only_step(
        converter: BoxOutputLinearNormalization,
        feature: torch.Tensor,
        samples: int) -> None:
    feature.grad = None
    entropy = converter(feature).entropy(num_samples=samples).mean()
    (-entropy).backward()


def benchmark_entropy_only(
        device: torch.device,
        batch_size: int,
        sample_counts: tuple[int, ...],
        warmup_steps: int,
        measure_steps: int,
        seed: int) -> list[dict[str, float]]:
    rows = []
    for samples in sample_counts:
        gc.collect()
        torch.cuda.empty_cache()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        converter = BoxOutputLinearNormalization(make_env_spec().action_space).to(device)
        feature = torch.zeros(batch_size, 16, device=device, requires_grad=True)
        for _ in range(warmup_steps):
            entropy_only_step(converter, feature, samples)
        synchronize(device)

        baseline_memory = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        timings = [
            cuda_elapsed_ms(
                device,
                lambda: entropy_only_step(converter, feature, samples),
            )
            for _ in range(measure_steps)
        ]
        peak_memory = torch.cuda.max_memory_allocated(device)
        rows.append({
            'samples': samples,
            'elements': samples * batch_size * 8,
            'median_ms': statistics.median(timings),
            'mean_ms': statistics.mean(timings),
            'std_ms': statistics.pstdev(timings),
            'min_ms': min(timings),
            'max_ms': max(timings),
            'peak_extra_mib': max(0, peak_memory - baseline_memory) / 1024 ** 2,
            'peak_total_mib': peak_memory / 1024 ** 2,
        })
        del feature, converter
    baseline_ms = rows[0]['median_ms']
    for row in rows:
        row['median_overhead_vs_first_pct'] = (
            (row['median_ms'] / baseline_ms - 1.0) * 100.0
        )
    torch.cuda.empty_cache()
    return rows


def configure_base_agent(entropy_samples: Optional[int]) -> QRLConf:
    conf = QRLConf()
    conf.num_critics = 1
    assert conf.actor is not None
    conf.actor.model.arch = (1024, 1024, 1024, 1024)
    conf.actor.model.input_mode = 'raw'
    min_dist = conf.actor.losses.min_dist
    min_dist.adaptive_entropy_regularizer = entropy_samples is not None
    min_dist.target_entropy = -8.0 if entropy_samples is not None else None
    min_dist.entropy_mc_samples = entropy_samples or 1
    min_dist.add_goal_as_future_state = False
    conf.actor.losses.behavior_cloning.weight = 0.0

    critic = conf.quasimetric_critic.model
    critic.encoder.kind = 'standard'
    critic.encoder.arch = (1024, 1024, 1024)
    critic.encoder.latent_size = 256
    critic.quasimetric_model.projector_arch = (1024, 1024)
    critic.quasimetric_model.quasimetric_head_spec = 'iqe(dim=2048,components=64)'
    critic.latent_dynamics.kind = 'mlp'
    critic.latent_dynamics.arch = (1024, 1024, 1024)
    critic.latent_dynamics.residual = True
    critic.latent_dynamics.history_length = 1
    return conf


def benchmark_full_actor(
        device: torch.device,
        batch_size: int,
        entropy_sample_counts: tuple[int, ...],
        warmup_steps: int,
        measure_steps: int,
        seed: int) -> list[dict[str, Any]]:
    rows = []
    modes: tuple[tuple[str, Optional[int]], ...] = (
        ('entropy_off', None),
        *((f'entropy_n{samples}', samples) for samples in entropy_sample_counts),
    )
    for mode, entropy_samples in modes:
        gc.collect()
        torch.cuda.empty_cache()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        conf = configure_base_agent(entropy_samples)
        total_steps = warmup_steps + measure_steps + 1
        agent, losses = conf.make(
            env_spec=make_env_spec(), total_optim_steps=total_steps
        )
        agent.to(device)
        losses.to(device)
        data = make_batch(batch_size, device)

        last_result = None
        for _ in range(warmup_steps):
            last_result = losses(agent, data, optimize=True, phase='actor')
        synchronize(device)
        baseline_memory = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)

        timings = []
        for _ in range(measure_steps):
            def step() -> None:
                nonlocal last_result
                last_result = losses(agent, data, optimize=True, phase='actor')

            timings.append(cuda_elapsed_ms(device, step))
        peak_memory = torch.cuda.max_memory_allocated(device)
        assert last_result is not None
        row: dict[str, Any] = {
            'mode': mode,
            'entropy_samples': entropy_samples,
            'median_ms': statistics.median(timings),
            'mean_ms': statistics.mean(timings),
            'std_ms': statistics.pstdev(timings),
            'min_ms': min(timings),
            'max_ms': max(timings),
            'peak_extra_mib': max(0, peak_memory - baseline_memory) / 1024 ** 2,
            'peak_total_mib': peak_memory / 1024 ** 2,
            'loss': float(last_result.loss.detach().cpu()),
        }
        if entropy_samples is not None:
            info = last_result.info['actor']['min_dist']
            row.update({
                'entropy': float(info['entropy'].cpu()),
                'entropy_alpha': float(info['entropy_alpha'].cpu()),
                'entropy_gap': float(info['entropy_gap'].cpu()),
            })
        rows.append(row)
        del data, losses, agent
    baseline_ms = rows[0]['median_ms']
    baseline_peak_mib = rows[0]['peak_total_mib']
    for row in rows:
        row['median_overhead_vs_entropy_off_pct'] = (
            (row['median_ms'] / baseline_ms - 1.0) * 100.0
        )
        row['peak_total_overhead_vs_entropy_off_mib'] = (
            row['peak_total_mib'] - baseline_peak_mib
        )
    torch.cuda.empty_cache()
    return rows


def print_table(title: str, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    print(f'\n{title}')
    widths = {
        column: max(len(column), *(len(str(row.get(column, ''))) for row in rows))
        for column in columns
    }
    print('  '.join(column.ljust(widths[column]) for column in columns))
    print('  '.join('-' * widths[column] for column in columns))
    for row in rows:
        rendered = []
        for column in columns:
            value = row.get(column, '')
            if isinstance(value, float):
                value = f'{value:.4f}'
            rendered.append(str(value).ljust(widths[column]))
        print('  '.join(rendered))


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type != 'cuda':
        raise RuntimeError('This script requires a CUDA device')
    if not torch.cuda.is_available():
        raise RuntimeError(
            'CUDA is unavailable. Check the PyTorch CUDA build, NVIDIA driver, '
            'and CUDA_VISIBLE_DEVICES.'
        )
    torch.cuda.set_device(device)

    device_index = device.index if device.index is not None else torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    metadata = {
        'python': platform.python_version(),
        'torch': torch.__version__,
        'torch_cuda': torch.version.cuda,
        'device': str(device),
        'gpu_name': properties.name,
        'gpu_total_memory_mib': properties.total_memory / 1024 ** 2,
        'batch_size': args.batch_size,
        'seed': args.seed,
    }
    print(json.dumps(metadata, ensure_ascii=False, indent=2))

    correctness = run_correctness_checks(device, args.seed)
    smoke = run_end_to_end_smoke(device, args.seed + 1, entropy_samples=32)
    print('\nCorrectness checks: PASS')
    print(json.dumps(correctness, indent=2))
    print('\nEnd-to-end adaptive actor update: PASS')
    print(json.dumps(smoke, indent=2))

    entropy_only = benchmark_entropy_only(
        device,
        args.batch_size,
        args.samples,
        args.warmup_steps,
        args.measure_steps,
        args.seed + 2,
    )
    print_table(
        'Entropy-only CUDA benchmark',
        entropy_only,
        (
            'samples', 'elements', 'median_ms', 'mean_ms',
            'median_overhead_vs_first_pct', 'peak_extra_mib',
        ),
    )

    full_actor = []
    if args.full_actor:
        full_actor = benchmark_full_actor(
            device,
            args.batch_size,
            args.full_actor_samples,
            args.full_actor_warmup_steps,
            args.full_actor_measure_steps,
            args.seed + 3,
        )
        print_table(
            'Base-sized AntMaze actor-update CUDA benchmark',
            full_actor,
            (
                'mode', 'median_ms', 'median_overhead_vs_entropy_off_pct',
                'peak_total_overhead_vs_entropy_off_mib', 'entropy',
                'entropy_alpha',
            ),
        )

    output = {
        'metadata': metadata,
        'correctness': correctness,
        'end_to_end_smoke': smoke,
        'entropy_only': entropy_only,
        'full_actor': full_actor,
    }
    print('\nRESULT_JSON_BEGIN')
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print('RESULT_JSON_END')
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(output, ensure_ascii=False, indent=2) + '\n'
        )
        print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()

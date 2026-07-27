#!/usr/bin/env python3
"""Compare QRL and GO-QRL-Max4 LayerNorm training speed."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


BASE_VARIANT = '1q_base'
SPLIT_VARIANT = 'split_layernorm_max4'
VARIANTS = (BASE_VARIANT, SPLIT_VARIANT)
LABELS = {
    BASE_VARIANT: 'QRL',
    SPLIT_VARIANT: 'GO-QRL-Max4 LayerNorm',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--warmup-steps', type=int, default=10)
    parser.add_argument('--measure-steps', type=int, default=30)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=20260722)
    parser.add_argument(
        '--json-output', type=Path,
        help='Optional path for raw runs and the aggregate comparison.',
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0 or args.measure_steps <= 0 or args.repeats <= 0:
        raise ValueError('batch-size, measure-steps, and repeats must be positive')
    if args.warmup_steps < 0:
        raise ValueError('warmup-steps must be non-negative')


def benchmark_command(args: argparse.Namespace, variant: str, seed: int) -> list[str]:
    return [
        sys.executable,
        '-m',
        'tools.benchmark_qrl_compute',
        '--variant',
        variant,
        '--device',
        args.device,
        '--batch-size',
        str(args.batch_size),
        '--warmup-steps',
        str(args.warmup_steps),
        '--measure-steps',
        str(args.measure_steps),
        '--seed',
        str(seed),
    ]


def parse_benchmark_output(stdout: str, variant: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get('variant') == variant:
            return row
    raise RuntimeError(
        f'Benchmark for {variant} did not emit a matching JSON result:\n{stdout}'
    )


def run_benchmark(
        args: argparse.Namespace, repo_root: Path, variant: str,
        repetition: int) -> dict[str, Any]:
    seed = args.seed + repetition
    command = benchmark_command(args, variant, seed)
    env = os.environ.copy()
    existing_pythonpath = env.get('PYTHONPATH')
    env['PYTHONPATH'] = str(repo_root) + (
        os.pathsep + existing_pythonpath if existing_pythonpath else ''
    )
    result = subprocess.run(
        command,
        cwd=repo_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f'Benchmark failed for {variant} (exit {result.returncode}).\n'
            f'stdout:\n{result.stdout}\nstderr:\n{result.stderr}'
        )
    row = parse_benchmark_output(result.stdout, variant)
    row['repetition'] = repetition + 1
    return row


def median(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return statistics.median(values) if values else None


def aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        variant_rows = [row for row in rows if row['variant'] == variant]
        parameter_counts = {
            int(row['agent_trainable_parameters']) for row in variant_rows
        }
        if len(parameter_counts) != 1:
            raise RuntimeError(
                f'Inconsistent parameter counts for {variant}: {parameter_counts}'
            )
        summary[variant] = {
            'label': LABELS[variant],
            'runs': len(variant_rows),
            'agent_trainable_parameters': parameter_counts.pop(),
            'ms_per_step_median': median(variant_rows, 'ms_per_step'),
            'steps_per_second_median': median(variant_rows, 'steps_per_second'),
            'samples_per_second_median': median(variant_rows, 'samples_per_second'),
            'peak_allocated_mb_median': median(variant_rows, 'peak_allocated_mb'),
            'peak_reserved_mb_median': median(variant_rows, 'peak_reserved_mb'),
        }

    base = summary[BASE_VARIANT]
    split = summary[SPLIT_VARIANT]
    base_ms = float(base['ms_per_step_median'])
    split_ms = float(split['ms_per_step_median'])
    summary['comparison'] = {
        'time_ratio': split_ms / base_ms,
        'slower_percent': (split_ms / base_ms - 1.0) * 100.0,
        'throughput_ratio': (
            float(split['samples_per_second_median'])
            / float(base['samples_per_second_median'])
        ),
        'peak_allocated_mb_delta': (
            None
            if base['peak_allocated_mb_median'] is None
            else float(split['peak_allocated_mb_median'])
            - float(base['peak_allocated_mb_median'])
        ),
    }
    return summary


def format_number(value: float | None, decimals: int = 2) -> str:
    return '-' if value is None else f'{value:,.{decimals}f}'


def print_summary(summary: dict[str, dict[str, Any]]) -> None:
    base = summary[BASE_VARIANT]
    split = summary[SPLIT_VARIANT]
    comparison = summary['comparison']
    print()
    print(f"{'metric':<24} {'1Q Base':>16} {'LayerNorm Max4':>20}")
    print('-' * 62)
    print(
        f"{'agent params':<24} "
        f"{base['agent_trainable_parameters']:>16,} "
        f"{split['agent_trainable_parameters']:>20,}"
    )
    for label, key in (
            ('median ms/step', 'ms_per_step_median'),
            ('median steps/s', 'steps_per_second_median'),
            ('median samples/s', 'samples_per_second_median'),
            ('peak allocated MiB', 'peak_allocated_mb_median'),
            ('peak reserved MiB', 'peak_reserved_mb_median')):
        print(
            f'{label:<24} '
            f'{format_number(base[key]):>16} '
            f'{format_number(split[key]):>20}'
        )
    print()
    print(
        'LayerNorm Max4 / Base time: '
        f"{comparison['time_ratio']:.3f}x "
        f"({comparison['slower_percent']:+.2f}%)"
    )
    print(
        'LayerNorm Max4 / Base throughput: '
        f"{comparison['throughput_ratio']:.3f}x"
    )
    if comparison['peak_allocated_mb_delta'] is not None:
        print(
            'LayerNorm Max4 peak allocation delta: '
            f"{comparison['peak_allocated_mb_delta']:+.1f} MiB"
        )


def main() -> None:
    args = parse_args()
    validate_args(args)
    repo_root = Path(__file__).resolve().parent.parent
    rows: list[dict[str, Any]] = []
    total_runs = args.repeats * len(VARIANTS)
    completed = 0
    for repetition in range(args.repeats):
        order = VARIANTS if repetition % 2 == 0 else tuple(reversed(VARIANTS))
        for variant in order:
            completed += 1
            print(
                f'[{completed}/{total_runs}] {LABELS[variant]} '
                f'(repeat {repetition + 1}/{args.repeats})',
                flush=True,
            )
            rows.append(run_benchmark(args, repo_root, variant, repetition))

    summary = aggregate(rows)
    print_summary(summary)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps({'runs': rows, 'summary': summary}, indent=2, sort_keys=True)
            + '\n'
        )
        print(f'JSON results: {args.json_output}')


if __name__ == '__main__':
    main()

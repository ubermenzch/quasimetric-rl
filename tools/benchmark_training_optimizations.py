#!/usr/bin/env python3
"""Benchmark each GPU training optimization on the real online QRL path."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
import itertools
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from typing import Optional


ROOT = Path(__file__).resolve().parents[1]
VENV_ROOT = ROOT / '.venv'
VENV_PYTHON = VENV_ROOT / 'bin/python'
if Path(sys.prefix).resolve() != VENV_ROOT.resolve():
    if not VENV_PYTHON.is_file():
        raise SystemExit(f'Missing QRL Python environment: {VENV_PYTHON}')
    os.execv(
        str(VENV_PYTHON),
        [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
    )
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.generate_go_qrl_hybrid_scale_tasks import (  # noqa: E402
    extra_arg_map,
    generate_all_tasks,
    task_scale,
)
from tools.run_qrl_queue import command_env, parse_config  # noqa: E402


OPTIMIZATION_DEFAULTS = (
    'training_optimizations.tf32=false',
    'training_optimizations.amp_dtype=null',
    'training_optimizations.fused_adamw=false',
    'training_optimizations.compile_heavy_modules=false',
    'training_optimizations.compile_mode=default',
    'training_optimizations.compile_fullgraph=false',
    'training_optimizations.compile_dynamic=false',
    'training_optimizations.compile_suppress_errors=true',
)

FEATURE_ARGS = {
    'tf32': ('training_optimizations.tf32=true',),
    'bf16': ('training_optimizations.amp_dtype=bfloat16',),
    'fused_adamw': ('training_optimizations.fused_adamw=true',),
    'compile': ('training_optimizations.compile_heavy_modules=true',),
}
OPTIMIZATION_FEATURES = tuple(FEATURE_ARGS)

INDIVIDUAL_GPU_VARIANTS = (
    ('baseline', 'tf32', 'bf16'),
    ('baseline', 'fused_adamw', 'compile'),
)

QUEUE_CONFIG = ROOT / 'configs/qrl_queue.env'

BENCHMARK_OVERRIDE_KEYS = frozenset({
    'eval_steps',
    'keep_only_latest_checkpoint',
    'log_steps',
    'resume_if_possible',
    'save_final_replay_buffer',
    'save_replay_buffer',
    'save_steps',
    'interaction.num_eval_episodes',
    'interaction.num_prefill_episodes',
    'interaction.num_rollouts_per_cycle',
    'interaction.num_samples_per_cycle',
    'interaction.num_test_episodes',
    'interaction.random_policy_env_steps',
    'interaction.total_env_steps',
})


@dataclass
class RunResult:
    variant: str
    gpu: int
    return_code: int
    wall_seconds: float
    train_seconds: Optional[float]
    first_segment_seconds: Optional[float]
    steady_seconds_per_batch: Optional[float]
    steady_cv_pct: Optional[float]
    steady_batches: int
    segments: int
    output_dir: str
    log_file: str
    error: Optional[str] = None


@dataclass(frozen=True)
class BaselineMetrics:
    steady_seconds_per_batch: float
    wall_seconds: float


def token_key(token: str) -> str:
    return token.split('=', 1)[0].lstrip('+')


def real_training_args(scale: str) -> list[str]:
    matches = [
        task for task in generate_all_tasks()
        if task.env_name == 'FetchSlide'
        and task.seed == '1000'
        and task_scale(task) == scale
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f'Expected one FetchSlide seed-1000 task for {scale}, got {len(matches)}'
        )
    task = matches[0]
    # Parse once here so malformed generated arguments fail before GPU work starts.
    extra_arg_map(task)
    return [
        token for token in shlex.split(task.extra_args)
        if not token_key(token).startswith('training_optimizations.')
        and token_key(token) not in BENCHMARK_OVERRIDE_KEYS
    ]


def variant_optimization_args(variant: str) -> list[str]:
    overrides = {
        token_key(token): token for token in OPTIMIZATION_DEFAULTS
    }
    for feature in variant_features(variant):
        for token in FEATURE_ARGS[feature]:
            overrides[token_key(token)] = token
    return list(overrides.values())


def variant_features(variant: str) -> frozenset[str]:
    if is_baseline_variant(variant):
        return frozenset()
    features = variant.split('+')
    unknown = set(features) - set(OPTIMIZATION_FEATURES)
    if unknown or len(features) != len(set(features)):
        raise ValueError(f'Invalid optimization variant {variant!r}')
    return frozenset(features)


def is_baseline_variant(variant: str) -> bool:
    return variant == 'baseline' or variant.startswith('baseline_')


def variant_from_mask(mask: int) -> str:
    if mask == 0:
        return 'baseline_factorial'
    return '+'.join(
        feature
        for bit, feature in enumerate(OPTIMIZATION_FEATURES)
        if mask & (1 << bit)
    )


def full_ablation_variants() -> tuple[str, ...]:
    return tuple(
        variant_from_mask(mask)
        for mask in range(1, 1 << len(OPTIMIZATION_FEATURES))
    )


def full_ablation_rounds() -> tuple[tuple[str, str], ...]:
    """Return time-blocked, GPU-balanced complementary configurations."""
    full_mask = (1 << len(OPTIMIZATION_FEATURES)) - 1
    complementary_pairs = tuple(
        (mask, full_mask ^ mask)
        for mask in range(1 << (len(OPTIMIZATION_FEATURES) - 1))
    )
    selected = None
    for orientations in itertools.product((0, 1), repeat=len(complementary_pairs)):
        # Keep the factorial baseline on GPU 0 so its full4 complement is on GPU 1.
        if orientations[0] != 0:
            continue
        gpu0_masks = [
            pair[orientation]
            for pair, orientation in zip(complementary_pairs, orientations)
        ]
        if all(
                sum(bool(mask & (1 << bit)) for mask in gpu0_masks) == 4
                for bit in range(len(OPTIMIZATION_FEATURES))):
            selected = tuple(
                (pair[orientation], pair[1 - orientation])
                for pair, orientation in zip(complementary_pairs, orientations)
            )
            break
    if selected is None:
        raise RuntimeError('Could not construct a balanced full-ablation design')

    # Put the factorial baseline/full4 pair last so it also serves as a late anchor.
    factorial = tuple(pair for pair in selected if 0 not in pair)
    baseline_pair = next(pair for pair in selected if 0 in pair)
    return (
        ('baseline_pre', 'baseline_pre'),
        *(tuple(variant_from_mask(mask) for mask in pair) for pair in factorial),
        tuple(variant_from_mask(mask) for mask in baseline_pair),
        ('baseline_post', 'baseline_post'),
    )


def benchmark_plans(
        gpus: tuple[int, int], suite: str,
) -> tuple[tuple[int, tuple[str, ...]], ...]:
    if suite == 'individual':
        return tuple(zip(gpus, INDIVIDUAL_GPU_VARIANTS))
    if suite != 'full':
        raise ValueError(f'Unknown benchmark suite {suite!r}')
    rounds = full_ablation_rounds()
    return (
        (gpus[0], tuple(round_pair[0] for round_pair in rounds)),
        (gpus[1], tuple(round_pair[1] for round_pair in rounds)),
    )


def make_command(
        *, output_root: Path, gpu: int, variant: str, scale: str,
        total_env_steps: int, prefill_episodes: int,
        samples_per_cycle: int) -> tuple[list[str], Path, Path]:
    run_name = f'gpu{gpu}_{scale}_{variant}'
    result_dir = output_root / 'results' / run_name
    log_file = output_root / 'logs' / f'{run_name}.log'
    command = [
        str(ROOT / '.venv/bin/python'),
        '-m',
        'online.main',
        *real_training_args(scale),
        'env.name=FetchSlide',
        'seed=1000',
        'device.index=0',
        f'output_base_dir={output_root / "results"}',
        f'output_folder={run_name}',
        'overwrite_output=True',
        'resume_if_possible=False',
        f'interaction.total_env_steps={total_env_steps}',
        f'interaction.num_prefill_episodes={prefill_episodes}',
        'interaction.num_rollouts_per_cycle=10',
        f'interaction.num_samples_per_cycle={samples_per_cycle}',
        f'interaction.random_policy_env_steps={prefill_episodes * 50}',
        'interaction.num_eval_episodes=1',
        'interaction.num_test_episodes=1',
        'eval_steps=null',
        'log_steps=500',
        'save_steps=1000000',
        'keep_only_latest_checkpoint=true',
        'save_replay_buffer=false',
        'save_final_replay_buffer=false',
        'timing.enabled=true',
        'timing.cuda_sync=true',
        'timing.log_jsonl=true',
        'timing.log_top_n=0',
        'timing.reset_after_log=true',
        *variant_optimization_args(variant),
    ]
    return command, result_dir, log_file


def validate_command_configuration(command: list[str]) -> None:
    validation_command = [*command[:3], '--cfg', 'job', '--resolve', *command[3:]]
    completed = subprocess.run(
        validation_command,
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode:
        details = completed.stderr.strip()
        raise RuntimeError(
            'Benchmark configuration preflight failed:\n'
            f'{details or "Hydra exited without an error message"}'
        )


def make_runtime_environment(
        *, gpu: int, output_root: Path,
        cache_namespace: str = 'preflight') -> dict[str, str]:
    environment = command_env(parse_config(QUEUE_CONFIG), str(gpu))
    environment['TORCHINDUCTOR_CACHE_DIR'] = str(
        output_root / f'inductor_cache_gpu{gpu}_{cache_namespace}'
    )
    return environment


def validate_runtime_environment(environment: dict[str, str]) -> None:
    code = '''
import quasimetric_rl.data.online  # register online environments
from quasimetric_rl.data.base import CREATE_ENV_REGISTRY

env = CREATE_ENV_REGISTRY['gcrl', 'FetchSlide']()
try:
    env.reset()
finally:
    env.close()
'''
    completed = subprocess.run(
        [str(VENV_PYTHON), '-c', code],
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            'FetchSlide runtime environment preflight failed:\n'
            f'{completed.stdout.strip()}'
        )


def gpu_state(gpu: int) -> tuple[str, int, int]:
    command = [
        'nvidia-smi',
        f'--id={gpu}',
        '--query-gpu=name,utilization.gpu,memory.used',
        '--format=csv,noheader,nounits',
    ]
    output = subprocess.check_output(command, text=True).strip()
    name, utilization, memory = (field.strip() for field in output.split(','))
    return name, int(utilization), int(memory)


def check_gpus(gpus: tuple[int, int], allow_busy: bool) -> None:
    for gpu in gpus:
        name, utilization, memory = gpu_state(gpu)
        print(
            f'GPU {gpu}: {name}, utilization={utilization}%, memory={memory} MiB',
            flush=True,
        )
        if not allow_busy and (utilization > 20 or memory > 1024):
            raise RuntimeError(
                f'GPU {gpu} is not idle. Re-run after it is free, or pass '
                '--allow-busy if sharing is intentional.'
            )


def parse_timing(
        result_dir: Path,
) -> tuple[float, float, float, float, int, int]:
    timing_file = result_dir / 'timing.jsonl'
    segments = []
    with timing_file.open() as handle:
        for line in handle:
            row = json.loads(line)
            record = row.get('records', {}).get('train/iteration')
            if record and int(record['count']) > 0:
                segments.append(record)
    if not segments:
        raise RuntimeError(f'No train/iteration records in {timing_file}')

    train_seconds = sum(float(record['total_s']) for record in segments)
    first_segment_seconds = float(segments[0]['total_s'])
    steady = segments[1:] if len(segments) > 1 else segments
    steady_seconds = sum(float(record['total_s']) for record in steady)
    steady_batches = sum(int(record['count']) for record in steady)
    steady_seconds_per_batch = steady_seconds / steady_batches
    segment_means = [
        float(record['total_s']) / int(record['count']) for record in steady
    ]
    steady_cv_pct = (
        statistics.pstdev(segment_means) / steady_seconds_per_batch * 100
        if len(segment_means) > 1 and steady_seconds_per_batch > 0 else 0.0
    )
    return (
        train_seconds,
        first_segment_seconds,
        steady_seconds_per_batch,
        steady_cv_pct,
        steady_batches,
        len(segments),
    )


def cleanup_checkpoints(result_dir: Path) -> None:
    for checkpoint in result_dir.glob('*.pth'):
        checkpoint.unlink()


def run_variant(
        *, output_root: Path, gpu: int, variant: str, scale: str,
        total_env_steps: int, prefill_episodes: int,
        samples_per_cycle: int, keep_checkpoints: bool) -> RunResult:
    command, result_dir, log_file = make_command(
        output_root=output_root,
        gpu=gpu,
        variant=variant,
        scale=scale,
        total_env_steps=total_env_steps,
        prefill_episodes=prefill_episodes,
        samples_per_cycle=samples_per_cycle,
    )
    log_file.parent.mkdir(parents=True, exist_ok=True)
    result_dir.parent.mkdir(parents=True, exist_ok=True)
    environment = make_runtime_environment(
        gpu=gpu,
        output_root=output_root,
        cache_namespace=variant,
    )
    inductor_cache = Path(environment['TORCHINDUCTOR_CACHE_DIR'])
    print(f'[GPU {gpu}] START {variant}', flush=True)
    started = time.perf_counter()
    with log_file.open('w') as log:
        print('command: ' + shlex.join(command), file=log, flush=True)
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    wall_seconds = time.perf_counter() - started
    if inductor_cache.exists():
        shutil.rmtree(inductor_cache)

    if completed.returncode != 0:
        error = f'exit code {completed.returncode}; inspect {log_file}'
        print(f'[GPU {gpu}] FAIL  {variant}: {error}', flush=True)
        return RunResult(
            variant, gpu, completed.returncode, wall_seconds,
            None, None, None, None, 0, 0,
            str(result_dir), str(log_file), error,
        )

    try:
        metrics = parse_timing(result_dir)
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        print(f'[GPU {gpu}] FAIL  {variant}: {error}', flush=True)
        return RunResult(
            variant, gpu, 0, wall_seconds,
            None, None, None, None, 0, 0,
            str(result_dir), str(log_file), error,
        )
    if not keep_checkpoints:
        cleanup_checkpoints(result_dir)
    train_s, first_s, steady_s, steady_cv, steady_batches, segments = metrics
    print(
        f'[GPU {gpu}] DONE  {variant}: wall={wall_seconds:.1f}s '
        f'steady={steady_s * 1000:.2f} ms/batch cv={steady_cv:.2f}%',
        flush=True,
    )
    return RunResult(
        variant, gpu, 0, wall_seconds, train_s, first_s, steady_s,
        steady_cv, steady_batches, segments,
        str(result_dir), str(log_file), None,
    )


def run_gpu_variants(
        *, output_root: Path, gpu: int, variants: tuple[str, ...], scale: str,
        total_env_steps: int, prefill_episodes: int,
        samples_per_cycle: int, keep_checkpoints: bool) -> list[RunResult]:
    return [
        run_variant(
            output_root=output_root,
            gpu=gpu,
            variant=variant,
            scale=scale,
            total_env_steps=total_env_steps,
            prefill_episodes=prefill_episodes,
            samples_per_cycle=samples_per_cycle,
            keep_checkpoints=keep_checkpoints,
        )
        for variant in variants
    ]


def run_full_ablation(
        *, output_root: Path, gpus: tuple[int, int], scale: str,
        total_env_steps: int, prefill_episodes: int,
        samples_per_cycle: int, keep_checkpoints: bool) -> list[RunResult]:
    rounds = full_ablation_rounds()
    results = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        for index, variants in enumerate(rounds, 1):
            print(
                f'Round {index}/{len(rounds)}: '
                f'GPU {gpus[0]}={variants[0]}, GPU {gpus[1]}={variants[1]}',
                flush=True,
            )
            futures = [
                executor.submit(
                    run_variant,
                    output_root=output_root,
                    gpu=gpu,
                    variant=variant,
                    scale=scale,
                    total_env_steps=total_env_steps,
                    prefill_episodes=prefill_episodes,
                    samples_per_cycle=samples_per_cycle,
                    keep_checkpoints=keep_checkpoints,
                )
                for gpu, variant in zip(gpus, variants)
            ]
            # Do not start the next pair until both GPUs finish this time block.
            results.extend(future.result() for future in futures)
    return results


def speedup(baseline: float, candidate: Optional[float]) -> Optional[float]:
    if candidate is None or candidate <= 0:
        return None
    return baseline / candidate


def format_speedup(value: Optional[float]) -> str:
    return 'N/A' if value is None else f'{value:.3f}x'


def baseline_metrics(results: list[RunResult]) -> dict[int, BaselineMetrics]:
    anchors: dict[int, list[RunResult]] = {}
    for result in results:
        if (
                result.variant in {'baseline', 'baseline_pre', 'baseline_post'}
                and result.error is None
                and result.steady_seconds_per_batch is not None):
            anchors.setdefault(result.gpu, []).append(result)
    return {
        gpu: BaselineMetrics(
            steady_seconds_per_batch=statistics.fmean(
                result.steady_seconds_per_batch for result in gpu_results
            ),
            wall_seconds=statistics.fmean(
                result.wall_seconds for result in gpu_results
            ),
        )
        for gpu, gpu_results in anchors.items()
    }


def result_speedups(
        result: RunResult, baselines: dict[int, BaselineMetrics],
) -> tuple[Optional[float], Optional[float]]:
    baseline = baselines.get(result.gpu)
    if baseline is None or result.error is not None:
        return None, None
    return (
        speedup(
            baseline.steady_seconds_per_batch,
            result.steady_seconds_per_batch,
        ),
        speedup(baseline.wall_seconds, result.wall_seconds),
    )


def render_report(results: list[RunResult], scale: str, suite: str) -> str:
    baselines = baseline_metrics(results)
    lines = [
        f'# QRL training optimization benchmark ({scale.upper()}/FetchSlide)',
        '',
        '| GPU | Variant | Wall s | Train s | First segment s | '
        'Steady ms/batch | Segment CV | Steady speedup | Wall speedup |',
        '|---:|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for result in sorted(results, key=lambda item: (item.gpu, item.variant)):
        if result.error is not None:
            lines.append(
                f'| {result.gpu} | {result.variant} | {result.wall_seconds:.2f} '
                f'| ERROR | ERROR | ERROR | ERROR | ERROR | ERROR |'
            )
            continue
        steady_ratio, wall_ratio = result_speedups(result, baselines)
        lines.append(
            f'| {result.gpu} | {result.variant} | {result.wall_seconds:.2f} '
            f'| {result.train_seconds:.2f} | {result.first_segment_seconds:.2f} '
            f'| {result.steady_seconds_per_batch * 1000:.3f} '
            f'| {result.steady_cv_pct:.2f}% '
            f'| {format_speedup(steady_ratio)} | {format_speedup(wall_ratio)} |'
        )

    if suite == 'full':
        ranked = []
        for result in results:
            if is_baseline_variant(result.variant) or result.error is not None:
                continue
            steady_ratio, wall_ratio = result_speedups(result, baselines)
            if steady_ratio is not None:
                ranked.append((steady_ratio, wall_ratio, result))
        ranked.sort(key=lambda item: item[0], reverse=True)
        lines.extend([
            '',
            '## Steady-state ranking',
            '',
            '| Rank | Variant | GPU | Steady speedup | Steady ms/batch | '
            'Segment CV | Wall speedup |',
            '|---:|---|---:|---:|---:|---:|---:|',
        ])
        for rank, (steady_ratio, wall_ratio, result) in enumerate(ranked, 1):
            lines.append(
                f'| {rank} | {result.variant} | {result.gpu} '
                f'| {format_speedup(steady_ratio)} '
                f'| {result.steady_seconds_per_batch * 1000:.3f} '
                f'| {result.steady_cv_pct:.2f}% '
                f'| {format_speedup(wall_ratio)} |'
            )

        normalized = {frozenset(): 1.0}
        for steady_ratio, _, result in ranked:
            normalized[variant_features(result.variant)] = steady_ratio
        if len(normalized) == 1 << len(OPTIMIZATION_FEATURES):
            lines.extend([
                '',
                '## Average main effects',
                '',
                '| Feature | Geometric-mean throughput multiplier |',
                '|---|---:|',
            ])
            for feature in OPTIMIZATION_FEATURES:
                ratios = []
                for features, without_speedup in normalized.items():
                    if feature in features:
                        continue
                    with_speedup = normalized[features | {feature}]
                    ratios.append(with_speedup / without_speedup)
                main_effect = math.exp(
                    sum(math.log(ratio) for ratio in ratios) / len(ratios)
                )
                lines.append(f'| {feature} | {main_effect:.3f}x |')

    lines.extend([
        '',
        '`Steady speedup > 1.0x` means faster than the baseline on the same GPU.',
        'For the full suite, each GPU baseline is the arithmetic mean of its '
        '`baseline_pre` and `baseline_post` anchors.',
        'The first timing segment is excluded from the steady-state metric for '
        'every variant.',
        '`Segment CV` is the coefficient of variation across steady timing '
        'segments; lower values indicate less timing noise.',
        'This is a short performance benchmark, not a learning-quality evaluation.',
    ])
    return '\n'.join(lines) + '\n'


def background_command(output_root: Path, argv: list[str]) -> list[str]:
    foreground_args = [arg for arg in argv if arg != '--background']
    has_output_dir = any(
        arg == '--output-dir' or arg.startswith('--output-dir=')
        for arg in foreground_args
    )
    if not has_output_dir:
        foreground_args.extend(('--output-dir', str(output_root)))
    return [str(VENV_PYTHON), str(Path(__file__).resolve()), *foreground_args]


def launch_background(output_root: Path, argv: list[str]) -> int:
    if output_root.exists():
        raise FileExistsError(f'Output directory already exists: {output_root}')
    log_file = Path(f'{output_root}.log')
    pid_file = Path(f'{output_root}.pid')
    if log_file.exists() or pid_file.exists():
        raise FileExistsError(
            f'Background log or PID file already exists for {output_root}'
        )

    environment = os.environ.copy()
    environment['PYTHONUNBUFFERED'] = '1'
    command = background_command(output_root, argv)
    with log_file.open('x') as log_handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pid_file.write_text(f'{process.pid}\n')
    print('Background benchmark started.')
    print(f'PID:        {process.pid}')
    print(f'Output:     {output_root}')
    print(f'Log:        {log_file}')
    print(f'PID file:   {pid_file}')
    print(f'Monitor:    tail -f {shlex.quote(str(log_file))}')
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', nargs=2, type=int, default=(6, 7))
    parser.add_argument(
        '--scale', choices=('l', 'xl', 'xxl', 'xxxl'), default='l',
    )
    parser.add_argument(
        '--suite', choices=('individual', 'full'), default='individual',
    )
    parser.add_argument('--total-env-steps', type=int)
    parser.add_argument('--prefill-episodes', type=int, default=20)
    parser.add_argument('--samples-per-cycle', type=int)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--allow-busy', action='store_true')
    parser.add_argument('--keep-checkpoints', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument(
        '--background', action='store_true',
        help='detach from the terminal and write output to a persistent log',
    )
    args = parser.parse_args()

    if args.background and args.dry_run:
        parser.error('--background and --dry-run cannot be used together')

    if args.total_env_steps is None:
        args.total_env_steps = 3000 if args.suite == 'full' else 2000
    if args.samples_per_cycle is None:
        args.samples_per_cycle = 200 if args.suite == 'full' else 250

    gpus = tuple(args.gpus)
    if gpus[0] == gpus[1]:
        parser.error('--gpus requires two different physical GPU indices')
    if args.total_env_steps <= args.prefill_episodes * 50:
        parser.error('--total-env-steps must exceed prefill transitions')
    if args.samples_per_cycle <= 0:
        parser.error('--samples-per-cycle must be positive')

    output_root = args.output_dir
    if output_root is None:
        stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        output_root = Path('/tmp') / f'qrl-training-opt-benchmark-{stamp}'
    output_root = output_root.resolve()

    if args.background:
        try:
            return launch_background(output_root, sys.argv[1:])
        except (OSError, subprocess.SubprocessError) as exc:
            parser.error(str(exc))

    plans = benchmark_plans(gpus, args.suite)
    prefill_transitions = args.prefill_episodes * 50
    cycles = 1 + math.ceil(
        (args.total_env_steps - prefill_transitions) / 500
    )
    batches_per_run = cycles * args.samples_per_cycle
    steady_batches_per_run = max(1, cycles - 1) * args.samples_per_cycle
    print(
        f'Suite: {args.suite}; runs={sum(len(plan[1]) for plan in plans)}; '
        f'batches/run={batches_per_run}; '
        f'steady_batches/run={steady_batches_per_run}; '
        f'steady_segments={max(1, cycles - 1)}',
        flush=True,
    )
    if args.dry_run:
        print(f'Output directory: {output_root}')
        for gpu, variants in plans:
            for variant in variants:
                command, _, _ = make_command(
                    output_root=output_root,
                    gpu=gpu,
                    variant=variant,
                    scale=args.scale,
                    total_env_steps=args.total_env_steps,
                    prefill_episodes=args.prefill_episodes,
                    samples_per_cycle=args.samples_per_cycle,
                )
                print(f'CUDA_VISIBLE_DEVICES={gpu} {shlex.join(command)}')
        return 0

    preflight_command, _, _ = make_command(
        output_root=output_root,
        gpu=gpus[0],
        variant='baseline',
        scale=args.scale,
        total_env_steps=args.total_env_steps,
        prefill_episodes=args.prefill_episodes,
        samples_per_cycle=args.samples_per_cycle,
    )
    validate_command_configuration(preflight_command)
    print('Configuration preflight: PASS', flush=True)
    validate_runtime_environment(make_runtime_environment(
        gpu=gpus[0], output_root=output_root,
    ))
    print('FetchSlide runtime preflight: PASS', flush=True)
    check_gpus(gpus, args.allow_busy)
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / 'results').mkdir()
    print(f'Output directory: {output_root}', flush=True)
    if args.suite == 'full':
        results = run_full_ablation(
            output_root=output_root,
            gpus=gpus,
            scale=args.scale,
            total_env_steps=args.total_env_steps,
            prefill_episodes=args.prefill_episodes,
            samples_per_cycle=args.samples_per_cycle,
            keep_checkpoints=args.keep_checkpoints,
        )
    else:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    run_gpu_variants,
                    output_root=output_root,
                    gpu=gpu,
                    variants=variants,
                    scale=args.scale,
                    total_env_steps=args.total_env_steps,
                    prefill_episodes=args.prefill_episodes,
                    samples_per_cycle=args.samples_per_cycle,
                    keep_checkpoints=args.keep_checkpoints,
                )
                for gpu, variants in plans
            ]
            results = [
                result for future in futures for result in future.result()
            ]

    report = render_report(results, args.scale, args.suite)
    (output_root / 'results.json').write_text(
        json.dumps([asdict(result) for result in results], indent=2) + '\n'
    )
    (output_root / 'report.md').write_text(report)
    print()
    print(report, end='')
    print(f'Full report: {output_root / "report.md"}')
    return int(any(result.error is not None for result in results))


if __name__ == '__main__':
    raise SystemExit(main())

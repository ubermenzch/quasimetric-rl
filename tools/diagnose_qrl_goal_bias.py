#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import importlib.util
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from d4rl_runtime import configure_d4rl_runtime


ROOT = Path(__file__).resolve().parents[1]
SCALING_CRL_ROOT = Path(os.environ.get("SCALING_CRL_ROOT", ROOT.parent / "scaling-crl"))
SCALING_SCRIPT = SCALING_CRL_ROOT / "tools/diagnose_qrl_goal_bias.py"
COVERAGE_LEVELS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95)


def configure_environment() -> None:
    sys.path.insert(0, str(ROOT))
    configure_d4rl_runtime(ROOT, require_library_paths=False)


def load_scaling_diagnostic():
    if not SCALING_SCRIPT.is_file():
        raise FileNotFoundError(
            "Missing cross-project diagnostic dependency: set SCALING_CRL_ROOT "
            "to a checkout containing tools/diagnose_qrl_goal_bias.py"
        )
    spec = importlib.util.spec_from_file_location("scaling_qrl_goal_bias", SCALING_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load diagnostic script: {SCALING_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resolve_cpu_workers(value: int, num_pairs: int) -> int:
    if value > 0:
        return value
    cpu_count = os.cpu_count() or 1
    return max(1, min(cpu_count, num_pairs, 8))


def find_real_matches(
    coords: np.ndarray,
    target_coords: np.ndarray,
    radius: float,
    workers: int,
) -> list[np.ndarray]:
    radius_sq = float(radius) * float(radius)

    def find_one(coord: np.ndarray) -> np.ndarray:
        diff = coords - coord
        sq_dist = np.einsum("ij,ij->i", diff, diff, optimize=True)
        return np.flatnonzero(sq_dist <= radius_sq)

    if workers <= 1 or len(target_coords) <= 1:
        return [find_one(coord) for coord in target_coords]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(find_one, target_coords))


def find_real_matches_gpu(
    coords: np.ndarray,
    target_coords: np.ndarray,
    radius: float,
    device: torch.device,
    batch_size: int,
) -> list[np.ndarray]:
    coords_t = torch.from_numpy(np.ascontiguousarray(coords)).to(device)
    targets_t = torch.from_numpy(np.ascontiguousarray(target_coords))
    radius_sq = float(radius) * float(radius)
    matches: list[np.ndarray] = []
    with torch.inference_mode():
        coords_norm = (coords_t * coords_t).sum(dim=1)
        for start in range(0, targets_t.shape[0], batch_size):
            end = min(start + batch_size, targets_t.shape[0])
            target_batch = targets_t[start:end].to(device)
            target_norm = (target_batch * target_batch).sum(dim=1)
            sq_dist = target_norm[:, None] + coords_norm[None, :] - 2.0 * (target_batch @ coords_t.T)
            sq_dist.clamp_(min=0.0)
            mask = sq_dist <= radius_sq
            for row in mask:
                matches.append(torch.nonzero(row, as_tuple=False).flatten().cpu().numpy())
    return matches


def finite_random_bounds(
    low: np.ndarray,
    high: np.ndarray,
    states_np: np.ndarray,
    target_dims: list[int],
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    low = low.astype(np.float32, copy=True)
    high = high.astype(np.float32, copy=True)
    target_set = set(target_dims)
    fixed_dims = []
    for dim in range(low.shape[0]):
        if dim in target_set:
            continue
        if not np.isfinite(low[dim]) or not np.isfinite(high[dim]):
            low[dim] = np.nanmin(states_np[:, dim]).astype(np.float32)
            high[dim] = np.nanmax(states_np[:, dim]).astype(np.float32)
            fixed_dims.append(dim)
        if high[dim] <= low[dim]:
            high[dim] = low[dim] + np.float32(1e-6)
    return low, high, fixed_dims


def gaussian_kl(mean_p: float, std_p: float, mean_q: float, std_q: float, eps: float) -> float:
    var_p = max(float(std_p) * float(std_p), eps)
    var_q = max(float(std_q) * float(std_q), eps)
    mean_delta = float(mean_q) - float(mean_p)
    return 0.5 * (var_p / var_q + mean_delta * mean_delta / var_q - 1.0 + np.log(var_q / var_p))


def gaussian_w2(mean_p: float, std_p: float, mean_q: float, std_q: float) -> float:
    mean_delta = float(mean_p) - float(mean_q)
    std_delta = float(std_p) - float(std_q)
    return float(np.sqrt(mean_delta * mean_delta + std_delta * std_delta))


def distribution_comparison_metrics(
    d_zero: torch.Tensor,
    d_real: torch.Tensor,
    d_random: torch.Tensor,
    real_stats: dict[str, float],
    rand_stats: dict[str, float],
    eps: float,
) -> dict[str, float]:
    real_mean = real_stats["mean"]
    real_std = real_stats["std"]
    rand_mean = rand_stats["mean"]
    rand_std = rand_stats["std"]
    zero_value = d_zero.item()

    kl_random_real = gaussian_kl(rand_mean, rand_std, real_mean, real_std, eps)
    kl_real_random = gaussian_kl(real_mean, real_std, rand_mean, rand_std, eps)
    out = {
        "kl_random_real": kl_random_real,
        "kl_real_random": kl_real_random,
        "sym_kl_random_real": 0.5 * (kl_random_real + kl_real_random),
        "w2_random_real": gaussian_w2(rand_mean, rand_std, real_mean, real_std),
        "zero_signed_z": (zero_value - real_mean) / (real_std + eps),
        "zero_abs_z": abs(zero_value - real_mean) / (real_std + eps),
        "random_mean_signed_z": (rand_mean - real_mean) / (real_std + eps),
        "random_mean_abs_z": abs(rand_mean - real_mean) / (real_std + eps),
    }

    real_values = d_real.to(torch.float64)
    random_values = d_random.to(torch.float64)
    for level in COVERAGE_LEVELS:
        tail = (1.0 - level) / 2.0
        low = real_values.quantile(tail).item()
        high = real_values.quantile(1.0 - tail).item()
        key = int(round(level * 100))
        out[f"real_central_p{key}_low"] = low
        out[f"real_central_p{key}_high"] = high
        out[f"zero_in_real_p{key}"] = float(low <= zero_value <= high)
        out[f"random_in_real_p{key}_frac"] = (
            ((random_values >= low) & (random_values <= high)).to(torch.float64).mean().item()
        )
    return out


def make_random_completed_goals_np(
    coord: np.ndarray,
    obs_dim: int,
    target_dims: list[int],
    low: np.ndarray,
    high: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    samples = np.empty((count, obs_dim), dtype=np.float32)
    target_set = set(target_dims)
    non_target_dims = [i for i in range(obs_dim) if i not in target_set]
    if non_target_dims:
        samples[:, non_target_dims] = rng.uniform(
            low=low[non_target_dims],
            high=high[non_target_dims],
            size=(count, len(non_target_dims)),
        ).astype(np.float32)

    samples[:, target_dims] = coord.astype(np.float32)
    return torch.from_numpy(samples)


def encode_goal_distances(
    agent,
    unique_anchors: torch.Tensor,
    goals: torch.Tensor,
    anchor_ids: torch.Tensor,
    batch_size: int,
    reduction: str,
    device: torch.device,
) -> torch.Tensor:
    critics = list(agent.critics)
    out = []
    unique_anchors = unique_anchors.to(device, non_blocking=True)
    anchor_ids = anchor_ids.to(device, non_blocking=True)

    with torch.inference_mode():
        anchor_embeddings = [critic.encoder(unique_anchors) for critic in critics]
        for start in range(0, goals.shape[0], batch_size):
            end = min(start + batch_size, goals.shape[0])
            goal_batch = goals[start:end].to(device, non_blocking=True)
            ids = anchor_ids[start:end]
            dists = []
            for critic, encoded_anchors in zip(critics, anchor_embeddings):
                anchor_batch = encoded_anchors.index_select(0, ids)
                goal_embeddings = critic.encoder(goal_batch)
                dists.append(critic.quasimetric_model(anchor_batch, goal_embeddings))
            stacked = torch.stack(dists, dim=0)
            if reduction == "first":
                reduced = stacked[0]
            elif reduction == "mean":
                reduced = stacked.mean(dim=0)
            elif reduction == "max":
                reduced = stacked.max(dim=0).values
            else:
                raise ValueError(reduction)
            out.append(reduced.detach().cpu())
    return torch.cat(out, dim=0)


def optimized_run(module, args: argparse.Namespace) -> dict:
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    result_dir = Path(args.result_dir).resolve()
    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else None
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    cfg, dataset, agent, checkpoint = module.load_agent_and_dataset(result_dir, checkpoint, device)
    states = dataset.raw_data.all_observations.to(torch.float32).cpu()
    if states.ndim != 2:
        raise RuntimeError(f"only flat observations are supported, got states shape={tuple(states.shape)}")
    obs_dim = states.shape[1]
    target_dims = args.target_dims
    if any(d < 0 or d >= obs_dim for d in target_dims):
        raise ValueError(f"target dims {target_dims} out of bounds for obs_dim={obs_dim}")

    states_np = states.numpy()
    coords_np = states_np[:, target_dims].astype(np.float32, copy=False)
    low, high = module.observation_bounds(dataset, args.random_low, args.random_high)
    low, high, empirical_bound_dims = finite_random_bounds(low, high, states_np, target_dims)
    module.validate_random_bounds(low, high, target_dims)
    n_states = states_np.shape[0]
    target_set = set(target_dims)
    non_target_dims = [i for i in range(obs_dim) if i not in target_set]

    anchor_indices = np.empty(args.num_pairs, dtype=np.int64)
    target_indices = np.empty(args.num_pairs, dtype=np.int64)
    for pair_id in range(args.num_pairs):
        anchor_indices[pair_id] = int(rng.integers(0, n_states))
        target_indices[pair_id] = int(rng.integers(0, n_states))

    target_coords = coords_np[target_indices]
    cpu_workers = resolve_cpu_workers(args.cpu_workers, args.num_pairs)
    if device.type == "cuda":
        real_matches = find_real_matches_gpu(
            coords_np,
            target_coords,
            args.goal_radius,
            device,
            args.match_batch_size,
        )
        match_backend = "gpu"
    else:
        real_matches = find_real_matches(coords_np, target_coords, args.goal_radius, cpu_workers)
        match_backend = "cpu_threads"

    rows = []
    skipped = 0
    goals_parts = []
    anchor_ids_parts = []
    unique_anchors = []
    slices = []

    for pair_id, (anchor_idx, target_idx, coord, real_indices) in enumerate(
        zip(anchor_indices, target_indices, target_coords, real_matches)
    ):
        if real_indices.shape[0] < args.min_real_matches:
            skipped += 1
            continue
        if real_indices.shape[0] > args.real_samples:
            real_indices = rng.choice(real_indices, size=args.real_samples, replace=False)

        canonical = torch.zeros(obs_dim, dtype=torch.float32)
        canonical[target_dims] = torch.from_numpy(coord)
        real_goals = states[torch.from_numpy(real_indices).long()]
        random_goals = make_random_completed_goals_np(
            coord=coord,
            obs_dim=obs_dim,
            target_dims=target_dims,
            low=low,
            high=high,
            count=args.random_samples,
            rng=rng,
        )

        pair_goals = torch.cat([canonical[None], real_goals, random_goals], dim=0)
        anchor_id = len(unique_anchors)
        unique_anchors.append(states[int(anchor_idx)])
        goals_parts.append(pair_goals)
        anchor_ids_parts.append(torch.full((pair_goals.shape[0],), anchor_id, dtype=torch.long))
        slices.append((len(rows), pair_id, int(anchor_idx), int(target_idx), coord, int(real_indices.shape[0]), pair_goals.shape[0], real_goals.shape[0]))

        row = {
            "pair_id": pair_id,
            "anchor_index": int(anchor_idx),
            "target_index": int(target_idx),
            "n_real_matches": int(real_indices.shape[0]),
        }
        for i, dim in enumerate(target_dims):
            row[f"target_dim_{dim}"] = float(coord[i])
        rows.append(row)

    if not rows:
        raise RuntimeError(
            f"no valid pairs found; skipped={skipped}. Increase --goal-radius or lower --min-real-matches."
        )

    goals = torch.cat(goals_parts, dim=0).pin_memory() if device.type == "cuda" else torch.cat(goals_parts, dim=0)
    anchor_ids = torch.cat(anchor_ids_parts, dim=0).pin_memory() if device.type == "cuda" else torch.cat(anchor_ids_parts, dim=0)
    anchors = torch.stack(unique_anchors, dim=0).pin_memory() if device.type == "cuda" else torch.stack(unique_anchors, dim=0)
    dists = encode_goal_distances(
        agent=agent,
        unique_anchors=anchors,
        goals=goals,
        anchor_ids=anchor_ids,
        batch_size=args.distance_batch_size,
        reduction=args.critic_reduction,
        device=device,
    )

    offset = 0
    for row_index, _pair_id, _anchor_idx, _target_idx, _coord, _n_real, goal_count, real_count in slices:
        pair_dists = dists[offset:offset + goal_count]
        offset += goal_count

        d_can = pair_dists[0]
        d_real = pair_dists[1:1 + real_count]
        d_rand = pair_dists[1 + real_count:]

        real_stats = module.summarize(d_real)
        rand_stats = module.summarize(d_rand)
        real_std = real_stats["std"]
        z_can = abs(d_can.item() - real_stats["mean"]) / (real_std + args.z_eps)
        z_rand = abs(rand_stats["mean"] - real_stats["mean"]) / (real_std + args.z_eps)
        distribution_metrics = distribution_comparison_metrics(
            d_zero=d_can,
            d_real=d_real,
            d_random=d_rand,
            real_stats=real_stats,
            rand_stats=rand_stats,
            eps=args.z_eps,
        )

        rows[row_index].update({
            "d_canonical": d_can.item(),
            "d_real_mean": real_stats["mean"],
            "d_real_std": real_stats["std"],
            "d_real_min": real_stats["min"],
            "d_real_max": real_stats["max"],
            "d_random_mean": rand_stats["mean"],
            "d_random_std": rand_stats["std"],
            "d_random_min": rand_stats["min"],
            "d_random_max": rand_stats["max"],
            "canonical_minus_real_mean": d_can.item() - real_stats["mean"],
            "random_minus_real_mean": rand_stats["mean"] - real_stats["mean"],
            "canonical_abs_z": z_can,
            "random_abs_z": z_rand,
            "canonical_minus_random_mean": d_can.item() - rand_stats["mean"],
        })
        rows[row_index].update(distribution_metrics)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix or f"{result_dir.name}_goal_bias"
    csv_path = output_dir / f"{prefix}.csv"
    json_path = output_dir / f"{prefix}_summary.json"

    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    def col(name: str) -> torch.Tensor:
        return torch.tensor([r[name] for r in rows], dtype=torch.float64)

    distribution_metric_names = [
        "kl_random_real",
        "kl_real_random",
        "sym_kl_random_real",
        "w2_random_real",
        "zero_signed_z",
        "zero_abs_z",
        "random_mean_signed_z",
        "random_mean_abs_z",
    ]
    coverage_metrics = {}
    for level in COVERAGE_LEVELS:
        key = int(round(level * 100))
        coverage_metrics[f"zero_in_real_p{key}_rate"] = col(f"zero_in_real_p{key}").mean().item()
        coverage_metrics[f"random_in_real_p{key}_mean"] = col(f"random_in_real_p{key}_frac").mean().item()
        coverage_metrics[f"random_in_real_p{key}_std"] = col(f"random_in_real_p{key}_frac").std(unbiased=False).item()
        coverage_metrics[f"real_central_p{key}_low"] = module.summarize(col(f"real_central_p{key}_low"))
        coverage_metrics[f"real_central_p{key}_high"] = module.summarize(col(f"real_central_p{key}_high"))

    summary = {
        "result_dir": str(result_dir),
        "checkpoint": str(checkpoint),
        "env_kind": cfg.env.kind,
        "env_name": cfg.env.name,
        "obs_dim": obs_dim,
        "target_dims": target_dims,
        "non_target_dims": non_target_dims,
        "goal_radius": args.goal_radius,
        "critic_reduction": args.critic_reduction,
        "num_pairs_requested": args.num_pairs,
        "num_pairs_used": len(rows),
        "num_pairs_skipped": skipped,
        "real_samples": args.real_samples,
        "random_samples": args.random_samples,
        "random_low": low.tolist(),
        "random_high": high.tolist(),
        "random_bounds_source": {
            "default": "observation_space",
            "empirical_dataset_dims": empirical_bound_dims,
            "note": "Non-finite non-target observation bounds are replaced with dataset min/max for random completion.",
        },
        "random_target_sampling": {
            "mode": "fixed_target_coord",
            "note": "Random goals use the exact sampled target coordinates; only non-target dimensions are randomly completed.",
        },
        "performance": {
            "implementation": "batched_gpu_distances_unique_anchor_encoding",
            "cpu_workers": cpu_workers,
            "torch_threads": torch.get_num_threads(),
            "distance_batch_size": args.distance_batch_size,
            "match_backend": match_backend,
            "match_batch_size": args.match_batch_size,
            "total_distance_pairs": int(goals.shape[0]),
        },
        "metrics": {
            "d_canonical": module.summarize(col("d_canonical")),
            "d_real_mean": module.summarize(col("d_real_mean")),
            "d_real_std": module.summarize(col("d_real_std")),
            "d_random_mean": module.summarize(col("d_random_mean")),
            "d_random_std": module.summarize(col("d_random_std")),
            "canonical_minus_real_mean": module.summarize(col("canonical_minus_real_mean")),
            "random_minus_real_mean": module.summarize(col("random_minus_real_mean")),
            "canonical_abs_z": module.summarize(col("canonical_abs_z")),
            "random_abs_z": module.summarize(col("random_abs_z")),
            "canonical_minus_random_mean": module.summarize(col("canonical_minus_random_mean")),
            **{name: module.summarize(col(name)) for name in distribution_metric_names},
        },
        "coverage": coverage_metrics,
        "csv_path": str(csv_path),
        "summary_path": str(json_path),
    }
    with json_path.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return summary


def parse_args(module) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure whether zero-filled canonical goals are geometrically biased relative to real states "
            "that satisfy the same goal condition. This local wrapper batches critic distance computation on GPU."
        )
    )
    parser.add_argument("--result-dir", required=True, help="Official QRL result directory containing config.yaml.")
    parser.add_argument("--checkpoint", default="", help="Checkpoint path. Defaults to latest checkpoint_*.pth in result dir.")
    parser.add_argument("--target-dims", type=module.parse_dims, default=module.parse_dims("0,1"), help="Comma-separated goal-relevant dims.")
    parser.add_argument("--goal-radius", type=float, default=0.5, help="Euclidean radius in target-dim space for real matches.")
    parser.add_argument("--num-pairs", type=int, default=2048)
    parser.add_argument("--real-samples", type=int, default=1024)
    parser.add_argument("--random-samples", type=int, default=1024)
    parser.add_argument("--min-real-matches", type=int, default=16)
    parser.add_argument("--critic-reduction", choices=["first", "mean", "max"], default="max")
    parser.add_argument("--distance-batch-size", type=int, default=65536)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--random-low", default=None, help="Explicit full-observation low bounds, comma-separated.")
    parser.add_argument("--random-high", default=None, help="Explicit full-observation high bounds, comma-separated.")
    parser.add_argument("--z-eps", type=float, default=1e-6)
    parser.add_argument("--output-dir", default="analysis/goal_bias")
    parser.add_argument("--output-prefix", default="")
    parser.add_argument("--cpu-workers", type=int, default=int(os.environ.get("GOAL_BIAS_CPU_WORKERS", "0")), help="CPU workers for real-goal matching. 0 chooses a bounded automatic value.")
    parser.add_argument("--torch-threads", type=int, default=int(os.environ.get("GOAL_BIAS_TORCH_THREADS", "0")), help="Torch CPU threads. 0 leaves PyTorch default unchanged.")
    parser.add_argument("--match-batch-size", type=int, default=32, help="Target-coordinate batch size for GPU real-goal matching.")
    return parser.parse_args()


def main() -> None:
    if not SCALING_SCRIPT.exists():
        raise FileNotFoundError(f"missing diagnostic script: {SCALING_SCRIPT}")
    configure_environment()
    module = load_scaling_diagnostic()
    args = parse_args(module)
    summary = optimized_run(module, args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

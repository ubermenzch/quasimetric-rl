#!/usr/bin/env python3
"""Audit existing dynamics ablations without modifying training artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONTROL_SLUGS = ("fetchpush", "fetchslide", "fetchpickandplace", "dmc_reacher_hard")
ENVIRONMENTS = (
    "FetchPush", "FetchSlide", "FetchPickAndPlace", "reacher_hard",
    "Pusher-v4", "AntNavigate-v4", "maze2d-large", "swimmer6", "manipulator_bring_ball",
)
VARIANTS = {
    f"A{i + 1:02d}": (separate, distance, activation)
    for i, (separate, distance, activation) in enumerate(
        (s, d, a)
        for s in (False, True)
        for d in ("iqe", "mse", "iqe_mse")
        for a in ("relu", "leaky_relu")
    )
}
SCALAR_TAGS = {
    "mse": "train_critic_00/latent_dynamics/mse",
    "sq_dists": "train_critic_00/latent_dynamics/sq_dists",
    "dist_p2n": "train_critic_00/latent_dynamics/dist_p2n",
    "dist_n2p": "train_critic_00/latent_dynamics/dist_n2p",
    "goal_std": "train_actor/min_dist/goal_latent_std_mean_00",
    "non_goal_std": "train_actor/min_dist/non_goal_latent_std_mean_00",
    "goal_norm": "train_actor/min_dist/goal_latent_norm_mean_00",
    "non_goal_norm": "train_actor/min_dist/non_goal_latent_norm_mean_00",
    "inner_gradient_norm": "train_actor/min_dist/latent_goal_gradient_norm_00",
}


def json_lines(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def flatten(value, prefix=""):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            result.update(flatten(item, f"{prefix}.{key}" if prefix else key))
        return result
    return {prefix: value}


def protocol(config):
    # Compare active training/evaluation configuration, excluding bookkeeping.
    keys = ("agent", "env", "interaction", "batch_size", "eval_steps", "save_steps")
    result = flatten({key: config[key] for key in keys})
    result = {key: value for key, value in result.items()
              if not key.startswith("agent.baselines.")}
    for key in (
        "agent.quasimetric_critic.losses.latent_dynamics.distance",
        "agent.quasimetric_critic.losses.latent_dynamics.mse_weight",
        "agent.quasimetric_critic.losses.latent_dynamics.iqe_weight",
    ):
        result.pop(key, None)
    return result


def load_scalar_events(run):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    merged = defaultdict(dict)
    for path in sorted(Path(run["run_dir"]).glob("events.out.tfevents.*")):
        accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
        accumulator.Reload()
        available = set(accumulator.Tags().get("scalars", []))
        for name, tag in SCALAR_TAGS.items():
            if tag not in available:
                continue
            for event in accumulator.Scalars(tag):
                if event.step > run["completed_steps"]:
                    continue
                assert math.isfinite(event.value), (path, tag, event.step)
                previous = merged[name].get(event.step)
                if previous is None or event.wall_time >= previous.wall_time:
                    merged[name][event.step] = event
    return merged


def scalar_summary(run):
    merged = load_scalar_events(run)
    result = {}
    for name in SCALAR_TAGS:
        events = sorted(merged[name].values(), key=lambda event: event.step)
        final = [event.value for event in events
                 if event.step >= 0.9 * run["completed_steps"]]
        selected = [event.value for event in events
                    if run["selected_steps"] - 20000 < event.step <= run["selected_steps"]]
        result[f"{name}_records"] = len(events)
        result[f"{name}_last_step"] = events[-1].step if events else ""
        result[f"{name}_last10pct"] = statistics.mean(final) if final else ""
        result[f"{name}_selected20k"] = statistics.mean(selected) if selected else ""
    return result


def ant_trajectories(rows):
    trajectories = []
    for row in rows:
        if row["environment"] != "AntNavigate-v4":
            continue
        merged = load_scalar_events(row)
        for step in range(20000, row["completed_steps"] + 1, 20000):
            record = {key: row[key] for key in ("environment", "variant", "seed")}
            record["env_steps"] = step
            for name in SCALAR_TAGS:
                window = [event.value for event in merged[name].values()
                          if step - 20000 < event.step <= step]
                record[name] = statistics.mean(window) if window else ""
            trajectories.append(record)
    return trajectories


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT.parent / "qrl-assets/results/queue")
    parser.add_argument("--output", type=Path, default=ROOT / "analysis/dynamics_mse")
    parser.add_argument("--scalars", action="store_true")
    parser.add_argument("--ant-trajectories", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    candidates = {path: path.name.split("dynfac_", 1)[1].split("-", 1)[0]
                  for path in args.results.glob("*dynfac_A??-*") if path.is_dir()}
    for slug in CONTROL_SLUGS:
        for seed in range(1000, 1003):
            for path in args.results.glob(
                f"official_qrl_1q_GO-QRL+Max4-M_200k_20kckpt_*_{slug}_online_s{seed}"
            ):
                candidates[path] = "A01"

    rows, configs, skipped = [], {}, []
    for path, variant in sorted(candidates.items()):
        if not (path / "test.log").exists() or not (path / "COMPLETE").exists():
            skipped.append(str(path))
            continue
        config = yaml.safe_load((path / "config.yaml").read_text())
        tests = json_lines(path / "test.log")
        assert len(tests) == 1, (path, len(tests))
        test = tests[0]
        losses = config["agent"]["quasimetric_critic"]["losses"]
        model = config["agent"]["quasimetric_critic"]["model"]
        distance = losses["latent_dynamics"].get("distance", "iqe")
        separate = losses.get("separate_latent_dynamics", False)
        activation = model["quasimetric_model"].get("projector_activation", "relu")
        assert (separate, distance, activation) == VARIANTS[variant], path
        assert losses["latent_dynamics"]["weight"] == 0.1, path
        if distance == "iqe_mse":
            assert losses["latent_dynamics"]["mse_weight"] == 1, path
            assert losses["latent_dynamics"]["iqe_weight"] == 1, path
        assert test["completed_training_env_steps"] == config["interaction"]["total_env_steps"], path
        assert test["seed"] == config["interaction"]["test_seed"], path
        assert test["num_episodes"] == config["interaction"]["num_test_episodes"], path
        evaluations = {row["env_steps"]: row for row in json_lines(path / "eval.log")}
        curve = [evaluations[step] for step in sorted(evaluations)
                 if step <= test["completed_training_env_steps"]]
        selected = evaluations[test["env_steps"]]
        assert selected["succ_rate"] == max(row["succ_rate"] for row in curve), path
        assert selected["seed_end"] < test["seed"], path
        row = {
            "environment": config["env"]["name"], "variant": variant,
            "distance": distance, "separate": separate, "activation": activation,
            "seed": config["seed"], "completed_steps": test["completed_training_env_steps"],
            "selected_steps": test["env_steps"], "test_episodes": test["num_episodes"],
            "test_success": test["succ_rate"], "test_hitting_time": test["hitting_time"],
            "validation_selected": selected["succ_rate"],
            "validation_final": curve[-1]["succ_rate"],
            "validation_checkpoint_mean": statistics.mean(item["succ_rate"] for item in curve),
            "git_commit": config.get("git_commit", ""), "run_dir": str(path),
        }
        rows.append(row)
        configs[(row["environment"], variant, row["seed"])] = protocol(config)

    groups = defaultdict(list)
    lookup = {}
    for row in rows:
        groups[(row["environment"], row["variant"])].append(row)
        key = (row["environment"], row["variant"], row["seed"])
        assert key not in lookup, key
        lookup[key] = row
    aggregate = []
    for (env, variant), group in sorted(groups.items()):
        values = [row["test_success"] for row in group]
        aggregate.append({
            "environment": env, "variant": variant, "n": len(group),
            "test_success_mean": statistics.mean(values),
            "test_success_std": statistics.stdev(values) if len(values) > 1 else "",
            "seed_successes": json.dumps({row["seed"]: row["test_success"] for row in group}),
            "validation_final_mean": statistics.mean(row["validation_final"] for row in group),
            "validation_checkpoint_mean": statistics.mean(row["validation_checkpoint_mean"] for row in group),
        })

    differences, comparisons = [], []
    for row in rows:
        if row["distance"] != "iqe_mse":
            continue
        number = int(row["variant"][1:])
        for offset in (4, 2):
            baseline_variant = f"A{number - offset:02d}"
            key = (row["environment"], baseline_variant, row["seed"])
            if key not in lookup:
                continue
            baseline = lookup[key]
            hybrid_config = configs[(row["environment"], row["variant"], row["seed"])]
            baseline_config = configs[key]
            diff = {name: [baseline_config.get(name), hybrid_config.get(name)]
                    for name in baseline_config.keys() | hybrid_config.keys()
                    if baseline_config.get(name) != hybrid_config.get(name)}
            if diff:
                differences.append({"environment": row["environment"], "seed": row["seed"],
                                    "baseline": baseline_variant, "hybrid": row["variant"],
                                    "differences": diff})
            comparisons.append({
                "environment": row["environment"], "seed": row["seed"],
                "baseline": baseline_variant, "hybrid": row["variant"],
                "baseline_success": baseline["test_success"], "hybrid_success": row["test_success"],
                "delta_pp": 100 * (row["test_success"] - baseline["test_success"]),
                "config_difference_count": len(diff),
            })

    audit = {
        "results_root": str(args.results), "runs": len(rows),
        "counts_by_variant": dict(sorted(Counter(row["variant"] for row in rows).items())),
        "environments": sorted({row["environment"] for row in rows}),
        "incomplete_local_runs": skipped, "paired_config_differences": differences,
        "expected_factorial_runs": 324,
        "missing_factorial_cells": [
            {"environment": env, "variant": variant, "seed": seed}
            for env in ENVIRONMENTS
            for variant in VARIANTS for seed in range(1000, 1003)
            if (env, variant, seed) not in lookup
        ],
    }
    write_csv(args.output / "runs.csv", rows)
    write_csv(args.output / "aggregate.csv", aggregate)
    write_csv(args.output / "paired_comparisons.csv", comparisons)
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({key: audit[key] for key in ("runs", "counts_by_variant", "environments")}), flush=True)
    print(f"Paired configuration differences: {len(differences)}", flush=True)
    for row in aggregate:
        print(f"{row['environment']:25s} {row['variant']} n={row['n']} "
              f"{100 * row['test_success_mean']:6.2f} +/- {100 * row['test_success_std']:6.2f}")
    if args.scalars:
        scalar_rows = []
        for i, row in enumerate(rows):
            scalar_rows.append({**row, **scalar_summary(row)})
            if (i + 1) % 10 == 0:
                print(f"Scalar audit: {i + 1}/{len(rows)}", flush=True)
        write_csv(args.output / "training_scalars.csv", scalar_rows)
    if args.ant_trajectories:
        write_csv(args.output / "ant_trajectories.csv", ant_trajectories(rows))


if __name__ == "__main__":
    main()

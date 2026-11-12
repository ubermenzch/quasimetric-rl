#!/usr/bin/env python3
"""Audit TensorBoard training scalars for the NQRL paper run matrix.

Run this script with the repository virtual environment, which contains
TensorBoard.  It does not alter training artifacts.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from analyze_nqrl_paper_results import (
    DATA_ROOT,
    ENVIRONMENTS,
    METHODS,
    SEEDS,
    artifact_dir,
    load_json_lines,
)


def load_scalars(run_dir):
    merged = {}
    parse_errors = []
    nonfinite = []
    event_file_count = 0
    train_data_scalar_count = 0
    for event_path in sorted(run_dir.glob("events.out.tfevents.*")):
        event_file_count += 1
        try:
            accumulator = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
            accumulator.Reload()
        except Exception as error:  # pragma: no cover - diagnostic path
            parse_errors.append(f"{event_path}: {error}")
            continue
        for tag in accumulator.Tags().get("scalars", []):
            events = accumulator.Scalars(tag)
            if tag.startswith("train") or tag.startswith("data/"):
                train_data_scalar_count += len(events)
                for event in events:
                    if not math.isfinite(float(event.value)):
                        nonfinite.append(
                            {
                                "event_file": str(event_path),
                                "tag": tag,
                                "step": int(event.step),
                            }
                        )
            for event in events:
                key = (tag, int(event.step))
                previous = merged.get(key)
                if previous is None or event.wall_time >= previous.wall_time:
                    merged[key] = event
    by_tag = defaultdict(list)
    for (tag, _), event in merged.items():
        by_tag[tag].append(event)
    for events in by_tag.values():
        events.sort(key=lambda event: (event.step, event.wall_time))
    return (
        by_tag,
        parse_errors,
        nonfinite,
        event_file_count,
        train_data_scalar_count,
    )


def window_mean(events, *, first: bool):
    if not events:
        return ""
    window = events[:20] if first else events[-20:]
    return statistics.mean(float(event.value) for event in window)


def main():
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    nonfinite = []
    parse_errors = []
    missing = []
    incomplete = []
    test_log_issues = []
    budget_mismatches = []
    event_file_count = 0
    train_data_scalar_count = 0
    for env in ENVIRONMENTS:
        for method in METHODS:
            for seed in SEEDS:
                run_dir = artifact_dir(env, method, seed)
                if not run_dir.exists():
                    missing.append(str(run_dir))
                    continue
                if not (run_dir / "COMPLETE").is_file():
                    incomplete.append(str(run_dir))
                test_path = run_dir / "test.log"
                test_rows = load_json_lines(test_path) if test_path.is_file() else []
                if len(test_rows) != 1:
                    test_log_issues.append(
                        {"run": str(run_dir), "test_rows": len(test_rows)}
                    )
                    continue
                test = test_rows[0]
                completed_env_steps = int(test["completed_training_env_steps"])
                if completed_env_steps != env.budget:
                    budget_mismatches.append(
                        {
                            "run": str(run_dir),
                            "completed_env_steps": completed_env_steps,
                            "expected_env_steps": env.budget,
                        }
                    )

                by_tag, errors, bad_values, num_files, num_values = load_scalars(run_dir)
                parse_errors.extend(errors)
                nonfinite.extend(bad_values)
                event_file_count += num_files
                train_data_scalar_count += num_values

                successes = by_tag.get("data/num_successes", [])
                transitions = by_tag.get("data/num_regular_transitions", [])
                iter_times = by_tag.get("train/iter_time", [])
                validation_success = by_tag.get("validation/succ_rate", [])
                if not successes or not transitions or not iter_times or not validation_success:
                    raise ValueError(f"Required audit scalars missing from {run_dir}")
                success_fraction = float(successes[-1].value) / float(transitions[-1].value)
                final_iter_median = statistics.median(
                    float(event.value) for event in iter_times[-20:]
                )
                c_critic = by_tag.get("train_c_learning/critic_loss", [])
                crl_actor = by_tag.get("train_crl/actor_loss", [])
                rows.append(
                    {
                        "environment": env.paper_name,
                        "method": method,
                        "seed": seed,
                        "run_directory": str(run_dir),
                        "completed_training_env_steps": completed_env_steps,
                        "test_success_rate": float(test["succ_rate"]),
                        "test_episode_return": float(test["epi_return"]),
                        "test_hitting_time": float(test["hitting_time"]),
                        "validation_peak_success": max(
                            float(event.value) for event in validation_success
                        ),
                        "validation_final_success": float(validation_success[-1].value),
                        "final_replay_successful_transition_fraction": success_fraction,
                        "median_iteration_seconds_last_20_records": final_iter_median,
                        "c_learning_critic_loss_first_20_mean": window_mean(
                            c_critic, first=True
                        ),
                        "c_learning_critic_loss_last_20_mean": window_mean(
                            c_critic, first=False
                        ),
                        "crl_actor_loss_last_20_mean": window_mean(
                            crl_actor, first=False
                        ),
                        "event_file_count": num_files,
                        "raw_train_data_scalar_count": num_values,
                    }
                )

    if missing:
        raise FileNotFoundError(f"Unexpected absent runs: {missing}")
    if incomplete:
        raise RuntimeError(f"Runs without COMPLETE markers: {incomplete}")
    if test_log_issues:
        raise RuntimeError(f"Invalid test logs: {test_log_issues}")
    if budget_mismatches:
        raise RuntimeError(f"Runs that did not reach the paper budget: {budget_mismatches}")
    if parse_errors:
        raise RuntimeError(f"TensorBoard parsing errors: {parse_errors}")
    if nonfinite:
        raise AssertionError(f"Non-finite train/data scalars: {nonfinite}")

    with (DATA_ROOT / "training_scalar_runs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    comparison_environments = {env.paper_name for env in ENVIRONMENTS}
    aggregate = []
    for method in METHODS:
        method_rows = [
            row for row in rows
            if row["method"] == method
            and row["environment"] in comparison_environments
        ]
        aggregate.append(
            {
                "method": method,
                "comparison_runs": len(method_rows),
                "mean_final_replay_successful_transition_fraction": statistics.mean(
                    row["final_replay_successful_transition_fraction"] for row in method_rows
                ),
                "median_of_run_median_iteration_seconds": statistics.median(
                    row["median_iteration_seconds_last_20_records"] for row in method_rows
                ),
            }
        )
    with (DATA_ROOT / "aggregate_training_scalars.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate[0].keys())
        writer.writeheader()
        writer.writerows(aggregate)

    available_aggregate = []
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        available_aggregate.append(
            {
                "method": method,
                "available_runs": len(method_rows),
                "mean_final_replay_successful_transition_fraction": statistics.mean(
                    row["final_replay_successful_transition_fraction"] for row in method_rows
                ),
                "median_of_run_median_iteration_seconds": statistics.median(
                    row["median_iteration_seconds_last_20_records"] for row in method_rows
                ),
            }
        )
    with (DATA_ROOT / "aggregate_training_scalars_available.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=available_aggregate[0].keys())
        writer.writeheader()
        writer.writerows(available_aggregate)

    def select(environment, method):
        return [
            row for row in rows
            if row["environment"] == environment and row["method"] == method
        ]

    c_learning_diagnostics = {}
    c_learning_environments = (
        "FetchPush",
        "PandaSlide-v3",
        "PandaPickAndPlace-v3",
        "PandaPushJoints-v3",
        "PandaPickAndPlaceJoints-v3",
    )
    for environment in c_learning_environments:
        selected = select(environment, "C-Learning")
        item = {
            "critic_loss_first_20_mean_over_seeds": statistics.mean(
                row["c_learning_critic_loss_first_20_mean"] for row in selected
            ),
            "critic_loss_last_20_mean_over_seeds": statistics.mean(
                row["c_learning_critic_loss_last_20_mean"] for row in selected
            ),
        }
        if environment.startswith("Panda"):
            item["mean_selected_test_return_divided_by_horizon_50"] = statistics.mean(
                row["test_episode_return"] / 50 for row in selected
            )
        c_learning_diagnostics[environment] = item

    antmaze_diagnostics = {}
    for method in METHODS:
        selected = select("AntMaze UMaze-v5", method)
        antmaze_diagnostics[method] = {
            "mean_per_seed_validation_peak": statistics.mean(
                row["validation_peak_success"] for row in selected
            ),
            "mean_final_validation_success": statistics.mean(
                row["validation_final_success"] for row in selected
            ),
            "mean_final_replay_successful_transition_fraction": statistics.mean(
                row["final_replay_successful_transition_fraction"] for row in selected
            ),
        }

    antnavigate_diagnostics = {}
    for method in METHODS:
        selected = select("AntNavigate-v4", method)
        antnavigate_diagnostics[method] = {
            "mean_test_success": statistics.mean(
                row["test_success_rate"] for row in selected
            ),
            "mean_per_seed_validation_peak": statistics.mean(
                row["validation_peak_success"] for row in selected
            ),
            "mean_final_validation_success": statistics.mean(
                row["validation_final_success"] for row in selected
            ),
            "mean_final_replay_successful_transition_fraction": statistics.mean(
                row["final_replay_successful_transition_fraction"] for row in selected
            ),
        }

    low_success_diagnostics = {
        "c_learning": c_learning_diagnostics,
        "antnavigate": antnavigate_diagnostics,
        "antmaze": antmaze_diagnostics,
        "c_learning_antnavigate_seed_rows": select(
            "AntNavigate-v4", "C-Learning"
        ),
        "crl_fetchpush_seed_rows": select("FetchPush", "CRL"),
        "td_infonce_seed_split_rows": {
            environment: select(environment, "TD-InfoNCE")
            for environment in (
                "PandaSlide-v3",
                "PandaPushJoints-v3",
                "PandaPickAndPlaceJoints-v3",
            )
        },
        "c_learning_reacher_seed_rows": select("reacher_hard", "C-Learning"),
        "additional_task_seed_rows": {
            environment: {method: select(environment, method) for method in METHODS}
            for environment in ("Pusher-v4", "FetchSlide")
        },
    }
    (DATA_ROOT / "low_success_diagnostics.json").write_text(
        json.dumps(low_success_diagnostics, indent=2) + "\n", encoding="utf-8"
    )

    summary = {
        "audited_runs": len(rows),
        "missing_runs": len(missing),
        "event_files": event_file_count,
        "raw_train_or_data_scalar_values": train_data_scalar_count,
        "runs_without_complete_marker": incomplete,
        "invalid_test_logs": test_log_issues,
        "budget_mismatches": budget_mismatches,
        "event_parse_errors": parse_errors,
        "nonfinite_train_or_data_scalars": nonfinite,
        "replay_metric_definition": (
            "last data/num_successes divided by last data/num_regular_transitions; "
            "the numerator counts transitions marked successful, not successful episodes"
        ),
        "timing_metric_definition": (
            "per run median of the final 20 train/iter_time records, then median over runs"
        ),
        "comparison_subset": "all fourteen tasks with complete, summary-matched artifacts for all six methods",
        "aggregate": aggregate,
        "available_run_aggregate": available_aggregate,
        "low_success_diagnostics_file": "low_success_diagnostics.json",
    }
    (DATA_ROOT / "training_scalar_audit.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

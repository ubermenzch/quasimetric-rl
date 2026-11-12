#!/usr/bin/env python3
"""Reproduce the NQRL paper tables and figures from the selected runs.

The final test summaries below use the author-provided values, with FetchSlide
updated to the latest local test logs at the author's request.
Local logs audit those summaries and derive validation-curve metrics for all
fourteen tasks and six methods.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPO_ROOT.parent / "qrl-assets" / "results" / "queue"
IMPORTED_RESULT_ROOT = REPO_ROOT.parent
PAPER_ROOT = REPO_ROOT / "iclr2027"
DATA_ROOT = PAPER_ROOT / "data"
FIGURE_ROOT = PAPER_ROOT / "figures"
SEEDS = range(1000, 1005)

METHODS = ("QRL", "TD-InfoNCE", "GCSL", "C-Learning", "CRL", "NQRL")
ARTIFACT_METHOD = {
    "QRL": "QRL-L",
    "TD-InfoNCE": "TD-InfoNCE-L",
    "GCSL": "GCSL-L",
    "C-Learning": "C-Learning-L",
    "CRL": "CRL-L",
    # Historical artifact IDs predate the paper's NQRL name.
    "NQRL": "CQRL+Min32-L-InnerLR1",
}


@dataclass(frozen=True)
class Environment:
    key: str
    paper_name: str
    budget: int
    suite: str
    stem: str
    slug: str
    eval_episodes: int
    test_episodes: int
    gcsl_factorized: bool = False


ENVIRONMENTS = (
    Environment(
        "antnavigate", "AntNavigate-v4", 500_000, "iclr_main_v1",
        "500k_50kckpt_val500_test1000", "antnavigate_v4", 500, 1000,
    ),
    Environment(
        "fetchpush", "FetchPush", 200_000, "iclr_main_v1",
        "200k_20kckpt_val1000_test1000", "fetchpush", 1000, 1000,
    ),
    Environment(
        "fetchpick", "FetchPickAndPlace", 200_000, "iclr_main_v1",
        "200k_20kckpt_val1000_test1000", "fetchpickandplace", 1000, 1000,
    ),
    Environment(
        "reacher", "reacher_hard", 200_000, "iclr_main_v1",
        "200k_20kckpt_val1000_test1000", "dmc_reacher_hard", 1000, 1000,
    ),
    Environment(
        "maze2d", "maze2d-large", 200_000, "iclr_main_v1",
        "200k_20kckpt_val1000_test1000", "maze2d_large", 1000, 1000,
    ),
    Environment(
        "pointmaze", "PointMaze Medium-v3", 200_400, "extended_v1",
        "200400steps_20kckpt_val100_test100", "pointmaze_medium_v3", 100, 100,
    ),
    Environment(
        "antmaze", "AntMaze UMaze-v5", 400_000, "extended_v1",
        "200200steps_20kckpt_val100_test100", "antmaze_umaze_v5", 100, 100,
        gcsl_factorized=True,
    ),
    Environment(
        "pandaslide", "PandaSlide-v3", 400_000, "extended_v1",
        "100k_20kckpt_val200_test200", "pandaslide_v3", 200, 200,
    ),
    Environment(
        "pandapick", "PandaPickAndPlace-v3", 400_000, "extended_v1",
        "100k_20kckpt_val200_test200", "pandapickandplace_v3", 200, 200,
    ),
    Environment(
        "pandaflip", "PandaFlip-v3", 400_000, "extended_v1",
        "100k_20kckpt_val200_test200", "pandaflip_v3", 200, 200,
    ),
    Environment(
        "pandapushj", "PandaPushJoints-v3", 400_000, "extended_v1",
        "100k_20kckpt_val200_test200", "pandapushjoints_v3", 200, 200,
        gcsl_factorized=True,
    ),
    Environment(
        "pandapickj", "PandaPickAndPlaceJoints-v3", 400_000, "extended_v1",
        "100k_20kckpt_val200_test200", "pandapickandplacejoints_v3", 200, 200,
        gcsl_factorized=True,
    ),
    Environment(
        "pusher", "Pusher-v4", 500_000, "iclr_main_v1",
        "500k_50kckpt_val500_test1000", "pusher_v4", 500, 1000,
        gcsl_factorized=True,
    ),
    Environment(
        "fetchslide", "FetchSlide", 500_000, "iclr_main_v1",
        "500k_50kckpt_val500_test1000", "fetchslide", 500, 1000,
    ),
)

PAIRED_METHODS = ("TD-InfoNCE", "NQRL")


# Mean and sample standard deviation over five training seeds, supplied by the
# author, except the explicitly requested latest FetchSlide results.
# Rows follow METHODS; columns follow ENVIRONMENTS.
FINAL_MEAN = {
    "QRL": (.049, .168, .133, .041, .658, .862, .000, .280, .374, .184, .118, .053, .248, .263),
    "TD-InfoNCE": (.179, .690, .358, .905, .249, .374, .008, .484, .510, .194, .778, .369, .793, .451),
    "GCSL": (.125, .090, .068, .207, .064, .088, .046, .202, .119, .180, .199, .071, .001, .046),
    "C-Learning": (.043, .071, .043, .728, .118, .080, .000, .003, .026, .048, .095, .025, .007, .014),
    "CRL": (.119, .290, .068, .112, .393, .606, .018, .222, .105, .134, .265, .068, .101, .252),
    "NQRL": (.980, .913, .723, .985, .732, .926, .424, .539, .535, .350, .951, .518, .441, .244),
}
FINAL_STD = {
    "QRL": (.046, .124, .043, .006, .077, .019, .000, .084, .079, .043, .007, .015, .082, .058),
    "TD-InfoNCE": (.018, .085, .190, .014, .017, .050, .004, .259, .021, .082, .324, .188, .426, .064),
    "GCSL": (.013, .008, .007, .033, .006, .037, .036, .035, .022, .041, .035, .018, .002, .024),
    "C-Learning": (.095, .001, .001, .309, .190, .017, .000, .004, .002, .010, .000, .000, .010, .018),
    "CRL": (.017, .222, .037, .028, .172, .059, .011, .057, .013, .051, .097, .040, .050, .053),
    "NQRL": (.011, .036, .156, .010, .039, .029, .046, .075, .044, .062, .019, .039, .211, .010),
}


# Preserve the author-supplied column mapping while exporting alphabetic task order.
_environment_order = sorted(range(len(ENVIRONMENTS)),
                            key=lambda i: ENVIRONMENTS[i].paper_name.casefold())
FINAL_MEAN = {method: tuple(values[i] for i in _environment_order)
              for method, values in FINAL_MEAN.items()}
FINAL_STD = {method: tuple(values[i] for i in _environment_order)
             for method, values in FINAL_STD.items()}
ENVIRONMENTS = tuple(ENVIRONMENTS[i] for i in _environment_order)


def artifact_qualifier(env: Environment, method: str) -> str:
    if env.key in {"fetchpush", "fetchpick", "fetchslide"} and method in {
        "TD-InfoNCE", "GCSL", "C-Learning"
    }:
        return "goalreprv2_"
    if env.suite == "iclr_main_v1" and method == "CRL":
        return "originalcrl2022_"
    if method == "GCSL" and env.gcsl_factorized:
        return "factorized_"
    return ""


def artifact_dir(env: Environment, method: str, seed: int) -> Path:
    token = ARTIFACT_METHOD[method]
    qualifier = artifact_qualifier(env, method)
    task_id = (
        f"{env.suite}_{token}_{env.stem}_{qualifier}{env.slug}_online_s{seed}"
    )
    primary = RESULT_ROOT / task_id
    imported = IMPORTED_RESULT_ROOT / task_id
    return imported if not primary.exists() and imported.is_dir() else primary


def load_json_lines(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def validation_curve(path: Path) -> tuple[list[dict], dict]:
    by_step: dict[tuple[int, int], dict] = {}
    duplicate_keys = set()
    conflicting_duplicate_keys = set()
    for row in load_json_lines(path):
        if row.get("split") != "validation":
            continue
        key = (int(row["env_steps"]), int(row.get("optim_steps", 0)))
        previous = by_step.get(key)
        if previous is not None:
            duplicate_keys.add(key)
            fields = ("succ_rate", "hitting_time", "epi_return")
            if any(float(previous[field]) != float(row[field]) for field in fields):
                conflicting_duplicate_keys.add(key)
        by_step[key] = row
    rows = sorted(by_step.values(), key=lambda row: (row["env_steps"], row.get("optim_steps", 0)))
    if not rows:
        raise ValueError(f"No validation records in {path}")
    return rows, {
        "duplicate_keys": len(duplicate_keys),
        "conflicting_duplicate_keys": len(conflicting_duplicate_keys),
    }


def interpolate(points: list[tuple[float, float]], x: float) -> float:
    if x <= points[0][0]:
        return points[0][1]
    if x >= points[-1][0]:
        return points[-1][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= x <= x1:
            if x1 == x0:
                return y1
            ratio = (x - x0) / (x1 - x0)
            return y0 + ratio * (y1 - y0)
    raise AssertionError("interpolation interval not found")


def trapz(xs: Iterable[float], ys: Iterable[float]) -> float:
    x_values = list(xs)
    y_values = list(ys)
    return sum(
        (x1 - x0) * (y0 + y1) / 2
        for x0, x1, y0, y1 in zip(
            x_values, x_values[1:], y_values, y_values[1:]
        )
    )


def write_final_csv() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    path = DATA_ROOT / "final_success.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("environment", "budget", "method", "mean", "sample_std", "source"))
        for env_idx, env in enumerate(ENVIRONMENTS):
            for method in METHODS:
                writer.writerow(
                    (
                        env.paper_name,
                        env.budget,
                        method,
                        f"{FINAL_MEAN[method][env_idx]:.3f}",
                        f"{FINAL_STD[method][env_idx]:.3f}",
                        (
                            "latest local selected-checkpoint test logs (author-requested update)"
                            if env.key == "fetchslide" else "author-provided final test summary"
                        ),
                    )
                )


def audit_local_tests() -> dict:
    audited = []
    absent = []
    mismatches = []
    nonfinite = []
    for env_idx, env in enumerate(ENVIRONMENTS):
        for method in METHODS:
            values = []
            for seed in SEEDS:
                path = artifact_dir(env, method, seed) / "test.log"
                if not path.exists():
                    absent.append(str(path))
                    continue
                rows = load_json_lines(path)
                if len(rows) != 1:
                    raise ValueError(f"Expected one test row in {path}, found {len(rows)}")
                row = rows[0]
                for key in ("succ_rate", "hitting_time", "epi_return"):
                    if not math.isfinite(float(row[key])):
                        nonfinite.append(f"{path}:{key}")
                values.append(float(row["succ_rate"]))
            if not values:
                continue
            if len(values) != 5:
                raise ValueError(f"Partial seed group for {env.paper_name}/{method}: {len(values)}")
            mean = statistics.mean(values)
            sample_std = statistics.stdev(values)
            target_mean = FINAL_MEAN[method][env_idx]
            target_std = FINAL_STD[method][env_idx]
            # The supplied table is rounded to three decimals.
            matches = abs(mean - target_mean) <= 5.1e-4 and abs(sample_std - target_std) <= 5.1e-4
            if not matches:
                mismatches.append(
                    {
                        "environment": env.paper_name,
                        "method": method,
                        "computed_mean": mean,
                        "provided_mean": target_mean,
                        "computed_sample_std": sample_std,
                        "provided_sample_std": target_std,
                    }
                )
            audited.append(
                {
                    "environment": env.paper_name,
                    "method": method,
                    "seeds": values,
                    "computed_mean": mean,
                    "computed_sample_std": sample_std,
                    "matches_author_summary": matches,
                    "run_directories": [str(artifact_dir(env, method, seed)) for seed in SEEDS],
                }
            )
    if absent:
        raise FileNotFoundError(f"Missing test logs: {absent}")
    if mismatches:
        raise AssertionError(f"Local test summaries disagree with the paper table: {mismatches}")
    if nonfinite:
        raise AssertionError(f"Non-finite test metrics: {nonfinite}")
    return {
        "audited_seed_runs": len(audited) * 5,
        "audited_method_environment_groups": len(audited),
        "missing_seed_runs": len(absent),
        "missing_logs": absent,
        "summary_mismatches": mismatches,
        "matched_method_environment_groups": sum(row["matches_author_summary"] for row in audited),
        "all_available_groups_match_author_summary_at_three_decimals": not mismatches,
        "nonfinite_test_metrics": nonfinite,
        "groups": audited,
    }


def aggregate_final_metrics() -> list[dict]:
    # Aggregate unrounded seed measurements, not the rounded display summaries.
    with (DATA_ROOT / "seed_test_success.csv").open(newline="") as handle:
        seed_rows = list(csv.DictReader(handle))
    raw = {}
    for env in ENVIRONMENTS:
        for method in METHODS:
            group = [r for r in seed_rows if r["environment"] == env.paper_name and r["method"] == method]
            assert sorted(int(r["training_seed"]) for r in group) == list(SEEDS)
            raw[env.paper_name, method] = [float(r["test_success"]) for r in group]
    result = []
    for method in METHODS:
        means = [statistics.mean(raw[env.paper_name, method]) for env in ENVIRONMENTS]
        stds = [statistics.stdev(raw[env.paper_name, method]) for env in ENVIRONMENTS]
        wins = sum(
            value == max(statistics.mean(raw[ENVIRONMENTS[idx].paper_name, other]) for other in METHODS)
            for idx, value in enumerate(means)
        )
        result.append(
            {
                "method": method,
                "macro_mean": statistics.mean(means),
                "median": statistics.median(means),
                "worst_task": min(means),
                "mean_seed_std": statistics.mean(stds),
                "wins": wins,
            }
        )
    path = DATA_ROOT / "aggregate_final_metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=result[0].keys())
        writer.writeheader()
        writer.writerows(result)
    return result


def analyze_learning_curves(
    included_envs=ENVIRONMENTS, methods=METHODS, prefix=""
) -> tuple[list[dict], list[dict], dict]:
    grid = tuple(index / 100 for index in range(20, 101))
    curve_rows = []
    run_rows = []
    nonfinite = []
    runs_with_duplicate_validation_keys = 0
    duplicate_validation_keys = 0
    conflicting_duplicate_validation_keys = 0

    for env in included_envs:
        for method in methods:
            for seed in SEEDS:
                run_dir = artifact_dir(env, method, seed)
                path = run_dir / "eval.log"
                if not path.exists():
                    raise FileNotFoundError(path)
                records, duplicate_summary = validation_curve(path)
                if duplicate_summary["duplicate_keys"]:
                    runs_with_duplicate_validation_keys += 1
                duplicate_validation_keys += duplicate_summary["duplicate_keys"]
                conflicting_duplicate_validation_keys += duplicate_summary[
                    "conflicting_duplicate_keys"
                ]
                points = []
                for row in records:
                    values = (row["succ_rate"], row["hitting_time"], row["epi_return"])
                    if not all(math.isfinite(float(value)) for value in values):
                        nonfinite.append(str(path))
                    points.append((float(row["env_steps"]) / env.budget, float(row["succ_rate"])))
                if points[0][0] > grid[0] or points[-1][0] < grid[-1] - 1e-9:
                    raise ValueError(f"Curve does not cover 20%-100% budget: {path}")
                sampled = [interpolate(points, x) for x in grid]
                auc_points = [(grid[0], interpolate(points, grid[0]))]
                auc_points.extend((x, y) for x, y in points if grid[0] < x < grid[-1])
                auc_points.append((grid[-1], interpolate(points, grid[-1])))
                auc = trapz(
                    (x for x, _ in auc_points),
                    (y for _, y in auc_points),
                ) / (grid[-1] - grid[0])
                first_hits = [x for x, y in points if y >= .5]
                peak_success = max(y for _, y in points)
                final_success = points[-1][1]
                selected = json.loads((run_dir / "best_checkpoint.json").read_text(encoding="utf-8"))
                validation = selected["validation"]
                test = selected["test"]
                run_rows.append(
                    {
                        "environment": env.paper_name,
                        "method": method,
                        "seed": seed,
                        "auc_20_100": auc,
                        "ever_reached_0_5": int(bool(first_hits)),
                        "first_0_5_budget_fraction": min(first_hits) if first_hits else "",
                        "final_to_peak_validation_ratio": (
                            final_success / peak_success if peak_success > 0 else ""
                        ),
                        "peak_to_final_validation_gap": peak_success - final_success,
                        "selected_checkpoint_fraction": float(validation["env_steps"]) / env.budget,
                        "selected_validation_success": float(validation["succ_rate"]),
                        "selected_test_success": float(test["succ_rate"]),
                        "absolute_validation_test_gap": abs(
                            float(validation["succ_rate"]) - float(test["succ_rate"])
                        ),
                    }
                )
                for fraction, success in zip(grid, sampled):
                    curve_rows.append(
                        {
                            "environment": env.paper_name,
                            "method": method,
                            "seed": seed,
                            "budget_fraction": fraction,
                            "validation_success": success,
                        }
                    )

    if nonfinite:
        raise AssertionError(f"Non-finite validation metrics: {nonfinite}")

    with (DATA_ROOT / f"{prefix}learning_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=run_rows[0].keys())
        writer.writeheader()
        writer.writerows(run_rows)
    with (DATA_ROOT / f"{prefix}learning_curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=curve_rows[0].keys())
        writer.writeheader()
        writer.writerows(curve_rows)

    summary = []
    for method in methods:
        rows = [row for row in run_rows if row["method"] == method]
        hits = [float(row["first_0_5_budget_fraction"]) for row in rows if row["ever_reached_0_5"]]
        summary.append(
            {
                "method": method,
                "auc_20_100": statistics.mean(float(row["auc_20_100"]) for row in rows),
                "runs_reaching_0_5_pct": 100 * statistics.mean(row["ever_reached_0_5"] for row in rows),
                "median_first_0_5_fraction_conditional": statistics.median(hits) if hits else "",
                "mean_final_to_peak_ratio_nonzero": statistics.mean(
                    float(row["final_to_peak_validation_ratio"])
                    for row in rows
                    if row["final_to_peak_validation_ratio"] != ""
                ),
                "mean_peak_to_final_gap": statistics.mean(
                    float(row["peak_to_final_validation_gap"]) for row in rows
                ),
                "mean_selected_checkpoint_fraction": statistics.mean(
                    float(row["selected_checkpoint_fraction"]) for row in rows
                ),
                "mean_absolute_validation_test_gap": statistics.mean(
                    float(row["absolute_validation_test_gap"]) for row in rows
                ),
            }
        )
    with (DATA_ROOT / f"{prefix}aggregate_learning_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    per_environment = []
    paired_auc_wins = 0
    paired_auc_comparisons = 0
    for env in included_envs:
        means = {
            method: statistics.mean(
                float(row["auc_20_100"])
                for row in run_rows
                if row["environment"] == env.paper_name and row["method"] == method
            )
            for method in methods
        }
        best_baseline = max(methods[:-1], key=lambda method: means[method])
        for seed in SEEDS:
            nqrl_auc = next(
                float(row["auc_20_100"])
                for row in run_rows
                if row["environment"] == env.paper_name
                and row["method"] == "NQRL"
                and row["seed"] == seed
            )
            baseline_auc = next(
                float(row["auc_20_100"])
                for row in run_rows
                if row["environment"] == env.paper_name
                and row["method"] == best_baseline
                and row["seed"] == seed
            )
            paired_auc_wins += int(nqrl_auc > baseline_auc)
            paired_auc_comparisons += 1
        per_environment.append(
            {
                "environment": env.paper_name,
                **{f"{method}_auc_20_100": means[method] for method in methods},
                "best_baseline": best_baseline,
                "nqrl_advantage_over_best_baseline": (
                    means["NQRL"] - means[best_baseline]
                ),
            }
        )
    with (DATA_ROOT / f"{prefix}per_environment_learning_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=per_environment[0].keys())
        writer.writeheader()
        writer.writerows(per_environment)

    macro_curve = []
    for method in methods:
        for fraction in grid:
            values = [
                float(row["validation_success"])
                for row in curve_rows
                if row["method"] == method and row["budget_fraction"] == fraction
            ]
            macro_curve.append(
                {
                    "method": method,
                    "budget_fraction": fraction,
                    "macro_mean_validation_success": statistics.mean(values),
                    "standard_error_over_runs": statistics.stdev(values) / math.sqrt(len(values)),
                }
            )
    with (DATA_ROOT / f"{prefix}macro_learning_curve.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=macro_curve[0].keys())
        writer.writeheader()
        writer.writerows(macro_curve)

    metadata = {
        "environments": [env.paper_name for env in included_envs],
        "excluded_environments": [env.paper_name for env in ENVIRONMENTS if env not in included_envs],
        "methods": list(methods),
        "training_seeds_per_environment": 5,
        "runs_per_method": len(included_envs) * 5,
        "auc_interval": [0.2, 1.0],
        "auc_interpolation": "linear between recorded validation checkpoints",
        "duplicate_validation_rule": (
            "for repeated (env_steps, optim_steps), retain the last JSONL record"
        ),
        "runs_with_duplicate_validation_keys": runs_with_duplicate_validation_keys,
        "duplicate_validation_keys": duplicate_validation_keys,
        "conflicting_duplicate_validation_keys": (
            conflicting_duplicate_validation_keys
        ),
        "curve_aggregation": f"macro mean over {len(included_envs)} environments and 5 training seeds",
        "nqrl_seed_paired_auc_wins_against_task_best_mean_baseline": paired_auc_wins,
        "nqrl_seed_paired_auc_comparisons": paired_auc_comparisons,
        "validation_test_separation": True,
        "nonfinite_validation_metrics": nonfinite,
    }
    return summary, macro_curve, metadata


PALETTE = {
    "QRL": "#5B6770",
    "TD-InfoNCE": "#2878B5",
    "GCSL": "#D99A2B",
    "C-Learning": "#8C6BB1",
    "CRL": "#3A9D5D",
    "NQRL": "#C23B3B",
}


def draw_final_success_figure() -> None:
    from reportlab.lib import colors
    from reportlab.pdfbase.pdfmetrics import stringWidth
    from reportlab.pdfgen import canvas

    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    width, height = 7.15 * 72, 65 + 26 * len(ENVIRONMENTS)
    output = FIGURE_ROOT / "final_success.pdf"
    pdf = canvas.Canvas(str(output), pagesize=(width, height))
    left, right, bottom, top = 126, 18, 37, 28
    plot_w, plot_h = width - left - right, height - bottom - top
    row_h = plot_h / len(ENVIRONMENTS)

    pdf.setTitle(f"Final test success rate on the {len(ENVIRONMENTS)} selected tasks")
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(left, height - 15, "Final test success rate")
    pdf.setFont("Helvetica", 6.5)
    for tick in (0, .2, .4, .6, .8, 1.0):
        x = left + tick * plot_w
        pdf.setStrokeColor(colors.HexColor("#DADADA"))
        pdf.setLineWidth(.35)
        pdf.line(x, bottom, x, height - top)
        pdf.setFillColor(colors.HexColor("#333333"))
        pdf.drawCentredString(x, bottom - 10, f"{tick:.1f}")

    offsets = tuple((index - 2.5) * min(2.9, row_h / 7) for index in range(6))
    for env_idx, env in enumerate(ENVIRONMENTS):
        y_center = height - top - (env_idx + .5) * row_h
        if env_idx % 2 == 0:
            pdf.setFillColor(colors.HexColor("#F5F5F5"))
            pdf.rect(left, y_center - row_h / 2, plot_w, row_h, stroke=0, fill=1)
        budget = f"{env.budget / 1000:g}k"
        label = f"{env.paper_name} ({budget})"
        pdf.setFillColor(colors.black)
        pdf.setFont("Helvetica", 6.2)
        pdf.drawRightString(left - 5, y_center - 2, label)
        for method_idx, method in enumerate(METHODS):
            mean = FINAL_MEAN[method][env_idx]
            std = FINAL_STD[method][env_idx]
            y = y_center + offsets[method_idx]
            x0 = left + max(0, mean - std) * plot_w
            x1 = left + min(1, mean + std) * plot_w
            x = left + mean * plot_w
            color = colors.HexColor(PALETTE[method])
            pdf.setStrokeColor(color)
            pdf.setFillColor(color)
            pdf.setLineWidth(1.0 if method == "NQRL" else .55)
            pdf.line(x0, y, x1, y)
            pdf.line(x0, y - 1.1, x0, y + 1.1)
            pdf.line(x1, y - 1.1, x1, y + 1.1)
            radius = 1.9 if method == "NQRL" else 1.35
            pdf.circle(x, y, radius, stroke=0, fill=1)

    legend_y = 12
    cursor = left
    pdf.setFont("Helvetica", 6.5)
    for method in METHODS:
        pdf.setFillColor(colors.HexColor(PALETTE[method]))
        pdf.circle(cursor, legend_y, 1.8 if method == "NQRL" else 1.4, stroke=0, fill=1)
        pdf.setFillColor(colors.black)
        pdf.drawString(cursor + 4, legend_y - 2, method)
        cursor += stringWidth(method, "Helvetica", 6.5) + 18
    pdf.save()


def draw_learning_curve_figure(macro_curve: list[dict]) -> None:
    from reportlab.lib import colors
    from reportlab.pdfbase.pdfmetrics import stringWidth
    from reportlab.pdfgen import canvas

    width, height = 7.15 * 72, 3.65 * 72
    output = FIGURE_ROOT / "learning_efficiency.pdf"
    pdf = canvas.Canvas(str(output), pagesize=(width, height))
    left, right, bottom, top = 42, 16, 34, 43
    plot_w, plot_h = width - left - right, height - bottom - top

    pdf.setTitle("Validation success over the interaction budget")
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(left, height - 15, f"Validation success: all {len(ENVIRONMENTS)} tasks")
    pdf.setFont("Helvetica", 6.5)
    for tick in (0, .2, .4, .6, .8, 1.0):
        y = bottom + tick * plot_h
        pdf.setStrokeColor(colors.HexColor("#DDDDDD"))
        pdf.setLineWidth(.35)
        pdf.line(left, y, width - right, y)
        pdf.setFillColor(colors.HexColor("#333333"))
        pdf.drawRightString(left - 5, y - 2, f"{tick:.1f}")
    for tick in (.2, .4, .6, .8, 1.0):
        x = left + (tick - .2) / .8 * plot_w
        pdf.setFillColor(colors.HexColor("#333333"))
        pdf.drawCentredString(x, bottom - 11, f"{int(tick * 100)}")
    pdf.setFont("Helvetica", 7)
    pdf.drawCentredString(left + plot_w / 2, 8, "Interaction budget used (%)")
    pdf.saveState()
    pdf.translate(10, bottom + plot_h / 2)
    pdf.rotate(90)
    pdf.drawCentredString(0, 0, "Macro validation success")
    pdf.restoreState()

    for method in METHODS:
        rows = [row for row in macro_curve if row["method"] == method]
        rows.sort(key=lambda row: row["budget_fraction"])
        pdf.setStrokeColor(colors.HexColor(PALETTE[method]))
        pdf.setLineWidth(2.2 if method == "NQRL" else 1.0)
        path = pdf.beginPath()
        for index, row in enumerate(rows):
            x = left + (float(row["budget_fraction"]) - .2) / .8 * plot_w
            y = bottom + float(row["macro_mean_validation_success"]) * plot_h
            if index == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        pdf.drawPath(path, stroke=1, fill=0)

    legend_y = height - 30
    cursor = left
    pdf.setFont("Helvetica", 6.5)
    for method in METHODS:
        pdf.setStrokeColor(colors.HexColor(PALETTE[method]))
        pdf.setLineWidth(2.2 if method == "NQRL" else 1.0)
        pdf.line(cursor, legend_y, cursor + 10, legend_y)
        pdf.setFillColor(colors.black)
        pdf.drawString(cursor + 13, legend_y - 2, method)
        cursor += stringWidth(method, "Helvetica", 6.5) + 29
    pdf.save()


def draw_paired_learning_figure(macro_curve: list[dict]) -> None:
    from reportlab.lib import colors
    from reportlab.pdfgen import canvas

    with (DATA_ROOT / "paired_learning_curves.csv").open(encoding="utf-8") as handle:
        curves = list(csv.DictReader(handle))
    width, height = 7.15 * 72, 2.9 * 72
    pdf = canvas.Canvas(str(FIGURE_ROOT / "paired_learning.pdf"), pagesize=(width, height))
    pdf.setTitle("NQRL and TD-InfoNCE: all 14 tasks and task-specific reversals")
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(34, height - 15, "NQRL and TD-InfoNCE: complete 14-task comparison")
    for index, method in enumerate(PAIRED_METHODS):
        x = 34 + index * 100
        pdf.setStrokeColor(colors.HexColor(PALETTE[method]))
        pdf.setLineWidth(1.7)
        pdf.line(x, height - 29, x + 12, height - 29)
        pdf.setFillColor(colors.black)
        pdf.setFont("Helvetica", 7)
        pdf.drawString(x + 16, height - 31, method)
    bottom, top = 35, height - 55
    plot_h = top - bottom
    panel_w = (width - 48) / 3
    for panel, environment in enumerate((None, "Pusher-v4", "FetchSlide")):
        left = 34 + panel * panel_w
        plot_w = panel_w - 22
        pdf.setFillColor(colors.black)
        pdf.setFont("Helvetica-Bold", 8)
        pdf.drawString(left, top + 7, environment or "14-task macro mean")
        for tick in (0, .25, .5, .75, 1):
            y = bottom + tick * plot_h
            pdf.setStrokeColor(colors.HexColor("#DDDDDD"))
            pdf.setLineWidth(.35)
            pdf.line(left, y, left + plot_w, y)
            if panel == 0:
                pdf.setFillColor(colors.black)
                pdf.setFont("Helvetica", 6)
                pdf.drawRightString(left - 4, y - 2, f"{tick:g}")
        for tick in (.2, .4, .6, .8, 1):
            pdf.setFillColor(colors.black)
            pdf.setFont("Helvetica", 6)
            pdf.drawCentredString(left + (tick - .2) / .8 * plot_w, bottom - 10, f"{tick * 100:.0f}")
        for method in PAIRED_METHODS:
            if environment is None:
                points = [
                    (float(row["budget_fraction"]), float(row["macro_mean_validation_success"]))
                    for row in macro_curve if row["method"] == method
                ]
            else:
                points = []
                for step in range(20, 101):
                    fraction = step / 100
                    values = [
                        float(row["validation_success"]) for row in curves
                        if row["environment"] == environment and row["method"] == method
                        and float(row["budget_fraction"]) == fraction
                    ]
                    assert len(values) == len(SEEDS)
                    points.append((fraction, statistics.mean(values)))
            pdf.setStrokeColor(colors.HexColor(PALETTE[method]))
            pdf.setLineWidth(1.8)
            path = pdf.beginPath()
            for index, (fraction, success) in enumerate(sorted(points)):
                x = left + (fraction - .2) / .8 * plot_w
                y = bottom + success * plot_h
                if index == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            pdf.drawPath(path, stroke=1, fill=0)
    pdf.setFillColor(colors.black)
    pdf.setFont("Helvetica", 7)
    pdf.drawCentredString(width / 2, 9, "Interaction budget used (%)")
    pdf.saveState()
    pdf.translate(9, bottom + plot_h / 2)
    pdf.rotate(90)
    pdf.drawCentredString(0, 0, "Validation success")
    pdf.restoreState()
    pdf.save()


def main() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    local_audit = audit_local_tests()
    write_final_csv()
    final_metrics = aggregate_final_metrics()
    learning_metrics, macro_curve, learning_metadata = analyze_learning_curves()
    paired_metrics, paired_curve, paired_metadata = analyze_learning_curves(
        ENVIRONMENTS, PAIRED_METHODS, "paired_"
    )
    draw_final_success_figure()
    draw_learning_curve_figure(macro_curve)
    draw_paired_learning_figure(paired_curve)
    audit = {
        "result_root": str(RESULT_ROOT),
        "imported_result_root": str(IMPORTED_RESULT_ROOT),
        "final_table": {
            "source": "author-provided summaries; FetchSlide updated from latest local test logs",
            "training_seeds": 5,
            "standard_deviation": "sample standard deviation across training seeds",
        },
        "local_test_audit": local_audit,
        "learning_curve_analysis": learning_metadata,
        "paired_learning_curve_analysis": paired_metadata,
        "paired_aggregate_learning_metrics": paired_metrics,
        "aggregate_final_metrics": final_metrics,
        "aggregate_learning_metrics": learning_metrics,
    }
    (DATA_ROOT / "audit_summary.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "local_test_audit": {
            key: value for key, value in local_audit.items() if key != "groups"
        },
        "aggregate_final_metrics": final_metrics,
        "aggregate_learning_metrics": learning_metrics,
        "learning_curve_analysis": learning_metadata,
        "paired_aggregate_learning_metrics": paired_metrics,
        "paired_learning_curve_analysis": paired_metadata,
    }, indent=2))


if __name__ == "__main__":
    main()

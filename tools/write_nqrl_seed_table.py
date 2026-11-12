#!/usr/bin/env python3
"""Write individual-seed validation and test results from main-comparison logs."""

import argparse
import csv
import hashlib
import json
import math
import statistics

from analyze_nqrl_paper_results import (
    DATA_ROOT, ENVIRONMENTS, METHODS, PAPER_ROOT, SEEDS, artifact_dir,
    interpolate, load_json_lines, trapz, validation_curve,
)


CSV_NAME = "seed_test_success.csv"
AUDIT_NAME = "seed_test_success_audit.json"
TABLE_PATH = PAPER_ROOT / "tables/seed_test_success.tex"
SELECTION_ORDER = [
    "success_count:max", "hitting_time:min", "epi_return:max",
    "env_steps:max", "optim_steps:max",
]


def validation_auc(records, budget):
    """Use the same unsmoothed, piecewise-linear integral as the main AUC table."""
    points = [(float(r["env_steps"]) / budget, float(r["succ_rate"])) for r in records]
    assert points[0][0] <= 0.2 and points[-1][0] == 1.0
    interval = [(0.2, interpolate(points, 0.2))]
    interval.extend((x, y) for x, y in points if 0.2 < x < 1.0)
    interval.append((1.0, interpolate(points, 1.0)))
    return trapz((x for x, _ in interval), (y for _, y in interval)) / 0.8


def collect_runs():
    rows, sources = [], []
    for env in ENVIRONMENTS:
        for seed in SEEDS:
            for method in METHODS:
                directory = artifact_dir(env, method, seed)
                path = directory / "test.log"
                records = load_json_lines(path)
                assert len(records) == 1, f"Expected one selected-checkpoint test: {path}"
                record = records[0]
                selected_path = directory / "best_checkpoint.json"
                selected = json.loads(selected_path.read_text())
                assert record == selected["test"], f"Test/checkpoint metadata mismatch: {path}"
                assert selected["selection_order"] == SELECTION_ORDER
                validation_path = directory / "eval.log"
                validations, _ = validation_curve(validation_path)
                best = max(validations, key=lambda r: (
                    r["success_count"], -r["hitting_time"], r["epi_return"],
                    r["env_steps"], r["optim_steps"],
                ))
                assert best == selected["validation"], f"Selection rule mismatch: {selected_path}"
                peak = max(r["succ_rate"] for r in validations)
                assert peak == best["succ_rate"]
                final = validations[-1]
                for validation in validations:
                    assert validation["num_episodes"] == env.eval_episodes
                    assert 0 <= validation["success_count"] <= validation["num_episodes"]
                    assert validation["succ_rate"] == validation["success_count"] / validation["num_episodes"]
                    assert all(math.isfinite(float(validation[k])) for k in
                               ("succ_rate", "hitting_time", "epi_return"))
                assert record["env_steps"] == best["env_steps"]
                assert record["optim_steps"] == best["optim_steps"]
                assert record["selected_checkpoint"] == best["checkpoint"]
                assert final["env_steps"] == record["completed_training_env_steps"] == env.budget
                assert record["split"] == "test"
                count, episodes = record["success_count"], record["num_episodes"]
                assert isinstance(count, int) and 0 <= count <= episodes
                assert episodes == env.test_episodes
                value = float(record["succ_rate"])
                assert math.isfinite(value) and value == count / episodes
                rows.append(dict(
                    environment=env.paper_name, method=method, training_seed=seed,
                    peak_validation_success=peak,
                    selected_validation_success=best["succ_rate"],
                    selected_env_steps=record["env_steps"],
                    selected_budget_fraction=record["env_steps"] / env.budget,
                    final_validation_success=final["succ_rate"],
                    validation_auc_20_100=validation_auc(validations, env.budget),
                    test_success=value, successful_episodes=count,
                    test_episodes=episodes,
                    interaction_budget=env.budget,
                    validation_episodes=best["num_episodes"],
                    selected_validation_success_count=best["success_count"],
                    selected_validation_mean_return=best["epi_return"],
                    selected_validation_selection_hitting_time=best["hitting_time"],
                    selected_optim_steps=record["optim_steps"],
                    selected_checkpoint=record["selected_checkpoint"],
                    final_validation_success_count=final["success_count"],
                    final_validation_mean_return=final["epi_return"],
                    final_validation_env_steps=final["env_steps"],
                    final_validation_optim_steps=final["optim_steps"],
                    completed_training_env_steps=record["completed_training_env_steps"],
                    completed_training_optim_steps=record["completed_training_optim_steps"],
                    validation_checkpoints=len(validations),
                    validation_seed_start=best["seed"], validation_seed_end=best["seed_end"],
                    test_mean_return=record["epi_return"],
                    test_seed_start=record["seed"], test_seed_end=record["seed_end"],
                ))
                sources.append(dict(
                    environment=env.paper_name, method=method, training_seed=seed,
                    test_log=str(path),
                    test_log_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    validation_log=str(validation_path),
                    validation_log_sha256=hashlib.sha256(validation_path.read_bytes()).hexdigest(),
                    checkpoint_metadata=str(selected_path),
                    checkpoint_metadata_sha256=hashlib.sha256(selected_path.read_bytes()).hexdigest(),
                ))
    groups = validate_runs(rows)
    audit = dict(
        tasks=len(ENVIRONMENTS), methods=list(METHODS), training_seeds=list(SEEDS),
        test_records=len(rows), matched_main_table_groups=len(groups),
        matched_auc_table_groups=len(groups),
        main_table_source="final_success.csv",
        main_table_source_sha256=hashlib.sha256((DATA_ROOT / "final_success.csv").read_bytes()).hexdigest(),
        auc_table_source="per_task_auc.csv",
        auc_table_source_sha256=hashlib.sha256((DATA_ROOT / "per_task_auc.csv").read_bytes()).hexdigest(),
        auc_runs_source="per_task_auc_runs.csv",
        auc_runs_source_sha256=hashlib.sha256((DATA_ROOT / "per_task_auc_runs.csv").read_bytes()).hexdigest(),
        metric="Success is the fraction of evaluation episodes reaching the goal at least once",
        selection_order=SELECTION_ORDER,
        selected_checkpoints_at_validation_peak=len(rows),
        checkpoint_units="Exact environment interactions, distinct from optimizer updates; budget fractions use actual run budgets",
        auc="Normalized piecewise-linear validation-success integral over 20%--100% of the interaction budget, without smoothing or extrapolation",
        aggregation="Mean and sample SD (ddof=1) over five training seeds; matched to Table 1 and the per-task AUC table",
        display="Success and AUC to three decimals, budget percentage to one decimal, steps/counts exact; highest success and AUC across methods within each task/seed are bold, including ties",
        supplementary_csv_fields="Returns, validation episode counts and seed ranges, checkpoint identities and optimizer steps; validation hitting time is retained only to verify checkpoint selection",
        groups=groups, sources=sources,
    )
    return rows, audit


def validate_runs(rows):
    expected = [(env.paper_name, method, seed) for env in ENVIRONMENTS
                for seed in SEEDS for method in METHODS]
    actual = [(r["environment"], r["method"], int(r["training_seed"])) for r in rows]
    assert actual == expected and len(set(actual)) == 420
    main = {(r["environment"], r["method"]): r for r in
            csv.DictReader((DATA_ROOT / "final_success.csv").open())}
    auc_table = {(r["environment"], r["method"]): r for r in
                 csv.DictReader((DATA_ROOT / "per_task_auc.csv").open())}
    auc_runs = {(r["environment"], r["method"], int(r["seed"])): r for r in
                csv.DictReader((DATA_ROOT / "per_task_auc_runs.csv").open())}
    groups = []
    for env in ENVIRONMENTS:
        for method in METHODS:
            selected = [r for r in rows if (r["environment"], r["method"]) == (env.paper_name, method)]
            values, auc_values = [], []
            for row in selected:
                count, episodes = int(row["successful_episodes"]), int(row["test_episodes"])
                value = float(row["test_success"])
                assert 0 <= count <= episodes == env.test_episodes
                assert math.isfinite(value) and value == count / episodes
                peak, end = float(row["peak_validation_success"]), float(row["final_validation_success"])
                assert 0 <= end <= peak <= 1
                assert peak == float(row["selected_validation_success"])
                assert int(row["validation_episodes"]) == env.eval_episodes
                assert peak == int(row["selected_validation_success_count"]) / env.eval_episodes
                assert end == int(row["final_validation_success_count"]) / env.eval_episodes
                assert int(row["interaction_budget"]) == int(row["final_validation_env_steps"]) == env.budget
                assert int(row["completed_training_env_steps"]) == env.budget
                steps = int(row["selected_env_steps"])
                assert 0 < steps <= env.budget
                assert float(row["selected_budget_fraction"]) == steps / env.budget
                auc = float(row["validation_auc_20_100"])
                assert math.isfinite(auc) and 0 <= auc <= peak + 1e-12
                key = env.paper_name, method, int(row["training_seed"])
                assert math.isclose(auc, float(auc_runs[key]["normalized_auc"]), rel_tol=0, abs_tol=1e-12)
                values.append(value)
                auc_values.append(auc)
            assert len(values) == 5
            mean, sd = statistics.mean(values), statistics.stdev(values)
            reference = main[env.paper_name, method]
            for value, key in ((mean, "mean"), (sd, "sample_std")):
                assert abs(value - float(reference[key])) <= 0.000500001, (
                    f"Seed results disagree with main table: {env.paper_name}/{method}/{key}"
                )
            auc_mean, auc_sd = statistics.mean(auc_values), statistics.stdev(auc_values)
            for value, key in ((auc_mean, "auc_mean"), (auc_sd, "auc_std")):
                assert math.isclose(value, float(auc_table[env.paper_name, method][key]),
                                    rel_tol=0, abs_tol=1e-12)
            groups.append(dict(environment=env.paper_name, method=method,
                               mean=mean, sample_std=sd, auc_mean=auc_mean, auc_sample_std=auc_sd))
    return groups


def render_table(rows):
    by_key = {(r["environment"], r["method"], int(r["training_seed"])): r for r in rows}
    metrics = ("peak_validation_success", "final_validation_success",
               "validation_auc_20_100", "test_success")
    best = {(env.paper_name, seed, metric): max(
        float(by_key[env.paper_name, method, seed][metric]) for method in METHODS
    ) for env in ENVIRONMENTS for seed in SEEDS for metric in metrics}
    header = r"Seed & Peak val. & Step & Budget (\%) & End val. & Val. AUC & Test & Successes \\"
    lines = [
        r"\begingroup", r"\nqrltablestyle", r"\setlength{\tabcolsep}{5.5pt}",
        r"\renewcommand{\arraystretch}{1.1}",
        r"\setlength{\LTleft}{0pt}", r"\setlength{\LTright}{0pt}",
        r"\setlength{\LTcapwidth}{\textwidth}",
        r"\begin{longtable}{@{}>{\centering\arraybackslash}m{\dimexpr0.10\textwidth-1.75\tabcolsep\relax}*{6}{>{\centering\arraybackslash}m{\dimexpr0.125\textwidth-1.75\tabcolsep\relax}}>{\centering\arraybackslash}m{\dimexpr0.15\textwidth-1.75\tabcolsep\relax}@{}}",
        r"\caption{\normalsize Validation and test results for all individual training seeds.",
        r"Peak val., End val., and Test report Success rate (\%); Val. AUC is also scaled to percent.",
        r"Peak val. is the highest validation success, attained at the selected",
        r"checkpoint in every run. Step gives its environment-interaction",
        r"count, with Budget expressing the fraction of the run's interaction budget",
        r"as a percentage. End val. is validation success at training completion.",
        r"Val. AUC uses the normalized 20\%--100\% interval from Table~\ref{tab:auc-by-task}.",
        r"Test is success at the selected checkpoint, as in Table~\ref{tab:main};",
        r"Successes gives successful/total test episodes. Each group specifies",
        r"the validation episode count $n_{\mathrm{val}}$ per checkpoint.",
        r"Bold marks the highest success and AUC across methods within each task",
        r"and training seed, with all ties determined from values before rounding.}",
        r"\label{tab:seed-success}\\",
        r"\toprule", header, r"\midrule", r"\endfirsthead",
        r"\caption[]{\normalsize Individual-seed validation and test results (continued).}\\",
        r"\toprule", header, r"\midrule", r"\endhead",
        r"\midrule", r"\multicolumn{8}{c}{\emph{Continued on next page}}\\",
        r"\endfoot", r"\bottomrule", r"\endlastfoot",
    ]
    for env_index, env in enumerate(ENVIRONMENTS):
        name = env.paper_name.replace("_", r"\_")
        for method_index, method in enumerate(METHODS):
            if env_index or method_index:
                lines.append(r"\addlinespace[4pt]")
            label = r"\method{} (Ours)" if method == "NQRL" else method
            lines.append(r"\multicolumn{8}{@{}c@{}}{\textbf{" + name + r"}\quad " + label
                         + r"\quad ($n_{\mathrm{val}}=" + str(env.eval_episodes) + r"$)}\\*")
            for seed in SEEDS:
                row = by_key[env.paper_name, method, seed]

                def cell(metric):
                    value = float(row[metric])
                    text = f"{100 * value:.1f}"
                    if value == best[env.paper_name, seed, metric]:
                        text = r"\mathbf{" + text + "}"
                    return "$" + text + "$"

                style = (r"\mathbf" if float(row["test_success"]) ==
                         best[env.paper_name, seed, "test_success"] else "")
                count = (r"\nqrlcount{" + style + "}{" +
                         str(int(row["successful_episodes"])) + "}{" +
                         str(int(row["test_episodes"])) + "}")
                cells = [str(seed), cell("peak_validation_success"),
                         str(int(row["selected_env_steps"])),
                         f"{100 * float(row['selected_budget_fraction']):.1f}",
                         cell("final_validation_success"), cell("validation_auc_20_100"),
                         cell("test_success"), count]
                # Repeat the task at every method so each five-seed block is self-contained.
                suffix = r" \\" if seed == SEEDS[-1] else r" \\*"
                lines.append(" & ".join(cells) + suffix)
    lines += [r"\end{longtable}", r"\endgroup"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", action="store_true", help="Re-extract validation and selected-checkpoint tests for all 420 runs")
    args = parser.parse_args()
    if args.collect:
        rows, audit = collect_runs()
        with (DATA_ROOT / CSV_NAME).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (DATA_ROOT / AUDIT_NAME).write_text(json.dumps(audit, indent=2) + "\n")
    else:
        rows = list(csv.DictReader((DATA_ROOT / CSV_NAME).open()))
        validate_runs(rows)
    TABLE_PATH.write_text(render_table(rows))
    print(f"Wrote {TABLE_PATH}: 14 tasks, 6 methods, 5 seeds, {len(rows)} validation/test rows; 84 mean/SD groups match each of the main success and AUC tables.")


if __name__ == "__main__":
    main()

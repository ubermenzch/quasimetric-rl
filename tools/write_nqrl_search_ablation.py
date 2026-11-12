#!/usr/bin/env python3
"""Collect audited K=0/8/16/32 runs and reproduce search-ablation tables.

Use --collect to import the configuration audit and recheck archived file hashes.
Without it, regenerate the tables from the bundled per-run CSV and provenance.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
from pathlib import Path

from nqrl_table_labels import task_label

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "iclr2027"
DATA = PAPER / "data"
STEPS = (0, 8, 16, 32)
SEEDS = tuple(range(1000, 1005))
SELECTED = ("AntNavigate-v4", "FetchPush", "PandaPickAndPlaceJoints-v3",
            "PandaPushJoints-v3", "PandaSlide-v3", "Pusher-v4")
ALIASES = {"AntMaze_UMaze-v5": "AntMaze UMaze-v5",
           "PointMaze_Medium-v3": "PointMaze Medium-v3"}
FIELDS = ("environment", "training_seed", "search_steps", "test_success",
          "test_episodes", "test_seed_start", "interaction_budget", "queue_index",
          "task_id", "search_mode", "inner_lr")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def csv_text(rows):
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def collect(audit_path):
    audit = json.loads(audit_path.read_text())
    accepted = [r for r in audit["runs"] if r["accepted"] and r["search_steps"] in STEPS
                and ALIASES.get(r["environment"], r["environment"]) in SELECTED]
    assert len(accepted) == 120
    records = []
    for r in accepted:
        assert not r["configuration_differences"]
        assert all(r["architecture_checks"].values()) and all(r["integrity_checks"].values())
        assert r["search_mode"] == "min" and r["inner_lr"] == 1.0
        assert r["dynamics_loss"] == dict(weight=0.1, distance="iqe_mse", mse_weight=1.0, iqe_weight=1.0)
        for name, expected in r["source_sha256"].items():
            assert sha(Path(r["run_dir"]) / name) == expected, (r["task_id"], name)
        row = {key: r[key] for key in FIELDS}
        row["environment"] = ALIASES.get(row["environment"], row["environment"])
        records.append(row)
    records.sort(key=lambda r: (r["environment"].casefold(), r["search_steps"], r["training_seed"]))
    runs_path = DATA / "goal_distance_steps_runs.csv"
    runs_path.write_text(csv_text(records))
    provenance = {
        "criterion": audit["criterion"], "metric": audit["metric"],
        "normalized_missing_defaults": audit["normalized_missing_defaults"],
        "source_audit_sha256": sha(audit_path),
        "runs_csv_sha256": sha(runs_path), "run_count": 120,
        "selected_task_count": 6, "selected_run_count": 120,
        "selection": "Six tasks selected after inspection for an overall upward success trend; not representative of the full benchmark average.",
        "runs": accepted,
    }
    (DATA / "goal_distance_steps_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def validated_runs():
    runs = read_csv(DATA / "goal_distance_steps_runs.csv")
    provenance = json.loads((DATA / "goal_distance_steps_provenance.json").read_text())
    assert sha(DATA / "goal_distance_steps_runs.csv") == provenance["runs_csv_sha256"]
    environments = sorted({r["environment"] for r in runs}, key=str.casefold)
    by_key = {(r["environment"], int(r["search_steps"]), int(r["training_seed"])): r for r in runs}
    assert len(runs) == len(by_key) == provenance["run_count"] == 120
    assert set(environments) == set(SELECTED)
    assert set(by_key) == {(e, k, s) for e in environments for k in STEPS for s in SEEDS}
    main = {(r["environment"], int(r["training_seed"])): r
            for r in read_csv(DATA / "seed_test_success.csv") if r["method"] == "NQRL"}
    proof = {r["task_id"]: r for r in provenance["runs"]}
    assert len(proof) == len(provenance["runs"]) == 120
    for r in runs:
        p = proof[r["task_id"]]
        assert p["accepted"] and not p["configuration_differences"]
        assert all(p["architecture_checks"].values()) and all(p["integrity_checks"].values())
        for field in ("test_success", "search_steps", "training_seed", "test_episodes", "test_seed_start", "interaction_budget"):
            assert float(r[field]) == p[field]
        assert r["search_mode"] == "min" and float(r["inner_lr"]) == 1
        reference = main[r["environment"], int(r["training_seed"])]
        if int(r["search_steps"]) == 32:
            assert float(r["test_success"]) == float(reference["test_success"])
    return runs, environments, by_key


def summary(runs, environments):
    params = {r["environment"]: int(r["trainable_parameters"])
              for r in read_csv(DATA / "nqrl_parameter_counts.csv")}
    rows = []
    for env in environments:
        for k in STEPS:
            group = [r for r in runs if r["environment"] == env and int(r["search_steps"]) == k]
            values = [float(r["test_success"]) for r in group]
            budgets = {int(r["interaction_budget"]) for r in group}
            assert len(group) == 5 and len(budgets) == 1
            rows.append(dict(environment=env, model_parameters=params[env], interaction_budget=budgets.pop(),
                             num_seeds=5, gradient_steps=k, success_mean=statistics.mean(values),
                             success_spread=statistics.stdev(values)))
    return rows


def render_table(rows):
    envs = list(dict.fromkeys(r["environment"] for r in rows))
    values = {(r["environment"], r["gradient_steps"]): r for r in rows}
    caption = r"""Completion-search ablation on six tasks. Each entry is test success
(\%): mean $\pm$ sample standard deviation across five training seeds.
AQRL/$K$ uses $K$ search updates; $K=0$ disables search.
All other settings match the corresponding main-table runs.
Macro rows weight tasks equally; $\Delta_{32-0}$ is in percentage points (pp).
Bold marks the highest mean among the four budgets.
Individual-seed ablation results for these six tasks are in
Appendix~\ref{app:search-ablation}."""
    label = "tab:goal-distance-steps"
    lines = [r"\begin{table*}[!t]", "\\caption{" + caption + "}", "\\label{" + label + "}",
             r"\centering\nqrltablestyle", r"\setlength{\tabcolsep}{3pt}",
             r"\renewcommand{\arraystretch}{1.12}", r"\nqrlpairsetup{$\mathbf{99.99}$}{$\mathbf{99.99}$}",
             r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccc@{}}", r"\toprule",
             r"Task & \method/0 & \method/8 & \method/16 & \method/32 & $\Delta_{32-0}$ \\", r"\midrule"]
    for env in envs:
        group = [values[env, k] for k in STEPS]
        best = max(r["success_mean"] for r in group)
        cells = []
        for r in group:
            weight = r"\mathbf" if r["success_mean"] == best else ""
            cells.append(rf"\nqrlpm{{{weight}}}{{{100*r['success_mean']:.2f}}}{{{100*r['success_spread']:.2f}}}")
        delta = 100 * (group[-1]["success_mean"] - group[0]["success_mean"])
        lines.append(task_label(env) + " & " + " & ".join(cells) + f" & ${delta:+.2f}$ " + r"\\")
    macro = [100 * statistics.mean(values[e, k]["success_mean"] for e in envs) for k in STEPS]
    def macro_cell(m, gain=False):
        value = f"{m-macro[0]:+.2f}" if gain else f"{m:.2f}"
        return "$" + (r"\mathbf{" + value + "}" if m == max(macro) else value) + "$"
    lines.extend([r"\midrule", "Macro success & " + " & ".join(macro_cell(m) for m in macro)
                  + f" & ${macro[-1]-macro[0]:+.2f}$ " + r"\\",
                  "Gain over $K=0$ (pp) & " + " & ".join(macro_cell(m, gain=True) for m in macro) + r" & -- \\",
                  r"\bottomrule", r"\end{tabular*}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def render_seeds(by_key):
    lines = [r"\begin{table*}[htbp]",
             r"\caption{Individual-seed test success (\%) for the six search-ablation tasks. Each row compares the same training seed at four search budgets. Means and sample SD appear in \tabref{tab:goal-distance-steps}.}",
             r"\label{tab:search-seeds}", r"\centering\nqrltablestyle",
             r"\renewcommand{\arraystretch}{1.10}", r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccc@{}}",
             r"\toprule", r"Task & Seed & $K=0$ & $K=8$ & $K=16$ & $K=32$ \\", r"\midrule"]
    for i, env in enumerate(SELECTED):
        if i:
            lines.append(r"\midrule")
        for seed in SEEDS:
            values = [100 * float(by_key[env, k, seed]["test_success"]) for k in STEPS]
            lines.append((task_label(env) if seed == SEEDS[0] else "") + f" & {seed} & "
                         + " & ".join(f"${v:.2f}$" for v in values) + " " + r"\\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular*}", r"\end{table*}"]) + "\n"


def generated_files():
    runs, environments, by_key = validated_runs()
    chosen = summary(runs, environments)
    assert len(chosen) == 24
    files = {"data/goal_distance_steps.csv": csv_text(chosen),
             "tables/goal_distance_steps.tex": render_table(chosen),
             "tables/search_seed_success.tex": render_seeds(by_key)}
    audit = {
        "statistic": "sample standard deviation across five training seeds (ddof=1)",
        "aggregate_source_sha256": hashlib.sha256(files["data/goal_distance_steps.csv"].encode()).hexdigest(),
        "verified_main_seed_source_sha256": sha(DATA / "seed_test_success.csv"),
        "runs_source_sha256": sha(DATA / "goal_distance_steps_runs.csv"),
        "selected_six_macro_pct": [100*statistics.mean(r["success_mean"] for r in chosen if r["gradient_steps"] == k) for k in STEPS],
        "numerically_cross_checked_settings": [dict(r, training_seeds=list(SEEDS),
            successes=[float(by_key[r["environment"], r["gradient_steps"], s]["test_success"]) for s in SEEDS]) for r in chosen],
        "main_table_cross_checked_runs": 30,
        "distance_panel": "Removed: previous distance summaries do not describe these audited runs.",
    }
    files["data/goal_distance_spread_audit.json"] = json.dumps(audit, indent=2) + "\n"
    return files


def check():
    for name, expected in generated_files().items():
        assert (PAPER / name).read_text() == expected, f"Regenerate {name}"
    a = json.loads((DATA / "goal_distance_spread_audit.json").read_text())
    assert all(math.isclose(x, y, abs_tol=1e-10) for x, y in zip(
        a["selected_six_macro_pct"], [67.54, 68.92333333333333, 71.28, 72.36666666666667]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", type=Path, help="Path to the archived configuration audit JSON")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.collect:
        collect(args.collect)
    if args.check:
        check()
    else:
        for name, content in generated_files().items():
            (PAPER / name).write_text(content)
        check()
    print("Verified six tasks: 120 runs, 24 settings, 30 main-table matches.")


if __name__ == "__main__":
    main()

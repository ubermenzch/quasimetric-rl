#!/usr/bin/env python3
"""Static consistency checks for the NQRL manuscript and generated data."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import shutil
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP

from PIL import Image
from nqrl_table_labels import task_label

from analyze_nqrl_paper_results import (
    DATA_ROOT,
    ENVIRONMENTS,
    FINAL_MEAN,
    FINAL_STD,
    METHODS,
    PAIRED_METHODS,
    PAPER_ROOT,
    SEEDS,
    artifact_dir,
    interpolate,
    validation_curve,
)
from write_nqrl_dynamics_tables import TABLE_NAMES, read_csv, render_tables, summarize
from analyze_nqrl_baseline_validation import summarize as summarize_baseline_validation
from analyze_nqrl_goal_dimensions import derive as derive_goal_dimensions, render_table as render_goal_dimensions
from analyze_nqrl_execution_metrics import (
    collect_runs as collect_execution_runs,
    collect_valid_arrivals,
    render_table as render_execution_table,
    summarize as summarize_execution,
    summarize_valid_arrivals,
)
from write_nqrl_seed_table import (
    collect_runs as collect_seed_tests, render_table as render_seed_tests,
    validate_runs as validate_seed_tests,
)


def citation_keys(tex: str) -> set[str]:
    keys = set()
    for match in re.finditer(r"\\cite[tp]?\{([^}]*)\}", tex):
        keys.update(key.strip() for key in match.group(1).split(","))
    return keys


def display_value(value: float) -> str:
    # Remove integration roundoff before rounding decimal ties for display.
    return str(Decimal(str(round(value, 12))).quantize(Decimal(".001"), rounding=ROUND_HALF_UP))


def check_best_highlighting(cells, values, *, maximize, context):
    assert len(cells) == len(values), context
    eligible = [value for value in values if value is not None]
    best = (max if maximize else min)(eligible) if eligible else None
    expected = [value is not None and value == best for value in values]
    assert [r"\mathbf" in cell for cell in cells] == expected, (
        f"Incorrect best-value highlighting: {context}"
    )


def expand_appendix(source):
    return re.sub(
        r"\\input\{(appendix/[^}]+)\}",
        lambda match: (PAPER_ROOT / match.group(1)).read_text(),
        source,
    )


def check_figure_text(path):
    if shutil.which("pdffonts") is not None:
        fonts = subprocess.run(["pdffonts", str(path)], check=True,
                               capture_output=True, text=True).stdout
        font_dir = Path(os.environ.get("NQRL_FONT_DIR", PAPER_ROOT))
        times_available = all((font_dir / name).is_file()
                              for name in ("times.ttf", "timesbd.ttf", "timesi.ttf"))
        if times_available:
            assert "TimesNewRoman" in fonts, f"Missing Times New Roman figure font: {path}"
            assert not any(name in fonts for name in ("LiberationSerif", "FreeSerif")), (
                f"Substitute font remains after Times New Roman installation: {path}"
            )
        else:
            assert "LiberationSerif" in fonts, f"Missing Times-compatible figure font: {path}"
            print(f"PENDING: Times New Roman unavailable; substitute font in {path.name}")
        assert "DejaVuSans" not in fonts, f"Obsolete sans-serif figure font: {path}"
    if shutil.which("pdftotext") is None:
        print("SKIP: PDF text-bound checks require pdftotext")
        return
    result = subprocess.run(["pdftotext", "-bbox", str(path), "-"],
                            check=True, capture_output=True)
    root = ET.fromstring(result.stdout)
    pages = root.findall(".//{*}page")
    assert len(pages) == 1, f"Figure must be a single page: {path}"
    page = pages[0]
    width, height = float(page.attrib["width"]), float(page.attrib["height"])
    if path.name == "nqrl_network_22m.pdf":
        assert (width, height) == (560, 366), "Unexpected compact network dimensions"
    if path.name == "dynamics_prediction.pdf":
        assert (width, height) == (560, 300), "Dynamics figure must be 25% shorter at unchanged width"
    if path.name == "state_encoder_comparison.pdf":
        assert (width, height) == (460, 144), "Unexpected state-encoder comparison dimensions"
    if path.name == "goal_fraction_success.pdf":
        assert (width, height) == (560, 132), "Unexpected compact ratio-success/gain panel dimensions"
    if path.name == "task_environments_grid.pdf":
        assert (width, height) == (576, 424), "Unexpected compact environment-grid dimensions"
    if path.name == "task_success_grid.pdf":
        assert (width, height) == (576, 457), "Unexpected compact learning-grid dimensions"
    words = page.findall(".//{*}word")
    assert words, f"Missing figure labels: {path}"
    for word in words:
        x0, y0, x1, y1 = (float(word.attrib[key]) for key in ("xMin", "yMin", "xMax", "yMax"))
        assert 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height, (
            f"Figure text outside page: {path.name}, {word.text}"
        )
    if path.name in ('state_encoder_comparison.pdf', 'nqrl_network_22m.pdf',
                     'goal_fraction_success.pdf', 'task_environments_grid.pdf', 'task_success_grid.pdf'):
        for i, first in enumerate(words):
            for second in words[i+1:]:
                dx = min(float(first.attrib['xMax']), float(second.attrib['xMax'])) - max(
                    float(first.attrib['xMin']), float(second.attrib['xMin']))
                dy = min(float(first.attrib['yMax']), float(second.attrib['yMax'])) - max(
                    float(first.attrib['yMin']), float(second.attrib['yMin']))
                assert dx <= .5 or dy <= .5, (
                    f'Overlapping figure labels: {path.name}: {first.text!r}, {second.text!r}'
                )
    if path.name == "nqrl_network_22m.pdf":
        text = " ".join(word.text or "" for word in words)
        for term in ("LayerNorm", "SiLU", "softplus", "tanh", "shared", "training"):
            assert term in text, f"Missing network structure label: {term}"
        for obsolete in ("569", "512", "1024", "2048", "21.67", "width", "shape"):
            assert obsolete not in text, f"Implementation detail remains in network figure: {obsolete}"
    if path.name == "state_encoder_comparison.pdf":
        text = " ".join(word.text or "" for word in words)
        for term in ("QRL", "NQRL", "State encoder", "Goal encoder", "Non-goal encoder",
                     "reacher_hard"):
            assert term in text, f"Missing state-encoder comparison label: {term}"
        qrl, nqrl = (next(word for word in words if word.text == term) for term in ("QRL", "NQRL"))
        assert float(qrl.attrib['xMax']) < float(nqrl.attrib['xMin'])
        assert abs(float(qrl.attrib['yMin'])-float(nqrl.attrib['yMin'])) < 1, (
            "State-encoder method panels must be side by side"
        )
        for left, right in ((0, 220), (220, 460)):
            panel = [word for word in words if left < float(word.attrib['xMin']) < right]
            state = next(word for word in panel if word.text == 'x')
            encoder = next(word for word in panel if word.text == 'encoder')
            latent = next(word for word in panel if word.text in ('z', 'zg(s)'))
            assert float(state.attrib['xMax']) < float(encoder.attrib['xMin'])
            assert float(encoder.attrib['xMax']) < float(latent.attrib['xMin']), (
                'State-encoder information flow must run left to right'
            )
    if path.name == 'goal_fraction_success.pdf':
        text = ' '.join(word.text or '' for word in words)
        assert 'NQRL gain' in text and 'Gain (pp)' in text and 'Success' in text
    if path.name == "dynamics_prediction.pdf":
        legend = " ".join(word.text or "" for word in words if float(word.attrib['yMax']) < 30)
        for variant in ("IQE-only", "IQE+MSE"):
            for loss in ("IQE loss", "MSE loss"):
                assert f"{variant}: {loss}" in legend, f"Missing dynamics error legend: {variant}: {loss}"
        success_legend = " ".join(word.text or "" for word in words
                                 if 175 < float(word.attrib['yMin']) < 195)
        assert "IQE-only" in success_legend and "IQE+MSE" in success_legend, (
            "Validation-success panel needs its own supervision legend"
        )


def check_goal_distance_steps() -> None:
    from write_nqrl_search_ablation import check, validated_runs, SELECTED, STEPS
    check()
    runs, _, by_key = validated_runs()
    means = {(env, k): statistics.mean(float(by_key[env, k, seed]["test_success"])
                                     for seed in SEEDS) for env in SELECTED for k in STEPS}
    assert sum(means[env, k] > means[env, 0] for env in SELECTED for k in STEPS[1:]) == 15
    assert all(means[env, 32] > means[env, 0] for env in SELECTED)
    assert sum(means[env, 32] == max(means[env, k] for k in STEPS) for env in SELECTED) == 4
    assert sum(float(by_key["Pusher-v4", 32, s]["test_success"]) >
               float(by_key["Pusher-v4", 0, s]["test_success"]) for s in SEEDS) == 2
    paper = (PAPER_ROOT / "nqrl_iclr2027.tex").read_text()
    for name in ("goal_distance_steps", "search_seed_success"):
        assert paper.count("\\input{tables/" + name + ".tex}") == 1
    for obsolete in ("4.72", "42.60", "47.32", "11.08", "five-task ablation", "Panel (b) reports"):
        assert obsolete not in paper
    assert "search_all_tasks" not in paper and "tab:search-all-tasks" not in paper
    for required in ("67.54", "68.92", "71.28", "72.37", "4.83", "21.12"):
        assert required in paper
    main_runs = read_csv(DATA_ROOT / "seed_test_success.csv")
    macro = statistics.mean(float(r["test_success"]) for r in main_runs if r["method"] == "NQRL") * 100
    assert math.isclose(macro, 66.14428571428571, abs_tol=1e-10)
    assert "66.2" not in paper and "66.1" in paper
    assert "$\\mathbf{66.1}$" in (PAPER_ROOT / "tables/main_success.tex").read_text()


def check_goal_dimension_analysis():
    rows, audit = derive_goal_dimensions()
    recorded = read_csv(DATA_ROOT / 'goal_dimension_analysis.csv')
    assert recorded == [{key: str(value) for key, value in row.items()} for row in rows]
    assert json.loads((DATA_ROOT / 'goal_dimension_analysis.json').read_text()) == audit
    paper = (PAPER_ROOT / 'nqrl_iclr2027.tex').read_text()
    experiments = paper.split(r'\section{Experiments}', 1)[1].split(r'\section{Conclusion}', 1)[0]
    assert re.findall(r'\\subsection\{([^}]+)\}', experiments) == [
        'Experimental setup', 'Goal-reaching success', 'Sample efficiency',
        'Gradient-search gains', 'Dynamics prediction error and the MSE constraint',
        'Non-goal coordinate coverage',
    ], 'Coordinate-coverage analysis must be Section 5.6'
    coverage = experiments.split(r'\label{sec:goal-coverage}', 1)[1]
    assert r'\input{tables/goal_dimension_analysis.tex}' in coverage
    assert r'\ref{fig:goal-fraction-success}' in coverage
    assert r'\label{fig:goal-fraction-success}' in coverage
    assert 'figures/goal_fraction_success.pdf' in coverage
    composite = next(block for block in re.findall(r'\\begin\{figure\}.*?\\end\{figure\}', coverage, re.S)
                     if r'\label{fig:goal-fraction-success}' in block)
    assert r'\input{tables/goal_dimension_analysis.tex}' in composite
    assert composite.count(r'\caption{') == 1
    assert r'\includegraphics[width=\textwidth]{figures/goal_fraction_success.pdf}' in composite
    assert r'\begin{minipage}' not in composite, 'Table panel must retain full width at 10 pt'
    assert 'NQRL-minus-QRL gain (pp)' in composite
    assert 'tasks are averaged within' in composite and 'seed;' in composite
    assert '(c) All 14 tasks' in composite and r'$d_s,d_g$' in composite
    assert 'tab:goal-dimensions' not in paper, 'Merged table must use the figure reference'
    for correlations in (audit, audit['qrl_success_correlation']):
        assert f"Pearson $r={correlations['pearson_r']:.3f}$" in coverage
        assert f"Spearman $\\rho={correlations['spearman_rho']:.3f}$" in coverage
    table = (PAPER_ROOT / 'tables/goal_dimension_analysis.tex').read_text()
    assert table == render_goal_dimensions(rows)
    assert r'\begin{table}' not in table and r'\caption{' not in table
    assert r'\begin{tabular*}{\textwidth}' in table
    assert '(c) Task-level results' in table
    assert len(rows) == 14 and [r['environment'] for r in rows] == [e.paper_name for e in ENVIRONMENTS]
    for row in rows:
        prefix = row['environment'].replace('_', r'\_') + ' &'
        line = next(line for line in table.splitlines() if line.startswith(prefix))
        check_best_highlighting(line.split('&')[4:6],
                                [Decimal(row[key]) for key in ('qrl_success', 'nqrl_success')],
                                maximize=True, context=f"Coordinate analysis: {row['environment']}")
    print('OK: all 14 task coordinate fractions, QRL-success and NQRL-gain correlations')


def check_goal_fraction_success():
    runs = read_csv(DATA_ROOT / 'goal_fraction_success_runs.csv')
    tasks = read_csv(DATA_ROOT / 'goal_fraction_success_tasks.csv')
    seed_means = read_csv(DATA_ROOT / 'goal_fraction_success_seed_means.csv')
    curves = read_csv(DATA_ROOT / 'goal_fraction_success_curves.csv')
    audit = json.loads((DATA_ROOT / 'goal_fraction_success_audit.json').read_text())
    assert (len(runs), len(tasks), len(seed_means), len(curves)) == (140, 28, 110, 22)
    assert audit['selected_test_records'] == 140 and audit['unique_fractions'] == 11
    assert audit['axes'] == dict(xlim=[.45, 1.0], ylim=[0, 1], panels=['QRL', 'NQRL'])
    assert audit['training_seeds'] == list(SEEDS)
    for name, digest in audit['source_sha256'].items():
        assert hashlib.sha256((DATA_ROOT/name).read_bytes()).hexdigest() == digest
    key = lambda r: (r['environment'], r['method'], int(r['seed']))
    assert {key(r) for r in runs} == {(e.paper_name, m, s) for e in ENVIRONMENTS
                                     for m in ('QRL', 'NQRL') for s in SEEDS}
    sources = {key(r): r for r in audit['sources']}
    assert len(sources) == 140
    environments = {e.paper_name: e for e in ENVIRONMENTS}
    dims = {r['environment']: r for r in read_csv(DATA_ROOT/'nqrl_parameter_counts.csv')}
    for row in runs:
        env = environments[row['environment']]
        path = artifact_dir(env, row['method'], int(row['seed'])) / 'test.log'
        source = sources[key(row)]
        assert source['test_log'] == str(path)
        assert row['test_sha256'] == source['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        assert len(records) == 1
        record = records[0]
        assert record['split'] == 'test'
        assert float(row['test_success']) == record['succ_rate'] == record['success_count']/record['num_episodes']
        assert int(row['successful_episodes']) == record['success_count']
        assert int(row['num_episodes']) == record['num_episodes'] == env.test_episodes
        assert int(row['selected_env_steps']) == record['env_steps']
        state, goal = int(dims[env.paper_name]['state_dim']), int(dims[env.paper_name]['goal_dim'])
        assert (int(row['state_coordinates']), int(row['goal_coordinates']),
                int(row['non_goal_coordinates'])) == (state, goal, state-goal)
    from fractions import Fraction
    ratio = lambda r: Fraction(int(r['ratio_numerator']), int(r['ratio_denominator']))
    for row in tasks:
        d = dims[row['environment']]
        assert ratio(row) == Fraction(int(d['state_dim'])-int(d['goal_dim']), int(d['state_dim']))
        values = [float(r['test_success']) for r in runs
                  if (r['environment'], r['method']) == (row['environment'], row['method'])]
        assert len(values) == int(row['seeds']) == 5
        assert math.isclose(statistics.mean(values), float(row['success_mean']), abs_tol=1e-12)
        assert math.isclose(statistics.stdev(values), float(row['success_std']), abs_tol=1e-12)
    for method in ('QRL', 'NQRL'):
        selected = [r for r in curves if r['method'] == method]
        fractions = [ratio(r) for r in selected]
        assert len(set(fractions)) == 11 and fractions == sorted(fractions)
        for row in selected:
            members = [r['environment'] for r in tasks if r['method'] == method and ratio(r) == ratio(row)]
            assert row['environments'] == '|'.join(members)
            assert int(row['task_count']) == len(members)
            assert float(row['non_goal_fraction']) == float(ratio(row))
            group = [r for r in seed_means if r['method'] == method and ratio(r) == ratio(row)]
            assert sorted(int(r['seed']) for r in group) == list(SEEDS)
            values = []
            for seed in SEEDS:
                originals = [float(r['test_success']) for r in runs
                             if r['method'] == method and r['environment'] in members and int(r['seed']) == seed]
                mean = math.fsum(originals)/len(members)
                stored = next(r for r in group if int(r['seed']) == seed)
                assert int(stored['task_count']) == len(members)
                assert math.isclose(float(stored['test_success']), mean, abs_tol=1e-12)
                values.append(mean)
            mean = math.fsum(values)/5
            sd = math.sqrt(math.fsum((v-mean)**2 for v in values)/4)
            assert int(row['seeds']) == 5
            assert math.isclose(float(row['success_mean']), mean, abs_tol=1e-12)
            assert math.isclose(float(row['success_std']), sd, abs_tol=1e-12)
    assert b'/ca .18' in (PAPER_ROOT/'figures/goal_fraction_success.pdf').read_bytes()
    print('OK: ratio-success curves match 140 original tests; exact-ratio grouping and five-seed sample SD verified')


def check_goal_fraction_gain():
    from fractions import Fraction
    runs = read_csv(DATA_ROOT/'goal_fraction_gain_runs.csv')
    tasks = read_csv(DATA_ROOT/'goal_fraction_gain_tasks.csv')
    seeds = read_csv(DATA_ROOT/'goal_fraction_gain_seed_means.csv')
    curves = read_csv(DATA_ROOT/'goal_fraction_gain_curves.csv')
    audit = json.loads((DATA_ROOT/'goal_fraction_gain_audit.json').read_text())
    original = {(r['environment'], r['method'], int(r['seed'])): r
                for r in read_csv(DATA_ROOT/'goal_fraction_success_runs.csv')}
    assert (len(runs), len(tasks), len(seeds), len(curves)) == (70, 14, 55, 11)
    assert (audit['original_test_records'], audit['paired_seed_differences']) == (140, 70)
    assert audit['training_seeds'] == list(SEEDS)
    assert audit['axes'] == dict(xlim=[.45, 1.0], ylim=[-20, 100], unit='percentage points')
    for name, digest in audit['source_sha256'].items():
        assert hashlib.sha256((DATA_ROOT/name).read_bytes()).hexdigest() == digest
    gain_by_key = {}
    task_ratios = {}
    for row in runs:
        key = row['environment'], int(row['seed'])
        qrl = original[key[0], 'QRL', key[1]]
        nqrl = original[key[0], 'NQRL', key[1]]
        assert float(row['qrl_success']) == float(qrl['test_success'])
        assert float(row['nqrl_success']) == float(nqrl['test_success'])
        gain = 100*(float(nqrl['test_success'])-float(qrl['test_success']))
        assert math.isclose(float(row['gain_pp']), gain, abs_tol=1e-12)
        fraction = Fraction(int(qrl['non_goal_coordinates']), int(qrl['state_coordinates']))
        assert fraction == Fraction(int(row['ratio_numerator']), int(row['ratio_denominator']))
        assert float(fraction) == float(row['non_goal_fraction'])
        assert key not in gain_by_key
        gain_by_key[key] = gain
        task_ratios[key[0]] = fraction
    assert set(gain_by_key) == {(env.paper_name, seed) for env in ENVIRONMENTS for seed in SEEDS}

    def check_summary(row, values):
        assert len(values) == int(row['seeds']) == 5
        mean = math.fsum(values)/5
        sd = math.sqrt(math.fsum((v-mean)**2 for v in values)/4)
        assert math.isclose(float(row['gain_mean_pp']), mean, abs_tol=1e-12)
        assert math.isclose(float(row['gain_std_pp']), sd, abs_tol=1e-12)

    for row in tasks:
        assert Fraction(int(row['ratio_numerator']), int(row['ratio_denominator'])) == task_ratios[row['environment']]
        check_summary(row, [gain_by_key[row['environment'], seed] for seed in SEEDS])
    fraction = lambda row: Fraction(int(row['ratio_numerator']), int(row['ratio_denominator']))
    assert [fraction(row) for row in curves] == sorted(set(task_ratios.values()))
    for row in curves:
        members = [env.paper_name for env in ENVIRONMENTS if task_ratios[env.paper_name] == fraction(row)]
        assert row['environments'].split('|') == members
        assert int(row['task_count']) == len(members)
        values = []
        for seed in SEEDS:
            mean = math.fsum(gain_by_key[env, seed] for env in members)/len(members)
            stored = [r for r in seeds if fraction(r) == fraction(row) and int(r['seed']) == seed]
            assert len(stored) == 1 and int(stored[0]['task_count']) == len(members)
            assert math.isclose(float(stored[0]['gain_pp']), mean, abs_tol=1e-12)
            values.append(mean)
        check_summary(row, values)
        assert -20 <= float(row['gain_mean_pp'])-float(row['gain_std_pp'])
        assert float(row['gain_mean_pp'])+float(row['gain_std_pp']) <= 100
    print('OK: gain curve and sample SD match all 70 paired-seed differences; all bands fit the displayed range')


def check_dynamics_validation():
    runs = read_csv(DATA_ROOT / 'dynamics_validation_runs.csv')
    curves = read_csv(DATA_ROOT / 'dynamics_validation_curves.csv')
    audit = json.loads((DATA_ROOT / 'dynamics_validation_audit.json').read_text())
    assert len(runs) == audit['checkpoints'] == 150 and len(curves) == 50
    assert len(audit['sources']) == audit['runs'] == 6
    auxiliary = {(r['variant'], int(r['seed'])): r for r in read_csv(DATA_ROOT / 'dynamics_mse_runs.csv')}
    assert {(r['variant'], r['seed']) for r in audit['sources']} == set(auxiliary)
    for source in audit['sources']:
        key = source['variant'], source['seed']
        path = Path(auxiliary[key]['run_dir']) / 'eval.log'
        assert str(path) == source['eval_log']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source['sha256']
        records, duplicates = validation_curve(path)
        assert duplicates['duplicate_keys'] == 0
        archived = [r for r in runs if (r['variant'], int(r['seed'])) == key]
        assert [int(r['env_steps']) for r in records] == list(range(20000, 500001, 20000))
        assert len(archived) == len(records) == 25
        for actual, saved in zip(records, archived):
            assert actual['split'] == 'validation'
            assert int(saved['env_steps']) == actual['env_steps']
            assert int(saved['num_episodes']) == actual['num_episodes'] == 100
            assert float(saved['validation_success']) == actual['succ_rate'] == actual['success_count'] / 100
            assert int(saved['episode_seed']) == actual['seed']
            assert int(saved['episode_seed_end']) == actual['seed_end']
    assert {(r['variant'], int(r['env_steps'])) for r in curves} == {
        (variant, step) for variant in ('A01', 'A05') for step in range(20000, 500001, 20000)}
    for row in curves:
        values = [float(r['validation_success']) for r in runs
                  if (r['variant'], r['env_steps']) == (row['variant'], row['env_steps'])]
        assert len(values) == int(row['seeds']) == 3
        assert all(0 <= value <= 1 for value in values)
        assert math.isclose(statistics.mean(values), float(row['success_mean']), abs_tol=1e-14)
        assert math.isclose(statistics.stdev(values), float(row['success_std']), abs_tol=1e-14)
        if row['variant'] == 'A01':
            assert float(row['success_mean']) == float(row['success_std']) == 0
    print('OK: dynamics success panel checked against 150 original validation records; all means and sample SDs match')


def main() -> None:
    tex_path = PAPER_ROOT / "nqrl_iclr2027.tex"
    bib_path = PAPER_ROOT / "nqrl_iclr2027.bib"
    readme_path = PAPER_ROOT / "README.md"
    reference_audit_path = PAPER_ROOT / "REFERENCE_AUDIT.md"
    tex = tex_path.read_text(encoding="utf-8")
    table_inputs = re.findall(r"\\input\{(tables/[^}]+)\}", tex)
    expected_tables = {
        "tables/goal_distance_steps.tex",
        "tables/main_success.tex",
        "tables/learning_process.tex",
        "tables/evaluation_schedule.tex",
        "tables/validation_auc.tex",
        "tables/baseline_implementations.tex",
        "tables/task_goal_conditions.tex",
        "tables/experiment_configuration.tex",
        "tables/goal_dimension_analysis.tex",
        "tables/seed_test_success.tex",
    }
    assert set(table_inputs) == expected_tables
    assert len(table_inputs) == len(expected_tables)
    assert table_inputs == [
        "tables/main_success.tex", "tables/validation_auc.tex", "tables/goal_distance_steps.tex",
        "tables/goal_dimension_analysis.tex", "tables/baseline_implementations.tex",
        "tables/task_goal_conditions.tex",
        "tables/experiment_configuration.tex", "tables/evaluation_schedule.tex",
        "tables/learning_process.tex", "tables/seed_test_success.tex",
    ], "Table fragments must follow manuscript order, with coordinate coverage inside Figure 4"
    seed_rows = list(csv.DictReader((DATA_ROOT / "seed_test_success.csv").open()))
    validate_seed_tests(seed_rows)
    collected_seed_rows, seed_audit = collect_seed_tests()
    assert seed_rows == [{key: str(value) for key, value in row.items()}
                         for row in collected_seed_rows], "Per-seed table data differ from original validation/test records"
    assert json.loads((DATA_ROOT / "seed_test_success_audit.json").read_text()) == seed_audit
    seed_table = (PAPER_ROOT / "tables/seed_test_success.tex").read_text()
    assert seed_table == render_seed_tests(seed_rows), "Per-seed table differs from verified validation/test data"
    learning_seed_rows = {(r['environment'], r['method'], int(r['seed'])): r for r in
                          csv.DictReader((DATA_ROOT / "learning_runs.csv").open())}
    for row in seed_rows:
        key = row['environment'], row['method'], int(row['training_seed'])
        prior = learning_seed_rows[key]
        assert float(row['test_success']) == float(prior['selected_test_success'])
        assert float(row['peak_validation_success']) == float(prior['selected_validation_success'])
        assert float(row['selected_budget_fraction']) == float(prior['selected_checkpoint_fraction'])
        assert math.isclose(float(row['validation_auc_20_100']), float(prior['auc_20_100']),
                            rel_tol=0, abs_tol=1e-12)
        assert math.isclose(float(row['peak_validation_success']) - float(row['final_validation_success']),
                            float(prior['peak_to_final_validation_gap']), rel_tol=0, abs_tol=1e-12)
    seed_data_lines = [line for line in seed_table.splitlines() if re.match(r'100[0-4] &', line)]
    assert len(seed_data_lines) == 420
    assert len(re.findall(r'\\multicolumn\{8\}\{@\{\}c@\{\}\}\{\\textbf', seed_table)) == 84
    seed_by_key = {(r['environment'], r['method'], int(r['training_seed'])): r for r in seed_rows}
    for env_index, env in enumerate(ENVIRONMENTS):
        for seed_index, seed in enumerate(SEEDS):
            method_cells = [seed_data_lines[env_index * 30 + method_index * 5 + seed_index].split(' & ')
                            for method_index in range(len(METHODS))]
            assert all(len(cells) == 8 and cells[0] == str(seed) for cells in method_cells)
            for column, metric in ((1, 'peak_validation_success'), (4, 'final_validation_success'),
                                   (5, 'validation_auc_20_100'), (6, 'test_success'),
                                   (7, 'successful_episodes')):
                values = [float(seed_by_key[env.paper_name, method, seed][metric]) for method in METHODS]
                check_best_highlighting([cells[column] for cells in method_cells], values, maximize=True,
                                        context=f'Individual runs: {env.paper_name}/{seed}/{metric}')
            assert all(r'\mathbf' not in cells[column] for cells in method_cells for column in (0, 2, 3))
    print("OK: all 420 individual runs match original validation/test records and checkpoint selection; 84 mean/SD groups match each of the main success and AUC tables")
    task_goals = (PAPER_ROOT / "tables/task_goal_conditions.tex").read_text()
    task_goal_rows = [line.split(" & ") for line in task_goals.splitlines()
                      if " & " in line and not line.startswith("Task & ")]
    assert [row[0] for row in task_goal_rows] == [
        env.paper_name.replace("_", r"\_") for env in ENVIRONMENTS
    ], "Task-goal table must cover all fourteen tasks in manuscript order"
    goal_dimensions = {row["environment"]: int(row["goal_dim"]) for row in
                       csv.DictReader((DATA_ROOT / "nqrl_parameter_counts.csv").open())}
    for env, row in zip(ENVIRONMENTS, task_goal_rows):
        assert len(row) == 4, f"Malformed task-goal row: {env.paper_name}"
        indices = tuple(map(int, row[2].strip("$").split(",")))
        expected_indices = ((3, 4, 5) if env.paper_name.startswith("Fetch") else
                            tuple(range(goal_dimensions[env.paper_name])))
        assert indices == expected_indices, f"Incorrect goal indices: {env.paper_name}"
    main_text, appendix_text = tex.split(r"\appendix", 1)
    assert r'\input{tables/goal_dimension_analysis.tex}' not in appendix_text
    for retired in ("tables/execution_metrics.tex", "sec:first-arrival", "tab:execution", "app:execution"):
        assert retired not in tex, f"Retired arrival metric remains in the paper: {retired}"
    assert not re.search(r"success.state (?:step )?fraction", tex, re.I)
    arrival_table = (PAPER_ROOT / "archive/execution_metrics.tex").read_text()
    assert "longtable" not in arrival_table and "Success-state" not in arrival_table
    assert "Test success" not in arrival_table and r"\ref{tab:main}" in arrival_table
    arrival_rows = [line.split(" & ") for line in arrival_table.splitlines()
                    if " & " in line and not line.startswith("Task & ")]
    assert len(arrival_rows) == 14 and all(len(row) == 7 for row in arrival_rows)
    assert [row[0] for row in arrival_rows] == [env.paper_name.replace("_", r"\_")
                                             for env in ENVIRONMENTS]
    assert not re.search(r'\^\{\([1-4]\)\}', arrival_table)
    for name in ('main_success', 'validation_auc', 'learning_process',
                 'experiment_configuration', 'goal_dimension_analysis', 'goal_distance_steps',
                 'seed_test_success'):
        table = (PAPER_ROOT / f'tables/{name}.tex').read_text()
        assert '(Ours)' in table, f'Missing authorship marker in {name}'
    for relative in table_inputs:
        tex = tex.replace(
            rf"\input{{{relative}}}",
            (PAPER_ROOT / relative).read_text(encoding="utf-8"),
        )
    tex = expand_appendix(tex)
    # Discretionary line breaks do not change a task label.
    tex = tex.replace(r"\allowbreak{}", "")
    assert len(re.findall(r'\\begin\{table\*?\}', tex)) == 8, (
        'Expected eight floating tables; coordinate coverage stays in Figure 4'
    )
    assert len(re.findall(r'\\begin\{longtable\}', tex)) == 1, "Expected one additional per-seed longtable"
    assert r'\usepackage{longtable}' in tex
    assert not re.search(r'\\(?:small|scriptsize|footnotesize|tiny|fontsize|resizebox|scalebox)\b', tex), (
        'Manuscript tables, algorithm and appendix contents must retain the template font size'
    )
    assert r'\begin{table}' not in (PAPER_ROOT / 'tables/goal_dimension_analysis.tex').read_text(), (
        'The merged coordinate table must not retain an independent table number'
    )
    bib = bib_path.read_text(encoding="utf-8")
    prose = "\n".join((tex, bib, readme_path.read_text(encoding="utf-8")))

    assert "\\title{Nearest-Goal Quasimetric Reinforcement Learning for Partially Specified Goals}" in tex
    abstract = tex.split(r'\begin{abstract}', 1)[1].split(r'\end{abstract}', 1)[0]
    assert 'nearby valid goal completion' not in abstract
    assert 'latent goal completion with low estimated reaching cost' in abstract
    assert r"\subsection*{AI use statement}" in tex
    ai_statement = tex.split(r"\subsection*{AI use statement}", 1)[1].split(r"\subsection*", 1)[0]
    for use in ('experimental logs', 'interpretation', 'research hypotheses', 'translation',
                'language editing', 'responsibility'):
        assert use in ai_statement, f'AI disclosure omits actual use: {use}'
    forbidden = ("CQRL", "Min32", "TBD", "placeholder", "32-step")
    found = [term for term in forbidden if term.lower() in prose.lower()]
    assert not found, f"Forbidden manuscript terms remain: {found}"

    bib_keys = set(re.findall(r"^@\w+\{([^,]+),", bib, flags=re.MULTILINE))
    cited = citation_keys(tex)
    required_method_sources = {
        'wang2023qrl', 'wang2022iqe', 'ghosh2021gcsl',
        'eysenbach2021clearning', 'eysenbach2022crl', 'zheng2024tdinfonce',
    }
    assert required_method_sources <= cited, 'Missing method or baseline attribution'
    assert r"\nocite" not in tex, "Bibliography must contain only actual citations"
    assert cited == bib_keys, f"Citation mismatch: missing={cited-bib_keys}, unused={bib_keys-cited}"
    reference_audit = reference_audit_path.read_text(encoding="utf-8")
    audited_keys = set(
        re.findall(r"^\| ([a-z][A-Za-z0-9_-]*) \|", reference_audit, flags=re.MULTILINE)
    )
    assert audited_keys == cited, (
        f"Reference-audit mismatch: missing={cited-audited_keys}, "
        f"extra={audited_keys-cited}"
    )
    assert bib.count("{") == bib.count("}"), "Unbalanced BibTeX braces"
    assert "https://openreview.net/forum?id=tc5qisoB-C" in bib

    labels = re.findall(r"\\label\{([^}]+)\}", tex)
    refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", tex))
    assert len(labels) == len(set(labels)), "Duplicate LaTeX labels"
    assert refs <= set(labels), f"Undefined LaTeX references: {refs-set(labels)}"

    uncommented = re.sub(r"(?m)(?<!\\)%.*$", "", tex)
    assert uncommented.count("{") == uncommented.count("}"), "Unbalanced braces"
    assert uncommented.count("\\begin{") == uncommented.count("\\end{"), (
        "Unbalanced LaTeX environments"
    )

    budget_labels = {
        200_000: "200k",
        200_400: "200k",
        400_000: "400k",
        500_000: "500k",
    }
    for env_idx, env in enumerate(ENVIRONMENTS):
        label = task_label(env.paper_name)
        prefix = f"{label} ({budget_labels[env.budget]}) &"
        lines = [line for line in tex.splitlines() if line.startswith(prefix)]
        assert len(lines) == 1, f"Expected one result row for {env.paper_name}"
        pairs = re.findall(r"\\nqrlpm\{(?:\\mathbf)?\}\{(\d?\.\d{3})\}\{(\d?\.\d{3})\}", lines[0])
        pairs = [tuple('0' + v if v.startswith('.') else v for v in pair) for pair in pairs]
        expected = [
            (f"{FINAL_MEAN[method][env_idx]:.3f}", f"{FINAL_STD[method][env_idx]:.3f}")
            for method in METHODS
        ]
        assert pairs == expected, f"Main-table mismatch for {env.paper_name}: {pairs} != {expected}"
        cells = lines[0].split("&")[1:]
        best = max(FINAL_MEAN[method][env_idx] for method in METHODS)
        assert [r"\mathbf" in cell for cell in cells] == [
            FINAL_MEAN[method][env_idx] == best for method in METHODS
        ], f"Incorrect winner highlighting for {env.paper_name}"

    main_table = tex.split(r"\label{tab:main}", 1)[1].split(r"\end{table*}", 1)[0]
    macro_line = next(line for line in main_table.splitlines() if line.startswith("Macro mean &"))
    displayed_macro = [v if not v.startswith('.') else '0' + v
                       for v in re.findall(r"\d?\.\d{3}", macro_line)]
    expected_macro = [
        str((sum(Decimal(str(value)) for value in FINAL_MEAN[method]) / len(ENVIRONMENTS)).quantize(
            Decimal(".001"), rounding=ROUND_HALF_UP
        )) for method in METHODS
    ]
    assert displayed_macro == expected_macro, "Main-table macro means do not match the reported rows"
    check_best_highlighting(macro_line.split("&")[1:],
                            [statistics.mean(FINAL_MEAN[method]) for method in METHODS],
                            maximize=True, context="Main success macro means")

    with (DATA_ROOT / "final_success.csv").open(encoding="utf-8") as handle:
        final_rows = list(csv.DictReader(handle))
    assert len(final_rows) == len(ENVIRONMENTS) * len(METHODS)
    task_order = [env.paper_name for env in ENVIRONMENTS]
    assert task_order == sorted(task_order, key=str.casefold)
    assert list(dict.fromkeys(row["environment"] for row in final_rows)) == task_order
    for filename in ("main_success.tex", "evaluation_schedule.tex", "validation_auc.tex"):
        table = (PAPER_ROOT / "tables" / filename).read_text().replace(r"\allowbreak{}", "")
        positions = [table.index((env.paper_name.replace("_", r"\_")
                                  if filename == "evaluation_schedule.tex"
                                  else task_label(env.paper_name)) + " ") for env in ENVIRONMENTS]
        assert positions == sorted(positions), f"Task order differs in {filename}"

    with (DATA_ROOT / "per_task_auc.csv").open(
        encoding="utf-8"
    ) as handle:
        task_auc_rows = list(csv.DictReader(handle))
        auc_rows = {}
        for row in task_auc_rows:
            target = auc_rows.setdefault(row["environment"], {})
            target[f'{row["method"]}_auc_20_100'] = row["auc_mean"]
            target[f'{row["method"]}_auc_std'] = row["auc_std"]
    assert len(task_auc_rows) == 84
    assert list(auc_rows) == task_order
    checkpoints = read_csv(DATA_ROOT / 'validation_checkpoints.csv')
    curve_rows = read_csv(DATA_ROOT / 'per_task_learning_curves.csv')
    checked_curve_points = 0
    for env in ENVIRONMENTS:
        for method in METHODS:
            curves = []
            for seed in SEEDS:
                records, _ = validation_curve(artifact_dir(env, method, seed) / 'eval.log')
                original = [(float(r['env_steps']), float(r['succ_rate'])) for r in records]
                archived = [(float(r['env_steps']), float(r['validation_success']))
                            for r in checkpoints
                            if (r['environment'], r['method'], int(r['seed']))
                            == (env.paper_name, method, seed)]
                assert archived == original, (env.paper_name, method, seed)
                curves.append(original)
            start = max(c[0][0] for c in curves)
            grid = sorted({x for curve in curves for x, _ in curve if start <= x <= env.budget})
            summaries = [r for r in curve_rows
                         if (r['environment'], r['method']) == (env.paper_name, method)]
            assert [float(r['env_steps']) for r in summaries] == grid
            for step, row in zip(grid, summaries):
                values = [interpolate(curve, step) for curve in curves]
                assert int(row['seeds']) == len(values) == 5
                assert all(0 <= value <= 1 for value in values)
                assert abs(statistics.mean(values) - float(row['success_mean'])) < 1e-12
                assert abs(statistics.stdev(values) - float(row['success_std'])) < 1e-12
                checked_curve_points += 1
    assert checked_curve_points == len(curve_rows)
    grid_audit = json.loads((DATA_ROOT / 'learning_grid_audit.json').read_text())
    assert grid_audit['layout'] == [4, 4]
    assert grid_audit['empty_cells'] == [[3, 2], [3, 3]]
    assert [p['environment'] for p in grid_audit['panels']] == task_order
    assert [(p['row'], p['column']) for p in grid_audit['panels']] == [
        (i // 4, i % 4) for i in range(14)]
    assert all(p['ylim'] == [0, 1] for p in grid_audit['panels'])
    assert grid_audit['source_sha256'] == hashlib.sha256(
        (DATA_ROOT / 'per_task_learning_curves.csv').read_bytes()).hexdigest()
    assert 'sample standard deviation (ddof=1)' in grid_audit['band']
    seed_aucs = read_csv(DATA_ROOT / "per_task_auc_runs.csv")
    assert len(seed_aucs) == 420
    for row in task_auc_rows:
        group = [r for r in seed_aucs
                 if (r["environment"], r["method"]) == (row["environment"], row["method"])]
        assert sorted(int(r["seed"]) for r in group) == list(SEEDS)
        values = [float(r["normalized_auc"]) for r in group]
        assert abs(statistics.mean(values) - float(row["auc_mean"])) < 1e-12
        assert abs(statistics.stdev(values) - float(row["auc_std"])) < 1e-12
    auc_block = tex.split(r"\label{tab:auc-by-task}", 1)[1].split(r"\end{table}", 1)[0]
    for env in ENVIRONMENTS:
        label = task_label(env.paper_name)
        candidates = [
            line for line in auc_block.splitlines()
            if line.startswith(f"{label} &")
        ]
        assert len(candidates) == 1, f"Expected one AUC row for {env.paper_name}"
        displayed = [v if not v.startswith('.') else '0' + v
                     for v in re.findall(r"\d?\.\d{3}", candidates[0])]
        expected = [
            display_value(float(auc_rows[env.paper_name][f'{method}_{field}']))
            for method in METHODS for field in ('auc_20_100', 'auc_std')
        ]
        assert displayed == expected, (
            f"AUC-table mismatch for {env.paper_name}: {displayed} != {expected}"
        )
        check_best_highlighting(
            candidates[0].split("&")[1:],
            [float(auc_rows[env.paper_name][f'{method}_auc_20_100']) for method in METHODS],
            maximize=True, context=f"AUC: {env.paper_name}",
        )

    with (DATA_ROOT / "aggregate_learning_metrics.csv").open(
        encoding="utf-8"
    ) as handle:
        learning_rows = {row["method"]: row for row in csv.DictReader(handle)}
    with (DATA_ROOT / "aggregate_training_scalars.csv").open(
        encoding="utf-8"
    ) as handle:
        scalar_rows = {row["method"]: row for row in csv.DictReader(handle)}
    process_block = tex.split(r"\label{tab:process}", 1)[1].split(
        r"\end{table*}", 1
    )[0].replace("\n", " ")
    process_rows = {}
    process_cells = {}
    for method in METHODS:
        token = re.escape(r"\method{} (Ours)" if method == "NQRL" else method)
        match = re.search(token + r"\s*&(.+?)\\\\", process_block)
        assert match, f"Missing process-table row for {method}"
        process_rows[method] = [float(value) for value in re.findall(r"\d+\.\d+", match.group(1))]
        process_cells[method] = match.group(1).split("&")
    for method in METHODS:
        learning = learning_rows[method]
        scalars = scalar_rows[method]
        expected = [
            round(float(learning["mean_final_to_peak_ratio_nonzero"]), 3),
            round(float(learning["mean_peak_to_final_gap"]), 3),
            round(float(scalars["mean_final_replay_successful_transition_fraction"]), 3),
        ]
        displayed = process_rows[method]
        assert displayed == expected, (
            f"Process-table mismatch for {method}: {displayed} != {expected}"
        )
    for index, (records, key, maximize) in enumerate((
        (learning_rows, "mean_final_to_peak_ratio_nonzero", True),
        (learning_rows, "mean_peak_to_final_gap", False),
        (scalar_rows, "mean_final_replay_successful_transition_fraction", True),
    )):
        check_best_highlighting([process_cells[method][index] for method in METHODS],
                                [float(records[method][key]) for method in METHODS],
                                maximize=maximize, context=f"Training diagnostics: {key}")

    audit = json.loads((DATA_ROOT / "audit_summary.json").read_text(encoding="utf-8"))
    assert len(ENVIRONMENTS) == 14
    expected_runs = len(ENVIRONMENTS) * len(METHODS) * len(SEEDS)
    runs_per_method = len(ENVIRONMENTS) * len(SEEDS)
    assert audit["local_test_audit"]["audited_seed_runs"] == expected_runs
    assert audit["local_test_audit"]["missing_seed_runs"] == 0
    assert audit["local_test_audit"]["summary_mismatches"] == []
    assert audit["local_test_audit"]["matched_method_environment_groups"] == 84
    assert audit["local_test_audit"]["nonfinite_test_metrics"] == []
    curve_audit = audit["learning_curve_analysis"]
    assert curve_audit["excluded_environments"] == []
    assert curve_audit["runs_per_method"] == runs_per_method
    assert curve_audit["runs_with_duplicate_validation_keys"] == 62
    assert curve_audit["duplicate_validation_keys"] == 62
    assert curve_audit["conflicting_duplicate_validation_keys"] == 44
    scalar_audit = json.loads(
        (DATA_ROOT / "training_scalar_audit.json").read_text(encoding="utf-8")
    )
    assert scalar_audit["audited_runs"] == expected_runs
    assert scalar_audit["missing_runs"] == 0
    assert scalar_audit["runs_without_complete_marker"] == []
    assert scalar_audit["invalid_test_logs"] == []
    assert scalar_audit["budget_mismatches"] == []
    assert scalar_audit["event_parse_errors"] == []
    assert scalar_audit["nonfinite_train_or_data_scalars"] == []
    assert scalar_audit["event_files"] > 0
    assert scalar_audit["raw_train_or_data_scalar_values"] > 0
    assert all(int(row["comparison_runs"]) == runs_per_method for row in scalar_rows.values())

    paired_audit = audit["paired_learning_curve_analysis"]
    assert paired_audit["methods"] == list(PAIRED_METHODS)
    assert paired_audit["excluded_environments"] == []
    assert paired_audit["runs_per_method"] == runs_per_method
    assert paired_audit["nqrl_seed_paired_auc_wins_against_task_best_mean_baseline"] == 57
    assert curve_audit["nqrl_seed_paired_auc_wins_against_task_best_mean_baseline"] == 56
    for row in audit["paired_aggregate_learning_metrics"]:
        reference = learning_rows[row["method"]]
        for key in ("auc_20_100", "runs_reaching_0_5_pct"):
            assert abs(float(reference[key]) - row[key]) < 1e-12
    for name in ("Pusher-v4", "FetchSlide"):
        assert float(auc_rows[name]["TD-InfoNCE_auc_20_100"]) > float(auc_rows[name]["NQRL_auc_20_100"])

    low_success = json.loads(
        (DATA_ROOT / "low_success_diagnostics.json").read_text(encoding="utf-8")
    )
    c_learning = low_success["c_learning"]
    assert round(c_learning["FetchPush"]["critic_loss_first_20_mean_over_seeds"], 3) == .491
    assert round(c_learning["FetchPush"]["critic_loss_last_20_mean_over_seeds"], 3) == .039
    panda_names = (
        "PandaSlide-v3",
        "PandaPickAndPlace-v3",
        "PandaPushJoints-v3",
        "PandaPickAndPlaceJoints-v3",
    )
    normalized_returns = [
        round(c_learning[name]["mean_selected_test_return_divided_by_horizon_50"], 3)
        for name in panda_names
    ]
    assert normalized_returns == [.659, .821, .866, .836]
    antmaze = low_success["antmaze"]
    assert [
        round(antmaze[method]["mean_per_seed_validation_peak"], 3)
        for method in ("TD-InfoNCE", "GCSL", "CRL")
    ] == [.030, .094, .062]
    antnavigate = low_success["antnavigate"]
    assert round(antnavigate["TD-InfoNCE"]["mean_test_success"], 3) == .179
    assert round(antnavigate["NQRL"]["mean_test_success"], 3) == .980
    c_antnavigate = low_success["c_learning_antnavigate_seed_rows"]
    assert [round(row["test_success_rate"], 3) for row in c_antnavigate] == [
        .213, 0, 0, 0, 0
    ]
    assert max(
        row["final_replay_successful_transition_fraction"]
        for row in c_antnavigate[1:]
    ) <= .000454
    c_pusher = low_success["additional_task_seed_rows"]["Pusher-v4"]["C-Learning"]
    assert [row["test_success_rate"] for row in c_pusher] == [0, .016, .019, 0, 0]
    pusher = next(env for env in ENVIRONMENTS if env.key == "pusher")
    for seed in SEEDS:
        curve, _ = validation_curve(artifact_dir(pusher, "C-Learning", seed) / "eval.log")
        assert len(curve) == 10
        assert len({row["succ_rate"] for row in curve}) == 1
        assert len({row["epi_return"] for row in curve}) == 1

    with (DATA_ROOT / "learning_runs.csv").open(encoding="utf-8") as handle:
        learning_run_rows = list(csv.DictReader(handle))
    assert len(learning_run_rows) == expected_runs
    for method, count in (("NQRL", 48), ("TD-InfoNCE", 23)):
        assert sum(int(row["ever_reached_0_5"]) for row in learning_run_rows if row["method"] == method) == count
    antnavigate_runs = [
        row for row in learning_run_rows
        if row["environment"] == "AntNavigate-v4"
    ]
    assert sum(
        row["ever_reached_0_5"] == "1"
        for row in antnavigate_runs
        if row["method"] == "NQRL"
    ) == 5
    assert sum(
        row["ever_reached_0_5"] == "1"
        for row in antnavigate_runs
        if row["method"] != "NQRL"
    ) == 0

    figure_names = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", tex)
    assert figure_names == [
        "figures/navigation_goal_concept_cropped.png", "figures/state_encoder_comparison.pdf",
        "figures/dynamics_prediction.pdf",
        "figures/goal_fraction_success.pdf",
        "figures/nqrl_network_22m.pdf",
        "figures/task_environments_grid.pdf", "figures/task_success_grid.pdf",
    ]
    for relative in figure_names:
        path = PAPER_ROOT / relative
        assert path.exists() and path.stat().st_size > 1000, f"Missing figure: {path}"
        if path.suffix == ".pdf":
            assert path.read_bytes().startswith(b"%PDF-"), f"Invalid PDF header: {path}"
            check_figure_text(path)
        else:
            assert path.suffix == ".png", f"Unsupported figure format: {path}"
            with Image.open(path) as figure:
                assert figure.format == "PNG" and figure.mode in ("RGB", "RGBA")
                assert figure.width / 5.5 >= 300, f"Figure below 300 dpi at text width: {path}"
                assert figure.height > 0
                figure.verify()
    introduction = tex.split(r"\section{Introduction}", 1)[1].split(r"\section{Related Work}", 1)[0]
    opening = tex.split(r"\section{Introduction}", 1)[0]
    assert figure_names[0] in opening and r"\begin{figure}[!b]" in opening, (
        "The navigation figure must be queued for the first-page bottom after the abstract"
    )
    assert r"\ref{fig:goal-concept}" in introduction
    encoder_section = tex.split(r"\subsection{Goal/non-goal representation}", 1)[1].split(
        r"\subsection{Nearest conditional completion (NCC)}", 1)[0]
    assert r"\ref{fig:state-encoders}" in encoder_section
    assert r"\label{fig:state-encoders}" in encoder_section
    assert "figures/state_encoder_comparison.pdf" in encoder_section
    assert r"\includegraphics[width=0.82\textwidth]{figures/state_encoder_comparison.pdf}" in encoder_section
    assert "fig:architecture" not in tex and "qrl_nqrl_comparison.pdf" not in tex
    with Image.open(PAPER_ROOT / "figures/navigation_goal_concept.png") as original:
        with Image.open(PAPER_ROOT / figure_names[0]) as cropped:
            assert original.size == (4054, 1663) and cropped.size == (4054, 1383)
            assert cropped.mode == original.mode
            assert cropped.tobytes() == original.crop((0, 280, 4054, 1663)).tobytes()
    dynamics_figure = next(
        block for block in re.findall(r"\\begin\{figure\}.*?\\end\{figure\}", tex, re.S)
        if r"\label{fig:dynamics-prediction}" in block
    )
    assert r"\begin{tabular*}" not in dynamics_figure
    assert "(c) Validation success" in dynamics_figure
    assert r"\begin{table}" not in dynamics_figure and "tab:dynamics-rescue" not in tex
    assert r"Shading shows mean $\pm$ sample SD" in dynamics_figure
    assert "same three training seeds" in dynamics_figure and "100 episodes" in dynamics_figure
    assert "Checkpoint selection prioritizes validation success, then lower mean hitting" in tex
    assert "higher mean episode return, and later training step" in tex
    assert b"/ca .18" in (PAPER_ROOT / 'figures/dynamics_prediction.pdf').read_bytes()
    figure_hashes = [hashlib.sha256((PAPER_ROOT / name).read_bytes()).hexdigest()
                     for name in figure_names]
    assert len(figure_hashes) == len(set(figure_hashes)), "Duplicate figure content"

    dynamics_rows = read_csv(DATA_ROOT / "dynamics_mse_runs.csv")
    dynamics_curves = read_csv(DATA_ROOT / "dynamics_mse_ant_trajectories.csv")
    assert len(dynamics_rows) == 6 and len(dynamics_curves) == 150
    assert {(row["environment"], row["variant"]) for row in dynamics_rows} == {
        ("AntNavigate-v4", "A01"), ("AntNavigate-v4", "A05")
    }
    assert {row["variant"] for row in dynamics_curves} == {"A01", "A05"}
    trajectory_summary = read_csv(DATA_ROOT / 'dynamics_trajectory_summary.csv')
    assert len(trajectory_summary) == 100
    for row in trajectory_summary:
        values = [float(point[row['metric']]) for point in dynamics_curves
                  if point['variant'] == row['variant']
                  and point['env_steps'] == row['window_end']]
        assert len(values) == int(row['seeds']) == 3
        assert math.isclose(statistics.mean(values), float(row['mean']), rel_tol=1e-12)
        assert math.isclose(statistics.stdev(values), float(row['sample_std']), rel_tol=1e-12)
        assert float(row['mean']) - float(row['sample_std']) > 0
    assert all(row["recorded_model_size"] == "GO-QRL-M" for row in dynamics_rows)
    assert all(
        (row["recorded_completion_mode"], int(row["recorded_completion_steps"])) == ("max", 4)
        for row in dynamics_rows
    ), "Auxiliary dynamics protocol does not match the recorded settings"
    for name, content in render_tables(dynamics_rows, dynamics_curves).items():
        assert (PAPER_ROOT / "archive" / name).read_text() == content, (
            f"Dynamics table differs from the accompanying data: {name}"
        )
    dynamics_summary = read_csv(DATA_ROOT / "dynamics_mse_summary.csv")
    assert dynamics_summary == [
        {key: str(value) for key, value in row.items()} for row in summarize(dynamics_rows)
    ]
    dynamics_cells = [line.split("&")[1:] for line in
                      (PAPER_ROOT / "archive" / TABLE_NAMES[0]).read_text().splitlines()
                      if line.startswith(("IQE-only &", "IQE+MSE &"))]
    dynamics_by_variant = {row["variant"]: row for row in dynamics_summary}
    for index, (key, maximize) in enumerate((
        ("iqe_selected20k_mean", False),
        ("mse_selected20k_mean", False),
        ("test_success_mean_pct", True),
    )):
        check_best_highlighting([cells[index] for cells in dynamics_cells],
                                [float(dynamics_by_variant[variant][key])
                                 for variant in ("A01", "A05")],
                                maximize=maximize, context=f"Dynamics: {key}")
    provenance = json.loads((DATA_ROOT / "dynamics_mse_provenance.json").read_text())
    assert len(provenance["antnavigate_runs"]) == 6
    for row in provenance["antnavigate_runs"]:
        assert row["completion_mode"] == "max" and row["completion_steps"] == 4
        assert row["completion_lr"] == .01
        assert "latent_goal_mode=max, latent_goal_steps=4" in row["runtime_model_repr"]
    ant_iqe = [
        row for row in dynamics_rows
        if row["environment"] == "AntNavigate-v4" and row["variant"] == "A01"
    ]
    ant_hybrid = [
        row for row in dynamics_rows
        if row["environment"] == "AntNavigate-v4" and row["variant"] == "A05"
    ]
    assert [float(row["test_success"]) for row in ant_iqe] == [0, 0, 0]
    assert [float(row["test_success"]) for row in ant_hybrid] == [.26, .20, .23]
    assert r"\label{sec:dynamics-rescue}" in tex and r"\label{app:dynamics}" not in tex
    execution_runs = collect_execution_runs()
    execution_summary = summarize_execution(execution_runs)
    valid_runs = collect_valid_arrivals(execution_runs)
    valid_summary = summarize_valid_arrivals(valid_runs)
    first_arrival_audit = json.loads((DATA_ROOT / 'first_arrival_audit.json').read_text())
    assert first_arrival_audit['seed_status_counts'] == dict(Counter(row['status'] for row in valid_runs))
    assert first_arrival_audit['group_status_counts'] == dict(Counter(row['status'] for row in valid_summary))
    assert arrival_table == render_execution_table(valid_summary)
    arrival_means = {(row['environment'], row['method']): row['valid_first_arrival_steps_mean']
                     for row in valid_summary}
    for env in ENVIRONMENTS:
        prefix = env.paper_name.replace("_", r"\_") + " &"
        line = next(line for line in arrival_table.splitlines() if line.startswith(prefix))
        check_best_highlighting(line.split("&")[1:],
                                [arrival_means[env.paper_name, method] for method in METHODS],
                                maximize=False, context=f"First arrival: {env.paper_name}")
    for row in valid_summary:
        if row['status'] == 'reported':
            assert row['valid_success_seeds'] == 5
        else:
            assert row['valid_first_arrival_steps_mean'] is None
            assert row['valid_first_arrival_steps_std'] is None
    original_runs = {(row['environment'], row['method'], row['seed']): row for row in execution_runs}
    for row in valid_runs:
        original = original_runs[row['environment'], row['method'], row['seed']]
        mean = original['successful_episode_first_arrival_steps']
        eligible = original['successful_episodes'] > 0 and mean > 1
        assert row['successful_episodes'] == original['successful_episodes']
        assert row['successful_episode_first_arrival_steps'] == mean
        assert row['valid_seed'] == eligible
        assert row['valid_first_arrival_steps'] == (mean if eligible else None)
        assert row['test_sha256'] == original['test_sha256']
    assert set(row['status'] for row in valid_summary) <= {'reported', 'insufficient_valid_seeds'}
    assert 'NR' not in arrival_table
    reported = sum(row['status'] == 'reported' for row in valid_summary)
    print(f'OK: archived arrival records preserved: {reported} complete five-seed estimates; '
          f'{len(valid_summary) - reported} groups fail the seed-mean eligibility rule')
    for name, expected in (('execution_metrics_runs.csv', execution_runs),
                           ('execution_metrics_summary.csv', execution_summary),
                           ('first_arrival_runs.csv', valid_runs),
                           ('first_arrival_summary.csv', valid_summary)):
        with (DATA_ROOT / name).open() as handle:
            recorded = list(csv.DictReader(handle))
        assert recorded == [
            {key: '' if value is None else str(value) for key, value in row.items()}
            for row in expected
        ], name
    for name in ('nqrl_iclr2027.tex',):
        source = (PAPER_ROOT / name).read_text()
        assert not re.search(r'relabel|hindsight|重标记|200\.4', source, re.I)
        assert r'\label{app:baseline-validation}' not in source
        assert r'\label{app:dynamics}' not in source
        assert not re.search(r'\\(?:subsection|paragraph)\{[^}]*[?？]', source)
        assert source.count(r'\input{tables/dynamics_mse_rescue.tex}') == 0
        assert set(re.findall(r'\\input\{(tables/[^}]+)\}', source)) == expected_tables
        for relative in expected_tables:
            source = source.replace(rf'\input{{{relative}}}', (PAPER_ROOT / relative).read_text())
        source = expand_appendix(source)
        assert not re.search(r'\b[ML][ -](?:tier|capacity)|[ML] (?:档|容量档)', source)
        source = re.sub(r'(?m)(?<!\\)%.*$', '', source)
        language_labels = re.findall(r'\\label\{([^}]+)\}', source)
        assert set(language_labels) == set(labels), name
        assert len(language_labels) == len(set(language_labels)), name
        assert set(re.findall(r'\\(?:ref|eqref)\{([^}]+)\}', source)) <= set(language_labels), name
        assert citation_keys(source) == cited, name
        assert source.count('{') == source.count('}'), name
        stack = []
        for operation, environment in re.findall(r'\\(begin|end)\{([^}]+)\}', source):
            if operation == 'begin':
                stack.append(environment)
            else:
                assert stack and stack.pop() == environment, (name, environment)
        assert not stack, name

    appendix_source = tex.split(r"\appendix", 1)[1]
    appendix_labels = re.findall(r"\\label\{(app:[^}]+)\}", appendix_source)
    contents = (PAPER_ROOT / "appendix/contents.tex").read_text()
    assert re.findall(r"\\pageref\{([^}]+)\}", contents) == appendix_labels
    assert r"\label{alg:nqrl}" in tex
    network = json.loads((DATA_ROOT / "network_architecture_audit.json").read_text())
    assert network["trainable_parameters"] == 21668891
    assert (network["goal_latent_size"], network["non_goal_latent_size"]) == (57, 455)
    assert network["modules"]["actor.backbone"]["input_size"] == 569
    assert network["modules"]["critics.0.latent_dynamics.backbone"]["input_size"] == 520
    policy = network['configuration']['actor']
    assert policy['model']['input_mode'] == 'split_latent'
    search = policy['losses']['min_dist']
    assert (search['latent_goal_mode'], search['latent_goal_steps'],
            search['latent_goal_optim'], search['latent_goal_keep_best']) == ('min', 32, 'sgd', True)
    unique_components = [v["parameters"] for k, v in network["modules"].items()
                         if k != "critics.0.quasimetric_model"]
    assert sum(unique_components) == network["trainable_parameters"]
    scenes = json.loads((DATA_ROOT / "environment_visualization_audit.json").read_text())
    assert [s["environment"] for s in scenes] == [e.paper_name for e in ENVIRONMENTS]
    counts = {r["environment"]: r for r in read_csv(DATA_ROOT / "nqrl_parameter_counts.csv")}
    for scene in scenes:
        image = PAPER_ROOT / scene["image"]
        assert image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert hashlib.sha256(image.read_bytes()).hexdigest() == scene["image_sha256"]
        assert len(scene["observation"]) == int(counts[scene["environment"]]["state_dim"])
        assert scene["width"] == 768 and scene["height"] == 432
    scene_by_key = {scene["key"]: scene for scene in scenes}
    lift_scene = scene_by_key["fetchpick"]
    process = lift_scene["scripted_process"]
    assert "scripted" in lift_scene["scene"] and "not a learned-policy" in lift_scene["scene"]
    assert process["object_lift"] > .12
    assert len(process["actions"]) == process["stages"][-1]["end_step"] == 48
    assert abs(lift_scene["observation"][5] - process["initial_object_position"][2]
               - process["object_lift"]) < 1e-6
    assert all(-1 <= value <= 1 for action in process["actions"] for value in action)
    assert scene_by_key["pandapickj"]["backend"] == "PandaPickAndPlaceJoints-v3"
    assert scene_by_key["pandapickj"]["camera"] != scene_by_key["pandapick"]["camera"]
    assert "scripted" in tex and "Initial-state views" not in tex

    check_goal_distance_steps()
    check_goal_dimension_analysis()
    check_goal_fraction_success()
    check_goal_fraction_gain()
    check_dynamics_validation()
    with (DATA_ROOT / "baseline_validation.csv").open(encoding="utf-8") as handle:
        baseline_validation = list(csv.DictReader(handle))
    assert baseline_validation == [
        {key: str(value) for key, value in row.items()}
        for row in summarize_baseline_validation()
    ], "Baseline validation summaries differ from completed run records"
    preflight = json.loads(
        (PAPER_ROOT.parent / "runs/baseline_paper_validation/preflight.json").read_text()
    )
    assert len(preflight["tasks"]) == 12
    assert all(task["status"] == "PASS" for task in preflight["tasks"])

    expected_data = {
        "aggregate_final_metrics.csv",
        "aggregate_learning_metrics.csv",
        "aggregate_training_scalars.csv",
        "aggregate_training_scalars_available.csv",
        "audit_summary.json",
        "final_success.csv",
        "seed_test_success.csv",
        "seed_test_success_audit.json",
        "learning_curves.csv",
        "learning_runs.csv",
        "low_success_diagnostics.json",
        "macro_learning_curve.csv",
        "per_environment_learning_metrics.csv",
        "training_scalar_audit.json",
        "training_scalar_runs.csv",
        "paired_learning_runs.csv",
        "paired_learning_curves.csv",
        "paired_aggregate_learning_metrics.csv",
        "paired_per_environment_learning_metrics.csv",
        "paired_macro_learning_curve.csv",
        "dynamics_mse_runs.csv",
        "dynamics_mse_ant_trajectories.csv",
        "dynamics_mse_summary.csv",
        "dynamics_mse_provenance.json",
        "dynamics_mse_evidence.csv",
        "goal_distance_steps.csv",
        "baseline_validation.csv",
        "execution_metrics_runs.csv",
        "execution_metrics_summary.csv",
        "execution_metrics_audit.json",
        "first_arrival_runs.csv",
        "first_arrival_summary.csv",
        "first_arrival_audit.json",
        "per_task_auc.csv",
        "per_task_auc_runs.csv",
        "per_task_learning_curves.csv",
        "validation_checkpoints.csv",
        "per_task_learning_audit.json",
        "online_feedback_runs.csv",
        "online_feedback_summary.csv",
        "online_feedback_audit.json",
        "nqrl_parameter_counts.csv",
        "dynamics_trajectory_summary.csv",
        "dynamics_validation_runs.csv",
        "dynamics_validation_curves.csv",
        "dynamics_validation_audit.json",
        "goal_dimension_analysis.csv",
        "goal_dimension_analysis.json",
        "goal_fraction_success_runs.csv",
        "goal_fraction_success_tasks.csv",
        "goal_fraction_success_seed_means.csv",
        "goal_fraction_success_curves.csv",
        "goal_fraction_success_audit.json",
        "goal_fraction_gain_runs.csv",
        "goal_fraction_gain_tasks.csv",
        "goal_fraction_gain_seed_means.csv",
        "goal_fraction_gain_curves.csv",
        "goal_fraction_gain_audit.json",
    }
    missing_data = [name for name in expected_data if not (DATA_ROOT / name).exists()]
    assert not missing_data, f"Missing generated data: {missing_data}"

    print(
        f"OK: {len(ENVIRONMENTS)} tasks, {len(METHODS)} methods, "
        f"{len(cited)} matched citation entries, {len(figure_names)} figures, "
        f"3 dynamics panels from {len(dynamics_rows)} auxiliary runs, "
        "24 six-task gradient-update settings, 3 completed baseline sanity checks"
    )


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["--search-ablation-only"]:
        check_goal_distance_steps()
        print("OK: search data, seed coverage, configuration provenance, tables, prose claims, and unrounded macro mean")
    else:
        main()

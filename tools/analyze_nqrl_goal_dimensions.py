#!/usr/bin/env python3
"""Relate goal-coordinate fractions to QRL success and NQRL's task-level gains."""

import argparse
import csv
import hashlib
import json
import statistics
from decimal import Decimal
from fractions import Fraction

from scipy.stats import pearsonr, spearmanr
from nqrl_table_labels import task_label

from analyze_nqrl_paper_results import DATA_ROOT, PAPER_ROOT, ENVIRONMENTS, SEEDS, artifact_dir, load_json_lines


def derive():
    dimensions = list(csv.DictReader((DATA_ROOT / 'nqrl_parameter_counts.csv').open()))
    results = {(r['environment'], r['method']): Decimal(r['mean'])
               for r in csv.DictReader((DATA_ROOT / 'final_success.csv').open())}
    rows = []
    for row in dimensions:
        env = row['environment']
        state, goal = int(row['state_dim']), int(row['goal_dim'])
        qrl, nqrl = results[env, 'QRL'], results[env, 'NQRL']
        assert 0 < goal < state and int(row['verified_training_seeds']) == 5
        rows.append(dict(environment=env, state_coordinates=state, goal_coordinates=goal,
                         non_goal_coordinates=state-goal, non_goal_fraction=(state-goal)/state,
                         qrl_success=str(qrl), nqrl_success=str(nqrl),
                         success_gain_pp=str(100*(nqrl-qrl))))
    assert len(rows) == 14
    assert [r['environment'] for r in rows] == sorted((r['environment'] for r in rows), key=str.casefold)
    x = [r['non_goal_fraction'] for r in rows]
    y = [float(r['success_gain_pp']) for r in rows]
    pearson, spearman = pearsonr(x, y), spearmanr(x, y)
    qrl_success = [float(r['qrl_success']) for r in rows]
    qrl_pearson, qrl_spearman = pearsonr(x, qrl_success), spearmanr(x, qrl_success)
    audit = dict(
        tasks=len(rows), unit='one task, not one training seed',
        fraction='(state_coordinates - goal_coordinates) / state_coordinates',
        gain='100 * (NQRL mean test success - QRL mean test success), percentage points',
        means='Unchanged published Table 1 means; no task exclusions or outcome-dependent grouping',
        pearson_r=float(pearson.statistic), pearson_p_two_sided=float(pearson.pvalue),
        spearman_rho=float(spearman.statistic), spearman_p_two_sided=float(spearman.pvalue),
        qrl_success_correlation=dict(
            pearson_r=float(qrl_pearson.statistic), pearson_p_two_sided=float(qrl_pearson.pvalue),
            spearman_rho=float(qrl_spearman.statistic), spearman_p_two_sided=float(qrl_spearman.pvalue),
        ),
        p_value_method='SciPy two-sided parametric/asymptotic tests; Spearman uses average ranks for ties',
        interpretation='Exploratory association across heterogeneous tasks; not a controlled estimate of coordinate effects',
        source_sha256={name: hashlib.sha256((DATA_ROOT/name).read_bytes()).hexdigest()
                       for name in ('nqrl_parameter_counts.csv', 'final_success.csv')},
    )
    return rows, audit


def render_table(rows):
    lines = [
        r'\begin{table}[!htb]',
        r'\caption{State dimensions and Success rate (\%) at the main-result checkpoints.',
        r'$d_s,d_g$ are state and goal dimensions; $f_n=(d_s-d_g)/d_s$ is the goal-unspecified fraction.',
        r'$\Delta$ is AQRL minus QRL in percentage points. Bold marks the higher mean.',
        r'}',
        r'\label{tab:goal-dimensions}', r'\nqrltablestyle', r'\centering',
        r'\setlength{\tabcolsep}{5pt}', r'\renewcommand{\arraystretch}{1.0}',
        r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}ccccccc@{}}',
        r'\toprule',
        r'Task & $d_s$ & $d_g$ & $f_n$ (\%) & QRL & \method{} (Ours) & $\Delta$ (pp) \\',
        r'\midrule',
    ]
    for row in rows:
        means = [Decimal(row[key]) for key in ('qrl_success', 'nqrl_success')]
        cells = ['$' + (r'\mathbf{' + f'{100*v:.1f}' + '}' if v == max(means) else f'{100*v:.1f}') + '$'
                 for v in means]
        lines.append(task_label(row['environment']) + ' & ' +
                     f"{row['state_coordinates']} & {row['goal_coordinates']} & " +
                     f"{100*row['non_goal_fraction']:.1f} & " + ' & '.join(cells) +
                     f" & ${Decimal(row['success_gain_pp']):+.1f}$" + r' \\')
    lines += [r'\bottomrule', r'\end{tabular*}', r'\end{table}']
    return '\n'.join(lines) + '\n'


def summarize_success_curves(runs):
    """Average exact-ratio tasks within each seed, then aggregate five seeds."""
    task_rows, seed_rows, curve_rows = [], [], []
    for method in ('QRL', 'NQRL'):
        for env in ENVIRONMENTS:
            group = [r for r in runs if (r['method'], r['environment']) == (method, env.paper_name)]
            assert sorted(int(r['seed']) for r in group) == list(SEEDS)
            ratios = {Fraction(int(r['non_goal_coordinates']), int(r['state_coordinates'])) for r in group}
            assert len(ratios) == 1
            ratio = ratios.pop()
            values = [float(r['test_success']) for r in group]
            task_rows.append(dict(method=method, environment=env.paper_name,
                                  ratio_numerator=ratio.numerator, ratio_denominator=ratio.denominator,
                                  non_goal_fraction=float(ratio), success_mean=statistics.mean(values),
                                  success_std=statistics.stdev(values), seeds=len(values)))
        fractions = sorted({Fraction(r['ratio_numerator'], r['ratio_denominator']) for r in task_rows
                            if r['method'] == method})
        for ratio in fractions:
            tasks = [r['environment'] for r in task_rows if r['method'] == method
                     and Fraction(r['ratio_numerator'], r['ratio_denominator']) == ratio]
            means = []
            for seed in SEEDS:
                values = [float(r['test_success']) for r in runs
                          if r['method'] == method and r['environment'] in tasks and int(r['seed']) == seed]
                assert len(values) == len(tasks)
                value = statistics.mean(values)
                means.append(value)
                seed_rows.append(dict(method=method, ratio_numerator=ratio.numerator,
                                      ratio_denominator=ratio.denominator, non_goal_fraction=float(ratio),
                                      seed=seed, task_count=len(tasks), test_success=value))
            curve_rows.append(dict(method=method, ratio_numerator=ratio.numerator,
                                   ratio_denominator=ratio.denominator, non_goal_fraction=float(ratio),
                                   task_count=len(tasks), environments='|'.join(tasks), seeds=len(means),
                                   success_mean=statistics.mean(means), success_std=statistics.stdev(means)))
    return task_rows, seed_rows, curve_rows


def collect_success_curves():
    dimensions = {r['environment']: r for r in csv.DictReader((DATA_ROOT / 'nqrl_parameter_counts.csv').open())}
    published = {(r['environment'], r['method']): r for r in
                 csv.DictReader((DATA_ROOT / 'final_success.csv').open())}
    runs, sources = [], []
    for method in ('QRL', 'NQRL'):
        for env in ENVIRONMENTS:
            state, goal = (int(dimensions[env.paper_name][key]) for key in ('state_dim', 'goal_dim'))
            for seed in SEEDS:
                path = artifact_dir(env, method, seed) / 'test.log'
                records = load_json_lines(path)
                assert len(records) == 1 and records[0]['split'] == 'test', str(path)
                record = records[0]
                assert record['num_episodes'] == env.test_episodes
                assert record['succ_rate'] == record['success_count'] / record['num_episodes']
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                runs.append(dict(environment=env.paper_name, method=method, seed=seed,
                                 state_coordinates=state, goal_coordinates=goal,
                                 non_goal_coordinates=state-goal, test_success=record['succ_rate'],
                                 successful_episodes=record['success_count'], num_episodes=record['num_episodes'],
                                 selected_env_steps=record['env_steps'], test_sha256=digest))
                sources.append(dict(environment=env.paper_name, method=method, seed=seed,
                                    test_log=str(path), sha256=digest))
    task_rows, seed_rows, curve_rows = summarize_success_curves(runs)
    for row in task_rows:
        reference = published[row['environment'], row['method']]
        for derived, key in (('success_mean', 'mean'), ('success_std', 'sample_std')):
            assert abs(row[derived] - float(reference[key])) <= .000500001, (row, reference)
    for name, records in (('goal_fraction_success_runs.csv', runs),
                          ('goal_fraction_success_tasks.csv', task_rows),
                          ('goal_fraction_success_seed_means.csv', seed_rows),
                          ('goal_fraction_success_curves.csv', curve_rows)):
        with (DATA_ROOT / name).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    audit = dict(
        methods=['QRL', 'NQRL'], tasks=len(ENVIRONMENTS), training_seeds=list(SEEDS),
        selected_test_records=len(runs), unique_fractions=len(curve_rows)//2,
        grouping='Exact rational non-goal coordinate fractions; no bins or rounded-ratio grouping',
        mean='Equal-weight task average at each exact ratio within each seed, then equal-weight mean over five seeds',
        band='Mean +/- sample SD (ddof=1) of five seed-level task averages; excludes between-task spread',
        markers='Filled markers: ratio-group means. Hollow markers: individual task means at shared ratios.',
        lines='Straight connections between observed ratios; no fitting, smoothing or extrapolation',
        axes=dict(xlim=[.45, 1.0], ylim=[0, 1], panels=['QRL', 'NQRL']),
        correlation_scope='Existing correlations use all 14 individual tasks, not the 11 plotting groups',
        source_sha256={name: hashlib.sha256((DATA_ROOT/name).read_bytes()).hexdigest()
                       for name in ('nqrl_parameter_counts.csv', 'final_success.csv')},
        sources=sources,
    )
    (DATA_ROOT / 'goal_fraction_success_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(f"Collected {len(runs)} selected tests; {len(curve_rows)//2} exact ratios per method.")


def derive_gain_curves():
    """Compute paired-seed success differences from the existing 140 tests."""
    source_name = 'goal_fraction_success_runs.csv'
    source = list(csv.DictReader((DATA_ROOT/source_name).open()))
    by_key = {(r['environment'], r['method'], int(r['seed'])): r for r in source}
    assert len(source) == len(by_key) == 140
    runs, tasks, seed_means, curves = [], [], [], []
    for env in ENVIRONMENTS:
        task_runs = []
        for seed in SEEDS:
            qrl, nqrl = (by_key[env.paper_name, method, seed] for method in ('QRL', 'NQRL'))
            ratio = Fraction(int(qrl['non_goal_coordinates']), int(qrl['state_coordinates']))
            assert ratio == Fraction(int(nqrl['non_goal_coordinates']), int(nqrl['state_coordinates']))
            qrl_success, nqrl_success = float(qrl['test_success']), float(nqrl['test_success'])
            row = dict(environment=env.paper_name, seed=seed,
                       ratio_numerator=ratio.numerator, ratio_denominator=ratio.denominator,
                       non_goal_fraction=float(ratio), qrl_success=qrl_success,
                       nqrl_success=nqrl_success, gain_pp=100*(nqrl_success-qrl_success))
            runs.append(row)
            task_runs.append(row)
        values = [r['gain_pp'] for r in task_runs]
        tasks.append(dict(environment=env.paper_name, ratio_numerator=ratio.numerator,
                          ratio_denominator=ratio.denominator, non_goal_fraction=float(ratio),
                          gain_mean_pp=statistics.mean(values), gain_std_pp=statistics.stdev(values), seeds=5))
    fractions = sorted({Fraction(r['ratio_numerator'], r['ratio_denominator']) for r in tasks})
    for ratio in fractions:
        members = [r['environment'] for r in tasks
                   if Fraction(r['ratio_numerator'], r['ratio_denominator']) == ratio]
        values = []
        for seed in SEEDS:
            value = statistics.mean(r['gain_pp'] for r in runs
                                    if r['environment'] in members and r['seed'] == seed)
            values.append(value)
            seed_means.append(dict(ratio_numerator=ratio.numerator, ratio_denominator=ratio.denominator,
                                   non_goal_fraction=float(ratio), seed=seed, task_count=len(members),
                                   gain_pp=value))
        curves.append(dict(ratio_numerator=ratio.numerator, ratio_denominator=ratio.denominator,
                           non_goal_fraction=float(ratio), task_count=len(members),
                           environments='|'.join(members), seeds=5,
                           gain_mean_pp=statistics.mean(values), gain_std_pp=statistics.stdev(values)))
    for name, rows in (('goal_fraction_gain_runs.csv', runs), ('goal_fraction_gain_tasks.csv', tasks),
                       ('goal_fraction_gain_seed_means.csv', seed_means), ('goal_fraction_gain_curves.csv', curves)):
        with (DATA_ROOT/name).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    audit = dict(
        tasks=len(tasks), training_seeds=list(SEEDS), paired_seed_differences=len(runs),
        original_test_records=len(source), unique_fractions=len(fractions),
        gain='100 * (NQRL test success - QRL test success), percentage points; not relative percent change',
        pairing='Subtract the two selected-test success rates for each task and matching training-seed ID',
        aggregation='Average task gains at each exact fraction within each seed, then mean and sample SD over five seeds',
        band='Mean +/- sample SD (ddof=1) of five paired-seed gain averages; not a difference of SDs',
        markers='Hollow markers: individual task mean gains at shared fractions',
        axes=dict(xlim=[.45, 1.0], ylim=[-20, 100], unit='percentage points'),
        lines='Straight connections between observed fractions; no fit or smoothing',
        source_sha256={name: hashlib.sha256((DATA_ROOT/name).read_bytes()).hexdigest()
                       for name in (source_name, 'nqrl_parameter_counts.csv', 'final_success.csv')},
    )
    assert all(-20 <= row['gain_mean_pp']-row['gain_std_pp']
               <= row['gain_mean_pp']+row['gain_std_pp'] <= 100 for row in curves)
    assert all(-20 <= row['gain_mean_pp'] <= 100 for row in tasks)
    (DATA_ROOT/'goal_fraction_gain_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(f"Derived {len(runs)} paired-seed gains at {len(fractions)} exact fractions.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collect-success-curves', action='store_true',
                        help='Also extract QRL/NQRL selected tests for ratio curves with five-seed bands')
    parser.add_argument('--derive-gain-curves', action='store_true',
                        help='Derive paired-seed NQRL-minus-QRL gains from the archived selected tests')
    args = parser.parse_args()
    rows, audit = derive()
    with (DATA_ROOT / 'goal_dimension_analysis.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (DATA_ROOT / 'goal_dimension_analysis.json').write_text(json.dumps(audit, indent=2) + '\n')
    (PAPER_ROOT / 'tables/goal_dimension_analysis.tex').write_text(render_table(rows))
    print(f"NQRL gains, 14 tasks: Pearson r={audit['pearson_r']:.3f}; Spearman rho={audit['spearman_rho']:.3f}")
    qrl = audit['qrl_success_correlation']
    print(f"QRL success, 14 tasks: Pearson r={qrl['pearson_r']:.3f}; Spearman rho={qrl['spearman_rho']:.3f}")
    if args.collect_success_curves:
        collect_success_curves()
    if args.derive_gain_curves:
        derive_gain_curves()


if __name__ == '__main__':
    main()

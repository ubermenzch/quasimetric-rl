#!/usr/bin/env python3
"""Preserve historical execution metrics, which are no longer submitted in the paper."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter
from decimal import Decimal

from analyze_nqrl_paper_results import (
    DATA_ROOT, ENVIRONMENTS, METHODS, PAPER_ROOT, SEEDS, artifact_dir,
    load_json_lines,
)


def valid_arrival(test, horizon):
    """Apply the seed-mean threshold before any display rounding."""
    count = int(test['success_count'])
    mean = first_arrival_steps(test, horizon)
    if not count:
        status = 'no_successful_episodes'
    elif mean <= 1:
        status = 'mean_not_above_one'
    else:
        status = 'valid'
    return count, mean, status == 'valid', status


def collect_valid_arrivals(legacy_rows):
    environments = {env.paper_name: env for env in ENVIRONMENTS}
    rows = []
    for row in legacy_rows:
        env = environments[row['environment']]
        path = artifact_dir(env, row['method'], row['seed'])
        test = load_json_lines(path / 'test.log')[0]
        count, mean, valid, status = valid_arrival(test, row['horizon'])
        rows.append({
            'environment': row['environment'], 'method': row['method'], 'seed': row['seed'],
            'test_sha256': row['test_sha256'], 'successful_episodes': count,
            'successful_episode_first_arrival_steps': mean,
            'valid_seed': valid, 'status': status,
            'valid_first_arrival_steps': mean if valid else None,
        })
    return rows


def summarize_valid_arrivals(rows):
    summaries = []
    for env in ENVIRONMENTS:
        for method in METHODS:
            group = [row for row in rows
                     if (row['environment'], row['method']) == (env.paper_name, method)]
            assert sorted(row['seed'] for row in group) == list(SEEDS)
            means = [row['valid_first_arrival_steps'] for row in group
                     if row['valid_first_arrival_steps'] is not None]
            assert len(means) == sum(row['valid_seed'] for row in group)
            assert all(mean > 1 for mean in means)
            status = 'reported' if len(means) == len(SEEDS) else 'insufficient_valid_seeds'
            summaries.append({
                'environment': env.paper_name, 'method': method, 'status': status,
                'valid_success_seeds': len(means),
                'valid_first_arrival_steps_mean': statistics.mean(means) if status == 'reported' else None,
                'valid_first_arrival_steps_std': statistics.stdev(means) if status == 'reported' else None,
            })
    return summaries


def first_arrival_steps(test, horizon):
    """Remove the known H+1 contribution of unsuccessful episodes exactly."""
    count, episodes = int(test['success_count']), int(test['num_episodes'])
    total = Decimal(str(test['hitting_time'])) * episodes
    if not count:
        assert abs(total - episodes * (horizon + 1)) < Decimal('0.000001')
        return None
    successful_total = total - (episodes - count) * (horizon + 1)
    integral_total = round(successful_total)
    assert abs(successful_total - integral_total) < Decimal('0.000001')
    assert count <= integral_total <= count * horizon
    return integral_total / count


def success_state_fraction(test, horizon, environment):
    if 'success_state_step_fraction' in test:
        value = float(test['success_state_step_fraction'])
        assert math.isfinite(value) and 0 <= value <= test['succ_rate'] + 1e-12
        return value, 'recorded_success_flags'
    if test['success_count'] == 0:
        return 0.0, 'zero_success_implies_zero_success_steps'
    # These two wrappers return the binary success indicator as reward even
    # during evaluation. Other tasks use dense returns that cannot recover it.
    if environment in ('reacher', 'maze2d'):
        episodes = int(test['num_episodes'])
        total = Decimal(str(test['epi_return'])) * episodes
        assert abs(total - round(total)) < Decimal('0.000001')
        value = float(total / (episodes * horizon))
        assert 0 <= value <= test['succ_rate'] + 1e-12
        return value, 'binary_success_return_divided_by_horizon'
    return None, 'not_recorded_dense_return_not_invertible'


def collect_runs():
    rows = []
    for env in ENVIRONMENTS:
        for method in METHODS:
            for seed in SEEDS:
                path = artifact_dir(env, method, seed)
                tests = load_json_lines(path / 'test.log')
                assert len(tests) == 1 and (path / 'COMPLETE').is_file(), path
                test = tests[0]
                assert test['split'] == 'test' and test['num_episodes'] == env.test_episodes
                assert test['success_count'] / test['num_episodes'] == test['succ_rate']
                horizons = set(re.findall(r'episode_length[=:]\s*(\d+)',
                                          (path / 'output.log').read_text()))
                assert len(horizons) == 1, (path, horizons)
                horizon = int(horizons.pop())
                best = json.loads((path / 'best_checkpoint.json').read_text())
                validation = best['validation']
                assert best['test'] == test, path
                assert int(validation['env_steps']) == test['env_steps'], path
                assert int(validation['optim_steps']) == test['optim_steps'], path
                assert validation['seed_end'] < test['seed'], path
                arrival = first_arrival_steps(test, horizon)
                if 'successful_episode_first_arrival_steps' in test:
                    measured = test['successful_episode_first_arrival_steps']
                    assert (measured is None and arrival is None) or math.isclose(measured, arrival)
                fraction, source = success_state_fraction(test, horizon, env.key)
                rows.append({
                    'environment': env.paper_name, 'method': method, 'seed': seed,
                    'selected_env_steps': test['env_steps'],
                    'selected_optim_steps': test['optim_steps'],
                    'episode_seed_start': test['seed'], 'episode_seed_end': test['seed_end'],
                    'num_episodes': test['num_episodes'], 'horizon': horizon,
                    'test_success': test['succ_rate'], 'successful_episodes': test['success_count'],
                    'success_state_step_fraction': fraction,
                    'success_state_fraction_source': source,
                    'successful_episode_first_arrival_steps': arrival,
                    'legacy_censored_hitting_time': test['hitting_time'],
                    'test_sha256': hashlib.sha256((path / 'test.log').read_bytes()).hexdigest(),
                    'checkpoint_available_locally': any(path.glob('*agent*.pth')),
                })
    return rows


def summarize(rows):
    summaries = []
    for env in ENVIRONMENTS:
        for method in METHODS:
            group = [row for row in rows
                     if (row['environment'], row['method']) == (env.paper_name, method)]
            assert sorted(row['seed'] for row in group) == list(SEEDS)
            summary = {'environment': env.paper_name, 'method': method, 'training_seeds': len(group)}
            for metric in ('test_success', 'success_state_step_fraction',
                           'successful_episode_first_arrival_steps'):
                values = [row[metric] for row in group if row[metric] is not None]
                summary[metric + '_available_seeds'] = len(values)
                # Missing occupancy measurements must not silently select an
                # easier subset of seeds. Arrival is defined only on success.
                complete = metric != 'success_state_step_fraction' or len(values) == len(group)
                summary[metric + '_mean'] = statistics.mean(values) if values and complete else None
                summary[metric + '_std'] = statistics.stdev(values) if len(values) > 1 and complete else None
            summaries.append(summary)
    return summaries


def render_table(rows):
    lines = [
        r'% Generated by tools/analyze_nqrl_execution_metrics.py.',
        r'\begin{table*}[t]',
        r'\caption{Mean first-arrival steps in successful test episodes, using the',
        r'checkpoints and episodes in Table~\ref{tab:main}. Entries are means $\pm$',
        r'sample SD across five training seeds. A seed is valid only if it has',
        r'successful episodes and its unrounded mean exceeds 1 step.',
        r'All five seeds must be valid; \textemdash{} indicates otherwise.',
        r'All successful episodes within a valid seed contribute, including step-1 arrivals.',
        r'Bold marks the lowest reported mean for each task.',
        r'Compare arrival times alongside the success rates in Table~\ref{tab:main}.}',
        r'\label{tab:execution}',
        r'\centering\scriptsize',
        r'\setlength{\tabcolsep}{2pt}',
        r'\renewcommand{\arraystretch}{1.12}',
        r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrrr@{}}',
        r'\toprule',
        r'Task & QRL & TD-InfoNCE & GCSL & C-Learning & CRL & \method{} (Ours) \\',
        r'\midrule',
    ]
    by_setting = {(row['environment'], row['method']): row for row in rows}
    assert len(by_setting) == len(rows) == len(ENVIRONMENTS) * len(METHODS)
    metric = 'valid_first_arrival_steps'
    for env in ENVIRONMENTS:
        cells = []
        reported_means = [float(by_setting[env.paper_name, method][metric + '_mean'])
                          for method in METHODS
                          if by_setting[env.paper_name, method]['status'] == 'reported']
        best = min(reported_means) if reported_means else None
        for method in METHODS:
            row = by_setting[env.paper_name, method]
            mean, sd = row[metric + '_mean'], row[metric + '_std']
            if row['status'] == 'insufficient_valid_seeds':
                cells.append(r'\textemdash')
            else:
                assert row['status'] == 'reported' and int(row['valid_success_seeds']) == len(SEEDS)
                assert mean not in (None, '') and sd not in (None, '')
                spread = f'{float(sd):.1f}'
                value = f'{float(mean):.1f}\\!\\pm\\!{spread}'
                cells.append('$' + (r'\mathbf{' + value + '}' if float(mean) == best else value) + '$')
        task = env.paper_name.replace('_', r'\_')
        lines.append(task + ' & ' + ' & '.join(cells) + r' \\')
    lines.extend((r'\bottomrule', r'\end{tabular*}', r'\end{table*}'))
    return '\n'.join(lines) + '\n'


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    archive = PAPER_ROOT / 'archive'
    archive.mkdir(exist_ok=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tables-only', action='store_true',
                        help='Render the first-arrival table from archived summaries without rewriting data')
    args = parser.parse_args()
    if args.tables_only:
        with (DATA_ROOT / 'first_arrival_summary.csv').open() as handle:
            summaries = list(csv.DictReader(handle))
        (archive / 'execution_metrics.tex').write_text(render_table(summaries))
        print('Archived first-arrival table; excluded from the submitted manuscript.')
        return
    rows = collect_runs()
    summaries = summarize(rows)
    write_csv(DATA_ROOT / 'execution_metrics_runs.csv', rows)
    write_csv(DATA_ROOT / 'execution_metrics_summary.csv', summaries)
    valid_rows = collect_valid_arrivals(rows)
    valid_summary = summarize_valid_arrivals(valid_rows)
    write_csv(DATA_ROOT / 'first_arrival_runs.csv', valid_rows)
    write_csv(DATA_ROOT / 'first_arrival_summary.csv', valid_summary)
    (archive / 'execution_metrics.tex').write_text(render_table(valid_summary))
    valid_audit = {
        'rule': 'Require successes and an unrounded successful-episode mean above 1 in every training seed; aggregate only complete five-seed groups.',
        'interpretation': 'Filter seed results; all successful episodes within a valid seed, including step-1 arrivals, contribute to its mean.',
        'scope': 'First-arrival statistics only; Table 1 retains its original test-success definition.',
        'recovery': '(num_episodes * hitting_time - (num_episodes - success_count) * (horizon + 1)) / success_count',
        'aggregation': 'Equal weight per training seed; mean and sample SD (ddof=1); round only for table display.',
        'seed_status_counts': dict(Counter(row['status'] for row in valid_rows)),
        'group_status_counts': dict(Counter(row['status'] for row in valid_summary)),
        'legacy_data': 'Legacy test summaries and unfiltered arrival calculations are retained separately.',
    }
    (DATA_ROOT / 'first_arrival_audit.json').write_text(json.dumps(valid_audit, indent=2) + '\n')
    print(json.dumps(valid_audit, indent=2))
    audit = {
        'tests': len(rows),
        'success_state_fraction_sources': dict(Counter(row['success_state_fraction_source'] for row in rows)),
        'groups_with_complete_success_state_fraction': sum(
            row['success_state_step_fraction_mean'] is not None for row in summaries),
        'first_arrival_rule': '(N * censored_hitting_time - (N - successes) * (H + 1)) / successes',
        'undefined_first_arrival_seeds': sum(row['successful_episodes'] == 0 for row in rows),
        'aggregation': 'Seed-level metrics first; sample SD; occupancy requires all five seeds; arrival excludes zero-success seeds with n reported',
        'binary_reward_evidence': [
            'quasimetric_rl/data/online/maze2d/__init__.py: OnlineMazeGoalEnv.step returns float(is_success)',
            'dm_control/suite/reacher.py: get_reward uses tolerance with default zero margin; DMC wrapper success is reward >= 1 - 1e-7',
        ],
    }
    (DATA_ROOT / 'execution_metrics_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()

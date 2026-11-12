#!/usr/bin/env python3
"""Audit online feedback diagnostics without interpreting them as causal proof."""

import csv
import hashlib
import json
import statistics

from analyze_nqrl_paper_results import DATA_ROOT, ENVIRONMENTS, SEEDS, artifact_dir, validation_curve
from audit_nqrl_training_scalars import load_scalars


def main():
    rows, tags, sources = [], set(), []
    for env in sorted(ENVIRONMENTS, key=lambda env: env.paper_name.casefold()):
        for seed in SEEDS:
            path = artifact_dir(env, 'NQRL', seed)
            scalars, errors, nonfinite, _, _ = load_scalars(path)
            assert not errors and not nonfinite, path
            tags.update(scalars)
            counts = {e.step: e.value for e in scalars['data/num_regular_transitions']}
            successes = {e.step: e.value for e in scalars['data/num_successes']}
            curve, _ = validation_curve(path / 'eval.log')
            row = {'environment': env.paper_name, 'seed': seed}
            for label, lo, hi in [('early', 0, .2), ('late', .8, 1)]:
                steps = sorted(s for s in counts.keys() & successes.keys()
                               if lo * env.budget < s <= hi * env.budget and counts[s] > 0)
                assert steps, (env, seed, label)
                row[label + '_replay_success_fraction'] = statistics.mean(
                    successes[s] / counts[s] for s in steps)
                validations = [v['succ_rate'] for v in curve
                               if lo * env.budget < v['env_steps'] <= hi * env.budget]
                assert validations, (env, seed, label)
                row[label + '_validation_success'] = statistics.mean(validations)
                row[label + '_scalar_records'] = len(steps)
                row[label + '_validation_records'] = len(validations)
            rows.append(row)
            for event in sorted(path.glob('events.out.tfevents.*')):
                sources.append({'environment': env.paper_name, 'seed': seed,
                                'file': event.name,
                                'sha256': hashlib.sha256(event.read_bytes()).hexdigest()})
        print('Audited', env.paper_name, flush=True)
    def write(name, records):
        with (DATA_ROOT / name).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=records[0])
            writer.writeheader()
            writer.writerows(records)
    write('online_feedback_runs.csv', rows)
    summary = []
    metrics = ('early_replay_success_fraction', 'late_replay_success_fraction',
               'early_validation_success', 'late_validation_success')
    for env in sorted(ENVIRONMENTS, key=lambda env: env.paper_name.casefold()):
        group = [r for r in rows if r['environment'] == env.paper_name]
        summary.append({'environment': env.paper_name, **{
            f'{metric}_{stat}': fn(r[metric] for r in group)
            for metric in metrics for stat, fn in [('mean', statistics.mean), ('std', statistics.stdev)]}})
    write('online_feedback_summary.csv', summary)
    (DATA_ROOT / 'online_feedback_audit.json').write_text(json.dumps({
        'runs': len(rows), 'windows': {'early': [0, .2], 'late': [.8, 1]},
        'boundary_rule': 'exclusive lower, inclusive upper; mean within seed before cross-seed summary',
        'available_tags': sorted(tags), 'sources': sources,
        'interpretation': 'Replay and validation trends are indirect evidence only. No region-matched true goal-distance error or controlled frozen-data comparison is recorded.',
    }, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Compare searched and random completions to goal-matched replay probes."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np


ROOT = Path('/backup/zhangcheng/qrl-assets/results/queue')
TASKS = (
    ('PointMaze-M', 'pointmaze_medium_v3', (0, 1), 0.45, True),
    ('FetchPnP', 'fetchpickandplace', (3, 4, 5), 0.05, False),
    ('FetchPush', 'fetchpush', (3, 4, 5), 0.05, False),
)


def selected_probes(directory: Path, cutoff: int, count: int):
    by_update = {}
    with (directory / 'diagnostics/probes.jsonl').open() as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:  # A running job may be writing a line.
                continue
            if (record['env_steps'] <= cutoff
                    and record['arrays'].startswith('probes/completion_')):
                by_update[record['update']] = record
    records = sorted(by_update.values(), key=lambda x: x['update'])[-count:]
    if len(records) != count:
        raise ValueError(f'{directory.name}: only {len(records)} eligible probes')
    return records


def measure(directory: Path, goal_dims: tuple, radius: float,
            cutoff: int, count: int, seed: int, inclusive: bool = False):
    sums = np.zeros(4, dtype=np.float64)
    matched = total = 0
    records = selected_probes(directory, cutoff, count)
    for record in records:
        filename = directory / 'diagnostics' / record['arrays']
        with np.load(filename, allow_pickle=False) as data:
            goals = data['raw_goals']
            bank = data['reference_states']
            bank_latent = data['reference_latents']
            excluded = data['self_match_mask']
            initial = data['initial_latent']
            completed = data['completed_latent']
            free_size = data['search_h'].shape[-1]
        goal_size = initial.shape[-1] - free_size
        if not np.allclose(initial[:, :goal_size], completed[:, :goal_size]):
            raise ValueError(f'{filename}: search changed the goal latent')
        goal_distance = np.linalg.norm(
            goals[:, None, goal_dims] - bank[None, :, goal_dims], axis=-1)
        within_goal = goal_distance <= radius if inclusive else goal_distance < radius
        candidates = within_goal & ~excluded
        rng = np.random.default_rng(seed * 10_000_000 + int(record['update']))
        for query in range(len(goals)):
            total += 1
            valid = np.flatnonzero(candidates[query])
            if not len(valid):
                continue
            random_pool = np.flatnonzero(~excluded[query])
            if not len(random_pool):
                continue
            random_latent = initial[query].copy()
            random_latent[goal_size:] = bank_latent[rng.choice(random_pool), goal_size:]
            reference = bank_latent[valid]
            sums += [
                np.linalg.norm(reference - initial[query], axis=-1).min(),
                np.linalg.norm(reference - completed[query], axis=-1).min(),
                np.linalg.norm(reference - random_latent, axis=-1).min(),
                np.linalg.norm(completed[query] - random_latent),
            ]
            matched += 1
    if not matched:
        raise ValueError(f'{directory.name}: no goal-matched replay states')
    return sums / matched, matched, total, (records[0]['env_steps'], records[-1]['env_steps'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cutoff', type=int, default=160_000)
    parser.add_argument('--probes', type=int, default=20)
    args = parser.parse_args()
    for label, slug, dims, radius, inclusive in TASKS:
        results = []
        for seed in (1000, 1001, 1002):
            directories = [p for p in ROOT.glob(f'*AQRL-Displacement*{slug}*s{seed}')
                           if '__attempt_' not in p.name]
            if len(directories) != 1:
                raise ValueError(f'{label} seed {seed}: {directories}')
            results.append(measure(directories[0], dims, radius,
                                   args.cutoff, args.probes, seed, inclusive))
        means = np.stack([r[0] for r in results])
        coverage = sum(r[1] for r in results) / sum(r[2] for r in results)
        print(f'{label} radius={radius} matched={sum(r[1] for r in results)}/'
              f'{sum(r[2] for r in results)} ({coverage:.2%}) '
              f'steps={results[0][3]}')
        for index, metric in enumerate(('initial', 'searched', 'random', 'searched_to_random')):
            values = means[:, index].tolist()
            print(f'  {metric}: {statistics.mean(values):.2f} ± '
                  f'{statistics.stdev(values):.2f}; seeds='
                  f'{[round(value, 3) for value in values]}')


if __name__ == '__main__':
    main()

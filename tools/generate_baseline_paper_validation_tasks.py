#!/usr/bin/env python3
"""Prepare one-seed original-task checks; never launch the scheduler."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quasimetric_rl.data.online.paper_validation import SPECS
from tools.generate_online_baseline_tasks import task_line
from tools.run_qrl_queue import Task, read_tasks


@dataclasses.dataclass(frozen=True)
class Experiment:
    algorithm: str
    name: str
    steps: int
    save_steps: int
    kind: str = 'paper_validation'
    overrides: tuple = ()


# Budgets are validation budgets, not a claim of bitwise paper reproduction.
EXPERIMENTS = (
    Experiment('td_infonce', 'FetchPush', 1_000_000, 20_000, 'gcrl'),
    Experiment('td_infonce', 'FetchPickAndPlace', 1_000_000, 20_000, 'gcrl'),
    Experiment('td_infonce', 'FetchSlide', 1_000_000, 20_000, 'gcrl'),
    Experiment('crl', 'FetchReach', 200_000, 10_000, 'gcrl'),
    Experiment('crl', 'FetchPush', 3_000_000, 50_000, 'gcrl'),
    Experiment('crl', 'CRLAntUMaze', 10_000_000, 100_100),
    Experiment('gcsl', 'GCSLFourRooms', 200_000, 2_000),
    Experiment('gcsl', 'GCSLSawyerPush', 1_000_000, 10_000),
    Experiment('gcsl', 'GCSLSawyerDoor', 1_000_000, 10_000),
    Experiment('c_learning', 'CLearningSawyerReach', 3_000_000, 50_000),
    Experiment('c_learning', 'CLearningSawyerPush', 3_000_000, 49_981),
    Experiment('c_learning', 'CLearningSawyerWindow', 3_000_000, 49_981),
)
DISPLAY_NAMES = {'td_infonce': 'TD-InfoNCE', 'crl': 'CRL', 'gcsl': 'GCSL', 'c_learning': 'C-Learning'}
PREFIX = 'paper_validation_v1_'
DEFAULT_MANIFEST = ROOT / 'configs/baseline_paper_validation_4alg_3task_1seed.tsv'
DEFAULT_REPORT = ROOT / 'runs/baseline_paper_validation/preflight.json'
QUICK_MANIFEST = ROOT / 'configs/crl_quick_validation_1task_100k.tsv'
QUICK_REPORT = ROOT / 'runs/baseline_paper_validation/crl_quick_preflight.json'
QUICK_EXPERIMENTS = (Experiment('crl', 'CRLPointSmall', 100_000, 5_000),)
MAIN_ADDITIONS = (
    Experiment('td_infonce', 'FetchReach', 500_000, 10_000, 'gcrl'),
    Experiment('c_learning', 'CLearningSawyerDrawer', 3_000_000, 49_981),
    Experiment('crl', 'CRLSawyerPush', 3_000_000, 49_950),
    Experiment('crl', 'CRLSawyerBin', 3_000_000, 49_950),
    Experiment('gcsl', 'GCSLLunarLander', 200_000, 2_000),
    Experiment('gcsl', 'GCSLClaw', 1_000_000, 10_000,
               overrides=('agent.baselines.gcbc.action_discretization=factorized',)),
    Experiment('c_learning', 'CLearningJacoReach', 3_000_000, 50_000,
               overrides=('agent.baselines.c_learning.discount=0.9',
                          'agent.baselines.c_learning.odds_clip=2.0')),
    Experiment('c_learning', 'CLearningFinger', 3_000_000, 50_000,
               overrides=('agent.baselines.c_learning.discount=0.99',
                          'agent.baselines.c_learning.odds_clip=50.0')),
    Experiment('c_learning', 'CLearningHand', 3_000_000, 50_000,
               overrides=('agent.baselines.c_learning.discount=0.3',
                          'agent.baselines.c_learning.odds_clip=2.0')),
)
MAIN_MANIFEST = ROOT / 'configs/baseline_paper_validation_main_available_1seed.tsv'
MAIN_REPORT = ROOT / 'runs/baseline_paper_validation/main_preflight.json'


def make_task(experiment: Experiment) -> Task:
    horizon = SPECS[experiment.name][0] if experiment.kind == 'paper_validation' else 50
    args = [
        f'env.kind={experiment.kind}', f'agent.algorithm={experiment.algorithm}',
        'batch_size=256', 'interaction.exploration_eps=0',
        f'interaction.env_steps_per_cycle={horizon}',
        'interaction.num_eval_episodes=50', 'interaction.num_test_episodes=200',
        'interaction.validation_seed=1000000', 'interaction.test_seed=2000000',
        'eval_steps=null', f'save_steps={experiment.save_steps}',
        'keep_only_latest_checkpoint=true',
        'save_replay_buffer=true', 'save_final_replay_buffer=true',
    ]
    if experiment.kind == 'paper_validation':
        goal_dims = ','.join(map(str, SPECS[experiment.name][3]))
        args.append(f'interaction.baseline_goal_dims=[{goal_dims}]')
    args.extend(experiment.overrides)
    return Task(
        task_id=f'{PREFIX}{DISPLAY_NAMES[experiment.algorithm]}_{experiment.name}_s1000',
        mode='online', env_name=experiment.name, seed='1000',
        steps=str(experiment.steps), params='source-default', extra_args=' '.join(args),
    )


def generate_tasks(*, suite='paper'):
    if suite == 'paper':
        return [make_task(experiment) for experiment in EXPERIMENTS]
    if suite == 'paper-main':
        return [make_task(experiment) for experiment in (*EXPERIMENTS, *MAIN_ADDITIONS)]
    if suite == 'crl-quick':
        return [dataclasses.replace(
            make_task(experiment),
            task_id=f'crl_quick_v1_CRL_{experiment.name}_s1000',
        ) for experiment in QUICK_EXPERIMENTS]
    raise ValueError(f'Unknown validation suite: {suite}')


def render_tasks(tasks, *, suite='paper'):
    if suite == 'crl-quick':
        return '\n'.join([
            '# task_id\tmode\tenv_name\tseed\tsteps\tparams\textra_args',
            '# CRL quick diagnostic: upstream point_Small, 100k steps, unchanged CRL defaults.',
            '# Success uses distance < 0.25; upstream reward still uses distance < 2.0.',
            '# See docs/BASELINE_PAPER_VALIDATION.md. Enqueue only after preflight passes.',
            *(task_line(task) for task in tasks),
        ]) + '\n'
    return '\n'.join([
        '# task_id\tmode\tenv_name\tseed\tsteps\tparams\textra_args',
        '# Original-paper TASK validation with unchanged local baseline implementations.',
        '# One training seed (1000). No model-size presets; see docs/BASELINE_PAPER_VALIDATION.md.',
        '# This is a candidate manifest, not a certificate that dependencies are available.',
        *(task_line(task) for task in tasks),
    ]) + '\n'


def append_verified(path: Path, tasks, report):
    verified = {row['task_id'] for row in report['tasks'] if row['status'] == 'PASS'}
    existing = {task.task_id: task for task in read_tasks(path)}
    additions = []
    for task in tasks:
        if task.task_id not in verified:
            continue
        if task.task_id in existing:
            if existing[task.task_id] != task:
                raise ValueError(f'Existing task has different settings: {task.task_id}')
        else:
            additions.append(task)
    if not additions:
        return 0
    current = path.read_text() if path.exists() else ''
    block = '\n# Upstream task checks; preflight passed, seed=1000.\n'
    block += '\n'.join(task_line(task) for task in additions) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(current + ('\n' if current and not current.endswith('\n') else '') + block)
    if path.exists():
        temporary.chmod(path.stat().st_mode)
    os.replace(temporary, path)
    return len(additions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=('paper', 'paper-main', 'crl-quick'), default='paper')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--append-to', type=Path)
    parser.add_argument('--preflight-report', type=Path)
    args = parser.parse_args()
    if args.output is None:
        args.output = {'paper': DEFAULT_MANIFEST, 'paper-main': MAIN_MANIFEST,
                       'crl-quick': QUICK_MANIFEST}[args.suite]
    if args.preflight_report is None:
        args.preflight_report = {'paper': DEFAULT_REPORT, 'paper-main': MAIN_REPORT,
                                 'crl-quick': QUICK_REPORT}[args.suite]
    tasks = generate_tasks(suite=args.suite)
    manifest = render_tasks(tasks, suite=args.suite)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(manifest)
    print(f'Wrote {len(tasks)} candidate tasks to {args.output}')
    if args.append_to is not None:
        report = json.loads(args.preflight_report.read_text())
        if report['manifest'] != manifest:
            raise ValueError('Preflight configuration does not match the current manifest')
        import hashlib
        for relative_path, expected_hash in report['source_hashes'].items():
            current_hash = hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
            if current_hash != expected_hash:
                raise ValueError(f'Source changed after preflight: {relative_path}; rerun preflight')
        count = append_verified(args.append_to, tasks, report)
        blocked = sum(row['status'] != 'PASS' for row in report['tasks'])
        print(f'Appended {count} verified tasks; {blocked} tasks remain blocked. Scheduler not started.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

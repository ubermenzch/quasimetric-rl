#!/usr/bin/env python3
"""Queue QRL's five published online settings, reusing paired FetchPush runs."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.generate_qrl_original_config_tasks import (
    QUEUE, TOTAL_STEPS, generate_tasks as paired_tasks, parameter_count,
    validate_original_repo,
)
from tools.generate_selected_success_l_min32_tasks import atomic_write, task_line
from tools.run_qrl_queue import read_tasks, task_fingerprint

OUTPUT = ROOT / 'configs/iclr_rebuttal_qrl_paper_online_5settings_1seed.tsv'
PAPER = 'https://proceedings.mlr.press/v202/wang23al/wang23al.pdf'
SETTINGS = ('FetchReach', 'FetchReachImage', 'FetchPush', 'FetchPushImage', 'FetchSlide')
IMAGE_OVERRIDES = {
    'agent.quasimetric_critic.model.encoder.arch': '[512]',
    'agent.quasimetric_critic.losses.latent_dynamics.weight': '10',
    'interaction.num_samples_per_cycle': '125',
}


def generate_tasks():
    template = paired_tasks()[0]  # Exactly reuse the AQRL-repository FetchPush seed 1000.
    tasks = []
    for name in SETTINGS:
        image_based = name.endswith('Image')
        values = dict(token.split('=', 1) for token in shlex.split(template.extra_args))
        if image_based:
            values.update(IMAGE_OVERRIDES)
        state_dim = 10 if name == 'FetchReach' else 25
        tasks.append(replace(
            template,
            task_id=template.task_id.replace('fetchpush_online_', f'{name.lower()}_online_'),
            env_name=name,
            params=str(parameter_count(state_dim=state_dim, image_based=image_based)),
            extra_args=' '.join(f'{key}={value}' for key, value in values.items()),
        ))
    assert len(tasks) == len({task.task_id for task in tasks}) == 5
    return tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--append', action='store_true')
    args = parser.parse_args()
    validate_original_repo()
    requested = generate_tasks()
    before = QUEUE.read_text()
    existing = read_tasks(QUEUE)
    from tools.set_pending_logs_only import preserve_storage_policy
    requested = preserve_storage_policy(requested, existing)
    by_id = {task.task_id: task for task in existing}
    if len(by_id) != len(existing):
        raise ValueError('The task list contains duplicate IDs')
    additions = []
    reused = []
    for task in requested:
        previous = by_id.get(task.task_id)
        if previous:
            if task_fingerprint(previous) != task_fingerprint(task):
                raise ValueError(f'Existing task definition differs: {task.task_id}')
            reused.append(task.task_id)
        else:
            additions.append(task)
    report = dict(
        requested=len(requested), added=len(additions), reused=len(reused),
        reused_task_ids=reused, seed=1000, interaction_budget=TOTAL_STEPS,
        environments=list(SETTINGS), code_repository=str(ROOT),
        source_paper=PAPER, source_sections=['Figure 5', 'Appendix C.3'],
        image_overrides=IMAGE_OVERRIDES,
        validation_episodes=50, validation_interval=2000,
        purpose='Single-seed implementation checks; no outcomes are asserted before completion.',
        source_comparison='Reuse the two-repository five-seed FetchPush experiment.',
        original_task_count=len(existing),
        queue_sha256_before=hashlib.sha256(before.encode()).hexdigest(),
    )
    if args.append:
        if QUEUE.read_text() != before:
            raise RuntimeError('Task list changed; rerun before appending')
        if additions:
            backup = QUEUE.with_name('tasks.tsv.before_qrl_paper_validation_20261107')
            if not backup.exists():
                backup.write_text(before)
            atomic_write(
                QUEUE, before.rstrip('\n') + '\n\n'
                '# Appendix C.5: QRL original online settings, seed 1000; FetchPush reused.\n'
                + '\n'.join(task_line(task) for task in additions) + '\n',
            )
        after = read_tasks(QUEUE)
        assert after[:len(existing)] == existing
        assert len(after) == len(existing) + len(additions)
        assert len({task.task_id for task in after}) == len(after)
        requested_ids = {task.task_id for task in requested}
        report['task_indices'] = {
            task.task_id: index for index, task in enumerate(after, 1)
            if task.task_id in requested_ids
        }
        report['queue_sha256_after'] = hashlib.sha256(QUEUE.read_bytes()).hexdigest()
        atomic_write(OUTPUT, '# task_id\tmode\tenv_name\tseed\tsteps\tparams\textra_args\n'
                     + '\n'.join(task_line(task) for task in requested) + '\n')
        atomic_write(OUTPUT.with_suffix('.audit.json'), json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

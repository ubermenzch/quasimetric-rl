#!/usr/bin/env python3
"""Queue paired QRL-original-configuration runs in AQRL and original QRL code."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quasimetric_rl.model_size import (
    _configured_mlp_parameter_count, load_model_size_preset, qrl_agent_parameter_count,
)
from tools.generate_selected_success_l_min32_tasks import atomic_write, task_line
from tools.run_qrl_queue import (
    TASK_CODE_REPO_ARG, TASK_CODE_REVISION_ARG, TASK_NATIVE_QRL_ARG,
    Task, read_tasks, task_fingerprint,
)

ORIGINAL_REPO = ROOT / 'reference_repos/qrl-original'
ORIGINAL_REVISION = '0a611a41815c83103d92cdbff1283f99266a7360'
SUBMODULE_REVISION = '43687c5a1d7567fe84874c9993bf0f94318b25ff'
QUEUE = ROOT / 'runs/qrl_queue/tasks.tsv'
SEEDS = tuple(range(1000, 1005))
TOTAL_STEPS = 1_000_000

COMMON_ARGS = (
    'env.kind=gcrl',
    'batch_size=256',
    'agent.num_critics=2',
    'agent.actor.losses.min_dist.adaptive_entropy_regularizer=true',
    'agent.actor.losses.min_dist.add_goal_as_future_state=true',
    'agent.actor.losses.behavior_cloning.weight=0',
    'agent.actor.losses.actor_optim.lr=3e-5',
    'agent.actor.losses.entropy_weight_optim.lr=3e-4',
    'agent.quasimetric_critic.losses.critic_optim.lr=1e-4',
    'agent.quasimetric_critic.losses.lagrange_mult_optim.lr=0.01',
    'agent.quasimetric_critic.losses.latent_dynamics.weight=0.1',
    'agent.quasimetric_critic.losses.global_push.softplus_beta=0.1',
    'agent.quasimetric_critic.losses.global_push.softplus_offset=15',
    'interaction.num_prefill_episodes=200',
    'interaction.num_samples_per_cycle=500',
    'interaction.num_rollouts_per_cycle=10',
    'interaction.num_eval_episodes=50',
    'interaction.exploration_eps=0.3',
    'log_steps=250',
)

AQRL_REPO_ARGS = (
    '+qrl_model_size=s',
    'agent.algorithm=qrl',
    'interaction.original_qrl_evaluation=true',
    'agent.training_schedule=joint',
    'agent.quasimetric_critic.losses.separate_latent_dynamics=false',
    'agent.quasimetric_critic.losses.latent_dynamics.distance=iqe',
    'agent.actor.losses.min_dist.latent_goal_mode=none',
    'interaction.validation_seed=416923159',
    'interaction.test_seed=416923209',
    'interaction.num_test_episodes=1000',
    # AQRL evaluates at checkpoint boundaries. Match QRL's 2k evaluation
    # interval while retaining only best/latest/final snapshots locally.
    'eval_steps=null',
    'save_steps=2000',
    'keep_only_latest_checkpoint=true',
    'keep_only_best_and_final_checkpoints=true',
    'save_replay_buffer=true',
    'save_final_replay_buffer=true',
    'resume_if_possible=true',
    f'{TASK_CODE_REPO_ARG}=.',
)

ORIGINAL_REPO_ARGS = (
    'eval_steps=2000',
    'save_steps=50000',
    f'{TASK_CODE_REPO_ARG}=reference_repos/qrl-original',
    f'{TASK_CODE_REVISION_ARG}={ORIGINAL_REVISION}',
    f'{TASK_NATIVE_QRL_ARG}=true',
)


def parameter_count(state_dim: int = 25, action_dim: int = 4, *,
                    image_based: bool = False) -> int:
    if image_based:
        state_dim = 1024  # Output dimension of the original Atari convolutional torso.
    preset = load_model_size_preset('qrl', 's')
    actor = preset.actor.model
    actor_count = _configured_mlp_parameter_count(
        2 * state_dim, actor.arch, 2 * action_dim,
        kind=str(actor.get('mlp_kind', 'plain')),
    )
    single_count = qrl_agent_parameter_count(state_dim, action_dim, 's')
    count = 2 * (single_count - actor_count) + actor_count
    if image_based:
        encoder = preset.quasimetric_critic.model.encoder
        full_encoder = _configured_mlp_parameter_count(
            state_dim, encoder.arch, encoder.latent_size, kind='plain',
        )
        image_encoder = _configured_mlp_parameter_count(
            state_dim, (512,), encoder.latent_size, kind='plain',
        )
        # Each critic and the actor have independent torsos. The actor shares
        # its torso between its current-state and goal-image inputs.
        torso = ((3 * 8 * 8 + 1) * 32 + (32 * 4 * 4 + 1) * 64
                 + (64 * 3 * 3 + 1) * 64)
        count += 2 * (image_encoder - full_encoder) + 3 * torso
    return count


def generate_tasks(total_steps: int = TOTAL_STEPS) -> list[Task]:
    tasks = []
    for label, extra in (
        ('AQRLRepo-QRLOriginalConfig', AQRL_REPO_ARGS),
        ('OriginalRepo-QRLOriginalConfig', ORIGINAL_REPO_ARGS),
    ):
        for seed in SEEDS:
            tasks.append(Task(
                task_id=(f'iclr_rebuttal_v1_{label}_{total_steps}steps_'
                         f'fetchpush_online_s{seed}'),
                mode='online', env_name='FetchPush', seed=str(seed),
                steps=str(total_steps), params=str(parameter_count()),
                extra_args=' '.join((*COMMON_ARGS, *extra)),
            ))
    assert len(tasks) == len({task.task_id for task in tasks}) == 10
    return tasks


def validate_original_repo() -> None:
    for repo, revision in (
        (ORIGINAL_REPO, ORIGINAL_REVISION),
        (ORIGINAL_REPO / 'third_party/torch-quasimetric', SUBMODULE_REVISION),
    ):
        actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
        dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo, text=True).strip()
        if actual != revision or dirty:
            raise ValueError(f'Original source must be clean at {revision}: {repo}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--append', action='store_true')
    parser.add_argument('--steps', type=int, default=TOTAL_STEPS)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error('--steps must be positive')
    validate_original_repo()
    requested = generate_tasks(args.steps)
    before = QUEUE.read_text()
    existing = read_tasks(QUEUE)
    from tools.set_pending_logs_only import preserve_storage_policy
    requested = preserve_storage_policy(requested, existing)
    by_id = {task.task_id: task for task in existing}
    additions = []
    for task in requested:
        previous = by_id.get(task.task_id)
        if previous and task_fingerprint(previous) != task_fingerprint(task):
            raise ValueError(f'Existing task definition differs: {task.task_id}')
        if not previous:
            additions.append(task)
    report = dict(requested=len(requested), added=len(additions),
                  reused=len(requested) - len(additions), environment='FetchPush',
                  seeds=list(SEEDS), interaction_budget=args.steps,
                  original_repo=str(ORIGINAL_REPO), original_revision=ORIGINAL_REVISION,
                  submodule_revision=SUBMODULE_REVISION,
                  native_source_unchanged=True,
                  primary_comparison='50-episode validation success every 2000 environment steps',
                  original_validation_seed=416923159,
                  checkpoint_note='AQRL: every 2k with pruning; original source: every 50k, unmodified saving code',
                  source_goal_sampling='Equal random/future mixture; original implementation defaults',
                  existing_tasks=len(existing),
                  queue_sha256_before=hashlib.sha256(before.encode()).hexdigest())
    if args.append:
        if QUEUE.read_text() != before:
            raise RuntimeError('Task list changed; rerun before appending')
        if additions:
            backup = QUEUE.with_name('tasks.tsv.before_qrl_original_config_20261107')
            if not backup.exists():
                backup.write_text(before)
            atomic_write(QUEUE, before.rstrip('\n') + '\n\n'
                         '# FetchPush: original QRL configuration in AQRL and original QRL code; '
                         'five training seeds per repository.\n'
                         + '\n'.join(task_line(task) for task in additions) + '\n')
        after = read_tasks(QUEUE)
        assert after[:len(existing)] == existing
        assert len(after) == len(existing) + len(additions)
        assert len({task.task_id for task in after}) == len(after)
        report['task_indices'] = {task.task_id: i for i, task in enumerate(after, 1)
                                  if task.task_id in {row.task_id for row in requested}}
        report['queue_sha256_after'] = hashlib.sha256(QUEUE.read_bytes()).hexdigest()
        out = ROOT / f'configs/iclr_rebuttal_qrl_original_config_fetchpush_{args.steps}steps_2repo_5seed.tsv'
        atomic_write(out, '# task_id\tmode\tenv_name\tseed\tsteps\tparams\textra_args\n'
                     + '\n'.join(task_line(task) for task in requested) + '\n')
        atomic_write(out.with_suffix('.audit.json'), json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

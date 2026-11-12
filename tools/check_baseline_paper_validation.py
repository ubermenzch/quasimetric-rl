#!/usr/bin/env python3
"""Smoke-test the queued configuration on CPU without starting full training."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import shlex
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import hydra
import gym
import attrs
import numpy as np
import torch

from online.main import Conf, summarize_evaluation
from online.trainer import Trainer
from quasimetric_rl import utils
from tools.generate_baseline_paper_validation_tasks import (
    DEFAULT_REPORT, MAIN_REPORT, QUICK_REPORT, generate_tasks, render_tasks,
)


def source_hashes():
    paths = (
        'quasimetric_rl/modules/gcrl_baselines.py',
        'quasimetric_rl/data/online/memory.py',
        'quasimetric_rl/data/online/paper_validation/__init__.py',
        'quasimetric_rl/data/online/paper_validation/reference.py',
        'quasimetric_rl/data/online/paper_validation/c_learning.py',
        'configs/baseline_paper_validation_protocols.json',
        'online/trainer.py', 'online/main.py',
        'tools/generate_baseline_paper_validation_tasks.py',
        'tools/check_baseline_paper_validation.py',
    )
    return {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths}


def task_config(task):
    overrides = shlex.split(task.extra_args) + [
        f'env.name={task.env_name}', f'seed={task.seed}',
        f'interaction.total_env_steps={task.steps}',
    ]
    with hydra.initialize(version_base=None, config_path=None):
        cfg = Conf.from_DictConfig(hydra.compose(config_name='config', overrides=overrides))
        # Conf is a singleton, so even deepcopy(cfg) returns the same object.
        # Detach its attrs fields for tools that hold several task configs.
        detached = object.__new__(Conf)
        for field in attrs.fields(Conf):
            setattr(detached, field.name, copy.deepcopy(getattr(cfg, field.name)))
        return detached


def check_task(task, updates):
    cfg = task_config(task)
    torch_seed, np_seed = utils.split_seed(cfg.seed, 2)
    np.random.seed(np.random.Generator(np.random.PCG64(np_seed)).integers(1 << 31))
    torch.manual_seed(np.random.Generator(np.random.PCG64(torch_seed)).integers(1 << 31))
    # Only memory reservation and device differ from the queued configuration.
    cfg.env.init_num_transitions = 2800
    cfg.env.increment_num_transitions = 1400
    replay = cfg.env.make()
    collect_env = None
    try:
        trainer = Trainer(
            agent_conf=cfg.agent, device=torch.device('cpu'), replay=replay,
            batch_size=cfg.batch_size, interaction_conf=cfg.interaction,
        )
        collect_env = trainer.make_collect_env()
        collect_env.seed(cfg.seed)
        if cfg.env.kind == 'paper_validation':
            from quasimetric_rl.data.online.paper_validation import SPECS
            expected = SPECS[cfg.env.name]
            assert tuple(replay.env_spec.observation_shape) == (expected[1],)
            if isinstance(replay.env_spec.action_space, gym.spaces.Discrete):
                assert replay.env_spec.action_space.n == expected[2]
                assert tuple(replay.env_spec.action_space.shape) == ()
            else:
                assert tuple(replay.env_spec.action_space.shape) == (expected[2],)
            assert trainer.agent.goal_dims == expected[3]
        for _ in range(2):
            rollout = trainer.collect_random_rollout(store=False, env=collect_env)
            assert torch.isfinite(rollout.all_observations).all()
            assert torch.isfinite(rollout.rewards).all()
            # Ensure a training partition exists even in this two-episode smoke test.
            replay.add_rollout(rollout, training=True)
        actor_before = [p.detach().clone() for p in trainer.agent.actor.parameters()]
        losses = []
        for _ in range(updates):
            result = trainer.losses(trainer.agent, trainer.sample(), optimize=True)
            assert torch.isfinite(result.loss), result.info
            losses.append(float(result.loss))
            assert all(torch.isfinite(p).all() for p in trainer.agent.parameters())
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in trainer.agent.parameters())
        assert any(not torch.equal(a, b) for a, b in zip(actor_before, trainer.agent.actor.parameters()))
        batch = trainer.sample()
        with torch.no_grad():
            actions = trainer.agent.act(batch.observations, batch.future_observations).mean
        assert torch.isfinite(actions).all()
        if isinstance(replay.env_spec.action_space, gym.spaces.Discrete):
            assert actions.dtype == torch.int64
            assert all(replay.env_spec.action_space.contains(a) for a in actions.cpu().numpy())
        else:
            low = torch.as_tensor(replay.env_spec.action_space.low)
            high = torch.as_tensor(replay.env_spec.action_space.high)
            assert ((actions >= low) & (actions <= high)).all()
        stream = io.BytesIO()
        torch.save(trainer.agent.state_dict(), stream)
        stream.seek(0)
        restored = copy.deepcopy(trainer.agent)
        restored.load_state_dict(torch.load(stream, weights_only=False))
        with torch.no_grad():
            torch.testing.assert_close(
                restored.act(batch.observations, batch.future_observations).mean, actions,
                rtol=0, atol=0,
            )
        evaluated = trainer.evaluate(num_episodes=2, seed=1_000_000)
        summary = summarize_evaluation(
            evaluated, split='smoke', seed=1_000_000,
            env_steps=replay.num_transitions_realized, optim_steps=updates,
            episode_length=replay.episode_length,
        )
        if cfg.env.kind == 'paper_validation':
            assert np.isfinite(summary['final_goal_distance'])
        return dict(
            task_id=task.task_id, status='PASS', algorithm=cfg.agent.algorithm,
            env=cfg.env.name, horizon=replay.episode_length,
            goal_dims=list(trainer.agent.goal_dims), updates=updates, losses=losses,
            trainable_parameters=sum(p.numel() for p in trainer.agent.parameters() if p.requires_grad),
            algorithm_config=attrs.asdict(getattr(
                cfg.agent.baselines, 'gcbc' if cfg.agent.algorithm == 'gcsl' else cfg.agent.algorithm)),
            smoke_evaluation=summary,
            protocol_metadata=(collect_env.protocol_metadata()
                               if cfg.env.kind == 'paper_validation' else None),
        )
    finally:
        if collect_env is not None:
            collect_env.close()
        replay.env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=('paper', 'paper-main', 'crl-quick'), default='paper')
    parser.add_argument('--report', type=Path)
    parser.add_argument('--updates', type=int, default=3)
    args = parser.parse_args()
    if args.updates < 1:
        parser.error('--updates must be positive')
    if args.report is None:
        args.report = {'paper': DEFAULT_REPORT, 'paper-main': MAIN_REPORT,
                       'crl-quick': QUICK_REPORT}[args.suite]
    torch.set_num_threads(1)
    tasks = generate_tasks(suite=args.suite)
    report = dict(manifest=render_tasks(tasks, suite=args.suite), source_hashes=source_hashes(), tasks=[])
    for task in tasks:
        print(f'Checking {task.task_id}', flush=True)
        try:
            row = check_task(task, args.updates)
        except Exception as exc:
            row = dict(task_id=task.task_id, status='BLOCKED', reason=f'{type(exc).__name__}: {exc}')
            traceback.print_exc()
        report['tasks'].append(row)
        print(f"{row['status']}: {task.task_id}", flush=True)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n')
    passed = sum(row['status'] == 'PASS' for row in report['tasks'])
    print(f'{passed}/{len(tasks)} passed. Report: {args.report}. No full jobs started.')
    return 0 if passed == len(tasks) else 1


if __name__ == '__main__':
    raise SystemExit(main())

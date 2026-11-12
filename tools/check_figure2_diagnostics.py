#!/usr/bin/env python3
"""Exercise the actual Figure 2 L model and diagnostics without a full run."""

import argparse
import csv
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hydra
from omegaconf import OmegaConf
import torch

from online.main import Conf, resolve_split_encoder_goal_dims
from online.trainer import Trainer
from online.dynamics_diagnostics import DynamicsDiagnostics, collect_fixed_probe, tensor_digest
from quasimetric_rl import utils


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/figure2_diagnostics_preflight')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'configs/figure2_iqe_only_l_antnavigate_5seed.tsv').open() as handle:
        rows = list(csv.reader((line for line in handle if not line.startswith('#')), delimiter='\t'))
    assert len(rows) == 5 and {int(row[3]) for row in rows} == set(range(1000, 1005))
    queue = (ROOT / 'runs/qrl_queue/tasks.tsv').read_text().splitlines()
    for row in rows:
        assert '\t'.join(row) in queue
    row = rows[0]
    with hydra.initialize(version_base=None, config_path=None):
        cfg = Conf.from_DictConfig(hydra.compose(config_name='config', overrides=shlex.split(row[6]) + [
            f'env.name={row[2]}', f'seed={row[3]}', f'interaction.total_env_steps={row[4]}',
        ]))
    resolve_split_encoder_goal_dims(cfg)
    assert cfg.dynamics_diagnostics.enabled
    torch.set_num_threads(2)
    torch.manual_seed(cfg.seed)
    # Only CPU placement and replay allocation differ from the queued run.
    cfg.device.type, cfg.device.index = 'cpu', None
    cfg.env.init_num_transitions = cfg.env.increment_num_transitions = 2000
    replay = cfg.env.make()
    try:
        trainer = Trainer(agent_conf=cfg.agent, device=torch.device('cpu'), replay=replay,
                          batch_size=cfg.batch_size, interaction_conf=cfg.interaction)
        count = sum(p.numel() for p in trainer.agent.parameters())
        assert count == 21_668_891, count
        rng = utils.rng_state_dict()
        diagnostic = DynamicsDiagnostics(cfg.dynamics_diagnostics, trainer, args.output,
                                         start_env_steps=0, start_optim_steps=0, experiment_conf=cfg)
        torch.testing.assert_close(torch.get_rng_state(), rng['torch'], rtol=0, atol=0)
        assert replay.num_transitions_realized == 0
        repeated = collect_fixed_probe(replay, cfg.dynamics_diagnostics)
        assert tensor_digest(repeated) == diagnostic.probe_hash
        print('L model and shared held-out probe: PASS', flush=True)
        env = trainer.make_collect_env()
        try:
            env.seed(8_300_000)
            trainer.collect_random_rollout(env=env)
        finally:
            env.close()
        info = trainer.train_step(trainer.sample())
        diagnostic.observe_training_step(info)
        diagnostic.finish_training_cycle(1000, 1)
        diagnostic.record(1000, 1, checkpoint=True)
        summary = json.loads((diagnostic.root / 'env00001000_opt00000001.json').read_text())
        for panel in ('fixed', 'replay'):
            stats = summary['panels'][panel]['critic_00']
            assert not stats['nonfinite_tensors'], stats
            assert stats['projector_activations']
            assert stats['gradients']['dynamics_parameters']['mse_effective_l2'] == 0
        assert all(torch.isfinite(p).all() for p in trainer.agent.parameters())
        report = {'status': 'PASS', 'parameters': count, 'probe_sha256': diagnostic.probe_hash,
                  'probe_transitions': len(diagnostic.probe['actions']), 'optimizer_updates': 1,
                  'training_settings': OmegaConf.to_container(OmegaConf.structured(cfg.agent), resolve=True)}
        (args.output / 'preflight.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({key: value for key, value in report.items() if key != 'training_settings'}), flush=True)
    finally:
        replay.env.close()


if __name__ == '__main__':
    main()

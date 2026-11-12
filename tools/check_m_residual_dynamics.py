#!/usr/bin/env python3
"""Preflight both M-residual dynamics arms and their recorded evidence on CPU."""

import argparse
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hydra
from omegaconf import OmegaConf
import torch

from online.dynamics_diagnostics import DynamicsDiagnostics, tensor_digest
from online.main import Conf, resolve_split_encoder_goal_dims, summarize_evaluation
from online.trainer import Trainer
from quasimetric_rl import utils
from quasimetric_rl.model_size import cqrl_agent_parameter_count, load_model_size_preset
from tools.run_qrl_queue import read_tasks, task_preserves_checkpoints_after_completion, task_training_args


MANIFEST = ROOT / 'configs/figure2_m_residual_antnavigate_6seed.tsv'


def configurations(manifest=MANIFEST):
    tasks = read_tasks(manifest)
    assert len(tasks) == len({t.task_id for t in tasks}) == 6
    result = []
    with hydra.initialize(version_base=None, config_path=None):
        for task in tasks:
            assert task_preserves_checkpoints_after_completion(task)
            cfg = Conf.from_DictConfig(hydra.compose(config_name='config', overrides=[
                *task_training_args(task), f'env.name={task.env_name}',
                f'seed={task.seed}', f'interaction.total_env_steps={task.steps}',
                f'output_folder={task.task_id}',
            ]))
            resolve_split_encoder_goal_dims(cfg)
            # online.Conf is a singleton. Snapshot its values before composing
            # the next task; instantiate only the arm currently under test.
            result.append((task, OmegaConf.structured(cfg)))
    assert {cfg.seed for _, cfg in result} == {1000, 1001, 1002}
    for seed in (1000, 1001, 1002):
        pair = [cfg for _, cfg in result if cfg.seed == seed]
        assert len(pair) == 2
        configs = [OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True) for cfg in pair]
        weights = []
        for cfg in configs:
            weights.append(cfg['agent']['quasimetric_critic']['losses']['latent_dynamics'].pop('mse_weight'))
            cfg.pop('output_folder')
        assert weights == [0., 1.] and configs[0] == configs[1]
    assert cqrl_agent_parameter_count(29, 8, 2, 'm') == 4_396_305
    assert load_model_size_preset('cqrl', 'm').quasimetric_critic.model.encoder.get('mlp_kind', 'plain') == 'plain'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/m_residual_dynamics_preflight')
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    configs = configurations(manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    reports = []
    initial_hash = None
    for task, queued_conf in configs[:2]:
        cfg = Conf.from_DictConfig(queued_conf)
        directory = args.output / ('iqe_only' if cfg.agent.quasimetric_critic.losses.latent_dynamics.mse_weight == 0 else 'iqe_mse')
        directory.mkdir(exist_ok=True)
        queued_config = OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True, enum_to_str=True)
        # CPU and small replay allocation only; retain the actual network,
        # diagnostic panels and optimizer settings of the queued experiment.
        cfg.device.type, cfg.device.index = 'cpu', None
        cfg.env.init_num_transitions = cfg.env.increment_num_transitions = 2000
        torch.manual_seed(cfg.seed)
        replay = cfg.env.make()
        try:
            trainer = Trainer(agent_conf=cfg.agent, device=torch.device('cpu'), replay=replay,
                              batch_size=cfg.batch_size, interaction_conf=cfg.interaction)
            parameters = sum(p.numel() for p in trainer.agent.parameters() if p.requires_grad)
            assert parameters == 4_391_126
            model_hash = tensor_digest(trainer.agent.state_dict())
            initial_hash = initial_hash or model_hash
            assert model_hash == initial_hash
            rng = utils.rng_state_dict()
            diagnostic = DynamicsDiagnostics(cfg.dynamics_diagnostics, trainer, directory,
                                             start_env_steps=0, start_optim_steps=0, experiment_conf=cfg)
            torch.testing.assert_close(torch.get_rng_state(), rng['torch'], rtol=0, atol=0)
            assert replay.num_transitions_realized == 0
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
            path = diagnostic.root / 'env00001000_opt00000001.pth'
            saved = torch.load(path, weights_only=False)
            assert 'agent' in saved and set(saved['panels']) == {'fixed', 'replay'}
            for panel in ('fixed', 'replay'):
                stats = saved['summary']['panels'][panel]['critic_00']
                assert not stats['nonfinite_tensors']
                assert len(stats['projector_activations']) == 6
                assert set(stats['branches']) == {'full', 'goal', 'non_goal'}
                effective_mse = stats['gradients']['dynamics_parameters']['mse_effective_l2']
                assert (effective_mse > 0) == (cfg.agent.quasimetric_critic.losses.latent_dynamics.mse_weight > 0)
            batches = torch.load(diagnostic.root / 'training_env00001000_opt00000001.pth', weights_only=False)
            assert batches['critics']['critic_00'].shape == (1, 4)
            with zipfile.ZipFile(diagnostic.root / f'source_{diagnostic.provenance}.zip') as archive:
                assert archive.read(str(manifest.relative_to(ROOT))) == manifest.read_bytes()
                assert archive.read('configs/model_size/cqrl/m_residual.yaml') == (ROOT / 'configs/model_size/cqrl/m_residual.yaml').read_bytes()
            evaluated = trainer.evaluate(num_episodes=2, seed=8_400_000)
            summary = summarize_evaluation(evaluated, split='validation', seed=8_400_000,
                                           env_steps=1000, optim_steps=1, episode_length=replay.episode_length)
            diagnostic.save_evaluation(evaluated, summary)
            evaluation = torch.load(diagnostic.root / 'validation_env00001000_opt00000001.pth', weights_only=False)
            assert evaluation['result']['is_success'].shape == (2,)
            report = dict(task_id=task.task_id, parameters=parameters, initial_model_sha256=model_hash,
                          probe_sha256=diagnostic.probe_hash, probe_transitions=len(diagnostic.probe['actions']),
                          optimizer_updates=1, evaluation_episodes=2, queued_config=queued_config)
            reports.append(report)
            print(task.task_id, 'PASS', flush=True)
        finally:
            replay.env.close()
    assert len({r['probe_sha256'] for r in reports}) == 1
    assert all(r['probe_transitions'] == 1024 for r in reports)
    (args.output / 'preflight.json').write_text(json.dumps({
        'status': 'PASS', 'validated_tasks': len(configs), 'arms': reports,
    }, indent=2) + '\n')
    print('PASS: six configurations, both actual models/updates/evaluations, shared fixed probe and saved evidence.')


if __name__ == '__main__':
    main()

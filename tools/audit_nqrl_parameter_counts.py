#!/usr/bin/env python3
"""Count NQRL parameters from archived configurations, excluding frozen targets."""

import csv
import hashlib

import gym
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf, SCMode

from analyze_nqrl_paper_results import DATA_ROOT, ENVIRONMENTS, SEEDS, artifact_dir
from generate_extended_online_tasks import ENVIRONMENTS as EXTENDED
from generate_iclr_main_table_tasks import ENVIRONMENTS as MAIN
from quasimetric_rl.data.env_spec import EnvSpec
from quasimetric_rl.modules import QRLConf


def main():
    torch.set_num_threads(1)
    specs = {e.name.replace('_', ' ').casefold(): e for e in (*EXTENDED, *MAIN)}
    rows = []
    for env in ENVIRONMENTS:
        spec = specs[env.paper_name.replace('_', ' ').casefold()]
        configs = [yaml.safe_load((artifact_dir(env, 'NQRL', seed) / 'config.yaml').read_text())
                   for seed in SEEDS]
        reference = configs[0]['agent']
        for config in configs[1:]:
            for key in ('quasimetric_critic', 'actor', 'num_critics'):
                assert reference[key] == config['agent'][key], (env, key)
        parameters = dict(reference)
        parameters.pop('baselines', None)
        parameters['model_size'] = None  # Historical label has no architectural effect.
        config = OmegaConf.to_container(
            OmegaConf.merge(OmegaConf.structured(QRLConf()), parameters),
            structured_config_mode=SCMode.INSTANTIATE)
        env_spec = EnvSpec(
            observation_space=gym.spaces.Box(-np.inf, np.inf, shape=(spec.state_dim,), dtype=np.float32),
            observation_space_is_dict=True,
            action_space=gym.spaces.Box(-1, 1, shape=(spec.action_dim,), dtype=np.float32))
        agent, losses = config.make(env_spec=env_spec, total_optim_steps=1)
        path = artifact_dir(env, 'NQRL', 1000)
        runtime = (path / 'output.log').read_text()
        assert f'observation_shape=torch.Size([{spec.state_dim}])' in runtime
        latent = config.quasimetric_critic.model.encoder.goal_latent_size
        assert f'goal_latent_size={latent},' in runtime
        rows.append({
            'environment': env.paper_name, 'state_dim': spec.state_dim,
            'action_dim': spec.action_dim, 'goal_dim': spec.goal_dim,
            'trainable_parameters': sum(p.numel() for p in agent.parameters() if p.requires_grad),
            'config_sha256': hashlib.sha256((path / 'config.yaml').read_bytes()).hexdigest(),
            'verified_training_seeds': len(configs),
        })
        del agent, losses
    with (DATA_ROOT / 'nqrl_parameter_counts.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print('Counted archived models for all 14 tasks; checked all five seed configurations.')
    print('Parameter range:', min(r['trainable_parameters'] for r in rows),
          max(r['trainable_parameters'] for r in rows))


if __name__ == '__main__':
    main()

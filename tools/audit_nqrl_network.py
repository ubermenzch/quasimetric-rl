#!/usr/bin/env python3
"""Resolve and inspect the archived AntNavigate network used in the paper."""

import json
from pathlib import Path
import sys

import gym
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf, SCMode

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_nqrl_paper_results import DATA_ROOT, ENVIRONMENTS, artifact_dir
from quasimetric_rl.data.env_spec import EnvSpec
from quasimetric_rl.modules import QRLConf


def inspect_network():
    torch.set_num_threads(1)
    task = next(env for env in ENVIRONMENTS if env.key == "antnavigate")
    config_path = artifact_dir(task, "NQRL", 1000) / "config.yaml"
    parameters = yaml.safe_load(config_path.read_text())["agent"]
    parameters.pop("baselines", None)
    parameters["model_size"] = None
    config = OmegaConf.to_container(
        OmegaConf.merge(OmegaConf.structured(QRLConf()), parameters),
        structured_config_mode=SCMode.INSTANTIATE,
    )
    spec = EnvSpec(
        observation_space=gym.spaces.Box(-np.inf, np.inf, shape=(29,), dtype=np.float32),
        observation_space_is_dict=True,
        action_space=gym.spaces.Box(-1, 1, shape=(8,), dtype=np.float32),
    )
    agent, losses = config.make(env_spec=spec, total_optim_steps=1)
    enc = config.quasimetric_critic.model.encoder
    modules = {}
    for name, module in agent.named_modules():
        if type(module).__name__ in ("ResidualMLP", "QuasimetricModel", "IQE"):
            modules[name] = {
                "type": type(module).__name__,
                "parameters": sum(p.numel() for p in module.parameters() if p.requires_grad),
                "input_size": getattr(module, "input_size", None),
                "output_size": getattr(module, "output_size", None),
                "width": getattr(module, "width", None),
                "residual_blocks": len(module.blocks) if hasattr(module, "blocks") else None,
                "block_depth": getattr(module, "residual_block_size", None),
            }
    result = {
        "environment": task.paper_name,
        "trainable_parameters": sum(p.numel() for p in agent.parameters() if p.requires_grad),
        "goal_dims": list(enc.goal_dims),
        "goal_latent_size": enc.goal_latent_size,
        "non_goal_latent_size": enc.non_goal_latent_size,
        "goal_arch": list(enc.goal_arch),
        "non_goal_arch": list(enc.non_goal_arch),
        "modules": modules,
        "module_tree": str(agent),
        "configuration": parameters,
    }
    assert result["trainable_parameters"] == 21668891
    return result


if __name__ == "__main__":
    result = inspect_network()
    path = DATA_ROOT / "network_architecture_audit.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("module_tree", "configuration")}, indent=2))

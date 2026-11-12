"""Checkpoint-aligned dynamics measurements, isolated from online optimization."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import zipfile

import attrs
import numpy as np
import torch
from omegaconf import OmegaConf

from quasimetric_rl import utils


@attrs.define(kw_only=True)
class DynamicsDiagnosticsConf:
    enabled: bool = False
    interval: int = attrs.field(default=5000, validator=attrs.validators.gt(0))
    probe_seed: int = 7_300_000
    probe_episodes: int = attrs.field(default=8, validator=attrs.validators.gt(0))
    samples_per_episode: int = attrs.field(default=32, validator=attrs.validators.gt(1))
    save_interval_models: bool = False
    save_training_batches: bool = False
    source_manifests: tuple[str, ...] = ('configs/figure2_iqe_only_l_antnavigate_5seed.tsv',)


@contextlib.contextmanager
def isolated_measurement(module=None):
    rng = utils.rng_state_dict()
    modes = [] if module is None else [(m, m.training) for m in module.modules()]
    try:
        if module is not None:
            module.eval()
        yield
    finally:
        for child, training in modes:
            child.training = training
        utils.load_rng_state(rng)


def tensor_digest(tensors):
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode('ascii'))
        digest.update(str((value.dtype, tuple(value.shape))).encode('ascii'))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temporary, path)


def finite_json(value):
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def distribution(value):
    value = value.detach().double()
    finite = torch.isfinite(value)
    valid = value[finite]
    result = {'nonfinite_fraction': (~finite).double().mean().item()}
    for name, quantile in [('p50', 0.5), ('p95', 0.95), ('p99', 0.99)]:
        result[name] = torch.quantile(valid, quantile).item() if valid.numel() else None
    result['mean'] = valid.mean().item() if valid.numel() else None
    result['max'] = valid.max().item() if valid.numel() else None
    return result


def representation(value):
    value = value.detach().double()
    return {
        'nonfinite_fraction': (~torch.isfinite(value)).double().mean().item(),
        'rms': value.square().mean().sqrt().item(),
        'norm': distribution(value.norm(dim=-1)),
        'std_mean': value.std(dim=0, unbiased=False).mean().item(),
        'variance_mean': value.var(dim=0, unbiased=False).mean().item(),
        'constant_coordinate_fraction': (
            value.std(dim=0, unbiased=False) < 1e-8).double().mean().item(),
    }


def gradient_stats(values):
    count = sum(value.numel() for value in values)
    return {
        'l2': math.sqrt(sum(value.detach().double().square().sum().item() for value in values)),
        'zero_fraction': sum((value == 0).sum().item() for value in values) / count,
        'nonfinite_fraction': sum((~torch.isfinite(value)).sum().item() for value in values) / count,
    }


def gradient_cosine(left, right):
    dot = sum((a.detach().double() * b.detach().double()).sum().item()
              for a, b in zip(left, right))
    denom = gradient_stats(left)['l2'] * gradient_stats(right)['l2']
    return dot / denom if denom > 0 else None


def measure_critic(critic, transitions, loss_conf):
    """Use raw, unweighted losses; autograd.grad never writes parameter .grad."""
    device = next(critic.parameters()).device
    data = {key: value.to(device) for key, value in transitions.items()}
    activation_stats = {}
    handles = []
    projection_side = 'target'

    def activation_hook(name):
        def hook(module, inputs, output):
            x = inputs[0].detach().double()
            if isinstance(module, torch.nn.SiLU):
                sigmoid = x.sigmoid()
                derivative = sigmoid * (1 + x * (1 - sigmoid))
            elif isinstance(module, torch.nn.LeakyReLU):
                derivative = torch.where(x > 0, 1., module.negative_slope)
            else:
                derivative = (x > 0).double()
            activation_stats[f'{projection_side}/{name}'] = {
                'type': type(module).__name__,
                'nonpositive_input_fraction': (x <= 0).double().mean().item(),
                'zero_output_fraction': (output.detach() == 0).double().mean().item(),
                'small_derivative_fraction': (derivative.abs() < 1e-6).double().mean().item(),
                'nonfinite_fraction': (~torch.isfinite(output.detach())).double().mean().item(),
            }
        return hook

    with isolated_measurement(critic), torch.enable_grad(), torch.autocast(device_type=device.type, enabled=False):
        try:
            for name, module in critic.quasimetric_model.projector.named_modules():
                if isinstance(module, (torch.nn.ReLU, torch.nn.LeakyReLU, torch.nn.SiLU)):
                    handles.append(module.register_forward_hook(activation_hook(name)))
            with torch.no_grad():
                current = critic.encoder(data['observations'])
                target = critic.encoder(data['next_observations'])
                projected_target = critic.quasimetric_model.project(target)
            prediction = critic.predict_next_latent(current, data['actions'])
            projection_side = 'prediction'
            projected_prediction = critic.quasimetric_model.project(prediction)
            distances = critic.quasimetric_model.forward_projected(
                projected_prediction, projected_target, bidirectional=True)
            mse = (prediction - target).square().mean()
            iqe = distances.square().mean()
            parameters = tuple(p for p in critic.latent_dynamics.parameters() if p.requires_grad)
            targets = (prediction,) + parameters
            iqe_grad = torch.autograd.grad(iqe, targets, retain_graph=True, allow_unused=True)
            mse_grad = torch.autograd.grad(mse, targets, allow_unused=True)
            iqe_grad = tuple(torch.zeros_like(p) if g is None else g for p, g in zip(targets, iqe_grad))
            mse_grad = tuple(torch.zeros_like(p) if g is None else g for p, g in zip(targets, mse_grad))
        finally:
            for handle in handles:
                handle.remove()

    tensors = {
        **{key: value.detach().cpu() for key, value in transitions.items()},
        'current_latent': current.detach().cpu(),
        'target_latent': target.detach().cpu(),
        'predicted_latent': prediction.detach().cpu(),
        'projected_target': projected_target.detach().cpu(),
        'projected_prediction': projected_prediction.detach().cpu(),
        'bidirectional_distances': distances.detach().cpu(),
        'iqe_gradient_predicted_latent': iqe_grad[0].detach().cpu(),
        'mse_gradient_predicted_latent': mse_grad[0].detach().cpu(),
    }
    stats = {
        'mse': mse.item(), 'sq_dists': iqe.item(),
        'dist_p2n': distances[:, 0].mean().item(),
        'dist_n2p': distances[:, 1].mean().item(),
        'projected_mse': (projected_prediction - projected_target).square().mean().item(),
        'projector_activations': activation_stats,
        'gradients': {}, 'branches': {},
    }
    for name, value in [('current', current), ('target', target), ('prediction', prediction),
                        ('projected_target', projected_target), ('projected_prediction', projected_prediction)]:
        stats[name] = representation(value)
    goal_size = getattr(critic.encoder, 'goal_latent_size', None)
    branches = {'full': slice(None)}
    if goal_size is not None:
        branches.update(goal=slice(0, goal_size), non_goal=slice(goal_size, None))
    for name, branch in branches.items():
        predicted, actual, previous = (v.detach().double()[:, branch] for v in (prediction, target, current))
        error = (predicted - actual).square().mean(-1)
        variance = actual.var(dim=0, unbiased=False).mean().item()
        persistence = (previous - actual).square().mean().item()
        stats['branches'][name] = {
            'mse_per_transition': distribution(error),
            'target_variance': variance,
            'mse_over_target_variance': error.mean().item() / variance if variance > 1e-12 else None,
            'persistence_mse': persistence,
            'mse_over_persistence': error.mean().item() / persistence if persistence > 1e-12 else None,
            'target': representation(actual), 'prediction': representation(predicted),
        }
    effective_iqe = loss_conf.weight * (loss_conf.iqe_weight if loss_conf.distance == 'iqe_mse'
                                        else float(loss_conf.distance == 'iqe'))
    effective_mse = loss_conf.weight * (loss_conf.mse_weight if loss_conf.distance == 'iqe_mse'
                                        else float(loss_conf.distance == 'mse'))
    for name, index in [('predicted_latent', slice(0, 1)), ('dynamics_parameters', slice(1, None))]:
        iq, ms = iqe_grad[index], mse_grad[index]
        stats['gradients'][name] = {
            'iqe_raw': gradient_stats(iq), 'mse_raw': gradient_stats(ms),
            'cosine': gradient_cosine(iq, ms),
            'iqe_effective_l2': effective_iqe * gradient_stats(iq)['l2'],
            'mse_effective_l2': effective_mse * gradient_stats(ms)['l2'],
        }
    stats['nonfinite_tensors'] = [name for name, value in tensors.items() if not torch.isfinite(value).all()]
    return finite_json(stats), tensors


def collect_fixed_probe(replay, conf):
    chunks = []
    with isolated_measurement():
        random.seed(conf.probe_seed)
        np.random.seed(conf.probe_seed)
        torch.manual_seed(conf.probe_seed)
        action_rng = np.random.default_rng(conf.probe_seed)
        env = replay.create_env()
        try:
            space = replay.env_spec.action_space
            if not np.isfinite(space.low).all() or not np.isfinite(space.high).all():
                raise ValueError('Dynamics probes require finite Box action bounds')
            for episode in range(conf.probe_episodes):
                env.seed(conf.probe_seed + episode)
                rollout = replay.collect_rollout(
                    lambda _obs, _goal, _space: action_rng.uniform(space.low, space.high).astype(space.dtype),
                    env=env)
                indices = torch.linspace(0, replay.episode_length - 1, conf.samples_per_episode).long()
                chunks.append({
                    'observations': rollout.all_observations[indices].clone(),
                    'actions': rollout.actions[indices].clone(),
                    'next_observations': rollout.all_observations[indices + 1].clone(),
                })
        finally:
            env.close()
    return {name: torch.cat([chunk[name] for chunk in chunks]) for name in chunks[0]}


class DynamicsDiagnostics:
    def __init__(self, conf, trainer, output_dir, *, start_env_steps, start_optim_steps, experiment_conf):
        self.conf, self.trainer = conf, trainer
        self.root = Path(output_dir) / 'dynamics_diagnostics'
        self.root.mkdir(exist_ok=True)
        self.last_step = start_env_steps
        if trainer.algorithm != 'qrl' or experiment_conf.agent.quasimetric_critic.model.latent_dynamics.kind != 'mlp':
            raise ValueError('Dynamics diagnostics currently require a QRL agent with MLP dynamics')
        probe_path = self.root / 'fixed_probe.pth'
        if probe_path.exists():
            saved = torch.load(probe_path, map_location='cpu', weights_only=False)
            if saved['protocol'] != self.probe_protocol():
                raise ValueError('Fixed probe protocol changed on resume')
            self.probe = saved['transitions']
            if tensor_digest(self.probe) != saved['sha256']:
                raise ValueError('Fixed probe checksum mismatch')
        else:
            if start_optim_steps > 0:
                raise ValueError('Cannot silently add step-zero diagnostics to an already trained run')
            self.probe = collect_fixed_probe(trainer.replay, conf)
            utils.atomic_torch_save({
                'protocol': self.probe_protocol(), 'transitions': self.probe,
                'sha256': tensor_digest(self.probe),
            }, probe_path)
        self.probe_hash = tensor_digest(self.probe)
        self.provenance = self.save_provenance(experiment_conf)
        # A crash may leave measurements beyond the last committed training state.
        for path in self.root.iterdir():
            match = re.fullmatch(r'(?:training_|validation_|test_)?env(\d+)_opt(\d+)\.(?:json|pth)', path.name)
            if match is None:
                continue
            env, opt = map(int, match.groups())
            if (env, opt) > (start_env_steps, start_optim_steps):
                path.unlink()
        self.last_record = None
        self.last_checkpoint = False
        self.training_values = {}
        self.training_start_optim = start_optim_steps + 1
        initial = self.root / 'env00000000_opt00000000.pth'
        if start_optim_steps > 0 and not initial.exists():
            raise ValueError('Missing initial diagnostic checkpoint on resume')
        if start_optim_steps == 0:
            self.record(0, 0, checkpoint=True)

    def probe_protocol(self):
        return {
            'seed': self.conf.probe_seed, 'episodes': self.conf.probe_episodes,
            'samples_per_episode': self.conf.samples_per_episode,
            'environment': self.trainer.replay.name,
            'episode_length': self.trainer.replay.episode_length,
            'policy': 'uniform_box_actions', 'used_for_training': False,
            'collection_env_steps': self.conf.probe_episodes * self.trainer.replay.episode_length,
        }

    def save_provenance(self, conf):
        root = Path(__file__).resolve().parents[1]
        sources = sorted(path for directory in ('online', 'quasimetric_rl', 'third_party/torch-quasimetric/torchqmet')
                         for path in (root / directory).rglob('*.py'))
        sources += sorted((root / 'configs/model_size').rglob('*.yaml'))
        sources += [root / relative for relative in (
            'tools/run_qrl_queue.py', 'tools/checkpoint_cleanup.py',
            'tools/check_m_residual_dynamics.py',
        ) if (root / relative).is_file()]
        for relative in self.conf.source_manifests:
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError(f'Missing or non-repository diagnostic source manifest: {relative}')
            sources.append(path)
        content = {str(path.relative_to(root)): path.read_bytes() for path in sources if path.exists()}
        hashes = {name: hashlib.sha256(value).hexdigest() for name, value in content.items()}
        versions = {dist.metadata['Name']: dist.version for dist in importlib.metadata.distributions()}
        device = next(self.trainer.agent.parameters()).device
        runtime = {
            'python': platform.python_version(), 'platform': platform.platform(),
            'device': str(device), 'torch_version': str(torch.__version__),
            'cuda_version': torch.version.cuda, 'cudnn_version': torch.backends.cudnn.version(),
            'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.machine(),
            'cpu_threads': torch.get_num_threads(), 'default_dtype': str(torch.get_default_dtype()),
            'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        }
        config_yaml = OmegaConf.to_yaml(conf)
        identity = hashlib.sha256(json.dumps([hashes, versions, config_yaml, runtime], sort_keys=True).encode()).hexdigest()
        archive = self.root / f'source_{identity}.zip'
        if not archive.exists():
            with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as handle:
                for name, value in content.items():
                    handle.writestr(name, value)
                handle.writestr('resolved_config.yaml', config_yaml)
        write_json(self.root / f'provenance_{identity}.json', {
            'schema_version': 1, 'source_sha256': hashes, 'versions': versions,
            'runtime': runtime,
            'resolved_config': OmegaConf.to_container(OmegaConf.structured(conf), resolve=True, enum_to_str=True),
            'num_parameters': sum(p.numel() for p in self.trainer.agent.parameters()),
            'measurement_mode': 'eval_fp32; encoder targets detached; no optimizer updates',
            'probe_protocol': self.probe_protocol(), 'probe_sha256': self.probe_hash,
        })
        return identity

    def record(self, env_steps, optim_steps, *, checkpoint=False, writer=None):
        if not checkpoint and env_steps // self.conf.interval <= self.last_step // self.conf.interval:
            return None
        stem = f'env{env_steps:08d}_opt{optim_steps:08d}'
        if self.last_record == (env_steps, optim_steps) and (not checkpoint or self.last_checkpoint):
            return stem + '.json'
        with isolated_measurement(self.trainer.agent):
            random.seed(self.conf.probe_seed + env_steps)
            np.random.seed(self.conf.probe_seed + env_steps)
            torch.manual_seed(self.conf.probe_seed + env_steps)
            panels = {'fixed': self.probe}
            if self.trainer.replay.num_transitions_realized:
                batch = self.trainer.replay.sample(len(self.probe['actions']))
                panels['replay'] = {name: getattr(batch, name).detach().cpu()
                                    for name in ('observations', 'actions', 'next_observations')}
            summary = {
                'schema_version': 1, 'env_steps': env_steps, 'optim_steps': optim_steps,
                'provenance': self.provenance, 'probe_sha256': self.probe_hash,
                'checkpoint_measurement': checkpoint, 'panels': {},
            }
            raw = {'env_steps': env_steps, 'optim_steps': optim_steps, 'panels': {}}
            for panel, transitions in panels.items():
                summary['panels'][panel], raw['panels'][panel] = {}, {}
                for index, critic in enumerate(self.trainer.agent.critics):
                    stats, tensors = measure_critic(
                        critic, transitions, self.trainer.losses.critic_losses[index].latent_dynamics)
                    key = f'critic_{index:02d}'
                    summary['panels'][panel][key] = stats
                    raw['panels'][panel][key] = tensors
                    if writer is not None:
                        for metric in ('mse', 'sq_dists', 'dist_p2n', 'dist_n2p', 'projected_mse'):
                            if stats[metric] is not None:
                                writer.add_scalar(f'dynamics_probe/{panel}/{key}/{metric}', stats[metric], env_steps)
            if checkpoint or self.conf.save_interval_models:
                raw['agent'] = self.trainer.agent.state_dict()
            raw['summary'] = summary
            utils.atomic_torch_save(raw, self.root / (stem + '.pth'))
            write_json(self.root / (stem + '.json'), summary)
        self.last_step, self.last_record = env_steps, (env_steps, optim_steps)
        self.last_checkpoint = checkpoint
        return stem + '.json'

    def observe_training_step(self, info):
        for key, critic_info in info.items():
            if key.startswith('critic_') and 'latent_dynamics' in critic_info:
                values = critic_info['latent_dynamics']
                self.training_values.setdefault(key, []).append(tuple(
                    values[name].detach() for name in ('mse', 'sq_dists', 'dist_p2n', 'dist_n2p')))

    def finish_training_cycle(self, env_steps, optim_steps):
        summary = {'env_steps': env_steps, 'optim_steps': optim_steps,
                   'first_optim_step': self.training_start_optim, 'critics': {}}
        batch_records = {}
        for key, rows in self.training_values.items():
            values = torch.stack(list(itertools.chain.from_iterable(rows))).reshape(-1, 4).double().cpu().numpy()
            batch_records[key] = torch.from_numpy(values)
            metrics = {}
            for index, name in enumerate(('mse', 'sq_dists', 'dist_p2n', 'dist_n2p')):
                column = values[:, index]
                finite = column[np.isfinite(column)]
                metrics[name] = {
                    'count': len(column), 'finite_count': len(finite),
                    'mean': float(finite.mean()) if len(finite) else None,
                    'std': float(finite.std()) if len(finite) else None,
                    'min': float(finite.min()) if len(finite) else None,
                    'max': float(finite.max()) if len(finite) else None,
                }
            summary['critics'][key] = metrics
        if self.conf.save_training_batches:
            utils.atomic_torch_save({
                'env_steps': env_steps, 'optim_steps': optim_steps,
                'first_optim_step': self.training_start_optim,
                'metric_names': ('mse', 'sq_dists', 'dist_p2n', 'dist_n2p'),
                'critics': batch_records,
            }, self.root / f'training_env{env_steps:08d}_opt{optim_steps:08d}.pth')
        write_json(self.root / f'training_env{env_steps:08d}_opt{optim_steps:08d}.json', finite_json(summary))
        self.training_values.clear()
        self.training_start_optim = optim_steps + 1

    def save_evaluation(self, result, summary):
        filename = f"{summary['split']}_env{summary['env_steps']:08d}_opt{summary['optim_steps']:08d}.pth"
        utils.atomic_torch_save({'summary': summary, 'result': attrs.asdict(result)}, self.root / filename)

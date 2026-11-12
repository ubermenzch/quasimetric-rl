"""Small scalar logs and compressed diagnostic samples; no model parameters."""
from collections.abc import Mapping
import json
import hashlib
import importlib.metadata
import math
import os
from pathlib import Path
import platform
import resource
import time
import zipfile

import numpy as np
import torch


class RunDiagnostics:
    def __init__(self, output_dir, *, protocol=None):
        self.root = Path(output_dir) / 'diagnostics'
        self.root.mkdir(parents=True, exist_ok=True)
        self.started = time.perf_counter()
        self.eval_index = 0
        self.probe_bytes = 0
        self.probe_count = 0
        self.training_log_count = 0
        source_root = Path.cwd()
        sources = ('online/main.py', 'online/trainer.py', 'quasimetric_rl/modules/__init__.py',
                   'quasimetric_rl/modules/actor/model.py', 'quasimetric_rl/modules/actor/losses/min_dist.py',
                   'quasimetric_rl/modules/quasimetric_critic/models/encoder.py',
                   'quasimetric_rl/modules/quasimetric_critic/losses/latent_dynamics.py',
                   'quasimetric_rl/modules/quasimetric_critic/losses/global_push.py',
                   'quasimetric_rl/modules/quasimetric_critic/losses/local_constraint.py',
                   'quasimetric_rl/modules/goal_marker.py',
                   'quasimetric_rl/data/base.py', 'quasimetric_rl/data/online/memory.py',
                   'quasimetric_rl/model_size.py', 'configs/model_size/cqrl/l.yaml',
                   'configs/model_size/qrl/l_aqrl_backbone.yaml', 'configs/model_size/qrl/s.yaml',
                   'quasimetric_rl/data/online/gcrl/fetch_envs.py',
                   'quasimetric_rl/data/online/dmc/__init__.py',
                   'quasimetric_rl/data/online/maze2d/__init__.py',
                   'quasimetric_rl/data/online/gymnasium_robotics/__init__.py',
                   'quasimetric_rl/data/online/panda_gym/__init__.py',
                   'quasimetric_rl/data/online/gym_mujoco/__init__.py')
        source_hashes = {}
        versions = {}
        for package in ('gym', 'gym-robotics', 'gymnasium', 'gymnasium-robotics',
                        'mujoco', 'mujoco-py', 'dm-control', 'panda-gym', 'd4rl'):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        with zipfile.ZipFile(self.root / 'source_code.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for name in sources:
                path = source_root / name
                if path.is_file():
                    source_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
                    archive.write(path, name)
            for name in ('rebuttal_diagnostics.py', 'run_native_qrl_logs_only.py', 'git_metadata.py'):
                path = Path(__file__).parent / name
                archive.write(path, 'instrumentation/' + name)
        self.write_json('protocol.json', {
            'protocol': protocol, 'python': platform.python_version(),
            'torch': torch.__version__, 'numpy': np.__version__,
            'started_at_unix': time.time(),
            'checkpoint_storage': 'disabled',
            'reference_scope': 'uniform replay sample, not the entire replay buffer',
            'source_root': str(source_root), 'source_sha256': source_hashes,
            'package_versions': versions,
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'cpu_threads': os.environ.get('OMP_NUM_THREADS'),
        })

    def write_json(self, name, data):
        target = self.root / name
        temporary = target.with_suffix(target.suffix + '.tmp')
        temporary.write_text(json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + '\n')
        temporary.replace(target)

    def record_runtime(self, trainer):
        agent = trainer.agent
        original_protocol = getattr(trainer, 'original_qrl_evaluation', not hasattr(trainer, 'validation_seed'))
        critics = []
        for critic in getattr(agent, 'critics', ()):
            encoder = critic.encoder
            critics.append(dict(
                parameter_count=sum(p.numel() for p in critic.parameters()),
                goal_dims=list(encoder.goal_dims) if hasattr(encoder, 'goal_dims') else None,
                goal_latent_size=getattr(encoder, 'goal_latent_size', None),
                non_goal_latent_size=getattr(encoder, 'non_goal_latent_size', None),
            ))
        self.write_json('runtime_model.json', dict(
            agent=str(agent), parameter_count=sum(p.numel() for p in agent.parameters()),
            critics=critics, actor_input_mode=getattr(getattr(agent, 'actor', None), 'input_mode', 'raw'),
            goal_marker_dims=getattr(agent, 'goal_marker_dims', None),
            observation_shape=list(trainer.replay.env_spec.observation_shape),
            action_shape=list(trainer.replay.env_spec.action_shape),
            episode_length=int(trainer.replay.episode_length),
            validation_seed=getattr(trainer, 'validation_seed', getattr(trainer, 'eval_seed', None)),
            test_seed=getattr(trainer, 'test_seed', None),
            evaluation_episode_seeding='single_stream' if original_protocol else 'base_plus_episode_index',
            evaluation_preserves_training_rng=not original_protocol,
            matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
            cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
        ))

    def _append(self, name, data):
        with (self.root / name).open('a') as stream:
            stream.write(json.dumps(data, sort_keys=True, allow_nan=False) + '\n')
            stream.flush()

    @staticmethod
    def scalars(value, prefix=''):
        result = {}
        if isinstance(value, Mapping):
            for key, item in value.items():
                result.update(RunDiagnostics.scalars(item, f'{prefix}/{key}' if prefix else str(key)))
        elif isinstance(value, torch.Tensor):
            scalar = float(value.detach().float().mean().cpu()) if value.numel() else None
            result[prefix] = scalar if scalar is None or math.isfinite(scalar) else None
        elif isinstance(value, (int, float, bool, np.number)):
            scalar = float(value)
            result[prefix] = scalar if math.isfinite(scalar) else None
        return result

    def _arrays(self, name, values, metadata):
        arrays = {key: (value.detach().float() if value.dtype == torch.bfloat16 else value.detach()).cpu().numpy() if isinstance(value, torch.Tensor)
                  else np.asarray(value) for key, value in values.items() if value is not None}
        arrays['metadata_json'] = np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False))
        target = self.root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix('.npz.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(target)
        return target.stat().st_size

    def record_training(self, env_steps, optim_steps, info, *, iteration_seconds=None,
                        data=None, agent=None):
        self.training_log_count += 1
        metrics = self.scalars(info)
        entry = {
            'env_steps': int(env_steps), 'optim_steps': int(optim_steps),
            'wall_time_unix': time.time(), 'elapsed_seconds': time.perf_counter() - self.started,
            'iteration_seconds': iteration_seconds, 'metrics': metrics,
            'nonfinite_or_empty_metric_keys': [key for key, value in metrics.items() if value is None],
        }
        if (self.training_log_count - 1) % 10 == 0:
            if agent is not None:
                values = {}
                for name, parameter in agent.named_parameters():
                    tensor = parameter.detach().float()
                    values[name + '/norm'] = tensor.norm()
                    values[name + '/max_abs'] = tensor.abs().max()
                    if parameter.grad is not None:
                        values[name + '/gradient_norm'] = parameter.grad.detach().float().norm()
                entry['parameter_statistics'] = self.scalars(values)
            if data is not None:
                arrays = {name: getattr(data, name)[:16] for name in (
                    'observations', 'actions', 'next_observations', 'future_observations',
                    'rewards', 'terminals', 'timeouts', 'future_steps',
                    'replay_episode_indices', 'replay_timesteps',
                    'goal_marker_sources', 'goal_marker_targets') if getattr(data, name, None) is not None}
                if getattr(data, 'next_observations', None) is not None:
                    arrays['random_actor_goals'] = torch.roll(data.next_observations, 1, 0)[:16]
                    marker_dims = getattr(agent, 'goal_marker_dims', None)
                    if marker_dims is not None:
                        from quasimetric_rl.modules.goal_marker import make_goal_marker
                        arrays['random_actor_goal_markers'] = make_goal_marker(
                            arrays['random_actor_goals'], marker_dims)
                        arrays['future_actor_goal_markers'] = make_goal_marker(
                            data.future_observations[:16], marker_dims)
                if agent is not None and getattr(data, 'future_steps', None) is not None:
                    # The observed trajectory length is a sampled path length,
                    # not a measurement of the optimal reaching cost.
                    with torch.no_grad():
                        for index, critic in enumerate(getattr(agent, 'critics', ())):
                            first = critic.encoder(data.observations[:16])
                            future = critic.encoder(data.future_observations[:16])
                            arrays[f'critic_{index:02d}_observed_future_distance'] = critic.quasimetric_model(first, future)
                self.record_probe('training_batch', int(optim_steps), arrays, env_steps=int(env_steps),
                                  metadata={'scope': 'first_16_examples_of_logged_batch',
                                            'future_steps_meaning': 'observed trajectory length, not optimal cost'})
        self._append('training.jsonl', entry)

    def record_evaluation(self, result, summary):
        self.eval_index += 1
        if isinstance(result, Mapping):
            values = result
        elif hasattr(type(result), '__attrs_attrs__'):
            values = {field.name: getattr(result, field.name) for field in type(result).__attrs_attrs__}
        else:
            values = vars(result)
        values = {key: value for key, value in values.items() if isinstance(value, torch.Tensor)}
        clean_summary = {key: value for key, value in summary.items()
                         if isinstance(value, (str, int, float, bool, type(None)))}
        clean_summary = {key: (value if not isinstance(value, float) or math.isfinite(value) else None)
                         for key, value in clean_summary.items()}
        name = f'evaluations/{self.eval_index:05d}_{summary.get("split", "validation")}.npz'
        self._arrays(name, values, clean_summary)
        self._append('evaluations.jsonl', dict(clean_summary, arrays=name, wall_time_unix=time.time()))

    def record_probe(self, kind, update, arrays, *, env_steps=None, metadata=None):
        name = f'probes/{kind}_{update:08d}.npz'
        meta = dict(metadata or {}, update=int(update), env_steps=env_steps,
                    wall_time_unix=time.time(), reference_scope='sampled_replay_bank')
        size = self._arrays(name, arrays, meta)
        self.probe_count += 1
        self.probe_bytes += size
        self._append('probes.jsonl', dict(meta, arrays=name, bytes=size))

    def finish(self, state, *, env_steps=None, optim_steps=None, error=None):
        usage = resource.getrusage(resource.RUSAGE_SELF)
        gpu_ready = torch.cuda.is_initialized()
        self.write_json('run_summary.json', dict(
            state=state, env_steps=env_steps, optim_steps=optim_steps, error=error,
            elapsed_seconds=time.perf_counter() - self.started,
            finished_at_unix=time.time(), probe_count=self.probe_count,
            probe_bytes=self.probe_bytes, evaluation_count=self.eval_index,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if gpu_ready else None,
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved() if gpu_ready else None,
            peak_host_rss_kib=usage.ru_maxrss, cpu_user_seconds=usage.ru_utime,
            cpu_system_seconds=usage.ru_stime,
        ))

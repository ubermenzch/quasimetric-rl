from typing import *

import os

import attrs
import glob
import logging
import json
import signal
import time

import hydra
import hydra.types
import hydra.core.config_store
from omegaconf import DictConfig

import torch
import torch.backends.cudnn
import torch.multiprocessing

import quasimetric_rl
from quasimetric_rl import utils, pdb_if_DEBUG, FLAGS

from quasimetric_rl.utils.steps_counter import StepsCounter
from quasimetric_rl.modules import InfoT
from quasimetric_rl.base_conf import BaseConf

from .trainer import Trainer, InteractionConf


class TrainingInterrupted(Exception):
    pass


@utils.singleton
@attrs.define(kw_only=True)
class Conf(BaseConf):
    output_base_dir: str = attrs.field(default=os.path.join(os.path.dirname(__file__), 'results'))

    resume_if_possible: bool = False

    env: quasimetric_rl.data.online.ReplayBuffer.Conf = quasimetric_rl.data.online.ReplayBuffer.Conf()

    batch_size: int = attrs.field(default=256, validator=attrs.validators.gt(0))
    interaction: InteractionConf = InteractionConf()

    log_steps: int = attrs.field(default=250, validator=attrs.validators.gt(0))
    eval_steps: int = attrs.field(default=2000, validator=attrs.validators.gt(0))
    save_steps: int = attrs.field(default=50000, validator=attrs.validators.gt(0))
    keep_only_latest_checkpoint: bool = True
    save_replay_buffer: bool = True
    save_final_replay_buffer: bool = True
    timing: utils.TimingConf = utils.TimingConf()



cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name='config', node=Conf())


@pdb_if_DEBUG
@hydra.main(version_base=None, config_name="config")
def train(dict_cfg: DictConfig):
    cfg: Conf = Conf.from_DictConfig(dict_cfg)
    if cfg.agent.training_schedule != 'joint':
        raise RuntimeError(
            "agent.training_schedule is currently implemented for offline training only. "
            "Online phased training needs an explicit replay/collection policy."
        )
    writer = cfg.setup_for_experiment()  # checking & setup logging
    profiler = cfg.timing.make(output_dir=cfg.output_dir, device=cfg.device.make())

    with profiler.record('setup/make_replay_buffer'):
        replay_buffer = cfg.env.make()

    with profiler.record('setup/make_trainer'):
        trainer = Trainer(
            agent_conf=cfg.agent,
            device=cfg.device.make(),
            replay=replay_buffer,
            batch_size=cfg.batch_size,
            interaction_conf=cfg.interaction,
            profiler=profiler,
        )

    val_results: List[dict] = []
    val_summaries: List[dict] = []
    step_counter = StepsCounter(
        alert_intervals=dict(
            log=cfg.log_steps,
            save=cfg.save_steps,
            eval=cfg.eval_steps,
        ),
    )
    next_cycle_sample = 0
    cycle_env_steps = 0

    def save(env_steps, optim_steps, *, suffix=None, extra=dict()):
        with profiler.record('checkpoint/save'):
            desc = f"env{env_steps:08d}_opt{optim_steps:08d}"
            if suffix is not None:
                desc += f'_{suffix}'
            utils.mkdir(cfg.output_dir)
            fullpath = os.path.join(cfg.output_dir, f'checkpoint_{desc}.pth')
            state_dicts = dict(
                env_steps=env_steps,
                optim_steps=optim_steps,
                agent=trainer.agent.state_dict(),
                losses=trainer.losses.state_dict(),
                rng=utils.rng_state_dict(),
                loop_state=dict(
                    next_cycle_sample=next_cycle_sample,
                    cycle_env_steps=cycle_env_steps,
                    replay_env_steps_offset=env_steps_offset,
                    steps_counter=step_counter.state_dict(),
                ),
                val_result=val_results[-1] if len(val_results) else None,
                val_summaries=val_summaries,
                **extra,
            )
            include_replay = cfg.save_replay_buffer and (suffix != 'final' or cfg.save_final_replay_buffer)
            if include_replay:
                state_dicts['replay'] = trainer.replay.state_dict()
            utils.atomic_torch_save(state_dicts, fullpath)
            if cfg.keep_only_latest_checkpoint:
                utils.prune_checkpoints(cfg.output_dir, fullpath)
            relpath = os.path.join('.', os.path.relpath(fullpath, os.path.dirname(__file__)))
            logging.info(f"Checkpointed to {relpath}")

    def load(ckpt) -> dict:
        state_dicts = torch.load(ckpt, map_location='cpu', weights_only=False)
        trainer.agent.load_state_dict(state_dicts['agent'])
        trainer.losses.load_state_dict(state_dicts['losses'])
        if 'replay' in state_dicts:
            trainer.replay.load_state_dict(state_dicts['replay'])
            logging.info(
                f"Restored replay buffer with "
                f"{trainer.replay.num_transitions_realized} transitions"
            )
        else:
            logging.warning(
                "Checkpoint has no replay buffer. Online resume will restore "
                "model/optimizer state but must collect a new replay buffer."
            )
        utils.load_rng_state(state_dicts.get('rng'))
        val_result = state_dicts.get('val_result')
        if val_result is not None:
            val_results.clear()
            val_results.append(val_result)
        saved_val_summaries = state_dicts.get('val_summaries')
        if saved_val_summaries is not None:
            val_summaries.clear()
            val_summaries.extend(saved_val_summaries)
        loop_state = state_dicts.get('loop_state', {})
        step_counter.load_state_dict(loop_state.get('steps_counter'))
        relpath = os.path.join('.', os.path.relpath(ckpt, os.path.dirname(__file__)))
        logging.info(f"Loaded from {relpath}")
        return state_dicts

    ckpts = {}
    for ckpt in sorted(glob.glob(os.path.join(glob.escape(cfg.output_dir), 'checkpoint_env*_opt*.pth'))):
        base = os.path.basename(ckpt).rsplit('.', 1)[0]
        parts = base.split('_')
        if len(parts) < 3:
            continue
        env_part, opt_part = parts[1], parts[2]
        if not (env_part.startswith('env') and opt_part.startswith('opt')):
            continue
        try:
            env_steps = int(env_part.removeprefix('env'))
            optim_steps = int(opt_part.removeprefix('opt'))
        except ValueError:
            continue
        ckpts[env_steps, optim_steps] = ckpt

    loaded_replay = False
    replay_env_steps_offset = 0
    if cfg.resume_if_possible and len(ckpts) > 0:
        start_env_steps, start_optim_steps = max(ckpts.keys())
        logging.info(f'Load from existing checkpoint: {ckpts[start_env_steps, start_optim_steps]}')
        loaded_state_dicts = load(ckpts[start_env_steps, start_optim_steps])
        loaded_replay = 'replay' in loaded_state_dicts
        loop_state = loaded_state_dicts.get('loop_state', {})
        if loaded_replay:
            next_cycle_sample = int(loop_state.get('next_cycle_sample', trainer.num_samples_per_cycle))
            replay_env_steps_offset = int(loop_state.get('replay_env_steps_offset', 0))
            cycle_env_steps = int(loop_state.get('cycle_env_steps', start_env_steps))
        logging.info(f'Fast forward to env_steps={start_env_steps} optim_steps={start_optim_steps}')
    else:
        start_env_steps, start_optim_steps = 0, 0

    original_total_env_steps = trainer.total_env_steps
    env_steps_offset = replay_env_steps_offset if loaded_replay else start_env_steps
    if loaded_replay and env_steps_offset + trainer.replay.num_transitions_realized > cycle_env_steps:
        next_cycle_sample = 0
    if start_env_steps > 0 and not loaded_replay:
        trainer.total_env_steps = max(0, original_total_env_steps - start_env_steps)
        logging.info(
            f'Resume remaining_env_steps={trainer.total_env_steps} '
            f'from original_total_env_steps={original_total_env_steps}'
        )
    elif loaded_replay:
        trainer.total_env_steps = max(0, original_total_env_steps - env_steps_offset)
        logging.info(
            f'Resume with restored replay buffer at replay_env_steps_offset={env_steps_offset}; '
            f'continuing for local_total_env_steps={trainer.total_env_steps}'
        )

    def eval(env_steps, optim_steps):
        with profiler.record('eval/total'):
            val_result = trainer.evaluate()
        val_results.clear()
        val_results.append(dict(
            env_steps=env_steps,
            optim_steps=optim_steps,
            result=attrs.asdict(val_result),
        ))
        epi_return = val_result.episode_return
        succ_rate = val_result.is_success
        succ_rate_ts = val_result.timestep_is_success.mean(dtype=torch.float32, dim=-1)
        hitting_time = torch.where(
            val_result.hitting_time < 0, trainer.replay.episode_length + 1, val_result.hitting_time,
        )
        val_summaries.append(dict(
            env_steps=env_steps,
            optim_steps=optim_steps,
            epi_return=epi_return,
            succ_rate_ts=succ_rate_ts,
            succ_rate=succ_rate,
            hitting_time=hitting_time,
        ))
        for k, v in val_summaries[-1].items():
            if k == 'env_steps':
                continue
            if isinstance(v, torch.Tensor):
                v = v.to(torch.float64).mean().item()
            writer.add_scalar(f"eval/{k}", v, env_steps)
        with open(os.path.join(cfg.output_dir, 'eval.log'), 'a') as f:
            print(
                json.dumps({
                    k: (v.to(torch.float64).mean().item() if isinstance(v, torch.Tensor) else v)
                    for k, v in val_summaries[-1].items()
                }),
                file=f,
            )
        logging.info(
            f"EVAL: " +
            "  ".join([
                f"env_steps={env_steps}",
                f"optim_steps={optim_steps}",
                f"succ_rate={succ_rate.to(torch.float64).mean().item():.2%}",
                f"epi_return={epi_return.to(torch.float64).mean().item():.4f}",
            ])
        )

    def log_tensorboard(env_steps, info: InfoT, prefix: str):
        for k, v in info.items():
            if isinstance(v, Mapping):
                log_tensorboard(env_steps, v, prefix=f"{prefix}{k}/")
                continue
            if isinstance(v, torch.Tensor):
                v = v.to(torch.float64).mean().item()
            writer.add_scalar(f"{prefix}{k}", v, env_steps)

    def current_env_steps() -> int:
        return min(
            env_steps_offset + trainer.replay.num_transitions_realized,
            env_steps_offset + trainer.total_env_steps,
        )

    def handle_signal(signum, _frame):
        raise TrainingInterrupted(f"received signal {signum}")

    old_sigterm = signal.getsignal(signal.SIGTERM)
    old_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    # Training loop
    optim_steps = start_optim_steps
    try:
        if start_optim_steps == 0:
            eval(0, 0); save(0, 0)
            profiler.log(writer=writer, step=0, step_name='env_steps', extra=dict(optim_steps=0))
        if trainer.total_env_steps > 0:
            for local_optim_steps, (local_env_steps, next_iter_new_env_step, data, data_info) in enumerate(
                    trainer.iter_training_data(start_cycle_sample=next_cycle_sample), start=1):
                optim_steps = start_optim_steps + local_optim_steps
                env_steps = env_steps_offset + local_env_steps
                cycle_env_steps = env_steps

                with profiler.record('train/iteration'):
                    iter_t0 = time.time()
                    train_info = trainer.train_step(data)
                    iter_time = time.time() - iter_t0
                next_cycle_sample = int(data_info.get('cycle_sample', 0)) + 1

                # bookkeep
                if not next_iter_new_env_step:
                    continue  # just train more, only eval/log/save right before new env step
                next_cycle_sample = trainer.num_samples_per_cycle
                step_counter.update_to_then_record_alerts(env_steps)

                if step_counter.alerts.eval:
                    eval(env_steps, optim_steps)

                if step_counter.alerts.save:
                    save(env_steps, optim_steps)

                if step_counter.alerts.log:
                    log_tensorboard(env_steps, data_info, 'data/')
                    log_tensorboard(env_steps, train_info, 'train_')
                    writer.add_scalar("train/iter_time", iter_time, env_steps)
                    writer.add_scalar("train/optim_steps", optim_steps, env_steps)
                    profiler.log(writer=writer, step=env_steps, step_name='env_steps', extra=dict(optim_steps=optim_steps))
                next_cycle_sample = 0

        final_env_steps = env_steps_offset + trainer.total_env_steps
        next_cycle_sample = trainer.num_samples_per_cycle
        cycle_env_steps = final_env_steps
        eval(final_env_steps, optim_steps)
        save(final_env_steps, optim_steps, suffix='final')
        profiler.log(writer=writer, step=final_env_steps, step_name='env_steps', extra=dict(optim_steps=optim_steps))
        open(cfg.completion_file, 'a').close()
    except TrainingInterrupted as exc:
        env_steps = current_env_steps()
        logging.warning(f"Interrupted: {exc}. Saving resumable checkpoint at env_steps={env_steps} optim_steps={optim_steps}")
        save(env_steps, optim_steps)
        raise SystemExit(130)
    finally:
        signal.signal(signal.SIGTERM, old_sigterm)
        signal.signal(signal.SIGINT, old_sigint)


if __name__ == '__main__':
    if 'MUJOCO_GL' not in os.environ:
        os.environ['MUJOCO_GL'] = 'egl'

    # set up some hydra flags before parsing
    os.environ['HYDRA_FULL_ERROR'] = str(int(FLAGS.DEBUG))

    train()

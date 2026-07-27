from typing import *

import os

import glob
import attrs
import logging
import signal
import time

import hydra
import hydra.types
import hydra.core.config_store
from omegaconf import DictConfig

from tqdm.auto import tqdm
import numpy as np
import torch
import torch.backends.cudnn

import quasimetric_rl
from quasimetric_rl import utils, pdb_if_DEBUG, FLAGS

from quasimetric_rl.utils.steps_counter import StepsCounter
from quasimetric_rl.modules import InfoT
from quasimetric_rl.base_conf import BaseConf
from quasimetric_rl.model_size import register_model_size_presets

from .trainer import Trainer


class TrainingInterrupted(Exception):
    pass


def training_phases(training_schedule: str) -> Tuple[str, ...]:
    if training_schedule == 'joint':
        return ('all',)
    if training_schedule == 'critic_then_dynamics_then_actor':
        return ('critic', 'latent_dynamics', 'actor')
    if training_schedule == 'critic_then_dynamics_then_goal_set_distance_then_actor':
        return ('critic', 'latent_dynamics', 'goal_set_distance', 'actor')
    return ()


@utils.singleton
@attrs.define(kw_only=True)
class Conf(BaseConf):
    output_base_dir: str = attrs.field(default=os.path.join(os.path.dirname(__file__), 'results'))

    resume_if_possible: bool = False

    env: quasimetric_rl.data.Dataset.Conf = quasimetric_rl.data.Dataset.Conf()

    batch_size: int = attrs.field(default=4096, validator=attrs.validators.gt(0))
    num_workers: int = attrs.field(default=8, validator=attrs.validators.ge(0))
    total_optim_steps: int = attrs.field(default=int(2e5), validator=attrs.validators.gt(0))

    log_steps: int = attrs.field(default=250, validator=attrs.validators.gt(0))
    save_steps: int = attrs.field(default=10000, validator=attrs.validators.gt(0))
    # Retention policy for periodic Agent-only evaluation snapshots.
    keep_only_latest_checkpoint: bool = False
    timing: utils.TimingConf = utils.TimingConf()



cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name='config', node=Conf())
register_model_size_presets(cs)


@pdb_if_DEBUG
@hydra.main(version_base=None, config_name="config")
def train(dict_cfg: DictConfig):
    cfg: Conf = Conf.from_DictConfig(dict_cfg)
    writer = cfg.setup_for_experiment()  # checking & setup logging
    profiler = cfg.timing.make(output_dir=cfg.output_dir, device=cfg.device.make())

    with profiler.record('setup/make_dataset'):
        dataset = cfg.env.make()

    batch_kwargs = dict(drop_last=True)

    with profiler.record('setup/make_trainer'):
        trainer = Trainer(
            agent_conf=cfg.agent,
            device=cfg.device.make(),
            dataset=dataset,
            batch_size=cfg.batch_size,
            total_optim_steps=cfg.total_optim_steps,
            dataloader_kwargs=batch_kwargs,
            profiler=profiler,
            data_seed=cfg.seed,
        )
    if cfg.agent.training_schedule in (
            'critic_then_dynamics_then_actor',
            'critic_then_dynamics_then_goal_set_distance_then_actor'):
        if not cfg.agent.quasimetric_critic.losses.separate_latent_dynamics:
            raise RuntimeError(
                f"agent.training_schedule={cfg.agent.training_schedule} requires "
                "agent.quasimetric_critic.losses.separate_latent_dynamics=true"
            )
        if cfg.agent.actor is None:
            raise RuntimeError(
                f"agent.training_schedule={cfg.agent.training_schedule} requires agent.actor to be enabled"
            )
    if cfg.agent.training_schedule == 'critic_then_dynamics_then_goal_set_distance_then_actor':
        if (not cfg.agent.goal_set_distance.enabled
                or cfg.agent.goal_set_distance.losses.implementation != 'learned'):
            raise RuntimeError(
                "agent.training_schedule=critic_then_dynamics_then_goal_set_distance_then_actor requires "
                "a learned goal-set objective"
            )
    phases = training_phases(cfg.agent.training_schedule)
    if not phases:
        raise ValueError(f"Unknown training schedule: {cfg.agent.training_schedule!r}")

    # Periodic Agent-only snapshots are evaluation artifacts. The rolling and
    # final checkpoints carry the full optimizer/RNG/data state needed to resume.
    resume_checkpoint = os.path.join(
        cfg.output_dir, utils.RESUME_CHECKPOINT_FILENAME
    )

    def checkpoint_metadata(epoch, it, checkpoint_kind):
        return dict(
            epoch=epoch,
            it=it,
            optim_steps=optim_steps,
            checkpoint_kind=checkpoint_kind,
        )

    def full_checkpoint_state(epoch, it, checkpoint_kind):
        return dict(
            **checkpoint_metadata(epoch, it, checkpoint_kind),
            agent=trainer.agent.state_dict(),
            losses=trainer.losses.state_dict(),
            rng=utils.rng_state_dict(),
            data=trainer.data_state_dict(),
            loop_state=dict(
                optim_steps=optim_steps,
                phase_index=phase_index,
                phase_step=phase_step,
                phase_name=phase_name,
                training_schedule=cfg.agent.training_schedule,
                periodic_save_pending=periodic_save_pending,
                steps_counter=step_counter.state_dict(),
            ),
        )

    def write_checkpoint(state_dicts, fullpath, checkpoint_kind):
        with profiler.record('checkpoint/save'):
            utils.mkdir(cfg.output_dir)
            utils.atomic_torch_save(state_dicts, fullpath)
        relpath = os.path.join('.', os.path.relpath(fullpath, os.path.dirname(__file__)))
        logging.info(
            f"Saved {checkpoint_kind} checkpoint at "
            f"optim_steps={optim_steps} to {relpath}"
        )

    def save_full(epoch, it, *, fullpath, checkpoint_kind):
        write_checkpoint(
            full_checkpoint_state(epoch, it, checkpoint_kind),
            fullpath,
            checkpoint_kind,
        )

    def save_resume(epoch, it):
        save_full(
            epoch,
            it,
            fullpath=resume_checkpoint,
            checkpoint_kind='resume',
        )

    def save_agent(epoch, it):
        filename = utils.agent_checkpoint_filename(optim_steps)
        fullpath = os.path.join(cfg.output_dir, filename)
        state_dicts = dict(
            **checkpoint_metadata(epoch, it, 'agent'),
            agent=trainer.agent.state_dict(),
        )
        write_checkpoint(state_dicts, fullpath, 'Agent-only')
        if cfg.keep_only_latest_checkpoint:
            utils.prune_checkpoints(
                cfg.output_dir,
                fullpath,
                pattern='agent_checkpoint_step*.pth',
                preserve_final=False,
            )

    def save_periodic(epoch, it):
        # Write the archive first. If the process stops between these writes,
        # resuming from the previous full checkpoint will recreate it safely.
        save_agent(epoch, it)
        save_resume(epoch, it)

    def save_final(epoch, it):
        fullpath = os.path.join(
            cfg.output_dir,
            f'checkpoint_{epoch:05d}_{it:05d}_final.pth',
        )
        save_full(
            epoch,
            it,
            fullpath=fullpath,
            checkpoint_kind='final',
        )

    def load(ckpt):
        state_dicts = torch.load(ckpt, map_location='cpu', weights_only=False)
        if 'losses' not in state_dicts:
            raise RuntimeError(
                f"Checkpoint {ckpt} is Agent-only and cannot resume training. "
                f"Use {utils.RESUME_CHECKPOINT_FILENAME} or a *_final.pth checkpoint."
            )
        utils.validate_training_cursor(state_dicts, trainer.num_batches)
        trainer.agent.load_state_dict(state_dicts['agent'])
        trainer.losses.load_state_dict(state_dicts['losses'])
        trainer.load_data_state_dict(state_dicts.get('data'))
        utils.load_rng_state(state_dicts.get('rng'))
        loop_state = state_dicts.get('loop_state', {})
        step_counter.load_state_dict(loop_state.get('steps_counter'))
        relpath = os.path.join('.', os.path.relpath(ckpt, os.path.dirname(__file__)))
        logging.info(f"Loaded from {relpath}")
        return state_dicts

    # step counter to keep track of when to save
    step_counter = StepsCounter(
        alert_intervals=dict(
            log=cfg.log_steps,
            save=cfg.save_steps,
        ),
    )
    num_total_optim_steps = cfg.total_optim_steps * len(phases)
    num_total_epochs = int(np.ceil(num_total_optim_steps / trainer.num_batches))
    logging.info(
        "Training plan: "
        f"schedule={cfg.agent.training_schedule} "
        f"total_optim_steps_per_phase={cfg.total_optim_steps} "
        f"num_phases={len(phases)} "
        f"num_batches_per_epoch={trainer.num_batches} "
        f"total_optim_steps={num_total_optim_steps} "
        f"total_epochs={num_total_epochs}"
    )
    optim_steps = 0
    phase_index = 0
    phase_step = 0
    phase_name = 'joint'
    periodic_save_pending = False

    ckpts = {}  # Legacy/full final checkpoints: (epoch, iter) -> path.
    for ckpt in sorted(glob.glob(os.path.join(glob.escape(cfg.output_dir), 'checkpoint_*.pth'))):
        checkpoint_key = utils.full_checkpoint_key(ckpt)
        if checkpoint_key is None:
            continue
        epoch, it, _is_final = checkpoint_key
        ckpts[epoch, it] = ckpt

    selected_ckpt = None
    selected_cursor = None
    if cfg.resume_if_possible:
        # A completed run should resume from final if cleanup was interrupted.
        # During an active run the rolling checkpoint supersedes older full
        # checkpoints, including a final copied in as a continuation source.
        if os.path.exists(cfg.completion_file) and ckpts:
            selected_cursor = max(ckpts.keys())
            selected_ckpt = ckpts[selected_cursor]
        elif os.path.isfile(resume_checkpoint):
            selected_ckpt = resume_checkpoint
        elif ckpts:
            selected_cursor = max(ckpts.keys())
            selected_ckpt = ckpts[selected_cursor]

    loaded_checkpoint = selected_ckpt is not None
    if loaded_checkpoint:
        logging.info(f'Load from existing checkpoint: {selected_ckpt}')
        loaded_state_dicts = load(selected_ckpt)
        loaded_loop_state = loaded_state_dicts.get('loop_state', {})
        if selected_cursor is None:
            start_epoch = int(loaded_state_dicts.get('epoch', 0))
            start_it = int(loaded_state_dicts.get('it', 0))
        else:
            start_epoch, start_it = selected_cursor
        optim_steps = int(loaded_loop_state.get(
            'optim_steps',
            start_epoch * trainer.num_batches + start_it + 1,
        ))
        phase_index = int(loaded_loop_state.get(
            'phase_index',
            0 if cfg.agent.training_schedule == 'joint' else optim_steps // cfg.total_optim_steps,
        ))
        phase_step = int(loaded_loop_state.get(
            'phase_step',
            0 if cfg.agent.training_schedule == 'joint' else optim_steps % cfg.total_optim_steps,
        ))
        phase_name = str(loaded_loop_state.get('phase_name', phase_name))
        periodic_save_pending = bool(loaded_loop_state.get('periodic_save_pending', False))
        if 'data' not in loaded_state_dicts:
            next_batch_idx = min(start_it + 1, trainer.num_batches)
            if 'loop_state' not in loaded_state_dicts and start_epoch == 0 and start_it == 0:
                next_batch_idx = 0
                optim_steps = 0
                phase_index = 0
                phase_step = 0
            trainer.load_data_state_dict(dict(
                epoch=start_epoch,
                next_batch_idx=next_batch_idx,
            ))
        logging.info(f'Fast forward to epoch={start_epoch} iter={start_it}')
    else:
        start_epoch, start_it = 0, 0

    def log_tensorboard(optim_steps, info: InfoT, prefix: str):  # logging helper
        for k, v in info.items():
            if isinstance(v, Mapping):
                log_tensorboard(optim_steps, v, prefix=f"{prefix}{k}/")
                continue
            if isinstance(v, torch.Tensor):
                v = v.to(torch.float64).mean().item()
            writer.add_scalar(f"{prefix}{k}", v, optim_steps)

    if not loaded_checkpoint:
        save_resume(0, 0)
        profiler.log(writer=writer, step=0, step_name='optim_steps')

    pending_signal: Optional[int] = None

    def handle_signal(signum, _frame):
        nonlocal pending_signal
        pending_signal = signum

    def stop_at_step_boundary():
        if pending_signal is not None:
            raise TrainingInterrupted(f"received signal {pending_signal}")

    def update_step_alerts():
        nonlocal periodic_save_pending
        step_counter.update_then_record_alerts()
        periodic_save_pending = step_counter.alerts.save or periodic_save_pending

    def save_periodic_at_epoch_boundary():
        nonlocal periodic_save_pending
        if not periodic_save_pending:
            return
        periodic_save_pending = False
        save_periodic(
            trainer.batch_iterator.epoch,
            trainer.batch_iterator.next_batch_idx,
        )

    old_sigterm = signal.getsignal(signal.SIGTERM)
    old_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    try:
        if cfg.agent.training_schedule == 'joint' and trainer.batch_iterator.epoch < num_total_epochs:
            for epoch in range(trainer.batch_iterator.epoch, num_total_epochs):
                epoch_desc = f"Train epoch {epoch:05d}/{num_total_epochs:05d}"
                for epoch, it, data, data_info in tqdm(trainer.iter_training_data(), total=trainer.num_batches, desc=epoch_desc):
                    update_step_alerts()
                    optim_steps += 1
                    with profiler.record('train/iteration'):
                        iter_t0 = time.time()
                        train_info = trainer.train_step(data)
                        iter_time = time.time() - iter_t0
                    trainer.advance_batch()

                    if step_counter.alerts.log:
                        log_tensorboard(optim_steps, data_info, 'data/')
                        log_tensorboard(optim_steps, train_info, 'train_')
                        writer.add_scalar("train/iter_time", iter_time, optim_steps)
                        profiler.log(writer=writer, step=optim_steps, step_name='optim_steps')
                    stop_at_step_boundary()
                trainer.advance_epoch()
                save_periodic_at_epoch_boundary()
                stop_at_step_boundary()
        elif cfg.agent.training_schedule in (
                'critic_then_dynamics_then_actor',
                'critic_then_dynamics_then_goal_set_distance_then_actor'):
            phase_index = max(0, min(phase_index, len(phases)))
            while phase_index < len(phases):
                phase_name = phases[phase_index]
                while phase_step < cfg.total_optim_steps:
                    epoch_desc = (
                        f"{phase_name} phase {phase_step:08d}/{cfg.total_optim_steps:08d} "
                        f"epoch {trainer.batch_iterator.epoch:05d}"
                    )
                    for epoch, it, data, data_info in tqdm(
                            trainer.iter_training_data(),
                            total=trainer.num_batches,
                            desc=epoch_desc):
                        update_step_alerts()
                        optim_steps += 1
                        phase_step += 1
                        with profiler.record('train/iteration'):
                            iter_t0 = time.time()
                            train_info = trainer.train_step(data, phase=phase_name)
                            iter_time = time.time() - iter_t0
                        trainer.advance_batch()

                        if step_counter.alerts.log:
                            log_tensorboard(optim_steps, data_info, 'data/')
                            log_tensorboard(optim_steps, train_info, 'train_')
                            writer.add_scalar("train/iter_time", iter_time, optim_steps)
                            writer.add_scalar("train/phase_index", phase_index, optim_steps)
                            writer.add_scalar("train/phase_step", phase_step, optim_steps)
                            profiler.log(writer=writer, step=optim_steps, step_name='optim_steps')

                        stop_at_step_boundary()
                        if phase_step >= cfg.total_optim_steps:
                            break
                    if trainer.batch_iterator.next_batch_idx >= trainer.num_batches:
                        trainer.advance_epoch()
                        save_periodic_at_epoch_boundary()
                        stop_at_step_boundary()
                phase_index += 1
                phase_step = 0

        save_final(
            trainer.batch_iterator.epoch,
            trainer.batch_iterator.next_batch_idx,
        )
        profiler.log(writer=writer, step=optim_steps, step_name='optim_steps')
        open(cfg.completion_file, 'a').close()
        if utils.rm_if_exists(resume_checkpoint):
            logging.info(
                f"Removed rolling resume checkpoint after successful completion: "
                f"{resume_checkpoint}"
            )
    except TrainingInterrupted as exc:
        interrupted_epoch = trainer.batch_iterator.epoch
        interrupted_it = trainer.batch_iterator.next_batch_idx
        logging.warning(
            f"Interrupted: {exc}. Saving resumable checkpoint at "
            f"epoch={interrupted_epoch} it={interrupted_it}"
        )
        save_resume(interrupted_epoch, interrupted_it)
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

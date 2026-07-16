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
    save_steps: int = attrs.field(default=2000, validator=attrs.validators.gt(0))
    keep_only_latest_checkpoint: bool = True
    timing: utils.TimingConf = utils.TimingConf()



cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name='config', node=Conf())


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

    # save, load, and resume
    def save(epoch, it, *, suffix=None, extra=dict()):
        with profiler.record('checkpoint/save'):
            desc = f"{epoch:05d}_{it:05d}"
            if suffix is not None:
                desc += f'_{suffix}'
            utils.mkdir(cfg.output_dir)
            fullpath = os.path.join(cfg.output_dir, f'checkpoint_{desc}.pth')
            state_dicts = dict(
                epoch=epoch,
                it=it,
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
                    steps_counter=step_counter.state_dict(),
                ),
                **extra,
            )
            utils.atomic_torch_save(state_dicts, fullpath)
            if cfg.keep_only_latest_checkpoint:
                utils.prune_checkpoints(cfg.output_dir, fullpath)
            relpath = os.path.join('.', os.path.relpath(fullpath, os.path.dirname(__file__)))
            logging.info(f"Checkpointed to {relpath}")

    def load(ckpt):
        state_dicts = torch.load(ckpt, map_location='cpu', weights_only=False)
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

    ckpts = {}  # (epoch, iter) -> path
    for ckpt in sorted(glob.glob(os.path.join(glob.escape(cfg.output_dir), 'checkpoint_*.pth'))):
        parts = os.path.basename(ckpt).rsplit('.', 1)[0].split('_')
        if len(parts) < 3 or not parts[1].isdigit() or not parts[2].isdigit():
            continue
        epoch, it = int(parts[1]), int(parts[2])
        ckpts[epoch, it] = ckpt

    if cfg.resume_if_possible and len(ckpts) > 0:
        start_epoch, start_it = max(ckpts.keys())
        logging.info(f'Load from existing checkpoint: {ckpts[start_epoch, start_it]}')
        loaded_state_dicts = load(ckpts[start_epoch, start_it])
        optim_steps = int(loaded_state_dicts.get('loop_state', {}).get(
            'optim_steps',
            start_epoch * trainer.num_batches + start_it + 1,
        ))
        phase_index = int(loaded_state_dicts.get('loop_state', {}).get(
            'phase_index',
            0 if cfg.agent.training_schedule == 'joint' else optim_steps // cfg.total_optim_steps,
        ))
        phase_step = int(loaded_state_dicts.get('loop_state', {}).get(
            'phase_step',
            0 if cfg.agent.training_schedule == 'joint' else optim_steps % cfg.total_optim_steps,
        ))
        phase_name = str(loaded_state_dicts.get('loop_state', {}).get('phase_name', phase_name))
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

    if start_epoch == 0 and start_it == 0:
        save(0, 0)
        profiler.log(writer=writer, step=0, step_name='optim_steps')

    current_epoch = start_epoch
    current_it = start_it

    def handle_signal(signum, _frame):
        raise TrainingInterrupted(f"received signal {signum}")

    old_sigterm = signal.getsignal(signal.SIGTERM)
    old_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    try:
        if cfg.agent.training_schedule == 'joint' and trainer.batch_iterator.epoch < num_total_epochs:
            for epoch in range(trainer.batch_iterator.epoch, num_total_epochs):
                epoch_desc = f"Train epoch {epoch:05d}/{num_total_epochs:05d}"
                for epoch, it, data, data_info in tqdm(trainer.iter_training_data(), total=trainer.num_batches, desc=epoch_desc):
                    current_epoch = epoch
                    current_it = it
                    step_counter.update_then_record_alerts()
                    optim_steps += 1
                    with profiler.record('train/iteration'):
                        iter_t0 = time.time()
                        train_info = trainer.train_step(data)
                        iter_time = time.time() - iter_t0
                    trainer.advance_batch()

                    if step_counter.alerts.save:
                        save(epoch, it)

                    if step_counter.alerts.log:
                        log_tensorboard(optim_steps, data_info, 'data/')
                        log_tensorboard(optim_steps, train_info, 'train_')
                        writer.add_scalar("train/iter_time", iter_time, optim_steps)
                        profiler.log(writer=writer, step=optim_steps, step_name='optim_steps')
                trainer.advance_epoch()
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
                        current_epoch = epoch
                        current_it = it
                        step_counter.update_then_record_alerts()
                        optim_steps += 1
                        phase_step += 1
                        with profiler.record('train/iteration'):
                            iter_t0 = time.time()
                            train_info = trainer.train_step(data, phase=phase_name)
                            iter_time = time.time() - iter_t0
                        trainer.advance_batch()

                        if step_counter.alerts.save:
                            save(epoch, it)

                        if step_counter.alerts.log:
                            log_tensorboard(optim_steps, data_info, 'data/')
                            log_tensorboard(optim_steps, train_info, 'train_')
                            writer.add_scalar("train/iter_time", iter_time, optim_steps)
                            writer.add_scalar("train/phase_index", phase_index, optim_steps)
                            writer.add_scalar("train/phase_step", phase_step, optim_steps)
                            profiler.log(writer=writer, step=optim_steps, step_name='optim_steps')

                        if phase_step >= cfg.total_optim_steps:
                            break
                    if trainer.batch_iterator.next_batch_idx >= trainer.num_batches:
                        trainer.advance_epoch()
                phase_index += 1
                phase_step = 0

        save(trainer.batch_iterator.epoch, trainer.batch_iterator.next_batch_idx, suffix='final')
        profiler.log(writer=writer, step=optim_steps, step_name='optim_steps')
        open(cfg.completion_file, 'a').close()
    except TrainingInterrupted as exc:
        logging.warning(
            f"Interrupted: {exc}. Saving resumable checkpoint at "
            f"epoch={current_epoch} it={current_it}"
        )
        save(current_epoch, current_it)
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

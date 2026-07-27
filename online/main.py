from typing import *

import os

import attrs
import glob
import logging
import json
import re
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
from quasimetric_rl.data.base import GOAL_SET_DIMS_REGISTRY
from quasimetric_rl.model_size import register_model_size_presets

from .trainer import Trainer, InteractionConf


class TrainingInterrupted(Exception):
    pass


ONLINE_CHECKPOINT_KIND_COMMITTED = 'online_committed'
ONLINE_CHECKPOINT_KIND_INTERRUPTED = 'online_interrupted'
_ONLINE_CHECKPOINT_RE = re.compile(
    r'^checkpoint_env(\d+)_opt(\d+)(?:_[^.]+)?\.pth$'
)


def online_checkpoint_key(path: str) -> Optional[Tuple[int, int]]:
    match = _ONLINE_CHECKPOINT_RE.fullmatch(os.path.basename(path))
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def committed_online_checkpoint_issue(
        state_dicts: Mapping[str, Any], *, expected_env_steps: int,
        expected_optim_steps: int, num_samples_per_cycle: int) -> Optional[str]:
    """Return why a checkpoint is not a complete online cycle boundary."""
    try:
        env_steps = int(state_dicts['env_steps'])
        optim_steps = int(state_dicts['optim_steps'])
    except (KeyError, TypeError, ValueError):
        return 'missing or invalid env_steps/optim_steps'
    if (env_steps, optim_steps) != (expected_env_steps, expected_optim_steps):
        return (
            f'filename cursor ({expected_env_steps}, {expected_optim_steps}) '
            f'does not match payload cursor ({env_steps}, {optim_steps})'
        )

    checkpoint_kind = state_dicts.get('checkpoint_kind')
    if checkpoint_kind is not None and checkpoint_kind != ONLINE_CHECKPOINT_KIND_COMMITTED:
        return f'checkpoint_kind={checkpoint_kind!r}'

    # Checkpoints written before checkpoint_kind was introduced are committed
    # only when they contain the checkpoint-bound validation record.
    validation_summary = state_dicts.get('validation_summary')
    if not isinstance(validation_summary, Mapping):
        return 'missing checkpoint-bound validation_summary'
    try:
        validation_cursor = (
            int(validation_summary['env_steps']),
            int(validation_summary['optim_steps']),
        )
    except (KeyError, TypeError, ValueError):
        return 'invalid validation_summary cursor'
    if validation_cursor != (env_steps, optim_steps):
        return (
            f'validation cursor {validation_cursor} does not match '
            f'checkpoint cursor ({env_steps}, {optim_steps})'
        )

    loop_state = state_dicts.get('loop_state')
    if not isinstance(loop_state, Mapping):
        return 'missing loop_state'
    try:
        next_cycle_sample = int(loop_state['next_cycle_sample'])
        cycle_env_steps = int(loop_state['cycle_env_steps'])
    except (KeyError, TypeError, ValueError):
        return 'invalid loop_state cursor'
    if next_cycle_sample != num_samples_per_cycle:
        return (
            f'next_cycle_sample={next_cycle_sample}, expected complete cycle '
            f'value {num_samples_per_cycle}'
        )
    if cycle_env_steps != env_steps:
        return f'cycle_env_steps={cycle_env_steps}, expected {env_steps}'

    for key in ('agent', 'losses', 'rng', 'replay'):
        if key not in state_dicts:
            return f'missing {key}'
    return None


def load_latest_committed_online_checkpoint(
        output_dir: str, *, num_samples_per_cycle: int,
) -> Optional[Tuple[str, int, int, dict]]:
    """Load the newest checkpoint known to be at a committed cycle boundary."""
    candidates = []
    for path in glob.glob(os.path.join(glob.escape(output_dir), 'checkpoint_env*_opt*.pth')):
        key = online_checkpoint_key(path)
        if key is not None:
            candidates.append((*key, path))
    candidates.sort(reverse=True)

    for env_steps, optim_steps, path in candidates:
        try:
            try:
                state_dicts = torch.load(
                    path, map_location='cpu', weights_only=False, mmap=True,
                )
            except TypeError:
                state_dicts = torch.load(path, map_location='cpu', weights_only=False)
        except Exception as exc:
            logging.warning('Skipping unreadable online checkpoint %s: %s', path, exc)
            continue
        issue = committed_online_checkpoint_issue(
            state_dicts,
            expected_env_steps=env_steps,
            expected_optim_steps=optim_steps,
            num_samples_per_cycle=num_samples_per_cycle,
        )
        if issue is None:
            return path, env_steps, optim_steps, state_dicts
        logging.warning('Skipping uncommitted online checkpoint %s: %s', path, issue)
        del state_dicts

    if candidates:
        logging.warning(
            'Found %d online checkpoint(s) in %s, but none is a complete '
            'checkpoint-level resume point; restarting this task from step 0',
            len(candidates), output_dir,
        )
    return None


@attrs.frozen
class OnlineResumePlan:
    requested_total_env_steps: int
    start_env_steps: int
    env_steps_offset: int
    local_total_env_steps: int
    remaining_env_steps: int


def resolve_online_resume_plan(
        *, requested_total_env_steps: int, start_env_steps: int,
        loaded_replay: bool, replay_env_steps_offset: int,
        replay_env_steps: int) -> OnlineResumePlan:
    if requested_total_env_steps < start_env_steps:
        raise ValueError(
            f'Requested total_env_steps={requested_total_env_steps} is below '
            f'the latest checkpoint at env_steps={start_env_steps}'
        )
    if loaded_replay:
        replay_cursor = replay_env_steps_offset + replay_env_steps
        if replay_cursor != start_env_steps:
            raise RuntimeError(
                f'Restored replay cursor {replay_cursor} does not match '
                f'checkpoint env_steps={start_env_steps}'
            )
        env_steps_offset = replay_env_steps_offset
        local_total_env_steps = requested_total_env_steps - env_steps_offset
        remaining_env_steps = local_total_env_steps - replay_env_steps
    else:
        env_steps_offset = start_env_steps
        local_total_env_steps = requested_total_env_steps - start_env_steps
        remaining_env_steps = local_total_env_steps
    if remaining_env_steps < 0:
        raise RuntimeError(f'Invalid negative remaining_env_steps={remaining_env_steps}')
    return OnlineResumePlan(
        requested_total_env_steps=requested_total_env_steps,
        start_env_steps=start_env_steps,
        env_steps_offset=env_steps_offset,
        local_total_env_steps=local_total_env_steps,
        remaining_env_steps=remaining_env_steps,
    )


@utils.singleton
@attrs.define(kw_only=True)
class Conf(BaseConf):
    output_base_dir: str = attrs.field(default=os.path.join(os.path.dirname(__file__), 'results'))

    resume_if_possible: bool = False

    env: quasimetric_rl.data.online.ReplayBuffer.Conf = quasimetric_rl.data.online.ReplayBuffer.Conf()

    batch_size: int = attrs.field(default=256, validator=attrs.validators.gt(0))
    interaction: InteractionConf = InteractionConf()

    log_steps: int = attrs.field(default=250, validator=attrs.validators.gt(0))
    eval_steps: Optional[int] = attrs.field(
        default=None, validator=attrs.validators.optional(attrs.validators.gt(0)),
    )
    save_steps: int = attrs.field(default=20000, validator=attrs.validators.gt(0))
    keep_only_latest_checkpoint: bool = True
    save_replay_buffer: bool = True
    save_final_replay_buffer: bool = True
    timing: utils.TimingConf = utils.TimingConf()



cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name='config', node=Conf())
register_model_size_presets(cs)


def resolve_split_encoder_goal_dims(cfg: Conf) -> None:
    """Fill SplitEncoder goal dimensions from the selected environment."""
    encoder = cfg.agent.quasimetric_critic.model.encoder
    if encoder.kind != 'split' or encoder.goal_dims is not None:
        return
    key = (cfg.env.kind, cfg.env.name)
    try:
        encoder.goal_dims = GOAL_SET_DIMS_REGISTRY[key]
    except KeyError as exc:
        raise ValueError(
            f'No default SplitEncoder goal dimensions for {key!r}; set '
            'agent.quasimetric_critic.model.encoder.goal_dims explicitly.'
        ) from exc


def summarize_evaluation(
        result, *, split: str, seed: int, env_steps: int, optim_steps: int,
        episode_length: int) -> dict:
    success_count = int(result.is_success.sum(dtype=torch.int64).item())
    num_episodes = int(result.is_success.numel())
    hitting_time = torch.where(
        result.hitting_time < 0,
        episode_length + 1,
        result.hitting_time + 1,
    )
    return {
        'split': split,
        'seed': int(seed),
        'seed_end': int(seed) + num_episodes - 1,
        'num_episodes': num_episodes,
        'env_steps': int(env_steps),
        'optim_steps': int(optim_steps),
        'success_count': success_count,
        'succ_rate': success_count / num_episodes,
        'hitting_time': hitting_time.to(torch.float64).mean().item(),
        'epi_return': result.episode_return.to(torch.float64).mean().item(),
    }


def validation_sort_key(summary: Mapping[str, Any]) -> tuple:
    """Higher is better: success, speed, return, then later checkpoint."""
    return (
        int(summary['success_count']),
        -float(summary['hitting_time']),
        float(summary['epi_return']),
        int(summary['env_steps']),
        int(summary['optim_steps']),
    )


def select_best_validation(summaries: Sequence[Mapping[str, Any]]) -> dict:
    candidates = [
        summary for summary in summaries
        if summary.get('split') == 'validation'
        and summary.get('checkpoint')
        and summary.get('agent_checkpoint')
    ]
    if not candidates:
        raise RuntimeError('No checkpoint-bound validation results are available')
    return dict(max(candidates, key=validation_sort_key))


@pdb_if_DEBUG
@hydra.main(version_base=None, config_name="config")
def train(dict_cfg: DictConfig):
    cfg: Conf = Conf.from_DictConfig(dict_cfg)
    resolve_split_encoder_goal_dims(cfg)
    if cfg.eval_steps is not None and cfg.eval_steps != cfg.save_steps:
        raise ValueError(
            'Independent online evaluation is disabled: eval_steps must be null '
            'or equal save_steps so every validation result is checkpoint-bound.'
        )
    if cfg.agent.training_schedule != 'joint':
        raise RuntimeError(
            "agent.training_schedule is currently implemented for offline training only. "
            "Online phased training needs an explicit replay/collection policy."
        )
    writer = cfg.setup_for_experiment()  # checking & setup logging
    completion_marker_existed = os.path.exists(cfg.completion_file)
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
            candidate_seed=cfg.seed,
        )
    requested_total_env_steps = trainer.total_env_steps

    val_results: List[dict] = []
    val_summaries: List[dict] = []
    step_counter = StepsCounter(
        alert_intervals=dict(
            log=cfg.log_steps,
            save=cfg.save_steps,
        ),
    )
    next_cycle_sample = 0
    cycle_env_steps = 0

    def checkpoint_desc(env_steps, optim_steps, suffix=None):
        desc = f"env{env_steps:08d}_opt{optim_steps:08d}"
        if suffix is not None:
            desc += f'_{suffix}'
        return desc

    def save(env_steps, optim_steps, *, suffix=None, extra=None):
        with profiler.record('checkpoint/save'):
            desc = checkpoint_desc(env_steps, optim_steps, suffix)
            utils.mkdir(cfg.output_dir)
            fullpath = os.path.join(cfg.output_dir, f'checkpoint_{desc}.pth')
            state_dicts = dict(
                env_steps=env_steps,
                optim_steps=optim_steps,
                checkpoint_kind=ONLINE_CHECKPOINT_KIND_COMMITTED,
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
                target_total_env_steps=requested_total_env_steps,
                **(extra or {}),
            )
            include_replay = cfg.save_replay_buffer and (suffix != 'final' or cfg.save_final_replay_buffer)
            if include_replay:
                state_dicts['replay'] = trainer.replay.state_dict()
            utils.atomic_torch_save(state_dicts, fullpath)
            if cfg.keep_only_latest_checkpoint:
                utils.prune_checkpoints(cfg.output_dir, fullpath)
            relpath = os.path.join('.', os.path.relpath(fullpath, os.path.dirname(__file__)))
            logging.info(f"Checkpointed to {relpath}")
            return fullpath

    def load(ckpt, state_dicts=None) -> dict:
        if state_dicts is None:
            state_dicts = torch.load(ckpt, map_location='cpu', weights_only=False)
        trainer.agent.load_state_dict(state_dicts['agent'])
        trainer.losses.load_state_dict(state_dicts['losses'])
        trainer.set_scheduler_horizon(trainer.scheduler_horizon)
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

    loaded_replay = False
    replay_env_steps_offset = 0
    resume_checkpoint = None
    if cfg.resume_if_possible:
        resume_checkpoint = load_latest_committed_online_checkpoint(
            cfg.output_dir,
            num_samples_per_cycle=trainer.num_samples_per_cycle,
        )
    if resume_checkpoint is not None:
        ckpt, start_env_steps, start_optim_steps, checkpoint_state = resume_checkpoint
        logging.info(f'Load from committed checkpoint: {ckpt}')
        loaded_state_dicts = load(ckpt, checkpoint_state)
        loaded_replay = 'replay' in loaded_state_dicts
        loop_state = loaded_state_dicts.get('loop_state', {})
        if loaded_replay:
            next_cycle_sample = int(loop_state.get('next_cycle_sample', trainer.num_samples_per_cycle))
            replay_env_steps_offset = int(loop_state.get('replay_env_steps_offset', 0))
            cycle_env_steps = int(loop_state.get('cycle_env_steps', start_env_steps))
        logging.info(f'Fast forward to env_steps={start_env_steps} optim_steps={start_optim_steps}')
    else:
        start_env_steps, start_optim_steps = 0, 0

    if completion_marker_existed and start_env_steps == 0:
        raise RuntimeError(
            'Output is marked complete but has no resumable full checkpoint'
        )
    if (completion_marker_existed
            and requested_total_env_steps > start_env_steps
            and not loaded_replay):
        raise RuntimeError(
            'Extending a completed online run requires a checkpoint with its '
            'replay buffer; refusing an inexact empty-replay continuation'
        )
    resume_plan = resolve_online_resume_plan(
        requested_total_env_steps=requested_total_env_steps,
        start_env_steps=start_env_steps,
        loaded_replay=loaded_replay,
        replay_env_steps_offset=replay_env_steps_offset,
        replay_env_steps=(trainer.replay.num_transitions_realized if loaded_replay else 0),
    )
    env_steps_offset = resume_plan.env_steps_offset
    trainer.total_env_steps = resume_plan.local_total_env_steps
    if loaded_replay and env_steps_offset + trainer.replay.num_transitions_realized > cycle_env_steps:
        next_cycle_sample = 0
    if start_env_steps > 0 and not loaded_replay:
        logging.info(
            f'Resume without replay at env_steps={start_env_steps}; '
            f'collecting remaining_env_steps={resume_plan.remaining_env_steps}'
        )
    elif loaded_replay:
        logging.info(
            f'Resume with restored replay buffer at replay_env_steps_offset={env_steps_offset}; '
            f'continuing for remaining_env_steps={resume_plan.remaining_env_steps} '
            f'to requested_total_env_steps={requested_total_env_steps}'
        )
    if completion_marker_existed and requested_total_env_steps > start_env_steps:
        os.unlink(cfg.completion_file)
        logging.info(
            f'Removed stale completion marker before extending training from '
            f'{start_env_steps} to {requested_total_env_steps} env steps'
        )

    def write_json(path, value):
        tmppath = path + '.tmp'
        with open(tmppath, 'w') as f:
            json.dump(value, f, indent=2, sort_keys=True)
            f.write('\n')
        os.replace(tmppath, path)

    def eval(
            env_steps, optim_steps, *, split, seed, num_episodes,
            summary_extra=None):
        with profiler.record('eval/total'):
            result = trainer.evaluate(num_episodes=num_episodes, seed=seed)
        summary = summarize_evaluation(
            result,
            split=split,
            seed=seed,
            env_steps=env_steps,
            optim_steps=optim_steps,
            episode_length=trainer.replay.episode_length,
        )
        summary.update(summary_extra or {})
        raw_result = dict(
            env_steps=env_steps,
            optim_steps=optim_steps,
            split=split,
            seed=seed,
            result=attrs.asdict(result),
        )
        if split == 'validation':
            val_results.clear()
            val_results.append(raw_result)
        for key in ('succ_rate', 'hitting_time', 'epi_return'):
            writer.add_scalar(f'{split}/{key}', summary[key], env_steps)
        log_name = 'eval.log' if split == 'validation' else 'test.log'
        with open(os.path.join(cfg.output_dir, log_name), 'a') as f:
            print(json.dumps(summary, sort_keys=True), file=f)
        logging.info(
            f"{split.upper()}: " +
            "  ".join([
                f"env_steps={env_steps}",
                f"optim_steps={optim_steps}",
                f"succ_rate={summary['succ_rate']:.2%}",
                f"hitting_time={summary['hitting_time']:.2f}",
                f"epi_return={summary['epi_return']:.4f}",
            ])
        )
        return result, summary

    def validate_and_save(env_steps, optim_steps):
        desc = checkpoint_desc(env_steps, optim_steps)
        summary_extra = {
            'checkpoint': f'checkpoint_{desc}.pth',
            'agent_checkpoint': f'agent_checkpoint_{desc}.pth',
        }
        result, summary = eval(
            env_steps,
            optim_steps,
            split='validation',
            seed=trainer.validation_seed,
            num_episodes=trainer.num_eval_episodes,
            summary_extra=summary_extra,
        )
        val_summaries.append(summary)
        utils.atomic_torch_save(
            {
                'env_steps': env_steps,
                'optim_steps': optim_steps,
                'agent': trainer.agent.state_dict(),
                'validation_result': attrs.asdict(result),
                'validation_summary': summary,
            },
            os.path.join(cfg.output_dir, summary['agent_checkpoint']),
        )
        return save(
            env_steps,
            optim_steps,
            extra={'validation_summary': summary},
        )

    def evaluate_selected_best(final_env_steps, final_optim_steps):
        best = select_best_validation(val_summaries)
        agent_path = os.path.join(cfg.output_dir, best['agent_checkpoint'])
        state = torch.load(agent_path, map_location='cpu', weights_only=False)
        trainer.agent.load_state_dict(state['agent'])
        test_summary_extra = {
            'selected_checkpoint': best['checkpoint'],
            'selected_agent_checkpoint': best['agent_checkpoint'],
            'completed_training_env_steps': final_env_steps,
            'completed_training_optim_steps': final_optim_steps,
        }
        test_result, test_summary = eval(
            best['env_steps'],
            best['optim_steps'],
            split='test',
            seed=trainer.test_seed,
            num_episodes=trainer.num_test_episodes,
            summary_extra=test_summary_extra,
        )
        selected_path = os.path.join(cfg.output_dir, 'selected_best_agent.pth')
        utils.atomic_torch_save(
            {
                'agent': trainer.agent.state_dict(),
                'validation_summary': best,
                'test_result': attrs.asdict(test_result),
                'test_summary': test_summary,
            },
            selected_path,
        )
        selection = {
            'selection_order': [
                'success_count:max',
                'hitting_time:min',
                'epi_return:max',
                'env_steps:max',
                'optim_steps:max',
            ],
            'validation': best,
            'test': test_summary,
            'selected_model': os.path.basename(selected_path),
        }
        write_json(os.path.join(cfg.output_dir, 'best_checkpoint.json'), selection)
        logging.info(
            'SELECTED BEST: checkpoint=%s validation_success=%.2f%% '
            'validation_hitting_time=%.2f test_success=%.2f%%',
            best['checkpoint'],
            100 * best['succ_rate'],
            best['hitting_time'],
            100 * test_summary['succ_rate'],
        )
        return selection

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

                if step_counter.alerts.save:
                    validate_and_save(env_steps, optim_steps)

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
        final_is_validated = any(
            summary.get('env_steps') == final_env_steps
            and summary.get('optim_steps') == optim_steps
            and summary.get('checkpoint')
            for summary in val_summaries
        )
        if not final_is_validated:
            validate_and_save(final_env_steps, optim_steps)
        evaluate_selected_best(final_env_steps, optim_steps)
        profiler.log(writer=writer, step=final_env_steps, step_name='env_steps', extra=dict(optim_steps=optim_steps))
        open(cfg.completion_file, 'a').close()
    except TrainingInterrupted as exc:
        env_steps = current_env_steps()
        logging.warning(
            f"Interrupted: {exc} at env_steps={env_steps} optim_steps={optim_steps}. "
            "Discarding progress after the latest committed checkpoint."
        )
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

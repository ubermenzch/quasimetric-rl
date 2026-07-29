from __future__ import annotations
from typing import *
from typing_extensions import Self

import time
import attrs
import logging
import contextlib

import gym.spaces
import numpy as np
import torch
import torch.utils.data

from quasimetric_rl import utils
from quasimetric_rl.modules import QRLConf, QRLAgent, QRLLosses, InfoT
from quasimetric_rl.data import BatchData, EpisodeData, MultiEpisodeData
from quasimetric_rl.data.online import ReplayBuffer, FixedLengthEnvWrapper
from quasimetric_rl.utils import TimingProfiler, tqdm


def first_nonzero(arr: torch.Tensor, dim: bool = -1, invalid_val: int = -1):
    mask = (arr != 0)
    return torch.where(mask.any(dim=dim), mask.to(torch.uint8).argmax(dim=dim), invalid_val)


@attrs.define(kw_only=True)
class EvalEpisodeResult:
    timestep_reward: torch.Tensor
    episode_return: torch.Tensor
    timestep_is_success: torch.Tensor
    is_success: torch.Tensor
    hitting_time: torch.Tensor

    @classmethod
    def from_timestep_reward_is_success(cls, timestep_reward: torch.Tensor, timestep_is_success: torch.Tensor) -> Self:
        return cls(
            timestep_reward=timestep_reward,
            episode_return=timestep_reward.sum(-1),
            timestep_is_success=timestep_is_success,
            is_success=timestep_is_success.any(dim=-1),
            hitting_time=first_nonzero(timestep_is_success, dim=-1),  # NB this is off by 1
        )


@attrs.define(kw_only=True)
class InteractionConf:
    total_env_steps: int = attrs.field(default=int(1e6), validator=attrs.validators.gt(0))

    # Explicit episode/sample counts override the transition-based defaults.
    num_prefill_episodes: Optional[int] = attrs.field(
        default=None, validator=attrs.validators.optional(attrs.validators.ge(0)),
    )
    num_samples_per_cycle: Optional[int] = attrs.field(
        default=None, validator=attrs.validators.optional(attrs.validators.ge(0)),
    )
    num_rollouts_per_cycle: Optional[int] = attrs.field(
        default=None, validator=attrs.validators.optional(attrs.validators.ge(0)),
    )
    num_eval_episodes: Optional[int] = attrs.field(
        default=1000, validator=attrs.validators.optional(attrs.validators.gt(0)),
    )
    num_test_episodes: int = attrs.field(default=1000, validator=attrs.validators.gt(0))
    validation_seed: int = 1_000_000
    test_seed: int = 2_000_000

    prefill_env_steps: Optional[int] = attrs.field(
        default=None, validator=attrs.validators.optional(attrs.validators.gt(0)),
    )
    random_policy_env_steps: Optional[int] = attrs.field(
        default=None, validator=attrs.validators.optional(attrs.validators.ge(0)),
    )
    env_steps_per_cycle: int = attrs.field(default=500, validator=attrs.validators.gt(0))
    eval_env_steps: int = attrs.field(default=2_500, validator=attrs.validators.gt(0))
    optim_steps_per_env_step: float = attrs.field(
        default=1.0, validator=attrs.validators.gt(0),
    )

    exploration_eps: float = attrs.field(default=0.3, validator=attrs.validators.ge(0))


@attrs.frozen
class ResolvedInteractionSchedule:
    num_prefill_episodes: int
    num_samples_per_cycle: int
    num_rollouts_per_cycle: int
    num_eval_episodes: int


def resolve_interaction_schedule(
        conf: InteractionConf, episode_length: int, *,
        default_prefill_env_steps: int = 10_000) -> ResolvedInteractionSchedule:
    if episode_length <= 0:
        raise ValueError(f'episode_length must be positive, got {episode_length}')

    num_prefill_episodes = conf.num_prefill_episodes
    if num_prefill_episodes is None:
        prefill_env_steps = (
            default_prefill_env_steps
            if conf.prefill_env_steps is None
            else conf.prefill_env_steps
        )
        num_prefill_episodes = int(np.ceil(prefill_env_steps / episode_length))

    num_rollouts_per_cycle = conf.num_rollouts_per_cycle
    if num_rollouts_per_cycle is None:
        num_rollouts_per_cycle = max(1, round(conf.env_steps_per_cycle / episode_length))

    num_samples_per_cycle = conf.num_samples_per_cycle
    if num_samples_per_cycle is None:
        cycle_env_steps = num_rollouts_per_cycle * episode_length
        num_samples_per_cycle = max(
            1, round(cycle_env_steps * conf.optim_steps_per_env_step),
        )

    num_eval_episodes = conf.num_eval_episodes
    if num_eval_episodes is None:
        num_eval_episodes = max(1, round(conf.eval_env_steps / episode_length))

    return ResolvedInteractionSchedule(
        num_prefill_episodes=num_prefill_episodes,
        num_samples_per_cycle=num_samples_per_cycle,
        num_rollouts_per_cycle=num_rollouts_per_cycle,
        num_eval_episodes=num_eval_episodes,
    )


def add_gaussian_exploration(
        action: torch.Tensor, action_space: gym.spaces.Space,
        std_fraction: float) -> torch.Tensor:
    """Add Gaussian noise scaled to the environment's finite action bounds."""
    if std_fraction == 0:
        return action
    if not isinstance(action_space, gym.spaces.Box):
        raise TypeError('Online Gaussian exploration requires a Box action space')
    low = torch.as_tensor(action_space.low, dtype=action.dtype, device=action.device)
    high = torch.as_tensor(action_space.high, dtype=action.dtype, device=action.device)
    if not torch.isfinite(low).all() or not torch.isfinite(high).all():
        raise ValueError('Online Gaussian exploration requires finite action bounds')
    scale = (high - low) / 2
    noisy_action = action + torch.randn_like(action) * std_fraction * scale
    return torch.maximum(torch.minimum(noisy_action, high), low)


class Trainer(object):
    agent: QRLAgent
    losses: QRLLosses
    device: torch.device
    replay: ReplayBuffer
    batch_size: int

    total_env_steps: int

    num_prefill_episodes: int
    num_samples_per_cycle: int
    num_rollouts_per_cycle: int
    num_eval_episodes: int
    num_test_episodes: int
    validation_seed: int
    test_seed: int
    exploration_eps: float
    random_policy_env_steps: int
    profiler: Optional[TimingProfiler]
    replay_sampling: str

    def set_scheduler_horizon(self, total_optim_steps: int) -> None:
        """Keep restored scheduler progress but use the newly requested horizon."""
        if hasattr(self.losses, 'set_scheduler_horizon'):
            self.losses.set_scheduler_horizon(total_optim_steps)
            return
        schedulers = []
        if self.losses.actor_loss is not None:
            schedulers.extend((
                self.losses.actor_loss.actor_sched,
                self.losses.actor_loss.entropy_weight_sched,
            ))
        if (self.losses.goal_set_distance_loss is not None
                and self.losses.goal_set_distance_loss.sched is not None):
            schedulers.append(self.losses.goal_set_distance_loss.sched)
        for critic_loss in self.losses.critic_losses:
            schedulers.extend((
                critic_loss.critic_sched,
                critic_loss.lagrange_mult_sched,
            ))
            if critic_loss.latent_dynamics_sched is not None:
                schedulers.append(critic_loss.latent_dynamics_sched)
        for scheduler in schedulers:
            scheduler.T_max = total_optim_steps

    def get_total_optim_steps(self, total_env_steps: int):
        current_env_steps = self.replay.num_episodes_realized * self.replay.episode_length
        prefill_env_steps = min(
            self.num_prefill_episodes * self.replay.episode_length,
            total_env_steps,
        )
        env_steps_after_prefill = max(current_env_steps, prefill_env_steps)
        remaining_env_steps = max(total_env_steps - env_steps_after_prefill, 0)

        # Training performs one optimization cycle immediately after prefill,
        # then one more cycle after each group of policy rollouts.
        num_cycles = 1
        if remaining_env_steps != 0:
            assert self.num_rollouts_per_cycle > 0
            env_steps_per_cycle = self.num_rollouts_per_cycle * self.replay.episode_length
            num_cycles += int(np.ceil(remaining_env_steps / env_steps_per_cycle))

        return self.num_samples_per_cycle * num_cycles

    def __init__(self, *, agent_conf: QRLConf,
                 device: torch.device,
                 replay: ReplayBuffer,
                 batch_size: int,
                 interaction_conf: InteractionConf,
                 profiler: Optional[TimingProfiler] = None,
                 eval_seed: int = 416923159,
                 candidate_seed: int = 0):

        self.device = device
        self.replay = replay
        self.eval_seed = eval_seed
        self.batch_size = batch_size
        self.algorithm = agent_conf.algorithm

        self.exploration_eps = interaction_conf.exploration_eps
        self.total_env_steps = interaction_conf.total_env_steps

        if self.algorithm == 'td_infonce':
            baseline_conf = agent_conf.baselines.td_infonce
            default_prefill_env_steps = baseline_conf.min_replay_size
            default_random_policy_env_steps = default_prefill_env_steps
            self.replay_sample_max_transitions = baseline_conf.max_replay_size
            self.replay_sample_max_episodes = None
        elif self.algorithm == 'crl':
            baseline_conf = agent_conf.baselines.crl
            default_prefill_env_steps = baseline_conf.min_replay_size
            default_random_policy_env_steps = 0
            self.replay_sample_max_transitions = baseline_conf.max_replay_size
            self.replay_sample_max_episodes = None
        elif self.algorithm in ('gcbc', 'gcsl'):
            baseline_conf = agent_conf.baselines.gcbc
            default_prefill_env_steps = baseline_conf.start_policy_timesteps
            default_random_policy_env_steps = baseline_conf.explore_timesteps
            self.gcbc_validation_fraction = baseline_conf.validation_fraction
            self.replay_sample_max_transitions = None
            self.replay_sample_max_episodes = baseline_conf.replay_capacity_trajectories
        elif self.algorithm == 'c_learning':
            baseline_conf = agent_conf.baselines.c_learning
            default_prefill_env_steps = baseline_conf.initial_collect_steps
            default_random_policy_env_steps = default_prefill_env_steps
            self.replay_sample_max_transitions = baseline_conf.replay_buffer_capacity
            self.replay_sample_max_episodes = None
        else:
            baseline_conf = None
            default_prefill_env_steps = 10_000
            default_random_policy_env_steps = default_prefill_env_steps
            self.replay_sample_max_transitions = None
            self.replay_sample_max_episodes = None
        if self.algorithm not in ('gcbc', 'gcsl'):
            self.gcbc_validation_fraction = 0.0

        if baseline_conf is not None and batch_size != baseline_conf.batch_size:
            raise ValueError(
                f'{self.algorithm} reference default requires batch_size='
                f'{baseline_conf.batch_size}, got {batch_size}'
            )

        schedule_conf = interaction_conf
        if baseline_conf is not None:
            schedule_conf = attrs.evolve(
                schedule_conf,
                optim_steps_per_env_step=baseline_conf.updates_per_env_step,
            )
        if (
                self.algorithm in ('gcbc', 'gcsl')
                and interaction_conf.num_prefill_episodes is None
                and interaction_conf.prefill_env_steps is None):
            schedule_conf = attrs.evolve(
                schedule_conf,
                num_prefill_episodes=(
                    default_prefill_env_steps // replay.episode_length + 1
                ),
                num_rollouts_per_cycle=(
                    1 if interaction_conf.num_rollouts_per_cycle is None
                    else interaction_conf.num_rollouts_per_cycle
                ),
                num_samples_per_cycle=(
                    replay.episode_length
                    if interaction_conf.num_samples_per_cycle is None
                    else interaction_conf.num_samples_per_cycle
                ),
            )
        schedule = resolve_interaction_schedule(
            schedule_conf, replay.episode_length,
            default_prefill_env_steps=default_prefill_env_steps,
        )
        self.random_policy_env_steps = (
            default_random_policy_env_steps
            if interaction_conf.random_policy_env_steps is None
            else interaction_conf.random_policy_env_steps
        )
        self.num_samples_per_cycle = schedule.num_samples_per_cycle
        self.num_rollouts_per_cycle = schedule.num_rollouts_per_cycle
        self.num_eval_episodes = schedule.num_eval_episodes
        self.num_test_episodes = interaction_conf.num_test_episodes
        self.validation_seed = interaction_conf.validation_seed
        self.test_seed = interaction_conf.test_seed
        validation_end = self.validation_seed + self.num_eval_episodes - 1
        test_end = self.test_seed + self.num_test_episodes - 1
        if max(self.validation_seed, self.test_seed) <= min(validation_end, test_end):
            raise ValueError(
                'validation and test episode seed ranges must be disjoint, got '
                f'{self.validation_seed}..{validation_end} and '
                f'{self.test_seed}..{test_end}'
            )
        self.num_prefill_episodes = schedule.num_prefill_episodes
        logging.info(
            'Resolved interaction schedule for episode_length=%d: '
            'prefill_episodes=%d, rollouts_per_cycle=%d, '
            'samples_per_cycle=%d, random_policy_env_steps=%d, eval_episodes=%d',
            replay.episode_length,
            self.num_prefill_episodes,
            self.num_rollouts_per_cycle,
            self.num_samples_per_cycle,
            self.random_policy_env_steps,
            self.num_eval_episodes,
        )
        self.profiler = profiler
        self.replay_sampling = (
            'uniform_future_pair'
            if agent_conf.algorithm in ('gcbc', 'gcsl')
            else 'geometric_future'
        )
        if agent_conf.algorithm == 'crl':
            self.replay.future_observation_discount = (
                agent_conf.baselines.crl.discount
            )
        self.replay.transition_history_length = max(
            self.replay.transition_history_length,
            agent_conf.required_transition_history_length,
        )

        total_optim_steps = self.get_total_optim_steps(interaction_conf.total_env_steps)
        self.agent, self.losses = agent_conf.make(
            env_spec=replay.env_spec,
            total_optim_steps=total_optim_steps,
            profiler=profiler,
            goal_set_dims=(
                replay.goal_set_dims
                if agent_conf.algorithm != 'qrl'
                else (
                    replay.goal_set_dims
                    if agent_conf.goal_set_distance.enabled
                    and agent_conf.goal_set_distance.losses.goal_dims is None
                    else None
                )
            ),
        )
        self.agent.to(device)
        self.losses.to(device)
        self.scheduler_horizon = total_optim_steps
        if getattr(self.losses, 'goal_set_distance_loss', None) is not None:
            self.losses.goal_set_distance_loss.set_observation_bounds_provider(replay.observation_bounds)
            self.losses.goal_set_distance_loss.set_candidate_state_provider(
                replay.sample_goal_conditioned_observations
            )
            self.losses.goal_set_distance_loss.set_candidate_seed(candidate_seed)

        logging.info('Agent:\n\t' + str(self.agent).replace('\n', '\n\t') + '\n\n')
        logging.info('Losses:\n\t' + str(self.losses).replace('\n', '\n\t') + '\n\n')

    def make_collect_env(self) -> FixedLengthEnvWrapper:
        return self.replay.create_env()

    def make_evaluate_env(self, seed: Optional[int] = None) -> FixedLengthEnvWrapper:
        env = self.replay.create_env()
        # a hack to expose more signal from some envs :)
        if hasattr(env, 'reward_mode') and len(self.replay.env_spec.observation_shape) == 1:
            env.unwrapped.reward_mode = 'dense'
        env.seed(self.eval_seed if seed is None else seed)
        return env

    def sample(self) -> BatchData:
        with self._record('data/sample_replay'):
            if self.replay_sampling == 'uniform_future_pair':
                batch = self.replay.sample_uniform_future_pairs(
                    self.batch_size, training_only=True,
                    max_episodes=self.replay_sample_max_episodes,
                )
            else:
                batch = self.replay.sample(
                    self.batch_size,
                    max_transitions=self.replay_sample_max_transitions,
                )
        with self._record('data/to_device'):
            return batch.to(self.device)

    def _record(self, name: str):
        if self.profiler is None:
            return contextlib.nullcontext()
        return self.profiler.record(name)

    def _store_rollout(self, rollout: EpisodeData) -> None:
        training = not (
            self.algorithm in ('gcbc', 'gcsl')
            and np.random.rand() < self.gcbc_validation_fraction
        )
        self.replay.add_rollout(rollout, training=training)

    def collect_random_rollout(self, *, store: bool = True, env: Optional[FixedLengthEnvWrapper] = None) -> EpisodeData:
        with self._record('env/random_rollout'):
            if self.algorithm in ('gcbc', 'gcsl'):
                random_actor = lambda _obs, _goal, _space: (
                    self.agent.actor.sample_uniform_action().cpu().numpy()
                )
            else:
                random_actor = lambda _obs, _goal, space: space.sample()
            rollout = self.replay.collect_rollout(
                random_actor,
                env=env,
            )
        if store:
            with self._record('env/add_rollout'):
                self._store_rollout(rollout)
        return rollout

    def collect_rollout(self, *, eval: bool = False, store: bool = True,
                        env: Optional[FixedLengthEnvWrapper] = None) -> EpisodeData:
        assert self.agent.actor is not None

        def actor(obs: torch.Tensor, goal: torch.Tensor, space: gym.spaces.Space):
            with self._record('env/actor_to_device'):
                obs = obs[None].to(self.device)
                goal = goal[None].to(self.device)
            with self._record('env/actor_forward'):
                adistn = self.agent.act(obs, goal)
            if eval or self.algorithm in ('gcbc', 'gcsl'):
                with self._record('env/action_to_cpu'):
                    a = adistn.mean.cpu().numpy()[0]
            else:
                with self._record('env/action_sample'):
                    a_t = adistn.sample()
                    a_t = add_gaussian_exploration(
                        a_t, space, self.exploration_eps,
                    )
                with self._record('env/action_to_cpu'):
                    a = a_t.cpu().numpy()[0]
            return a

        with torch.no_grad(), self.agent.mode(False), \
                self._record('env/policy_rollout_eval' if eval else 'env/policy_rollout_train'):
            rollout = self.replay.collect_rollout(actor, env=env)
        if store:
            with self._record('env/add_rollout'):
                self._store_rollout(rollout)
        return rollout

    def evaluate(
            self, *, num_episodes: Optional[int] = None,
            seed: Optional[int] = None) -> EvalEpisodeResult:
        num_episodes = self.num_eval_episodes if num_episodes is None else num_episodes
        if num_episodes <= 0:
            raise ValueError(f'num_episodes must be positive, got {num_episodes}')

        rng_state = utils.rng_state_dict()
        env = None
        try:
            episode_seed_start = self.eval_seed if seed is None else int(seed)
            with self._record('eval/make_env'):
                env = self.make_evaluate_env(episode_seed_start)
            rollouts = []
            with self._record('eval/rollouts'):
                for episode_idx in tqdm(range(num_episodes), desc='evaluate'):
                    env.seed(episode_seed_start + episode_idx)
                    rollouts.append(self.collect_rollout(eval=True, store=False, env=env))
            with self._record('eval/aggregate'):
                mrollouts = MultiEpisodeData.cat(rollouts)
                return EvalEpisodeResult.from_timestep_reward_is_success(
                    mrollouts.rewards.reshape(
                        num_episodes, env.episode_length,
                    ),
                    mrollouts.transition_infos['is_success'].reshape(
                        num_episodes, env.episode_length,
                    ),
                )
        finally:
            if env is not None:
                env.close()
            utils.load_rng_state(rng_state)

    def iter_training_data(self, *, start_cycle_sample: int = 0) -> Iterator[Tuple[int, bool, BatchData, InfoT]]:
        r"""
        Yield data to train on for each optimization iteration.

        yield (
            env steps,
            whether this is last yield before collecting new env steps,
            data,
            info,
        )
        """
        if not 0 <= start_cycle_sample <= self.num_samples_per_cycle:
            raise ValueError(
                f"start_cycle_sample={start_cycle_sample} must be in "
                f"[0, {self.num_samples_per_cycle}]"
            )

        def yield_data(first_cycle_sample: int = 0):
            num_transitions = self.replay.num_transitions_realized
            for icyc in tqdm(range(first_cycle_sample, self.num_samples_per_cycle), desc=f"{num_transitions} env steps, train batches"):
                data_t0 = time.time()
                data = self.sample()
                info = dict(
                    data_time=(time.time() - data_t0),
                    cycle_sample=icyc,
                    num_episodes=self.replay.num_episodes_realized,
                    num_regular_transitions=self.replay.num_transitions_realized,
                    num_successes=self.replay.num_successful_transitions,
                    replay_capacity=self.replay.episodes_capacity,
                    reward=data.rewards,
                )

                yield num_transitions, (icyc == self.num_samples_per_cycle - 1), data, info

        total_env_steps = self.total_env_steps

        num_prefill_episodes = min(
            self.num_prefill_episodes,
            total_env_steps // self.replay.episode_length,
        )
        if self.replay.num_episodes_realized < num_prefill_episodes:
            with self._record('env/make_collect_env'):
                env = self.make_collect_env()  # always make fresh collect env before collecting. GCRL envs don't like reusing.
            with self._record('env/prefill'):
                for _ in tqdm(
                        range(self.replay.num_episodes_realized, num_prefill_episodes),
                        desc='prefill'):
                    if self.algorithm == 'crl':
                        self.collect_rollout(env=env)
                    else:
                        self.collect_random_rollout(env=env)
        else:
            logging.info(
                f"Skipping prefill because replay already has "
                f"{self.replay.num_transitions_realized} transitions"
            )
        assert self.replay.num_transitions_realized <= total_env_steps

        yield from yield_data(start_cycle_sample)

        while self.replay.num_transitions_realized < total_env_steps:
            with self._record('env/make_collect_env'):
                env = self.make_collect_env()
            with self._record('env/rollout_cycle'):
                for _ in range(self.num_rollouts_per_cycle):
                    if (
                            self.replay.num_transitions_realized
                            < self.random_policy_env_steps):
                        self.collect_random_rollout(env=env)
                    else:
                        self.collect_rollout(env=env)

                    if self.replay.num_transitions_realized >= total_env_steps:
                        break

            yield from yield_data()

    def train_step(self, data: BatchData, *, optimize: bool = True, phase: str = 'all') -> InfoT:
        with self._record('train/losses_total'):
            return self.losses(self.agent, data, optimize=optimize, phase=phase).info

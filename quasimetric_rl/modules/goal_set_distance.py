from typing import *

import contextlib

import attrs
import torch
import torch.nn.functional as F

from ..data import BatchData, EnvSpec
from .optim import AdamWSpec, OptimWrapper
from .quasimetric_critic import CriticBatchInfo
from .utils import LossBase, LossResult, MLP, Module


class GoalSetDistance(Module):
    """One latent goal-set distance head for every QRL critic."""

    @attrs.define(kw_only=True)
    class Conf:
        arch: Tuple[int, ...] = (512, 512)
        nonnegative_output: bool = True

        def make(self, *, latent_size: int, num_critics: int) -> 'GoalSetDistance':
            return GoalSetDistance(
                latent_size=latent_size,
                num_critics=num_critics,
                arch=self.arch,
                nonnegative_output=self.nonnegative_output,
            )

    latent_size: int
    num_critics: int
    heads: torch.nn.ModuleList
    nonnegative_output: bool

    def __init__(self, *, latent_size: int, num_critics: int, arch: Tuple[int, ...],
                 nonnegative_output: bool):
        super().__init__()
        self.latent_size = latent_size
        self.num_critics = num_critics
        self.heads = torch.nn.ModuleList([
            MLP(latent_size * 2, 1, hidden_sizes=arch, zero_init_last_fc=True)
            for _ in range(num_critics)
        ])
        self.nonnegative_output = nonnegative_output

    def forward(self, critic_idx: int, state_latent: torch.Tensor,
                padded_goal_latent: torch.Tensor) -> torch.Tensor:
        if not 0 <= critic_idx < self.num_critics:
            raise IndexError(f'critic_idx={critic_idx} outside [0, {self.num_critics})')
        pred = self.heads[critic_idx](torch.cat([state_latent, padded_goal_latent], dim=-1)).squeeze(-1)
        return F.softplus(pred) if self.nonnegative_output else pred

    def extra_repr(self) -> str:
        return f'latent_size={self.latent_size}, num_critics={self.num_critics}'


class GoalSetDistanceLoss(LossBase):
    @attrs.define(kw_only=True)
    class Conf:
        weight: float = attrs.field(default=1.0, validator=attrs.validators.gt(0))
        num_goal_samples: int = attrs.field(default=16, validator=attrs.validators.gt(0))
        # Retained for compatibility with existing task files. The new goal-set
        # construction samples non-goal dimensions from observed state support,
        # so this radius is intentionally not used.
        goal_condition_radius: float = attrs.field(default=0.5, validator=attrs.validators.gt(0))
        # If omitted, use the environment's registered goal-state dimensions.
        # Set explicitly to override the built-in mapping for a custom task.
        goal_dims: Optional[Tuple[int, ...]] = attrs.field(
            default=None,
            converter=lambda dims: None if dims is None else tuple(dims),
        )
        include_goal_state: bool = False
        optim: AdamWSpec.Conf = AdamWSpec.Conf(lr=1e-4)

        def make(self, model: GoalSetDistance, total_optim_steps: int,
                 env_spec: EnvSpec, goal_dims: Optional[Tuple[int, ...]]) -> 'GoalSetDistanceLoss':
            goal_dims = self.goal_dims if self.goal_dims is not None else goal_dims
            if goal_dims is None:
                raise ValueError(
                    'GoalSetDistance requires goal_dims or an environment-provided default'
                )
            return GoalSetDistanceLoss(
                model,
                total_optim_steps=total_optim_steps,
                env_spec=env_spec,
                weight=self.weight,
                num_goal_samples=self.num_goal_samples,
                goal_dims=goal_dims,
                include_goal_state=self.include_goal_state,
                optim_spec=self.optim.make(),
            )

    observation_shape: torch.Size
    weight: float
    num_goal_samples: int
    goal_dims: Tuple[int, ...]
    include_goal_state: bool
    optim: OptimWrapper
    sched: torch.optim.lr_scheduler._LRScheduler
    profiler: Optional[Any]
    observation_bounds_provider: Optional[Callable[..., Tuple[torch.Tensor, torch.Tensor]]]

    def __init__(self, model: GoalSetDistance, *, total_optim_steps: int, env_spec: EnvSpec,
                 weight: float, num_goal_samples: int, goal_dims: Tuple[int, ...],
                 include_goal_state: bool, optim_spec: AdamWSpec):
        super().__init__()
        if len(env_spec.observation_shape) != 1:
            raise RuntimeError(
                'GoalSetDistance requires vector observations. Image goal sets need an '
                'environment-specific state sampler.')
        obs_dim = env_spec.observation_shape[0]
        if not goal_dims or min(goal_dims) < 0 or max(goal_dims) >= obs_dim:
            raise ValueError(f'Invalid goal_dims={goal_dims!r} for observation dim {obs_dim}')
        self.observation_shape = env_spec.observation_shape
        self.weight = weight
        self.num_goal_samples = num_goal_samples
        self.goal_dims = tuple(goal_dims)
        self.include_goal_state = include_goal_state
        self.optim, self.sched = optim_spec.create_optim_scheduler(model.parameters(), total_optim_steps)
        self.profiler = None
        self.observation_bounds_provider = None

    def _record(self, name: str):
        return contextlib.nullcontext() if self.profiler is None else self.profiler.record(name)

    def padded_goal_state(self, goal_states: torch.Tensor) -> torch.Tensor:
        """Keep goal coordinates and zero every non-goal coordinate."""
        padded = torch.zeros_like(goal_states)
        padded[..., list(self.goal_dims)] = goal_states[..., list(self.goal_dims)]
        return padded

    def _flatten_observations(self, observations: torch.Tensor) -> torch.Tensor:
        return observations.reshape(-1, *self.observation_shape)

    def _gather_state_goal_pairs(self, data: BatchData) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = data.observations
        raw_goal = torch.roll(data.next_observations, 1, dims=0)
        return state, raw_goal, self.padded_goal_state(raw_goal)

    def set_observation_bounds_provider(
            self, provider: Callable[..., Tuple[torch.Tensor, torch.Tensor]]) -> None:
        """Set the dataset or replay-buffer source of global observation bounds."""
        self.observation_bounds_provider = provider

    def _sample_goal_condition_states(self, raw_goal_states: torch.Tensor) -> torch.Tensor:
        """Sample non-goal dimensions within dataset or replay-buffer bounds."""
        if self.observation_bounds_provider is None:
            raise RuntimeError('GoalSetDistanceLoss requires an observation-bounds provider')
        num_pairs = raw_goal_states.shape[0]
        low, high = self.observation_bounds_provider(
            device=raw_goal_states.device,
        )
        low, high = low.to(dtype=raw_goal_states.dtype), high.to(dtype=raw_goal_states.dtype)
        shape = (num_pairs, self.num_goal_samples, *self.observation_shape)
        candidates = low + torch.rand(
            shape, device=raw_goal_states.device, dtype=raw_goal_states.dtype,
        ) * (high - low)
        candidates[..., list(self.goal_dims)] = raw_goal_states[:, None, list(self.goal_dims)]
        if self.include_goal_state:
            candidates = torch.cat([raw_goal_states[:, None, :], candidates], dim=1)
        return candidates

    def _forward_loss(self, model: GoalSetDistance, critic_batch_infos: Collection[CriticBatchInfo],
                      data: BatchData) -> LossResult:
        if len(critic_batch_infos) != model.num_critics:
            raise RuntimeError('GoalSetDistance head count must match the critic count')
        with self._record('train/goal_set_distance/gather_pairs'):
            states, raw_goals, padded_goals = self._gather_state_goal_pairs(data)
            candidates = self._sample_goal_condition_states(raw_goals)

        preds, targets = [], []
        with torch.no_grad(), self._record('train/goal_set_distance/critic_targets'):
            flat_candidates = candidates.flatten(0, 1)
            for idx, critic_batch_info in enumerate(critic_batch_infos):
                critic = critic_batch_info.critic
                z_state = critic.encoder(states)
                z_padded_goal = critic.encoder(padded_goals)
                z_candidates = critic.encoder(flat_candidates).unflatten(0, candidates.shape[:2])
                dists = critic.quasimetric_model(
                    z_state[:, None, :].expand_as(z_candidates), z_candidates)
                targets.append(dists.min(dim=1).values)
                preds.append((z_state, z_padded_goal))

        predictions = torch.stack([
            model(idx, z_state, z_padded_goal)
            for idx, (z_state, z_padded_goal) in enumerate(preds)
        ], dim=-1)
        target = torch.stack(targets, dim=-1)
        mse = F.mse_loss(predictions, target)
        loss = self.weight * mse
        return LossResult(
            loss=loss,
            info=dict(
                loss=loss, mse=mse, pred=predictions.mean(), target=target.mean(),
                abs_error=(predictions - target).abs().mean(), weight=self.weight,
                num_goal_samples=self.num_goal_samples,
            ),
        )

    def forward(self, model: GoalSetDistance, critic_batch_infos: Collection[CriticBatchInfo],
                data: BatchData, *, optimize: bool = True) -> LossResult:
        with self.optim.update_context(optimize=optimize):
            result = self._forward_loss(model, critic_batch_infos, data)
            with self._record('train/goal_set_distance/backward'):
                result.loss.backward()
        if optimize:
            self.sched.step()
        return result

    def __call__(self, model: GoalSetDistance, critic_batch_infos: Collection[CriticBatchInfo],
                 data: BatchData, *, optimize: bool = True) -> LossResult:
        return torch.nn.Module.__call__(self, model, critic_batch_infos, data, optimize=optimize)

    def extra_repr(self) -> str:
        return f'num_goal_samples={self.num_goal_samples}, goal_dims={self.goal_dims}'


@attrs.define(kw_only=True)
class GoalSetDistanceConf:
    enabled: bool = False
    model: GoalSetDistance.Conf = GoalSetDistance.Conf()
    losses: GoalSetDistanceLoss.Conf = GoalSetDistanceLoss.Conf()

    def make(self, *, env_spec: EnvSpec, total_optim_steps: int, latent_size: int,
             num_critics: int, goal_dims: Optional[Tuple[int, ...]] = None) -> Tuple[
                 Optional[GoalSetDistance], Optional[GoalSetDistanceLoss]]:
        if not self.enabled:
            return None, None
        model = self.model.make(latent_size=latent_size, num_critics=num_critics)
        return model, self.losses.make(
            model,
            total_optim_steps=total_optim_steps,
            env_spec=env_spec,
            goal_dims=goal_dims,
        )


__all__ = ['GoalSetDistance', 'GoalSetDistanceLoss', 'GoalSetDistanceConf']

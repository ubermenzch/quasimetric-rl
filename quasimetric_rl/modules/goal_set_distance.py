from typing import *

import contextlib

import attrs
import torch
import torch.nn.functional as F

from ..data import BatchData, EnvSpec
from ..data.env_spec.input_encoding import InputEncoding
from .optim import AdamWSpec, OptimWrapper
from .quasimetric_critic import CriticBatchInfo
from .utils import LossBase, LossResult, MLP, Module


class GoalSetDistance(Module):
    @attrs.define(kw_only=True)
    class Conf:
        arch: Tuple[int, ...] = (512, 512)
        nonnegative_output: bool = True

        def make(self, *, env_spec: EnvSpec) -> 'GoalSetDistance':
            return GoalSetDistance(
                env_spec=env_spec,
                arch=self.arch,
                nonnegative_output=self.nonnegative_output,
            )

    observation_shape: torch.Size
    observation_encoding: InputEncoding
    backbone: MLP
    nonnegative_output: bool

    def __init__(self, *, env_spec: EnvSpec, arch: Tuple[int, ...],
                 nonnegative_output: bool):
        super().__init__()
        self.observation_shape = env_spec.observation_shape
        self.observation_encoding = env_spec.make_observation_input()
        self.backbone = MLP(
            self.observation_encoding.output_size * 2,
            1,
            hidden_sizes=arch,
            zero_init_last_fc=True,
        )
        self.nonnegative_output = nonnegative_output

    def forward(self, state: torch.Tensor, goal_state: torch.Tensor) -> torch.Tensor:
        state_goal = torch.stack([state, goal_state], dim=-len(self.observation_shape) - 1)
        model_input = self.observation_encoding(state_goal).flatten(-2, -1)
        pred = self.backbone(model_input).squeeze(-1)
        if self.nonnegative_output:
            pred = F.softplus(pred)
        return pred

    def __call__(self, state: torch.Tensor, goal_state: torch.Tensor) -> torch.Tensor:
        return super().__call__(state, goal_state)

    def extra_repr(self) -> str:
        return f"observation_shape={self.observation_shape}, nonnegative_output={self.nonnegative_output}"


class GoalSetDistanceLoss(LossBase):
    @attrs.define(kw_only=True)
    class Conf:
        weight: float = attrs.field(default=1.0, validator=attrs.validators.gt(0))
        num_goal_samples: int = attrs.field(default=16, validator=attrs.validators.gt(0))
        goal_condition_radius: float = attrs.field(default=0.5, validator=attrs.validators.gt(0))
        goal_dims: Tuple[int, ...] = attrs.field(default=(0, 1), converter=tuple)
        sample_shape: str = attrs.field(default='ball', validator=attrs.validators.in_(('ball', 'box')))
        include_goal_state: bool = True
        add_goal_as_future_state: bool = True
        critic_reduction: str = attrs.field(
            default='max',
            validator=attrs.validators.in_(('first', 'mean', 'max', 'min')),
        )
        optim: AdamWSpec.Conf = AdamWSpec.Conf(lr=1e-4)

        def make(self, model: GoalSetDistance, total_optim_steps: int,
                 env_spec: EnvSpec) -> 'GoalSetDistanceLoss':
            return GoalSetDistanceLoss(
                model,
                total_optim_steps=total_optim_steps,
                env_spec=env_spec,
                weight=self.weight,
                num_goal_samples=self.num_goal_samples,
                goal_condition_radius=self.goal_condition_radius,
                goal_dims=self.goal_dims,
                sample_shape=self.sample_shape,
                include_goal_state=self.include_goal_state,
                add_goal_as_future_state=self.add_goal_as_future_state,
                critic_reduction=self.critic_reduction,
                optim_spec=self.optim.make(),
            )

    observation_shape: torch.Size
    weight: float
    num_goal_samples: int
    goal_condition_radius: float
    goal_dims: Tuple[int, ...]
    sample_shape: str
    include_goal_state: bool
    add_goal_as_future_state: bool
    critic_reduction: str
    optim: OptimWrapper
    sched: torch.optim.lr_scheduler._LRScheduler
    profiler: Optional[Any]

    def __init__(self, model: GoalSetDistance, *, total_optim_steps: int, env_spec: EnvSpec,
                 weight: float, num_goal_samples: int, goal_condition_radius: float,
                 goal_dims: Tuple[int, ...], sample_shape: str, include_goal_state: bool,
                 add_goal_as_future_state: bool, critic_reduction: str, optim_spec: AdamWSpec):
        super().__init__()
        self.observation_shape = env_spec.observation_shape
        self.weight = weight
        self.num_goal_samples = num_goal_samples
        self.goal_condition_radius = goal_condition_radius
        self.goal_dims = tuple(goal_dims)
        self.sample_shape = sample_shape
        self.include_goal_state = include_goal_state
        self.add_goal_as_future_state = add_goal_as_future_state
        self.critic_reduction = critic_reduction
        self.optim, self.sched = optim_spec.create_optim_scheduler(model.parameters(), total_optim_steps)
        self.profiler = None

    def _record(self, name: str):
        if self.profiler is None:
            return contextlib.nullcontext()
        return self.profiler.record(name)

    def _flatten_observations(self, observations: torch.Tensor) -> torch.Tensor:
        return observations.reshape(-1, *self.observation_shape)

    def _gather_state_goal_pairs(self, data: BatchData) -> Tuple[torch.Tensor, torch.Tensor]:
        state = data.observations
        goal_state = torch.roll(data.next_observations, 1, dims=0)
        if self.add_goal_as_future_state:
            goal_state = torch.stack([goal_state, data.future_observations], dim=0)
            state = state.expand_as(goal_state)
        return self._flatten_observations(state), self._flatten_observations(goal_state)

    def _sample_goal_condition_states(self, goal_states: torch.Tensor) -> torch.Tensor:
        if len(self.observation_shape) != 1:
            raise RuntimeError(
                "GoalSetDistanceLoss goal-condition sampling requires vector observations. "
                "For image observations, add an environment-specific sampler."
            )

        obs_dim = self.observation_shape[0]
        if len(self.goal_dims) == 0:
            raise RuntimeError("GoalSetDistanceLoss requires at least one goal_dim")
        if min(self.goal_dims) < 0 or max(self.goal_dims) >= obs_dim:
            raise RuntimeError(
                f"goal_dims={self.goal_dims!r} are invalid for observation dim {obs_dim}"
            )

        num_random_samples = self.num_goal_samples
        candidates = goal_states[:, None, :].expand(-1, num_random_samples, -1).clone()
        dims = torch.as_tensor(self.goal_dims, device=goal_states.device, dtype=torch.long)
        noise_shape = (goal_states.shape[0], num_random_samples, len(self.goal_dims))

        if self.sample_shape == 'box':
            noise = torch.empty(noise_shape, device=goal_states.device, dtype=goal_states.dtype)
            noise.uniform_(-self.goal_condition_radius, self.goal_condition_radius)
        elif self.sample_shape == 'ball':
            direction = torch.randn(noise_shape, device=goal_states.device, dtype=goal_states.dtype)
            direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            radius = torch.rand(
                goal_states.shape[0],
                num_random_samples,
                1,
                device=goal_states.device,
                dtype=goal_states.dtype,
            )
            radius = radius.pow(1.0 / len(self.goal_dims)) * self.goal_condition_radius
            noise = direction * radius
        else:
            raise ValueError(f"Unknown goal-condition sample_shape: {self.sample_shape!r}")

        candidates[..., dims] = goal_states[:, None, dims] + noise
        if self.include_goal_state:
            candidates = torch.cat([goal_states[:, None, :], candidates], dim=1)
        return candidates

    def _critic_goal_set_targets(self, critic_batch_infos: Collection[CriticBatchInfo],
                                 states: torch.Tensor, goal_states: torch.Tensor) -> torch.Tensor:
        candidates = self._sample_goal_condition_states(goal_states)
        num_pairs, num_candidates = candidates.shape[:2]
        flat_candidates = candidates.reshape(num_pairs * num_candidates, *self.observation_shape)
        critic_targets: List[torch.Tensor] = []

        for critic_batch_info in critic_batch_infos:
            critic = critic_batch_info.critic
            z_state = critic.encoder(states)
            z_candidates = critic.encoder(flat_candidates).reshape(num_pairs, num_candidates, -1)
            z_state = z_state[:, None, :].expand_as(z_candidates)
            candidate_dists = critic.quasimetric_model(z_state, z_candidates)
            critic_targets.append(candidate_dists.min(dim=1).values)

        stacked_targets = torch.stack(critic_targets, dim=-1)
        if self.critic_reduction == 'first':
            return stacked_targets[..., 0]
        if self.critic_reduction == 'mean':
            return stacked_targets.mean(dim=-1)
        if self.critic_reduction == 'max':
            return stacked_targets.max(dim=-1).values
        if self.critic_reduction == 'min':
            return stacked_targets.min(dim=-1).values
        raise ValueError(f"Unknown critic_reduction: {self.critic_reduction!r}")

    def _forward_loss(self, model: GoalSetDistance, critic_batch_infos: Collection[CriticBatchInfo],
                      data: BatchData) -> LossResult:
        if len(critic_batch_infos) == 0:
            raise RuntimeError("GoalSetDistanceLoss requires at least one critic")

        with self._record('train/goal_set_distance/gather_pairs'):
            states, goal_states = self._gather_state_goal_pairs(data)
        with self._record('train/goal_set_distance/model'):
            pred = model(states, goal_states)
        with torch.no_grad(), self._record('train/goal_set_distance/critic_target'):
            target = self._critic_goal_set_targets(critic_batch_infos, states, goal_states)

        mse = F.mse_loss(pred, target)
        loss = self.weight * mse
        info = dict(
            loss=loss,
            mse=mse,
            pred=pred.mean(),
            target=target.mean(),
            abs_error=(pred - target).abs().mean(),
            weight=self.weight,
            num_goal_samples=self.num_goal_samples,
            goal_condition_radius=self.goal_condition_radius,
        )
        return LossResult(loss=loss, info=info)

    def forward(self, model: GoalSetDistance, critic_batch_infos: Collection[CriticBatchInfo],
                data: BatchData, *, optimize: bool = True) -> LossResult:
        with self.optim.update_context(optimize=optimize):
            result = self._forward_loss(model, critic_batch_infos, data)
            with self._record('train/goal_set_distance/backward'):
                result.loss.backward()

        if optimize:
            with self._record('train/goal_set_distance/scheduler_step'):
                self.sched.step()
        return result

    def __call__(self, model: GoalSetDistance, critic_batch_infos: Collection[CriticBatchInfo],
                 data: BatchData, *, optimize: bool = True) -> LossResult:
        return torch.nn.Module.__call__(self, model, critic_batch_infos, data, optimize=optimize)

    def extra_repr(self) -> str:
        return (
            f"weight={self.weight:g}, num_goal_samples={self.num_goal_samples}, "
            f"goal_condition_radius={self.goal_condition_radius:g}, goal_dims={self.goal_dims}, "
            f"sample_shape={self.sample_shape}, critic_reduction={self.critic_reduction}"
        )


@attrs.define(kw_only=True)
class GoalSetDistanceConf:
    enabled: bool = False
    model: GoalSetDistance.Conf = GoalSetDistance.Conf()
    losses: GoalSetDistanceLoss.Conf = GoalSetDistanceLoss.Conf()

    def make(self, *, env_spec: EnvSpec, total_optim_steps: int) -> Tuple[
        Optional[GoalSetDistance],
        Optional[GoalSetDistanceLoss],
    ]:
        if not self.enabled:
            return None, None
        model = self.model.make(env_spec=env_spec)
        losses = self.losses.make(model, total_optim_steps=total_optim_steps, env_spec=env_spec)
        return model, losses


__all__ = ['GoalSetDistance', 'GoalSetDistanceLoss', 'GoalSetDistanceConf']

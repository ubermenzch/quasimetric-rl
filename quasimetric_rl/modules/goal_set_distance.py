from typing import *

import contextlib
import math

import attrs
import torch
import torch.nn.functional as F

from ..data import BatchData, EnvSpec
from .optim import AdamWSpec, OptimWrapper
from .quasimetric_critic import CriticBatchInfo
from .utils import LossBase, LossResult, MLP, Module


GOAL_SET_IMPLEMENTATIONS = ('learned', 'direct')
GOAL_SET_AGGREGATIONS = ('hard_min', 'lme_min', 'median', 'hard_max', 'lme_max')
GOAL_SET_CANDIDATE_SAMPLING_MODES = ('uniform_bounds', 'dataset_radius')


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
    IMPLEMENTATIONS = GOAL_SET_IMPLEMENTATIONS
    AGGREGATIONS = GOAL_SET_AGGREGATIONS
    CANDIDATE_SAMPLING_MODES = GOAL_SET_CANDIDATE_SAMPLING_MODES

    @attrs.define(kw_only=True)
    class Conf:
        implementation: str = attrs.field(
            default='learned', validator=attrs.validators.in_(GOAL_SET_IMPLEMENTATIONS)
        )
        aggregation: str = attrs.field(
            default='hard_min', validator=attrs.validators.in_(GOAL_SET_AGGREGATIONS)
        )
        lme_temperature: float = attrs.field(default=1.0, validator=attrs.validators.gt(0))
        weight: float = attrs.field(default=1.0, validator=attrs.validators.gt(0))
        num_goal_samples: int = attrs.field(default=16, validator=attrs.validators.gt(0))
        goal_condition_radius: float = attrs.field(default=0.5, validator=attrs.validators.gt(0))
        candidate_sampling: str = attrs.field(
            default='uniform_bounds',
            validator=attrs.validators.in_(GOAL_SET_CANDIDATE_SAMPLING_MODES),
        )
        dataset_max_sampling_attempts: int = attrs.field(
            default=256, validator=attrs.validators.gt(0)
        )
        # If omitted, use the environment's registered goal-state dimensions.
        # Set explicitly to override the built-in mapping for a custom task.
        goal_dims: Optional[Tuple[int, ...]] = attrs.field(
            default=None,
            converter=lambda dims: None if dims is None else tuple(dims),
        )
        # Retained as an explicit invariant for old configs: the raw goal is
        # always candidate zero and counts toward num_goal_samples.
        include_goal_state: bool = True
        optim: AdamWSpec.Conf = AdamWSpec.Conf(lr=1e-4)

        def make(self, model: Optional[GoalSetDistance], total_optim_steps: int,
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
                implementation=self.implementation,
                aggregation=self.aggregation,
                lme_temperature=self.lme_temperature,
                weight=self.weight,
                num_goal_samples=self.num_goal_samples,
                goal_dims=goal_dims,
                include_goal_state=self.include_goal_state,
                candidate_sampling=self.candidate_sampling,
                goal_condition_radius=self.goal_condition_radius,
                dataset_max_sampling_attempts=self.dataset_max_sampling_attempts,
                optim_spec=self.optim.make(),
            )

    observation_shape: torch.Size
    implementation: str
    aggregation: str
    lme_temperature: float
    weight: float
    num_goal_samples: int
    goal_dims: Tuple[int, ...]
    include_goal_state: bool
    candidate_sampling: str
    goal_condition_radius: float
    dataset_max_sampling_attempts: int
    optim: Optional[OptimWrapper]
    sched: Optional[torch.optim.lr_scheduler._LRScheduler]
    profiler: Optional[Any]
    observation_bounds_provider: Optional[Callable[..., Tuple[torch.Tensor, torch.Tensor]]]
    candidate_state_provider: Optional[Callable[..., Tuple[torch.Tensor, float]]]
    candidate_seed: int
    candidate_step: int
    last_candidate_fallback_fraction: float
    last_candidate_count_mean: float
    last_candidate_count_min: float
    last_candidate_shortfall_fraction: float

    def __init__(self, model: Optional[GoalSetDistance], *, total_optim_steps: int,
                 env_spec: EnvSpec, implementation: str, aggregation: str,
                 lme_temperature: float, weight: float, num_goal_samples: int,
                 goal_dims: Tuple[int, ...], include_goal_state: bool,
                 candidate_sampling: str, goal_condition_radius: float,
                 dataset_max_sampling_attempts: int, optim_spec: AdamWSpec):
        super().__init__()
        if len(env_spec.observation_shape) != 1:
            raise RuntimeError(
                'GoalSetDistance requires vector observations. Image goal sets need an '
                'environment-specific state sampler.')
        obs_dim = env_spec.observation_shape[0]
        if not goal_dims or min(goal_dims) < 0 or max(goal_dims) >= obs_dim:
            raise ValueError(f'Invalid goal_dims={goal_dims!r} for observation dim {obs_dim}')
        if implementation == 'learned' and model is None:
            raise ValueError('Learned goal-set objectives require a GoalSetDistance model')
        if implementation == 'direct' and model is not None:
            raise ValueError('Direct goal-set objectives must not create a GoalSetDistance model')
        if not include_goal_state:
            raise ValueError('The raw goal must be included in every goal-state candidate set')
        self.observation_shape = env_spec.observation_shape
        self.implementation = implementation
        self.aggregation = aggregation
        self.lme_temperature = lme_temperature
        self.weight = weight
        self.num_goal_samples = num_goal_samples
        self.goal_dims = tuple(goal_dims)
        self.include_goal_state = include_goal_state
        self.candidate_sampling = candidate_sampling
        self.goal_condition_radius = goal_condition_radius
        self.dataset_max_sampling_attempts = dataset_max_sampling_attempts
        if model is None:
            self.optim = self.sched = None
        else:
            self.optim, self.sched = optim_spec.create_optim_scheduler(
                model.parameters(), total_optim_steps
            )
        self.profiler = None
        self.observation_bounds_provider = None
        self.candidate_state_provider = None
        self.candidate_seed = 0
        self.candidate_step = 0
        self.last_candidate_fallback_fraction = 0.0
        self.last_candidate_count_mean = float(num_goal_samples)
        self.last_candidate_count_min = float(num_goal_samples)
        self.last_candidate_shortfall_fraction = 0.0

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

    def set_candidate_state_provider(
            self, provider: Callable[..., Tuple[torch.Tensor, float]]) -> None:
        """Set the dataset/replay source used by dataset-radius sampling."""
        self.candidate_state_provider = provider

    def set_candidate_seed(self, seed: int) -> None:
        """Start a candidate-only random stream independent of global PyTorch RNG."""
        self.candidate_seed = int(seed)
        self.candidate_step = 0

    def candidate_rng_state_dict(self) -> Dict[str, int]:
        return dict(seed=self.candidate_seed, step=self.candidate_step)

    def load_candidate_rng_state_dict(self, state: Optional[Mapping[str, Any]]) -> None:
        if state is None:
            return
        self.candidate_seed = int(state.get('seed', self.candidate_seed))
        self.candidate_step = int(state.get('step', self.candidate_step))

    def advance_candidate_step(self) -> None:
        self.candidate_step += 1

    def _candidate_step_seed(self) -> int:
        return (
            self.candidate_seed
            + self.candidate_step * 0x9E3779B97F4A7C15
        ) & 0x7FFFFFFFFFFFFFFF

    def _candidate_generator(self, device: torch.device) -> torch.Generator:
        generator = torch.Generator(device=device)
        # Each optimization step is addressable directly, so unrelated random
        # draws and checkpoint resumes cannot shift the candidate sequence.
        generator.manual_seed(self._candidate_step_seed())
        return generator

    def _sample_goal_condition_states_with_mask(
            self, raw_goal_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the raw goal followed by a fixed number of sampled states."""
        num_additional = self.num_goal_samples - 1
        fallback_fraction = 0.0
        if num_additional == 0:
            additional = raw_goal_states.new_empty(*raw_goal_states.shape[:-1], 0, *self.observation_shape)
            additional_mask = torch.empty(
                *raw_goal_states.shape[:-1], 0,
                dtype=torch.bool,
                device=raw_goal_states.device,
            )
        elif self.candidate_sampling == 'uniform_bounds':
            if self.observation_bounds_provider is None:
                raise RuntimeError('Uniform candidate sampling requires observation bounds')
            low, high = self.observation_bounds_provider(device=raw_goal_states.device)
            low = low.to(dtype=raw_goal_states.dtype)
            high = high.to(dtype=raw_goal_states.dtype)
            shape = (*raw_goal_states.shape[:-1], num_additional, *self.observation_shape)
            additional = low + torch.rand(
                shape,
                device=raw_goal_states.device,
                dtype=raw_goal_states.dtype,
                generator=self._candidate_generator(raw_goal_states.device),
            ) * (high - low)
            additional[..., list(self.goal_dims)] = raw_goal_states[..., None, list(self.goal_dims)]
            additional_mask = torch.ones(
                *raw_goal_states.shape[:-1], num_additional,
                dtype=torch.bool,
                device=raw_goal_states.device,
            )
        elif self.candidate_sampling == 'dataset_radius':
            if self.candidate_state_provider is None:
                raise RuntimeError('Dataset-radius sampling requires a candidate-state provider')
            additional, fallback_fraction = self.candidate_state_provider(
                raw_goal_states,
                goal_dims=self.goal_dims,
                num_samples=num_additional,
                radius=self.goal_condition_radius,
                seed=self._candidate_step_seed(),
                max_attempts=self.dataset_max_sampling_attempts,
            )
            additional_mask = torch.ones(
                *raw_goal_states.shape[:-1], num_additional,
                dtype=torch.bool,
                device=raw_goal_states.device,
            )
        else:
            raise RuntimeError(f'Unknown candidate_sampling={self.candidate_sampling!r}')

        raw_goal = raw_goal_states[..., None, :]
        raw_goal_mask = torch.ones(
            *raw_goal_states.shape[:-1], 1,
            dtype=torch.bool,
            device=raw_goal_states.device,
        )
        candidates = torch.cat([raw_goal, additional], dim=-2)
        candidate_mask = torch.cat([raw_goal_mask, additional_mask], dim=-1)
        candidate_counts = candidate_mask.sum(dim=-1)
        self.last_candidate_fallback_fraction = float(fallback_fraction)
        self.last_candidate_count_mean = candidate_counts.to(torch.float32).mean().item()
        self.last_candidate_count_min = candidate_counts.min().item()
        self.last_candidate_shortfall_fraction = (
            candidate_counts < self.num_goal_samples
        ).to(torch.float32).mean().item()
        return candidates, candidate_mask

    def _sample_goal_condition_states(self, raw_goal_states: torch.Tensor) -> torch.Tensor:
        candidates, _ = self._sample_goal_condition_states_with_mask(raw_goal_states)
        return candidates

    @staticmethod
    def _validated_candidate_mask(
            distances: torch.Tensor, candidate_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if candidate_mask is None:
            candidate_mask = torch.ones_like(distances, dtype=torch.bool)
        if candidate_mask.shape != distances.shape:
            raise ValueError(
                f'candidate_mask shape={tuple(candidate_mask.shape)} does not match '
                f'distances shape={tuple(distances.shape)}'
            )
        candidate_mask = candidate_mask.to(device=distances.device, dtype=torch.bool)
        if not candidate_mask.any(dim=-1).all():
            raise ValueError('Every candidate set must contain at least the raw goal')
        return candidate_mask

    def aggregate_distances(
            self, distances: torch.Tensor,
            candidate_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Aggregate the final candidate axis into one set objective."""
        if distances.shape[-1] <= 0:
            raise ValueError('Cannot aggregate an empty candidate set')
        candidate_mask = self._validated_candidate_mask(distances, candidate_mask)
        if self.aggregation == 'hard_min':
            return distances.masked_fill(~candidate_mask, torch.inf).min(dim=-1).values
        if self.aggregation == 'hard_max':
            return distances.masked_fill(~candidate_mask, -torch.inf).max(dim=-1).values
        if self.aggregation == 'median':
            sorted_distances = distances.masked_fill(~candidate_mask, torch.inf).sort(dim=-1).values
            counts = candidate_mask.sum(dim=-1)
            lower = ((counts - 1) // 2).unsqueeze(-1)
            upper = (counts // 2).unsqueeze(-1)
            lower_value = sorted_distances.gather(-1, lower).squeeze(-1)
            upper_value = sorted_distances.gather(-1, upper).squeeze(-1)
            return 0.5 * (lower_value + upper_value)
        return self._lme_distance(distances, self.aggregation, candidate_mask)

    def _lme_distance(
            self, distances: torch.Tensor, aggregation: str,
            candidate_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        candidate_mask = self._validated_candidate_mask(distances, candidate_mask)
        log_num_candidates = candidate_mask.sum(dim=-1).to(distances.dtype).log()
        if aggregation == 'lme_min':
            return -self.lme_temperature * (
                torch.logsumexp(
                    (-distances / self.lme_temperature).masked_fill(
                        ~candidate_mask, -torch.inf
                    ),
                    dim=-1,
                )
                - log_num_candidates
            )
        if aggregation == 'lme_max':
            return self.lme_temperature * (
                torch.logsumexp(
                    (distances / self.lme_temperature).masked_fill(
                        ~candidate_mask, -torch.inf
                    ),
                    dim=-1,
                )
                - log_num_candidates
            )
        raise RuntimeError(f'Expected an LME aggregation, got {aggregation!r}')

    def aggregation_diagnostics(
            self, distances: torch.Tensor,
            candidate_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Compare LME and hard order statistics on the same candidates."""
        candidate_mask = self._validated_candidate_mask(distances, candidate_mask)
        if self.aggregation in ('hard_min', 'lme_min'):
            hard = distances.masked_fill(~candidate_mask, torch.inf).min(dim=-1).values
            lme = self._lme_distance(distances, 'lme_min', candidate_mask)
            logits = -distances / self.lme_temperature
            gap = lme - hard
        elif self.aggregation in ('hard_max', 'lme_max'):
            hard = distances.masked_fill(~candidate_mask, -torch.inf).max(dim=-1).values
            lme = self._lme_distance(distances, 'lme_max', candidate_mask)
            logits = distances / self.lme_temperature
            gap = hard - lme
        else:
            return {}

        logits = logits.masked_fill(~candidate_mask, -torch.inf)
        log_weights = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        weights = log_weights.exp()
        safe_log_weights = log_weights.masked_fill(~candidate_mask, 0)
        entropy = -(weights * safe_log_weights).sum(dim=-1)
        counts = candidate_mask.sum(dim=-1)
        entropy_fraction = torch.where(
            counts > 1,
            entropy / counts.to(distances.dtype).log().clamp_min(1e-12),
            torch.zeros_like(entropy),
        )
        return dict(
            hard_reference=hard,
            lme_reference=lme,
            lme_hard_gap=gap,
            lme_hard_relative_gap=gap / hard.abs().clamp_min(1e-6),
            lme_weight_entropy_fraction=entropy_fraction,
            lme_effective_candidates=entropy.exp(),
        )

    @staticmethod
    def _gather_candidate_latent(
            candidate_latents: torch.Tensor, candidate_index: torch.Tensor) -> torch.Tensor:
        gather_index = candidate_index[..., None, None].expand(
            *candidate_index.shape, 1, candidate_latents.shape[-1]
        )
        return torch.gather(candidate_latents, -2, gather_index).squeeze(-2)

    def direct_actor_distance(
            self, critic: Any, next_latent: torch.Tensor,
            candidate_latents: torch.Tensor,
            candidate_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.direct_actor_distance_with_info(
            critic, next_latent, candidate_latents, candidate_mask
        )[0]

    def direct_actor_distance_with_info(
            self, critic: Any, next_latent: torch.Tensor,
            candidate_latents: torch.Tensor,
            candidate_mask: Optional[torch.Tensor] = None) -> Tuple[
                torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute a direct set objective while keeping critic parameters frozen."""
        if self.implementation != 'direct':
            raise RuntimeError('direct_actor_distance is only valid for direct objectives')
        candidate_mask = self._validated_candidate_mask(
            candidate_latents[..., 0], candidate_mask
        )
        expanded_next = next_latent.unsqueeze(-2).expand_as(candidate_latents)
        if self.aggregation in ('lme_min', 'lme_max'):
            distances = critic.quasimetric_model(expanded_next, candidate_latents)
            return self.aggregate_distances(
                distances, candidate_mask
            ), self.aggregation_diagnostics(
                distances.detach(), candidate_mask
            )

        # Hard order statistics only backpropagate through the selected one or
        # two candidates. Select without a graph, then recompute those distances.
        with torch.no_grad():
            selection_distances = critic.quasimetric_model(
                expanded_next.detach(), candidate_latents
            )
            diagnostics = self.aggregation_diagnostics(selection_distances, candidate_mask)
            if self.aggregation == 'hard_min':
                selected_indices = (
                    selection_distances.masked_fill(
                        ~candidate_mask, torch.inf
                    ).argmin(dim=-1),
                )
            elif self.aggregation == 'hard_max':
                selected_indices = (
                    selection_distances.masked_fill(
                        ~candidate_mask, -torch.inf
                    ).argmax(dim=-1),
                )
            elif self.aggregation == 'median':
                order = selection_distances.masked_fill(
                    ~candidate_mask, torch.inf
                ).argsort(dim=-1)
                counts = candidate_mask.sum(dim=-1)
                lower = ((counts - 1) // 2).unsqueeze(-1)
                upper = (counts // 2).unsqueeze(-1)
                selected_indices = (
                    order.gather(-1, lower).squeeze(-1),
                    order.gather(-1, upper).squeeze(-1),
                )
            else:
                raise RuntimeError(f'Unknown aggregation={self.aggregation!r}')

        selected_distances = [
            critic.quasimetric_model(
                next_latent,
                self._gather_candidate_latent(candidate_latents, index),
            )
            for index in selected_indices
        ]
        return torch.stack(selected_distances, dim=-1).mean(dim=-1), diagnostics

    def _forward_loss(self, model: GoalSetDistance, critic_batch_infos: Collection[CriticBatchInfo],
                      data: BatchData) -> LossResult:
        if len(critic_batch_infos) != model.num_critics:
            raise RuntimeError('GoalSetDistance head count must match the critic count')
        with self._record('train/goal_set_distance/gather_pairs'):
            states, raw_goals, padded_goals = self._gather_state_goal_pairs(data)
            candidates, candidate_mask = self._sample_goal_condition_states_with_mask(raw_goals)

        preds, targets, target_diagnostics = [], [], []
        with torch.no_grad(), self._record('train/goal_set_distance/critic_targets'):
            flat_candidates = candidates.reshape(-1, self.observation_shape[0])
            for idx, critic_batch_info in enumerate(critic_batch_infos):
                critic = critic_batch_info.critic
                z_state = critic.encoder(states)
                z_padded_goal = critic.encoder(padded_goals)
                z_candidates = critic.encoder(flat_candidates).reshape(
                    *candidates.shape[:-1], -1
                )
                dists = critic.quasimetric_model(
                    z_state[..., None, :].expand_as(z_candidates), z_candidates)
                targets.append(self.aggregate_distances(dists, candidate_mask))
                target_diagnostics.append(self.aggregation_diagnostics(dists, candidate_mask))
                preds.append((z_state, z_padded_goal))

        predictions = torch.stack([
            model(idx, z_state, z_padded_goal)
            for idx, (z_state, z_padded_goal) in enumerate(preds)
        ], dim=-1)
        target = torch.stack(targets, dim=-1)
        mse = F.mse_loss(predictions, target)
        loss = self.weight * mse
        info = dict(
            loss=loss, mse=mse, pred=predictions.mean(), target=target.mean(),
            abs_error=(predictions - target).abs().mean(), weight=self.weight,
            num_goal_samples=self.num_goal_samples,
            candidate_fallback_fraction=self.last_candidate_fallback_fraction,
            candidate_count_mean=self.last_candidate_count_mean,
            candidate_count_min=self.last_candidate_count_min,
            candidate_shortfall_fraction=self.last_candidate_shortfall_fraction,
        )
        if target_diagnostics and target_diagnostics[0]:
            for key in target_diagnostics[0]:
                values = torch.stack([diagnostics[key] for diagnostics in target_diagnostics], dim=-1)
                info[key] = values.mean()
                if key == 'lme_hard_gap':
                    info['lme_hard_gap_max'] = values.max()
        return LossResult(loss=loss, info=info)

    def forward(self, model: Optional[GoalSetDistance], critic_batch_infos: Collection[CriticBatchInfo],
                data: BatchData, *, optimize: bool = True) -> LossResult:
        if self.implementation != 'learned' or model is None or self.optim is None or self.sched is None:
            raise RuntimeError('Only learned goal-set objectives have a standalone training loss')
        with self.optim.update_context(optimize=optimize):
            result = self._forward_loss(model, critic_batch_infos, data)
            with self._record('train/goal_set_distance/backward'):
                result.loss.backward()
        if optimize:
            self.sched.step()
            self.advance_candidate_step()
        return result

    def __call__(self, model: Optional[GoalSetDistance], critic_batch_infos: Collection[CriticBatchInfo],
                 data: BatchData, *, optimize: bool = True) -> LossResult:
        return torch.nn.Module.__call__(self, model, critic_batch_infos, data, optimize=optimize)

    def extra_repr(self) -> str:
        return (
            f'implementation={self.implementation}, aggregation={self.aggregation}, '
            f'num_goal_samples={self.num_goal_samples}, goal_dims={self.goal_dims}, '
            f'candidate_sampling={self.candidate_sampling}, candidate_seed={self.candidate_seed}'
        )


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
        model = (
            self.model.make(latent_size=latent_size, num_critics=num_critics)
            if self.losses.implementation == 'learned'
            else None
        )
        return model, self.losses.make(
            model,
            total_optim_steps=total_optim_steps,
            env_spec=env_spec,
            goal_dims=goal_dims,
        )


__all__ = ['GoalSetDistance', 'GoalSetDistanceLoss', 'GoalSetDistanceConf']

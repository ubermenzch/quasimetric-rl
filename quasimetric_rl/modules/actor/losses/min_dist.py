from typing import *

import attrs

import torch
import torch.nn as nn

from ....data import BatchData, EnvSpec

from ...utils import LatentTensor, LossResult, grad_mul
from ..model import Actor
from ...quasimetric_critic import QuasimetricCritic, CriticBatchInfo
from ...quasimetric_critic.models.encoder import SplitEncoder

from . import ActorLossBase


LATENT_GOAL_MODES = ('none', 'min', 'max')
LATENT_GOAL_OPTIMIZERS = ('adam', 'sgd', 'rmsg')
LATENT_GOAL_SEARCHES = ('direct', 'residual', 'bounded_residual')


@attrs.define(kw_only=True)
class ActorObsGoalCriticInfo:
    r"""
    Similar to CriticBatchInfo, but does not store the latents for the data batch.

    Instead, for the batch of observation and goal pairs which the actor is activated with,
    this stores the latents for them.
    """
    critic: QuasimetricCritic
    zo: LatentTensor
    zg: LatentTensor


class MinDistLoss(ActorLossBase):
    @attrs.define(kw_only=True)
    class Conf:
        # config / argparse uses this to specify behavior

        adaptive_entropy_regularizer: bool = True
        target_entropy: Optional[float] = None
        entropy_mc_samples: int = attrs.field(
            default=100, validator=attrs.validators.gt(0)
        )

        # If set, in addition to use random goals, also use future state in the same trajectory as goals.
        # We enable this for online settings, following Contrastive RL.
        add_goal_as_future_state: bool = True

        latent_goal_mode: str = attrs.field(
            default='none', validator=attrs.validators.in_(LATENT_GOAL_MODES)
        )
        latent_goal_steps: int = attrs.field(default=8, validator=attrs.validators.ge(0))
        latent_goal_optim: str = attrs.field(
            default='adam', validator=attrs.validators.in_(LATENT_GOAL_OPTIMIZERS)
        )
        latent_goal_lr: float = attrs.field(default=0.01, validator=attrs.validators.gt(0))
        latent_goal_betas: Tuple[float, float] = attrs.field(
            default=(0.9, 0.999), converter=tuple
        )
        latent_goal_eps: float = attrs.field(default=1e-8, validator=attrs.validators.gt(0))
        latent_goal_keep_best: bool = False
        latent_goal_log_displacement: bool = False
        latent_goal_probe_replay: bool = False
        latent_goal_probe_interval: int = attrs.field(default=1000, validator=attrs.validators.gt(0))
        latent_goal_reference_size: int = attrs.field(default=2048, validator=attrs.validators.gt(0))
        latent_goal_probe_queries: int = attrs.field(default=32, validator=attrs.validators.gt(0))
        latent_goal_search: str = attrs.field(
            default='direct', validator=attrs.validators.in_(LATENT_GOAL_SEARCHES)
        )
        latent_goal_residual_radius: float = attrs.field(
            default=1.0, validator=attrs.validators.gt(0)
        )
        # Real-state alternative to latent optimization. The original sampled
        # target is included in this total; remaining candidates come from replay.
        replay_goal_candidates: int = attrs.field(default=0, validator=attrs.validators.ge(0))
        replay_goal_radius: float = attrs.field(default=0.05, validator=attrs.validators.gt(0))
        replay_goal_log_diagnostics: bool = False

        def make(self, env_spec: EnvSpec) -> 'MinDistLoss':
            return MinDistLoss(
                env_spec=env_spec,
                adaptive_entropy_regularizer=self.adaptive_entropy_regularizer,
                target_entropy=self.target_entropy,
                entropy_mc_samples=self.entropy_mc_samples,
                add_goal_as_future_state=self.add_goal_as_future_state,
                latent_goal_mode=self.latent_goal_mode,
                latent_goal_steps=self.latent_goal_steps,
                latent_goal_optim=self.latent_goal_optim,
                latent_goal_lr=self.latent_goal_lr,
                latent_goal_betas=self.latent_goal_betas,
                latent_goal_eps=self.latent_goal_eps,
                latent_goal_keep_best=self.latent_goal_keep_best,
                latent_goal_log_displacement=self.latent_goal_log_displacement,
                latent_goal_probe_replay=self.latent_goal_probe_replay,
                latent_goal_probe_interval=self.latent_goal_probe_interval,
                latent_goal_reference_size=self.latent_goal_reference_size,
                latent_goal_probe_queries=self.latent_goal_probe_queries,
                latent_goal_search=self.latent_goal_search,
                latent_goal_residual_radius=self.latent_goal_residual_radius,
                replay_goal_candidates=self.replay_goal_candidates,
                replay_goal_radius=self.replay_goal_radius,
                replay_goal_log_diagnostics=self.replay_goal_log_diagnostics,
            )

    add_goal_as_future_state: bool
    raw_entropy_weight: Optional[nn.Parameter]  # set if using adaptive entropy regularization
    target_entropy: Optional[float] = None  # set if using adaptive entropy regularization
    entropy_mc_samples: int
    latent_goal_mode: str
    latent_goal_steps: int
    latent_goal_optim: str
    latent_goal_lr: float
    latent_goal_betas: Tuple[float, float]
    latent_goal_eps: float
    latent_goal_keep_best: bool
    latent_goal_log_displacement: bool
    latent_goal_search: str
    latent_goal_residual_radius: float

    def __init__(self, *, env_spec: EnvSpec,
                 adaptive_entropy_regularizer: bool,
                 target_entropy: Optional[float] = None,
                 entropy_mc_samples: int = 100,
                 add_goal_as_future_state: bool,
                 latent_goal_mode: str = 'none',
                 latent_goal_steps: int = 8,
                 latent_goal_optim: str = 'adam',
                 latent_goal_lr: float = 0.01,
                 latent_goal_betas: Tuple[float, float] = (0.9, 0.999),
                 latent_goal_eps: float = 1e-8,
                 latent_goal_keep_best: bool = False,
                 latent_goal_log_displacement: bool = False,
                 latent_goal_probe_replay: bool = False,
                 latent_goal_probe_interval: int = 1000,
                 latent_goal_reference_size: int = 2048,
                 latent_goal_probe_queries: int = 32,
                 latent_goal_search: str = 'direct',
                 latent_goal_residual_radius: float = 1.0,
                 replay_goal_candidates: int = 0,
                 replay_goal_radius: float = 0.05,
                 replay_goal_log_diagnostics: bool = False):
        super().__init__()
        if not env_spec.action_dtype.is_floating_point:
            raise RuntimeError(
                'Discrete action spaces do not support optimizing actor by backpropagation through the critic. '
                'Set agent.actor=null to turn of actor optimization.'
            )

        self.add_goal_as_future_state = add_goal_as_future_state
        self.goal_marker_dims = None
        if latent_goal_mode not in LATENT_GOAL_MODES:
            raise ValueError(f'Unknown latent_goal_mode={latent_goal_mode!r}')
        if latent_goal_optim not in LATENT_GOAL_OPTIMIZERS:
            raise ValueError(f'Unknown latent_goal_optim={latent_goal_optim!r}')
        if latent_goal_search not in LATENT_GOAL_SEARCHES:
            raise ValueError(f'Unknown latent_goal_search={latent_goal_search!r}')
        if (latent_goal_steps < 0 or latent_goal_lr <= 0
                or latent_goal_eps <= 0 or latent_goal_residual_radius <= 0):
            raise ValueError(
                'Latent goal steps must be non-negative; lr, eps, and residual '
                'radius must be positive'
            )
        if (len(latent_goal_betas) != 2
                or not all(0 <= beta < 1 for beta in latent_goal_betas)):
            raise ValueError(
                f'Expected two latent goal Adam betas in [0, 1), got {latent_goal_betas!r}'
            )
        if latent_goal_keep_best and latent_goal_mode == 'none':
            raise ValueError(
                'latent_goal_keep_best requires latent_goal_mode=min or max'
            )
        if latent_goal_search != 'direct' and latent_goal_mode == 'none':
            raise ValueError(
                'Non-direct latent goal search requires latent_goal_mode=min or max'
            )
        self.latent_goal_mode = latent_goal_mode
        self.latent_goal_steps = latent_goal_steps
        self.latent_goal_optim = latent_goal_optim
        self.latent_goal_lr = latent_goal_lr
        self.latent_goal_betas = tuple(latent_goal_betas)
        self.latent_goal_eps = latent_goal_eps
        self.latent_goal_keep_best = latent_goal_keep_best
        self.latent_goal_log_displacement = latent_goal_log_displacement
        if min(latent_goal_probe_interval, latent_goal_reference_size, latent_goal_probe_queries) <= 0:
            raise ValueError('Replay diagnostic interval, reference size and query count must be positive')
        if latent_goal_probe_replay and latent_goal_mode not in ('min', 'max'):
            raise ValueError('Completion support diagnostics require latent search')
        self.latent_goal_probe_replay = latent_goal_probe_replay
        self.latent_goal_probe_interval = latent_goal_probe_interval
        self.latent_goal_reference_size = latent_goal_reference_size
        self.latent_goal_probe_queries = latent_goal_probe_queries
        self._replay_reference_provider = None
        self._support_seed = 0
        self._support_calls = 0
        self._support_cache = {}
        self._diagnostic_sink = None
        self._last_search_probe = None
        self.latent_goal_search = latent_goal_search
        self.latent_goal_residual_radius = latent_goal_residual_radius
        if replay_goal_candidates < 0 or replay_goal_radius <= 0:
            raise ValueError('Replay candidate count must be nonnegative and radius positive')
        if replay_goal_candidates and latent_goal_mode != 'none':
            raise ValueError('Replay goal selection and latent search are separate controls')
        self.replay_goal_candidates = replay_goal_candidates
        self.replay_goal_radius = replay_goal_radius
        self.replay_goal_log_diagnostics = replay_goal_log_diagnostics
        self._replay_candidate_provider = None
        if replay_goal_candidates:
            # Persist this separate sampling stream across checkpoint resumes.
            self.register_buffer('replay_candidate_seed', torch.tensor(0, dtype=torch.long))
            self.register_buffer('replay_candidate_draws', torch.tensor(0, dtype=torch.long))
        if entropy_mc_samples <= 0:
            raise ValueError(
                f'Expected a positive entropy sample count, got {entropy_mc_samples}'
            )
        self.entropy_mc_samples = entropy_mc_samples
        if adaptive_entropy_regularizer:
            self.raw_entropy_weight = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
            self.target_entropy = (
                env_spec.get_action_entropy_reg_target()
                if target_entropy is None
                else float(target_entropy)
            )
        else:
            self.register_parameter('raw_entropy_weight', None)
            self.target_entropy = None

    def adaptive_entropy_loss(
            self, entropy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.target_entropy is None or self.raw_entropy_weight is None:
            raise RuntimeError('Adaptive entropy regularization is disabled')
        alpha = grad_mul(self.raw_entropy_weight.exp(), -1)
        return alpha * (self.target_entropy - entropy), alpha

    def gather_obs_goal_pairs(
            self, critic_batch_infos: Collection[CriticBatchInfo], data: BatchData,
            *, goal_set_distance_loss: Optional[Any] = None,
            use_sampled_goal_non_goal_latent: bool = False,
            reencode_full_targets: bool = False) -> Tuple[
                torch.Tensor, torch.Tensor, Collection[ActorObsGoalCriticInfo]]:
        r"""
        Returns (
            obs,
            goal,
            [ (latent_obs, latent_goal) for each critic ],
        )
        """

        obs, goal = self.gather_raw_obs_goal_pairs(data)
        if self.goal_marker_dims is not None:
            if goal_set_distance_loss is not None:
                raise ValueError('Goal markers and sampled goal-set distances are separate baselines')
            from ...goal_marker import make_goal_marker
            goal = make_goal_marker(goal, self.goal_marker_dims)
            reencode_full_targets = True
        if goal_set_distance_loss is not None:
            # Actor and GSD use the same goal-set representation: goal dims are
            # retained while every non-goal dimension is zeroed.
            goal = goal_set_distance_loss.padded_goal_state(goal)

        actor_obs_goal_critic_infos: List[ActorObsGoalCriticInfo] = []

        for critic_batch_info in critic_batch_infos:
            critic = critic_batch_info.critic
            if isinstance(critic.encoder, SplitEncoder):
                if goal_set_distance_loss is not None:
                    raise RuntimeError(
                        'Split latent goals cannot be combined with GoalSetDistance'
                    )
                # The critic has already stepped in joint training, so re-encode
                # both inputs with its current encoder weights.
                zo = critic.encoder(obs)
                zg = (
                    critic.encoder(goal)
                    if use_sampled_goal_non_goal_latent
                    else critic.encoder.encode_actor_goal(goal)
                )
            elif goal_set_distance_loss is None and not reencode_full_targets:
                zo = critic_batch_info.zx
                zg = torch.roll(critic_batch_info.zy, 1, dims=0)  # randomize in the same way:)
                if self.add_goal_as_future_state:
                    # add future_observations
                    zg = torch.stack([
                        zg,
                        critic_batch_info.critic.encoder(data.future_observations),
                    ], 0)
            else:
                # Match SSE's current encoder version in the new input
                # controls. Preserve the original QRL raw-input path above.
                zo, zg = critic_batch_info.critic.encoder(
                    torch.stack([obs, goal], dim=0)
                ).unbind(0)
            if self.add_goal_as_future_state:
                zo = zo.expand_as(zg)

            actor_obs_goal_critic_infos.append(ActorObsGoalCriticInfo(
                critic=critic_batch_info.critic,
                zo=zo,
                zg=zg,
            ))

        return obs, goal, actor_obs_goal_critic_infos

    @staticmethod
    def _latent_part_diagnostics(
            encoder: SplitEncoder, latent: torch.Tensor) -> Dict[str, torch.Tensor]:
        goal_latent, non_goal_latent = encoder.split_latent(latent)
        result: Dict[str, torch.Tensor] = {}
        for name, part in (
                ('goal_latent', goal_latent),
                ('non_goal_latent', non_goal_latent)):
            flat = part.reshape(-1, part.shape[-1])
            per_dim_std = flat.std(dim=0, unbiased=False)
            result[f'{name}_mean'] = flat.mean()
            result[f'{name}_std_mean'] = per_dim_std.mean()
            result[f'{name}_std_min'] = per_dim_std.min()
            result[f'{name}_norm_mean'] = torch.linalg.vector_norm(flat, dim=-1).mean()
            result[f'{name}_active_fraction'] = (per_dim_std > 1e-6).to(
                flat.dtype
            ).mean()
        return result

    @staticmethod
    def _standardized_residual_rms(residual: torch.Tensor) -> torch.Tensor:
        return residual.square().mean(dim=-1).sqrt()

    def _project_bounded_residual(self, residual: torch.Tensor) -> torch.Tensor:
        residual_rms = self._standardized_residual_rms(residual).unsqueeze(-1)
        scale = (
            self.latent_goal_residual_radius
            / residual_rms.clamp_min(self.latent_goal_eps)
        ).clamp(max=1.0)
        return residual * scale

    def _optimize_latent_goal(
            self, critic: QuasimetricCritic, predicted_latent: torch.Tensor,
            initial_goal_latent: torch.Tensor) -> Tuple[
                torch.Tensor, Dict[str, torch.Tensor]]:
        if self.latent_goal_mode not in ('min', 'max'):
            raise RuntimeError('Latent goal optimization requires min or max mode')
        if not isinstance(critic.encoder, SplitEncoder):
            raise RuntimeError('Latent goal optimization requires SplitEncoder')

        goal_latent, initial_non_goal_latent = critic.encoder.split_latent(
            initial_goal_latent.detach()
        )
        if initial_non_goal_latent.dtype in (torch.float16, torch.bfloat16):
            # The inner optimization repeatedly differentiates with respect to
            # this latent. Retain FP32 search state while allowing the large
            # projector and dynamics MLPs to run under autocast.
            goal_latent = goal_latent.float()
            initial_non_goal_latent = initial_non_goal_latent.float()
            predicted_latent = predicted_latent.float()
        residual_search = self.latent_goal_search in (
            'residual', 'bounded_residual'
        )
        if residual_search:
            # Optimize relative movement in batch-standardized coordinates.
            flat_initial_h = initial_non_goal_latent.reshape(
                -1, initial_non_goal_latent.shape[-1]
            )
            residual_scale = flat_initial_h.std(dim=0, unbiased=False).clamp_min(
                self.latent_goal_eps
            )
            search_variable = torch.zeros_like(
                initial_non_goal_latent
            ).requires_grad_(True)

            def candidate_h(variable: torch.Tensor) -> torch.Tensor:
                return initial_non_goal_latent + residual_scale * variable
        else:
            residual_scale = None
            search_variable = initial_non_goal_latent.clone().requires_grad_(True)

            def candidate_h(variable: torch.Tensor) -> torch.Tensor:
                # LayerNorm defines the encoder output, not the inner search space.
                if critic.encoder.branch_normalization == 'layernorm':
                    return variable
                return critic.encoder.normalize_non_goal_part(variable)

        first_moment = torch.zeros_like(search_variable)
        second_moment = torch.zeros_like(search_variable)
        beta1, beta2 = self.latent_goal_betas
        direction = -1.0 if self.latent_goal_mode == 'min' else 1.0
        frozen_prediction = predicted_latent.detach()

        with torch.no_grad():
            projected_prediction = critic.quasimetric_model.project(
                frozen_prediction
            )
            initial_h = candidate_h(search_variable.detach())
            initial_dist = critic.quasimetric_model.forward_projected(
                projected_prediction,
                critic.quasimetric_model.project(
                    critic.encoder.join_parts(goal_latent, initial_h)
                ),
            )
            best_h = initial_h
            best_dist = initial_dist
            best_step = torch.zeros_like(initial_dist, dtype=torch.long)
            best_residual_rms = torch.zeros_like(initial_dist)
        gradient_norms: List[torch.Tensor] = []
        update_norms: List[torch.Tensor] = []
        cost_trace = {}
        probe_due = (self.latent_goal_probe_replay
                     and self._support_calls % self.latent_goal_probe_interval == 0)
        query_indices = torch.linspace(
            0, initial_h.numel() // initial_h.shape[-1] - 1,
            min(self.latent_goal_probe_queries, initial_h.numel() // initial_h.shape[-1]),
            device=initial_h.device,
        ).long() if probe_due else None
        h_trace, sample_costs, sample_gradient_norms = [], [], []

        for step in range(1, self.latent_goal_steps + 1):
            with torch.enable_grad():
                current_h = candidate_h(search_variable)
                completed_goal = critic.encoder.join_parts(
                    goal_latent, current_h
                )
                inner_dist = critic.quasimetric_model.forward_projected(
                    projected_prediction,
                    critic.quasimetric_model.project(completed_goal),
                )
                if not torch.isfinite(inner_dist).all():
                    raise FloatingPointError(
                        f'Non-finite latent goal {self.latent_goal_mode} distance '
                        f'at inner step {step}'
                    )
                gradient, = torch.autograd.grad(
                    inner_dist.sum(), search_variable
                )
            if not torch.isfinite(gradient).all():
                raise FloatingPointError(
                    f'Non-finite latent goal gradient at inner step {step}'
                )
            if self.latent_goal_log_displacement and step - 1 in (0, 1, 2, 4, 8, 16, 32):
                cost_trace[f'latent_goal_cost_step_{step - 1:02d}'] = inner_dist.detach().mean()
            if probe_due:
                h_trace.append(current_h.detach().reshape(-1, current_h.shape[-1])[query_indices].clone())
                sample_costs.append(inner_dist.detach().reshape(-1)[query_indices].clone())
                sample_gradient_norms.append(gradient.detach().norm(dim=-1).reshape(-1)[query_indices].clone())
            if self.latent_goal_keep_best:
                with torch.no_grad():
                    candidate_dist = inner_dist.detach()
                    candidate_residual_rms = (
                        self._standardized_residual_rms(search_variable.detach())
                        if residual_search
                        else torch.zeros_like(candidate_dist)
                    )
                    improved = (
                        candidate_dist < best_dist
                        if self.latent_goal_mode == 'min'
                        else candidate_dist > best_dist
                    )
                    best_h = torch.where(
                        improved.unsqueeze(-1), current_h.detach(), best_h
                    )
                    best_dist = torch.where(improved, candidate_dist, best_dist)
                    best_step = torch.where(
                        improved,
                        torch.full_like(best_step, step - 1),
                        best_step,
                    )
                    best_residual_rms = torch.where(
                        improved, candidate_residual_rms, best_residual_rms
                    )

            gradient = gradient.detach()
            if self.latent_goal_optim == 'adam':
                first_moment = beta1 * first_moment + (1 - beta1) * gradient
                second_moment = beta2 * second_moment + (1 - beta2) * gradient.square()
                corrected_first = first_moment / (1 - beta1 ** step)
                corrected_second = second_moment / (1 - beta2 ** step)
                update = self.latent_goal_lr * corrected_first / (
                    corrected_second.sqrt() + self.latent_goal_eps
                )
            elif self.latent_goal_optim == 'sgd':
                update = self.latent_goal_lr * gradient
            elif self.latent_goal_optim == 'rmsg':
                gradient_rms = gradient.square().mean(
                    dim=-1, keepdim=True
                ).sqrt().clamp_min(self.latent_goal_eps)
                update = self.latent_goal_lr * gradient / gradient_rms
            else:
                raise AssertionError(self.latent_goal_optim)
            gradient_norms.append(torch.linalg.vector_norm(gradient, dim=-1).mean())
            update_norms.append(torch.linalg.vector_norm(update, dim=-1).mean())
            search_variable = (search_variable + direction * update).detach()
            if self.latent_goal_search == 'bounded_residual':
                search_variable = self._project_bounded_residual(search_variable)
            search_variable = search_variable.requires_grad_(True)

        with torch.no_grad():
            last_h = candidate_h(search_variable.detach())
            last_residual_rms = (
                self._standardized_residual_rms(search_variable.detach())
                if residual_search
                else torch.zeros_like(initial_dist)
            )
            last_inner_dist = critic.quasimetric_model.forward_projected(
                projected_prediction,
                critic.quasimetric_model.project(
                    critic.encoder.join_parts(goal_latent, last_h)
                ),
            )
        if not torch.isfinite(last_inner_dist).all():
            raise FloatingPointError('Non-finite final latent goal distance')

        if self.latent_goal_keep_best:
            improved = (
                last_inner_dist < best_dist
                if self.latent_goal_mode == 'min'
                else last_inner_dist > best_dist
            )
            best_h = torch.where(improved.unsqueeze(-1), last_h, best_h)
            best_dist = torch.where(improved, last_inner_dist, best_dist)
            best_step = torch.where(
                improved,
                torch.full_like(best_step, self.latent_goal_steps),
                best_step,
            )
            best_residual_rms = torch.where(
                improved, last_residual_rms, best_residual_rms
            )
        else:
            best_h = last_h
            best_dist = last_inner_dist
            best_step.fill_(self.latent_goal_steps)
            best_residual_rms = last_residual_rms

        final_h = best_h
        final_inner_dist = best_dist
        completed_goal = critic.encoder.join_parts(goal_latent, final_h)
        if probe_due:
            h_trace.append(last_h.detach().reshape(-1, last_h.shape[-1])[query_indices].clone())
            sample_costs.append(last_inner_dist.reshape(-1)[query_indices].clone())
            self._last_search_probe = dict(
                query_indices=query_indices, search_h=torch.stack(h_trace),
                search_cost=torch.stack(sample_costs),
                predicted_latent=frozen_prediction.reshape(-1, frozen_prediction.shape[-1])[query_indices],
                search_gradient_norm=torch.stack(sample_gradient_norms) if sample_gradient_norms else initial_h.new_empty((0, len(query_indices))),
                retained_step=best_step.reshape(-1)[query_indices],
                retained_cost=final_inner_dist.reshape(-1)[query_indices],
            )

        improvement = (
            initial_dist - final_inner_dist
            if self.latent_goal_mode == 'min'
            else final_inner_dist - initial_dist
        )
        zero_diagnostic = initial_dist.new_zeros(())
        gradient_norm = (
            torch.stack(gradient_norms).mean()
            if gradient_norms else zero_diagnostic
        )
        gradient_norm_max = (
            torch.stack(gradient_norms).max()
            if gradient_norms else zero_diagnostic
        )
        update_norm = (
            torch.stack(update_norms).mean()
            if update_norms else zero_diagnostic
        )
        update_norm_max = (
            torch.stack(update_norms).max()
            if update_norms else zero_diagnostic
        )
        diagnostics = dict(
            latent_goal_initial_dist=initial_dist.mean(),
            latent_goal_final_inner_dist=final_inner_dist.mean(),
            latent_goal_last_inner_dist=last_inner_dist.mean(),
            latent_goal_improvement=improvement.mean(),
            latent_goal_initial_norm=torch.linalg.vector_norm(
                initial_h, dim=-1
            ).mean(),
            latent_goal_initial_norm_max=torch.linalg.vector_norm(
                initial_h, dim=-1
            ).max(),
            latent_goal_gradient_norm=gradient_norm,
            latent_goal_gradient_norm_max=gradient_norm_max,
            latent_goal_update_norm=update_norm,
            latent_goal_update_norm_max=update_norm_max,
            latent_goal_final_norm=torch.linalg.vector_norm(
                final_h, dim=-1
            ).mean(),
            latent_goal_final_norm_max=torch.linalg.vector_norm(
                final_h, dim=-1
            ).max(),
            latent_goal_best_step=best_step.to(torch.float32).mean(),
            latent_goal_best_step_max=best_step.max(),
            latent_goal_selected_initial_fraction=(best_step == 0).to(
                final_inner_dist.dtype
            ).mean(),
        )
        if self.latent_goal_log_displacement:
            cost_trace[f'latent_goal_cost_step_{self.latent_goal_steps:02d}'] = last_inner_dist.mean()
            diagnostics.update(cost_trace)
        if self.latent_goal_log_displacement:
            with torch.no_grad():
                initial_norm = torch.linalg.vector_norm(initial_h.float(), dim=-1)
                denominator = initial_norm.clamp_min(self.latent_goal_eps)
                diagnostics['latent_goal_initial_norm_near_zero_fraction'] = (
                    initial_norm < self.latent_goal_eps
                ).float().mean()
                # "final" is the retained candidate; "last" is the raw K-th
                # iterate. Measure per-example vector differences, not a
                # difference between initial/final mean vector norms.
                for label, candidate in (('final', final_h), ('last', last_h)):
                    displacement = torch.linalg.vector_norm(
                        candidate.float() - initial_h.float(), dim=-1,
                    )
                    flat = displacement.reshape(-1)
                    prefix = f'latent_goal_{label}'
                    diagnostics.update({
                        f'{prefix}_displacement': flat.mean(),
                        f'{prefix}_displacement_std': flat.std(unbiased=False),
                        f'{prefix}_displacement_median': torch.quantile(flat, 0.5),
                        f'{prefix}_displacement_p95': torch.quantile(flat, 0.95),
                        f'{prefix}_displacement_max': flat.max(),
                        f'{prefix}_relative_displacement': (displacement / denominator).mean(),
                        f'{prefix}_displacement_rms': (flat / initial_h.shape[-1] ** 0.5).mean(),
                    })
        if residual_scale is not None:
            diagnostics.update(
                latent_goal_residual_scale_mean=residual_scale.mean(),
                latent_goal_residual_scale_min=residual_scale.min(),
                latent_goal_residual_scale_max=residual_scale.max(),
                latent_goal_last_residual_rms=last_residual_rms.mean(),
                latent_goal_last_residual_rms_max=last_residual_rms.max(),
                latent_goal_final_residual_rms=best_residual_rms.mean(),
                latent_goal_final_residual_rms_max=best_residual_rms.max(),
            )
        return completed_goal.detach(), diagnostics

    def set_replay_reference_provider(self, provider: Callable, *, seed: int) -> None:
        self._replay_reference_provider = provider
        self._support_seed = int(seed)

    def set_diagnostic_sink(self, sink: Callable) -> None:
        self._diagnostic_sink = sink

    @torch.no_grad()
    def _completion_support(self, critic, raw_goals, initial_latent, completed_latent):
        """Compare completions to an independently sampled replay reference bank.

        Exclude the original target state itself from both comparisons. Cache
        each probe with its update index so ordinary scalar logging cannot miss
        probes between logging boundaries; analyses can deduplicate that index.
        """
        self._support_calls += 1
        if (self._support_calls - 1) % self.latent_goal_probe_interval:
            return self._support_cache
        if self._replay_reference_provider is None:
            raise RuntimeError('Completion diagnostics have no replay reference provider')
        reference = self._replay_reference_provider(
            num_samples=self.latent_goal_reference_size,
            seed=self._support_seed + self._support_calls,
            device=raw_goals.device,
        )
        if reference.ndim != 2 or not len(reference):
            raise ValueError('Replay reference must be a nonempty matrix of complete states')
        flat_raw = raw_goals.reshape(-1, raw_goals.shape[-1])
        count = min(self.latent_goal_probe_queries, len(flat_raw))
        indices = torch.linspace(0, len(flat_raw) - 1, count, device=flat_raw.device).long()
        raw = flat_raw[indices]
        initial = initial_latent.reshape(-1, initial_latent.shape[-1])[indices].float()
        completed = completed_latent.reshape(-1, completed_latent.shape[-1])[indices].float()
        encoded = critic.encoder(reference).float()
        same_state = (raw[:, None, :] == reference[None, :, :]).all(-1)
        valid = (~same_state).any(-1)
        stats = dict(
            latent_goal_support_probe_update=initial.new_tensor(self._support_calls),
            latent_goal_support_reference_size=initial.new_tensor(len(reference)),
            latent_goal_support_queries=initial.new_tensor(count),
            latent_goal_support_valid_queries=valid.sum().float(),
            latent_goal_support_self_matches_excluded=same_state.sum().float(),
        )
        for label, first, last, bank in (
                ('full', initial, completed, encoded),
                ('non_goal', critic.encoder.split_latent(initial)[1],
                 critic.encoder.split_latent(completed)[1], critic.encoder.split_latent(encoded)[1])):
            # Direct differences avoid cancellation for nearly identical
            # replay embeddings with large norms.
            before = torch.cdist(first, bank, compute_mode='donot_use_mm_for_euclid_dist').masked_fill(same_state, float('inf')).min(-1).values[valid]
            after = torch.cdist(last, bank, compute_mode='donot_use_mm_for_euclid_dist').masked_fill(same_state, float('inf')).min(-1).values[valid]
            if not before.numel():
                continue
            prefix = f'latent_goal_support_{label}'
            stats.update({
                f'{prefix}_initial_l2': before.mean(),
                f'{prefix}_final_l2': after.mean(),
                f'{prefix}_initial_rms': before.mean() / first.shape[-1] ** .5,
                f'{prefix}_final_rms': after.mean() / first.shape[-1] ** .5,
                f'{prefix}_final_p95': torch.quantile(after, .95),
                f'{prefix}_increased_fraction': (after > before).float().mean(),
            })
        self._support_cache = stats
        if self._diagnostic_sink is not None:
            payload = dict(self._last_search_probe or {})
            payload.update(raw_goals=raw, initial_latent=initial, completed_latent=completed,
                           reference_states=reference, reference_latents=encoded,
                           self_match_mask=same_state)
            self._diagnostic_sink('completion', self._support_calls, payload)
        self._last_search_probe = None
        return stats

    def set_replay_candidate_provider(self, provider: Callable, *, seed: int) -> None:
        if not self.replay_goal_candidates:
            raise RuntimeError('Replay goal selection is disabled')
        self._replay_candidate_provider = provider
        self.replay_candidate_seed.fill_(seed)

    @torch.no_grad()
    def _select_replay_goal(self, critic, predicted_latent, raw_goals, initial_latent):
        """Select a complete, unmodified replay state within the task tolerance.

        Selection is detached. The caller recomputes distance to the chosen
        target with action gradients enabled, just as for ANCC completions.
        """
        if not isinstance(critic.encoder, SplitEncoder):
            raise RuntimeError('Replay goal selection requires SplitEncoder')
        if self._replay_candidate_provider is None:
            raise RuntimeError('Replay goal selection has no replay provider')
        count = self.replay_goal_candidates
        seed = int(self.replay_candidate_seed.item() + self.replay_candidate_draws.item())
        probe_due = (self.replay_goal_log_diagnostics and self._diagnostic_sink is not None
                     and int(self.replay_candidate_draws.item()) % self.latent_goal_probe_interval == 0)
        self.replay_candidate_draws.add_(1)
        if count > 1:
            candidates, fallback = self._replay_candidate_provider(
                raw_goals, goal_dims=critic.encoder.goal_dims,
                num_samples=count - 1, radius=self.replay_goal_radius,
                seed=seed, max_attempts=256,
            )
            expected = (*raw_goals.shape[:-1], count - 1, raw_goals.shape[-1])
            if candidates.shape != expected:
                raise ValueError(f'Unexpected replay candidate shape {candidates.shape}; expected {expected}')
        else:
            candidates, fallback = raw_goals.unsqueeze(-2)[..., :0, :], 0.0
        flat_query = predicted_latent.detach().reshape(-1, predicted_latent.shape[-1])
        flat_raw = raw_goals.reshape(-1, raw_goals.shape[-1])
        flat_candidates = candidates.reshape(flat_raw.shape[0], count - 1, flat_raw.shape[-1])
        best_latent = initial_latent.detach().reshape(flat_query.shape)
        best_raw = flat_raw.clone()
        initial_cost = critic.quasimetric_model(flat_query, best_latent)
        best_cost = initial_cost.clone()
        goal_dims = list(critic.encoder.goal_dims)
        query_indices = torch.linspace(0, len(flat_raw) - 1,
                                       min(self.latent_goal_probe_queries, len(flat_raw)),
                                       device=flat_raw.device).long() if probe_due else None
        logged_costs, logged_valid = [], []
        chosen_index = torch.zeros(len(flat_raw), dtype=torch.long, device=flat_raw.device)
        rejected = torch.zeros((), device=flat_query.device)
        # Bound peak memory: only one small candidate chunk reaches the encoder
        # and projector at a time, and no candidate activations retain gradients.
        for start in range(0, count - 1, 4):
            states = flat_candidates[:, start:start + 4]
            offsets = (states[..., goal_dims] - flat_raw[:, None, goal_dims]).norm(dim=-1)
            valid = offsets <= self.replay_goal_radius
            rejected += (~valid).sum()
            latents = critic.encoder(states.flatten(0, 1)).reshape(
                states.shape[0], states.shape[1], -1,
            )
            costs = critic.quasimetric_model(flat_query[:, None, :], latents)
            costs = costs.masked_fill(~valid, float('inf'))
            if probe_due:
                logged_costs.append(costs[query_indices].clone())
                logged_valid.append(valid[query_indices].clone())
            chunk_cost, index = costs.min(dim=-1)
            selected = torch.arange(states.shape[0], device=states.device)
            improved = chunk_cost < best_cost
            best_latent = torch.where(improved[:, None], latents[selected, index], best_latent)
            best_raw = torch.where(improved[:, None], states[selected, index], best_raw)
            chosen_index = torch.where(improved, start + index + 1, chosen_index)
            best_cost = torch.minimum(best_cost, chunk_cost)
        offsets = (best_raw[..., goal_dims] - flat_raw[..., goal_dims]).norm(dim=-1)
        info = dict(
            replay_initial_cost=initial_cost.mean(),
            replay_selected_cost=best_cost.mean(),
            replay_improved_fraction=(best_cost < initial_cost).float().mean(),
            replay_candidate_fallback_fraction=flat_query.new_tensor(fallback),
            replay_candidate_count=flat_query.new_tensor(count),
            replay_candidate_rejected_fraction=rejected / max(flat_raw.shape[0] * (count - 1), 1),
            replay_selected_goal_offset=offsets.mean(),
            replay_selected_goal_offset_max=offsets.max(),
            replay_goal_radius=flat_query.new_tensor(self.replay_goal_radius),
        )
        if probe_due:
            self._diagnostic_sink('replay_selection', int(self.replay_candidate_draws.item()), dict(
                raw_goals=flat_raw[query_indices],
                candidate_states=flat_candidates[query_indices],
                candidate_costs=torch.cat(logged_costs, -1) if logged_costs else initial_cost.new_empty((len(query_indices), 0)),
                candidate_valid=torch.cat(logged_valid, -1) if logged_valid else chosen_index.new_empty((len(query_indices), 0)),
                initial_cost=initial_cost[query_indices], selected_cost=best_cost[query_indices],
                selected_index=chosen_index[query_indices], selected_state=best_raw[query_indices],
                predicted_latent=flat_query[query_indices], sampling_seed=flat_raw.new_tensor(seed, dtype=torch.long),
            ))
        return best_latent.reshape_as(initial_latent).detach(), info

    def gather_raw_obs_goal_pairs(self, data: BatchData) -> Tuple[torch.Tensor, torch.Tensor]:
        obs = data.observations
        goal = torch.roll(data.next_observations, 1, dims=0)  # randomize :)
        if self.add_goal_as_future_state:
            # add future_observations
            goal = torch.stack([goal, data.future_observations], 0)
            obs = obs.expand_as(goal)
        return obs, goal

    def forward(self, actor: Actor, critic_batch_infos: Collection[CriticBatchInfo], data: BatchData, *,
                goal_set_distance: Optional[Any] = None,
                goal_set_distance_loss: Optional[Any] = None) -> LossResult:
        if goal_set_distance_loss is None and goal_set_distance is not None:
            raise RuntimeError('Goal-set model requires a goal-set objective')
        if goal_set_distance_loss is not None:
            is_learned = goal_set_distance_loss.implementation == 'learned'
            if is_learned != (goal_set_distance is not None):
                raise RuntimeError(
                    'Learned objectives require a model; direct objectives must not have one'
                )
        with torch.no_grad():
            use_sampled_goal_non_goal_latent = (
                actor.input_mode == 'split_latent'
                and (self.latent_goal_mode in ('min', 'max') or self.replay_goal_candidates > 0)
            )
            obs, goal, actor_obs_goal_critic_infos = self.gather_obs_goal_pairs(
                critic_batch_infos,
                data,
                goal_set_distance_loss=goal_set_distance_loss,
                use_sampled_goal_non_goal_latent=use_sampled_goal_non_goal_latent,
                reencode_full_targets=actor.input_mode in ('masked', 'goal_only'),
            )

        actor_input_critic_idx = None
        if actor.input_mode in ('latent', 'split_latent'):
            actor_input_critic_idx = 0
            actor_input_info = actor_obs_goal_critic_infos[actor_input_critic_idx]
            obs = actor_input_info.zo.detach()
            if actor.input_mode == 'split_latent':
                if not isinstance(actor_input_info.critic.encoder, SplitEncoder):
                    raise RuntimeError('split_latent actor input requires SplitEncoder')
                goal, _non_goal = actor_input_info.critic.encoder.split_latent(
                    actor_input_info.zg.detach()
                )
            else:
                goal = actor_input_info.zg.detach()

        actor_distn = actor(obs, goal)
        action = actor_distn.rsample()
        info: Dict[str, torch.Tensor] = {}

        dists: List[torch.Tensor] = []
        set_diagnostics: List[Dict[str, torch.Tensor]] = []
        direct_candidates = None
        direct_candidate_mask = None
        replay_raw_goals = None
        if self.replay_goal_candidates:
            if goal_set_distance_loss is not None:
                raise RuntimeError('Replay selection cannot be combined with goal-set objectives')
            _, replay_raw_goals = self.gather_raw_obs_goal_pairs(data)
        if (goal_set_distance_loss is not None
                and goal_set_distance_loss.implementation == 'direct'):
            with torch.no_grad():
                _, raw_goal = self.gather_raw_obs_goal_pairs(data)
                direct_candidates, direct_candidate_mask = (
                    goal_set_distance_loss._sample_goal_condition_states_with_mask(raw_goal)
                )
            info['candidate_fallback_fraction'] = torch.as_tensor(
                goal_set_distance_loss.last_candidate_fallback_fraction,
                device=action.device,
            )
            info['candidate_count_mean'] = torch.as_tensor(
                goal_set_distance_loss.last_candidate_count_mean,
                device=action.device,
            )
            info['candidate_count_min'] = torch.as_tensor(
                goal_set_distance_loss.last_candidate_count_min,
                device=action.device,
            )
            info['candidate_shortfall_fraction'] = torch.as_tensor(
                goal_set_distance_loss.last_candidate_shortfall_fraction,
                device=action.device,
            )

        if actor_input_critic_idx is not None:
            info['latent_input_critic_idx'] = torch.as_tensor(
                actor_input_critic_idx, device=action.device
            )

        for idx, actor_obs_goal_critic_info in enumerate(actor_obs_goal_critic_infos):
            critic = actor_obs_goal_critic_info.critic
            with critic.requiring_grad(False):
                zp = critic.predict_next_latent(
                    actor_obs_goal_critic_info.zo.detach(), action
                )
                if self.replay_goal_candidates:
                    selected_goal, selection_info = self._select_replay_goal(
                        critic, zp, replay_raw_goals, actor_obs_goal_critic_info.zg,
                    )
                    dist = critic.quasimetric_model(zp, selected_goal)
                    for key, value in selection_info.items():
                        info[f'{key}_{idx:02d}'] = value
                elif self.latent_goal_mode in ('min', 'max'):
                    completed_goal, completion_diagnostics = self._optimize_latent_goal(
                        critic, zp, actor_obs_goal_critic_info.zg
                    )
                    dist = critic.quasimetric_model(zp, completed_goal)
                    for key, value in completion_diagnostics.items():
                        info[f'{key}_{idx:02d}'] = value
                    if self.latent_goal_probe_replay:
                        raw_obs, raw_goals = self.gather_raw_obs_goal_pairs(data)
                        if self._last_search_probe is not None:
                            query = self._last_search_probe['query_indices']
                            self._last_search_probe.update(
                                raw_current_states=raw_obs.reshape(-1, raw_obs.shape[-1])[query],
                                actions=action.detach().reshape(-1, action.shape[-1])[query],
                            )
                        support = self._completion_support(
                            critic, raw_goals, actor_obs_goal_critic_info.zg, completed_goal,
                        )
                        for key, value in support.items():
                            info[f'{key}_{idx:02d}'] = value
                elif goal_set_distance is None:
                    if direct_candidates is None:
                        dist = critic.quasimetric_model(
                            zp, actor_obs_goal_critic_info.zg.detach()
                        )
                    else:
                        with torch.no_grad():
                            flat_candidates = direct_candidates.reshape(
                                -1, direct_candidates.shape[-1]
                            )
                            candidate_latents = critic.encoder(flat_candidates).reshape(
                                *direct_candidates.shape[:-1], -1
                            )
                        dist, aggregation_diagnostics = (
                            goal_set_distance_loss.direct_actor_distance_with_info(
                                critic,
                                zp,
                                candidate_latents,
                                direct_candidate_mask,
                            )
                        )
                        set_diagnostics.append(aggregation_diagnostics)
                        for key, value in aggregation_diagnostics.items():
                            info[f'{key}_{idx:02d}'] = value.mean()
                else:
                    with goal_set_distance.requiring_grad(False):
                        dist = goal_set_distance(
                            idx, zp, actor_obs_goal_critic_info.zg.detach()
                        )
            info[f'dist_{idx:02d}'] = dist.mean()
            if isinstance(critic.encoder, SplitEncoder):
                for key, value in self._latent_part_diagnostics(
                        critic.encoder, actor_obs_goal_critic_info.zo.detach()).items():
                    info[f'{key}_{idx:02d}'] = value
            dists.append(dist)

        if set_diagnostics and set_diagnostics[0]:
            for key in set_diagnostics[0]:
                values = torch.stack([
                    diagnostics[key] for diagnostics in set_diagnostics
                ], dim=-1)
                info[key] = values.mean()
                if key == 'lme_hard_gap':
                    info['lme_hard_gap_max'] = values.max()

        max_dist = info['dist_max'] = torch.stack(dists, -1).max(-1).values.mean()
        loss = max_dist
        if self.target_entropy is not None:
            info['target_entropy'] = self.target_entropy
            entropy = info['entropy'] = actor_distn.entropy(
                num_samples=self.entropy_mc_samples
            ).mean()
            entropy_loss, alpha = self.adaptive_entropy_loss(entropy)
            info['entropy_alpha'] = alpha
            info['entropy_gap'] = self.target_entropy - entropy
            info['entropy_mc_samples'] = torch.as_tensor(
                self.entropy_mc_samples, device=entropy.device
            )
            loss += entropy_loss
        return LossResult(loss=loss, info=info)

    def extra_repr(self) -> str:
        return (
            f'add_goal_as_future_state={self.add_goal_as_future_state}, '
            f'target_entropy={self.target_entropy}, '
            f'entropy_mc_samples={self.entropy_mc_samples}, '
            f'latent_goal_mode={self.latent_goal_mode}, '
            f'latent_goal_steps={self.latent_goal_steps}, '
            f'latent_goal_optim={self.latent_goal_optim}, '
            f'latent_goal_lr={self.latent_goal_lr}, '
            f'latent_goal_keep_best={self.latent_goal_keep_best}, '
            f'latent_goal_search={self.latent_goal_search}, '
            f'latent_goal_residual_radius={self.latent_goal_residual_radius}'
            f', replay_goal_candidates={self.replay_goal_candidates}, '
            f'replay_goal_radius={self.replay_goal_radius}'
        )

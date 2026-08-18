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
        latent_goal_search: str = attrs.field(
            default='direct', validator=attrs.validators.in_(LATENT_GOAL_SEARCHES)
        )
        latent_goal_residual_radius: float = attrs.field(
            default=1.0, validator=attrs.validators.gt(0)
        )

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
                latent_goal_search=self.latent_goal_search,
                latent_goal_residual_radius=self.latent_goal_residual_radius,
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
                 latent_goal_search: str = 'direct',
                 latent_goal_residual_radius: float = 1.0):
        super().__init__()
        if not env_spec.action_dtype.is_floating_point:
            raise RuntimeError(
                'Discrete action spaces do not support optimizing actor by backpropagation through the critic. '
                'Set agent.actor=null to turn of actor optimization.'
            )

        self.add_goal_as_future_state = add_goal_as_future_state
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
        self.latent_goal_search = latent_goal_search
        self.latent_goal_residual_radius = latent_goal_residual_radius
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
            use_sampled_goal_non_goal_latent: bool = False) -> Tuple[
                torch.Tensor, torch.Tensor, Collection[ActorObsGoalCriticInfo]]:
        r"""
        Returns (
            obs,
            goal,
            [ (latent_obs, latent_goal) for each critic ],
        )
        """

        obs, goal = self.gather_raw_obs_goal_pairs(data)
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
            elif goal_set_distance_loss is None:
                zo = critic_batch_info.zx
                zg = torch.roll(critic_batch_info.zy, 1, dims=0)  # randomize in the same way:)
                if self.add_goal_as_future_state:
                    # add future_observations
                    zg = torch.stack([
                        zg,
                        critic_batch_info.critic.encoder(data.future_observations),
                    ], 0)
            else:
                # In joint training, the critic optimizer has already stepped
                # since critic_batch_info was created. Re-encode both sides so
                # the goal-set objective never mixes stale and fresh latents.
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
                and self.latent_goal_mode in ('min', 'max')
            )
            obs, goal, actor_obs_goal_critic_infos = self.gather_obs_goal_pairs(
                critic_batch_infos,
                data,
                goal_set_distance_loss=goal_set_distance_loss,
                use_sampled_goal_non_goal_latent=use_sampled_goal_non_goal_latent,
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
                if self.latent_goal_mode in ('min', 'max'):
                    completed_goal, completion_diagnostics = self._optimize_latent_goal(
                        critic, zp, actor_obs_goal_critic_info.zg
                    )
                    dist = critic.quasimetric_model(zp, completed_goal)
                    for key, value in completion_diagnostics.items():
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
        )

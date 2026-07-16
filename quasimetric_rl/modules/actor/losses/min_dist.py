from typing import *

import attrs

import torch
import torch.nn as nn

from ....data import BatchData, EnvSpec

from ...utils import LatentTensor, LossResult, grad_mul
from ..model import Actor
from ...quasimetric_critic import QuasimetricCritic, CriticBatchInfo

from . import ActorLossBase



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

        # If set, in addition to use random goals, also use future state in the same trajectory as goals.
        # We enable this for online settings, following Contrastive RL.
        add_goal_as_future_state: bool = True

        def make(self, env_spec: EnvSpec) -> 'MinDistLoss':
            return MinDistLoss(
                env_spec=env_spec,
                adaptive_entropy_regularizer=self.adaptive_entropy_regularizer,
                add_goal_as_future_state=self.add_goal_as_future_state,
            )

    add_goal_as_future_state: bool
    raw_entropy_weight: Optional[nn.Parameter]  # set if using adaptive entropy regularization
    target_entropy: Optional[float] = None  # set if using adaptive entropy regularization

    def __init__(self, *, env_spec: EnvSpec,
                 adaptive_entropy_regularizer: bool,
                 add_goal_as_future_state: bool):
        super().__init__()
        if not env_spec.action_dtype.is_floating_point:
            raise RuntimeError(
                'Discrete action spaces do not support optimizing actor by backpropagation through the critic. '
                'Set agent.actor=null to turn of actor optimization.'
            )

        self.add_goal_as_future_state = add_goal_as_future_state
        if adaptive_entropy_regularizer:
            self.raw_entropy_weight = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
            self.target_entropy = env_spec.get_action_entropy_reg_target()
        else:
            self.register_parameter('raw_entropy_weight', None)
            self.target_entropy = None

    def gather_obs_goal_pairs(self, critic_batch_infos: Collection[CriticBatchInfo], data: BatchData,
                               *, goal_set_distance_loss: Optional[Any] = None) -> Tuple[
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
            if goal_set_distance_loss is None:
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
            obs, goal, actor_obs_goal_critic_infos = self.gather_obs_goal_pairs(
                critic_batch_infos, data, goal_set_distance_loss=goal_set_distance_loss)

        if actor.input_mode == 'latent':
            actor_input_critic_idx = torch.randint(
                len(actor_obs_goal_critic_infos),
                (),
                device=data.observations.device,
            ).item()
            actor_input_info = actor_obs_goal_critic_infos[actor_input_critic_idx]
            obs = actor_input_info.zo.detach()
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

        if actor.input_mode == 'latent':
            info['latent_input_critic_idx'] = torch.as_tensor(actor_input_critic_idx, device=action.device)

        for idx, actor_obs_goal_critic_info in enumerate(actor_obs_goal_critic_infos):
            critic = actor_obs_goal_critic_info.critic
            with critic.requiring_grad(False):
                zp = critic.latent_dynamics(actor_obs_goal_critic_info.zo.detach(), action)
                if goal_set_distance is None:
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
                        dist, diagnostics = goal_set_distance_loss.direct_actor_distance_with_info(
                            critic, zp, candidate_latents, direct_candidate_mask
                        )
                        set_diagnostics.append(diagnostics)
                        for key, value in diagnostics.items():
                            info[f'{key}_{idx:02d}'] = value.mean()
                else:
                    # Freeze GSD parameters while retaining the action -> zp ->
                    # GSD gradient used to optimize the actor.
                    with goal_set_distance.requiring_grad(False):
                        dist = goal_set_distance(idx, zp, actor_obs_goal_critic_info.zg.detach())
            info[f'dist_{idx:02d}'] = dist.mean()
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
        loss = max_dist  # pick the most pessimistic

        if self.target_entropy is not None:
            # add entropy regularization

            info['target_entropy'] = self.target_entropy
            entropy = info['entropy'] = actor_distn.entropy().mean()

            alpha = info['entropy_alpha'] = grad_mul(self.raw_entropy_weight.exp(), -1)  # minimax :)
            loss += alpha * (self.target_entropy - entropy)

        return LossResult(loss=loss, info=info)

    def extra_repr(self) -> str:
        return f"add_goal_as_future_state={self.add_goal_as_future_state}, target_entropy={self.target_entropy}"

from typing import *
from typing import Any, Mapping

import attrs

import torch

from . import actor, quasimetric_critic, gcrl_baselines
from . import goal_set_distance as goal_set_distance_module
from .quasimetric_critic.models.encoder import SplitEncoder

from ..data import EnvSpec, BatchData
from .utils import LossResult, Module, InfoT
from ..utils import TimingProfiler


class QRLAgent(Module):
    actor: Optional[actor.Actor]
    critics: Collection[quasimetric_critic.QuasimetricCritic]
    goal_set_distance: Optional[goal_set_distance_module.GoalSetDistance]
    goal_set_dims: Optional[Tuple[int, ...]]

    def __init__(self, actor: Optional['actor.Actor'],
                 critics: Collection[quasimetric_critic.QuasimetricCritic],
                 goal_set_distance: Optional['goal_set_distance_module.GoalSetDistance'] = None,
                 goal_set_dims: Optional[Tuple[int, ...]] = None):
        super().__init__()
        self.add_module('actor', actor)
        self.critics = torch.nn.ModuleList(critics)
        self.add_module('goal_set_distance', goal_set_distance)
        self.goal_set_dims = goal_set_dims

    def act(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.distributions.Distribution:
        if self.actor is None:
            raise RuntimeError("This agent has no actor.")
        if self.goal_set_dims is not None:
            padded_goal = torch.zeros_like(goal)
            padded_goal[..., list(self.goal_set_dims)] = goal[..., list(self.goal_set_dims)]
            goal = padded_goal
        if self.actor.input_mode == 'raw':
            return self.actor(obs, goal)
        if self.actor.input_mode not in ('latent', 'split_latent'):
            raise ValueError(f"Unknown actor input_mode: {self.actor.input_mode!r}")
        critic = self.critics[0]
        with torch.no_grad():
            z_obs = critic.encoder(obs)
            if self.actor.input_mode == 'split_latent':
                if not isinstance(critic.encoder, SplitEncoder):
                    raise RuntimeError('split_latent actor input requires SplitEncoder')
                z_goal = critic.encoder.encode_goal_part(goal)
            else:
                z_goal = critic.encoder.encode_actor_goal(goal)
        return self.actor(z_obs, z_goal)


class QRLLosses(Module):
    actor_loss: Optional[actor.ActorLosses]
    critic_losses: Collection[quasimetric_critic.QuasimetricCriticLosses]
    goal_set_distance_loss: Optional[goal_set_distance_module.GoalSetDistanceLoss]
    profiler: Optional[TimingProfiler]

    def __init__(self, actor_loss: Optional['actor.ActorLosses'],
                 critic_losses: Collection[quasimetric_critic.QuasimetricCriticLosses],
                 goal_set_distance_loss: Optional['goal_set_distance_module.GoalSetDistanceLoss'] = None,
                 profiler: Optional[TimingProfiler] = None):
        super().__init__()
        self.add_module('actor_loss', actor_loss)
        self.critic_losses = torch.nn.ModuleList(critic_losses)
        self.add_module('goal_set_distance_loss', goal_set_distance_loss)
        self.profiler = profiler
        if self.actor_loss is not None:
            self.actor_loss.profiler = profiler
        if self.goal_set_distance_loss is not None:
            self.goal_set_distance_loss.profiler = profiler
        for critic_loss in self.critic_losses:
            critic_loss.profiler = profiler

    def _record(self, name: str):
        if self.profiler is None:
            import contextlib
            return contextlib.nullcontext()
        return self.profiler.record(name)

    def _make_critic_batch_infos(self, agent: QRLAgent, data: BatchData) -> List[quasimetric_critic.CriticBatchInfo]:
        # compute CriticBatchInfo
        critic_batch_infos: List[quasimetric_critic.CriticBatchInfo] = []
        for idx, (critic, critic_loss) in enumerate(zip(agent.critics, self.critic_losses)):
            with self._record(f'train/critic_{idx:02d}/encoder'):
                zx, zy = critic.encoder(torch.stack([data.observations, data.next_observations], dim=0)).unbind(0)
            critic_batch_info = quasimetric_critic.CriticBatchInfo(
                critic=critic,
                zx=zx,
                zy=zy,
                px=critic.quasimetric_model.project(zx),
                py=critic.quasimetric_model.project(zy),
            )
            critic_batch_infos.append(critic_batch_info)
        return critic_batch_infos

    def forward(self, agent: QRLAgent, data: BatchData, *, optimize: bool = True,
                phase: str = 'all') -> LossResult:
        if phase not in ('all', 'critic', 'latent_dynamics', 'actor', 'goal_set_distance'):
            raise ValueError(f"Unknown training phase: {phase!r}")
        if (phase == 'goal_set_distance'
                and self.goal_set_distance_loss is not None
                and self.goal_set_distance_loss.implementation == 'direct'):
            raise ValueError('Direct goal-set objectives have no standalone model-training phase')

        needs_critic_grad = phase in ('all', 'critic')
        if needs_critic_grad:
            critic_batch_infos = self._make_critic_batch_infos(agent, data)
        else:
            with torch.no_grad():
                critic_batch_infos = self._make_critic_batch_infos(agent, data)
        loss_results: Dict[str, LossResult] = {}

        if phase in ('all', 'critic', 'latent_dynamics'):
            critic_phase = phase if phase in ('critic', 'latent_dynamics') else 'all'
            for idx, (critic_loss, critic_batch_info) in enumerate(zip(self.critic_losses, critic_batch_infos)):
                with self._record(f'train/critic_{idx:02d}/total'):
                    loss_results[f"critic_{idx:02d}"] = critic_loss(
                        data,
                        critic_batch_info,
                        optimize=optimize,
                        phase=critic_phase,
                    )

        if (phase in ('all', 'goal_set_distance')
                and self.goal_set_distance_loss is not None
                and self.goal_set_distance_loss.implementation == 'learned'):
            if agent.goal_set_distance is None:
                raise RuntimeError("Learned goal-set objective has no GoalSetDistance model")
            with self._record('train/goal_set_distance/total'):
                loss_results['goal_set_distance'] = self.goal_set_distance_loss(
                    agent.goal_set_distance,
                    critic_batch_infos,
                    data,
                    optimize=optimize,
                )

        if phase in ('all', 'actor') and self.actor_loss is not None:
            with self._record('train/actor/total'):
                loss_results['actor'] = self.actor_loss(
                    agent.actor,
                    critic_batch_infos,
                    data,
                    goal_set_distance=agent.goal_set_distance,
                    goal_set_distance_loss=self.goal_set_distance_loss,
                    optimize=optimize,
                )

        return LossResult.combine(loss_results)

    # for type hints
    def __call__(self, agent: QRLAgent, data: BatchData, *, optimize: bool = True,
                 phase: str = 'all') -> LossResult:
        return super().__call__(agent, data, optimize=optimize, phase=phase)

    def state_dict(self):
        optim_scheds = {}
        if self.actor_loss is not None:
            optim_scheds['actor'] = dict(
                actor_optim=self.actor_loss.actor_optim.state_dict(),
                actor_sched=self.actor_loss.actor_sched.state_dict(),
                entropy_weight_optim=self.actor_loss.entropy_weight_optim.state_dict(),
                entropy_weight_sched=self.actor_loss.entropy_weight_sched.state_dict(),
            )
        if self.goal_set_distance_loss is not None:
            goal_set_state = dict(
                candidate_rng=self.goal_set_distance_loss.candidate_rng_state_dict(),
            )
            if (self.goal_set_distance_loss.optim is not None
                    and self.goal_set_distance_loss.sched is not None):
                goal_set_state.update(
                    optim=self.goal_set_distance_loss.optim.state_dict(),
                    sched=self.goal_set_distance_loss.sched.state_dict(),
                )
            optim_scheds['goal_set_distance'] = goal_set_state
        for idx, critic_loss in enumerate(self.critic_losses):
            critic_optim_scheds = dict(
                critic_optim=critic_loss.critic_optim.state_dict(),
                critic_sched=critic_loss.critic_sched.state_dict(),
                lagrange_mult_optim=critic_loss.lagrange_mult_optim.state_dict(),
                lagrange_mult_sched=critic_loss.lagrange_mult_sched.state_dict(),
            )
            if critic_loss.latent_dynamics_optim is not None:
                critic_optim_scheds.update(
                    latent_dynamics_optim=critic_loss.latent_dynamics_optim.state_dict(),
                    latent_dynamics_sched=critic_loss.latent_dynamics_sched.state_dict(),
                )
            optim_scheds[f"critic_{idx:02d}"] = critic_optim_scheds
        return dict(
            module=super().state_dict(),
            optim_scheds=optim_scheds,
        )

    def load_state_dict(self, state_dict: Mapping[str, Any]):
        super().load_state_dict(state_dict['module'])
        optim_scheds = state_dict['optim_scheds']
        if self.actor_loss is not None:
            self.actor_loss.actor_optim.load_state_dict(optim_scheds['actor']['actor_optim'])
            self.actor_loss.actor_sched.load_state_dict(optim_scheds['actor']['actor_sched'])
            self.actor_loss.entropy_weight_optim.load_state_dict(optim_scheds['actor']['entropy_weight_optim'])
            self.actor_loss.entropy_weight_sched.load_state_dict(optim_scheds['actor']['entropy_weight_sched']),
        if self.goal_set_distance_loss is not None and 'goal_set_distance' in optim_scheds:
            goal_set_state = optim_scheds['goal_set_distance']
            if (self.goal_set_distance_loss.optim is not None
                    and self.goal_set_distance_loss.sched is not None
                    and 'optim' in goal_set_state):
                self.goal_set_distance_loss.optim.load_state_dict(goal_set_state['optim'])
                self.goal_set_distance_loss.sched.load_state_dict(goal_set_state['sched'])
            self.goal_set_distance_loss.load_candidate_rng_state_dict(
                goal_set_state.get('candidate_rng')
            )
        for idx, critic_loss in enumerate(self.critic_losses):
            critic_loss.critic_optim.load_state_dict(optim_scheds[f"critic_{idx:02d}"]['critic_optim'])
            critic_loss.critic_sched.load_state_dict(optim_scheds[f"critic_{idx:02d}"]['critic_sched'])
            critic_loss.lagrange_mult_optim.load_state_dict(optim_scheds[f"critic_{idx:02d}"]['lagrange_mult_optim'])
            critic_loss.lagrange_mult_sched.load_state_dict(optim_scheds[f"critic_{idx:02d}"]['lagrange_mult_sched'])
            if critic_loss.latent_dynamics_optim is not None:
                critic_optim_scheds = optim_scheds[f"critic_{idx:02d}"]
                if 'latent_dynamics_optim' in critic_optim_scheds:
                    critic_loss.latent_dynamics_optim.load_state_dict(critic_optim_scheds['latent_dynamics_optim'])
                    critic_loss.latent_dynamics_sched.load_state_dict(critic_optim_scheds['latent_dynamics_sched'])


@attrs.define(kw_only=True)
class QRLConf:
    algorithm: str = attrs.field(
        default='qrl',
        validator=attrs.validators.in_(('qrl', *gcrl_baselines.BASELINE_ALGORITHMS)),
    )
    baselines: 'gcrl_baselines.GCRLBaselinesConf' = gcrl_baselines.GCRLBaselinesConf()
    # Informational label populated by reusable model-size presets.
    model_size: Optional[str] = attrs.field(
        default=None,
        validator=attrs.validators.optional(
            attrs.validators.in_((
                'QRL-S', 'QRL-M', 'QRL-L',
                'GO-QRL-S', 'GO-QRL-M', 'GO-QRL-L',
                'TD-InfoNCE-M', 'CRL-M', 'GCBC-M', 'C-Learning-M',
            ))
        ),
    )
    actor: Optional['actor.ActorConf'] = actor.ActorConf()
    quasimetric_critic: 'quasimetric_critic.QuasimetricCriticConf' = quasimetric_critic.QuasimetricCriticConf()
    goal_set_distance: 'goal_set_distance_module.GoalSetDistanceConf' = (
        goal_set_distance_module.GoalSetDistanceConf()
    )
    num_critics: int = attrs.field(default=2, validator=attrs.validators.gt(0))
    training_schedule: str = attrs.field(
        default='joint',
        validator=attrs.validators.in_((
            'joint',
            'critic_then_dynamics_then_actor',
            'critic_then_dynamics_then_goal_set_distance_then_actor',
        )),
    )

    @property
    def required_transition_history_length(self) -> int:
        if self.algorithm != 'qrl':
            return 0
        latent_dynamics_conf = self.quasimetric_critic.model.latent_dynamics
        if latent_dynamics_conf.kind == 'transformer':
            return latent_dynamics_conf.history_length
        return 0

    def make(self, *, env_spec: EnvSpec, total_optim_steps: int,
             profiler: Optional[TimingProfiler] = None,
             goal_set_dims: Optional[Tuple[int, ...]] = None) -> Tuple[QRLAgent, QRLLosses]:
        if self.algorithm != 'qrl':
            if goal_set_dims is None:
                raise ValueError(
                    f'agent.algorithm={self.algorithm} requires registered goal dimensions'
                )
            return self.baselines.make(
                self.algorithm,
                env_spec=env_spec,
                goal_dims=goal_set_dims,
            )
        encoder_conf = self.quasimetric_critic.model.encoder
        encoder_conf.resolve_split_parameterization(env_spec=env_spec)
        latent_goal_mode = (
            'none'
            if self.actor is None
            else self.actor.losses.min_dist.latent_goal_mode
        )
        actor_input_mode = None if self.actor is None else self.actor.model.input_mode
        if actor_input_mode in ('latent', 'split_latent') and self.num_critics != 1:
            raise ValueError(
                f'agent.actor.model.input_mode={actor_input_mode} requires agent.num_critics=1; '
                'the latent actor must use a single, stable critic encoder.'
            )
        if actor_input_mode == 'split_latent' and encoder_conf.kind != 'split':
            raise ValueError('agent.actor.model.input_mode=split_latent requires encoder.kind=split')
        if encoder_conf.kind == 'split' and self.goal_set_distance.enabled:
            raise ValueError('SplitEncoder cannot be combined with GoalSetDistance')
        if latent_goal_mode != 'none':
            if self.num_critics != 1:
                raise ValueError('Latent goal optimization requires agent.num_critics=1')
            if self.actor.model.input_mode not in ('latent', 'split_latent'):
                raise ValueError(
                    'Latent goal optimization requires a latent actor input mode'
                )
            if encoder_conf.kind != 'split':
                raise ValueError('Latent goal optimization requires encoder.kind=split')
        if self.goal_set_distance.enabled:
            implementation = self.goal_set_distance.losses.implementation
            gsd_schedule = 'critic_then_dynamics_then_goal_set_distance_then_actor'
            if implementation == 'learned' and self.training_schedule == 'critic_then_dynamics_then_actor':
                raise ValueError(
                    'Learned goal-set objectives require joint training or an explicit '
                    'goal-set-distance training phase'
                )
            if implementation == 'direct' and self.training_schedule == gsd_schedule:
                raise ValueError('Direct goal-set objectives have no goal-set-distance training phase')
            if implementation == 'direct' and self.actor is None:
                raise ValueError('Direct goal-set objectives require an actor')
        if self.actor is None:
            actor = actor_losses = None
        else:
            actor_goal_latent_size = (
                encoder_conf.goal_latent_size
                if self.actor.model.input_mode == 'split_latent'
                else encoder_conf.latent_size
            )
            actor, actor_losses = self.actor.make(
                env_spec=env_spec,
                total_optim_steps=total_optim_steps,
                latent_size=encoder_conf.latent_size,
                goal_latent_size=actor_goal_latent_size,
            )
        critics, critic_losses = zip(*[
            self.quasimetric_critic.make(env_spec=env_spec, total_optim_steps=total_optim_steps)
            for _ in range(self.num_critics)
        ])
        effective_goal_set_dims = None
        if self.goal_set_distance.enabled:
            effective_goal_set_dims = (
                self.goal_set_distance.losses.goal_dims
                if self.goal_set_distance.losses.goal_dims is not None
                else goal_set_dims
            )
        goal_set_distance, goal_set_distance_loss = self.goal_set_distance.make(
            env_spec=env_spec,
            total_optim_steps=total_optim_steps,
            latent_size=self.quasimetric_critic.model.encoder.latent_size,
            num_critics=self.num_critics,
            goal_dims=effective_goal_set_dims,
        )
        return QRLAgent(
            actor=actor,
            critics=critics,
            goal_set_distance=goal_set_distance,
            goal_set_dims=effective_goal_set_dims,
        ), QRLLosses(
            actor_loss=actor_losses,
            critic_losses=critic_losses,
            goal_set_distance_loss=goal_set_distance_loss,
            profiler=profiler)

__all__ = [
    'QRLAgent', 'QRLLosses', 'QRLConf', 'InfoT',
    'gcrl_baselines',
]

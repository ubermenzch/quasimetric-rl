from typing import *
from typing import Any, Mapping

import attrs

import torch

from . import actor, quasimetric_critic
from . import goal_set_distance as goal_set_distance_module

from ..data import EnvSpec, BatchData
from .utils import LossResult, Module, InfoT
from ..utils import TimingProfiler


class QRLAgent(Module):
    actor: Optional[actor.Actor]
    critics: Collection[quasimetric_critic.QuasimetricCritic]
    goal_set_distance: Optional[goal_set_distance_module.GoalSetDistance]

    def __init__(self, actor: Optional['actor.Actor'],
                 critics: Collection[quasimetric_critic.QuasimetricCritic],
                 goal_set_distance: Optional['goal_set_distance_module.GoalSetDistance'] = None):
        super().__init__()
        self.add_module('actor', actor)
        self.critics = torch.nn.ModuleList(critics)
        self.add_module('goal_set_distance', goal_set_distance)

    def act(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.distributions.Distribution:
        if self.actor is None:
            raise RuntimeError("This agent has no actor.")
        if self.actor.input_mode == 'raw':
            return self.actor(obs, goal)
        if self.actor.input_mode != 'latent':
            raise ValueError(f"Unknown actor input_mode: {self.actor.input_mode!r}")
        critic_idx = torch.randint(len(self.critics), (), device=obs.device).item()
        critic = self.critics[critic_idx]
        with torch.no_grad():
            z_obs, z_goal = critic.encoder(torch.stack([obs, goal], dim=0)).unbind(0)
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

        if phase in ('all', 'goal_set_distance') and self.goal_set_distance_loss is not None:
            if agent.goal_set_distance is None:
                raise RuntimeError("Goal-set distance loss is enabled but agent.goal_set_distance is None")
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
            optim_scheds['goal_set_distance'] = dict(
                optim=self.goal_set_distance_loss.optim.state_dict(),
                sched=self.goal_set_distance_loss.sched.state_dict(),
            )
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
            self.goal_set_distance_loss.optim.load_state_dict(optim_scheds['goal_set_distance']['optim'])
            self.goal_set_distance_loss.sched.load_state_dict(optim_scheds['goal_set_distance']['sched'])
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
        latent_dynamics_conf = self.quasimetric_critic.model.latent_dynamics
        if latent_dynamics_conf.kind == 'transformer':
            return latent_dynamics_conf.history_length
        return 0

    def make(self, *, env_spec: EnvSpec, total_optim_steps: int,
             profiler: Optional[TimingProfiler] = None) -> Tuple[QRLAgent, QRLLosses]:
        if self.actor is None:
            actor = actor_losses = None
        else:
            actor, actor_losses = self.actor.make(
                env_spec=env_spec,
                total_optim_steps=total_optim_steps,
                latent_size=self.quasimetric_critic.model.encoder.latent_size,
            )
        critics, critic_losses = zip(*[
            self.quasimetric_critic.make(env_spec=env_spec, total_optim_steps=total_optim_steps)
            for _ in range(self.num_critics)
        ])
        goal_set_distance, goal_set_distance_loss = self.goal_set_distance.make(
            env_spec=env_spec,
            total_optim_steps=total_optim_steps,
            latent_size=self.quasimetric_critic.model.encoder.latent_size,
            num_critics=self.num_critics,
        )
        return QRLAgent(
            actor=actor,
            critics=critics,
            goal_set_distance=goal_set_distance,
        ), QRLLosses(
            actor_loss=actor_losses,
            critic_losses=critic_losses,
            goal_set_distance_loss=goal_set_distance_loss,
            profiler=profiler)

__all__ = ['QRLAgent', 'QRLLosses', 'QRLConf', 'InfoT']

from typing import *

import abc
import contextlib
import itertools
import attrs

import torch

from ....data import BatchData
from ...utils import LossBase, LossResult, LatentTensor
from ..models import QuasimetricCritic
from ...optim import OptimWrapper, AdamWSpec


@attrs.define(kw_only=True)
class CriticBatchInfo:
    r"""
    All critic outputs needed to compute losses for a single critic over a batch.
    """
    critic: QuasimetricCritic
    zx: LatentTensor
    zy: LatentTensor
    px: torch.Tensor
    py: torch.Tensor


class CriticLossBase(LossBase):
    @abc.abstractmethod
    def forward(self, data: BatchData, critic_batch_info: CriticBatchInfo) -> LossResult:
        pass

    # for type hints
    def __call__(self, data: BatchData, critic_batch_info: CriticBatchInfo, **kwargs) -> LossResult:
        return super().__call__(data, critic_batch_info, **kwargs)


from .global_push import GlobalPushLoss
from .local_constraint import LocalConstraintLoss
from .latent_dynamics import LatentDynamicsLoss


class QuasimetricCriticLosses(CriticLossBase):
    @attrs.define(kw_only=True)
    class Conf:
        global_push: GlobalPushLoss.Conf = GlobalPushLoss.Conf()
        local_constraint: LocalConstraintLoss.Conf = LocalConstraintLoss.Conf()
        latent_dynamics: LatentDynamicsLoss.Conf = LatentDynamicsLoss.Conf()

        separate_latent_dynamics: bool = False
        critic_optim: AdamWSpec.Conf = AdamWSpec.Conf(lr=1e-4)
        latent_dynamics_optim_uses_critic_optim: bool = True
        latent_dynamics_optim: AdamWSpec.Conf = AdamWSpec.Conf(lr=1e-4)
        lagrange_mult_optim: AdamWSpec.Conf = AdamWSpec.Conf(lr=1e-2)

        def make(self, critic: QuasimetricCritic, total_optim_steps: int) -> 'QuasimetricCriticLosses':
            return QuasimetricCriticLosses(
                critic,
                total_optim_steps=total_optim_steps,
                global_push=self.global_push.make(),
                local_constraint=self.local_constraint.make(),
                latent_dynamics=self.latent_dynamics.make(),
                separate_latent_dynamics=self.separate_latent_dynamics,
                critic_optim_spec=self.critic_optim.make(),
                latent_dynamics_optim_spec=(
                    self.critic_optim if self.latent_dynamics_optim_uses_critic_optim
                    else self.latent_dynamics_optim
                ).make(),
                lagrange_mult_optim_spec=self.lagrange_mult_optim.make(),
            )

    global_push: GlobalPushLoss
    local_constraint: LocalConstraintLoss
    latent_dynamics: LatentDynamicsLoss
    separate_latent_dynamics: bool

    critic_optim: OptimWrapper
    critic_sched: torch.optim.lr_scheduler._LRScheduler
    latent_dynamics_optim: Optional[OptimWrapper]
    latent_dynamics_sched: Optional[torch.optim.lr_scheduler._LRScheduler]
    lagrange_mult_optim: OptimWrapper
    lagrange_mult_sched: torch.optim.lr_scheduler._LRScheduler
    profiler: Optional[Any]

    def __init__(self, critic: QuasimetricCritic, *, total_optim_steps: int, global_push: GlobalPushLoss,
                 local_constraint: LocalConstraintLoss, latent_dynamics: LatentDynamicsLoss,
                 separate_latent_dynamics: bool, critic_optim_spec: AdamWSpec,
                 latent_dynamics_optim_spec: AdamWSpec, lagrange_mult_optim_spec: AdamWSpec):
        super().__init__()
        self.global_push = global_push
        self.local_constraint = local_constraint
        self.latent_dynamics = latent_dynamics
        self.separate_latent_dynamics = separate_latent_dynamics

        critic_params = (
            itertools.chain(critic.encoder.parameters(), critic.quasimetric_model.parameters())
            if separate_latent_dynamics else critic.parameters()
        )
        self.critic_optim, self.critic_sched = critic_optim_spec.create_optim_scheduler(
            critic_params, total_optim_steps)
        if separate_latent_dynamics:
            self.latent_dynamics_optim, self.latent_dynamics_sched = (
                latent_dynamics_optim_spec.create_optim_scheduler(critic.latent_dynamics.parameters(), total_optim_steps)
            )
        else:
            self.latent_dynamics_optim = None
            self.latent_dynamics_sched = None
        self.lagrange_mult_optim, self.lagrange_mult_sched = lagrange_mult_optim_spec.create_optim_scheduler(
            local_constraint.parameters(), total_optim_steps)
        assert len(list(local_constraint.parameters())) == 1
        self.profiler = None

    def _record(self, name: str):
        if self.profiler is None:
            return contextlib.nullcontext()
        return self.profiler.record(name)

    def _forward_critic_only(self, data: BatchData, critic_batch_info: CriticBatchInfo, *,
                             optimize: bool = True) -> LossResult:
        loss_results = {}
        with self.critic_optim.update_context(optimize=optimize), \
                self.lagrange_mult_optim.update_context(optimize=optimize):
            with self._record('train/critic/global_push'):
                loss_results['global_push'] = self.global_push(data, critic_batch_info)
            with self._record('train/critic/local_constraint'):
                loss_results['local_constraint'] = self.local_constraint(data, critic_batch_info)
            with self._record('train/critic/combine_losses'):
                result = LossResult.combine(loss_results)
            with self._record('train/critic/backward'):
                result.loss.backward()

        if optimize:
            with self._record('train/critic/scheduler_step'):
                self.critic_sched.step()
                self.lagrange_mult_sched.step()
        return result

    def _forward_latent_dynamics_only(self, data: BatchData, critic_batch_info: CriticBatchInfo, *,
                                      optimize: bool = True) -> LossResult:
        if not self.separate_latent_dynamics:
            raise RuntimeError(
                "phase='latent_dynamics' requires "
                "agent.quasimetric_critic.losses.separate_latent_dynamics=true"
            )
        assert self.latent_dynamics_optim is not None
        assert self.latent_dynamics_sched is not None
        with self.latent_dynamics_optim.update_context(optimize=optimize):
            with self._record('train/critic/latent_dynamics'):
                result = self.latent_dynamics(
                    data,
                    critic_batch_info,
                    detach_critic_outputs=True,
                )
            with self._record('train/critic/latent_dynamics_backward'):
                result.loss.backward()

        if optimize:
            with self._record('train/critic/latent_dynamics_scheduler_step'):
                self.latent_dynamics_sched.step()
        return result

    def forward(self, data: BatchData, critic_batch_info: CriticBatchInfo, *,
                optimize: bool = True, phase: str = 'all') -> LossResult:
        if phase == 'critic':
            return self._forward_critic_only(data, critic_batch_info, optimize=optimize)
        if phase == 'latent_dynamics':
            return self._forward_latent_dynamics_only(data, critic_batch_info, optimize=optimize)
        if phase != 'all':
            raise ValueError(f"Unknown critic training phase: {phase!r}")

        if not self.separate_latent_dynamics:
            loss_results = {}
            with self.critic_optim.update_context(optimize=optimize), \
                    self.lagrange_mult_optim.update_context(optimize=optimize):

                with self._record('train/critic/global_push'):
                    loss_results['global_push'] = self.global_push(data, critic_batch_info)
                with self._record('train/critic/local_constraint'):
                    loss_results['local_constraint'] = self.local_constraint(data, critic_batch_info)
                with self._record('train/critic/latent_dynamics'):
                    loss_results['latent_dynamics'] = self.latent_dynamics(data, critic_batch_info)
                with self._record('train/critic/combine_losses'):
                    result = LossResult.combine(loss_results)
                with self._record('train/critic/backward'):
                    result.loss.backward()

            if optimize:
                with self._record('train/critic/scheduler_step'):
                    self.critic_sched.step()
                    self.lagrange_mult_sched.step()
            return result

        assert self.latent_dynamics_optim is not None
        assert self.latent_dynamics_sched is not None
        loss_results = {}
        with self.critic_optim.update_context(optimize=optimize), \
                self.lagrange_mult_optim.update_context(optimize=optimize), \
                self.latent_dynamics_optim.update_context(optimize=optimize):
            with self._record('train/critic/global_push'):
                loss_results['global_push'] = self.global_push(data, critic_batch_info)
            with self._record('train/critic/local_constraint'):
                loss_results['local_constraint'] = self.local_constraint(data, critic_batch_info)
            with self._record('train/critic/combine_losses'):
                result = LossResult.combine(loss_results)
            with self._record('train/critic/backward'):
                result.loss.backward()

            with self._record('train/critic/latent_dynamics'):
                loss_results['latent_dynamics'] = self.latent_dynamics(
                    data,
                    critic_batch_info,
                    detach_critic_outputs=True,
                )
            with self._record('train/critic/latent_dynamics_backward'):
                loss_results['latent_dynamics'].loss.backward()

        if optimize:
            with self._record('train/critic/scheduler_step'):
                self.critic_sched.step()
                self.lagrange_mult_sched.step()
            with self._record('train/critic/latent_dynamics_scheduler_step'):
                self.latent_dynamics_sched.step()

        return LossResult.combine(loss_results)

    # for type hints
    def __call__(self, data: BatchData, critic_batch_info: CriticBatchInfo, *,
                 optimize: bool = True, phase: str = 'all') -> LossResult:
        return torch.nn.Module.__call__(self, data, critic_batch_info, optimize=optimize, phase=phase)

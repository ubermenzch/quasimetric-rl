from typing import *

import attrs

import torch

from ....data import BatchData

from ...utils import LossResult
from ..model import Actor
from ...quasimetric_critic import CriticBatchInfo

from . import ActorLossBase



class BCLoss(ActorLossBase):
    @attrs.define(kw_only=True)
    class Conf:
        # config / argparse uses this to specify behavior

        weight: float = attrs.field(default=0, validator=attrs.validators.ge(0))

        def make(self) -> 'BCLoss':
            return BCLoss(weight=self.weight)

    weight: float

    def __init__(self, *, weight: float):
        super().__init__()
        self.weight = weight

    def forward(self, actor: Actor, critic_batch_infos: Collection[CriticBatchInfo], data: BatchData) -> LossResult:
        if self.weight == 0:
            return LossResult(loss=0, info={})
        info = {}
        if actor.input_mode == 'latent':
            critic_batch_infos = list(critic_batch_infos)
            critic_idx = torch.randint(
                len(critic_batch_infos),
                (),
                device=data.observations.device,
            ).item()
            critic = critic_batch_infos[critic_idx].critic
            with torch.no_grad():
                obs, goal = critic.encoder(torch.stack([
                    data.observations,
                    data.future_observations,
                ], dim=0)).unbind(0)
            info['latent_input_critic_idx'] = torch.as_tensor(critic_idx, device=data.observations.device)
        else:
            obs = data.observations
            goal = data.future_observations
        actor_distn = actor(obs, goal)
        log_prob: torch.Tensor = actor_distn.log_prob(data.actions).mean()
        loss = -log_prob * self.weight
        info['log_prob'] = log_prob
        return LossResult(loss=loss, info=info)

    def extra_repr(self) -> str:
        return f"weight={self.weight:g}"

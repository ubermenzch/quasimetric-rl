from typing import *

import attrs

import torch
import torch.nn.functional as F

from ....data import BatchData

from ...utils import LatentTensor, LossResult
from ..models import QuasimetricCritic

from . import CriticLossBase, CriticBatchInfo



class GlobalPushLoss(CriticLossBase):
    @attrs.define(kw_only=True)
    class Conf:
        # config / argparse uses this to specify behavior

        # smaller => smoother loss
        softplus_beta: float = attrs.field(default=0.1, validator=attrs.validators.gt(0))

        # should be greater than most GT distances between sampled pairs
        softplus_offset: float = attrs.field(default=15, validator=attrs.validators.ge(0))

        def make(self) -> 'GlobalPushLoss':
            return GlobalPushLoss(
                softplus_beta=self.softplus_beta,
                softplus_offset=self.softplus_offset,
            )

    softplus_beta: float
    softplus_offset: float

    def __init__(self, *, softplus_beta: float, softplus_offset: float):
        super().__init__()
        self.softplus_beta = softplus_beta
        self.softplus_offset = softplus_offset
        self.goal_marker_dims = None
        self.goal_marker_fraction = 0.05

    def forward(self, data: BatchData, critic_batch_info: CriticBatchInfo) -> LossResult:
        # To randomly pair zx, zy, we just roll over zy by 1, because zx and zy
        # are latents of randomly ordered random batches.
        target_proj = torch.roll(critic_batch_info.py, 1, dims=0)
        marker_count = 0
        if self.goal_marker_dims is not None:
            from ...goal_marker import make_goal_marker
            marker_count = min(target_proj.shape[0], max(1, round(target_proj.shape[0] * self.goal_marker_fraction)))
            sampled_goals = torch.roll(data.next_observations, 1, dims=0)[:marker_count]
            markers = make_goal_marker(sampled_goals, self.goal_marker_dims)
            critic = critic_batch_info.critic
            marker_proj = critic.quasimetric_model.project(critic.encoder(markers))
            target_proj = torch.cat((marker_proj, target_proj[marker_count:]), dim=0)
        dists = critic_batch_info.critic.quasimetric_model.forward_projected(
            critic_batch_info.px, target_proj,
        )
        # Sec 3.2. Transform so that we penalize large distances less.
        tsfm_dist: torch.Tensor = F.softplus(self.softplus_offset - dists, beta=self.softplus_beta)
        tsfm_dist = tsfm_dist.mean()
        info = dict(dist=dists.mean(), tsfm_dist=tsfm_dist)
        if self.goal_marker_dims is not None:
            info['goal_marker_fraction'] = marker_count / target_proj.shape[0]
        return LossResult(loss=tsfm_dist, info=info)

    def extra_repr(self) -> str:
        return f"softplus_beta={self.softplus_beta:g}, softplus_offset={self.softplus_offset:g}"

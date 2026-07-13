from typing import *

import attrs
import contextlib

import torch
import torch.nn.functional as F

from ....data import BatchData

from ...utils import LossResult

from . import CriticLossBase, CriticBatchInfo



class LatentDynamicsLoss(CriticLossBase):
    r"""
    Section 3.4
    """

    @attrs.define(kw_only=True)
    class Conf:
        # config / argparse uses this to specify behavior

        weight: float = attrs.field(default=0.1, validator=attrs.validators.gt(0))
        distance: str = attrs.field(default='iqe', validator=attrs.validators.in_(('iqe', 'mse')))

        def make(self) -> 'LatentDynamicsLoss':
            return LatentDynamicsLoss(
                weight=self.weight,
                distance=self.distance,
            )

    weight: float
    distance: str

    def __init__(self, *, weight: float, distance: str):
        super().__init__()
        self.weight = weight
        self.distance = distance

    @contextlib.contextmanager
    def _quasimetric_model_requiring_grad(self, critic_batch_info: CriticBatchInfo, flag: bool):
        params = list(critic_batch_info.critic.quasimetric_model.parameters())
        requires_grad = [p.requires_grad for p in params]
        for p in params:
            p.requires_grad_(flag)
        try:
            yield
        finally:
            for p, rg in zip(params, requires_grad):
                p.requires_grad_(rg)

    def forward(self, data: BatchData, critic_batch_info: CriticBatchInfo, *,
                detach_critic_outputs: bool = False) -> LossResult:
        if (
            data.history_observations is not None
            and data.history_actions is not None
            and hasattr(critic_batch_info.critic.latent_dynamics, 'forward_sequence')
        ):
            return self.forward_sequence(data, critic_batch_info, detach_critic_outputs=detach_critic_outputs)

        zx = critic_batch_info.zx
        py = critic_batch_info.py
        if detach_critic_outputs:
            zx = zx.detach()
            py = py.detach()

        pred_zy = critic_batch_info.critic.latent_dynamics(zx, data.actions)
        grad_context = (
            self._quasimetric_model_requiring_grad(critic_batch_info, False)
            if detach_critic_outputs else contextlib.nullcontext()
        )
        with grad_context:
            dists = critic_batch_info.critic.quasimetric_model.forward_projected(
                critic_batch_info.critic.quasimetric_model.project(pred_zy),
                py,
                bidirectional=True,
            )
        sq_dists = dists.square().mean()
        target_zy = critic_batch_info.zy.detach() if detach_critic_outputs else critic_batch_info.zy
        mse = F.mse_loss(pred_zy, target_zy)
        loss_value = mse if self.distance == 'mse' else sq_dists

        dist_p2n, dist_n2p = dists.unbind(-1)
        return LossResult(
            loss=loss_value * self.weight,
            info=dict(
                distance_is_mse=float(self.distance == 'mse'),
                mse=mse,
                sq_dists=sq_dists,
                dist_p2n=dist_p2n.mean(),
                dist_n2p=dist_n2p.mean(),
            ),
        )

    def forward_sequence(self, data: BatchData, critic_batch_info: CriticBatchInfo, *,
                         detach_critic_outputs: bool = False) -> LossResult:
        assert data.history_observations is not None
        assert data.history_actions is not None
        critic = critic_batch_info.critic
        batch_size, obs_len = data.history_observations.shape[:2]
        hist_len = obs_len - 1
        z_history = critic.encoder(data.history_observations.flatten(0, 1)).unflatten(0, (batch_size, obs_len))
        z_context = z_history
        z_target = z_history[:, 1:]
        if detach_critic_outputs:
            z_context = z_context.detach()
            z_target = z_target.detach()

        pred_z = critic.latent_dynamics.forward_sequence(
            z_context,
            data.history_actions,
            data.history_mask,
        )
        grad_context = (
            self._quasimetric_model_requiring_grad(critic_batch_info, False)
            if detach_critic_outputs else contextlib.nullcontext()
        )
        with grad_context:
            pred_p = critic.quasimetric_model.project(pred_z.flatten(0, 1))
            target_p = critic.quasimetric_model.project(z_target.flatten(0, 1))
            dists = critic.quasimetric_model.forward_projected(
                pred_p,
                target_p,
                bidirectional=True,
            ).unflatten(0, (batch_size, hist_len))
        if data.history_mask is not None:
            mask = data.history_mask.to(dists.dtype)
            denom = mask.sum().clamp_min(1)
            sq_dists = (dists.square().sum(dim=-1) * mask).sum() / denom / 2
            dist_p2n = (dists[..., 0] * mask).sum() / denom
            dist_n2p = (dists[..., 1] * mask).sum() / denom
            mse = (
                (pred_z - z_target).square().sum(dim=-1) * mask
            ).sum() / denom / pred_z.shape[-1]
        else:
            sq_dists = dists.square().mean()
            dist_p2n, dist_n2p = dists.unbind(-1)
            dist_p2n = dist_p2n.mean()
            dist_n2p = dist_n2p.mean()
            mse = F.mse_loss(pred_z, z_target)
        loss_value = mse if self.distance == 'mse' else sq_dists
        return LossResult(
            loss=loss_value * self.weight,
            info=dict(
                distance_is_mse=float(self.distance == 'mse'),
                mse=mse,
                sq_dists=sq_dists,
                dist_p2n=dist_p2n,
                dist_n2p=dist_n2p,
            ),
        )

    def extra_repr(self) -> str:
        return f"weight={self.weight:g}, distance={self.distance}"

from typing import *

import attrs

import torch
import torch.nn as nn

from .encoder import Encoder, SplitEncoder
from .quasimetric_model import QuasimetricModel
from .latent_dynamics import LatentDynamics

from ...utils import Module

from ....data import EnvSpec


class QuasimetricCritic(Module):
    @attrs.define(kw_only=True)
    class Conf:
        # config / argparse uses this to specify behavior

        encoder: Encoder.Conf = Encoder.Conf()
        quasimetric_model: QuasimetricModel.Conf = QuasimetricModel.Conf()
        latent_dynamics: LatentDynamics.Conf = LatentDynamics.Conf()

        def make(self, *, env_spec: EnvSpec) -> 'QuasimetricCritic':
            encoder = self.encoder.make(
                env_spec=env_spec,
            )
            quasimetric_model = self.quasimetric_model.make(
                input_size=encoder.latent_size,
            )
            latent_dynamics = self.latent_dynamics.make(
                latent_size=encoder.latent_size,
                env_spec=env_spec,
            )
            return QuasimetricCritic(encoder, quasimetric_model, latent_dynamics)

    encoder: Union[Encoder, SplitEncoder]
    quasimetric_model: QuasimetricModel
    latent_dynamics: LatentDynamics

    raw_lagrange_multiplier: nn.Parameter  # for the QRL constrained optimization


    def __init__(self, encoder: Union[Encoder, SplitEncoder],
                 quasimetric_model: QuasimetricModel,
                 latent_dynamics: LatentDynamics):
        super().__init__()
        self.encoder = encoder
        self.quasimetric_model = quasimetric_model
        self.latent_dynamics = latent_dynamics

    def forward(self, x: torch.Tensor, y: torch.Tensor, *, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        # The basic interface is a V- or Q-function.
        zx = self.encoder(x)
        zy = self.encoder(y)
        if action is not None:
            zx = self.predict_next_latent(zx, action)
        return self.quasimetric_model(zx, zy)

    def predict_next_latent(
            self, zx: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.encoder.normalize_latent(self.latent_dynamics(zx, action))

    def predict_next_latent_sequence(
            self, z_history: torch.Tensor, action_history: torch.Tensor,
            history_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not hasattr(self.latent_dynamics, 'forward_sequence'):
            raise RuntimeError('Configured latent dynamics does not support sequences')
        predicted = self.latent_dynamics.forward_sequence(
            z_history,
            action_history,
            history_mask,
        )
        return self.encoder.normalize_latent(predicted)

    # for type hints
    def __call__(self, x: torch.Tensor, y: torch.Tensor, *, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        return super().__call__(x, y, action=action)

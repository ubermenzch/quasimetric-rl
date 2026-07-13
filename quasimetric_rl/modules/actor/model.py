from typing import *

import attrs

import torch
import torch.nn as nn

from ..utils import MLP

from ...data import EnvSpec
from ...data.env_spec.input_encoding import InputEncoding
from ...data.env_spec.act_distn import ActionOutputConverter


class Actor(nn.Module):
    @attrs.define(kw_only=True)
    class Conf:
        # config / argparse uses this to specify behavior

        arch: Tuple[int, ...] = (512, 512)
        input_mode: str = attrs.field(default='raw', validator=attrs.validators.in_(('raw', 'latent')))

        def make(self, *, env_spec: EnvSpec, latent_size: Optional[int] = None) -> 'Actor':
            return Actor(
                env_spec=env_spec,
                arch=self.arch,
                input_mode=self.input_mode,
                latent_size=latent_size,
            )

    observation_shape: torch.Size
    input_mode: str
    observation_encoding: InputEncoding
    backbone: MLP
    action_output: ActionOutputConverter

    def __init__(self, *, env_spec: EnvSpec, arch: Tuple[int, ...],
                 input_mode: str, latent_size: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.observation_shape = env_spec.observation_shape
        self.input_mode = input_mode
        self.observation_encoding = env_spec.make_observation_input()

        self.action_output = env_spec.make_action_output_distn()
        if input_mode == 'raw':
            backbone_input_size = self.observation_encoding.output_size * 2  # add goal
        elif input_mode == 'latent':
            if latent_size is None:
                raise ValueError("latent actor input_mode requires latent_size")
            backbone_input_size = latent_size * 2
        else:
            raise ValueError(f"Unknown actor input_mode: {input_mode!r}")
        self.backbone = MLP(backbone_input_size, self.action_output.input_size, hidden_sizes=arch,
                            zero_init_last_fc=True)

    def forward(self, o: torch.Tensor, g: torch.Tensor) -> torch.distributions.Distribution:
        if self.input_mode == 'raw':
            og = torch.stack([o, g], dim=-len(self.observation_shape) - 1)
            actor_input = self.observation_encoding(og).flatten(-2, -1)
        else:
            actor_input = torch.cat([o, g], dim=-1)
        return self.action_output(self.backbone(actor_input))

    # for type hint
    def __call__(self, o: torch.Tensor, g: torch.Tensor) -> torch.distributions.Distribution:
        return super().__call__(o, g)

    def extra_repr(self) -> str:
        return f"observation_shape={self.observation_shape}, input_mode={self.input_mode}"

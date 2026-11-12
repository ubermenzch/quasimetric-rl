from typing import *

import attrs

import torch
import torch.nn as nn

from ..utils import MLP_KINDS, make_mlp

from ...data import EnvSpec
from ...data.env_spec.input_encoding import InputEncoding
from ...data.env_spec.act_distn import ActionOutputConverter


RAW_ACTOR_INPUT_MODES = ('raw', 'masked', 'goal_only')
ACTOR_INPUT_MODES = (*RAW_ACTOR_INPUT_MODES, 'latent', 'split_latent')


class Actor(nn.Module):
    @attrs.define(kw_only=True)
    class Conf:
        # config / argparse uses this to specify behavior

        arch: Tuple[int, ...] = (512, 512)
        mlp_kind: str = attrs.field(
            default='plain', validator=attrs.validators.in_(MLP_KINDS)
        )
        residual_block_size: int = attrs.field(
            default=4, validator=attrs.validators.gt(0)
        )
        input_mode: str = attrs.field(
            default='raw', validator=attrs.validators.in_(ACTOR_INPUT_MODES)
        )
        goal_dims: Optional[Tuple[int, ...]] = attrs.field(
            default=None, converter=lambda dims: None if dims is None else tuple(dims),
        )

        def make(
                self, *, env_spec: EnvSpec, latent_size: Optional[int] = None,
                goal_latent_size: Optional[int] = None) -> 'Actor':
            return Actor(
                env_spec=env_spec,
                arch=self.arch,
                input_mode=self.input_mode,
                latent_size=latent_size,
                goal_latent_size=goal_latent_size,
                goal_dims=self.goal_dims,
                mlp_kind=self.mlp_kind,
                residual_block_size=self.residual_block_size,
            )

    observation_shape: torch.Size
    input_mode: str
    observation_encoding: InputEncoding
    backbone: nn.Module
    action_output: ActionOutputConverter

    def __init__(
            self, *, env_spec: EnvSpec, arch: Tuple[int, ...],
            input_mode: str, latent_size: Optional[int] = None,
            goal_latent_size: Optional[int] = None, mlp_kind: str = 'plain',
            residual_block_size: int = 4,
            goal_dims: Optional[Tuple[int, ...]] = None, **kwargs):
        super().__init__(**kwargs)
        self.observation_shape = env_spec.observation_shape
        self.input_mode = input_mode
        self.observation_encoding = env_spec.make_observation_input()

        self.action_output = env_spec.make_action_output_distn()
        if input_mode == 'raw':
            backbone_input_size = self.observation_encoding.output_size * 2  # add goal
        elif input_mode in ('masked', 'goal_only'):
            if len(self.observation_shape) != 1:
                raise ValueError('Masked and goal-only actors require vector states')
            state_size = self.observation_shape[0]
            if (not goal_dims or len(set(goal_dims)) != len(goal_dims)
                    or min(goal_dims) < 0 or max(goal_dims) >= state_size):
                raise ValueError('Masked and goal-only actors require valid goal_dims')
            self.goal_dims = tuple(goal_dims)
            mask = torch.zeros(state_size)
            mask[list(self.goal_dims)] = 1
            self.register_buffer('goal_mask', mask, persistent=False)
            backbone_input_size = (
                3 * state_size if input_mode == 'masked'
                else state_size + len(self.goal_dims)
            )
        elif input_mode in ('latent', 'split_latent'):
            if latent_size is None:
                raise ValueError(f"{input_mode} actor input_mode requires latent_size")
            if input_mode == 'split_latent' and goal_latent_size is None:
                raise ValueError(
                    'split_latent actor input_mode requires goal_latent_size'
                )
            if goal_latent_size is None:
                goal_latent_size = latent_size
            backbone_input_size = latent_size + goal_latent_size
        else:
            raise ValueError(f"Unknown actor input_mode: {input_mode!r}")
        self.backbone = make_mlp(
            backbone_input_size,
            self.action_output.input_size,
            hidden_sizes=arch,
            kind=mlp_kind,
            residual_block_size=residual_block_size,
            activation_fn=nn.SiLU if mlp_kind == 'residual' else nn.ReLU,
            zero_init_last_fc=True,
        )

    def forward(self, o: torch.Tensor, g: torch.Tensor) -> torch.distributions.Distribution:
        if self.input_mode == 'raw':
            og = torch.stack([o, g], dim=-len(self.observation_shape) - 1)
            actor_input = self.observation_encoding(og).flatten(-2, -1)
        elif self.input_mode == 'masked':
            # Apply the same goal transform during training and inference.
            # torch.where also removes nonfinite values in unspecified slots.
            mask = self.goal_mask.to(dtype=g.dtype).expand_as(g)
            masked_goal = torch.where(mask.bool(), g, torch.zeros_like(g))
            actor_input = torch.cat([o, masked_goal, mask], dim=-1)
        elif self.input_mode == 'goal_only':
            actor_input = torch.cat([o, g[..., list(self.goal_dims)]], dim=-1)
        else:
            actor_input = torch.cat([o, g], dim=-1)
        features = self.backbone(actor_input)
        if features.dtype in (torch.float16, torch.bfloat16):
            # Distribution construction uses special functions such as
            # log_ndtr, which are not consistently implemented or stable in
            # low precision. The backbone contains the expensive matmuls; keep
            # only the small distribution head in FP32.
            features = features.float()
        return self.action_output(features)

    # for type hint
    def __call__(self, o: torch.Tensor, g: torch.Tensor) -> torch.distributions.Distribution:
        return super().__call__(o, g)

    def extra_repr(self) -> str:
        return f"observation_shape={self.observation_shape}, input_mode={self.input_mode}"

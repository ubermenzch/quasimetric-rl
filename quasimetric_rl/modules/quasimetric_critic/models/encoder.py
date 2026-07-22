from typing import *

import attrs

import torch
import torch.nn as nn

from ...utils import MLP, LatentTensor

from ....data import EnvSpec
from ....data.env_spec.input_encoding import InputEncoding


ENCODER_KINDS = ('standard', 'split')
SPLIT_BRANCH_NORMALIZATIONS = ('none', 'rmsnorm', 'layernorm')


class Encoder(nn.Module):
    r"""
    (*, *input_shape)                      Input
           |
     [input_encoding]                      e.g., AtariTorso network to map image input to a flat vector
           |
        (*, d)                             Encoded 1-D input
           |
    [MLP specified by arch]                ENCODER
           |
        (*, latent_size)                   1-D Latent
    """

    @attrs.define(kw_only=True)
    class Conf:
        # config / argparse uses this to specify behavior

        kind: str = attrs.field(
            default='standard', validator=attrs.validators.in_(ENCODER_KINDS)
        )
        arch: Tuple[int, ...] = (512, 512)
        latent_size: int = 128
        goal_dims: Optional[Tuple[int, ...]] = attrs.field(
            default=None,
            converter=lambda dims: None if dims is None else tuple(dims),
        )
        goal_arch: Tuple[int, ...] = (384, 384)
        non_goal_arch: Tuple[int, ...] = (384, 384)
        goal_latent_size: int = attrs.field(default=64, validator=attrs.validators.gt(0))
        non_goal_latent_size: int = attrs.field(default=64, validator=attrs.validators.gt(0))
        branch_normalization: str = attrs.field(
            default='none',
            validator=attrs.validators.in_(SPLIT_BRANCH_NORMALIZATIONS),
        )

        def make(self, *, env_spec: EnvSpec) -> Union['Encoder', 'SplitEncoder']:
            if self.kind == 'split':
                if self.goal_dims is None:
                    raise ValueError('SplitEncoder requires encoder.goal_dims')
                if self.goal_latent_size + self.non_goal_latent_size != self.latent_size:
                    raise ValueError(
                        'SplitEncoder goal_latent_size + non_goal_latent_size must equal '
                        f'latent_size, got {self.goal_latent_size} + '
                        f'{self.non_goal_latent_size} != {self.latent_size}'
                    )
                return SplitEncoder(
                    env_spec=env_spec,
                    goal_dims=self.goal_dims,
                    goal_arch=self.goal_arch,
                    non_goal_arch=self.non_goal_arch,
                    goal_latent_size=self.goal_latent_size,
                    non_goal_latent_size=self.non_goal_latent_size,
                    branch_normalization=self.branch_normalization,
                )
            if self.branch_normalization != 'none':
                raise ValueError(
                    'encoder.branch_normalization is only valid for encoder.kind=split'
                )
            return Encoder(
                env_spec=env_spec,
                arch=self.arch,
                latent_size=self.latent_size,
            )

    input_shape: torch.Size
    input_encoding: InputEncoding
    encoder: MLP
    latent_size: int

    def __init__(self, *, env_spec: EnvSpec,
                 arch: Tuple[int, ...], latent_size: int, **kwargs):
        super().__init__(**kwargs)
        self.input_shape = env_spec.observation_shape
        self.input_encoding = env_spec.make_observation_input()
        encoder_input_size = self.input_encoding.output_size
        self.encoder = MLP(encoder_input_size, latent_size, hidden_sizes=arch)
        self.latent_size = latent_size

    def forward(self, x: torch.Tensor) -> LatentTensor:
        return self.encoder(self.input_encoding(x))

    def encode_actor_goal(self, goal: torch.Tensor) -> LatentTensor:
        return self(goal)

    # for type hint
    def __call__(self, x: torch.Tensor) -> LatentTensor:
        return super().__call__(x)

    def extra_repr(self) -> str:
        return f"input_shape={self.input_shape}, latent_size={self.latent_size}"


class SplitEncoder(nn.Module):
    """Encode goal and non-goal coordinates with independent MLPs."""

    input_shape: torch.Size
    latent_size: int
    goal_dims: Tuple[int, ...]
    non_goal_dims: Tuple[int, ...]
    goal_latent_size: int
    non_goal_latent_size: int
    goal_encoder: MLP
    non_goal_encoder: MLP
    goal_normalization: nn.Module
    non_goal_normalization: nn.Module

    def __init__(
            self, *, env_spec: EnvSpec, goal_dims: Tuple[int, ...],
            goal_arch: Tuple[int, ...], non_goal_arch: Tuple[int, ...],
            goal_latent_size: int, non_goal_latent_size: int,
            branch_normalization: str = 'none'):
        super().__init__()
        if len(env_spec.observation_shape) != 1:
            raise RuntimeError('SplitEncoder currently supports vector observations only')
        observation_size = int(env_spec.observation_shape[0])
        goal_dims = tuple(goal_dims)
        if not goal_dims or len(set(goal_dims)) != len(goal_dims):
            raise ValueError(f'goal_dims must be non-empty and unique, got {goal_dims!r}')
        if min(goal_dims) < 0 or max(goal_dims) >= observation_size:
            raise ValueError(
                f'Invalid goal_dims={goal_dims!r} for observation size {observation_size}'
            )
        goal_dim_set = set(goal_dims)
        non_goal_dims = tuple(dim for dim in range(observation_size) if dim not in goal_dim_set)
        if not non_goal_dims:
            raise ValueError('SplitEncoder requires at least one non-goal dimension')
        if branch_normalization not in SPLIT_BRANCH_NORMALIZATIONS:
            raise ValueError(
                f'Unknown split branch normalization: {branch_normalization!r}'
            )

        self.input_shape = env_spec.observation_shape
        self.goal_dims = goal_dims
        self.non_goal_dims = non_goal_dims
        self.goal_latent_size = goal_latent_size
        self.non_goal_latent_size = non_goal_latent_size
        self.latent_size = goal_latent_size + non_goal_latent_size
        self.branch_normalization = branch_normalization
        self.goal_encoder = MLP(
            len(goal_dims), goal_latent_size, hidden_sizes=goal_arch
        )
        self.non_goal_encoder = MLP(
            len(non_goal_dims), non_goal_latent_size, hidden_sizes=non_goal_arch
        )
        if branch_normalization == 'rmsnorm':
            self.goal_normalization = nn.RMSNorm(
                goal_latent_size, elementwise_affine=False
            )
            self.non_goal_normalization = nn.RMSNorm(
                non_goal_latent_size, elementwise_affine=False
            )
        elif branch_normalization == 'layernorm':
            self.goal_normalization = nn.LayerNorm(
                goal_latent_size, elementwise_affine=False
            )
            self.non_goal_normalization = nn.LayerNorm(
                non_goal_latent_size, elementwise_affine=False
            )
        else:
            self.goal_normalization = nn.Identity()
            self.non_goal_normalization = nn.Identity()

    def normalize_goal_part(self, latent: LatentTensor) -> LatentTensor:
        if latent.shape[-1] != self.goal_latent_size:
            raise ValueError(
                f'Expected goal latent size {self.goal_latent_size}, '
                f'got {latent.shape[-1]}'
            )
        return self.goal_normalization(latent)

    def normalize_non_goal_part(self, latent: LatentTensor) -> LatentTensor:
        if latent.shape[-1] != self.non_goal_latent_size:
            raise ValueError(
                f'Expected non-goal latent size {self.non_goal_latent_size}, '
                f'got {latent.shape[-1]}'
            )
        return self.non_goal_normalization(latent)

    def encode_goal_part(self, observation: torch.Tensor) -> LatentTensor:
        return self.normalize_goal_part(
            self.goal_encoder(observation[..., list(self.goal_dims)])
        )

    def encode_non_goal_part(self, observation: torch.Tensor) -> LatentTensor:
        return self.normalize_non_goal_part(
            self.non_goal_encoder(observation[..., list(self.non_goal_dims)])
        )

    def join_parts(
            self, goal_latent: torch.Tensor,
            non_goal_latent: torch.Tensor) -> LatentTensor:
        if goal_latent.shape[:-1] != non_goal_latent.shape[:-1]:
            raise ValueError(
                'goal and non-goal latent batch shapes must match, got '
                f'{tuple(goal_latent.shape)} and {tuple(non_goal_latent.shape)}'
            )
        if goal_latent.shape[-1] != self.goal_latent_size:
            raise ValueError(
                f'Expected goal latent size {self.goal_latent_size}, '
                f'got {goal_latent.shape[-1]}'
            )
        if non_goal_latent.shape[-1] != self.non_goal_latent_size:
            raise ValueError(
                f'Expected non-goal latent size {self.non_goal_latent_size}, '
                f'got {non_goal_latent.shape[-1]}'
            )
        return torch.cat([goal_latent, non_goal_latent], dim=-1)

    def split_latent(self, latent: LatentTensor) -> Tuple[LatentTensor, LatentTensor]:
        if latent.shape[-1] != self.latent_size:
            raise ValueError(
                f'Expected latent size {self.latent_size}, got {latent.shape[-1]}'
            )
        return torch.split(
            latent, [self.goal_latent_size, self.non_goal_latent_size], dim=-1
        )

    def encode_actor_goal(self, goal: torch.Tensor) -> LatentTensor:
        goal_latent = self.encode_goal_part(goal)
        non_goal_latent = goal_latent.new_zeros(
            *goal_latent.shape[:-1],
            self.non_goal_latent_size,
        )
        return self.join_parts(goal_latent, non_goal_latent)

    def forward(self, observation: torch.Tensor) -> LatentTensor:
        return self.join_parts(
            self.encode_goal_part(observation),
            self.encode_non_goal_part(observation),
        )

    def __call__(self, observation: torch.Tensor) -> LatentTensor:
        return super().__call__(observation)

    def extra_repr(self) -> str:
        return (
            f'input_shape={self.input_shape}, latent_size={self.latent_size}, '
            f'goal_dims={self.goal_dims}, non_goal_dims={self.non_goal_dims}, '
            f'goal_latent_size={self.goal_latent_size}, '
            f'non_goal_latent_size={self.non_goal_latent_size}, '
            f'branch_normalization={self.branch_normalization!r}'
        )


__all__ = [
    'Encoder', 'SplitEncoder', 'ENCODER_KINDS',
    'SPLIT_BRANCH_NORMALIZATIONS',
]

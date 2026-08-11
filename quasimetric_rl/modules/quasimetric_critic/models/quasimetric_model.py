from typing import *

import attrs
import functools

import torch
import torch.nn as nn
import torchqmet

from ...utils import MLP_KINDS, LatentTensor, make_mlp


class L2(torchqmet.QuasimetricBase):
    r"""
    This is a *metric* (not quasimetric) that is used for debugging & comparison.
    """

    def __init__(self, input_size: int) -> None:
        super().__init__(input_size, num_components=1, guaranteed_quasimetric=True,
                         transforms=[], reduction='sum', discount=None)

    def compute_components(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r'''
        Inputs:
            x (torch.Tensor): Shape [..., input_size]
            y (torch.Tensor): Shape [..., input_size]

        Output:
            d (torch.Tensor): Shape [..., num_components]
        '''
        return (x - y).norm(p=2, dim=-1, keepdim=True)


def create_quasimetric_head_from_spec(spec: str) -> torchqmet.QuasimetricBase:
    # Only two are supported
    #   1. iqe(dim=xxx,components=xxx), Interval Quasimetric Embedding
    #   2. l2(dim=xxx), L2 distance

    def iqe(*, dim: int, components: int) -> torchqmet.IQE:
        assert dim % components == 0, "IQE: dim must be divisible by components"
        return torchqmet.IQE(dim, dim // components)

    def l2(*, dim: int) -> L2:
        return L2(dim)

    return eval(spec, dict(iqe=iqe, l2=l2), {})



class QuasimetricModel(nn.Module):
    r"""
    (*, input_shape)    (*, input_shape)        Input latents
           |                   |
        [MLP specified by projector_arch]       i.e., apply the same MLP on both inputs
           |                   |
        (*, d_proj)          (*, d_proj)        Projected latents
           +---------+---------+
                     |
    [quasimetric head specified by quasimetric_head_spec]
                     |
                    (*)                         Estimated quasimetric d(x, y)
                     or
                    (*, 2)                      if bidirectional=True
    """

    @attrs.define(kw_only=True)
    class Conf:
        # config / argparse uses this to specify behavior

        projector_arch: Tuple[int, ...] = (512,)
        projector_mlp_kind: str = attrs.field(
            default='plain', validator=attrs.validators.in_(MLP_KINDS)
        )
        projector_residual_block_size: int = attrs.field(
            default=4, validator=attrs.validators.gt(0)
        )
        projector_activation: str = attrs.field(
            default='relu',
            validator=attrs.validators.in_(('relu', 'leaky_relu', 'silu')),
        )
        projector_negative_slope: float = attrs.field(
            default=0.01,
            validator=attrs.validators.ge(0),
        )
        quasimetric_head_spec: str = 'iqe(dim=2048,components=64)'

        def make(self, *, input_size: int) -> 'QuasimetricModel':
            return QuasimetricModel(
                input_size=input_size,
                projector_arch=self.projector_arch,
                projector_mlp_kind=self.projector_mlp_kind,
                projector_residual_block_size=self.projector_residual_block_size,
                projector_activation=self.projector_activation,
                projector_negative_slope=self.projector_negative_slope,
                quasimetric_head_spec=self.quasimetric_head_spec,
            )

    input_size: int
    projector: nn.Module
    quasimetric_head: torchqmet.QuasimetricBase

    def __init__(self, *, input_size: int, projector_arch: Tuple[int, ...],
                 projector_mlp_kind: str = 'plain',
                 projector_residual_block_size: int = 4,
                 projector_activation: str = 'relu', projector_negative_slope: float = 0.01,
                 quasimetric_head_spec: str):
        super().__init__()
        self.input_size = input_size
        self.quasimetric_head = create_quasimetric_head_from_spec(quasimetric_head_spec)
        if projector_activation == 'relu':
            activation_fn = nn.ReLU
        elif projector_activation == 'leaky_relu':
            activation_fn = functools.partial(
                nn.LeakyReLU,
                negative_slope=projector_negative_slope,
            )
        elif projector_activation == 'silu':
            activation_fn = nn.SiLU
        else:
            raise ValueError(f'unknown projector activation: {projector_activation!r}')
        self.projector = make_mlp(
            input_size,
            self.quasimetric_head.input_size,
            hidden_sizes=projector_arch,
            kind=projector_mlp_kind,
            residual_block_size=projector_residual_block_size,
            activation_fn=activation_fn,
        )

    def project(self, z: LatentTensor) -> torch.Tensor:
        return self.projector(z)

    def forward_projected(self, px: torch.Tensor, py: torch.Tensor, *, bidirectional: bool = False) -> torch.Tensor:
        if bidirectional:
            px, py = torch.broadcast_tensors(px, py)
            px, py = torch.stack([px, py], dim=-2), torch.stack([py, px], dim=-2)  # [B x 2 x D]

        return self.quasimetric_head(px, py)

    def forward(self, zx: LatentTensor, zy: LatentTensor, *, bidirectional: bool = False) -> torch.Tensor:
        return self.forward_projected(
            self.project(zx),
            self.project(zy),
            bidirectional=bidirectional,
        )

    # for type hint
    def __call__(self, zx: LatentTensor, zy: LatentTensor, *, bidirectional: bool = False) -> torch.Tensor:
        return super().__call__(zx, zy, bidirectional=bidirectional)

    def extra_repr(self) -> str:
        return f"input_size={self.input_size}"

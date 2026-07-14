from typing import *

import attrs

import torch
import torch.nn as nn

from ...utils import MLP, LatentTensor

from ....data import EnvSpec
from ....data.env_spec.input_encoding import InputEncoding



class MlpLatentDynamics(MLP):
    action_input: InputEncoding
    residual: bool

    def __init__(self, *, latent_size:int, env_spec: EnvSpec, hidden_sizes: Tuple[int, ...], residual: bool):
        action_input = env_spec.make_action_input()
        super().__init__(
            latent_size + action_input.output_size,
            latent_size,
            hidden_sizes=hidden_sizes,
            zero_init_last_fc=residual,
        )
        self.action_input = action_input
        self.residual = residual

    @property
    def required_transition_history_length(self) -> int:
        return 0

    def forward(self, zx: LatentTensor, action: torch.Tensor) -> LatentTensor:
        # broadcast batch shapes before cat
        action = self.action_input(action)
        broadcast_bshape: torch.Size = torch.broadcast_shapes(zx.shape[:-1], action.shape[:-1])
        zx = zx.expand(broadcast_bshape + zx.shape[-1:])
        action = action.expand(broadcast_bshape + action.shape[-1:])

        zy = super().forward(
            torch.cat([zx, action], dim=-1)
        )
        if self.residual:
            zy = zx + zy
        return zy

    # for type hints
    def __call__(self, zx: LatentTensor, action: torch.Tensor) -> LatentTensor:
        return nn.Module.__call__(self, zx, action)

    def extra_repr(self) -> str:
        return super().extra_repr() + f"\nresidual={self.residual}"


class TransformerLatentDynamics(nn.Module):
    latent_size: int
    history_length: int
    action_input: InputEncoding
    action_size: int
    token_input: nn.Linear
    position_embedding: nn.Parameter
    transformer: nn.TransformerEncoder
    output: nn.Linear
    residual: bool

    def __init__(
        self,
        *,
        latent_size: int,
        env_spec: EnvSpec,
        history_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        ff_mult: int,
        dropout: float,
        residual: bool,
    ):
        super().__init__()
        assert history_length > 0
        self.latent_size = latent_size
        self.history_length = history_length
        self.action_input = env_spec.make_action_input()
        self.action_size = self.action_input.output_size
        self.residual = residual
        self.token_input = nn.Linear(latent_size + self.action_size, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(history_length, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output = nn.Linear(d_model, latent_size)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.token_input.weight)
            nn.init.zeros_(self.token_input.bias)
            nn.init.zeros_(self.position_embedding)
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    @property
    def required_transition_history_length(self) -> int:
        return self.history_length

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)

    def forward_sequence(
        self,
        z_history: torch.Tensor,
        action_history: torch.Tensor,
        history_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert z_history.shape[-2] == action_history.shape[-2] + 1
        length = action_history.shape[-2]
        assert length <= self.history_length
        action_history = self.action_input(action_history)
        tokens = torch.cat([z_history[..., :-1, :], action_history], dim=-1)
        batch_shape = tokens.shape[:-2]
        tokens = tokens.reshape(-1, length, tokens.shape[-1])
        x = self.token_input(tokens) + self.position_embedding[:length].unsqueeze(0)
        key_padding_mask = None
        if history_mask is not None:
            key_padding_mask = ~history_mask.reshape(-1, length).to(torch.bool)
        x = self.transformer(
            x,
            mask=self._causal_mask(length, x.device),
            src_key_padding_mask=key_padding_mask,
            # Avoid PyTorch 2.x re-detecting the causal CUDA mask every forward.
            is_causal=True,
        )
        delta = self.output(x).reshape(batch_shape + (length, self.latent_size))
        if self.residual:
            delta = z_history[..., :-1, :] + delta
        return delta

    def forward(self, zx: LatentTensor, action: torch.Tensor) -> LatentTensor:
        z_history = torch.stack([zx, zx], dim=-2)
        action_history = action.unsqueeze(-2)
        return self.forward_sequence(z_history, action_history)[..., -1, :]

    def __call__(self, zx: LatentTensor, action: torch.Tensor) -> LatentTensor:
        return nn.Module.__call__(self, zx, action)

    def extra_repr(self) -> str:
        return (
            f"latent_size={self.latent_size}, history_length={self.history_length}, "
            f"action_size={self.action_size}, residual={self.residual}"
        )


class LatentDynamics(nn.Module):
    @attrs.define(kw_only=True)
    class Conf:
        # config / argparse uses this to specify behavior

        kind: str = 'mlp'
        arch: Tuple[int, ...] = (512, 512)
        residual: bool = True
        # Counts input state frames: h=1 is current state only; h=2 adds one
        # historical state. The following state is the prediction target.
        history_length: int = attrs.field(default=8, validator=attrs.validators.gt(0))
        transformer_d_model: Optional[int] = attrs.field(
            default=None,
            validator=attrs.validators.optional(attrs.validators.gt(0)),
        )
        transformer_num_layers: int = attrs.field(default=2, validator=attrs.validators.gt(0))
        transformer_num_heads: int = attrs.field(default=8, validator=attrs.validators.gt(0))
        transformer_ff_mult: int = attrs.field(default=4, validator=attrs.validators.gt(0))
        transformer_dropout: float = attrs.field(default=0.0, validator=attrs.validators.ge(0))

        def make(self, *, latent_size: int, env_spec: EnvSpec) -> Union[MlpLatentDynamics, TransformerLatentDynamics]:
            if self.kind == 'mlp':
                return MlpLatentDynamics(
                    latent_size=latent_size,
                    env_spec=env_spec,
                    hidden_sizes=self.arch,
                    residual=self.residual,
                )
            if self.kind == 'transformer':
                d_model = self.transformer_d_model or latent_size
                return TransformerLatentDynamics(
                    latent_size=latent_size,
                    env_spec=env_spec,
                    history_length=self.history_length,
                    d_model=d_model,
                    num_layers=self.transformer_num_layers,
                    num_heads=self.transformer_num_heads,
                    ff_mult=self.transformer_ff_mult,
                    dropout=self.transformer_dropout,
                    residual=self.residual,
                )
            raise ValueError(f"unknown latent dynamics kind: {self.kind}")

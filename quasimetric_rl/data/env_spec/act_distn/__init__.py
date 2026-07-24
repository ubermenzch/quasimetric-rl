from typing import *
from typing_extensions import Protocol, Final

import abc
import warnings

import gym.spaces

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions
import torch.distributions.constraints

from .... import FLAGS



#-----------------------------------------------------------------------------#
#-------------------------------- output API ---------------------------------#
#-----------------------------------------------------------------------------#


class TensorDistributionProtocol(Protocol):
    batch_shape: torch.Size
    event_shape: torch.Size

    mode: torch.Tensor
    mean: torch.Tensor

    def sample(self, sample_shape=torch.Size()) -> torch.Tensor:
        pass

    def rsample(self, sample_shape=torch.Size()) -> torch.Tensor:
        pass

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        pass

    def entropy(self, num_samples: Optional[int] = None) -> torch.Tensor:
        pass


class ActionOutputConverter(nn.Module, metaclass=abc.ABCMeta):
    input_size: int

    def __init__(self, action_space: gym.spaces.Box) -> None:
        super().__init__()

    @abc.abstractmethod
    def forward(self, feature: torch.Tensor) -> TensorDistributionProtocol:
        pass

    def __call__(self, feature: torch.Tensor) -> TensorDistributionProtocol:
        return super().__call__(feature)


#-----------------------------------------------------------------------------#
#----------------------------------- impls -----------------------------------#
#-----------------------------------------------------------------------------#


class DiscreteOutputOneHot(ActionOutputConverter):
    input_size: Final[int]
    num_actions: Final[int]

    def __init__(self, action_space: gym.spaces.Discrete) -> None:
        super().__init__(action_space)
        self.input_size = self.num_actions = int(action_space.n)

    def forward(self, feature: torch.Tensor) -> torch.distributions.Distribution:
        return torch.distributions.Categorical(
            logits=feature,
            validate_args=FLAGS.DEBUG,
        )


class BoxOutputLinearNormalization(ActionOutputConverter):
    input_size: Final[int]

    kind: str
    action_shape: torch.Size
    _mean_values: Tuple[float, ...]
    _half_len_values: Tuple[float, ...]

    @property
    def output_distn_ty(self) -> str:
        return self.kind

    def __init__(self, action_space: gym.spaces.Box) -> None:
        super().__init__(action_space)
        self.input_size = torch.Size(action_space.shape).numel() * 2
        high = torch.as_tensor(action_space.high, dtype=torch.float32)
        low = torch.as_tensor(action_space.low, dtype=torch.float32)
        mean = (high + low) / 2
        half_len = ((high - low) / 2).clamp_min(1e-3)
        self.action_shape = mean.shape
        self._mean_values = tuple(
            float(value) for value in mean.reshape(-1).tolist()
        )
        self._half_len_values = tuple(
            float(value) for value in half_len.reshape(-1).tolist()
        )
        assert torch.as_tensor(
            action_space.bounded_above & action_space.bounded_below
        ).all(), "Must have bounded action space"

    def _reference_tensor(self, values: Tuple[float, ...]) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.float32).reshape(self.action_shape)

    def _action_bounds_like(
            self, feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bounds = feature.new_tensor(
            (self._mean_values, self._half_len_values)
        ).reshape(2, *self.action_shape)
        return bounds.unbind(dim=0)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'mean'] = self._reference_tensor(self._mean_values)
        destination[prefix + 'half_len'] = self._reference_tensor(
            self._half_len_values
        )

    def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs):
        for name, values in (
                ('mean', self._mean_values),
                ('half_len', self._half_len_values)):
            key = prefix + name
            loaded = state_dict.pop(key, None)
            expected = self._reference_tensor(values)
            if loaded is not None:
                observed = loaded.detach().to(device='cpu', dtype=torch.float32)
                if observed.shape != expected.shape or not torch.equal(
                        observed, expected):
                    warnings.warn(
                        f'Ignoring checkpoint {key}={observed.tolist()}; '
                        'action bounds are derived from the current environment '
                        f'and expected {expected.tolist()}',
                        RuntimeWarning,
                        stacklevel=2,
                    )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, feature: torch.Tensor) -> torch.distributions.Distribution:
        gmean, grawstd = feature.view(
            *feature.shape[:-1], 2, *self.action_shape
        ).unbind(dim=-len(self.action_shape) - 1)
        mean, half_len = self._action_bounds_like(feature)
        pre_tanh_distn = torch.distributions.Normal(
            loc=gmean,
            scale=F.softplus(grawstd) + 1e-4,
            validate_args=FLAGS.DEBUG,
        )

        # Acme (CRL) Tanh Normal
        from .utils import AcmeTanhTransformedDistribution, SampleDist
        distn = AcmeTanhTransformedDistribution(
            pre_tanh_distn,
            validate_args=FLAGS.DEBUG,
        )
        distn = torch.distributions.TransformedDistribution(
            distn,
            torch.distributions.AffineTransform(loc=mean, scale=half_len),
            validate_args=FLAGS.DEBUG,
        )
        distn = torch.distributions.Independent(
            distn,
            reinterpreted_batch_ndims=1,
            validate_args=FLAGS.DEBUG,
        )

        distn = SampleDist(
            distn,
            pre_tanh_distn=pre_tanh_distn,
            affine_loc=mean,
            affine_scale=half_len,
        )

        return distn

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Mapping, Optional, Tuple

import attrs
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data import BatchData, EnvSpec
from .utils import InfoT, LossResult, MLP, Module, ResidualMLP


BASELINE_ALGORITHMS = (
    'td_infonce', 'crl', 'scaling_crl', 'gcbc', 'gcsl', 'c_learning',
)
FETCH_MANIPULATION_ENVIRONMENTS = frozenset({
    ('gcrl', 'FetchPush'),
    ('gcrl', 'FetchSlide'),
    ('gcrl', 'FetchPickAndPlace'),
})
TD_INFONCE_REFERENCE_REVISION = '18f4e7e5872da9c3653f57661d01b4fbce85b50e'
CRL_REFERENCE_REVISION = 'ec7c3d346277b737bc2decffcd1b533d4b7ec105'
SCALING_CRL_REFERENCE_REVISION = '17acb519ddc4325c8662b1f8c68ed6a5f31857fc'
GCSL_REFERENCE_REVISION = 'cfae5609cee79e5a2228fb7653451023c41a64cb'
C_LEARNING_REFERENCE_REVISION = 'ec7c3d346277b737bc2decffcd1b533d4b7ec105'


def resolve_baseline_goal_dims(
        algorithm: str, *, env_kind: Optional[str], env_name: Optional[str],
        state_dim: int,
        success_goal_dims: Tuple[int, ...]) -> Tuple[int, ...]:
    """Resolve the hindsight-goal representation independently of success."""
    if algorithm not in BASELINE_ALGORITHMS:
        raise ValueError(f'Unknown GCRL baseline: {algorithm!r}')
    success_goal_dims = tuple(int(dim) for dim in success_goal_dims)
    _validate_goal_dims(success_goal_dims, state_dim)

    # The original 2022 CRL configuration defaults to start_index=0 and
    # end_index=-1, including the complete future state in every environment.
    if algorithm == 'crl':
        return tuple(range(state_dim))
    if algorithm == 'scaling_crl':
        return success_goal_dims
    if (env_kind, env_name) not in FETCH_MANIPULATION_ENVIRONMENTS:
        return success_goal_dims
    if state_dim < 6:
        raise ValueError(
            f'{env_kind}/{env_name} requires at least 6 state dimensions, '
            f'got {state_dim}'
        )
    if algorithm in ('td_infonce', 'c_learning'):
        return tuple(range(state_dim))
    return tuple(range(6))


def _positive_tuple(value):
    value = tuple(int(item) for item in value)
    if not value or min(value) <= 0:
        raise ValueError(f'Expected positive hidden sizes, got {value!r}')
    return value


@attrs.define(kw_only=True)
class TDInfoNCEConf:
    reference_revision: str = TD_INFONCE_REFERENCE_REVISION
    hidden_sizes: Tuple[int, ...] = attrs.field(
        default=(512, 512, 512, 512), converter=_positive_tuple,
    )
    representation_dim: int = attrs.field(default=16, validator=attrs.validators.gt(0))
    actor_lr: float = attrs.field(default=5e-5, validator=attrs.validators.gt(0))
    critic_lr: float = attrs.field(default=3e-4, validator=attrs.validators.gt(0))
    discount: float = attrs.field(
        default=0.99,
        validator=attrs.validators.and_(attrs.validators.ge(0), attrs.validators.le(1)),
    )
    tau: float = attrs.field(
        default=0.005,
        validator=attrs.validators.and_(attrs.validators.gt(0), attrs.validators.le(1)),
    )
    representation_norm: bool = True
    representation_temperature: float = attrs.field(
        default=1.0, validator=attrs.validators.gt(0),
    )
    batch_size: int = attrs.field(default=256, validator=attrs.validators.gt(0))
    updates_per_env_step: float = attrs.field(
        default=1.0, validator=attrs.validators.gt(0),
    )
    min_replay_size: int = attrs.field(default=10_000, validator=attrs.validators.gt(0))
    max_replay_size: int = attrs.field(default=1_000_000, validator=attrs.validators.gt(0))


@attrs.define(kw_only=True)
class CRLConf:
    reference_revision: str = CRL_REFERENCE_REVISION
    hidden_sizes: Tuple[int, ...] = attrs.field(
        default=(256, 256), converter=_positive_tuple,
    )
    representation_dim: int = attrs.field(default=64, validator=attrs.validators.gt(0))
    actor_lr: float = attrs.field(default=3e-4, validator=attrs.validators.gt(0))
    critic_lr: float = attrs.field(default=3e-4, validator=attrs.validators.gt(0))
    discount: float = attrs.field(
        default=0.99,
        validator=attrs.validators.and_(attrs.validators.ge(0), attrs.validators.le(1)),
    )
    contrastive_loss: str = attrs.field(
        default='binary_nce',
        validator=attrs.validators.in_((
            'fwd_infonce', 'bwd_infonce', 'sym_infonce', 'binary_nce',
        )),
    )
    energy: str = attrs.field(
        default='dot',
        validator=attrs.validators.in_(('norm', 'l2', 'dot', 'cosine')),
    )
    logsumexp_penalty: float = attrs.field(default=0.0, validator=attrs.validators.ge(0))
    activation: str = attrs.field(
        default='relu', validator=attrs.validators.in_(('relu', 'silu')),
    )
    representation_norm: bool = False
    entropy_coefficient: float = attrs.field(default=0.0, validator=attrs.validators.ge(0))
    random_goal_fraction: float = attrs.field(
        default=0.5, validator=attrs.validators.in_((0.0, 0.5, 1.0)),
    )
    adam_eps: float = attrs.field(default=1e-7, validator=attrs.validators.gt(0))
    batch_size: int = attrs.field(default=256, validator=attrs.validators.gt(0))
    updates_per_env_step: float = attrs.field(
        default=1.0, validator=attrs.validators.gt(0),
    )
    min_replay_size: int = attrs.field(default=10_000, validator=attrs.validators.gt(0))
    max_replay_size: int = attrs.field(default=1_000_000, validator=attrs.validators.gt(0))


def _scaling_crl_arch(value):
    value = _positive_tuple(value)
    if len(set(value)) != 1:
        raise ValueError(
            'Scaling-CRL requires a constant width for residual shortcuts, '
            f'got {value!r}'
        )
    if len(value) % 4:
        raise ValueError(
            'Scaling-CRL depth must be divisible by its four-layer residual '
            f'block size, got depth={len(value)}'
        )
    return value


@attrs.define(kw_only=True)
class ScalingCRLConf:
    """Defaults from Wang et al. (2025), separate from the 2022 CRL port."""

    reference_revision: str = SCALING_CRL_REFERENCE_REVISION
    hidden_sizes: Tuple[int, ...] = attrs.field(
        default=(256, 256, 256, 256), converter=_scaling_crl_arch,
    )
    representation_dim: int = attrs.field(default=64, validator=attrs.validators.gt(0))
    actor_lr: float = attrs.field(default=3e-4, validator=attrs.validators.gt(0))
    critic_lr: float = attrs.field(default=3e-4, validator=attrs.validators.gt(0))
    alpha_lr: float = attrs.field(default=3e-4, validator=attrs.validators.gt(0))
    discount: float = attrs.field(
        default=0.99,
        validator=attrs.validators.and_(attrs.validators.ge(0), attrs.validators.le(1)),
    )
    logsumexp_penalty: float = attrs.field(
        default=0.1, validator=attrs.validators.ge(0),
    )
    target_entropy_per_action: float = attrs.field(
        default=-0.5, validator=attrs.validators.lt(0),
    )
    batch_size: int = attrs.field(default=512, validator=attrs.validators.gt(0))
    # Official code: 800 SGD batches per 512 x 62 vectorized environment steps.
    updates_per_env_step: float = attrs.field(
        default=800 / (512 * 62), validator=attrs.validators.gt(0),
    )
    min_replay_size: int = attrs.field(default=1_000, validator=attrs.validators.gt(0))
    max_replay_size: int = attrs.field(default=10_000, validator=attrs.validators.gt(0))
    residual_block_size: int = attrs.field(default=4, validator=attrs.validators.gt(0))
    additive_exploration_std_fraction: float = attrs.field(
        default=0.0, validator=attrs.validators.ge(0),
    )
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    def __attrs_post_init__(self) -> None:
        if self.residual_block_size != 4:
            raise ValueError(
                'The official Scaling-CRL architecture uses four-layer '
                f'residual blocks, got {self.residual_block_size}'
            )
        if self.log_std_min >= self.log_std_max:
            raise ValueError(
                f'Expected log_std_min < log_std_max, got '
                f'{self.log_std_min} >= {self.log_std_max}'
            )


@attrs.define(kw_only=True)
class GCSLConf:
    reference_revision: str = GCSL_REFERENCE_REVISION
    hidden_sizes: Tuple[int, ...] = attrs.field(
        default=(400, 300), converter=_positive_tuple,
    )
    actor_lr: float = attrs.field(default=5e-4, validator=attrs.validators.gt(0))
    action_granularity: int = attrs.field(default=3, validator=attrs.validators.gt(1))
    start_policy_timesteps: int = attrs.field(
        default=1_000, validator=attrs.validators.ge(0),
    )
    explore_timesteps: int = attrs.field(
        default=10_000, validator=attrs.validators.ge(0),
    )
    validation_fraction: float = attrs.field(
        default=0.2,
        validator=attrs.validators.and_(attrs.validators.ge(0), attrs.validators.lt(1)),
    )
    replay_capacity_trajectories: int = attrs.field(
        default=20_000, validator=attrs.validators.gt(0),
    )
    batch_size: int = attrs.field(default=256, validator=attrs.validators.gt(0))
    updates_per_env_step: float = attrs.field(
        default=1.0, validator=attrs.validators.gt(0),
    )


@attrs.define(kw_only=True)
class CLearningConf:
    reference_revision: str = C_LEARNING_REFERENCE_REVISION
    hidden_sizes: Tuple[int, ...] = attrs.field(
        default=(256, 256), converter=_positive_tuple,
    )
    actor_lr: float = attrs.field(default=3e-4, validator=attrs.validators.gt(0))
    critic_lr: float = attrs.field(default=3e-4, validator=attrs.validators.gt(0))
    discount: float = attrs.field(
        default=0.99,
        validator=attrs.validators.and_(attrs.validators.ge(0), attrs.validators.lt(1)),
    )
    tau: float = attrs.field(
        default=0.005,
        validator=attrs.validators.and_(attrs.validators.gt(0), attrs.validators.le(1)),
    )
    odds_clip: float = attrs.field(default=20.0, validator=attrs.validators.gt(0))
    critic_loss_weight: float = attrs.field(
        default=0.5, validator=attrs.validators.ge(0),
    )
    relabel_next_probability: float = attrs.field(
        default=0.5,
        validator=attrs.validators.and_(attrs.validators.ge(0), attrs.validators.le(1)),
    )
    relabel_future_probability: float = attrs.field(
        default=0.0,
        validator=attrs.validators.and_(attrs.validators.ge(0), attrs.validators.le(1)),
    )
    batch_size: int = attrs.field(default=256, validator=attrs.validators.gt(0))
    updates_per_env_step: float = attrs.field(
        default=1.0, validator=attrs.validators.gt(0),
    )
    initial_collect_steps: int = attrs.field(
        default=10_000, validator=attrs.validators.gt(0),
    )
    replay_buffer_capacity: int = attrs.field(
        default=1_000_000, validator=attrs.validators.gt(0),
    )

    def __attrs_post_init__(self) -> None:
        if self.relabel_next_probability + self.relabel_future_probability != 0.5:
            raise ValueError(
                'C-Learning requires next and future relabel probabilities to sum to 0.5'
            )


@attrs.define(kw_only=True)
class GCRLBaselinesConf:
    td_infonce: TDInfoNCEConf = TDInfoNCEConf()
    crl: CRLConf = CRLConf()
    scaling_crl: ScalingCRLConf = ScalingCRLConf()
    # Serialized as ``gcbc`` for compatibility with existing checkpoints.
    gcbc: GCSLConf = GCSLConf()
    c_learning: CLearningConf = CLearningConf()

    def make(
            self, algorithm: str, *, env_spec: EnvSpec,
            goal_dims: Tuple[int, ...]) -> Tuple['GCRLBaselineAgent', 'GCRLBaselineLosses']:
        if algorithm not in BASELINE_ALGORITHMS:
            raise ValueError(f'Unknown GCRL baseline: {algorithm!r}')
        if len(env_spec.observation_shape) != 1:
            raise ValueError(
                f'{algorithm} currently supports vector observations only, got '
                f'{tuple(env_spec.observation_shape)}'
            )
        state_dim = env_spec.observation_shape.numel()
        action_dim = env_spec.action_shape.numel()
        _validate_goal_dims(goal_dims, state_dim)
        actor_log_std_bounds = None
        actor_min_std = None
        actor_action_granularity = None

        if algorithm == 'td_infonce':
            conf = self.td_infonce
            actor_hidden_sizes = conf.hidden_sizes
            actor_activation = 'relu'
            actor_init_scheme = 'td_infonce'
            actor_min_std = 1e-6
            critic = TDInfoNCECritic(
                state_dim=state_dim,
                action_dim=action_dim,
                goal_dim=len(goal_dims),
                hidden_sizes=conf.hidden_sizes,
                representation_dim=conf.representation_dim,
                representation_norm=conf.representation_norm,
                representation_temperature=conf.representation_temperature,
            )
            target_critic = TDInfoNCECritic(
                state_dim=state_dim,
                action_dim=action_dim,
                goal_dim=len(goal_dims),
                hidden_sizes=conf.hidden_sizes,
                representation_dim=conf.representation_dim,
                representation_norm=conf.representation_norm,
                representation_temperature=conf.representation_temperature,
            )
            target_critic.load_state_dict(critic.state_dict())
            target_critic.requires_grad_(False)
        elif algorithm == 'crl':
            conf = self.crl
            actor_hidden_sizes = conf.hidden_sizes
            actor_activation = conf.activation
            actor_init_scheme = 'original_crl_actor'
            actor_min_std = 1e-6
            critic = ContrastiveCritic(
                state_dim=state_dim,
                action_dim=action_dim,
                goal_dim=len(goal_dims),
                hidden_sizes=conf.hidden_sizes,
                representation_dim=conf.representation_dim,
                energy=conf.energy,
                representation_norm=conf.representation_norm,
                activation=conf.activation,
            )
            target_critic = None
        elif algorithm == 'scaling_crl':
            conf = self.scaling_crl
            actor_hidden_sizes = conf.hidden_sizes
            actor_activation = 'silu'
            actor_init_scheme = None
            actor_log_std_bounds = (conf.log_std_min, conf.log_std_max)
            critic = ScalingCRLCritic(
                state_dim=state_dim,
                action_dim=action_dim,
                goal_dim=len(goal_dims),
                hidden_sizes=conf.hidden_sizes,
                representation_dim=conf.representation_dim,
                residual_block_size=conf.residual_block_size,
            )
            target_critic = None
        elif algorithm in ('gcbc', 'gcsl'):
            conf = self.gcbc
            actor_hidden_sizes = conf.hidden_sizes
            actor_activation = 'relu'
            actor_init_scheme = 'torch_default'
            actor_action_granularity = conf.action_granularity
            critic = target_critic = None
        else:
            conf = self.c_learning
            actor_hidden_sizes = conf.hidden_sizes
            actor_activation = 'relu'
            actor_init_scheme = 'c_learning'
            actor_min_std = 0.0
            critic = CLearningCritic(
                state_dim=state_dim,
                action_dim=action_dim,
                goal_dim=len(goal_dims),
                hidden_sizes=conf.hidden_sizes,
            )
            target_critic = CLearningCritic(
                state_dim=state_dim,
                action_dim=action_dim,
                goal_dim=len(goal_dims),
                hidden_sizes=conf.hidden_sizes,
            )
            target_critic.load_state_dict(critic.state_dict())
            target_critic.requires_grad_(False)

        actor = GoalConditionedPolicy(
            env_spec=env_spec,
            goal_dim=len(goal_dims),
            hidden_sizes=actor_hidden_sizes,
            activation=actor_activation,
            init_scheme=actor_init_scheme,
            log_std_bounds=actor_log_std_bounds,
            min_std=actor_min_std,
            action_granularity=actor_action_granularity,
            residual_block_size=(
                self.scaling_crl.residual_block_size
                if algorithm == 'scaling_crl' else None
            ),
        )
        agent = GCRLBaselineAgent(
            algorithm=algorithm,
            actor=actor,
            critic=critic,
            target_critic=target_critic,
            goal_dims=goal_dims,
        )
        return agent, GCRLBaselineLosses(
            algorithm=algorithm,
            agent=agent,
            td_infonce_conf=self.td_infonce,
            crl_conf=self.crl,
            scaling_crl_conf=self.scaling_crl,
            gcsl_conf=self.gcbc,
            c_learning_conf=self.c_learning,
            action_dim=action_dim,
        )


def _validate_goal_dims(goal_dims: Tuple[int, ...], state_dim: int) -> None:
    if not goal_dims:
        raise ValueError('GCRL baselines require at least one goal coordinate')
    if len(set(goal_dims)) != len(goal_dims):
        raise ValueError(f'Goal dimensions must be unique, got {goal_dims!r}')
    if min(goal_dims) < 0 or max(goal_dims) >= state_dim:
        raise ValueError(
            f'Goal dimensions {goal_dims!r} are invalid for state_dim={state_dim}'
        )


def _activation_type(name: str):
    if name == 'relu':
        return nn.ReLU
    if name == 'silu':
        return nn.SiLU
    raise ValueError(f'Unknown activation: {name!r}')


def _initialize_mlp(mlp: MLP, scheme: str) -> None:
    with torch.no_grad():
        for module in mlp.modules():
            if not isinstance(module, nn.Linear):
                continue
            fan_in = module.weight.shape[1]
            if scheme == 'td_infonce':
                nn.init.uniform_(module.weight, -(3 / fan_in) ** 0.5, (3 / fan_in) ** 0.5)
                nn.init.zeros_(module.bias)
            elif scheme == 'original_crl_actor':
                if module is mlp.module[-1]:
                    # Acme's distribution head is a separate Haiku Linear and
                    # therefore uses Haiku's default truncated-normal init.
                    std = fan_in ** -0.5
                    nn.init.trunc_normal_(
                        module.weight, std=std,
                        a=-2 * std, b=2 * std,
                    )
                else:
                    nn.init.uniform_(
                        module.weight,
                        -(3 / fan_in) ** 0.5,
                        (3 / fan_in) ** 0.5,
                    )
                nn.init.zeros_(module.bias)
            elif scheme == 'original_crl_critic':
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif scheme == 'torch_default':
                module.reset_parameters()
            elif scheme == 'c_learning':
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
                if module is mlp.module[-1]:
                    nn.init.uniform_(
                        module.weight,
                        -(0.3 / fan_in) ** 0.5,
                        (0.3 / fan_in) ** 0.5,
                    )
            elif scheme != 'glorot':
                raise ValueError(f'Unknown MLP initialization scheme: {scheme!r}')


class DiscretizedActionDistribution:
    def __init__(self, logits: torch.Tensor, action_table: torch.Tensor):
        self.logits = logits
        self.action_table = action_table
        self._categorical = torch.distributions.Categorical(logits=logits)
        self.batch_shape = logits.shape[:-1]
        self.event_shape = action_table.shape[1:]

    @property
    def mean(self) -> torch.Tensor:
        return self.action_table[self.logits.argmax(dim=-1)]

    @property
    def mode(self) -> torch.Tensor:
        return self.mean

    def sample(self, sample_shape=torch.Size()) -> torch.Tensor:
        return self.action_table[self._categorical.sample(sample_shape)]

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        distance = (
            value.unsqueeze(-2) - self.action_table
        ).square().sum(dim=-1)
        return self._categorical.log_prob(distance.argmin(dim=-1))


class GoalConditionedPolicy(Module):
    def __init__(
            self, *, env_spec: EnvSpec, goal_dim: int,
            hidden_sizes: Tuple[int, ...], activation: str,
            init_scheme: Optional[str],
            log_std_bounds: Optional[Tuple[float, float]] = None,
            min_std: Optional[float] = None,
            action_granularity: Optional[int] = None,
            residual_block_size: Optional[int] = None):
        super().__init__()
        state_dim = env_spec.observation_shape.numel()
        self.action_output = env_spec.make_action_output_distn()
        if action_granularity is None:
            output_size = self.action_output.input_size
            action_table = None
        else:
            if not isinstance(env_spec.action_space, gym.spaces.Box):
                raise TypeError('GCSL action discretization requires Box actions')
            axes = [
                torch.linspace(float(low), float(high), action_granularity)
                for low, high in zip(
                    env_spec.action_space.low.reshape(-1),
                    env_spec.action_space.high.reshape(-1),
                )
            ]
            mesh = torch.meshgrid(*axes, indexing='xy')
            action_table = torch.stack(
                [values.reshape(-1) for values in mesh], dim=-1,
            ).reshape(-1, *env_spec.action_shape)
            output_size = action_granularity ** env_spec.action_shape.numel()
        if residual_block_size is None:
            self.backbone = MLP(
                state_dim + goal_dim,
                output_size,
                hidden_sizes=hidden_sizes,
                activation_fn=_activation_type(activation),
            )
        else:
            self.backbone = ResidualMLP(
                state_dim + goal_dim,
                output_size,
                hidden_sizes=hidden_sizes,
                residual_block_size=residual_block_size,
                activation_fn=_activation_type(activation),
            )
        if init_scheme is not None:
            _initialize_mlp(self.backbone, init_scheme)
        self.register_buffer('action_table', action_table)
        if (log_std_bounds is not None or min_std is not None) and not hasattr(
                self.action_output, 'from_mean_and_std'):
            raise TypeError('Custom policy standard deviation requires Box actions')
        if min_std is not None and min_std < 0:
            raise ValueError(f'Expected min_std >= 0, got {min_std}')
        self.log_std_bounds = log_std_bounds
        self.min_std = min_std

    def sample_uniform_action(self) -> torch.Tensor:
        if self.action_table is None:
            raise RuntimeError('Uniform discrete actions are only defined for GCSL')
        index = torch.randint(
            self.action_table.shape[0], (), device=self.action_table.device,
        )
        return self.action_table[index]

    def forward(
            self, observation: torch.Tensor,
            goal: torch.Tensor) -> torch.distributions.Distribution:
        features = self.backbone(torch.cat([observation, goal], dim=-1))
        if self.action_table is not None:
            return DiscretizedActionDistribution(features, self.action_table)
        if self.log_std_bounds is None and self.min_std is None:
            return self.action_output(features)

        action_shape = self.action_output.action_shape
        mean, raw_log_std = features.view(
            *features.shape[:-1], 2, *action_shape
        ).unbind(dim=-len(action_shape) - 1)
        if self.log_std_bounds is None:
            return self.action_output.from_mean_and_std(
                mean, F.softplus(raw_log_std) + self.min_std,
            )

        lower, upper = self.log_std_bounds
        log_std = lower + 0.5 * (upper - lower) * (
            torch.tanh(raw_log_std) + 1
        )
        return self.action_output.from_mean_and_std(mean, log_std.exp())


class ContrastiveCritic(Module):
    def __init__(
            self, *, state_dim: int, action_dim: int, goal_dim: int,
            hidden_sizes: Tuple[int, ...], representation_dim: int,
            energy: str, representation_norm: bool, activation: str):
        super().__init__()
        activation_fn = _activation_type(activation)
        self.sa_encoder = MLP(
            state_dim + action_dim,
            representation_dim,
            hidden_sizes=hidden_sizes,
            activation_fn=activation_fn,
        )
        self.goal_encoder = MLP(
            goal_dim,
            representation_dim,
            hidden_sizes=hidden_sizes,
            activation_fn=activation_fn,
        )
        _initialize_mlp(self.sa_encoder, 'original_crl_critic')
        _initialize_mlp(self.goal_encoder, 'original_crl_critic')
        self.energy = energy
        self.representation_norm = representation_norm

    def representations(
            self, observation: torch.Tensor, action: torch.Tensor,
            goal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa_repr = self.sa_encoder(torch.cat([observation, action], dim=-1))
        goal_repr = self.goal_encoder(goal)
        if self.representation_norm:
            sa_repr = F.normalize(sa_repr, dim=-1)
            goal_repr = F.normalize(goal_repr, dim=-1)
        return sa_repr, goal_repr

    def _energy(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if self.energy == 'norm':
            return -torch.sqrt((left - right).square().sum(dim=-1) + 1e-6)
        if self.energy == 'l2':
            return -(left - right).square().sum(dim=-1)
        if self.energy == 'dot':
            return (left * right).sum(dim=-1)
        if self.energy == 'cosine':
            return (
                (left * right).sum(dim=-1)
                / (
                    torch.linalg.vector_norm(left)
                    * torch.linalg.vector_norm(right)
                    + 1e-6
                )
            )
        raise RuntimeError(f'Unknown energy: {self.energy!r}')

    def pairwise(
            self, observation: torch.Tensor, action: torch.Tensor,
            goals: torch.Tensor) -> torch.Tensor:
        sa_repr, goal_repr = self.representations(observation, action, goals)
        return self._energy(sa_repr[:, None, :], goal_repr[None, :, :])

    def paired(
            self, observation: torch.Tensor, action: torch.Tensor,
            goal: torch.Tensor) -> torch.Tensor:
        sa_repr, goal_repr = self.representations(observation, action, goal)
        return self._energy(sa_repr, goal_repr)


class ScalingCRLCritic(Module):
    """Residual L2 critic from the official Scaling-CRL implementation."""

    def __init__(
            self, *, state_dim: int, action_dim: int, goal_dim: int,
            hidden_sizes: Tuple[int, ...], representation_dim: int,
            residual_block_size: int):
        super().__init__()
        self.sa_encoder = ResidualMLP(
            state_dim + action_dim,
            representation_dim,
            hidden_sizes=hidden_sizes,
            residual_block_size=residual_block_size,
            activation_fn=nn.SiLU,
        )
        self.goal_encoder = ResidualMLP(
            goal_dim,
            representation_dim,
            hidden_sizes=hidden_sizes,
            residual_block_size=residual_block_size,
            activation_fn=nn.SiLU,
        )

    def representations(
            self, observation: torch.Tensor, action: torch.Tensor,
            goal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self.sa_encoder(torch.cat([observation, action], dim=-1)),
            self.goal_encoder(goal),
        )

    @staticmethod
    def _energy(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return -torch.sqrt((left - right).square().sum(dim=-1))

    def pairwise(
            self, observation: torch.Tensor, action: torch.Tensor,
            goals: torch.Tensor) -> torch.Tensor:
        sa_repr, goal_repr = self.representations(observation, action, goals)
        return self._energy(sa_repr[:, None, :], goal_repr[None, :, :])

    def paired(
            self, observation: torch.Tensor, action: torch.Tensor,
            goal: torch.Tensor) -> torch.Tensor:
        sa_repr, goal_repr = self.representations(observation, action, goal)
        return self._energy(sa_repr, goal_repr)


class TDInfoNCECritic(Module):
    def __init__(
            self, *, state_dim: int, action_dim: int, goal_dim: int,
            hidden_sizes: Tuple[int, ...], representation_dim: int,
            representation_norm: bool, representation_temperature: float):
        super().__init__()
        sag_input_dim = state_dim + action_dim + goal_dim
        self.sag_encoders = nn.ModuleList([
            MLP(sag_input_dim, representation_dim, hidden_sizes=hidden_sizes)
            for _ in range(2)
        ])
        self.future_encoders = nn.ModuleList([
            MLP(goal_dim, representation_dim, hidden_sizes=hidden_sizes)
            for _ in range(2)
        ])
        for encoder in (*self.sag_encoders, *self.future_encoders):
            _initialize_mlp(encoder, 'td_infonce')
        self.representation_norm = representation_norm
        self.representation_temperature = representation_temperature

    def pairwise(
            self, observation: torch.Tensor, action: torch.Tensor,
            desired_goal: torch.Tensor,
            candidate_goals: torch.Tensor) -> torch.Tensor:
        sag_input = torch.cat([observation, action, desired_goal], dim=-1)
        outputs = []
        for sag_encoder, future_encoder in zip(
                self.sag_encoders, self.future_encoders):
            sag_repr = sag_encoder(sag_input)
            future_repr = future_encoder(candidate_goals)
            if self.representation_norm:
                sag_repr = sag_repr / (
                    torch.linalg.vector_norm(sag_repr, dim=-1, keepdim=True) + 1e-8
                )
                future_repr = future_repr / (
                    torch.linalg.vector_norm(future_repr, dim=-1, keepdim=True) + 1e-8
                )
                sag_repr = sag_repr / self.representation_temperature
            outputs.append(torch.einsum('ik,jk->ij', sag_repr, future_repr))
        return torch.stack(outputs, dim=-1)


class CLearningCritic(Module):
    def __init__(
            self, *, state_dim: int, action_dim: int, goal_dim: int,
            hidden_sizes: Tuple[int, ...]):
        super().__init__()
        input_dim = state_dim + action_dim + goal_dim
        self.classifiers = nn.ModuleList([
            MLP(input_dim, 1, hidden_sizes=hidden_sizes)
            for _ in range(2)
        ])

    def logits(
            self, observation: torch.Tensor, action: torch.Tensor,
            goal: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([observation, action, goal], dim=-1)
        return torch.cat([classifier(inputs) for classifier in self.classifiers], dim=-1)


class GCRLBaselineAgent(Module):
    def __init__(
            self, *, algorithm: str, actor: GoalConditionedPolicy,
            critic: Optional[Module], target_critic: Optional[Module],
            goal_dims: Tuple[int, ...]):
        super().__init__()
        self.algorithm = algorithm
        self.actor = actor
        self.add_module('critic', critic)
        self.add_module('target_critic', target_critic)
        self.goal_dims = tuple(goal_dims)

    def extract_goal(self, state_or_goal: torch.Tensor) -> torch.Tensor:
        index = torch.as_tensor(self.goal_dims, device=state_or_goal.device)
        return state_or_goal.index_select(-1, index)

    def act(
            self, observation: torch.Tensor,
            goal: torch.Tensor) -> torch.distributions.Distribution:
        return self.actor(observation, self.extract_goal(goal))


@contextmanager
def _frozen(module: Optional[nn.Module]):
    if module is None:
        yield
        return
    original = [parameter.requires_grad for parameter in module.parameters()]
    try:
        module.requires_grad_(False)
        yield
    finally:
        for parameter, requires_grad in zip(module.parameters(), original):
            parameter.requires_grad_(requires_grad)


class GCRLBaselineLosses(Module):
    goal_set_distance_loss = None

    def __init__(
            self, *, algorithm: str, agent: GCRLBaselineAgent,
            td_infonce_conf: TDInfoNCEConf, crl_conf: CRLConf,
            scaling_crl_conf: ScalingCRLConf,
            gcsl_conf: GCSLConf, c_learning_conf: CLearningConf,
            action_dim: int):
        super().__init__()
        self.algorithm = algorithm
        self.td_infonce_conf = td_infonce_conf
        self.crl_conf = crl_conf
        self.scaling_crl_conf = scaling_crl_conf
        self.gcsl_conf = gcsl_conf
        self.c_learning_conf = c_learning_conf
        self.action_dim = action_dim

        if algorithm == 'td_infonce':
            actor_lr = td_infonce_conf.actor_lr
            critic_lr = td_infonce_conf.critic_lr
        elif algorithm == 'crl':
            actor_lr = crl_conf.actor_lr
            critic_lr = crl_conf.critic_lr
        elif algorithm == 'scaling_crl':
            actor_lr = scaling_crl_conf.actor_lr
            critic_lr = scaling_crl_conf.critic_lr
        elif algorithm in ('gcbc', 'gcsl'):
            actor_lr = gcsl_conf.actor_lr
            critic_lr = None
        else:
            actor_lr = c_learning_conf.actor_lr
            critic_lr = c_learning_conf.critic_lr

        adam_eps = crl_conf.adam_eps if algorithm == 'crl' else 1e-8
        self.actor_optim = torch.optim.Adam(
            agent.actor.parameters(), lr=actor_lr, eps=adam_eps,
        )
        self.critic_optim = (
            None if agent.critic is None
            else torch.optim.Adam(
                agent.critic.parameters(), lr=critic_lr, eps=adam_eps,
            )
        )
        self.register_parameter('log_alpha', None)
        self.alpha_optim = None
        if algorithm == 'scaling_crl':
            self.log_alpha = nn.Parameter(torch.zeros(()))
            self.alpha_optim = torch.optim.Adam(
                (self.log_alpha,), lr=scaling_crl_conf.alpha_lr,
            )

    def set_scheduler_horizon(self, _total_optim_steps: int) -> None:
        # The reference implementations use constant learning rates.
        return None

    @staticmethod
    @torch.no_grad()
    def _soft_update(target: Optional[nn.Module], source: Optional[nn.Module], tau: float):
        if target is None or source is None:
            return
        for target_parameter, source_parameter in zip(
                target.parameters(), source.parameters()):
            target_parameter.lerp_(source_parameter, tau)

    def _td_infonce_losses(
            self, agent: GCRLBaselineAgent,
            data: BatchData) -> Tuple[torch.Tensor, torch.Tensor, InfoT]:
        critic = agent.critic
        target_critic = agent.target_critic
        assert isinstance(critic, TDInfoNCECritic)
        assert isinstance(target_critic, TDInfoNCECritic)
        conf = self.td_infonce_conf
        batch_size = data.num_transitions
        labels = torch.arange(batch_size, device=data.device)

        shift = int(torch.randint(batch_size, (), device=data.device).item())
        random_goal = agent.extract_goal(torch.roll(
            data.observations, shifts=shift, dims=0,
        ))
        next_goal = agent.extract_goal(data.next_observations)
        negative_goal = torch.roll(random_goal, shifts=-1, dims=0)

        positive_logits = critic.pairwise(
            data.observations, data.actions, random_goal, next_goal,
        )
        immediate_loss = torch.stack([
            F.cross_entropy(positive_logits[..., index], labels, reduction='none')
            for index in range(positive_logits.shape[-1])
        ], dim=-1)

        with torch.no_grad():
            next_action = agent.actor(data.next_observations, random_goal).sample()
            target_logits = target_critic.pairwise(
                data.next_observations, next_action, random_goal, negative_goal,
            ).amin(dim=-1)
            importance_weights = target_logits.softmax(dim=1)

        bootstrap_logits = critic.pairwise(
            data.observations, data.actions, random_goal, negative_goal,
        )
        bootstrap_loss = torch.stack([
            -(importance_weights * F.log_softmax(
                bootstrap_logits[..., index], dim=1,
            )).sum(dim=1)
            for index in range(bootstrap_logits.shape[-1])
        ], dim=-1)
        critic_loss = (
            (1 - conf.discount) * immediate_loss
            + conf.discount * bootstrap_loss
        ).mean()

        actor_goal = random_goal
        actor_dist = agent.actor(data.observations, actor_goal)
        actor_action = actor_dist.rsample()
        with _frozen(critic):
            actor_logits = critic.pairwise(
                data.observations, actor_action, actor_goal, actor_goal,
            ).amin(dim=-1)
            actor_loss = F.cross_entropy(actor_logits, labels)

        info = {
            'critic_loss': critic_loss,
            'actor_loss': actor_loss,
            'immediate_loss': immediate_loss.mean(),
            'bootstrap_loss': bootstrap_loss.mean(),
            'importance_diag': importance_weights.diagonal().mean(),
            'positive_logit': positive_logits.diagonal(dim1=0, dim2=1).mean(),
        }
        return critic_loss, actor_loss, info

    def _crl_losses(
            self, agent: GCRLBaselineAgent,
            data: BatchData) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], InfoT]:
        critic = agent.critic
        assert isinstance(critic, ContrastiveCritic)
        conf = self.crl_conf
        future_goal = agent.extract_goal(data.future_observations)
        logits = critic.pairwise(data.observations, data.actions, future_goal)
        labels = torch.arange(data.num_transitions, device=data.device)

        if conf.contrastive_loss == 'fwd_infonce':
            critic_loss = F.cross_entropy(logits, labels)
        elif conf.contrastive_loss == 'bwd_infonce':
            critic_loss = F.cross_entropy(logits.transpose(0, 1), labels)
        elif conf.contrastive_loss == 'sym_infonce':
            critic_loss = (
                F.cross_entropy(logits, labels)
                + F.cross_entropy(logits.transpose(0, 1), labels)
            )
        else:
            critic_loss = F.binary_cross_entropy_with_logits(
                logits, torch.eye(
                    data.num_transitions,
                    device=logits.device,
                    dtype=logits.dtype,
                ),
            )
        logsumexp = torch.logsumexp(logits, dim=1)
        critic_loss = critic_loss + conf.logsumexp_penalty * logsumexp.square().mean()

        if conf.random_goal_fraction == 0.0:
            actor_observation = data.observations
            actor_goal = future_goal
        elif conf.random_goal_fraction == 0.5:
            actor_observation = torch.cat([data.observations, data.observations], dim=0)
            actor_goal = torch.cat([future_goal, torch.roll(future_goal, 1, 0)], dim=0)
        else:
            actor_observation = data.observations
            actor_goal = torch.roll(future_goal, 1, 0)

        actor_dist = agent.actor(actor_observation, actor_goal)
        actor_action, log_prob = actor_dist.rsample_with_log_prob()
        with _frozen(critic):
            actor_value = critic.paired(actor_observation, actor_action, actor_goal)

        alpha = logits.new_tensor(conf.entropy_coefficient)
        alpha_loss = None
        actor_loss = (alpha * log_prob - actor_value).mean()
        info = {
            'critic_loss': critic_loss,
            'actor_loss': actor_loss,
            'binary_accuracy': (
                (logits > 0) == torch.eye(
                    data.num_transitions,
                    device=logits.device,
                    dtype=torch.bool,
                )
            ).to(torch.float32).mean(),
            'categorical_accuracy': (logits.argmax(1) == labels).to(torch.float32).mean(),
            'positive_logit': logits.diagonal().mean(),
            'negative_logit': (
                (logits.sum() - logits.diagonal().sum())
                / max(logits.numel() - data.num_transitions, 1)
            ),
            'logsumexp': logsumexp.square().mean(),
            'entropy': -log_prob.mean(),
            'alpha': alpha,
        }
        if alpha_loss is not None:
            info['alpha_loss'] = alpha_loss
        return critic_loss, actor_loss, alpha_loss, info

    def _scaling_crl_losses(
            self, agent: GCRLBaselineAgent,
            data: BatchData) -> Tuple[
                torch.Tensor, torch.Tensor, torch.Tensor, InfoT,
            ]:
        critic = agent.critic
        assert isinstance(critic, ScalingCRLCritic)
        assert self.log_alpha is not None
        conf = self.scaling_crl_conf
        future_goal = agent.extract_goal(data.future_observations)
        logits = critic.pairwise(data.observations, data.actions, future_goal)
        labels = torch.arange(data.num_transitions, device=data.device)

        critic_loss = F.cross_entropy(logits, labels)
        logsumexp = torch.logsumexp(logits + 1e-6, dim=1)
        critic_loss = (
            critic_loss
            + conf.logsumexp_penalty * logsumexp.square().mean()
        )

        actor_dist = agent.actor(data.observations, future_goal)
        # The reference computes the squashed-Gaussian Jacobian with an
        # explicit 1e-6 inside the logarithm. Its environments use unit action
        # bounds; retain the affine term so the common wrapper is also correct
        # for other finite Box bounds.
        pre_tanh_dist = actor_dist._pre_tanh_distn
        pre_tanh_action = pre_tanh_dist.rsample()
        unit_action = torch.tanh(pre_tanh_action)
        actor_action = (
            actor_dist._affine_loc
            + actor_dist._affine_scale * unit_action
        )
        log_prob = (
            pre_tanh_dist.log_prob(pre_tanh_action)
            - torch.log(1 - unit_action.square() + 1e-6)
            - actor_dist._affine_scale.abs().log()
        ).sum(dim=-1)
        with _frozen(critic):
            actor_value = critic.paired(
                data.observations, actor_action, future_goal,
            )

        alpha = self.log_alpha.exp()
        target_entropy = conf.target_entropy_per_action * self.action_dim
        actor_loss = (alpha.detach() * log_prob - actor_value).mean()
        alpha_loss = alpha * (-log_prob.detach() - target_entropy).mean()
        info = {
            'critic_loss': critic_loss,
            'actor_loss': actor_loss,
            'alpha_loss': alpha_loss,
            'categorical_accuracy': (
                logits.argmax(dim=1) == labels
            ).to(torch.float32).mean(),
            'positive_logit': logits.diagonal().mean(),
            'negative_logit': (
                (logits.sum() - logits.diagonal().sum())
                / max(logits.numel() - data.num_transitions, 1)
            ),
            'logsumexp': logsumexp.mean(),
            'entropy': -log_prob.mean(),
            'alpha': alpha,
        }
        return critic_loss, actor_loss, alpha_loss, info

    def _gcsl_loss(
            self, agent: GCRLBaselineAgent,
            data: BatchData) -> Tuple[torch.Tensor, InfoT]:
        future_goal = agent.extract_goal(data.future_observations)
        distribution = agent.actor(data.observations, future_goal)
        nll = -distribution.log_prob(data.actions)
        loss = nll.mean()
        return loss, {
            'actor_loss': loss,
            'negative_log_likelihood': nll.mean(),
        }

    def _c_learning_goals(
            self, agent: GCRLBaselineAgent,
            data: BatchData) -> torch.Tensor:
        batch_size = data.num_transitions
        conf = self.c_learning_conf
        num_next = round(batch_size * conf.relabel_next_probability)
        num_future = round(batch_size * conf.relabel_future_probability)
        half_batch = batch_size // 2
        if half_batch == 0:
            raise ValueError('C-Learning requires batch_size >= 2')
        if num_future != 0:
            raise NotImplementedError(
                'The source-default TD C-Learning port requires '
                'relabel_future_probability=0'
            )
        if num_next != half_batch:
            raise ValueError(
                f'Expected {half_batch} next goals for batch_size={batch_size}, '
                f'got {num_next}'
            )

        num_random = batch_size - num_next - num_future
        random_pool = data.observations[:num_random]
        permutation = torch.randperm(num_random, device=data.device)
        random_goal = agent.extract_goal(random_pool[permutation])
        next_goal = agent.extract_goal(data.next_observations[:num_next])
        return torch.cat([next_goal, random_goal], dim=0)

    def _c_learning_critic_loss(
            self, agent: GCRLBaselineAgent, data: BatchData,
            goals: torch.Tensor) -> Tuple[torch.Tensor, InfoT]:
        critic = agent.critic
        target_critic = agent.target_critic
        assert isinstance(critic, CLearningCritic)
        assert isinstance(target_critic, CLearningCritic)
        conf = self.c_learning_conf
        batch_size = data.num_transitions
        half_batch = batch_size // 2

        with torch.no_grad():
            next_action = agent.actor(data.next_observations, goals).sample()
            target_probability = target_critic.logits(
                data.next_observations, next_action, goals,
            ).sigmoid().amin(dim=-1)
            odds = target_probability / (1 - target_probability)
            odds = odds.clamp(max=conf.odds_clip)
            recursive_target = conf.discount * odds / (1 + conf.discount * odds)
            next_achieved_goal = agent.extract_goal(data.next_observations)
            reward = (next_achieved_goal == goals).all(dim=-1).to(odds.dtype)
            targets = reward + (1 - reward) * recursive_target
            sample_weights = torch.cat([
                torch.full_like(odds[:half_batch], 1 - conf.discount),
                1 + conf.discount * odds[half_batch:],
            ])

        logits = critic.logits(data.observations, data.actions, goals)
        per_classifier_loss = F.binary_cross_entropy_with_logits(
            logits, targets[:, None].expand_as(logits), reduction='none',
        )
        critic_loss = conf.critic_loss_weight * (
            sample_weights * per_classifier_loss.sum(dim=-1)
        ).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'classifier_probability': logits.sigmoid().mean(),
            'target_probability': targets.mean(),
            'importance_weight': sample_weights.mean(),
            'odds': odds.mean(),
        }

    def _c_learning_actor_loss(
            self, agent: GCRLBaselineAgent, data: BatchData,
            goals: torch.Tensor) -> Tuple[torch.Tensor, InfoT]:
        critic = agent.critic
        assert isinstance(critic, CLearningCritic)

        actor_dist = agent.actor(data.observations, goals)
        actor_action = actor_dist.rsample()
        with _frozen(critic):
            actor_probability = critic.logits(
                data.observations, actor_action, goals,
            ).sigmoid().amin(dim=-1)
            actor_loss = -actor_probability.mean()

        return actor_loss, {'actor_loss': actor_loss}

    def _c_learning_losses(
            self, agent: GCRLBaselineAgent,
            data: BatchData) -> Tuple[torch.Tensor, torch.Tensor, InfoT]:
        goals = self._c_learning_goals(agent, data)
        critic_loss, critic_info = self._c_learning_critic_loss(
            agent, data, goals,
        )
        actor_loss, actor_info = self._c_learning_actor_loss(
            agent, data, goals,
        )

        info = {**critic_info, **actor_info}
        return critic_loss, actor_loss, info

    def forward(
            self, agent: GCRLBaselineAgent, data: BatchData, *,
            optimize: bool = True, phase: str = 'all') -> LossResult:
        if phase != 'all':
            raise ValueError(
                f'{self.algorithm} only supports phase="all", got {phase!r}'
            )
        if self.critic_optim is not None:
            self.critic_optim.zero_grad()
        self.actor_optim.zero_grad()
        if self.alpha_optim is not None:
            self.alpha_optim.zero_grad()

        critic_already_optimized = False
        if self.algorithm == 'td_infonce':
            critic_loss, actor_loss, info = self._td_infonce_losses(agent, data)
            alpha_loss = None
        elif self.algorithm == 'crl':
            critic_loss, actor_loss, alpha_loss, info = self._crl_losses(agent, data)
        elif self.algorithm == 'scaling_crl':
            critic_loss, actor_loss, alpha_loss, info = (
                self._scaling_crl_losses(agent, data)
            )
        elif self.algorithm in ('gcbc', 'gcsl'):
            actor_loss, info = self._gcsl_loss(agent, data)
            critic_loss = actor_loss.new_zeros(())
            alpha_loss = None
        else:
            goals = self._c_learning_goals(agent, data)
            critic_loss, critic_info = self._c_learning_critic_loss(
                agent, data, goals,
            )
            if optimize:
                assert self.critic_optim is not None
                critic_loss.backward()
                self.critic_optim.step()
                critic_already_optimized = True
            actor_loss, actor_info = self._c_learning_actor_loss(
                agent, data, goals,
            )
            info = {**critic_info, **actor_info}
            alpha_loss = None

        if optimize:
            if self.critic_optim is not None and not critic_already_optimized:
                critic_loss.backward()
            actor_loss.backward()
            if alpha_loss is not None:
                alpha_loss.backward()
            if self.critic_optim is not None and not critic_already_optimized:
                self.critic_optim.step()
            self.actor_optim.step()
            if self.alpha_optim is not None:
                self.alpha_optim.step()

            if self.algorithm == 'td_infonce':
                self._soft_update(
                    agent.target_critic, agent.critic, self.td_infonce_conf.tau,
                )
            elif self.algorithm == 'c_learning':
                self._soft_update(
                    agent.target_critic, agent.critic, self.c_learning_conf.tau,
                )

        total_loss = critic_loss.detach() + actor_loss.detach()
        if alpha_loss is not None:
            total_loss = total_loss + alpha_loss.detach()
        return LossResult(loss=total_loss, info={self.algorithm: info})

    def state_dict(self, *args, **kwargs):
        state = {
            'module': super().state_dict(*args, **kwargs),
            'actor_optim': self.actor_optim.state_dict(),
        }
        if self.critic_optim is not None:
            state['critic_optim'] = self.critic_optim.state_dict()
        if self.alpha_optim is not None:
            state['alpha_optim'] = self.alpha_optim.state_dict()
        return state

    def load_state_dict(
            self, state_dict: Mapping[str, Any], strict: bool = True):
        result = super().load_state_dict(state_dict['module'], strict=strict)
        self.actor_optim.load_state_dict(state_dict['actor_optim'])
        if self.critic_optim is not None:
            self.critic_optim.load_state_dict(state_dict['critic_optim'])
        if self.alpha_optim is not None:
            self.alpha_optim.load_state_dict(state_dict['alpha_optim'])
        return result


__all__ = [
    'BASELINE_ALGORITHMS',
    'SCALING_CRL_REFERENCE_REVISION',
    'GCRLBaselinesConf',
    'GCRLBaselineAgent',
    'GCRLBaselineLosses',
    'GCSLConf',
    'ScalingCRLConf',
    'ScalingCRLCritic',
]

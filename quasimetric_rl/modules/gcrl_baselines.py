from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Mapping, Optional, Tuple

import attrs
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data import BatchData, EnvSpec
from .utils import InfoT, LossResult, MLP, Module


BASELINE_ALGORITHMS = ('td_infonce', 'crl', 'gcbc', 'gcsl', 'c_learning')


def _positive_tuple(value):
    value = tuple(int(item) for item in value)
    if not value or min(value) <= 0:
        raise ValueError(f'Expected positive hidden sizes, got {value!r}')
    return value


@attrs.define(kw_only=True)
class TDInfoNCEConf:
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


@attrs.define(kw_only=True)
class CRLConf:
    hidden_sizes: Tuple[int, ...] = attrs.field(
        default=(256, 256), converter=_positive_tuple,
    )
    representation_dim: int = attrs.field(default=64, validator=attrs.validators.gt(0))
    actor_lr: float = attrs.field(default=3e-4, validator=attrs.validators.gt(0))
    critic_lr: float = attrs.field(default=3e-4, validator=attrs.validators.gt(0))
    alpha_lr: float = attrs.field(default=3e-4, validator=attrs.validators.gt(0))
    discount: float = attrs.field(
        default=0.99,
        validator=attrs.validators.and_(attrs.validators.ge(0), attrs.validators.le(1)),
    )
    contrastive_loss: str = attrs.field(
        default='fwd_infonce',
        validator=attrs.validators.in_((
            'fwd_infonce', 'bwd_infonce', 'sym_infonce', 'binary_nce',
        )),
    )
    energy: str = attrs.field(
        default='norm',
        validator=attrs.validators.in_(('norm', 'l2', 'dot', 'cosine')),
    )
    logsumexp_penalty: float = attrs.field(default=0.1, validator=attrs.validators.ge(0))
    activation: str = attrs.field(
        default='silu', validator=attrs.validators.in_(('relu', 'silu')),
    )
    representation_norm: bool = False
    # None enables the JaxGCRL adaptive entropy coefficient. Set 0 for the
    # original vector-observation CRL configuration.
    entropy_coefficient: Optional[float] = attrs.field(
        default=None, validator=attrs.validators.optional(attrs.validators.ge(0)),
    )
    target_entropy_per_action: float = -0.5
    # JaxGCRL uses only achieved future goals for the actor. The original CRL
    # implementation uses an equal mixture of future and shuffled goals.
    random_goal_fraction: float = attrs.field(
        default=0.0, validator=attrs.validators.in_((0.0, 0.5, 1.0)),
    )


@attrs.define(kw_only=True)
class GCBCConf:
    hidden_sizes: Tuple[int, ...] = attrs.field(
        default=(400, 300), converter=_positive_tuple,
    )
    actor_lr: float = attrs.field(default=5e-4, validator=attrs.validators.gt(0))


@attrs.define(kw_only=True)
class CLearningConf:
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


@attrs.define(kw_only=True)
class GCRLBaselinesConf:
    td_infonce: TDInfoNCEConf = TDInfoNCEConf()
    crl: CRLConf = CRLConf()
    gcbc: GCBCConf = GCBCConf()
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

        if algorithm == 'td_infonce':
            conf = self.td_infonce
            actor_hidden_sizes = conf.hidden_sizes
            actor_activation = 'relu'
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
        elif algorithm in ('gcbc', 'gcsl'):
            conf = self.gcbc
            actor_hidden_sizes = conf.hidden_sizes
            actor_activation = 'relu'
            critic = target_critic = None
        else:
            conf = self.c_learning
            actor_hidden_sizes = conf.hidden_sizes
            actor_activation = 'relu'
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
            gcbc_conf=self.gcbc,
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


class GoalConditionedPolicy(Module):
    def __init__(
            self, *, env_spec: EnvSpec, goal_dim: int,
            hidden_sizes: Tuple[int, ...], activation: str):
        super().__init__()
        state_dim = env_spec.observation_shape.numel()
        self.backbone = MLP(
            state_dim + goal_dim,
            env_spec.make_action_output_distn().input_size,
            hidden_sizes=hidden_sizes,
            activation_fn=_activation_type(activation),
        )
        self.action_output = env_spec.make_action_output_distn()

    def forward(
            self, observation: torch.Tensor,
            goal: torch.Tensor) -> torch.distributions.Distribution:
        return self.action_output(self.backbone(torch.cat([observation, goal], dim=-1)))


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
            return F.cosine_similarity(left, right, dim=-1, eps=1e-6)
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
                sag_repr = F.normalize(sag_repr, dim=-1)
                future_repr = F.normalize(future_repr, dim=-1)
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
            gcbc_conf: GCBCConf, c_learning_conf: CLearningConf,
            action_dim: int):
        super().__init__()
        self.algorithm = algorithm
        self.td_infonce_conf = td_infonce_conf
        self.crl_conf = crl_conf
        self.gcbc_conf = gcbc_conf
        self.c_learning_conf = c_learning_conf
        self.action_dim = action_dim

        if algorithm == 'td_infonce':
            actor_lr = td_infonce_conf.actor_lr
            critic_lr = td_infonce_conf.critic_lr
        elif algorithm == 'crl':
            actor_lr = crl_conf.actor_lr
            critic_lr = crl_conf.critic_lr
        elif algorithm in ('gcbc', 'gcsl'):
            actor_lr = gcbc_conf.actor_lr
            critic_lr = None
        else:
            actor_lr = c_learning_conf.actor_lr
            critic_lr = c_learning_conf.critic_lr

        self.actor_optim = torch.optim.Adam(agent.actor.parameters(), lr=actor_lr)
        self.critic_optim = (
            None if agent.critic is None
            else torch.optim.Adam(agent.critic.parameters(), lr=critic_lr)
        )
        if algorithm == 'crl' and crl_conf.entropy_coefficient is None:
            self.log_alpha = nn.Parameter(torch.zeros(()))
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=crl_conf.alpha_lr)
        else:
            self.register_parameter('log_alpha', None)
            self.alpha_optim = None

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

        permutation = torch.randperm(batch_size, device=data.device)
        random_goal = agent.extract_goal(data.observations[permutation])
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
            identity = torch.eye(
                data.num_transitions, device=data.device, dtype=logits.dtype,
            )
            critic_loss = F.binary_cross_entropy_with_logits(logits, identity)
        logsumexp = torch.logsumexp(logits + 1e-6, dim=1)
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
        actor_action = actor_dist.rsample()
        log_prob = actor_dist.log_prob(actor_action)
        with _frozen(critic):
            actor_value = critic.paired(actor_observation, actor_action, actor_goal)

        if self.log_alpha is None:
            alpha = logits.new_tensor(conf.entropy_coefficient)
            alpha_loss = None
        else:
            alpha = self.log_alpha.exp()
            target_entropy = conf.target_entropy_per_action * self.action_dim
            alpha_loss = alpha * (-log_prob.detach() - target_entropy).mean()
        actor_loss = (alpha.detach() * log_prob - actor_value).mean()
        info = {
            'critic_loss': critic_loss,
            'actor_loss': actor_loss,
            'categorical_accuracy': (logits.argmax(1) == labels).to(torch.float32).mean(),
            'positive_logit': logits.diagonal().mean(),
            'negative_logit': (
                (logits.sum() - logits.diagonal().sum())
                / max(logits.numel() - data.num_transitions, 1)
            ),
            'logsumexp': logsumexp.mean(),
            'entropy': -log_prob.mean(),
            'alpha': alpha,
        }
        if alpha_loss is not None:
            info['alpha_loss'] = alpha_loss
        return critic_loss, actor_loss, alpha_loss, info

    def _gcbc_loss(
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

    def _c_learning_losses(
            self, agent: GCRLBaselineAgent,
            data: BatchData) -> Tuple[torch.Tensor, torch.Tensor, InfoT]:
        critic = agent.critic
        target_critic = agent.target_critic
        assert isinstance(critic, CLearningCritic)
        assert isinstance(target_critic, CLearningCritic)
        conf = self.c_learning_conf
        batch_size = data.num_transitions
        half_batch = batch_size // 2
        if half_batch == 0:
            raise ValueError('C-Learning requires batch_size >= 2')

        num_random = batch_size - half_batch
        random_pool = data.observations[:num_random]
        permutation = torch.randperm(num_random, device=data.device)
        random_goal = agent.extract_goal(random_pool[permutation])
        next_goal = agent.extract_goal(data.next_observations[:half_batch])
        goals = torch.cat([next_goal, random_goal], dim=0)

        with torch.no_grad():
            next_action = agent.actor(data.next_observations, goals).sample()
            target_probability = target_critic.logits(
                data.next_observations, next_action, goals,
            ).sigmoid().amin(dim=-1)
            odds = target_probability / (1 - target_probability).clamp_min(1e-6)
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
        critic_loss = (sample_weights * per_classifier_loss.sum(dim=-1)).mean()

        actor_dist = agent.actor(data.observations, goals)
        actor_action = actor_dist.rsample()
        with _frozen(critic):
            actor_probability = critic.logits(
                data.observations, actor_action, goals,
            ).sigmoid().amin(dim=-1)
            actor_loss = -actor_probability.mean()

        info = {
            'critic_loss': critic_loss,
            'actor_loss': actor_loss,
            'classifier_probability': logits.sigmoid().mean(),
            'target_probability': targets.mean(),
            'importance_weight': sample_weights.mean(),
            'odds': odds.mean(),
        }
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

        if self.algorithm == 'td_infonce':
            critic_loss, actor_loss, info = self._td_infonce_losses(agent, data)
            alpha_loss = None
        elif self.algorithm == 'crl':
            critic_loss, actor_loss, alpha_loss, info = self._crl_losses(agent, data)
        elif self.algorithm in ('gcbc', 'gcsl'):
            actor_loss, info = self._gcbc_loss(agent, data)
            critic_loss = actor_loss.new_zeros(())
            alpha_loss = None
        else:
            critic_loss, actor_loss, info = self._c_learning_losses(agent, data)
            alpha_loss = None

        if optimize:
            if self.critic_optim is not None:
                critic_loss.backward()
            actor_loss.backward()
            if alpha_loss is not None:
                alpha_loss.backward()
            if self.critic_optim is not None:
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
    'GCRLBaselinesConf',
    'GCRLBaselineAgent',
    'GCRLBaselineLosses',
]

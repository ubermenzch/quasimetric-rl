from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import ceil, gcd, sqrt
from pathlib import Path
import re
from typing import Dict, Iterable, Tuple

from omegaconf import DictConfig, OmegaConf


MODEL_SIZE_ROOT = Path(__file__).resolve().parents[1] / 'configs' / 'model_size'
# S/M/L remain the automatic latent-capacity choices. Larger levels are explicit
# scaling tiers because they deliberately keep the L latent size.
MODEL_SIZE_LEVELS: Tuple[str, ...] = ('s', 'm', 'l')
QRL_MODEL_SIZE_LEVELS: Tuple[str, ...] = ('s', 'm', 'l', 'xl', 'xxl', 'xxxl')
SCALING_CRL_MODEL_SIZE_LEVELS: Tuple[str, ...] = ('m', 'l', 'xl', 'xxl', 'xxxl')
# Backward-compatible import for scripts created before Base was renamed QRL.
BASE_MODEL_SIZE_LEVELS = MODEL_SIZE_LEVELS


@dataclass(frozen=True)
class GOQRLBranchPlan:
    goal_arch: Tuple[int, ...]
    non_goal_arch: Tuple[int, ...]
    goal_latent_size: int
    non_goal_latent_size: int
    qrl_encoder_parameters: int
    goal_encoder_parameters: int
    non_goal_encoder_parameters: int
    goal_parameter_weight: int
    non_goal_parameter_weight: int
    minimum_ratio_applied: bool


def model_size_path(family: str, level: str) -> Path:
    family = family.lower().replace('-', '_')
    if family == 'base':
        family = 'qrl'
    return MODEL_SIZE_ROOT / family / f'{level.lower()}.yaml'


def load_model_size_preset(family: str, level: str) -> DictConfig:
    path = model_size_path(family, level)
    if not path.is_file():
        raise ValueError(
            f'Unknown {family!r} model-size level {level!r}; '
            f'expected a preset at {path}'
        )
    return OmegaConf.load(path)


def register_model_size_presets(config_store) -> None:
    """Expose editable YAML presets as Hydra config groups."""
    for family_dir in sorted(path for path in MODEL_SIZE_ROOT.iterdir() if path.is_dir()):
        groups = [f'{family_dir.name}_model_size']
        if family_dir.name == 'qrl':
            groups.append('base_model_size')
        for path in sorted(family_dir.glob('*.yaml')):
            node = OmegaConf.load(path)
            for group in groups:
                config_store.store(
                    group=group,
                    name=path.stem,
                    node=node,
                    package='agent',
                )


def select_qrl_model_size(state_dim: int) -> str:
    """Choose the smallest level whose half-latent strictly exceeds state_dim."""
    if state_dim <= 0:
        raise ValueError(f'state_dim must be positive, got {state_dim}')
    for level in MODEL_SIZE_LEVELS:
        preset = load_model_size_preset('qrl', level)
        latent_size = int(preset.quasimetric_critic.model.encoder.latent_size)
        if latent_size / 2 > state_dim:
            return level
    largest = load_model_size_preset('qrl', MODEL_SIZE_LEVELS[-1])
    largest_latent = int(largest.quasimetric_critic.model.encoder.latent_size)
    raise ValueError(
        f'No QRL model-size level can represent state_dim={state_dim} under '
        f'the latent_size / 2 > state_dim rule; largest latent is {largest_latent}'
    )


# Compatibility with the initial Base-S/M/L naming.
select_base_model_size = select_qrl_model_size


def _mlp_parameter_count(input_size: int, hidden_sizes, output_size: int) -> int:
    sizes = (input_size, *map(int, hidden_sizes), output_size)
    return sum(
        (input_dim + 1) * output_dim
        for input_dim, output_dim in zip(sizes, sizes[1:])
    )


def _residual_mlp_parameter_count(
        input_size: int, hidden_sizes, output_size: int,
        residual_block_size: int = 4) -> int:
    hidden_sizes = tuple(map(int, hidden_sizes))
    if not hidden_sizes or len(set(hidden_sizes)) != 1:
        raise ValueError(
            'Residual MLP parameter counting requires a non-empty constant-width '
            f'architecture, got {hidden_sizes}'
        )
    if residual_block_size <= 0 or len(hidden_sizes) % residual_block_size:
        raise ValueError(
            'Residual MLP depth must be divisible by residual_block_size, got '
            f'depth={len(hidden_sizes)}, block_size={residual_block_size}'
        )
    width = hidden_sizes[0]
    depth = len(hidden_sizes)
    # Input projection and every residual Dense have affine LayerNorm parameters.
    return (
        (input_size + 1) * width + 2 * width
        + depth * ((width + 1) * width + 2 * width)
        + (width + 1) * output_size
    )


def _configured_mlp_parameter_count(
        input_size: int, hidden_sizes, output_size: int, *,
        kind: str = 'plain', residual_block_size: int = 4) -> int:
    if kind == 'plain':
        return _mlp_parameter_count(input_size, hidden_sizes, output_size)
    if kind == 'residual':
        return _residual_mlp_parameter_count(
            input_size,
            hidden_sizes,
            output_size,
            residual_block_size,
        )
    raise ValueError(f'Unknown MLP kind {kind!r}')


def _iqe_head_dim(quasimetric) -> int:
    head_match = re.fullmatch(
        r'iqe\(dim=(\d+),components=(\d+)\)',
        str(quasimetric.quasimetric_head_spec).replace(' ', ''),
    )
    if head_match is None:
        raise ValueError(
            'Model-size parameter counting currently supports only the '
            'iqe(dim=...,components=...) head'
        )
    head_dim, components = map(int, head_match.groups())
    if head_dim % components != 0:
        raise ValueError(f'IQE dim {head_dim} is not divisible by {components} components')
    return head_dim


def qrl_agent_parameter_count(state_dim: int, action_dim: int, level: str) -> int:
    """Count trainable Agent parameters for a vector QRL preset."""
    if state_dim <= 0 or action_dim <= 0:
        raise ValueError(
            f'state_dim and action_dim must be positive, got {state_dim}, {action_dim}'
        )
    preset = load_model_size_preset('qrl', level)
    if int(preset.num_critics) != 1:
        raise ValueError('QRL model-size parameter counting requires one critic')
    encoder = preset.quasimetric_critic.model.encoder
    quasimetric = preset.quasimetric_critic.model.quasimetric_model
    dynamics = preset.quasimetric_critic.model.latent_dynamics
    actor = preset.actor.model
    if encoder.kind != 'standard' or actor.input_mode != 'raw':
        raise ValueError('QRL model-size parameter counting requires standard/raw architecture')

    latent_size = int(encoder.latent_size)
    head_dim = _iqe_head_dim(quasimetric)
    encoder_count = _configured_mlp_parameter_count(
        state_dim,
        encoder.arch,
        latent_size,
        kind=str(encoder.get('mlp_kind', 'plain')),
        residual_block_size=int(encoder.get('residual_block_size', 4)),
    )
    projector_count = _configured_mlp_parameter_count(
        latent_size,
        quasimetric.projector_arch,
        head_dim,
        kind=str(quasimetric.get('projector_mlp_kind', 'plain')),
        residual_block_size=int(
            quasimetric.get('projector_residual_block_size', 4)
        ),
    )
    dynamics_count = _configured_mlp_parameter_count(
        latent_size + action_dim,
        dynamics.arch,
        latent_size,
        kind=str(dynamics.get('mlp_kind', 'plain')),
        residual_block_size=int(dynamics.get('residual_block_size', 4)),
    )
    actor_count = _configured_mlp_parameter_count(
        2 * state_dim,
        actor.arch,
        2 * action_dim,
        kind=str(actor.get('mlp_kind', 'plain')),
        residual_block_size=int(actor.get('residual_block_size', 4)),
    )
    # IQE max-mean reduction contributes one trainable scalar.
    return encoder_count + projector_count + 1 + dynamics_count + actor_count


base_agent_parameter_count = qrl_agent_parameter_count


def scaling_crl_agent_parameter_count(
        state_dim: int, action_dim: int, goal_dim: int, level: str) -> int:
    """Count actor, both encoders, and entropy scalar for Scaling-CRL."""
    if min(state_dim, action_dim, goal_dim) <= 0:
        raise ValueError(
            'state_dim, action_dim, and goal_dim must be positive, got '
            f'{state_dim}, {action_dim}, {goal_dim}'
        )
    preset = load_model_size_preset('scaling_crl', level)
    baseline = preset.baselines.scaling_crl
    hidden_sizes = tuple(map(int, baseline.hidden_sizes))
    representation_dim = 64
    actor_count = _residual_mlp_parameter_count(
        state_dim + goal_dim, hidden_sizes, 2 * action_dim,
    )
    sa_encoder_count = _residual_mlp_parameter_count(
        state_dim + action_dim, hidden_sizes, representation_dim,
    )
    goal_encoder_count = _residual_mlp_parameter_count(
        goal_dim, hidden_sizes, representation_dim,
    )
    return actor_count + sa_encoder_count + goal_encoder_count + 1


def _branch_parameter_weights(
        goal_dim: int, non_goal_dim: int, min_goal_ratio: float,
) -> Tuple[int, int, bool]:
    if min_goal_ratio <= 0:
        raise ValueError(f'min_goal_ratio must be positive, got {min_goal_ratio}')
    raw_ratio = goal_dim / non_goal_dim
    if raw_ratio >= min_goal_ratio:
        divisor = gcd(goal_dim, non_goal_dim)
        return goal_dim // divisor, non_goal_dim // divisor, False

    # The supported presets use the exact, auditable 1:8 floor.
    if abs(min_goal_ratio - 0.125) > 1e-12:
        raise ValueError(
            'Automatic GO-QRL branch matching currently requires '
            f'min_goal_ratio=0.125, got {min_goal_ratio}'
        )
    return 1, 8, True


@lru_cache(maxsize=None)
def _branch_architecture_candidates(
        input_size: int, output_size: int,
        ideal_first: int, ideal_second: int, radius_fraction: float,
) -> Dict[int, Tuple[float, Tuple[int, int]]]:
    first_radius = max(48, ceil(ideal_first * radius_fraction))
    second_radius = max(48, ceil(ideal_second * radius_fraction))
    candidates: Dict[int, Tuple[float, Tuple[int, int]]] = {}
    for first in range(max(1, ideal_first - first_radius), ideal_first + first_radius + 1):
        for second in range(
                max(1, ideal_second - second_radius),
                ideal_second + second_radius + 1):
            count = _mlp_parameter_count(input_size, (first, second), output_size)
            score = (
                (first - ideal_first) ** 2
                + (second - ideal_second) ** 2
                + 0.2 * (first - second) ** 2
            )
            existing = candidates.get(count)
            if existing is None or score < existing[0]:
                candidates[count] = (score, (first, second))
    return candidates


def match_go_qrl_split_encoder(
        state_dim: int, goal_dim: int, latent_size: int,
        reference_arch: Iterable[int], min_goal_ratio: float = 0.125,
        max_relative_budget_error: float = 0.0025,
        mlp_kind: str = 'plain', residual_block_size: int = 4,
) -> GOQRLBranchPlan:
    """Match two split MLPs to a QRL encoder's parameter budget.

    Two-layer plain MLPs are matched exactly. Deeper plain and residual MLP
    branches remain constant-width, so their nearest realizable total may
    differ slightly.
    """
    reference_arch = tuple(map(int, reference_arch))
    if state_dim <= 1 or not 0 < goal_dim < state_dim:
        raise ValueError(
            f'Expected 0 < goal_dim < state_dim, got {goal_dim}, {state_dim}'
        )
    if latent_size <= 1:
        raise ValueError(f'latent_size must exceed one, got {latent_size}')
    if len(reference_arch) < 2 or min(reference_arch) <= 0:
        raise ValueError(
            'Automatic GO-QRL branch matching requires at least two positive '
            'reference hidden widths, '
            f'got {reference_arch}'
        )

    non_goal_dim = state_dim - goal_dim
    goal_weight, non_goal_weight, floor_applied = _branch_parameter_weights(
        goal_dim, non_goal_dim, min_goal_ratio,
    )
    total_weight = goal_weight + non_goal_weight
    goal_share = goal_weight / total_weight
    goal_latent_size = min(
        latent_size - 1,
        ceil(latent_size * goal_weight / total_weight),
    )
    non_goal_latent_size = latent_size - goal_latent_size
    qrl_parameters = _configured_mlp_parameter_count(
        state_dim,
        reference_arch,
        latent_size,
        kind=mlp_kind,
        residual_block_size=residual_block_size,
    )
    target_goal_parameters = qrl_parameters * goal_share

    if mlp_kind == 'residual' or (
            mlp_kind == 'plain' and len(reference_arch) > 2):
        if len(set(reference_arch)) != 1:
            raise ValueError(
                'Deep plain and residual GO-QRL matching require a constant '
                'reference width, '
                f'got {reference_arch}'
            )
        reference_width = reference_arch[0]
        depth = len(reference_arch)

        def nearest_widths(input_size: int, output_size: int, target: float):
            candidates = []
            for width in range(1, 2 * reference_width + 1):
                arch = (width,) * depth
                count = _configured_mlp_parameter_count(
                    input_size,
                    arch,
                    output_size,
                    kind=mlp_kind,
                    residual_block_size=residual_block_size,
                )
                candidates.append((abs(count - target), width, count))
            return sorted(candidates)[:32]

        goal_candidates = nearest_widths(
            goal_dim, goal_latent_size, target_goal_parameters,
        )
        non_goal_candidates = nearest_widths(
            non_goal_dim,
            non_goal_latent_size,
            qrl_parameters - target_goal_parameters,
        )
        feasible = []
        goal_budget_tolerance = max_relative_budget_error
        if mlp_kind == 'plain':
            goal_budget_tolerance = max(goal_budget_tolerance, 0.003)
        for _goal_error, goal_width, goal_parameters in goal_candidates:
            for _non_goal_error, non_goal_width, non_goal_parameters in non_goal_candidates:
                if goal_parameters / non_goal_parameters < min_goal_ratio:
                    continue
                total_error = abs(
                    goal_parameters + non_goal_parameters - qrl_parameters
                ) / qrl_parameters
                goal_error = abs(
                    goal_parameters - target_goal_parameters
                ) / target_goal_parameters
                if goal_error > goal_budget_tolerance:
                    continue
                feasible.append((
                    total_error,
                    goal_error,
                    abs(goal_width - reference_width * sqrt(goal_share))
                    + abs(non_goal_width - reference_width * sqrt(1 - goal_share)),
                    goal_width,
                    non_goal_width,
                    goal_parameters,
                    non_goal_parameters,
                ))
        if not feasible:
            raise ValueError(
                f'Could not construct {mlp_kind} GO-QRL encoder branches for '
                f'state_dim={state_dim}, goal_dim={goal_dim}'
            )
        (
            _total_error, _goal_error, _width_error,
            goal_width, non_goal_width,
            goal_parameters, non_goal_parameters,
        ) = min(feasible)
        return GOQRLBranchPlan(
            goal_arch=(goal_width,) * depth,
            non_goal_arch=(non_goal_width,) * depth,
            goal_latent_size=goal_latent_size,
            non_goal_latent_size=non_goal_latent_size,
            qrl_encoder_parameters=qrl_parameters,
            goal_encoder_parameters=goal_parameters,
            non_goal_encoder_parameters=non_goal_parameters,
            goal_parameter_weight=goal_weight,
            non_goal_parameter_weight=non_goal_weight,
            minimum_ratio_applied=floor_applied,
        )
    if mlp_kind != 'plain':
        raise ValueError(f'Unknown GO-QRL encoder MLP kind {mlp_kind!r}')
    if len(reference_arch) != 2:
        raise ValueError(
            'Plain GO-QRL branch matching requires two reference hidden widths, '
            f'got {reference_arch}'
        )

    ideal_goal = tuple(
        max(1, round(width * sqrt(goal_share))) for width in reference_arch
    )
    ideal_non_goal = tuple(
        max(1, round(width * sqrt(1 - goal_share))) for width in reference_arch
    )

    for radius_fraction in (0.25, 0.4, 0.65, 1.0):
        goal_candidates = _branch_architecture_candidates(
            goal_dim, goal_latent_size, *ideal_goal, radius_fraction,
        )
        non_goal_candidates = _branch_architecture_candidates(
            non_goal_dim, non_goal_latent_size, *ideal_non_goal, radius_fraction,
        )
        feasible = []
        for goal_parameters, (goal_score, goal_arch) in goal_candidates.items():
            non_goal_parameters = qrl_parameters - goal_parameters
            non_goal_candidate = non_goal_candidates.get(non_goal_parameters)
            if non_goal_candidate is None:
                continue
            if goal_parameters / non_goal_parameters < min_goal_ratio:
                continue
            relative_error = (
                abs(goal_parameters - target_goal_parameters)
                / target_goal_parameters
            )
            if relative_error > max_relative_budget_error:
                continue
            non_goal_score, non_goal_arch = non_goal_candidate
            feasible.append((
                relative_error,
                goal_score + non_goal_score,
                goal_arch,
                non_goal_arch,
                goal_parameters,
                non_goal_parameters,
            ))
        if feasible:
            (
                _error, _score, goal_arch, non_goal_arch,
                goal_parameters, non_goal_parameters,
            ) = min(feasible)
            return GOQRLBranchPlan(
                goal_arch=goal_arch,
                non_goal_arch=non_goal_arch,
                goal_latent_size=goal_latent_size,
                non_goal_latent_size=non_goal_latent_size,
                qrl_encoder_parameters=qrl_parameters,
                goal_encoder_parameters=goal_parameters,
                non_goal_encoder_parameters=non_goal_parameters,
                goal_parameter_weight=goal_weight,
                non_goal_parameter_weight=non_goal_weight,
                minimum_ratio_applied=floor_applied,
            )

    raise ValueError(
        'Could not construct a two-branch GO-QRL encoder with exact QRL total '
        f'parameters for state_dim={state_dim}, goal_dim={goal_dim}, '
        f'latent_size={latent_size}, reference_arch={reference_arch}'
    )


def go_qrl_agent_parameter_count(
        state_dim: int, action_dim: int, goal_dim: int, level: str,
) -> int:
    """Count GO-QRL parameters after resolving its matched split encoder."""
    preset = load_model_size_preset('go_qrl', level)
    encoder = preset.quasimetric_critic.model.encoder
    quasimetric = preset.quasimetric_critic.model.quasimetric_model
    dynamics = preset.quasimetric_critic.model.latent_dynamics
    actor = preset.actor.model
    if encoder.kind != 'split' or actor.input_mode != 'split_latent':
        raise ValueError('GO-QRL counting requires split/split_latent architecture')
    plan = match_go_qrl_split_encoder(
        state_dim=state_dim,
        goal_dim=goal_dim,
        latent_size=int(encoder.latent_size),
        reference_arch=encoder.arch,
        min_goal_ratio=float(encoder.min_goal_parameter_ratio),
        mlp_kind=str(encoder.get('mlp_kind', 'plain')),
        residual_block_size=int(encoder.get('residual_block_size', 4)),
    )
    latent_size = int(encoder.latent_size)
    head_dim = _iqe_head_dim(quasimetric)
    projector_count = _configured_mlp_parameter_count(
        latent_size,
        quasimetric.projector_arch,
        head_dim,
        kind=str(quasimetric.get('projector_mlp_kind', 'plain')),
        residual_block_size=int(
            quasimetric.get('projector_residual_block_size', 4)
        ),
    )
    dynamics_count = _configured_mlp_parameter_count(
        latent_size + action_dim,
        dynamics.arch,
        latent_size,
        kind=str(dynamics.get('mlp_kind', 'plain')),
        residual_block_size=int(dynamics.get('residual_block_size', 4)),
    )
    actor_count = _configured_mlp_parameter_count(
        latent_size + plan.goal_latent_size,
        actor.arch,
        2 * action_dim,
        kind=str(actor.get('mlp_kind', 'plain')),
        residual_block_size=int(actor.get('residual_block_size', 4)),
    )
    return (
        plan.goal_encoder_parameters
        + plan.non_goal_encoder_parameters
        + projector_count + 1
        + dynamics_count
        + actor_count
    )


__all__ = [
    'BASE_MODEL_SIZE_LEVELS',
    'GOQRLBranchPlan',
    'MODEL_SIZE_LEVELS',
    'QRL_MODEL_SIZE_LEVELS',
    'MODEL_SIZE_ROOT',
    'SCALING_CRL_MODEL_SIZE_LEVELS',
    'base_agent_parameter_count',
    'go_qrl_agent_parameter_count',
    'load_model_size_preset',
    'match_go_qrl_split_encoder',
    'model_size_path',
    'qrl_agent_parameter_count',
    'register_model_size_presets',
    'scaling_crl_agent_parameter_count',
    'select_base_model_size',
    'select_qrl_model_size',
]

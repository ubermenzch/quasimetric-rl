from __future__ import annotations
from typing import *

import attrs
import itertools

import numpy as np
import torch
import torch.utils.data
import gym

from omegaconf import MISSING

from .utils import TensorCollectionAttrsMixin
from .env_spec import EnvSpec



#-----------------------------------------------------------------------------#
#-------------------------------- Batch data ---------------------------------#
#-----------------------------------------------------------------------------#

# What should be in a batch


@attrs.define(kw_only=True)
class BatchData(TensorCollectionAttrsMixin):  # TensorCollectionAttrsMixin has some util methods
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    rewards: torch.Tensor
    terminals: torch.Tensor
    timeouts: torch.Tensor

    future_observations: torch.Tensor  # sampled!

    # H input state frames followed by the final prediction target. The final
    # input frame is the sampled current state; earlier frames are its history.
    history_observations: Optional[torch.Tensor] = None  # [B, H + 1, *obs_shape]
    history_actions: Optional[torch.Tensor] = None  # [B, H, *action_shape]
    history_mask: Optional[torch.Tensor] = None  # [B, H], True for valid in-episode transitions

    @property
    def device(self) -> torch.device:
        return self.observations.device

    @property
    def batch_shape(self) -> torch.Size:
        return self.terminals.shape

    @property
    def num_transitions(self) -> int:
        return self.terminals.numel()



#-----------------------------------------------------------------------------#
#------------------------------- Episode data --------------------------------#
#-----------------------------------------------------------------------------#


@attrs.define(kw_only=True)
class MultiEpisodeData(TensorCollectionAttrsMixin):
    r"""
    The DATASET of MULTIPLE episodes
    """


    # For each episode, L: number of (s, a, s', r, d, to) pairs, so number of transitions (not observations)
    episode_lengths: torch.Tensor
    # cat all states from all episodes, where the last s' is added. I.e., each episode has L+1 states
    all_observations: torch.Tensor
    # cat all actions from all episodes. Each episode has L actions.
    actions: torch.Tensor
    # cat all rewards from all episodes. Each episode has L rewards.
    rewards: torch.Tensor
    # cat all terminals from all episodes. Each episode has L terminals.
    terminals: torch.Tensor
    # cat all timeouts from all episodes. Each episode has L timeouts.
    timeouts: torch.Tensor
    # cat all observation infos from all episodes. Each episode has L + 1 elements.
    observation_infos: Mapping[str, torch.Tensor] = attrs.Factory(dict)
    # cat all transition infos from all episodes. Each episode has L elements.
    transition_infos: Mapping[str, torch.Tensor] = attrs.Factory(dict)

    @property
    def num_episodes(self) -> int:
        return self.episode_lengths.shape[0]

    @property
    def num_transitions(self) -> int:
        return self.rewards.shape[0]

    def __attrs_post_init__(self):
        assert self.episode_lengths.ndim == 1
        N = self.num_transitions
        assert N > 0
        assert self.all_observations.ndim >= 1 and self.all_observations.shape[0] == (N + self.num_episodes), self.all_observations.shape
        assert self.actions.ndim >= 1 and self.actions.shape[0] == N
        assert self.rewards.ndim == 1 and self.rewards.shape[0] == N
        assert self.terminals.ndim == 1 and self.terminals.shape[0] == N
        assert self.timeouts.ndim == 1 and self.timeouts.shape[0] == N
        for k, v in self.observation_infos.items():
            assert v.shape[0] == N + self.num_episodes, k
        for k, v in self.transition_infos.items():
            assert v.shape[0] == N, k



@attrs.define(kw_only=True)
class EpisodeData(MultiEpisodeData):
    r"""
    A SINGLE episode
    """

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        assert self.num_episodes == 1

    @classmethod
    def from_simple_trajectory(cls,
                               observations: Union[np.ndarray, torch.Tensor],
                               actions: Union[np.ndarray, torch.Tensor],
                               next_observations: Union[np.ndarray, torch.Tensor],
                               rewards: Union[np.ndarray, torch.Tensor],
                               terminals: Union[np.ndarray, torch.Tensor],
                               timeouts: Union[np.ndarray, torch.Tensor]):
        observations = torch.tensor(observations)
        next_observations=torch.tensor(next_observations)
        all_observations = torch.cat([observations, next_observations[-1:]], dim=0)
        return cls(
            episode_lengths=torch.tensor([observations.shape[0]]),
            all_observations=all_observations,
            actions=torch.tensor(actions),
            rewards=torch.tensor(rewards),
            terminals=torch.tensor(terminals),
            timeouts=torch.tensor(timeouts),
        )


#-----------------------------------------------------------------------------#
#--------------------------------- dataset -----------------------------------#
#-----------------------------------------------------------------------------#


# Each env is specified with two strings:
#   + kind  # d4rl, gcrl, etc.
#   + spec  # maze2d-umaze-v1, FetchPushImage, etc.


LOAD_EPISODES_REGISTRY: Mapping[Tuple[str, str], Callable[[], Iterator[EpisodeData]]] = {}
CREATE_ENV_REGISTRY: Mapping[Tuple[str, str], Callable[[], gym.Env]] = {}

# Coordinates in the raw state observation that define goal attainment for the
# vector environments bundled with this repository.
GOAL_SET_DIMS_REGISTRY: Mapping[Tuple[str, str], Tuple[int, ...]] = {
    **{
        ('d4rl', name): (0, 1)
        for name in (
            'maze2d-umaze-v1',
            'maze2d-medium-v1',
            'maze2d-large-v1',
            'antmaze-umaze-v2',
            'antmaze-umaze-diverse-v2',
            'antmaze-medium-play-v2',
            'antmaze-medium-diverse-v2',
            'antmaze-large-play-v2',
            'antmaze-large-diverse-v2',
        )
    },
    ('gcrl', 'FetchReach'): (0, 1, 2),
    ('gcrl', 'FetchPush'): (3, 4, 5),
    ('gcrl', 'FetchSlide'): (3, 4, 5),
    ('gcrl', 'FetchPickAndPlace'): (3, 4, 5),
    ('gym_mujoco', 'Reacher-v4'): (0, 1),
    ('gym_mujoco', 'Pusher-v4'): (0, 1, 2),
    ('gym_mujoco', 'AntNavigate-v4'): (0, 1),
    ('dmc', 'reacher_easy'): (0, 1),
    ('dmc', 'reacher_hard'): (0, 1),
    ('dmc', 'swimmer6'): (0, 1),
    ('dmc', 'swimmer15'): (0, 1),
    ('dmc', 'quadruped_fetch'): (0, 1),
    ('dmc', 'manipulator_bring_ball'): (0, 1),
    ('dmc', 'manipulator_bring_peg'): (0, 1, 2, 3),
    ('online_maze', 'maze2d-medium'): (0, 1),
    ('online_maze', 'maze2d-large'): (0, 1),
}


def register_offline_env(kind: str, spec: str, *, load_episodes_fn, create_env_fn):
    r"""
    Each specific env (e.g., an offline env from d4rl) just needs to register

        1. how to load the episodes
        (this is optional in online settings. see ReplayBuffer)

        load_episodes_fn() -> Iterator[EpisodeData]

        2. how to create an env

        create_env_fn() -> gym.Env

     See d4rl/maze2d.py for example
    """
    assert (kind, spec) not in LOAD_EPISODES_REGISTRY
    LOAD_EPISODES_REGISTRY[(kind, spec)] = load_episodes_fn
    CREATE_ENV_REGISTRY[(kind, spec)] = create_env_fn


class Dataset:
    @attrs.define(kw_only=True)
    class Conf:
        # config / argparse uses this to specify behavior

        kind: str = MISSING  # d4rl, gcrl, etc.
        name: str = MISSING  # maze2d-umaze-v1, etc.

        # Defines how to fetch the future observation. smaller -> more recent
        future_observation_discount: float = attrs.field(default=0.99, validator=attrs.validators.and_(
            attrs.validators.ge(0.0),
            attrs.validators.le(1.0),
        ))
        transition_history_length: int = attrs.field(default=0, validator=attrs.validators.ge(0))

        def make(self, *, dummy: bool = False) -> 'Dataset':
            return Dataset(self.kind, self.name,
                           future_observation_discount=self.future_observation_discount,
                           transition_history_length=self.transition_history_length,
                           dummy=dummy)

    kind: str
    name: str
    future_observation_discount: float
    transition_history_length: int

    # Computed Attributes::

    # Data
    raw_data: MultiEpisodeData  # will contain all episodes

    # Env info
    env_spec: EnvSpec

    # Defines how to fetch the future observation. smaller -> more recent
    future_observation_discount: float

    # Auxiliary structures that helps fetching transitions of specific kinds
    # -----
    obs_indices_to_obs_index_in_episode: torch.Tensor
    indices_to_episode_indices: torch.Tensor  # episode indices refers to indices in this split
    indices_to_episode_timesteps: torch.Tensor
    max_episode_length: int
    # -----

    def create_env(self) -> gym.Env:
        return CREATE_ENV_REGISTRY[self.kind, self.name]()

    @property
    def goal_set_dims(self) -> Tuple[int, ...]:
        """State coordinates used to define goal sets for supported vector tasks."""
        try:
            return GOAL_SET_DIMS_REGISTRY[self.kind, self.name]
        except KeyError as exc:
            raise ValueError(
                f'No default goal-set dimensions for {(self.kind, self.name)!r}; '
                'set agent.goal_set_distance.losses.goal_dims explicitly.'
            ) from exc

    def load_episodes(self) -> Iterator[EpisodeData]:
        return LOAD_EPISODES_REGISTRY[self.kind, self.name]()

    def __init__(self, kind: str, name: str, *,
                 future_observation_discount: float,
                 transition_history_length: int = 0,
                 dummy: bool = False,  # when you don't want to load data, e.g., in analysis
                 ) -> None:
        self.kind = kind
        self.name = name
        self.future_observation_discount = future_observation_discount
        self.transition_history_length = transition_history_length

        self.env_spec = EnvSpec.from_env(self.create_env())

        assert 0 <= future_observation_discount
        self.future_observation_discount = future_observation_discount
        assert transition_history_length >= 0
        self.transition_history_length = transition_history_length

        if not dummy:
            episodes = tuple(self.load_episodes())
        else:
            from .online.utils import get_empty_episode
            episodes = (get_empty_episode(self.env_spec, episode_length=1),)

        obs_indices_to_obs_index_in_episode = []
        indices_to_episode_indices = []
        indices_to_episode_timesteps = []
        for eidx, episode in enumerate(episodes):
            l = episode.num_transitions
            obs_indices_to_obs_index_in_episode.append(torch.arange(l + 1, dtype=torch.int64))
            indices_to_episode_indices.append(torch.full([l], eidx, dtype=torch.int64))
            indices_to_episode_timesteps.append(torch.arange(l, dtype=torch.int64))

        assert len(episodes) > 0, "must have at least one episode"
        self.raw_data = MultiEpisodeData.cat(episodes)

        self.obs_indices_to_obs_index_in_episode = torch.cat(obs_indices_to_obs_index_in_episode, dim=0)
        self.indices_to_episode_indices = torch.cat(indices_to_episode_indices, dim=0)
        self.indices_to_episode_timesteps = torch.cat(indices_to_episode_timesteps, dim=0)
        self.max_episode_length = self.raw_data.episode_lengths.max().item()

    def get_observations(self, obs_indices: torch.Tensor):
        return self.raw_data.all_observations[obs_indices]

    @property
    def num_observations_available(self) -> int:
        """Number of valid observations that can be sampled globally."""
        return self.raw_data.all_observations.shape[0]

    def observation_bounds(self, *, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cached per-dimension bounds over all available observations."""
        if self.num_observations_available <= 0:
            raise RuntimeError('Cannot compute bounds for an empty dataset')
        cached = getattr(self, '_observation_bounds_cache', None)
        if cached is None:
            observations = self.raw_data.all_observations[:self.num_observations_available]
            cached = (observations.amin(dim=0), observations.amax(dim=0))
            self._observation_bounds_cache = cached
        low, high = cached
        if device is not None:
            low, high = low.to(device), high.to(device)
        return low, high

    def _goal_condition_coordinates(self, goal_dims: Tuple[int, ...]) -> torch.Tensor:
        """Return an incrementally maintained cache of valid goal coordinates."""
        observations = self.raw_data.all_observations
        available = self.num_observations_available
        caches = getattr(self, '_goal_condition_coordinate_caches', None)
        if caches is None:
            caches = self._goal_condition_coordinate_caches = {}
        cache_key = tuple(goal_dims)
        cache = caches.get(cache_key)

        needs_new_storage = (
            cache is None
            or cache['values'].shape[0] != observations.shape[0]
            or cache['values'].device != observations.device
            or cache['values'].dtype != observations.dtype
        )
        if needs_new_storage:
            values = torch.empty(
                (observations.shape[0], len(goal_dims)),
                device=observations.device,
                dtype=observations.dtype,
            )
            cached_count = 0
            if cache is not None and cache['count'] <= available:
                cached_count = min(cache['count'], cache['values'].shape[0], values.shape[0])
                values[:cached_count].copy_(cache['values'][:cached_count])
            cache = caches[cache_key] = dict(values=values, count=cached_count)
        elif cache['count'] > available:
            cache['count'] = 0

        if cache['count'] < available:
            goal_index = torch.as_tensor(goal_dims, device=observations.device)
            cache['values'][cache['count']:available].copy_(
                observations[cache['count']:available].index_select(-1, goal_index)
            )
            cache['count'] = available
        return cache['values'][:available]

    def _goal_condition_grid(
            self, goal_dims: Tuple[int, ...], cell_size: float) -> Dict[str, Any]:
        """Build or incrementally update a CPU grid over available observations."""
        coordinates = self._goal_condition_coordinates(goal_dims)
        if coordinates.device.type != 'cpu':
            raise RuntimeError('Goal-conditioned dataset grids require CPU dataset storage')
        caches = getattr(self, '_goal_condition_grid_caches', None)
        if caches is None:
            caches = self._goal_condition_grid_caches = {}
        cache_key = (tuple(goal_dims), float(cell_size))
        grid = caches.get(cache_key)
        if grid is None or grid['count'] > coordinates.shape[0]:
            grid = caches[cache_key] = dict(count=0, bins={})

        start = grid['count']
        if start < coordinates.shape[0]:
            cells = torch.floor(coordinates[start:] / cell_size).to(torch.int64).numpy()
            if cells.shape[0] > 0:
                sort_keys = tuple(cells[:, dim] for dim in reversed(range(cells.shape[1])))
                order = np.lexsort(sort_keys)
                sorted_cells = cells[order]
                boundaries = np.concatenate((
                    np.array([0], dtype=np.int64),
                    np.nonzero(np.any(sorted_cells[1:] != sorted_cells[:-1], axis=1))[0] + 1,
                    np.array([sorted_cells.shape[0]], dtype=np.int64),
                ))
                for left, right in zip(boundaries[:-1], boundaries[1:]):
                    cell = tuple(int(value) for value in sorted_cells[left])
                    new_indices = torch.from_numpy(order[left:right].copy()).to(torch.int64).add_(start)
                    existing = grid['bins'].get(cell)
                    grid['bins'][cell] = (
                        new_indices if existing is None else torch.cat([existing, new_indices])
                    )
            grid['count'] = coordinates.shape[0]
        return grid

    def sample_goal_conditioned_observations(
            self, raw_goal_states: torch.Tensor, *, goal_dims: Tuple[int, ...],
            num_samples: int, radius: float, seed: int,
            max_attempts: int = 256) -> Tuple[torch.Tensor, float]:
        """Sample ``num_samples`` exact-radius observations with replacement.

        The grid only creates a local proposal pool. Every accepted observation
        still passes the exact Euclidean-radius test. Slots that remain unresolved
        after ``max_attempts`` keep the raw goal as their fallback candidate.
        """
        if self.num_observations_available <= 0:
            raise RuntimeError('Cannot sample candidates from an empty dataset')
        if num_samples <= 0 or radius <= 0 or max_attempts <= 0:
            raise ValueError('num_samples, radius, and max_attempts must be positive')

        observations = self.raw_data.all_observations[:self.num_observations_available]
        if observations.ndim != 2 or raw_goal_states.shape[-1] != observations.shape[-1]:
            raise ValueError(
                f'Expected vector goals ending in {observations.shape[-1]} dimensions, '
                f'got shape={tuple(raw_goal_states.shape)}'
            )
        if not goal_dims or min(goal_dims) < 0 or max(goal_dims) >= observations.shape[-1]:
            raise ValueError(f'Invalid goal_dims={goal_dims!r}')

        if observations.device.type != 'cpu':
            raise RuntimeError('Goal-conditioned dataset sampling requires CPU dataset storage')
        storage_device = observations.device
        flat_goals = raw_goal_states.detach().reshape(-1, observations.shape[-1]).to(
            device=storage_device,
            dtype=observations.dtype,
        )
        num_pairs = flat_goals.shape[0]
        sampled = flat_goals[:, None, :].expand(
            num_pairs, num_samples, observations.shape[-1]
        ).clone()
        sampled_flat = sampled.reshape(-1, observations.shape[-1])
        num_raw_goal_fallbacks = 0

        goal_index = torch.as_tensor(goal_dims, device=storage_device)
        observation_goals = self._goal_condition_coordinates(goal_dims)
        query_goals = flat_goals.index_select(-1, goal_index)
        grid = self._goal_condition_grid(goal_dims, radius)
        generator = torch.Generator(device=storage_device)
        generator.manual_seed(int(seed))

        attempts_per_round = 16
        radius_squared = radius * radius
        goal_cells = torch.floor(query_goals / radius).to(torch.int64).numpy()
        unique_cells, inverse = np.unique(goal_cells, axis=0, return_inverse=True)
        neighbor_offsets = tuple(itertools.product((-1, 0, 1), repeat=len(goal_dims)))

        for cell_number, cell_values in enumerate(unique_cells):
            pair_indices_np = np.nonzero(inverse == cell_number)[0]
            pair_indices = torch.from_numpy(pair_indices_np.copy()).to(torch.int64)
            cell = tuple(int(value) for value in cell_values)
            neighboring_bins = []
            for offset in neighbor_offsets:
                neighbor = tuple(value + delta for value, delta in zip(cell, offset))
                indices = grid['bins'].get(neighbor)
                if indices is not None:
                    neighboring_bins.append(indices)
            pool = (
                torch.cat(neighboring_bins)
                if neighboring_bins
                else torch.empty(0, dtype=torch.int64)
            )

            if pool.numel() == 0:
                num_raw_goal_fallbacks += pair_indices.numel() * num_samples
                continue

            group_size = pair_indices.numel()
            local_pair_indices = torch.arange(group_size).repeat_interleave(num_samples)
            destinations = pair_indices.repeat_interleave(num_samples) * num_samples
            destinations += torch.arange(num_samples).repeat(group_size)
            unresolved = torch.arange(group_size * num_samples)

            attempts_used = 0
            while unresolved.numel() > 0 and attempts_used < max_attempts:
                attempts = min(attempts_per_round, max_attempts - attempts_used)
                pool_positions = torch.randint(
                    pool.numel(),
                    (unresolved.numel(), attempts),
                    generator=generator,
                )
                random_indices = pool[pool_positions]
                unresolved_pairs = local_pair_indices[unresolved]
                candidate_goals = observation_goals[random_indices]
                goals = query_goals[pair_indices[unresolved_pairs]][:, None, :]
                valid = (
                    candidate_goals - goals
                ).square().sum(dim=-1) <= radius_squared
                has_match = valid.any(dim=-1)
                matched_rows = torch.nonzero(has_match, as_tuple=False).squeeze(-1)
                if matched_rows.numel() > 0:
                    first_match = valid[matched_rows].to(torch.int64).argmax(dim=-1)
                    source_indices = random_indices[matched_rows, first_match]
                    target_indices = destinations[unresolved[matched_rows]]
                    sampled_flat[target_indices] = observations[source_indices]
                unresolved = unresolved[~has_match]
                attempts_used += attempts

            if unresolved.numel() > 0:
                num_raw_goal_fallbacks += unresolved.numel()

        candidates = sampled.reshape(
            *raw_goal_states.shape[:-1], num_samples, observations.shape[-1]
        ).to(device=raw_goal_states.device, dtype=raw_goal_states.dtype)
        fallback_fraction = num_raw_goal_fallbacks / (num_pairs * num_samples)
        return candidates, fallback_fraction

    def get_transition_history(self, indices: torch.Tensor, history_length: int):
        if history_length <= 0:
            return None, None, None
        eindices = self.indices_to_episode_indices[indices]
        tindices = self.indices_to_episode_timesteps[indices]
        obs_indices = indices + eindices
        # The sampled transition is the final (current) input frame. Earlier
        # frames provide history, so h=1 has no historical frame and h=2 has one.
        offsets = torch.arange(history_length, device=indices.device) - (history_length - 1)
        transition_indices = indices[:, None] + offsets
        valid = (tindices[:, None] + offsets) >= 0
        safe_transition_indices = torch.where(valid, transition_indices, indices[:, None])
        safe_obs_indices = torch.where(valid, obs_indices[:, None] + offsets, obs_indices[:, None])
        history_observations = torch.cat([
            self.get_observations(safe_obs_indices),
            self.get_observations((obs_indices + 1)[:, None]),
        ], dim=1)
        history_actions = self.raw_data.actions[safe_transition_indices]
        return history_observations, history_actions, valid

    def __getitem__(self, indices: torch.Tensor) -> BatchData:
        indices = torch.as_tensor(indices)
        eindices = self.indices_to_episode_indices[indices]
        obs_indices = indices + eindices  # index for `observation`: skip the s_last from previous episodes
        obs = self.get_observations(obs_indices)
        nobs = self.get_observations(obs_indices + 1)

        terminals = self.raw_data.terminals[indices]

        tindices = self.indices_to_episode_timesteps[indices]
        epilengths = self.raw_data.episode_lengths[eindices]  # max idx is this
        deltas = torch.arange(self.max_episode_length)
        pdeltas = torch.where(
            # test tidx + 1 + delta <= max_idx = epi_length
            (tindices[:, None] + deltas) < epilengths[:, None],
            self.future_observation_discount ** deltas,
            0,
        )
        deltas = torch.distributions.Categorical(
            probs=pdeltas,
        ).sample()
        future_observations = self.get_observations(obs_indices + 1 + deltas)
        history_observations = history_actions = history_mask = None
        if self.transition_history_length > 0:
            history_observations, history_actions, history_mask = self.get_transition_history(
                indices,
                self.transition_history_length,
            )

        return BatchData(
            observations=obs,
            actions=self.raw_data.actions[indices],
            next_observations=nobs,
            future_observations=future_observations,
            history_observations=history_observations,
            history_actions=history_actions,
            history_mask=history_mask,
            rewards=self.raw_data.rewards[indices],
            terminals=terminals,
            timeouts=self.raw_data.timeouts[indices],
        )

    def __len__(self):
        return self.raw_data.num_transitions

    def __repr__(self):
        return rf"""
{self.__class__.__name__}(
    kind={self.kind!r},
    name={self.name!r},
    future_observation_discount={self.future_observation_discount!r},
    transition_history_length={self.transition_history_length!r},
    env_spec={self.env_spec!r},
)""".lstrip('\n')

    def get_dataloader(self, *,
                       batch_size: int, shuffle: bool = False,
                       drop_last: bool = False,
                       pin_memory: bool = False,
                       num_workers: int = 0, persistent_workers: bool = False,
                       **kwargs) -> torch.utils.data.DataLoader:
        sampler = torch.utils.data.BatchSampler(
            torch.utils.data.RandomSampler(self),
            batch_size=batch_size,
            drop_last=drop_last,
        )
        return torch.utils.data.DataLoader(
            self,
            batch_size=None,
            sampler=sampler,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            **kwargs,
        )


def seed_worker(_):
    worker_seed = torch.utils.data.get_worker_info().seed % (2 ** 32)
    np.random.seed(worker_seed)


from . import d4rl  # register

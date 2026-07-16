from __future__ import annotations
from typing import *

import time
import logging
import contextlib

import torch
import torch.utils.data

from quasimetric_rl.modules import QRLConf, QRLAgent, QRLLosses, InfoT
from quasimetric_rl.data import BatchData, Dataset
from quasimetric_rl.utils import TimingProfiler



class ResumableRandomBatchIterator:
    def __init__(self, dataset: Dataset, *, batch_size: int, drop_last: bool,
                 seed: int = 0, epoch: int = 0, next_batch_idx: int = 0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = epoch
        self.next_batch_idx = next_batch_idx
        self._generator = torch.Generator()
        self._permutation = torch.empty(0, dtype=torch.int64)
        self._refresh_permutation()

    @property
    def num_batches(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def _refresh_permutation(self) -> None:
        self._generator.manual_seed(self.seed + self.epoch)
        self._permutation = torch.randperm(len(self.dataset), generator=self._generator)

    def state_dict(self) -> dict[str, Any]:
        return dict(
            seed=self.seed,
            epoch=self.epoch,
            next_batch_idx=self.next_batch_idx,
        )

    def load_state_dict(self, state: Mapping[str, Any] | None) -> None:
        if not state:
            return
        self.seed = int(state.get('seed', self.seed))
        self.epoch = int(state.get('epoch', self.epoch))
        self.next_batch_idx = int(state.get('next_batch_idx', self.next_batch_idx))
        self._refresh_permutation()

    def __iter__(self) -> Iterator[Tuple[int, int, BatchData]]:
        while self.next_batch_idx < self.num_batches:
            batch_idx = self.next_batch_idx
            start = batch_idx * self.batch_size
            stop = start + self.batch_size
            indices = self._permutation[start:stop]
            if self.drop_last and indices.numel() < self.batch_size:
                break
            data = self.dataset[indices]
            yield self.epoch, batch_idx, data

    def advance_batch(self) -> None:
        self.next_batch_idx += 1

    def advance_epoch(self) -> None:
        self.epoch += 1
        self.next_batch_idx = 0
        self._refresh_permutation()


class Trainer(object):
    agent: QRLAgent
    losses: QRLLosses
    device: torch.device
    dataset: Dataset
    batch_size: int
    batch_iterator: ResumableRandomBatchIterator
    profiler: Optional[TimingProfiler]

    def __init__(self, *,
                 agent_conf: QRLConf,
                 device: torch.device,
                 dataset: Dataset,
                 batch_size: int,
                 total_optim_steps: int,
                 dataloader_kwargs: Dict[str, Any] = {},
                 profiler: Optional[TimingProfiler] = None,
                 data_seed: int = 0):

        self.device = device
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.profiler = profiler
        self.dataset.transition_history_length = max(
            self.dataset.transition_history_length,
            agent_conf.required_transition_history_length,
        )

        self.agent, self.losses = agent_conf.make(
            env_spec=dataset.env_spec,
            total_optim_steps=total_optim_steps,
            profiler=profiler,
            goal_set_dims=(
                dataset.goal_set_dims
                if agent_conf.goal_set_distance.enabled
                and agent_conf.goal_set_distance.losses.goal_dims is None
                else None
            ),
        )
        self.agent.to(device)
        self.losses.to(device)
        if self.losses.goal_set_distance_loss is not None:
            self.losses.goal_set_distance_loss.set_observation_bounds_provider(dataset.observation_bounds)
            self.losses.goal_set_distance_loss.set_candidate_state_provider(
                dataset.sample_goal_conditioned_observations
            )
            self.losses.goal_set_distance_loss.set_candidate_seed(data_seed)

        logging.info('Agent:\n\t' + str(self.agent).replace('\n', '\n\t') + '\n\n')
        logging.info('Losses:\n\t' + str(self.losses).replace('\n', '\n\t') + '\n\n')

        self.batch_iterator = ResumableRandomBatchIterator(
            dataset,
            batch_size=batch_size,
            drop_last=bool(dataloader_kwargs.get('drop_last', False)),
            seed=data_seed,
        )

    def _record(self, name: str):
        if self.profiler is None:
            return contextlib.nullcontext()
        return self.profiler.record(name)

    @property
    def num_batches(self):
        return self.batch_iterator.num_batches

    def iter_training_data(self) -> Iterator[Tuple[int, int, BatchData, InfoT]]:
        r"""
        Yield data to train on for each optimization iteration.

        yield (
            epoch,
            batch index,
            data,
            info,
        )
        """
        data_t0 = time.time()
        data: BatchData
        for epoch, it, data in self.batch_iterator:
            with self._record('data/to_device'):
                data = data.to(self.device)
            yield epoch, it, data, dict(data_time=time.time() - data_t0)
            data_t0 = time.time()

    def advance_epoch(self) -> None:
        self.batch_iterator.advance_epoch()

    def advance_batch(self) -> None:
        self.batch_iterator.advance_batch()

    def data_state_dict(self) -> dict[str, Any]:
        return self.batch_iterator.state_dict()

    def load_data_state_dict(self, state: Mapping[str, Any] | None) -> None:
        self.batch_iterator.load_state_dict(state)

    def train_step(self, data: BatchData, *, optimize: bool = True, phase: str = 'all') -> InfoT:
        with self._record('train/losses_total'):
            return self.losses(self.agent, data, optimize=optimize, phase=phase).info

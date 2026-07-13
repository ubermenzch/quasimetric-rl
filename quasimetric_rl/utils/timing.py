from __future__ import annotations
from typing import *

import contextlib
import json
import logging
import time
from collections import defaultdict

import attrs
import torch


@attrs.define(kw_only=True)
class TimingConf:
    enabled: bool = False
    cuda_sync: bool = False
    log_jsonl: bool = True
    log_top_n: int = attrs.field(default=20, validator=attrs.validators.ge(0))
    reset_after_log: bool = True

    def make(self, *, output_dir: str, device: Optional[torch.device] = None) -> 'TimingProfiler':
        return TimingProfiler(
            enabled=self.enabled,
            cuda_sync=self.cuda_sync,
            jsonl_path=(f"{output_dir}/timing.jsonl" if self.log_jsonl else None),
            log_top_n=self.log_top_n,
            reset_after_log=self.reset_after_log,
            device=device,
        )


class TimingProfiler:
    def __init__(self, *, enabled: bool, cuda_sync: bool, jsonl_path: Optional[str],
                 log_top_n: int, reset_after_log: bool, device: Optional[torch.device] = None):
        self.enabled = enabled
        self.cuda_sync = cuda_sync
        self.jsonl_path = jsonl_path
        self.log_top_n = log_top_n
        self.reset_after_log = reset_after_log
        self.device = device
        self._records: MutableMapping[str, Dict[str, float]] = defaultdict(
            lambda: dict(total_s=0.0, count=0.0, max_s=0.0)
        )

    def _sync(self) -> None:
        if not self.enabled or not self.cuda_sync or not torch.cuda.is_available():
            return
        if self.device is not None and self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        else:
            torch.cuda.synchronize()

    @contextlib.contextmanager
    def record(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        self._sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            elapsed = time.perf_counter() - start
            rec = self._records[name]
            rec['total_s'] += elapsed
            rec['count'] += 1
            rec['max_s'] = max(rec['max_s'], elapsed)

    def snapshot(self, *, reset: bool = False) -> Dict[str, Dict[str, float]]:
        out = {
            name: dict(
                total_s=rec['total_s'],
                mean_s=(rec['total_s'] / rec['count'] if rec['count'] else 0.0),
                max_s=rec['max_s'],
                count=int(rec['count']),
            )
            for name, rec in sorted(self._records.items())
        }
        if reset:
            self._records.clear()
        return out

    def log(self, *, writer, step: int, step_name: str, extra: Optional[Mapping[str, Any]] = None) -> None:
        if not self.enabled:
            return
        records = self.snapshot(reset=self.reset_after_log)
        if not records:
            return

        for name, rec in records.items():
            tag = f"timing/{name}"
            writer.add_scalar(f"{tag}/total_s", rec['total_s'], step)
            writer.add_scalar(f"{tag}/mean_s", rec['mean_s'], step)
            writer.add_scalar(f"{tag}/max_s", rec['max_s'], step)
            writer.add_scalar(f"{tag}/count", rec['count'], step)

        if self.jsonl_path is not None:
            row = dict(
                step=step,
                step_name=step_name,
                wall_time=time.time(),
                records=records,
            )
            if extra:
                row.update(extra)
            with open(self.jsonl_path, 'a') as f:
                print(json.dumps(row, sort_keys=True), file=f)

        if self.log_top_n > 0:
            top = sorted(records.items(), key=lambda item: item[1]['total_s'], reverse=True)[:self.log_top_n]
            summary = "  ".join(
                f"{name}: total={rec['total_s']:.3f}s mean={rec['mean_s']:.4f}s n={rec['count']}"
                for name, rec in top
            )
            logging.info(f"TIMING {step_name}={step}: {summary}")

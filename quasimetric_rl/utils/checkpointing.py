from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def atomic_torch_save(obj: Any, path: str | os.PathLike[str]) -> None:
    path = Path(path)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def prune_checkpoints(output_dir: str | os.PathLike[str], keep_path: str | os.PathLike[str]) -> None:
    keep_path = Path(keep_path).resolve()
    for ckpt in Path(output_dir).glob("checkpoint_*.pth"):
        if ckpt.resolve() == keep_path:
            continue
        try:
            ckpt.unlink()
            logging.info(f"Removed old checkpoint {ckpt}")
        except FileNotFoundError:
            pass


def rng_state_dict() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        try:
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        except Exception as exc:
            logging.warning(f"Could not capture CUDA RNG state: {exc}")
    return state


def load_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "torch_cuda" in state and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(state["torch_cuda"])
        except Exception as exc:
            logging.warning(f"Could not restore CUDA RNG state: {exc}")

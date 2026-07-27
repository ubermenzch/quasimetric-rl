from __future__ import annotations

import logging
import os
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch


RESUME_CHECKPOINT_FILENAME = "checkpoint_resume_latest.pth"
_AGENT_CHECKPOINT_RE = re.compile(r"agent_checkpoint_step(\d+)\.pth$")
_FULL_CHECKPOINT_RE = re.compile(
    r"checkpoint_(?:(\d+)_(\d+)|env(\d+)_opt(\d+))(?:_final)?\.pth$"
)


def agent_checkpoint_filename(optim_steps: int) -> str:
    if optim_steps < 0:
        raise ValueError(f"optim_steps must be non-negative, got {optim_steps}")
    return f"agent_checkpoint_step{optim_steps:08d}.pth"


def agent_checkpoint_step(path: str | os.PathLike[str]) -> int | None:
    match = _AGENT_CHECKPOINT_RE.fullmatch(Path(path).name)
    return int(match.group(1)) if match is not None else None


def full_checkpoint_key(path: str | os.PathLike[str]) -> tuple[int, int, int] | None:
    match = _FULL_CHECKPOINT_RE.fullmatch(Path(path).name)
    if match is None:
        return None
    first = match.group(1) or match.group(3)
    second = match.group(2) or match.group(4)
    return (
        int(first),
        int(second),
        int(Path(path).name.endswith("_final.pth")),
    )


def atomic_torch_save(obj: Any, path: str | os.PathLike[str]) -> None:
    path = Path(path)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def prune_checkpoints(
        output_dir: str | os.PathLike[str],
        keep_path: str | os.PathLike[str],
        *,
        pattern: str = "checkpoint_*.pth",
        preserve_final: bool = True,
) -> None:
    """Delete matching checkpoints except the retained path and optional finals."""
    keep_path = Path(keep_path).resolve()
    for ckpt in Path(output_dir).glob(pattern):
        if ckpt.resolve() == keep_path or (
                preserve_final and ckpt.name.endswith("_final.pth")):
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


def validate_training_cursor(checkpoint: dict[str, Any], num_batches: int) -> None:
    """Reject checkpoints captured in the middle of an optimization step."""
    data_state = checkpoint.get("data")
    loop_state = checkpoint.get("loop_state")
    if not isinstance(data_state, dict) or not isinstance(loop_state, dict):
        return
    if "optim_steps" not in loop_state:
        return

    epoch = int(data_state.get("epoch", checkpoint.get("epoch", 0)))
    next_batch_idx = int(data_state.get("next_batch_idx", 0))
    optim_steps = int(loop_state["optim_steps"])
    expected_optim_steps = epoch * num_batches + next_batch_idx
    if optim_steps != expected_optim_steps:
        raise RuntimeError(
            "Checkpoint is not at a complete training-step boundary: "
            f"optim_steps={optim_steps}, but epoch={epoch}, "
            f"next_batch_idx={next_batch_idx}, and num_batches={num_batches} "
            f"imply {expected_optim_steps}. Restart this run from a clean output directory."
        )

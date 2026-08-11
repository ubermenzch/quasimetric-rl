#!/usr/bin/env python3
"""Shared helpers for deleting completed-task checkpoint files."""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


CHECKPOINTS_DELETED_MARKER = "CHECKPOINTS_DELETED"


@dataclass(frozen=True)
class CheckpointCleanupResult:
    checkpoint_count: int
    checkpoint_bytes: int
    already_clean: bool = False


def task_checkpoint_files(output_dir: Path) -> tuple[Path, ...]:
    if not output_dir.is_dir():
        return ()
    return tuple(sorted(
        path
        for path in output_dir.glob("*.pth")
        if path.is_file() or path.is_symlink()
    ))


def checkpoint_file_bytes(paths: tuple[Path, ...]) -> int:
    total = 0
    for path in paths:
        try:
            total += path.lstat().st_size
        except OSError:
            pass
    return total


def cleanup_task_checkpoints(
    output_dir: Path,
    task_id: str,
) -> CheckpointCleanupResult:
    """Delete top-level checkpoint files and atomically record the cleanup."""
    if not output_dir.is_dir():
        raise FileNotFoundError(f"task result directory does not exist: {output_dir}")

    checkpoint_files = task_checkpoint_files(output_dir)
    marker = output_dir / CHECKPOINTS_DELETED_MARKER
    if not checkpoint_files and marker.is_file():
        return CheckpointCleanupResult(0, 0, already_clean=True)

    checkpoint_bytes = checkpoint_file_bytes(checkpoint_files)
    for checkpoint in checkpoint_files:
        checkpoint.unlink()

    payload = {
        "version": 1,
        "task_id": task_id,
        "deleted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint_count": len(checkpoint_files),
        "checkpoint_bytes": checkpoint_bytes,
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{marker.name}.", suffix=".tmp", dir=marker.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, marker)
    finally:
        if temporary.exists():
            temporary.unlink()

    return CheckpointCleanupResult(
        checkpoint_count=len(checkpoint_files),
        checkpoint_bytes=checkpoint_bytes,
    )

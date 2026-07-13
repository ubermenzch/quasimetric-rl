#!/usr/bin/env python3
"""Verify the local environment needed by this repository's D4RL experiments."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "d4rl_v2_datasets.tsv"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-datasets", action="store_true")
    parser.add_argument("--checksums", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Construct dummy Maze2D and AntMaze datasets after dependency checks.",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail unless PyTorch can execute a CUDA tensor operation.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest() -> list[tuple[str, int, str]]:
    entries: list[tuple[str, int, str]] = []
    for line in MANIFEST.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        name, size, digest = line.split("\t")
        entries.append((name, int(size), digest))
    return entries


def main() -> None:
    args = parse_args()
    failures: list[str] = []

    if sys.version_info[:2] != (3, 9):
        failures.append(f"expected Python 3.9, got {sys.version.split()[0]}")
    print(f"Python: {sys.version.split()[0]}")

    dataset_dir_raw = os.environ.get("D4RL_DATASET_DIR", "")
    mujoco_dir_raw = os.environ.get("MUJOCO_PY_MUJOCO_PATH", "")
    dataset_dir = Path(dataset_dir_raw) if dataset_dir_raw else None
    mujoco_dir = Path(mujoco_dir_raw) if mujoco_dir_raw else None
    print(f"D4RL_DATASET_DIR: {dataset_dir_raw or '<unset>'}")
    print(f"MUJOCO_PY_MUJOCO_PATH: {mujoco_dir_raw or '<unset>'}")

    if mujoco_dir is None or not mujoco_dir.is_dir() or not (mujoco_dir / "bin" / "libmujoco210.so").is_file():
        failures.append("MuJoCo 2.1.0 is missing; expected bin/libmujoco210.so")
    else:
        os.environ.setdefault("MUJOCO_PATH", str(mujoco_dir))
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        library_dir = str(mujoco_dir / "bin")
        ld_paths = [path for path in os.environ.get("LD_LIBRARY_PATH", "").split(":") if path]
        if library_dir not in ld_paths:
            os.environ["LD_LIBRARY_PATH"] = ":".join([library_dir, *ld_paths])

    if not args.skip_datasets:
        if dataset_dir is None or not dataset_dir.is_dir():
            failures.append("D4RL dataset directory is missing")
        else:
            for filename, expected_size, expected_hash in read_manifest():
                path = dataset_dir / filename
                if not path.is_file():
                    failures.append(f"missing dataset: {filename}")
                    continue
                if path.stat().st_size != expected_size:
                    failures.append(f"size mismatch: {filename}")
                if args.checksums and sha256(path) != expected_hash:
                    failures.append(f"checksum mismatch: {filename}")

    try:
        import torch
        import gym
        import hydra
        import omegaconf
        import mujoco
        import d4rl
        import torchqmet

        print(f"torch: {torch.__version__}, CUDA wheel: {torch.version.cuda}")
        print(f"gym: {gym.__version__}, hydra: {hydra.__version__}, omegaconf: {omegaconf.__version__}")
        print(f"mujoco: {mujoco.__version__}, d4rl: {getattr(d4rl, '__version__', '1.1')}")
        print(f"torchqmet: {getattr(torchqmet, '__version__', 'installed')}")

        if not torch.cuda.is_available():
            print("CUDA: unavailable")
            if args.require_cuda:
                failures.append("PyTorch cannot access CUDA")
        else:
            name = torch.cuda.get_device_name(0)
            capability = torch.cuda.get_device_capability(0)
            print(f"GPU 0: {name}, compute capability: {capability[0]}.{capability[1]}")
            torch.zeros(1, device="cuda").add_(1).cpu()

        if args.smoke:
            from quasimetric_rl.data import Dataset

            for env_name in ("maze2d-umaze-v1", "antmaze-umaze-diverse-v2"):
                dataset = Dataset.Conf(kind="d4rl", name=env_name).make(dummy=True)
                print(
                    f"smoke dataset: {env_name}, "
                    f"observation={tuple(dataset.env_spec.observation_shape)}, "
                    f"action={tuple(dataset.env_spec.action_shape)}"
                )
    except Exception as exc:
        failures.append(f"Python dependency or smoke check failed: {type(exc).__name__}: {exc}")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        raise SystemExit(1)
    print("Environment verification passed.")


if __name__ == "__main__":
    main()

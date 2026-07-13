"""Shared runtime setup for standalone scripts that use D4RL and MuJoCo."""

from __future__ import annotations

import os
from pathlib import Path


def default_asset_root(repo_root: Path) -> Path:
    return Path(os.environ.get("QRL_ASSET_ROOT", repo_root.parent / "qrl-assets"))


def default_nvidia_library_dir() -> Path | None:
    candidates = [Path("/usr/local/nvidia/lib64"), Path("/usr/lib/nvidia")]
    candidates.extend(sorted(Path("/usr/lib").glob("nvidia-[0-9][0-9][0-9]")))
    return next((path for path in candidates if path.is_dir()), None)


def prepend_env_path(name: str, path: Path | str, *, require_exists: bool = False) -> None:
    value = str(path)
    if require_exists and not os.path.exists(value):
        return
    parts = [part for part in os.environ.get(name, "").split(os.pathsep) if part]
    if value not in parts:
        os.environ[name] = os.pathsep.join([value, *parts])


def configure_d4rl_runtime(repo_root: Path, *, require_library_paths: bool) -> Path:
    """Set the legacy D4RL/MuJoCo defaults and return the asset root.

    Existing user-provided variables always win. `require_library_paths` keeps
    compatibility with scripts that historically ignored nonexistent library
    directories.
    """
    asset_root = default_asset_root(repo_root)
    mujoco_path = asset_root / "mujoco/mujoco210"

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")
    os.environ.setdefault("D4RL_DATASET_DIR", str(asset_root / "d4rl/datasets"))
    os.environ.setdefault("MUJOCO_PY_MUJOCO_PATH", str(mujoco_path))
    os.environ.setdefault("MUJOCO_PATH", str(mujoco_path))

    prepend_env_path(
        "LD_LIBRARY_PATH",
        mujoco_path / "bin",
        require_exists=require_library_paths,
    )
    driver_library_dir = os.environ.get("QRL_DRIVER_LIBRARY_DIR", "")
    if not driver_library_dir:
        detected_driver_dir = default_nvidia_library_dir()
        driver_library_dir = str(detected_driver_dir) if detected_driver_dir is not None else ""
    if driver_library_dir:
        prepend_env_path(
            "LD_LIBRARY_PATH",
            driver_library_dir,
            require_exists=require_library_paths,
        )
    return asset_root

"""Shared runtime setup for standalone scripts that use D4RL and MuJoCo."""

from __future__ import annotations

import ctypes.util
import os
import sys
from pathlib import Path


RUNTIME_REEXEC_MARKER = "_QRL_D4RL_RUNTIME_REEXEC"
EGL_VENDOR_ENV = "__EGL_VENDOR_LIBRARY_FILENAMES"


def default_asset_root(repo_root: Path) -> Path:
    return Path(os.environ.get("QRL_ASSET_ROOT", repo_root.parent / "qrl-assets"))


def default_nvidia_library_dir() -> Path | None:
    candidates = [Path("/usr/local/nvidia/lib64"), Path("/usr/lib/nvidia")]
    candidates.extend(sorted(Path("/usr/lib").glob("nvidia-[0-9][0-9][0-9]")))
    return next((path for path in candidates if path.is_dir()), None)


def default_user_graphics_prefix(asset_root: Path) -> Path:
    return Path(
        os.environ.get(
            "QRL_USER_GRAPHICS_PREFIX", asset_root / "micromamba/envs/graphics"
        )
    )


def find_nvidia_egl_vendor_file(repo_root: Path) -> Path | None:
    vendor_dirs = (
        Path("/usr/share/glvnd/egl_vendor.d"),
        Path("/etc/glvnd/egl_vendor.d"),
        Path("/usr/local/share/glvnd/egl_vendor.d"),
    )
    for vendor_dir in vendor_dirs:
        vendor_files = sorted(vendor_dir.glob("*nvidia*.json"))
        if vendor_files:
            return vendor_files[0]

    if ctypes.util.find_library("EGL_nvidia") is None:
        return None
    bundled_vendor_file = repo_root / "configs/nvidia_egl_vendor.json"
    return bundled_vendor_file if bundled_vendor_file.is_file() else None


def configure_nvidia_egl_vendor(repo_root: Path) -> Path | None:
    existing = os.environ.get(EGL_VENDOR_ENV)
    if existing:
        return Path(existing.split(os.pathsep)[0])
    vendor_file = find_nvidia_egl_vendor_file(repo_root)
    if vendor_file is not None:
        os.environ[EGL_VENDOR_ENV] = str(vendor_file)
    return vendor_file


def prepend_env_path(name: str, path: Path | str, *, require_exists: bool = False) -> None:
    value = str(path)
    if require_exists and not os.path.exists(value):
        return
    parts = [part for part in os.environ.get(name, "").split(os.pathsep) if part]
    if value not in parts:
        os.environ[name] = os.pathsep.join([value, *parts])


def prepend_compiler_flag(name: str, flag: str) -> None:
    flags = os.environ.get(name, "").split()
    if flag not in flags:
        os.environ[name] = " ".join((flag, *flags))


def configure_d4rl_runtime(
    repo_root: Path,
    *,
    require_library_paths: bool,
    reexec_if_library_path_changed: bool = False,
    prefer_nvidia_egl_vendor: bool = False,
) -> Path:
    """Set the legacy D4RL/MuJoCo defaults and return the asset root.

    Existing user-provided variables always win. `require_library_paths` keeps
    compatibility with scripts that historically ignored nonexistent library
    directories. A process that needs to dynamically load MuJoCo should enable
    `reexec_if_library_path_changed`: glibc reads LD_LIBRARY_PATH at process
    startup, so mutating it in an already-running Python process is not enough.
    """
    original_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    asset_root = default_asset_root(repo_root)
    mujoco_path = asset_root / "mujoco/mujoco210"

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")
    os.environ.setdefault("D4RL_DATASET_DIR", str(asset_root / "d4rl/datasets"))
    os.environ.setdefault("MUJOCO_PY_MUJOCO_PATH", str(mujoco_path))
    os.environ.setdefault("MUJOCO_PATH", str(mujoco_path))
    if prefer_nvidia_egl_vendor:
        configure_nvidia_egl_vendor(repo_root)

    prepend_env_path(
        "LD_LIBRARY_PATH",
        mujoco_path / "bin",
        require_exists=require_library_paths,
    )
    graphics_prefix = default_user_graphics_prefix(asset_root)
    os.environ.setdefault("QRL_USER_GRAPHICS_PREFIX", str(graphics_prefix))
    graphics_include = graphics_prefix / "include"
    prepend_env_path("CPATH", graphics_include, require_exists=True)
    graphics_patchelf = graphics_prefix / "bin/patchelf"
    if graphics_patchelf.is_file():
        prepend_env_path("PATH", graphics_patchelf.parent, require_exists=True)
    prepend_compiler_flag("CFLAGS", "-DGLEW_NO_GLU")
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

    library_path_changed = (
        os.environ.get("LD_LIBRARY_PATH", "") != original_library_path
    )
    if (
        reexec_if_library_path_changed
        and library_path_changed
        and os.environ.get(RUNTIME_REEXEC_MARKER) != "1"
    ):
        env = os.environ.copy()
        env[RUNTIME_REEXEC_MARKER] = "1"
        if getattr(sys.stdout, "write_through", False):
            env.setdefault("PYTHONUNBUFFERED", "1")
        os.execve(sys.executable, [sys.executable, *sys.argv], env)
    return asset_root

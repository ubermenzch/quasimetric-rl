#!/usr/bin/env bash

# Source this file from the repository root or any shell before using D4RL.
# Override QRL_ASSET_ROOT to keep data and MuJoCo outside the Git checkout.

QRL_ENV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export QRL_ASSET_ROOT="$(realpath -m "${QRL_ASSET_ROOT:-${QRL_ENV_ROOT}/../qrl-assets}")"
export D4RL_DATASET_DIR="$(realpath -m "${D4RL_DATASET_DIR:-${QRL_ASSET_ROOT}/d4rl/datasets}")"
export MUJOCO_PY_MUJOCO_PATH="$(realpath -m "${MUJOCO_PY_MUJOCO_PATH:-${QRL_ASSET_ROOT}/mujoco/mujoco210}")"
export MUJOCO_PATH="$(realpath -m "${MUJOCO_PATH:-${MUJOCO_PY_MUJOCO_PATH}}")"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export D4RL_SUPPRESS_IMPORT_ERROR="${D4RL_SUPPRESS_IMPORT_ERROR:-1}"
export LD_LIBRARY_PATH="${MUJOCO_PY_MUJOCO_PATH}/bin${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# A standard NVIDIA driver installation exposes its libraries through the
# system linker. Set this only on hosts that need an explicit driver location.
if [[ -n "${QRL_DRIVER_LIBRARY_DIR:-}" ]]; then
    export LD_LIBRARY_PATH="${QRL_DRIVER_LIBRARY_DIR}:${LD_LIBRARY_PATH}"
fi

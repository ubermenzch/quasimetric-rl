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

# Match mujoco-py's legacy NVIDIA library discovery so its build-time path
# validation succeeds on bare-metal hosts as well as container hosts.
QRL_DETECTED_DRIVER_LIBRARY_DIR="${QRL_DRIVER_LIBRARY_DIR:-}"
if [[ -z "${QRL_DETECTED_DRIVER_LIBRARY_DIR}" ]]; then
    for candidate in /usr/local/nvidia/lib64 /usr/lib/nvidia; do
        if [[ -d "${candidate}" ]]; then
            QRL_DETECTED_DRIVER_LIBRARY_DIR="${candidate}"
            break
        fi
    done
fi
if [[ -z "${QRL_DETECTED_DRIVER_LIBRARY_DIR}" ]]; then
    candidates=(/usr/lib/nvidia-[0-9][0-9][0-9])
    for ((index=${#candidates[@]} - 1; index>=0; index--)); do
        if [[ -d "${candidates[index]}" ]]; then
            QRL_DETECTED_DRIVER_LIBRARY_DIR="${candidates[index]}"
            break
        fi
    done
fi
if [[ -n "${QRL_DETECTED_DRIVER_LIBRARY_DIR}" ]]; then
    export LD_LIBRARY_PATH="${QRL_DETECTED_DRIVER_LIBRARY_DIR}:${LD_LIBRARY_PATH}"
fi

#!/usr/bin/env bash
set -euo pipefail

# Provision the reproducible D4RL/MuJoCo environment used by this repository.
# The default downloads roughly 1.6 GB of D4RL data in addition to Python
# packages. All assets remain outside the Git checkout.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSET_ROOT="${QRL_ASSET_ROOT:-${ROOT_DIR}/../qrl-assets}"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
DATASET_MANIFEST="${D4RL_DATASET_MANIFEST:-${ROOT_DIR}/docs/d4rl_v2_datasets.tsv}"
MUJOCO_URL="${MUJOCO_URL:-https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz}"
MUJOCO_ARCHIVE_SHA256="${MUJOCO_ARCHIVE_SHA256:-a436ca2f4144c38b837205635bbd60ffe1162d5b44c87df22232795978d7d012}"
MAZE2D_DATASET_URL="${MAZE2D_DATASET_URL:-http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d}"
ANTMAZE_V2_DATASET_URL="${ANTMAZE_V2_DATASET_URL:-http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_v2}"
DOWNLOAD_RETRIES="${DOWNLOAD_RETRIES:-3}"

INSTALL_SYSTEM=1
INSTALL_PYTHON=1
INSTALL_MUJOCO=1
INSTALL_DATASETS=1
VERIFY=1


usage() {
    cat <<'EOF'
Usage: tools/setup_environment.sh [options]

Install the system packages, Python environment, MuJoCo 2.1.0, and the D4RL
datasets required by the repository's Maze2D and AntMaze experiments.

Options:
  --skip-system     Do not install Debian/Ubuntu system packages.
  --skip-python     Do not create or install the Python virtual environment.
  --skip-mujoco     Do not download MuJoCo 2.1.0.
  --skip-datasets   Do not download D4RL datasets.
  --skip-verify     Do not run the final CUDA/MuJoCo/D4RL preflight.
  -h, --help        Show this help message.

Environment overrides:
  QRL_ASSET_ROOT, VENV_DIR, PYTHON_BIN, TORCH_INDEX_URL,
  MUJOCO_URL, MUJOCO_ARCHIVE_SHA256, MAZE2D_DATASET_URL,
  ANTMAZE_V2_DATASET_URL, D4RL_DATASET_MANIFEST, DOWNLOAD_RETRIES.
EOF
}


while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-system) INSTALL_SYSTEM=0 ;;
        --skip-python) INSTALL_PYTHON=0 ;;
        --skip-mujoco) INSTALL_MUJOCO=0 ;;
        --skip-datasets) INSTALL_DATASETS=0 ;;
        --skip-verify) VERIFY=0 ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

if [[ "${ASSET_ROOT}" != /* ]]; then
    ASSET_ROOT="${ROOT_DIR}/${ASSET_ROOT}"
fi
export QRL_ASSET_ROOT="${ASSET_ROOT}"


require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Missing required command: $1" >&2
        exit 1
    fi
}


run_privileged() {
    if [[ "${EUID}" -eq 0 ]]; then
        "$@"
        return
    fi
    require_command sudo
    sudo "$@"
}


install_system_packages() {
    require_command apt-get
    run_privileged apt-get update
    run_privileged env DEBIAN_FRONTEND=noninteractive apt-get install -y \
        build-essential curl git patchelf libgl1-mesa-glx libosmesa6-dev \
        libglew-dev libglfw3 libglfw3-dev
}


sha256_matches() {
    local path="$1"
    local expected="$2"
    [[ -f "${path}" ]] && [[ "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]]
}


download_file() {
    local url="$1"
    local destination="$2"
    local temporary="${destination}.part"

    mkdir -p "$(dirname "${destination}")"
    rm -f "${temporary}"
    echo "Downloading ${url}"
    curl --fail --location --retry "${DOWNLOAD_RETRIES}" --retry-delay 2 \
        --output "${temporary}" "${url}"
    mv "${temporary}" "${destination}"
}


install_mujoco() {
    local mujoco_dir="${ASSET_ROOT}/mujoco/mujoco210"
    local archive="${ASSET_ROOT}/downloads/mujoco210-linux-x86_64.tar.gz"
    local install_parent
    local temporary_dir

    if [[ -f "${mujoco_dir}/bin/libmujoco210.so" ]]; then
        echo "MuJoCo already installed: ${mujoco_dir}"
        return
    fi
    if [[ -e "${mujoco_dir}" ]]; then
        echo "MuJoCo target exists but is incomplete: ${mujoco_dir}" >&2
        echo "Move or remove it, then rerun this script." >&2
        exit 1
    fi

    if ! sha256_matches "${archive}" "${MUJOCO_ARCHIVE_SHA256}"; then
        download_file "${MUJOCO_URL}" "${archive}"
    fi
    if ! sha256_matches "${archive}" "${MUJOCO_ARCHIVE_SHA256}"; then
        echo "MuJoCo archive checksum mismatch: ${archive}" >&2
        exit 1
    fi

    install_parent="$(dirname "${mujoco_dir}")"
    mkdir -p "${install_parent}"
    temporary_dir="$(mktemp -d "${install_parent}/.mujoco210.XXXXXX")"
    trap 'rm -rf "${temporary_dir}"' RETURN
    tar -xzf "${archive}" -C "${temporary_dir}"
    if [[ ! -f "${temporary_dir}/mujoco210/bin/libmujoco210.so" ]]; then
        echo "MuJoCo archive does not contain bin/libmujoco210.so" >&2
        exit 1
    fi
    mv "${temporary_dir}/mujoco210" "${mujoco_dir}"
    rm -rf "${temporary_dir}"
    trap - RETURN
    echo "Installed MuJoCo: ${mujoco_dir}"
}


dataset_url() {
    local filename="$1"
    case "${filename}" in
        maze2d-*.hdf5) printf '%s/%s\n' "${MAZE2D_DATASET_URL}" "${filename}" ;;
        Ant_maze_*.hdf5) printf '%s/%s\n' "${ANTMAZE_V2_DATASET_URL}" "${filename}" ;;
        *)
            echo "Unsupported dataset filename in manifest: ${filename}" >&2
            return 1
            ;;
    esac
}


dataset_is_valid() {
    local path="$1"
    local expected_size="$2"
    local expected_sha256="$3"
    [[ -f "${path}" ]] && [[ "$(stat -c '%s' "${path}")" == "${expected_size}" ]] && \
        sha256_matches "${path}" "${expected_sha256}"
}


install_datasets() {
    local dataset_dir="${ASSET_ROOT}/d4rl/datasets"
    local filename
    local expected_size
    local expected_sha256
    local destination
    local url

    require_command sha256sum
    require_command stat
    if [[ ! -f "${DATASET_MANIFEST}" ]]; then
        echo "Dataset manifest does not exist: ${DATASET_MANIFEST}" >&2
        exit 1
    fi
    mkdir -p "${dataset_dir}"

    while IFS=$'\t' read -r filename expected_size expected_sha256; do
        [[ -z "${filename}" || "${filename}" == \#* ]] && continue
        destination="${dataset_dir}/${filename}"
        if dataset_is_valid "${destination}" "${expected_size}" "${expected_sha256}"; then
            echo "Dataset already verified: ${filename}"
            continue
        fi
        url="$(dataset_url "${filename}")"
        download_file "${url}" "${destination}"
        if ! dataset_is_valid "${destination}" "${expected_size}" "${expected_sha256}"; then
            echo "Dataset validation failed: ${filename}" >&2
            exit 1
        fi
        echo "Downloaded and verified: ${filename}"
    done < "${DATASET_MANIFEST}"
}


if [[ "${INSTALL_SYSTEM}" -eq 1 ]]; then
    install_system_packages
fi

require_command curl
require_command sha256sum
require_command tar

if [[ "${INSTALL_PYTHON}" -eq 1 ]]; then
    VENV_DIR="${VENV_DIR}" PYTHON_BIN="${PYTHON_BIN:-python3.9}" \
        "${ROOT_DIR}/tools/bootstrap_environment.sh"
fi
if [[ "${INSTALL_MUJOCO}" -eq 1 ]]; then
    install_mujoco
fi
if [[ "${INSTALL_DATASETS}" -eq 1 ]]; then
    install_datasets
fi
if [[ "${VERIFY}" -eq 1 ]]; then
    source "${ROOT_DIR}/tools/qrl_env.sh"
    "${VENV_DIR}/bin/python" "${ROOT_DIR}/tools/verify_environment.py" \
        --checksums --smoke --require-cuda
fi

echo "Setup complete. Asset root: ${ASSET_ROOT}"

#!/usr/bin/env bash
set -euo pipefail

# Provision the reproducible D4RL and online simulator environment used here.
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
MICROMAMBA_URL="${MICROMAMBA_URL:-https://micro.mamba.pm/api/micromamba/linux-64/latest}"
BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON:-}"
PIP_RETRIES="${PIP_RETRIES:-3}"

INSTALL_SUBMODULES=1
INSTALL_PYTHON=1
INSTALL_MUJOCO=1
INSTALL_DATASETS=1
INSTALL_SIMULATORS=1
SIMULATORS_ONLY=0
VERIFY=1


usage() {
    cat <<'EOF'
Usage: tools/setup_environment.sh [options]

Initialize Git submodules, create the Python environment, install online
simulators and MuJoCo 2.1.0, and download the D4RL Maze2D/AntMaze datasets.

Options:
  --skip-submodules Do not initialize Git submodules.
  --skip-python     Do not create or install the Python virtual environment.
  --skip-mujoco     Do not download MuJoCo 2.1.0.
  --skip-datasets   Do not download D4RL datasets.
  --skip-simulators Do not install or verify online simulator packages.
  --simulators-only Only update and verify online simulators in an existing venv.
                    This skips submodules and D4RL datasets.
  --skip-verify     Do not run the final CUDA/MuJoCo/D4RL preflight.
  -h, --help        Show this help message.

Environment overrides:
  QRL_ASSET_ROOT, QRL_QUEUE_TASKS_FILE, VENV_DIR, PYTHON_BIN, TORCH_INDEX_URL, BOOTSTRAP_PYTHON,
  PIP_RETRIES,
  MICROMAMBA_URL, MICROMAMBA_ROOT, QRL_USER_GRAPHICS_PREFIX,
  MUJOCO_URL, MUJOCO_ARCHIVE_SHA256, MAZE2D_DATASET_URL,
  ANTMAZE_V2_DATASET_URL, D4RL_DATASET_MANIFEST, DOWNLOAD_RETRIES.
EOF
}


while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-submodules) INSTALL_SUBMODULES=0 ;;
        --skip-python) INSTALL_PYTHON=0 ;;
        --skip-mujoco) INSTALL_MUJOCO=0 ;;
        --skip-datasets) INSTALL_DATASETS=0 ;;
        --skip-simulators) INSTALL_SIMULATORS=0 ;;
        --simulators-only)
            SIMULATORS_ONLY=1
            INSTALL_SUBMODULES=0
            INSTALL_PYTHON=0
            INSTALL_DATASETS=0
            ;;
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

if [[ "${SIMULATORS_ONLY}" -eq 1 && "${INSTALL_SIMULATORS}" -eq 0 ]]; then
    echo "--simulators-only and --skip-simulators cannot be used together." >&2
    exit 2
fi

if [[ "${ASSET_ROOT}" != /* ]]; then
    ASSET_ROOT="${ROOT_DIR}/${ASSET_ROOT}"
fi
ASSET_ROOT="$(realpath -m "${ASSET_ROOT}")"
export QRL_ASSET_ROOT="${ASSET_ROOT}"
MICROMAMBA_ROOT="${MICROMAMBA_ROOT:-${ASSET_ROOT}/micromamba}"
MICROMAMBA_BIN="${MICROMAMBA_ROOT}/bin/micromamba"
MICROMAMBA_ENV="${MICROMAMBA_ROOT}/envs/python39"
GRAPHICS_ENV="${QRL_USER_GRAPHICS_PREFIX:-${MICROMAMBA_ROOT}/envs/graphics}"
QUEUE_TASKS_FILE="${QRL_QUEUE_TASKS_FILE:-${ROOT_DIR}/runs/qrl_queue/tasks.tsv}"


require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Missing required command: $1" >&2
        exit 1
    fi
}


is_python39() {
    local candidate="$1"
    command -v "${candidate}" >/dev/null 2>&1 && "${candidate}" - <<'PY'
import sys
raise SystemExit(0 if sys.version_info[:2] == (3, 9) else 1)
PY
}


resolve_download_python() {
    local candidate
    for candidate in "${BOOTSTRAP_PYTHON}" python3 python; do
        [[ -z "${candidate}" ]] && continue
        if command -v "${candidate}" >/dev/null 2>&1 && "${candidate}" - <<'PY'
import sys
raise SystemExit(0 if sys.version_info >= (3, 8) else 1)
PY
        then
            command -v "${candidate}"
            return
        fi
    done
    echo "A Python 3.8+ executable is required to bootstrap user-local Python 3.9." >&2
    echo "Set BOOTSTRAP_PYTHON to an existing Python executable and rerun." >&2
    exit 1
}


initialize_submodules() {
    require_command git
    git -C "${ROOT_DIR}" submodule update --init --recursive
}


sha256_matches() {
    local path="$1"
    local expected="$2"
    "${DOWNLOAD_PYTHON}" - "${path}" "${expected}" <<'PY'
import hashlib
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
expected = sys.argv[2]
if not path.is_file():
    raise SystemExit(1)
digest = hashlib.sha256()
with path.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
raise SystemExit(0 if digest.hexdigest() == expected else 1)
PY
}


download_file() {
    local url="$1"
    local destination="$2"
    local temporary="${destination}.part"

    mkdir -p "$(dirname "${destination}")"
    rm -f "${temporary}"
    echo "Downloading ${url}"
    "${DOWNLOAD_PYTHON}" - "${url}" "${temporary}" "${DOWNLOAD_RETRIES}" <<'PY'
import pathlib
import sys
import time
import urllib.request


def format_size(num_bytes):
    units = ("B", "KiB", "MiB", "GiB")
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024


def download_once(url, destination):
    request = urllib.request.Request(url, headers={"User-Agent": "qrl-environment-setup"})
    with urllib.request.urlopen(request) as response, destination.open("wb") as output:
        content_length = response.headers.get("Content-Length")
        total = int(content_length) if content_length and content_length.isdigit() else None
        received = 0
        start = last_update = time.monotonic()
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            output.write(block)
            received += len(block)
            now = time.monotonic()
            if now - last_update >= 0.2 or (total is not None and received >= total):
                speed = received / max(now - start, 1e-9)
                if total is None:
                    message = f"  {format_size(received)}  {format_size(speed)}/s"
                else:
                    percent = min(received / total * 100, 100.0)
                    message = (
                        f"  {percent:6.2f}%  {format_size(received)}/{format_size(total)}"
                        f"  {format_size(speed)}/s"
                    )
                print(f"\r{message}", end="", flush=True)
                last_update = now
    print()


url = sys.argv[1]
destination = pathlib.Path(sys.argv[2])
retries = int(sys.argv[3])
last_error = None
for attempt in range(retries):
    try:
        if attempt:
            print(f"Retrying download ({attempt + 1}/{retries})")
        download_once(url, destination)
        break
    except Exception as error:
        last_error = error
        destination.unlink(missing_ok=True)
        if attempt + 1 == retries:
            raise
        time.sleep(2)
if not destination.is_file():
    raise RuntimeError(f"download did not create {destination}: {last_error}")
PY
    mv "${temporary}" "${destination}"
}


install_micromamba() {
    local archive="${ASSET_ROOT}/downloads/micromamba-linux-64.tar.bz2"
    local temporary_dir

    if [[ -x "${MICROMAMBA_BIN}" ]]; then
        return
    fi

    mkdir -p "${MICROMAMBA_ROOT}"
    download_file "${MICROMAMBA_URL}" "${archive}"
    temporary_dir="$(mktemp -d "${MICROMAMBA_ROOT}/.extract.XXXXXX")"
    trap 'rm -rf "${temporary_dir}"' RETURN
    "${DOWNLOAD_PYTHON}" - "${archive}" "${temporary_dir}" <<'PY'
import pathlib
import sys
import tarfile

archive = pathlib.Path(sys.argv[1])
destination = pathlib.Path(sys.argv[2])
with tarfile.open(archive, "r:bz2") as tar:
    member = tar.getmember("bin/micromamba")
    tar.extract(member, destination)
PY
    mkdir -p "$(dirname "${MICROMAMBA_BIN}")"
    mv "${temporary_dir}/bin/micromamba" "${MICROMAMBA_BIN}"
    chmod +x "${MICROMAMBA_BIN}"
    rm -rf "${temporary_dir}"
    trap - RETURN
}


resolve_python39() {
    local requested="${PYTHON_BIN:-}"

    if [[ -n "${requested}" ]] && is_python39 "${requested}"; then
        command -v "${requested}"
        return
    fi
    if [[ -n "${requested}" ]]; then
        echo "Requested PYTHON_BIN is not an available Python 3.9: ${requested}" >&2
        echo "Provisioning a user-local Python 3.9 instead." >&2
    fi
    if is_python39 python3.9; then
        command -v python3.9
        return
    fi

    install_micromamba >&2
    if [[ ! -x "${MICROMAMBA_ENV}/bin/python" ]]; then
        echo "Creating user-local Python 3.9 with micromamba." >&2
        MAMBA_ROOT_PREFIX="${MICROMAMBA_ROOT}/root" "${MICROMAMBA_BIN}" create \
            --yes --prefix "${MICROMAMBA_ENV}" --channel conda-forge python=3.9 >&2
    fi
    if ! is_python39 "${MICROMAMBA_ENV}/bin/python"; then
        echo "micromamba did not create a usable Python 3.9 environment." >&2
        exit 1
    fi
    printf '%s\n' "${MICROMAMBA_ENV}/bin/python"
}


install_user_graphics_dependencies() {
    if [[ "$(uname -s)" != "Linux" ]]; then
        return
    fi
    if [[ -f "${GRAPHICS_ENV}/include/X11/Xlib.h" &&
          -f "${GRAPHICS_ENV}/include/X11/X.h" &&
          -f "${GRAPHICS_ENV}/include/GL/glew.h" &&
          -x "${GRAPHICS_ENV}/bin/patchelf" ]]; then
        echo "User-local graphics build dependencies already installed: ${GRAPHICS_ENV}"
        return
    fi

    install_micromamba
    echo "Installing user-local graphics build dependencies."
    if [[ -d "${GRAPHICS_ENV}/conda-meta" ]]; then
        MAMBA_ROOT_PREFIX="${MICROMAMBA_ROOT}/root" "${MICROMAMBA_BIN}" install \
            --yes --prefix "${GRAPHICS_ENV}" --channel conda-forge \
            xorg-libx11 xorg-xorgproto glew patchelf
    else
        MAMBA_ROOT_PREFIX="${MICROMAMBA_ROOT}/root" "${MICROMAMBA_BIN}" create \
            --yes --prefix "${GRAPHICS_ENV}" --channel conda-forge \
            xorg-libx11 xorg-xorgproto glew patchelf
    fi
    if [[ ! -f "${GRAPHICS_ENV}/include/X11/Xlib.h" ||
          ! -f "${GRAPHICS_ENV}/include/X11/X.h" ||
          ! -f "${GRAPHICS_ENV}/include/GL/glew.h" ||
          ! -x "${GRAPHICS_ENV}/bin/patchelf" ]]; then
        echo "User-local graphics environment is incomplete: ${GRAPHICS_ENV}" >&2
        exit 1
    fi
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
    "${DOWNLOAD_PYTHON}" - "${archive}" "${temporary_dir}" <<'PY'
import pathlib
import sys
import tarfile

archive = pathlib.Path(sys.argv[1])
destination = pathlib.Path(sys.argv[2])
with tarfile.open(archive, "r:gz") as tar:
    tar.extractall(destination)
PY
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
    "${DOWNLOAD_PYTHON}" - "${path}" "${expected_size}" <<'PY' && \
        sha256_matches "${path}" "${expected_sha256}"
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
raise SystemExit(0 if path.is_file() and path.stat().st_size == int(sys.argv[2]) else 1)
PY
}


install_datasets() {
    local dataset_dir="${ASSET_ROOT}/d4rl/datasets"
    local filename
    local expected_size
    local expected_sha256
    local destination
    local url

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


install_online_simulators() {
    local requirements_file="${ROOT_DIR}/requirements/online-simulators-py39.txt"
    if [[ ! -f "${requirements_file}" ]]; then
        echo "Simulator requirements do not exist: ${requirements_file}" >&2
        exit 1
    fi
    echo "Installing or repairing the pinned local online simulator stack."
    # The source server uses Gym 0.18 legacy MuJoCo backends. Their dynamics
    # differ from Gymnasium v4, so remove packages that would make the adapter
    # silently select another backend.
    if "${VENV_PYTHON}" -m pip show gymnasium >/dev/null 2>&1 || \
            "${VENV_PYTHON}" -m pip show gymnasium-robotics >/dev/null 2>&1; then
        echo "Removing Gymnasium packages to match the source server backends."
        "${VENV_PYTHON}" -m pip uninstall --yes gymnasium gymnasium-robotics
    fi
    "${VENV_PYTHON}" -m pip install \
        --retries "${PIP_RETRIES}" --prefer-binary \
        -r "${requirements_file}"
    # Gym 0.18.0's Pillow<=7.2.0 metadata cannot be satisfied by a Python 3.9
    # binary wheel. Its runtime dependencies are pinned in the file above.
    "${VENV_PYTHON}" -m pip install \
        --retries "${PIP_RETRIES}" --no-deps "gym==0.18.0"
}


initialize_queue_tasks() {
    if [[ -e "${QUEUE_TASKS_FILE}" ]]; then
        echo "Queue task list already exists: ${QUEUE_TASKS_FILE}"
        return
    fi
    mkdir -p "$(dirname "${QUEUE_TASKS_FILE}")"
    printf '%s\n' \
        '# Runtime queue task list. Add task rows locally before starting the scheduler.' \
        '# Format: task_id<TAB>mode<TAB>env_name<TAB>seed<TAB>steps<TAB>params<TAB>extra_args' \
        > "${QUEUE_TASKS_FILE}"
    echo "Created empty queue task list: ${QUEUE_TASKS_FILE}"
}


DOWNLOAD_PYTHON="$(resolve_download_python)"
initialize_queue_tasks
if [[ "${INSTALL_SUBMODULES}" -eq 1 ]]; then
    initialize_submodules
fi
if [[ "${INSTALL_PYTHON}" -eq 1 ]]; then
    PYTHON39_BIN="$(resolve_python39)"
    VENV_DIR="${VENV_DIR}" PYTHON_BIN="${PYTHON39_BIN}" \
        "${ROOT_DIR}/tools/bootstrap_environment.sh"
fi
VENV_PYTHON="${VENV_DIR}/bin/python"
if [[ ! -x "${VENV_PYTHON}" ]]; then
    echo "Virtual environment Python is missing: ${VENV_PYTHON}" >&2
    echo "Run without --skip-python or set VENV_DIR to an existing environment." >&2
    exit 1
fi
if ! "${VENV_PYTHON}" - <<'PY'
import sys
raise SystemExit(0 if sys.version_info[:2] == (3, 9) else 1)
PY
then
    echo "Existing virtual environment must use Python 3.9: ${VENV_DIR}" >&2
    exit 1
fi
install_user_graphics_dependencies
if [[ "${INSTALL_MUJOCO}" -eq 1 ]]; then
    install_mujoco
fi
if [[ "${INSTALL_DATASETS}" -eq 1 ]]; then
    install_datasets
fi
if [[ "${INSTALL_SIMULATORS}" -eq 1 ]]; then
    install_online_simulators
fi
if [[ "${VERIFY}" -eq 1 ]]; then
    source "${ROOT_DIR}/tools/qrl_env.sh"
    if [[ "${SIMULATORS_ONLY}" -eq 0 ]]; then
        "${VENV_PYTHON}" "${ROOT_DIR}/tools/verify_environment.py" \
            --checksums --smoke --require-cuda
    fi
    if [[ "${INSTALL_SIMULATORS}" -eq 1 ]]; then
        "${VENV_PYTHON}" "${ROOT_DIR}/tools/verify_online_simulators.py"
    fi
fi

echo "Setup complete. Asset root: ${ASSET_ROOT}"

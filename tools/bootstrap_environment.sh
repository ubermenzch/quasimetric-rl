#!/usr/bin/env bash
set -euo pipefail

# Create the Python environment used by the current D4RL experiments.
# Run `source tools/qrl_env.sh` after placing MuJoCo and datasets, then run
# `tools/verify_environment.py --smoke --require-cuda` before launching training.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.9}"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
PIP="${VENV_DIR}/bin/python -m pip"
D4RL_COMMIT="d842aa194b416e564e54b0730d9f934e3e32f854"
TORCH_VERSION="${TORCH_VERSION:-2.8.0}"
# Override with a wheel index compatible with the target driver. Set this to an
# empty string to install the selected torch version from the default index.
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python 3.9 is required; set PYTHON_BIN to a Python 3.9 executable." >&2
    exit 1
fi

"${PYTHON_BIN}" - <<'PY'
import sys
if sys.version_info[:2] != (3, 9):
    raise SystemExit(f"Expected Python 3.9, got {sys.version.split()[0]}")
PY

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

if [[ ! -f "${ROOT_DIR}/third_party/torch-quasimetric/setup.py" ]]; then
    echo "Missing torch-quasimetric submodule. Run:" >&2
    echo "  git submodule update --init --recursive" >&2
    exit 1
fi

${PIP} install --upgrade "pip==23.3.2" setuptools wheel
if [[ -n "${TORCH_INDEX_URL}" ]]; then
    ${PIP} install --index-url "${TORCH_INDEX_URL}" "torch==${TORCH_VERSION}"
else
    ${PIP} install "torch==${TORCH_VERSION}"
fi
${PIP} install -r "${ROOT_DIR}/requirements/offline-py39.txt"

# D4RL's declared mjrl dependency is only used by unsupported hand/kitchen
# suites. Install without dependency resolution to keep the point-maze setup
# stable and avoid pulling mutable Git main branches.
${PIP} install --no-deps "d4rl @ git+https://github.com/rail-berkeley/d4rl@${D4RL_COMMIT}"
${PIP} install -e "${ROOT_DIR}/third_party/torch-quasimetric"

echo "Environment created at ${VENV_DIR}"
echo "Next: source tools/qrl_env.sh && ${VENV_DIR}/bin/python tools/verify_environment.py --smoke --require-cuda"

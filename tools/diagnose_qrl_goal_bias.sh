#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${ROOT_DIR}"

source tools/qrl_env.sh
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="${QRL_PYTHON_BIN:-.venv/bin/python}"
exec "${PYTHON_BIN}" tools/diagnose_qrl_goal_bias.py "$@"

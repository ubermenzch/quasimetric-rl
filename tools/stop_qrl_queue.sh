#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="${BASH_SOURCE[0]%/*}"
if [[ "${SCRIPT_DIR}" == "${BASH_SOURCE[0]}" ]]; then SCRIPT_DIR=.; fi
cd "${SCRIPT_DIR}/.."
PYTHON_BIN="${QRL_PYTHON_BIN:-.venv/bin/python}"
exec "${PYTHON_BIN}" tools/qrl_queue_control.py stop "$@"

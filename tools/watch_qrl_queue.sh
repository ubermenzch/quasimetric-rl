#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONFIG="${CONFIG:-configs/qrl_queue.env}"
PYTHON_BIN="${QRL_PYTHON_BIN:-.venv/bin/python}"
exec "${PYTHON_BIN}" tools/watch_qrl_queue.py --config "${CONFIG}" "$@"

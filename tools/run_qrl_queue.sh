#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONFIG="${CONFIG:-configs/qrl_queue.env}"
PYTHON_BIN="${QRL_PYTHON_BIN:-.venv/bin/python}"
RUNNER_LOG="${QRL_QUEUE_RUNNER_LOG:-logs/qrl_queue/runner.log}"

mkdir -p "$(dirname "${RUNNER_LOG}")"
nohup setsid "${PYTHON_BIN}" tools/run_qrl_queue.py --config "${CONFIG}" "$@" \
  </dev/null >>"${RUNNER_LOG}" 2>&1 &

echo "QRL queue runner started in the background; log: ${RUNNER_LOG}"

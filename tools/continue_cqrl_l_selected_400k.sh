#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${QRL_PYTHON_BIN:-.venv/bin/python}"
TASK_LIST="configs/continue_cqrl_l_selected_400k.txt"
CONFIG="${CONFIG:-configs/qrl_queue.env}"

tools/stop_qrl_queue.sh
sleep 2
"${PYTHON_BIN}" tools/prepare_online_continuations.py \
  --config "${CONFIG}" \
  --target-env-steps 400000 \
  --task-list "${TASK_LIST}"

echo "Continuation tasks prepared; start the queue separately with CONFIG=${CONFIG} tools/run_qrl_queue.sh"

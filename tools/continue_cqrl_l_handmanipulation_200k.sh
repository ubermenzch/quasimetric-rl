#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${QRL_PYTHON_BIN:-.venv/bin/python}"
TASK_LIST="configs/continue_cqrl_l_handmanipulation_200k.txt"

tools/stop_qrl_queue.sh
sleep 2
"${PYTHON_BIN}" tools/prepare_online_continuations.py \
  --target-env-steps 200000 \
  --task-list "${TASK_LIST}"
tools/run_qrl_queue.sh

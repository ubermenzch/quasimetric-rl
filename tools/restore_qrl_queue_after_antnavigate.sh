#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONFIG="${CONFIG:-configs/qrl_queue.env}"
STATUS_DIR="runs/qrl_queue/status"
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
elif (($#)); then
  echo "Usage: $0 [--dry-run]" >&2
  exit 2
fi

if [[ ! -f "${CONFIG}" ]]; then
  echo "Missing complete queue config: ${CONFIG}" >&2
  exit 2
fi

FULL_TASKS="$(.venv/bin/python - "${CONFIG}" <<'PY'
import sys
from tools.run_qrl_queue import cfg, parse_config, resolve_path

config = parse_config(resolve_path(sys.argv[1]))
print(resolve_path(cfg(config, 'TASKS_FILE', 'runs/qrl_queue/tasks.tsv')))
PY
)"

if ((DRY_RUN == 0)); then
  scheduler_pid="$(sed -n 's/^pid=\([0-9][0-9]*\)$/\1/p' \
    runs/qrl_queue/qrl_queue.lock 2>/dev/null | head -n 1 || true)"
  tools/stop_qrl_queue.sh
  if [[ -n "${scheduler_pid}" ]]; then
    deadline=$((SECONDS + 30))
    while kill -0 "${scheduler_pid}" 2>/dev/null; do
      if ((SECONDS >= deadline)); then
        echo "Timed out waiting for scheduler PID ${scheduler_pid} to exit." >&2
        exit 2
      fi
      sleep 1
    done
  fi
fi

.venv/bin/python - "${FULL_TASKS}" "${STATUS_DIR}" "${DRY_RUN}" <<'PY'
import sys

from tools.run_qrl_queue import (
    read_status,
    read_tasks,
    resolve_path,
    status_extra_fields,
    write_status,
)

tasks = read_tasks(resolve_path(sys.argv[1]))
status_dir = resolve_path(sys.argv[2])
dry_run = sys.argv[3] == '1'
marker = 'antnavigate_ablations_only'
restored = 0

for task in tasks:
    status = read_status(status_dir, task.task_id)
    if (
        status.get('state') != 'PAUSED'
        or status.get('manual_pause_scope') != marker
    ):
        continue
    previous_state = status.get('pre_manual_pause_state', 'PENDING') or 'PENDING'
    restored_state = 'FAILED' if previous_state == 'FAILED' else 'PENDING'
    extra = status_extra_fields(status)
    previous_finished_at = extra.pop('pre_manual_pause_finished_at', '')
    for key in (
        'manual_pause_scope',
        'pre_manual_pause_state',
        'finished_at',
        'gpu',
        'pid',
    ):
        extra.pop(key, None)
    if extra.get('note') == 'paused_while_antnavigate_ablations_run':
        extra.pop('note')
    if restored_state == 'FAILED' and previous_finished_at:
        extra['finished_at'] = previous_finished_at
    if not dry_run:
        write_status(status_dir, task, restored_state, extra)
    restored += 1

verb = 'Would restore' if dry_run else 'Restored'
print(f'{verb} {restored} manually paused tasks.')
PY

if ((DRY_RUN)); then
  echo "DRY_RUN complete; scheduler and statuses were not changed."
  exit 0
fi

CONFIG="${CONFIG}" tools/run_qrl_queue.sh
echo "Complete queue restored. Tasks paused for the AntNavigate focus are schedulable again."

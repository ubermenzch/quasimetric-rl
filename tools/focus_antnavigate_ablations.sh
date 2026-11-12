#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# CONFIG always points at the complete queue. The generated focused config is
# stored under runs/ so it cannot replace the source config.
CONFIG="${CONFIG:-configs/qrl_queue.env}"
FOCUSED_TASKS="${FOCUSED_TASKS:-runs/qrl_queue/tasks_antnavigate_ablations_only.tsv}"
FOCUSED_CONFIG="${FOCUSED_CONFIG:-runs/qrl_queue/qrl_queue_antnavigate_ablations_only.env}"
FOCUSED_NOTIFICATION_STATE="runs/qrl_queue/status/notifications_antnavigate_ablations_only.json"
STATUS_DIR="runs/qrl_queue/status"
DRY_RUN=0

TARGET_MANIFESTS=(
  configs/go_qrl_plain_l_max4_antnavigate_5seed.tsv
  configs/go_qrl_residual_l_max1_antnavigate_5seed.tsv
)

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
elif (($#)); then
  echo "Usage: $0 [--dry-run]" >&2
  exit 2
fi

for required in "${CONFIG}" "${TARGET_MANIFESTS[@]}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing required file: ${required}" >&2
    exit 2
  fi
done

FULL_TASKS="$(.venv/bin/python - "${CONFIG}" <<'PY'
import sys
from tools.run_qrl_queue import cfg, parse_config, resolve_path

config = parse_config(resolve_path(sys.argv[1]))
print(resolve_path(cfg(config, 'TASKS_FILE', 'runs/qrl_queue/tasks.tsv')))
PY
)"
if [[ ! -f "${FULL_TASKS}" ]]; then
  echo "Complete task table does not exist: ${FULL_TASKS}" >&2
  exit 2
fi

mapfile -t target_ids < <(.venv/bin/python - "${FULL_TASKS}" "${TARGET_MANIFESTS[@]}" <<'PY'
import sys
from tools.run_qrl_queue import read_tasks, resolve_path

full_tasks = read_tasks(resolve_path(sys.argv[1]))
target_tasks = [
    task
    for source in sys.argv[2:]
    for task in read_tasks(resolve_path(source))
]
target_ids = [task.task_id for task in target_tasks]
full_ids = {task.task_id for task in full_tasks}
if len(target_ids) != 10 or len(set(target_ids)) != 10:
    raise SystemExit(f'Expected 10 unique AntNavigate tasks, found {len(set(target_ids))}')
if any(task.env_name != 'AntNavigate-v4' for task in target_tasks):
    raise SystemExit('Target manifests contain a non-AntNavigate environment')
missing = set(target_ids) - full_ids
if missing:
    raise SystemExit(f'{len(missing)} AntNavigate tasks are absent from {sys.argv[1]}')
for task_id in target_ids:
    print(task_id)
PY
)

declare -A target_id_set=()
for task_id in "${target_ids[@]}"; do
  target_id_set["${task_id}"]=1
done

full_task_count="$(.venv/bin/python - "${FULL_TASKS}" <<'PY'
import sys
from tools.run_qrl_queue import read_tasks, resolve_path
print(len(read_tasks(resolve_path(sys.argv[1]))))
PY
)"

# Stop only the scheduler. Existing training processes keep running until the
# exact non-ablation PIDs have been resolved and validated below.
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

non_ablation_pids=()
non_ablation_ids=()
for status_file in "${STATUS_DIR}"/*.status; do
  [[ -f "${status_file}" ]] || continue
  state="$(sed -n 's/^state=//p' "${status_file}" | head -n 1)"
  [[ "${state}" == "RUNNING" ]] || continue
  task_id="$(sed -n 's/^task_id=//p' "${status_file}" | head -n 1)"
  [[ -n "${task_id}" ]] || continue
  [[ -z "${target_id_set[$task_id]:-}" ]] || continue
  # Ignore unrelated status files that are not in the complete queue.
  if ! awk -F '\t' -v wanted="${task_id}" '$1 == wanted { found=1; exit } END { exit !found }' "${FULL_TASKS}"; then
    continue
  fi
  pid="$(sed -n 's/^pid=\([0-9][0-9]*\)$/\1/p' "${status_file}" | head -n 1)"
  [[ -n "${pid}" ]] || continue
  non_ablation_pids+=("${pid}")
  non_ablation_ids+=("${task_id}")
done

echo "Complete queue: ${full_task_count} tasks"
echo "AntNavigate ablations retained: ${#target_ids[@]}"
echo "Other tasks paused by exclusion: $((full_task_count - ${#target_ids[@]}))"
echo "Currently running non-ablation tasks to stop: ${#non_ablation_pids[@]}"

for index in "${!non_ablation_pids[@]}"; do
  pid="${non_ablation_pids[index]}"
  task_id="${non_ablation_ids[index]}"
  if ! kill -0 "${pid}" 2>/dev/null; then
    echo "Already stopped: ${task_id} (PID ${pid})"
    continue
  fi
  owner="$(ps -o user= -p "${pid}" | tr -d '[:space:]')"
  cmdline="$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)"
  if [[ "${owner}" != "$(id -un)" ]]; then
    echo "Refusing to signal PID ${pid}: owner is ${owner:-unknown}." >&2
    exit 2
  fi
  if [[ "${cmdline}" != *"-m online.main"* \
        || "${cmdline}" != *"output_folder=${task_id}"* ]]; then
    echo "Refusing to signal PID ${pid}: command does not match ${task_id}." >&2
    echo "Command: ${cmdline:-unavailable}" >&2
    exit 2
  fi
  if ((DRY_RUN)); then
    echo "DRY_RUN would stop non-ablation task ${task_id} (PID ${pid})"
  else
    kill -TERM "${pid}"
    echo "Sent SIGTERM to non-ablation task ${task_id} (PID ${pid})"
  fi
done

if ((DRY_RUN)); then
  echo "DRY_RUN complete; scheduler and training processes were not changed."
  exit 0
fi

echo "Waiting for non-ablation processes to exit..."
deadline=$((SECONDS + 180))
while :; do
  alive=()
  for pid in "${non_ablation_pids[@]}"; do
    kill -0 "${pid}" 2>/dev/null && alive+=("${pid}")
  done
  ((${#alive[@]} == 0)) && break
  if ((SECONDS >= deadline)); then
    echo "Timed out waiting for PIDs: ${alive[*]}" >&2
    echo "The scheduler remains stopped; the focused queue was not started." >&2
    exit 2
  fi
  sleep 2
done

# Persist an explicit manual PAUSED state for every unfinished non-ablation
# task. DONE tasks and tasks that were already PAUSED for another reason are
# intentionally left unchanged. The marker lets the restore script distinguish
# this temporary hold from genuine task failures or identity problems.
.venv/bin/python - "${FULL_TASKS}" "${STATUS_DIR}" "${TARGET_MANIFESTS[@]}" <<'PY'
import sys

from tools.run_qrl_queue import (
    read_status,
    read_tasks,
    resolve_path,
    status_extra_fields,
    write_status,
)

full_tasks = read_tasks(resolve_path(sys.argv[1]))
status_dir = resolve_path(sys.argv[2])
target_ids = {
    task.task_id
    for manifest in sys.argv[3:]
    for task in read_tasks(resolve_path(manifest))
}
marker = 'antnavigate_ablations_only'
newly_paused = 0
already_manually_paused = 0
already_paused_other_reason = 0
done = 0

for task in full_tasks:
    if task.task_id in target_ids:
        continue
    status = read_status(status_dir, task.task_id)
    state = status.get('state', 'PENDING') or 'PENDING'
    if state == 'DONE':
        done += 1
        continue
    if state == 'PAUSED':
        if status.get('manual_pause_scope') == marker:
            already_manually_paused += 1
        else:
            already_paused_other_reason += 1
        continue

    extra = status_extra_fields(status)
    previous_pid = status.get('pid', '')
    previous_gpu = status.get('gpu', '')
    previous_finished_at = status.get('finished_at', '')
    for key in ('gpu', 'pid', 'finished_at'):
        extra.pop(key, None)
    extra.update({
        'manual_pause_scope': marker,
        'pre_manual_pause_state': state,
        'pre_manual_pause_finished_at': previous_finished_at,
        'note': 'paused_while_antnavigate_ablations_run',
    })
    if previous_pid:
        extra['previous_pid'] = previous_pid
    if previous_gpu:
        extra['previous_gpu'] = previous_gpu
    write_status(status_dir, task, 'PAUSED', extra)
    newly_paused += 1

print(
    'Explicit PAUSED statuses: '
    f'new={newly_paused}, existing_focus={already_manually_paused}, '
    f'existing_other={already_paused_other_reason}, done_unchanged={done}.'
)
PY

.venv/bin/python - "${FULL_TASKS}" "${FOCUSED_TASKS}" "${TARGET_MANIFESTS[@]}" <<'PY'
import os
from pathlib import Path
import sys
import tempfile

from tools.run_qrl_queue import read_tasks, resolve_path

source = resolve_path(sys.argv[1])
destination = resolve_path(sys.argv[2])
target_ids = {
    task.task_id
    for manifest in sys.argv[3:]
    for task in read_tasks(resolve_path(manifest))
}

kept_lines = []
kept_ids = set()
for line in source.read_text(errors='replace').splitlines(keepends=True):
    stripped = line.strip()
    if not stripped or stripped.startswith('#'):
        kept_lines.append(line)
        continue
    task_id = line.split('\t', 1)[0]
    if task_id in target_ids:
        kept_lines.append(line)
        kept_ids.add(task_id)
if kept_ids != target_ids:
    raise SystemExit(
        f'Expected to retain {len(target_ids)} ablations, retained {len(kept_ids)}'
    )

destination.parent.mkdir(parents=True, exist_ok=True)
descriptor, temporary_name = tempfile.mkstemp(
    prefix=f'{destination.name}.', suffix='.tmp', dir=destination.parent,
)
temporary = Path(temporary_name)
try:
    with os.fdopen(descriptor, 'w') as handle:
        handle.writelines(kept_lines)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
finally:
    temporary.unlink(missing_ok=True)
print(f'Wrote focused task table with {len(kept_ids)} AntNavigate ablations.')
PY

cp "${CONFIG}" "${FOCUSED_CONFIG}"
for required_key in TASKS_FILE NOTIFY_STATE_FILE NOTIFY_QUEUE_DONE; do
  if ! grep -q "^${required_key}=" "${FOCUSED_CONFIG}"; then
    echo "${required_key} is missing from ${FOCUSED_CONFIG}." >&2
    exit 2
  fi
done
sed -i \
  -e "s|^TASKS_FILE=.*|TASKS_FILE=\"${FOCUSED_TASKS}\"|" \
  -e "s|^NOTIFY_STATE_FILE=.*|NOTIFY_STATE_FILE=\"${FOCUSED_NOTIFICATION_STATE}\"|" \
  -e 's|^NOTIFY_QUEUE_DONE=.*|NOTIFY_QUEUE_DONE=0|' \
  "${FOCUSED_CONFIG}"

CONFIG="${FOCUSED_CONFIG}" tools/run_qrl_queue.sh
echo "Queue restarted with only the 10 AntNavigate ablations schedulable."
echo "The complete task table and all completed statuses remain unchanged."
echo "After the ablations finish, restore the complete queue with:"
echo "  tools/restore_qrl_queue_after_antnavigate.sh"

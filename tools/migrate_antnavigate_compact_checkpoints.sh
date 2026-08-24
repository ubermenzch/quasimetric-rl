#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONFIG="${CONFIG:-configs/qrl_queue.env}"
STATUS_DIR="runs/qrl_queue/status"
TARGET_GLOBS=(
  "${STATUS_DIR}/ablation_GO-QRL+Max4-Plain-L_500k_50kckpt_val500_test1000_antnavigate_v4_l500k_online_s"*.status
  "${STATUS_DIR}/ablation_GO-QRL+Max1-ResidualLN-SiLU-L_500k_50kckpt_val500_test1000_antnavigate_v4_l500k_online_s"*.status
)

PIDS=()
for status_file in "${TARGET_GLOBS[@]}"; do
  [[ -f "${status_file}" ]] || continue
  pid="$(sed -n 's/^pid=\([0-9][0-9]*\)$/\1/p' "${status_file}" | head -n 1)"
  [[ -n "${pid}" ]] && PIDS+=("${pid}")
done

tools/stop_qrl_queue.sh --with-jobs

# The scheduler may already have exited while leaving RUNNING statuses and
# orphaned training processes. Validate each recorded process before signaling
# it directly so this recovery path cannot terminate an unrelated user job.
for pid in "${PIDS[@]}"; do
  if ! kill -0 "${pid}" 2>/dev/null; then
    continue
  fi
  owner="$(ps -o user= -p "${pid}" | tr -d '[:space:]')"
  cmdline="$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)"
  if [[ "${owner}" != "$(id -un)" ]]; then
    echo "Refusing to signal PID ${pid}: owner is ${owner:-unknown}." >&2
    exit 2
  fi
  if [[ "${cmdline}" != *"-m online.main"* \
        || "${cmdline}" != *"output_folder=ablation_GO-QRL+"* \
        || "${cmdline}" != *"antnavigate_v4_l500k_online_"* ]]; then
    echo "Refusing to signal PID ${pid}: command is not a target AntNavigate task." >&2
    echo "Command: ${cmdline:-unavailable}" >&2
    exit 2
  fi
  kill -TERM "${pid}"
  echo "Sent SIGTERM to target training PID ${pid}."
done

echo "Waiting for the 10 target training processes to exit..."
deadline=$((SECONDS + 180))
while :; do
  alive=()
  for pid in "${PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      alive+=("${pid}")
    fi
  done
  if ((${#alive[@]} == 0)); then
    break
  fi
  if ((SECONDS >= deadline)); then
    echo "Timed out waiting for target training PIDs to exit: ${alive[*]}" >&2
    echo "The queue remains stopped; no task definitions were changed." >&2
    exit 2
  fi
  sleep 2
done

.venv/bin/python tools/migrate_antnavigate_compact_checkpoints.py \
  --config "${CONFIG}" --yes

CONFIG="${CONFIG}" tools/run_qrl_queue.sh
echo "Compact checkpoint migration completed and the queue was restarted."

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

LOCK_FILE="${QRL_QUEUE_LOCK_FILE:-runs/qrl_queue/qrl_queue.lock}"
WITH_JOBS=0

usage() {
  cat <<'EOF'
Usage: tools/stop_qrl_queue.sh [--with-jobs] [--lock-file PATH]

Stop the QRL queue scheduler identified by its lock file.

  --with-jobs       Also send SIGTERM to the scheduler's process group.
                    Online jobs discard work after their latest committed
                    checkpoint and are requeued when the scheduler restarts.
  --lock-file PATH  Use a non-default queue lock file.
  -h, --help        Show this help.

By default, only the scheduler is stopped; running training jobs continue.
EOF
}

while (($#)); do
  case "$1" in
    --with-jobs)
      WITH_JOBS=1
      shift
      ;;
    --lock-file)
      if (($# < 2)); then
        echo "--lock-file requires a path" >&2
        exit 2
      fi
      LOCK_FILE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "${LOCK_FILE}" ]]; then
  echo "QRL queue scheduler is not running: lock file not found: ${LOCK_FILE}"
  exit 0
fi

SCHEDULER_PID="$(sed -n 's/^pid=\([0-9][0-9]*\)$/\1/p' "${LOCK_FILE}" | head -n 1)"
if [[ -z "${SCHEDULER_PID}" || "${SCHEDULER_PID}" -le 1 ]]; then
  echo "Refusing to stop scheduler: invalid PID in ${LOCK_FILE}" >&2
  exit 1
fi

if ! kill -0 "${SCHEDULER_PID}" 2>/dev/null; then
  echo "QRL queue scheduler is not running; stale PID ${SCHEDULER_PID} in ${LOCK_FILE}"
  exit 0
fi

CMDLINE="$(tr '\0' ' ' < "/proc/${SCHEDULER_PID}/cmdline" 2>/dev/null || true)"
if [[ "${CMDLINE}" != *"tools/run_qrl_queue.py"* ]]; then
  echo "Refusing to signal PID ${SCHEDULER_PID}: it is not the QRL queue scheduler" >&2
  echo "Command: ${CMDLINE:-unavailable}" >&2
  exit 1
fi

if ((WITH_JOBS)); then
  PROCESS_GROUP="$(ps -o pgid= -p "${SCHEDULER_PID}" | tr -d '[:space:]')"
  CURRENT_GROUP="$(ps -o pgid= -p "$$" | tr -d '[:space:]')"
  if [[ -z "${PROCESS_GROUP}" || ! "${PROCESS_GROUP}" =~ ^[0-9]+$ ]]; then
    echo "Refusing to stop jobs: cannot resolve scheduler process group" >&2
    exit 1
  fi
  if [[ "${PROCESS_GROUP}" == "${CURRENT_GROUP}" || "${PROCESS_GROUP}" -le 1 ]]; then
    echo "Refusing to signal unsafe process group ${PROCESS_GROUP}" >&2
    exit 1
  fi
  kill -TERM -- "-${PROCESS_GROUP}"
  echo "Sent SIGTERM to QRL scheduler process group ${PROCESS_GROUP} (scheduler PID ${SCHEDULER_PID})."
else
  kill -TERM "${SCHEDULER_PID}"
  echo "Sent SIGTERM to QRL scheduler PID ${SCHEDULER_PID}; running training jobs were left active."
fi

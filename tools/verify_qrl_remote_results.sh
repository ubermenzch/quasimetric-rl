#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONFIG="${QRL_QUEUE_CONFIG:-configs/qrl_queue.env}"
if [[ -f "${CONFIG}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${CONFIG}"
  set +a
fi

HOST="${QRL_RESULTS_REMOTE_HOST:-${REMOTE_RESULTS_SYNC_HOST:-}}"
PORT="${QRL_RESULTS_REMOTE_PORT:-${REMOTE_RESULTS_SYNC_PORT:-22}}"
USER_NAME="${QRL_RESULTS_REMOTE_USER:-${REMOTE_RESULTS_SYNC_USER:-}}"
REMOTE_ROOT="${QRL_RESULTS_REMOTE_ROOT:-${REMOTE_RESULTS_SYNC_ROOT:-}}"
KEY="${QRL_RESULTS_REMOTE_KEY:-${REMOTE_RESULTS_SYNC_KEY:-}}"
KNOWN_HOSTS="${QRL_RESULTS_REMOTE_KNOWN_HOSTS:-${REMOTE_RESULTS_SYNC_KNOWN_HOSTS:-}}"

for setting in HOST USER_NAME REMOTE_ROOT KEY KNOWN_HOSTS; do
  if [[ -z "${!setting}" ]]; then
    echo "Missing remote result setting: ${setting}" >&2
    echo "Configure REMOTE_RESULTS_SYNC_* in ${CONFIG} or use QRL_RESULTS_REMOTE_* overrides." >&2
    exit 2
  fi
done
if [[ "${REMOTE_ROOT}" != /* ]]; then
  echo "REMOTE_ROOT must be an absolute path: ${REMOTE_ROOT}" >&2
  exit 2
fi
if [[ ! -f "${KEY}" || ! -f "${KEY}.pub" ]]; then
  echo "Missing dedicated SSH key pair: ${KEY} and ${KEY}.pub" >&2
  exit 2
fi

chmod 600 "${KEY}"
mkdir -p "$(dirname "${KNOWN_HOSTS}")"
touch "${KNOWN_HOSTS}"
chmod 600 "${KNOWN_HOSTS}"

ssh_args=(
  -p "${PORT}"
  -i "${KEY}"
  -o IdentitiesOnly=yes
  -o BatchMode=yes
  -o ConnectTimeout=10
  -o StrictHostKeyChecking=accept-new
  -o "UserKnownHostsFile=${KNOWN_HOSTS}"
)

ssh "${ssh_args[@]}" "${USER_NAME}@${HOST}" bash -s -- "${REMOTE_ROOT}" <<'REMOTE'
set -euo pipefail
target="$1"
if [[ ! -d "${target}" ]]; then
  echo "Remote result root does not exist: ${target}" >&2
  exit 2
fi
if [[ ! -w "${target}" ]]; then
  echo "Remote result root is not writable: ${target}" >&2
  exit 2
fi
if ! command -v rsync >/dev/null 2>&1; then
  echo "rsync is not installed on the remote host" >&2
  exit 2
fi
probe="${target}/.qrl_transfer_probe_$$"
trap 'rm -f "${probe}"' EXIT
: >"${probe}"
rm -f "${probe}"
trap - EXIT
printf 'REMOTE_HOST=%s\n' "$(hostname)"
printf 'REMOTE_ROOT=%s\n' "${target}"
printf 'REMOTE_WRITABLE=yes\n'
printf 'REMOTE_RSYNC=%s\n' "$(command -v rsync)"
df -h "${target}" | tail -n 1
REMOTE

echo "SSH and remote QRL result storage verification passed."

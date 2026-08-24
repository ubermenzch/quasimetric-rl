#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export QRL_RESULTS_REMOTE_HOST="${QRL_RESULTS_REMOTE_HOST:-10.82.1.225}"
export QRL_RESULTS_REMOTE_PORT="${QRL_RESULTS_REMOTE_PORT:-8899}"
export QRL_RESULTS_REMOTE_USER="${QRL_RESULTS_REMOTE_USER:-zhangcheng}"
export QRL_RESULTS_REMOTE_ROOT="${QRL_RESULTS_REMOTE_ROOT:-/data2/zhangcheng/qrl-assets/results/queue}"
export QRL_RESULTS_REMOTE_KEY="${QRL_RESULTS_REMOTE_KEY:-runs/qrl_queue/ssh/id_ed25519_qrl_results_225}"
export QRL_RESULTS_REMOTE_KNOWN_HOSTS="${QRL_RESULTS_REMOTE_KNOWN_HOSTS:-runs/qrl_queue/ssh/known_hosts_225}"

exec "${ROOT_DIR}/tools/verify_qrl_remote_results.sh"

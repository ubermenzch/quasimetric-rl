#!/bin/sh
# Support both `sh tools/setup_baseline_paper_dependencies.sh` and Bash.
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi
set -Eeuo pipefail

QRL_SETUP_DRY_RUN=0
QRL_SETUP_CHECKS=1
QRL_SETUP_CHECKS_ONLY=0
while (($#)); do
    case "$1" in
        --dry-run) QRL_SETUP_DRY_RUN=1 ;;
        --skip-checks) QRL_SETUP_CHECKS=0 ;;
        --checks-only|--from-data) QRL_SETUP_CHECKS_ONLY=1 ;;
        -h|--help)
            printf '%s\n' \
                'Usage: sh tools/setup_baseline_paper_dependencies.sh [--dry-run] [--skip-checks] [--checks-only]' \
                'Install missing paper-validation dependencies in the existing .venv, without root.' \
                'Fetch isolated CRL MetaWorld. C-Learning Pen is skipped by user request.' \
                'No training jobs are started. Task preflight is a separate command.' \
                '--dry-run prints actions without installing, downloading or writing files.' \
                '--checks-only runs simulator checks without package installation or downloads.' \
                '--from-data is retained as an alias for --checks-only.' \
                '--skip-checks skips simulator smoke checks after installation.'
            exit 0 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
    shift
done

QRL_SETUP_ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$QRL_SETUP_ROOT"
source tools/qrl_env.sh
QRL_SETUP_PYTHON="$QRL_SETUP_ROOT/.venv/bin/python"
QRL_SETUP_REFERENCE="${QRL_PAPER_REFERENCE_ROOT:-/home/zhangcheng/gcrl-reference-repos}"
QRL_SETUP_META="$QRL_SETUP_REFERENCE/metaworld-crl"
QRL_SETUP_META_REV=a0009ed9a208ff9864a5c1368c04c273bb20dd06
QRL_SETUP_LOG_DIR="$QRL_SETUP_ROOT/logs/baseline_paper_dependencies"
QRL_SETUP_CONSTRAINTS="$QRL_SETUP_LOG_DIR/constraints-$(date +%Y%m%d-%H%M%S)-$$.txt"
export PATH="$QRL_SETUP_ROOT/.venv/bin:$PATH"

run() {
    if ((QRL_SETUP_DRY_RUN)); then
        printf '[dry-run]'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

[[ -x "$QRL_SETUP_PYTHON" ]] || die "Missing repository Python: $QRL_SETUP_PYTHON"
QRL_SETUP_TOOLS=()
if ((!QRL_SETUP_CHECKS_ONLY)); then
    QRL_SETUP_TOOLS+=(git gcc g++ make)
fi
for QRL_SETUP_TOOL in "${QRL_SETUP_TOOLS[@]}"; do
    command -v "$QRL_SETUP_TOOL" >/dev/null || die "Missing $QRL_SETUP_TOOL; this script uses this machine's existing compiler tools."
done

if ((!QRL_SETUP_DRY_RUN)); then
    mkdir -p "$QRL_SETUP_LOG_DIR"
    QRL_SETUP_LOG="$QRL_SETUP_LOG_DIR/setup-$(date +%Y%m%d-%H%M%S)-$$.log"
    exec > >(tee -a "$QRL_SETUP_LOG") 2>&1
    trap 'printf "Setup stopped at line %s (status %s). See %s\n" "$LINENO" "$?" "$QRL_SETUP_LOG" >&2' ERR
    printf 'Log: %s\n' "$QRL_SETUP_LOG"
    if ((!QRL_SETUP_CHECKS_ONLY)); then
        "$QRL_SETUP_PYTHON" -m pip list --format=freeze --disable-pip-version-check > "$QRL_SETUP_CONSTRAINTS"
    fi
elif ((!QRL_SETUP_CHECKS_ONLY)); then
    printf '[dry-run] Capture existing package versions in %s\n' "$QRL_SETUP_CONSTRAINTS"
fi

if ((!QRL_SETUP_CHECKS_ONLY)); then
printf '\n[1/5] Install missing build and simulator packages in .venv\n'
# SWIG ships a Linux binary wheel, so installing it needs no system package manager.
if ! command -v swig >/dev/null; then
    run "$QRL_SETUP_PYTHON" -m pip install --only-binary=:all: \
        -c "$QRL_SETUP_CONSTRAINTS" swig==4.3.1
fi
run "$QRL_SETUP_PYTHON" -m pip install --no-build-isolation \
    -c "$QRL_SETUP_CONSTRAINTS" \
    box2d-py==2.3.5 transforms3d==0.4.2
# ROBEL's release metadata pins Gym <0.14 and mujoco-py <2.1. Its remaining
# runtime requirements are NumPy and transforms3d, supplied explicitly here.
# Keep the repository's simulator versions; the DClaw smoke check below must
# establish runtime compatibility independently of this metadata exception.
run "$QRL_SETUP_PYTHON" -c \
    'from importlib.metadata import version; print("ROBEL runtime:", {name: version(name) for name in ("gym", "mujoco-py", "numpy", "transforms3d")})'
printf 'Installing ROBEL with an explicit legacy-metadata exception; keeping existing Gym and mujoco-py.\n'
run "$QRL_SETUP_PYTHON" -m pip install --no-deps \
    -c "$QRL_SETUP_CONSTRAINTS" robel==0.1.2

printf '\n[2/5] Install the pinned mjrl used by the original CRL requirements\n'
# This mjrl revision imports six but omits it from its package metadata.
run "$QRL_SETUP_PYTHON" -m pip install --only-binary=:all: \
    -c "$QRL_SETUP_CONSTRAINTS" six==1.16.0
run "$QRL_SETUP_PYTHON" -m pip install --no-build-isolation \
    -c "$QRL_SETUP_CONSTRAINTS" \
    'mjrl @ git+https://github.com/aravindr93/mjrl.git@3871d93763d3b49c4741e6daeaebbc605fe140dc'

printf '\n[3/5] Fetch CRL MetaWorld into its own reference directory\n'
run mkdir -p "$QRL_SETUP_REFERENCE"
if [[ -d "$QRL_SETUP_META/.git" ]]; then
    [[ -z "$(git -C "$QRL_SETUP_META" status --porcelain)" ]] || die "Tracked or untracked changes in $QRL_SETUP_META; refusing to replace them."
    if ! git -C "$QRL_SETUP_META" cat-file -e "$QRL_SETUP_META_REV^{commit}" 2>/dev/null; then
        run git -C "$QRL_SETUP_META" fetch origin "$QRL_SETUP_META_REV"
    fi
elif [[ -e "$QRL_SETUP_META" ]]; then
    die "$QRL_SETUP_META already exists and is not the expected Git checkout."
else
    run git clone https://github.com/Farama-Foundation/Metaworld.git "$QRL_SETUP_META"
fi
run git -C "$QRL_SETUP_META" checkout --detach "$QRL_SETUP_META_REV"
printf 'MetaWorld remains source-only; the C-Learning checkout is preserved.\n'
else
    printf 'Resuming at step 4; package installation and MetaWorld are skipped.\n'
fi

printf '\n[4/5] SKIPPED: C-Learning Pen validation (user request; original data unavailable)\n'

printf '\n[5/5] Check the base simulator dependencies\n'
if ((QRL_SETUP_CHECKS)); then
    run "$QRL_SETUP_PYTHON" - <<'PY'
import subprocess
import sys

checks = {
    'Box2D': 'import Box2D; print(Box2D.__version__)',
    'ROBEL DClaw': "import gym, robel; e=gym.make('DClawTurnFixed-v0'); e.reset(); e.step(e.action_space.sample()); e.close()",
    # Import mjrl first: D4RL suppresses its ImportError and leaves Gym IDs
    # unregistered, hiding the missing dependency behind UnregisteredEnv.
    'D4RL Door base': '''
from mjrl.envs.mujoco_env import MujocoEnv
import d4rl.hand_manipulation_suite
import gym
import numpy as np

env = gym.make('door-human-v0')
try:
    observation = env.reset()
    assert np.isfinite(observation).all()
    observation, reward, done, info = env.step(env.action_space.sample())
    assert np.isfinite(observation).all() and np.isfinite(reward)
finally:
    env.close()
''',
    'DMC imports': 'from dm_control import suite, manipulation',
}
failed = []
for name, program in checks.items():
    print('Checking ' + name, flush=True)
    try:
        result = subprocess.run([sys.executable, '-c', program], timeout=180)
        if result.returncode != 0:
            failed.append(name)
    except subprocess.TimeoutExpired:
        failed.append(name + ' (timeout)')
    print(name + (': FAIL' if any(item.startswith(name) for item in failed) else ': PASS'), flush=True)
if failed:
    print('Downloads/install steps finished; runtime compatibility still needs fixes: ' + ', '.join(failed))
    sys.exit(1)
PY
else
    printf 'Simulator smoke checks skipped.\n'
fi

printf '\nDependency preparation finished. No training tasks were launched.\n'
printf '%s\n' \
    'Task adapters are available; use tools/check_baseline_paper_validation.py --suite paper-main to validate them.' \
    'C-Learning Jaco/Finger/Hand use documented Appendix H.3 reconstructions.' \
    'C-Learning Pen is excluded from the current validation plan; no Pen data is downloaded.' \
    'See configs/baseline_paper_validation_protocols.json for protocol details and limitations.'

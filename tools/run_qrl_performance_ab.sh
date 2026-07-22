#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: tools/run_qrl_performance_ab.sh [options]

Options:
  --gpu INDEX          Physical GPU index passed through CUDA_VISIBLE_DEVICES (default: 0)
  --batch-size N       Training batch size (default: 4096)
  --warmup-steps N     Warmup training steps per process (default: 10)
  --measure-steps N    Measured training steps per process (default: 30)
  --repeats N          A/B repetitions per variant (default: 3)
  --results-dir PATH   Output directory (default: runs/qrl_perf/<timestamp>)
  --baseline-dir PATH  Detached HEAD worktree (default: /tmp/qrl-perf-baseline-<commit>)
  -h, --help           Show this help

The selected GPU must be idle. Baseline and optimized runs execute sequentially.
EOF
}

gpu=0
batch_size=4096
warmup_steps=10
measure_steps=30
repeats=3
results_dir=''
baseline_dir=''

while (($#)); do
  case "$1" in
    --gpu)
      gpu="$2"
      shift 2
      ;;
    --batch-size)
      batch_size="$2"
      shift 2
      ;;
    --warmup-steps)
      warmup_steps="$2"
      shift 2
      ;;
    --measure-steps)
      measure_steps="$2"
      shift 2
      ;;
    --repeats)
      repeats="$2"
      shift 2
      ;;
    --results-dir)
      results_dir="$2"
      shift 2
      ;;
    --baseline-dir)
      baseline_dir="$2"
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

for value_name in batch_size warmup_steps measure_steps repeats; do
  value="${!value_name}"
  if ! [[ "$value" =~ ^[0-9]+$ ]]; then
    echo "$value_name must be an integer, got: $value" >&2
    exit 2
  fi
done
if ((batch_size == 0 || measure_steps == 0 || repeats == 0)); then
  echo 'batch-size, measure-steps, and repeats must be positive' >&2
  exit 2
fi
if ! [[ "$gpu" =~ ^[0-9]+$ ]]; then
  echo "gpu must be a non-negative integer, got: $gpu" >&2
  exit 2
fi

repo="$(git rev-parse --show-toplevel)"
python_bin="$repo/.venv/bin/python"
benchmark="$repo/tools/benchmark_qrl_compute.py"
baseline_commit="$(git -C "$repo" rev-parse HEAD)"
baseline_short="$(git -C "$repo" rev-parse --short HEAD)"

if [[ -z "$baseline_dir" ]]; then
  baseline_dir="/tmp/qrl-perf-baseline-$baseline_short"
fi
if [[ -z "$results_dir" ]]; then
  results_dir="$repo/runs/qrl_perf/$(date +%Y%m%d-%H%M%S)"
fi

if [[ ! -x "$python_bin" ]]; then
  echo "Python virtual environment not found: $python_bin" >&2
  exit 1
fi
if [[ ! -f "$benchmark" ]]; then
  echo "Benchmark driver not found: $benchmark" >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo 'nvidia-smi is required for the GPU preflight check' >&2
  exit 1
fi

if [[ -e "$baseline_dir/.git" ]]; then
  existing_commit="$(git -C "$baseline_dir" rev-parse HEAD)"
  if [[ "$existing_commit" != "$baseline_commit" ]]; then
    echo "Existing baseline worktree is at $existing_commit, expected $baseline_commit" >&2
    echo "Choose another path with --baseline-dir." >&2
    exit 1
  fi
elif [[ -e "$baseline_dir" ]]; then
  echo "Baseline path exists but is not a git worktree: $baseline_dir" >&2
  exit 1
else
  git -C "$repo" worktree add --detach "$baseline_dir" "$baseline_commit"
fi

mkdir -p "$results_dir"
mkdir -p "$repo/runs/qrl_perf"
printf '%s\n' "$results_dir" > "$repo/runs/qrl_perf/LATEST"

{
  echo "started_at=$(date --iso-8601=seconds)"
  echo "hostname=$(hostname)"
  echo "repo=$repo"
  echo "baseline_dir=$baseline_dir"
  echo "baseline_commit=$baseline_commit"
  echo "gpu=$gpu"
  echo "batch_size=$batch_size"
  echo "warmup_steps=$warmup_steps"
  echo "measure_steps=$measure_steps"
  echo "repeats=$repeats"
  nvidia-smi -i "$gpu" --query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu \
    --format=csv,noheader
} | tee "$results_dir/metadata.txt"

run_one() {
  local implementation="$1"
  local source_root="$2"
  local variant="$3"
  local repetition="$4"
  local stem="${implementation}_${variant}_r${repetition}"

  echo "RUN implementation=$implementation variant=$variant repetition=$repetition"
  (
    cd "$source_root"
    CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONPATH="$source_root" \
      "$python_bin" "$benchmark" \
        --variant "$variant" \
        --device cuda:0 \
        --batch-size "$batch_size" \
        --warmup-steps "$warmup_steps" \
        --measure-steps "$measure_steps"
  ) \
    2> >(tee "$results_dir/${stem}.stderr.log" >&2) \
    | tee "$results_dir/${stem}.json"
}

variants=(1q_base 2q_base split_none split_max8)
for ((repetition = 1; repetition <= repeats; repetition++)); do
  if ((repetition % 2 == 1)); then
    implementations=(baseline optimized)
  else
    implementations=(optimized baseline)
  fi

  for implementation in "${implementations[@]}"; do
    if [[ "$implementation" == baseline ]]; then
      source_root="$baseline_dir"
    else
      source_root="$repo"
    fi
    for variant in "${variants[@]}"; do
      run_one "$implementation" "$source_root" "$variant" "$repetition"
    done
  done
done

PERF_RESULTS_DIR="$results_dir" "$python_bin" - <<'PY' | tee "$results_dir/summary.txt"
import json
import os
import statistics
from pathlib import Path

root = Path(os.environ['PERF_RESULTS_DIR'])
rows = []
for path in root.glob('*_r*.json'):
    text = path.read_text().strip().splitlines()
    if not text:
        continue
    row = json.loads(text[-1])
    row['implementation'] = (
        'baseline' if path.name.startswith('baseline_') else 'optimized'
    )
    rows.append(row)

variants = ('1q_base', '2q_base', 'split_none', 'split_max8')
print(
    f"{'variant':<14} {'base ms':>10} {'opt ms':>10} "
    f"{'speedup':>9} {'gain':>9} {'memory delta':>14}"
)
for variant in variants:
    baseline = [
        row for row in rows
        if row['variant'] == variant and row['implementation'] == 'baseline'
    ]
    optimized = [
        row for row in rows
        if row['variant'] == variant and row['implementation'] == 'optimized'
    ]
    if not baseline or not optimized:
        print(f'{variant:<14} incomplete')
        continue
    base_ms = statistics.median(row['ms_per_step'] for row in baseline)
    opt_ms = statistics.median(row['ms_per_step'] for row in optimized)
    base_mem = statistics.median(row['peak_allocated_mb'] for row in baseline)
    opt_mem = statistics.median(row['peak_allocated_mb'] for row in optimized)
    print(
        f'{variant:<14} {base_ms:10.2f} {opt_ms:10.2f} '
        f'{base_ms / opt_ms:8.3f}x {(base_ms - opt_ms) / base_ms:8.2%} '
        f'{opt_mem - base_mem:+11.1f} MiB'
    )
PY

echo "finished_at=$(date --iso-8601=seconds)" | tee -a "$results_dir/metadata.txt"
echo "Results: $results_dir"

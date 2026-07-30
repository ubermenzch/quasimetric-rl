#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="$ROOT/tools/run_splitlatent_best10_eval_2000_2999.sh"
LOG_DIR="$ROOT/logs/offline_eval"
RUN_PREFIX="1q_splitlatent_best10_eval2000_2999"
RESULT_PREFIX="official_qrl_1q_SplitLatentMax8_PMBase1q_bc0_100k_maze2d_umaze_offlinegoals"

seeds=(1000 1001 1002 1003 1004 1005 1006 1007 1008 1009)
steps=(80032 80032 15128 80032 5124 20008 75152 5124 15128 90036)
gpus=(1 2 3 6 7)
cpu_sets=("0-15" "16-31" "32-47" "48-63" "64-79")

run_tag="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
out_dir="$ROOT/analysis/offline_eval/${RUN_PREFIX}_${run_tag}"
coordinator_log="$LOG_DIR/${RUN_PREFIX}_${run_tag}_coordinator.log"

checkpoint_path() {
    local seed="$1"
    local step="$2"
    local padded_step
    printf -v padded_step '%08d' "$step"
    printf '%s/../qrl-assets/results/queue/%s_s%s/agent_checkpoint_step%s.pth' \
        "$ROOT" "$RESULT_PREFIX" "$seed" "$padded_step"
}

validate_inputs() {
    local idx checkpoint

    command -v taskset >/dev/null || {
        echo "taskset is required but was not found" >&2
        return 1
    }
    command -v flock >/dev/null || {
        echo "flock is required but was not found" >&2
        return 1
    }
    [[ -x "$ROOT/.venv/bin/python" ]] || {
        echo "Missing Python environment: $ROOT/.venv/bin/python" >&2
        return 1
    }
    [[ -f "$ROOT/tools/evaluate_offline_maze2d.py" ]] || {
        echo "Missing evaluator: $ROOT/tools/evaluate_offline_maze2d.py" >&2
        return 1
    }

    for idx in "${!seeds[@]}"; do
        checkpoint="$(checkpoint_path "${seeds[$idx]}" "${steps[$idx]}")"
        [[ -f "$checkpoint" ]] || {
            echo "Missing checkpoint: $checkpoint" >&2
            return 1
        }
    done

    for idx in "${!cpu_sets[@]}"; do
        taskset -c "${cpu_sets[$idx]}" true 2>/dev/null || {
            echo "CPU set ${cpu_sets[$idx]} is unavailable" >&2
            return 1
        }
    done
}

print_assignments() {
    local worker idx
    for worker in "${!gpus[@]}"; do
        for idx in "$worker" "$((worker + 5))"; do
            printf 'GPU=%s CPUs=%s train_seed=%s checkpoint_step=%s eval_seeds=2000-2999\n' \
                "${gpus[$worker]}" "${cpu_sets[$worker]}" \
                "${seeds[$idx]}" "${steps[$idx]}"
        done
    done
}

show_status() {
    local latest tag pid state summary_count gpu log last_progress completed failed

    latest="$(find "$ROOT/analysis/offline_eval" -maxdepth 1 -type d \
        -name "${RUN_PREFIX}_*" -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr | head -n 1 | cut -d' ' -f2-)"
    if [[ -z "$latest" ]]; then
        echo "No ${RUN_PREFIX} run directory found"
        return 1
    fi

    tag="${latest##*/${RUN_PREFIX}_}"
    pid=""
    [[ -f "$latest/coordinator.pid" ]] && pid="$(<"$latest/coordinator.pid")"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        state="RUNNING"
    else
        state="NOT RUNNING"
    fi
    summary_count="$(find "$latest" -maxdepth 1 -type f -name '*_summary.tsv' | wc -l)"

    printf 'run: %s\nstate: %s\ncoordinator_pid: %s\nsummaries: %s/10\n' \
        "$tag" "$state" "${pid:-unknown}" "$summary_count"

    for gpu in "${gpus[@]}"; do
        log="$LOG_DIR/${RUN_PREFIX}_${tag}_gpu${gpu}.log"
        if [[ ! -f "$log" ]]; then
            printf 'gpu%s: log not created\n' "$gpu"
            continue
        fi
        completed="$(grep -ac '^COMPLETED train_seed=' "$log" || true)"
        failed="$(grep -ac '^FAILED train_seed=' "$log" || true)"
        last_progress="$(tr '\r' '\n' < "$log" | grep -a 'eval official' | tail -n 1 || true)"
        printf 'gpu%s: completed=%s/2 failed=%s' "$gpu" "$completed" "$failed"
        [[ -n "$last_progress" ]] && printf ' | %s' "$last_progress"
        printf '\n'
    done

    if [[ -f "$LOG_DIR/${RUN_PREFIX}_${tag}_coordinator.log" ]]; then
        echo "coordinator tail:"
        tail -n 5 "$LOG_DIR/${RUN_PREFIX}_${tag}_coordinator.log"
    fi
}

run_foreground() {
    local worker gpu cpu_set worker_log idx seed step result_dir prefix
    local worker_status status summary_count
    local -a worker_pids=()

    cd "$ROOT"
    validate_inputs
    mkdir -p "$out_dir" "$LOG_DIR"

    exec 9>"$LOG_DIR/.${RUN_PREFIX}.lock"
    if ! flock -n 9; then
        echo "Another ${RUN_PREFIX} coordinator is already running" >&2
        return 1
    fi

    : > "$out_dir/worker.pids"
    print_assignments

    for worker in "${!gpus[@]}"; do
        gpu="${gpus[$worker]}"
        cpu_set="${cpu_sets[$worker]}"
        worker_log="$LOG_DIR/${RUN_PREFIX}_${run_tag}_gpu${gpu}.log"

        (
            worker_status=0
            for idx in "$worker" "$((worker + 5))"; do
                seed="${seeds[$idx]}"
                step="${steps[$idx]}"
                result_dir="$ROOT/../qrl-assets/results/queue/${RESULT_PREFIX}_s${seed}"
                prefix="train_s${seed}_best_return_step${step}_eval_s2000_s2999"

                printf 'STARTING train_seed=%s checkpoint_step=%s GPU=%s CPUs=%s\n' \
                    "$seed" "$step" "$gpu" "$cpu_set"
                if taskset -c "$cpu_set" "$ROOT/.venv/bin/python" \
                    "$ROOT/tools/evaluate_offline_maze2d.py" \
                    "$result_dir" \
                    --checkpoint "$step" \
                    --num-episodes 1000 \
                    --seed 2000 \
                    --device "cuda:$gpu" \
                    --num-envs 16 \
                    --num-workers 8 \
                    --out-dir "$out_dir" \
                    --prefix "$prefix"; then
                    printf 'COMPLETED train_seed=%s checkpoint_step=%s\n' "$seed" "$step"
                else
                    printf 'FAILED train_seed=%s checkpoint_step=%s\n' "$seed" "$step"
                    worker_status=1
                fi
            done
            exit "$worker_status"
        ) > "$worker_log" 2>&1 &

        worker_pids+=("$!")
        printf '%s\tGPU=%s\tCPUs=%s\tlog=%s\n' \
            "$!" "$gpu" "$cpu_set" "$worker_log" >> "$out_dir/worker.pids"
        printf 'Started GPU %s worker PID %s\n' "$gpu" "$!"
    done

    status=0
    for idx in "${!worker_pids[@]}"; do
        if ! wait "${worker_pids[$idx]}"; then
            echo "GPU ${gpus[$idx]} worker failed"
            status=1
        fi
    done

    summary_count="$(find "$out_dir" -maxdepth 1 -type f -name '*_summary.tsv' | wc -l)"
    echo "Generated $summary_count/10 summary TSV files"
    if [[ "$summary_count" -ne 10 ]]; then
        status=1
    fi
    if [[ "$status" -eq 0 ]]; then
        echo "ALL EVALUATIONS COMPLETED"
    else
        echo "EVALUATION FINISHED WITH FAILURES"
    fi
    return "$status"
}

case "${1:-start}" in
    start)
        validate_inputs
        mkdir -p "$out_dir" "$LOG_DIR"
        nohup env RUN_TAG="$run_tag" bash "$SCRIPT_PATH" --foreground \
            > "$coordinator_log" 2>&1 < /dev/null &
        coordinator_pid="$!"
        printf '%s\n' "$coordinator_pid" > "$out_dir/coordinator.pid"
        printf 'Started coordinator PID: %s\nOutput: %s\nLog: %s\n' \
            "$coordinator_pid" "$out_dir" "$coordinator_log"
        ;;
    --foreground)
        run_foreground
        ;;
    --dry-run)
        validate_inputs
        print_assignments
        ;;
    --status)
        show_status
        ;;
    *)
        echo "Usage: bash $SCRIPT_PATH [start|--dry-run|--status]" >&2
        exit 2
        ;;
esac

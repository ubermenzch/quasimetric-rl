#!/usr/bin/env bash

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="$ROOT/tools/run_boundedres_missing_eval_2000_2999.sh"
RUN_DIR="$ROOT/analysis/offline_eval/boundedres_all_agent_eval2000_2999_20260722_160453"
LOG_DIR="$ROOT/logs/offline_eval"
LOCK_PATH="$LOG_DIR/.boundedres_missing_steps_eval2000_2999.lock"
PID_PATH="$RUN_DIR/boundedres_missing_steps_eval.pid"
LATEST_LOG_PATH="$RUN_DIR/boundedres_missing_steps_eval.latest_log"
RESULT_PREFIX="official_qrl_1q_SplitLatentMax8_BoundedResR1_PMBase1q_bc0_100k_maze2d_umaze_offlinegoals"

training_seeds=(1000 1001 1002 1003 1004 1005 1006 1007 1008 1009)
missing_steps=(30012 35136 40016 45140 50020 55144 60024 65148 70028 75152)
all_steps=(
    5124 10004 15128 20008 25132
    30012 35136 40016 45140 50020
    55144 60024 65148 70028 75152
    80032 85156 90036 95160 100040
)

result_dir_template="$ROOT/../qrl-assets/results/queue/${RESULT_PREFIX}_s{seed}"

summary_path() {
    local step="$1"
    local padded_step
    printf -v padded_step '%08d' "$step"
    printf '%s/boundedres_agent_step%s_eval_s2000_s2999_summary.tsv' \
        "$RUN_DIR" "$padded_step"
}

summary_is_complete() {
    local step="$1"
    local path
    path="$(summary_path "$step")"
    [[ -f "$path" ]] || return 1

    awk -F '\t' -v expected_step="$step" '
        NR > 1 {
            rows++
            seeds[$4] = 1
            if (($6 + 0) != expected_step) bad = 1
        }
        END {
            for (seed in seeds) seed_count++
            exit !(rows == 10 && seed_count == 10 && !bad)
        }
    ' "$path"
}

validate_inputs() {
    local seed step padded_step checkpoint

    [[ -x "$ROOT/.venv/bin/python" ]] || {
        echo "Missing Python environment: $ROOT/.venv/bin/python" >&2
        return 1
    }
    [[ -f "$ROOT/tools/evaluate_offline_maze2d.py" ]] || {
        echo "Missing evaluator: $ROOT/tools/evaluate_offline_maze2d.py" >&2
        return 1
    }
    command -v flock >/dev/null || {
        echo "flock is required but was not found" >&2
        return 1
    }

    for seed in "${training_seeds[@]}"; do
        for step in "${missing_steps[@]}"; do
            printf -v padded_step '%08d' "$step"
            checkpoint="$ROOT/../qrl-assets/results/queue/${RESULT_PREFIX}_s${seed}/agent_checkpoint_step${padded_step}.pth"
            [[ -f "$checkpoint" ]] || {
                echo "Missing checkpoint: $checkpoint" >&2
                return 1
            }
        done
    done
}

merge_and_validate() {
    local step path combined combined_tmp rows
    local -a summary_files=()

    for step in "${all_steps[@]}"; do
        if ! summary_is_complete "$step"; then
            echo "Incomplete or missing summary for checkpoint step $step" >&2
            return 1
        fi
        path="$(summary_path "$step")"
        summary_files+=("$path")
    done

    combined="$RUN_DIR/boundedres_all_agent_checkpoints_eval_s2000_s2999_summary.tsv"
    combined_tmp="$RUN_DIR/.boundedres_all_agent_checkpoints_summary.tmp"
    awk '
        FNR == 1 {
            if (header_seen++) next
        }
        { print }
    ' "${summary_files[@]}" > "$combined_tmp"

    rows="$(awk 'END {print NR - 1}' "$combined_tmp")"
    if [[ "$rows" -ne 200 ]]; then
        echo "Expected 200 model rows, found $rows" >&2
        rm -f "$combined_tmp"
        return 1
    fi

    mv "$combined_tmp" "$combined"
    echo "COMPLETED: checkpoint summaries=20, model rows=$rows"
    echo "Combined summary: $combined"
}

run_foreground() {
    local step padded_step prefix status

    validate_inputs
    mkdir -p "$RUN_DIR" "$LOG_DIR"

    exec 9>"$LOCK_PATH"
    if ! flock -n 9; then
        echo "Another missing-checkpoint evaluation is already running" >&2
        return 1
    fi

    status=0
    for step in "${missing_steps[@]}"; do
        if summary_is_complete "$step"; then
            echo "SKIP checkpoint step $step: complete summary already exists"
            continue
        fi

        printf -v padded_step '%08d' "$step"
        prefix="boundedres_agent_step${padded_step}_eval_s2000_s2999"
        echo "START checkpoint step $step"

        if "$ROOT/.venv/bin/python" "$ROOT/tools/evaluate_offline_maze2d.py" \
            "$result_dir_template" \
            --training-seeds "1000,1001,1002,1003,1004,1005,1006,1007,1008,1009" \
            --checkpoint "$step" \
            --num-episodes 1000 \
            --seed 2000 \
            --gpus "0,1,2,3,4,5,6,7" \
            --num-envs 16 \
            --num-workers 8 \
            --out-dir "$RUN_DIR" \
            --prefix "$prefix"; then
            if summary_is_complete "$step"; then
                echo "DONE checkpoint step $step"
            else
                echo "FAILED checkpoint step $step: summary validation failed"
                status=1
            fi
        else
            echo "FAILED checkpoint step $step: evaluator returned nonzero"
            status=1
        fi
    done

    if [[ "$status" -ne 0 ]]; then
        echo "Evaluation finished with failures; rerun the same start command to resume"
        return 1
    fi

    merge_and_validate
}

show_status() {
    local pid latest_log state completed step

    pid=""
    latest_log=""
    [[ -f "$PID_PATH" ]] && pid="$(<"$PID_PATH")"
    [[ -f "$LATEST_LOG_PATH" ]] && latest_log="$(<"$LATEST_LOG_PATH")"

    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        state="RUNNING"
    else
        state="NOT RUNNING"
    fi

    completed=0
    for step in "${missing_steps[@]}"; do
        summary_is_complete "$step" && completed=$((completed + 1))
    done

    printf 'state: %s\npid: %s\nmissing-step summaries: %d/10\nlog: %s\n' \
        "$state" "${pid:-unknown}" "$completed" "${latest_log:-unknown}"
    if [[ -n "$latest_log" && -f "$latest_log" ]]; then
        echo "log tail:"
        tail -n 8 "$latest_log"
    fi
}

case "${1:-start}" in
    start)
        validate_inputs
        mkdir -p "$RUN_DIR" "$LOG_DIR"
        run_tag="$(date +%Y%m%d_%H%M%S)"
        coordinator_log="$LOG_DIR/boundedres_missing_steps_eval2000_2999_${run_tag}.log"
        nohup bash "$SCRIPT_PATH" --foreground > "$coordinator_log" 2>&1 < /dev/null &
        coordinator_pid="$!"
        printf '%s\n' "$coordinator_pid" > "$PID_PATH"
        printf '%s\n' "$coordinator_log" > "$LATEST_LOG_PATH"
        printf 'Started PID: %s\nLog: %s\n' "$coordinator_pid" "$coordinator_log"
        ;;
    --foreground)
        run_foreground
        ;;
    --status)
        show_status
        ;;
    --dry-run)
        validate_inputs
        printf 'Validated %d checkpoints; steps:' "$((${#training_seeds[@]} * ${#missing_steps[@]}))"
        printf ' %s' "${missing_steps[@]}"
        printf '\nGPUs: 0,1,2,3,4,5,6,7\nEpisodes per model: 1000 (seeds 2000-2999)\n'
        ;;
    *)
        echo "Usage: bash $SCRIPT_PATH [start|--foreground|--status|--dry-run]" >&2
        exit 2
        ;;
esac

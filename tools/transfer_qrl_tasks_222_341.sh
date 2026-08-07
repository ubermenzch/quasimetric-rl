#!/usr/bin/env bash

set -Eeuo pipefail

ROOT=/home/zhangcheng/qrl-assets/results/queue
KEY=/home/zhangcheng/.ssh/id_ed25519_qrl_transfer
REMOTE=zc@10.130.136.132
PORT=8899
DEST=/data/raid5/zhangcheng/quasimetric-rl/IQE-MSE-Leaky-Results

dry_run=false
if [[ ${1:-} == "--dry-run" ]]; then
    dry_run=true
elif [[ $# -ne 0 ]]; then
    echo "Usage: $0 [--dry-run]" >&2
    exit 2
fi

for command in find rsync ssh sort; do
    command -v "$command" >/dev/null || {
        echo "ERROR: required command not found: $command" >&2
        exit 2
    }
done

[[ -d $ROOT ]] || {
    echo "ERROR: result root not found: $ROOT" >&2
    exit 2
}
[[ -f $KEY ]] || {
    echo "ERROR: SSH private key not found: $KEY" >&2
    exit 2
}

ids=$(mktemp /tmp/qrl_tasks_222_341.XXXXXX.ids)
trap 'rm -f "$ids"' EXIT

while IFS= read -r directory; do
    name=${directory##*/}

    if [[ $name =~ ^ablation_GO-QRL\+Max4-dynfac_A0[2-6]-.*_(fetchpush|fetchslide|fetchpickandplace|reacher_hard|pusher_v4|antnavigate_v4|maze2d_large)_online_s100[0-2]$ ]] ||
       [[ $name =~ ^ablation_GO-QRL\+Max4-dynfac_A01-.*_(pusher_v4|antnavigate_v4|maze2d_large)_online_s100[0-2]$ ]] ||
       [[ $name =~ ^ablation_GO-QRL\+Max4-dynfac_(A07|A10)-.*_pusher_v4_online_s100[0-2]$ ]]; then
        printf '%s\n' "$name"
    fi
done < <(find "$ROOT" -mindepth 1 -maxdepth 1 -type d -print | sort) >"$ids"

selected=$(wc -l <"$ids")
complete=0
missing=0
while IFS= read -r id; do
    if [[ -f $ROOT/$id/COMPLETE ]]; then
        complete=$((complete + 1))
    else
        echo "MISSING/INCOMPLETE: $id" >&2
        missing=$((missing + 1))
    fi
done <"$ids"

echo "Selected tasks: $selected; complete tasks: $complete"
[[ $selected -eq 120 && $complete -eq 120 && $missing -eq 0 ]] || {
    echo "ERROR: expected exactly 120 complete tasks; transfer aborted" >&2
    exit 2
}

transfer_bytes=$(
    while IFS= read -r id; do
        find "$ROOT/$id" -type f ! -name '*.pth' -printf '%s\n'
    done <"$ids" | awk '{total += $1} END {printf "%.0f", total}'
)
awk -v bytes="$transfer_bytes" \
    'BEGIN {printf "Non-checkpoint payload: %.2f MiB\n", bytes / 1048576}'

ssh_command=(
    ssh
    -o IdentitiesOnly=yes
    -i "$KEY"
    -p "$PORT"
)
rsync_ssh="ssh -o IdentitiesOnly=yes -i $KEY -p $PORT"

echo "Checking destination: $REMOTE:$DEST"
"${ssh_command[@]}" "$REMOTE" \
    "mkdir -p '$DEST/_manifests' && test -w '$DEST' && df -h '$DEST'"

rsync_args=(
    -a
    --recursive
    --partial
    --append-verify
    --info=progress2
    --stats
    --files-from="$ids"
    "--exclude=*.pth"
    -e "$rsync_ssh"
)

if $dry_run; then
    echo "Dry run only; no result files will be written."
    rsync "${rsync_args[@]}" --dry-run --itemize-changes \
        "$ROOT/" "$REMOTE:$DEST/"
    exit 0
fi

echo "Uploading task manifest..."
rsync -a -e "$rsync_ssh" "$ids" \
    "$REMOTE:$DEST/_manifests/tasks222-341.ids"

echo "Transferring 120 task directories (all *.pth files excluded)..."
rsync "${rsync_args[@]}" "$ROOT/" "$REMOTE:$DEST/"

echo "Verifying destination..."
"${ssh_command[@]}" "$REMOTE" bash -s -- "$DEST" <<'REMOTE_CHECK'
set -Eeuo pipefail

dest=$1
ids=$dest/_manifests/tasks222-341.ids
directories=0
complete=0
eval_logs=0
test_logs=0
directories_with_pth=0

while IFS= read -r id; do
    [[ -d $dest/$id ]] && directories=$((directories + 1))
    [[ -f $dest/$id/COMPLETE ]] && complete=$((complete + 1))
    [[ -f $dest/$id/eval.log ]] && eval_logs=$((eval_logs + 1))
    [[ -f $dest/$id/test.log ]] && test_logs=$((test_logs + 1))

    if find "$dest/$id" -type f -name '*.pth' -print -quit | grep -q .; then
        directories_with_pth=$((directories_with_pth + 1))
    fi
done <"$ids"

printf 'dirs=%d COMPLETE=%d eval.log=%d test.log=%d dirs_with_pth=%d\n' \
    "$directories" "$complete" "$eval_logs" "$test_logs" \
    "$directories_with_pth"
du -sh "$dest"

[[ $directories -eq 120 ]]
[[ $complete -eq 120 ]]
[[ $eval_logs -eq 120 ]]
[[ $test_logs -eq 120 ]]
[[ $directories_with_pth -eq 0 ]]
REMOTE_CHECK

echo "Transfer and verification completed successfully."

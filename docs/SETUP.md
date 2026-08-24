# Portable Setup

This document describes the reproducible environment for the modified offline
Maze2D and AntMaze experiments. Source code, package pins, task files, and
dataset checksums live in Git. Virtual environments, MuJoCo, datasets, logs,
and checkpoints stay outside the repository.

The current D4RL experiment path requires Linux, Python 3.9, an NVIDIA driver
that can run a compatible PyTorch CUDA wheel, EGL/OpenGL libraries, and MuJoCo
2.1.0. It is portable across compatible NVIDIA CUDA hosts; it is not currently
a CPU-only, macOS, or non-NVIDIA setup because the legacy `mujoco-py` D4RL
stack depends on EGL and NVIDIA CUDA for the configured experiments.

## Clone The Repository

The upstream base commit is `0a611a41815c83103d92cdbff1283f99266a7360` from
`https://github.com/quasimetric-learning/quasimetric-rl.git`. The
`third_party/torch-quasimetric` submodule is required.

```bash
git clone --recurse-submodules <your-repository-url> qrl-official
cd qrl-official
git submodule update --init --recursive
```

All supplied paths are relative to this directory. By default, external assets
are stored in the sibling directory `../qrl-assets/`; set `QRL_ASSET_ROOT` only
when a different asset location is needed.

## Automated Setup

Confirm that the NVIDIA driver is available:

```bash
nvidia-smi
```

On a host with any existing Python 3.8+ interpreter, the following command
initializes Git submodules, creates a user-local Python 3.9 when needed,
creates the project virtual environment, downloads MuJoCo 2.1.0 and all nine
required D4RL datasets (about 1.6GB), verifies their checksums, and runs the
CUDA/MuJoCo smoke test. It does not require elevated privileges:

```bash
tools/setup_environment.sh
```

All QRL Python dependencies are installed into `.venv`. If Python 3.9 is not
already available, the script downloads micromamba into
`../qrl-assets/micromamba/` and creates Python 3.9 there, without modifying
the host Python installation. It also installs the X11 protocol, X11, and GLEW
development headers plus `patchelf` required by `mujoco-py` into
`../qrl-assets/micromamba/envs/graphics/`; this avoids requiring root access
when a server lacks `X11/Xlib.h`. `--skip-submodules`, `--skip-python`,
`--skip-mujoco`, `--skip-datasets`, `--skip-simulators`, and `--skip-verify`
allow partial setup.
It accepts `QRL_ASSET_ROOT` for a different asset location and URL overrides
for an internal mirror. Pip uses its default 15-second socket timeout and the
bootstrap script uses 3 retries for package downloads. MuJoCo and every dataset
are validated before use. Asset paths are normalized before launching
`mujoco-py`, whose legacy loader requires canonical library paths.

To add only the online DMC, Gym MuJoCo, Gymnasium-Robotics, and Panda-Gym
simulators to a server that already has this repository's `.venv`, run:

```bash
git pull
tools/setup_environment.sh --simulators-only
```

This is an incremental installation: matching packages and an existing MuJoCo
2.1 runtime are retained, while missing or mismatched components are repaired.
It does not initialize submodules, download D4RL datasets, or rebuild the
virtual environment. Gym 0.18 and Gymnasium intentionally coexist: the legacy
adapter selects its original Gym backend explicitly, while the Robotics and
Panda adapters use Gymnasium. Finally, the command creates, resets, and steps
every registered DMC, Gym MuJoCo, Gymnasium-Robotics, online-maze, and
Panda-Gym task. Set
`VENV_DIR=/path/to/venv` if the existing environment is not `.venv` in the
repository root.

NVIDIA drivers and the driver-provided EGL/OpenGL runtime cannot be installed
in a Python virtual environment. They are normally already present on an AI
server. The repository supplies user-local build headers but never invokes a
system package manager or privileged command; if the NVIDIA driver runtime is
absent, ask the server administrator to provision it before running setup.

The nested bootstrap script installs PyTorch 2.8.0 from the CUDA 12.8 wheel
index by default. This is a default, not a hardware requirement. Select a
compatible wheel index when the target driver needs a different CUDA runtime:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 \
    PYTHON_BIN=python3.9 tools/bootstrap_environment.sh
```

Set `TORCH_INDEX_URL=` to use the default Python package index. The remaining
direct dependencies are pinned in `requirements/offline-py39.txt`. D4RL is
installed without dependency resolution because its declared `mjrl` dependency
is not used by this repository's Maze2D and AntMaze experiments. The bootstrap
script also pins pip, setuptools, and wheel because Gym 0.18.0 cannot be built
with current setuptools releases. Pillow is pinned to a Python 3.9 binary-wheel
release so setup does not require system JPEG development headers. Gym is
installed separately without dependency resolution because its obsolete Pillow
upper bound conflicts with that binary wheel; its required dependencies remain
pinned in the requirements file.

## Place External Assets

Use the following asset layout next to the clone:

```text
../qrl-assets/
  d4rl/datasets/
    *.hdf5
  mujoco/mujoco210/
    bin/libmujoco210.so
    include/
    model/
```

`tools/setup_environment.sh` creates this layout automatically. To reuse an
existing artifact store instead, copy the assets while in the repository root:

```bash
mkdir -p ../qrl-assets/d4rl/datasets ../qrl-assets/mujoco
rsync -a --info=progress2 source-host:path/to/datasets/ \
    ../qrl-assets/d4rl/datasets/
rsync -a --info=progress2 source-host:path/to/mujoco210/ \
    ../qrl-assets/mujoco/mujoco210/
```

MuJoCo 2.1.0 should be obtained and used according to its license. The
manifest `docs/d4rl_v2_datasets.tsv` records the filenames, byte sizes, and
SHA-256 values for the required Maze2D v1 and AntMaze v2 datasets. Verify the
copy after exporting the runtime environment:

```bash
source tools/qrl_env.sh
awk -F '\t' '!/^#/ {print $3 "  " $1}' docs/d4rl_v2_datasets.tsv | \
    (cd "$D4RL_DATASET_DIR" && sha256sum -c -)
```

`tools/qrl_env.sh` derives `QRL_ASSET_ROOT` from `../qrl-assets` and exports
the D4RL, MuJoCo, EGL, and dynamic-library variables. It contains no
machine-specific path that needs editing.

## Verify And Run

Run the full preflight before launching experiments:

```bash
source tools/qrl_env.sh
.venv/bin/python tools/verify_environment.py --checksums --smoke --require-cuda
```

Create the local queue configuration from its portable relative-path example:

```bash
cp configs/qrl_queue.example.env configs/qrl_queue.env
CONFIG=configs/qrl_queue.env tools/run_qrl_queue.sh --once
```

`configs/qrl_queue.env` is ignored by Git, so local GPU selection and runtime
preferences do not leak into the repository. The setup script creates an empty
runtime task list at `runs/qrl_queue/tasks.tsv`; add local task rows there
before starting the scheduler.

### Result Storage Modes

The queue supports two explicit storage modes. Local-only mode is the default:
set `RESULTS_ROOT` to a filesystem with enough capacity and leave remote sync
disabled. On server 225, for example:

```bash
# configs/qrl_queue.env
RESULTS_ROOT="/data2/zhangcheng/qrl-assets/results/queue"
REMOTE_RESULTS_SYNC_ENABLED=0
```

Remote-mirror mode first writes the active training result locally, then copies
each newly completed task directory to an SSH-accessible result server. A copy
is considered successful only after `rsync` exits successfully and the remote
`COMPLETE` marker is verified. Failures are retried by later scheduler passes,
and the scheduler does not report the queue complete while requested copies are
pending. Verified local task directories are retained; this mode does not
delete local results.

Create a dedicated key on the training server. The private key and connection
state live under the Git-ignored `runs/` directory and must never be committed:

```bash
mkdir -p runs/qrl_queue/ssh
ssh-keygen -t ed25519 \
  -f runs/qrl_queue/ssh/id_ed25519_qrl_results \
  -N '' -C qrl-results
ssh-copy-id -i runs/qrl_queue/ssh/id_ed25519_qrl_results.pub \
  -p <ssh-port> <remote-user>@<remote-host>
```

Create the destination directory on the result server and ensure `rsync` is
installed on both machines. Then configure the ignored training-server file:

```bash
# configs/qrl_queue.env
RESULTS_ROOT="../qrl-assets/results/queue"
REMOTE_RESULTS_SYNC_ENABLED=1
REMOTE_RESULTS_SYNC_HOST="<remote-host>"
REMOTE_RESULTS_SYNC_PORT=<ssh-port>
REMOTE_RESULTS_SYNC_USER="<remote-user>"
REMOTE_RESULTS_SYNC_ROOT="/absolute/remote/results/queue"
REMOTE_RESULTS_SYNC_KEY="runs/qrl_queue/ssh/id_ed25519_qrl_results"
REMOTE_RESULTS_SYNC_KNOWN_HOSTS="runs/qrl_queue/ssh/known_hosts_remote_results"
REMOTE_RESULTS_SYNC_BASELINE_FILE="runs/qrl_queue/remote_sync/local_only_baseline.txt"
REMOTE_RESULTS_SYNC_TIMEOUT_SECONDS=300
REMOTE_RESULTS_SYNC_MAX_PER_PASS=2
```

Before enabling a scheduler for the first time, snapshot all result directories
that already exist locally. Those task IDs remain local-only; directories
created later are eligible for remote copying:

```bash
mkdir -p ../qrl-assets/results/queue
.venv/bin/python tools/snapshot_qrl_remote_sync_baseline.py \
  --config configs/qrl_queue.env \
  --output runs/qrl_queue/remote_sync/local_only_baseline.txt
tools/verify_qrl_remote_results.sh
```

The verification command records the remote host key, checks non-interactive
key authentication, confirms destination write access and `rsync`, and performs
a temporary write probe. Machine-specific values can also be supplied through
the `QRL_RESULTS_REMOTE_*` environment overrides.

Stop only the scheduler while allowing its current training jobs to finish:

```bash
tools/stop_qrl_queue.sh
```

Stop the scheduler and all jobs in its process group:

```bash
tools/stop_qrl_queue.sh --with-jobs
```

Queue rows use seven tab-separated columns: `task_id`, `mode`, `env_name`,
`seed`, `steps`, `params`, and `extra_args`. Record total agent parameters in
`params` when adding the task, either as an integer or a compact value such as
`11.6m`. Legacy six-column rows remain valid and show an empty parameter field.

The queue watcher's `#` column is recalculated from the current task-file order
on every refresh. It is a display position, not a persistent task identifier.
Use the stable first-column `task_id` when managing tasks.

### Phone Notifications

The queue runner supports ServerChan Turbo and ntfy. ServerChan is the default
choice for Android/WeChat delivery on networks where `ntfy.sh` is unavailable.
Obtain an `SCT` SendKey from [ServerChan](https://sct.ftqq.com/), and add it only
to the ignored local configuration. The SendKey is a credential: never commit
or print it.

```bash
# configs/qrl_queue.env
SERVERCHAN_SENDKEY="<SCT SendKey>"
NOTIFY_HOST_LABEL="<friendly server name>"
NTFY_TOPIC_URL=""
NOTIFY_NO_PENDING=0
NOTIFY_TASK_DONE=0
NOTIFY_TASK_FAILED=1
NOTIFY_QUEUE_DONE=1
```

Configure a ServerChan delivery channel, then send a lock-free test message:

```bash
.venv/bin/python tools/run_qrl_queue.py \
  --config configs/qrl_queue.env --test-notification
```

Set a different `NOTIFY_HOST_LABEL` on each machine (for example `226`, `L40`,
and `225`) even when they share one SendKey. Every notification title starts
with this label; the label is also included in the message body. When the
setting is empty, the operating-system hostname is used.

The running scheduler reloads this configuration on every polling cycle.
When notifications are first enabled, it records the current queue state as a
baseline instead of replaying historical events. By default, the completion
notification is sent only after every task is terminal and no training process
remains. A queue containing `FAILED` or `PAUSED` tasks is reported as finished
with issues rather than successfully completed. Reaching zero `PENDING` tasks
does not notify because currently `RUNNING` tasks may still be training; the
optional `NOTIFY_NO_PENDING` setting can restore that separate backlog-drained
notification when explicitly needed. Newly `FAILED` or `PAUSED` tasks are also
reported with their task metadata, error, log path, and a short log excerpt;
multiple failures found in one scheduler cycle are combined into one message.
Per-task completion notifications remain optional. Delivery errors are logged
and retried later without stopping training.

Permanently deleting a task removes its rows from the active task table and
local `tasks.tsv.before_*` histories, its status and status temporary file, all
attempt logs, result directory, and watcher ETA history. Deletion defaults to a
dry run and requires an exact expected count when executed:

```bash
.venv/bin/python tools/delete_qrl_tasks.py \
  --task-id 'official_qrl_1q_Base_M_200k_fetchpush_online_s1000'

.venv/bin/python tools/delete_qrl_tasks.py \
  --task-id 'official_qrl_1q_Base_M_200k_fetchpush_online_s1000' \
  --yes --expect 1
```

Tasks can also be selected without knowing their full ID by combining stable
metadata filters. Filters are ANDed across categories and ORed when a category
is repeated:

```bash
.venv/bin/python tools/delete_qrl_tasks.py \
  --task-id-glob 'official_qrl_1q_*' \
  --state PAUSED --env-name FetchPush --seed 1000
```

Stop the scheduler before deleting any task still present in the active task
table. A `RUNNING` status whose recorded PID no longer exists is displayed as
`STALE` and can be deleted; a task with a live PID is always rejected. Orphaned
status records can be selected explicitly with `--orphaned`.

To reclaim checkpoint storage from completed tasks while preserving their task
rows, status, attempt logs, configuration, TensorBoard data, and evaluation
summaries, add `--checkpoints-only`. This mode only accepts `DONE` tasks and
deletes every top-level `*.pth` file in each matched result directory. It uses
the same dry-run and exact-count confirmation flow:

```bash
.venv/bin/python tools/delete_qrl_tasks.py \
  --state DONE --env-name FetchPush --checkpoints-only

.venv/bin/python tools/delete_qrl_tasks.py \
  --state DONE --env-name FetchPush --checkpoints-only \
  --yes --expect 3
```

After a successful cleanup, the result directory contains a
`CHECKPOINTS_DELETED` manifest with the deletion time, file count, and reclaimed
bytes. The queue monitor displays `deleted` in that task's `ckpt` column.

### Automatic checkpoint cleanup

To reclaim storage automatically for an individual task, append this
queue-only option to its `extra_args` field:

```text
queue.delete_checkpoints_after_completion=true
```

The queue runner does not pass this option to Hydra and does not include it in
the training-definition fingerprint. It can therefore be added before a task
runs or to an already completed task. For an online task, cleanup starts only
after `COMPLETE`, `best_checkpoint.json`, and a non-empty `test.log` confirm
that validation-based checkpoint selection and final test evaluation have
finished. A `COMPLETE` marker alone is deliberately insufficient because the
offline training entry point writes it before the separate offline evaluation
workflow. The runner then deletes every top-level `*.pth` file, including
replay, periodic, final, agent-only, and selected-best checkpoints, while
preserving task/status records, logs, TensorBoard data, configuration, and
evaluation summaries. Cleanup is irreversible and prevents later
checkpoint-based continuation.

The running scheduler notices the flag on its next polling cycle. With the
scheduler stopped, process newly marked completed tasks once with:

```bash
.venv/bin/python tools/run_qrl_queue.py \
  --config configs/qrl_queue.env --sync-only
```

`tools/diagnose_qrl_goal_bias.py` is an optional cross-project analysis tool,
not a training requirement. It needs the companion `scaling-crl` checkout; set
`SCALING_CRL_ROOT` to that checkout when using the diagnostic.

## What To Version

Commit source code, `configs/qrl_tasks*.tsv`, `tools/`, `docs/`, requirements,
and small evaluation summaries. Keep virtual environments, external assets,
raw logs, TensorBoard files, replay buffers, and checkpoints outside Git.
Also keep `configs/qrl_queue.env`, ServerChan SendKeys, SSH private keys,
`known_hosts`, and remote-sync baseline/state files outside Git; commit only
`configs/qrl_queue.example.env` and the generic tooling.

Use a personal fork as the working repository while retaining the official
repository as `upstream`:

```bash
git remote rename origin upstream
git remote add origin git@github.com:<your-account>/<your-qrl-fork>.git
git fetch upstream
git push -u origin main
```

Do not use `git add -A` in this worktree: it may contain large generated
experiment artifacts. A separate repository only becomes preferable if this
work evolves into an independent project; retain the upstream URL and base
commit in that case.

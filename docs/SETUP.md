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
`--skip-mujoco`, `--skip-datasets`, and `--skip-verify` allow partial setup.
It accepts `QRL_ASSET_ROOT` for a different asset location and URL overrides
for an internal mirror. Pip uses its default 15-second socket timeout and the
bootstrap script uses 3 retries for package downloads. MuJoCo and every dataset
are validated before use. Asset paths are normalized before launching
`mujoco-py`, whose legacy loader requires canonical library paths.

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
preferences do not leak into the repository. The one-critic AntMaze ablations
are defined in `configs/qrl_tasks_1q_antmaze.tsv`.

`tools/diagnose_qrl_goal_bias.py` is an optional cross-project analysis tool,
not a training requirement. It needs the companion `scaling-crl` checkout; set
`SCALING_CRL_ROOT` to that checkout when using the diagnostic.

## What To Version

Commit source code, `configs/qrl_tasks*.tsv`, `tools/`, `docs/`, requirements,
and small evaluation summaries. Keep virtual environments, external assets,
raw logs, TensorBoard files, replay buffers, and checkpoints outside Git.

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

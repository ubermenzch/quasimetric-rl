# Online GCRL baselines

The online trainer supports four reward-free goal-conditioned baselines on the
seven validated environments in `ONLINE_GOAL_ENVS.md`:

| CLI value | Method | Implementation source |
| --- | --- | --- |
| `td_infonce` | TD-InfoNCE | `chongyi-zheng/td_infonce`, commit `18f4e7e5` |
| `crl` | CRL | JaxGCRL CRL, commit `7c53a074`; original Google CRL also inspected at commit `ec7c3d34` |
| `gcbc` | GCSL/GCBC | `dibyaghosh/gcsl`, commit `cfae5609` |
| `c_learning` | C-Learning | `google-research/google-research/c_learning`, commit `ec7c3d34` |

These agents do not optimize the task reward. C-Learning constructs its binary
next-goal labels internally, and the other methods learn from state/action
trajectories and hindsight goals. Environment rewards are still recorded by the
shared replay and evaluation code for reporting only.

## Fidelity choices

The rule for the comparison matrix is strict: start from the executable online
defaults in the reference repository, then change hidden-layer widths only to
meet the M parameter budget. A model-size YAML is not allowed to override a
loss, optimizer, representation dimension, batch size, replay rule, exploration
rule, initialization, or update schedule.

| Method | Reference defaults used by this port | M-only width change |
| --- | --- | --- |
| TD-InfoNCE | batch 256; actor/critic LR `5e-5`/`3e-4`; discount `0.99`; tau `0.005`; replay 10k-1M; UTD 1; twin normalized 16-D representations; random-goal actor; random behavior until 10k | none: `512 x 4` is already M-sized |
| CRL | batch 256; all LRs `3e-4`; discount `0.99`; replay 1k-10k; default JaxGCRL UTD `1001 / (256 * 62)` (approximately `1:16`); forward InfoNCE; norm energy; LSE coefficient `0.1`; 64-D representation; SiLU; no layer norm; adaptive entropy; stochastic untrained-policy prefill | `256 x 2` to `1152 x 2` |
| GCSL/GCBC | batch 256; LR `5e-4`; one update per environment step; start training after 1k; uniform discrete exploration until 10k, then greedy; action granularity 3; 20% validation trajectories; 20k-trajectory buffers; uniform strict-future hindsight pairs | `400 -> 300` to `2304 -> 1728`, preserving the width ratio |
| C-Learning | batch 256; actor/critic LR `3e-4`; discount `0.99`; tau `0.005`; random prefill 10k; replay 1M; one collect and one train step; pure TD relabeling with 50% next and 50% shuffled random goals; odds clip 20; critic before actor | `256 x 2` to `1184 x 2` |

The implementation deliberately retains source behavior that might otherwise
look like something to "fix". Examples include CRL's epsilon inside the norm
square root and its source cosine formula, GCSL's joint action discretization
for these seven action spaces, and C-Learning's unclamped odds denominator and
default policy standard-deviation transform. There are no task-specific rescue
settings or numerical-stability substitutions in the comparison tasks.

C-Learning's repository provides different explicit relabeling commands for
its Sawyer tasks. The seven environments here are not those Sawyer definitions,
so this matrix uses the code-level pure-TD default (`next=0.5`, `future=0.0`)
rather than importing a task-specific Sawyer override.

The following are common experimental-protocol adaptations, not algorithm
hyperparameters: PyTorch instead of the source JAX/TensorFlow runtime, the
repository's seven environment wrappers and fixed episode horizons, sequential
instead of vectorized collection, 200k environment interactions, and validation
at 20k checkpoints followed by one test of the selected checkpoint. Therefore
the paper should call these source-faithful ports under a common protocol, not
bitwise reproductions of the original runtime.

All comparison tasks explicitly set `batch_size=256`; the trainer rejects a
different shared batch size. Algorithm-owned defaults also override the shared
UTD and prefill defaults. The source values are regression-tested in
`tests/test_gcrl_baselines.py`.

All comparison tasks select an algorithm-specific M preset. TD-InfoNCE uses
`512 x 4`, CRL uses `1152 x 2`, GCSL uses `2304 -> 1728`, and C-Learning uses
`1184 x 2`. Representation dimensions and layer counts are unchanged. Frozen
target copies are excluded from the 4.0M-4.5M trainable-parameter count. Exact
ranges are in `configs/model_size/README.md`.

## Run one experiment

```bash
source tools/qrl_env.sh
.venv/bin/python -m online.main \
  env.kind=gcrl \
  env.name=FetchPush \
  agent.algorithm=td_infonce \
  +td_infonce_model_size=m \
  batch_size=256 \
  seed=1000 \
  interaction.total_env_steps=200000 \
  interaction.exploration_eps=0 \
  interaction.num_eval_episodes=200 \
  interaction.num_test_episodes=200 \
  interaction.validation_seed=1000 \
  interaction.test_seed=2000000 \
  eval_steps=null \
  save_steps=20000 \
  keep_only_latest_checkpoint=false \
  save_replay_buffer=true
```

Use `crl`, `gcbc`, or `c_learning` together with their corresponding
`+crl_model_size=m`, `+gcbc_model_size=m`, or
`+c_learning_model_size=m` preset. As with QRL and
GO-QRL, every 20k boundary is evaluated on the validation seeds and saved with
the replay and optimizer state. Training then reloads the checkpoint with the
best validation success, hitting time, and return tie-breakers, and evaluates
that model once on the disjoint test seeds. The selected artifact is
`selected_best_agent.pth`, with provenance in `best_checkpoint.json`.

## Seven-environment matrix

The checked-in task file contains four algorithms, seven environments, five
training seeds, and 200k environment steps, for 140 tasks total:

```bash
.venv/bin/python tools/generate_online_baseline_tasks.py
```

This writes `configs/gcrl_baselines_4alg_7env_5seed_200k.tsv`. Before submitting
the matrix, the real simulator/update smoke test can be rerun with:

```bash
source tools/qrl_env.sh
.venv/bin/python tools/smoke_test_gcrl_baselines.py
```

Server assignments can append one algorithm idempotently without adding the
other three:

```bash
.venv/bin/python tools/generate_online_baseline_tasks.py \
  --algorithms crl \
  --append-to runs/qrl_queue/tasks.tsv
```

The 35 legacy TD-InfoNCE tasks without `-M` in their IDs predate the strict
reference-default lock. Their architecture size is M, but their saved configs
do not contain all source-faithful initialization, distribution, replay, and
sampling settings. Do not use those results in the strict comparison matrix;
run the new `TD-InfoNCE-M` task IDs instead.

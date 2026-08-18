# Online GCRL baselines

The online trainer supports five reward-free goal-conditioned methods on the
seven validated environments in `ONLINE_GOAL_ENVS.md`. The existing strict
baseline matrix still contains the first four. They now have M and L capacity
presets; Scaling-CRL has a separate multi-level scale for the planned
QRL/GO-QRL capacity comparison.

| CLI value | Method | Implementation source |
| --- | --- | --- |
| `td_infonce` | TD-InfoNCE | `chongyi-zheng/td_infonce`, commit `18f4e7e5` |
| `crl` | CRL (2022) | `google-research/google-research/contrastive_rl`, commit `ec7c3d34` |
| `scaling_crl` | Scaling-CRL (2025) | `wang-kevin3290/scaling-crl`, commit `17acb519` |
| `gcsl` | GCSL | `dibyaghosh/gcsl`, commit `cfae5609` |
| `c_learning` | C-Learning | `google-research/google-research/c_learning`, commit `ec7c3d34` |

These agents do not optimize the task reward. C-Learning constructs its binary
next-goal labels internally, and the other methods learn from state/action
trajectories and hindsight goals. Environment rewards are still recorded by the
shared replay and evaluation code for reporting only.

## Fidelity choices

The rule for the comparison matrix is strict: start from the executable online
defaults in the reference repository, then change hidden-layer widths only to
meet M. L may increase depth to four when width-only scaling would create
excessively wide layers. A model-size YAML is not allowed to override a
loss, optimizer, representation dimension, batch size, replay rule, exploration
rule, initialization, or update schedule.

| Method | Reference defaults used by this port | M / L architecture |
| --- | --- | --- |
| TD-InfoNCE | batch 256; actor/critic LR `5e-5`/`3e-4`; discount `0.99`; tau `0.005`; replay 10k-1M; UTD 1; twin normalized 16-D representations; random-goal actor; random behavior until 10k | `512 x 4` / `1192 x 4` |
| CRL (2022) | batch 256; actor/critic LR `3e-4`; Adam epsilon `1e-7`; discount `0.99`; replay 10k-1M; UTD 1; all-pairs sigmoid BCE NCE; dot-product energy; 64-D unnormalized representations; ReLU; fixed entropy coefficient 0; equal future/shuffled actor-goal mixture; uniform-random behavior until 10k | `1152 x 2` / `1544 x 4` |
| Scaling-CRL (2025) | batch 512; actor/critic/alpha LR `3e-4`; discount `0.99`; replay axis 1k-10k; UTD 1:40; forward InfoNCE; negative L2 energy; LSE penalty `0.1`; 64-D representations; LayerNorm/SiLU four-layer residual blocks; learned entropy; initialized stochastic actor prefill | joint residual depth/width selected by `Scaling-CRL-{M,L,XL,XXL,XXXL}` |
| GCSL | batch 256; LR `5e-4`; one update per environment step; start training after 1k; uniform discrete exploration until 10k, then greedy; action granularity 3; 20% validation trajectories; 20k-trajectory buffers; uniform strict-future hindsight pairs | `2304 -> 1728` / `2680 x 4` |
| C-Learning | batch 256; actor/critic LR `3e-4`; discount `0.99`; tau `0.005`; random prefill 10k; replay 1M; one collect and one train step; pure TD relabeling with 50% next and 50% shuffled random goals; odds clip 20; critic before actor | `1184 x 2` / `1544 x 4` |

The implementation deliberately retains source behavior that might otherwise
look like something to "fix". Examples include CRL's unweighted `B x B` sigmoid
BCE with the diagonal as positives and every off-diagonal pair as a negative,
GCSL's joint action discretization for these seven action spaces, and
C-Learning's unclamped odds denominator and default policy standard-deviation
transform. There are no task-specific rescue settings or numerical-stability
substitutions in the comparison tasks.

C-Learning's repository provides different explicit relabeling commands for
its Sawyer tasks. The seven environments here are not those Sawyer definitions,
so this matrix uses the code-level pure-TD default (`next=0.5`, `future=0.0`)
rather than importing a task-specific Sawyer override.

The hindsight-goal representation is separate from the dimensions used to
score task success. In `FetchPush`, `FetchSlide`, and `FetchPickAndPlace`, the
success check remains object position `(3, 4, 5)`, while the baseline inputs
follow their reference state representations:

| Method | Fetch manipulation conditioning goal |
| --- | --- |
| CRL (2022), TD-InfoNCE, C-Learning | complete 25-D future/desired state `(0:25)` |
| Scaling-CRL | task goal coordinates `(3:6)` |
| GCSL | gripper and object positions `(0:6)` |

GCSL's replay buffer calls the environment's `extract_goal()` rather than
passing the whole simulator state. Its SawyerPush goal space contains only hand
and puck position. C-Learning's SawyerPush reproduction command does not set
`obs_to_goal`; its `log_subset=(3, 6)` setting affects metrics only, so the
training goal remains the complete state under the source default.

This distinction preserves gripper-motion learning during early exploration;
the three object coordinates are an evaluation predicate, not a shared goal
encoder for all methods. Original CRL uses `start_index=0` and
`end_index=-1`, so it conditions on the complete state in every validated
environment. The other methods retain their registered task-goal dimensions
outside the three Fetch manipulation tasks.

Fetch manipulation checkpoints trained before the goal-representation
separation used a 3-D baseline goal input. Their actor/critic parameter shapes
are incompatible with the corrected 25-D or 6-D inputs and they must be
retrained rather than resumed. Non-CRL replacement tasks use the `goalreprv2`
task-ID tag. All new CRL tasks use `originalcrl2022`, including non-Fetch tasks,
because the previous implementation used JaxGCRL's objective, network, entropy,
replay, and update defaults rather than the 2022 CRL defaults. Old CRL
checkpoints must not be resumed into the new implementation.

The following are common experimental-protocol adaptations, not algorithm
hyperparameters: PyTorch instead of the source JAX/TensorFlow runtime, the
repository's seven environment wrappers and fixed episode horizons, sequential
instead of vectorized collection, 200k environment interactions, and validation
at 20k checkpoints followed by one test of the selected checkpoint. Therefore
the paper should call these source-faithful ports under a common protocol, not
bitwise reproductions of the original runtime.

For Scaling-CRL specifically, the official queue axes are time by 512 parallel
environments. This sequential trainer interprets its published 1k/10k replay
sizes as total recent transitions. Multiplying those values by 512 would make
the replay and prefill interaction budget fundamentally different from the
other methods in this repository. This is a common-protocol adaptation and
must be reported as such. The future-goal geometric discount, initialized
stochastic-policy prefill, batch 512, and approximately 1:40 UTD are retained.

All comparison tasks explicitly set `batch_size=256`; the trainer rejects a
different shared batch size. Algorithm-owned defaults also override the shared
UTD and prefill defaults. The source values are regression-tested in
`tests/test_gcrl_baselines.py`.

The checked-in comparison matrix selects algorithm-specific M presets. L
presets are also available for a matched 21.5M-21.9M comparison. Representation
dimensions are unchanged. CRL, GCSL, and C-Learning use four hidden layers at L;
their M checkpoints cannot be resumed at L. Frozen target copies are excluded
from trainable-parameter counts. Exact ranges are in
`configs/model_size/README.md`.

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

Use `crl`, `gcsl`, or `c_learning` together with their corresponding model-size
group. Replace `m` with `l` in any of the four baseline overrides to select the
L budget. As with QRL and
GO-QRL, every 20k boundary is evaluated on the validation seeds and saved with
the replay and optimizer state. Training then reloads the checkpoint with the
best validation success, hitting time, and return tie-breakers, and evaluates
that model once on the disjoint test seeds. The selected artifact is
`selected_best_agent.pth`, with provenance in `best_checkpoint.json`.

Scaling-CRL is run independently with its own batch and preset, for example:

```bash
source tools/qrl_env.sh
.venv/bin/python -m online.main \
  env.kind=gcrl env.name=FetchSlide \
  agent.algorithm=scaling_crl \
  +scaling_crl_model_size=l \
  batch_size=512 \
  interaction.exploration_eps=0 \
  interaction.total_env_steps=500000
```

Use `l`, `xl`, `xxl`, or `xxxl` to select the four planned comparison tiers.
These are genuine Scaling-CRL models; the presets do not route through the
`crl` implementation. See `configs/model_size/README.md` for depth, width, and
parameter matching.

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

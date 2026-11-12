# M-residual dynamics comparison

The current study contains twelve fresh runs: six Min4 tasks in
[`configs/figure2_m_residual_min4_antnavigate_6seed.tsv`](../configs/figure2_m_residual_min4_antnavigate_6seed.tsv)
and six Min32 tasks in
[`configs/figure2_m_residual_min32_antnavigate_6seed.tsv`](../configs/figure2_m_residual_min32_antnavigate_6seed.tsv).
Each completion setting pairs IQE-only and IQE+MSE on seeds 1000–1002 using
a two-layer residual/LayerNorm/SiLU M network. The previous plain/ReLU M
configuration is retained as `+cqrl_model_size=m`; the new architecture is
`+cqrl_model_size=m_residual` (4,391,126 trainable parameters on AntNavigate).

## Matched training protocol

| Setting | Both arms |
| --- | --- |
| Environment | AntNavigate-v4; 29 state dimensions, 8 actions, goal coordinates 0/1 |
| Training seeds | 1000, 1001, 1002; fresh task IDs |
| Budget | 500,000 training environment steps per run (6,000,000 total across twelve runs) |
| Completion | Min4 or Min32; SGD LR 0.01 for both; direct search; keep best candidate |
| Dynamics | Joint training; outer weight 0.1; IQE weight 1 |
| MSE weight | 0 for IQE-only; 1 for IQE+MSE |
| Batch size | 256 |
| Validation | Every 20k steps, 100 episodes, seeds 1000–1099 |
| Selected-checkpoint test | 100 episodes, seeds 2000000–2000099; deterministic actions |

Both arms use `distance=iqe_mse`. Their training configurations differ only
in the MSE weight. A disabled loss term is omitted from the optimization
expression, so a nonfinite diagnostic MSE cannot enter through `0 * MSE`.
Its raw value and gradient are still measured. Seeds are paired and the
preflight checks identical initialization and identical fixed raw transitions.

Within a completion setting the two arms differ only in MSE weight. Across
Min4 and Min32 only the number of minimizing updates differs. This tests the
failure's persistence under the new network and minimizing completion;
it does not substitute for the L/Min32/LR1 main-method ablation. Each run has its
own online replay distribution. Retained raw transitions and model snapshots
allow later fixed-model or common-data analyses.

The earlier six Max4 runs are superseded. Their original
[`manifest`](../configs/figure2_m_residual_antnavigate_6seed.tsv), configurations,
checkpoints and diagnostic records are retained. Min4 and Min32 use new `v2`
task IDs and train from scratch; they never resume Max4 checkpoints. The old
Max4 rows are removed from the active queue so interrupted jobs cannot be
rescheduled as part of this study.

## Evidence retained in every run

All files below are under the run's `dynamics_diagnostics/` directory unless
otherwise specified. Recording is enabled in **both** arms from initialization.

| Artifact / interval | Contents and use |
| --- | --- |
| `fixed_probe.pth`, once | 1,024 raw `(s,a,s')` transitions, 128 evenly spaced samples from each of eight uniform-action episodes; seeds 7300000–7300007 and tensor SHA-256. The 8,000 probe environment steps never enter replay or the training budget. |
| `training_env*_opt*.pth`, each collection cycle | Every optimizer update's latent MSE, squared bidirectional IQE, and both directed distances, in update order with optimizer-step bounds. Nonfinite values are preserved. |
| `training_env*_opt*.json`, each collection cycle | Count, finite count, mean, SD, minimum and maximum over all updates. Supports correctly weighted windows and identifies missing/invalid observations. |
| `env*_opt*.json/.pth`, step zero and every 5k after prefill | The full agent state, raw transitions, current/target/predicted latents, projected values, directed distances and predicted-latent IQE/MSE gradients. Measures the same fixed probe and 1,024 contemporary replay samples separately. |
| Representation measurements | Full/goal/non-goal error distributions; latent norms, variance, dispersion, constant-coordinate fraction; MSE normalized by target variance and by a persistence predictor. Projected MSE and projected representation statistics are also recorded. |
| Projector and gradient measurements | Per-layer SiLU activation/derivative statistics on target and prediction; separate raw/effective IQE and MSE gradient norms for predicted latents and dynamics parameters, zero/nonfinite fractions and gradient cosine. Full models and inputs permit finer gradient or activation analyses later. |
| Validation-aligned diagnostics, every 20k | Exact measurements linked from `eval.log`, with the evaluated model cursor. These do not replace checkpoint error with a preceding training-window average. |
| `validation_env*_opt*.pth`, `test_env*_opt*.pth` | Per-episode rewards, success masks, first-arrival and goal-distance records returned by evaluation, plus seed/checkpoint summaries. |
| `source_*.zip`, `provenance_*.json` | Active Python code including local changes and scheduler/cleanup code, model YAMLs, this batch's task manifest, resolved configuration, package versions, source hashes, probe hash and parameter count. Records Python/OS, device, Torch/CUDA/cuDNN versions, TF32 and determinism settings. Each resumed source/configuration/runtime has its own identity. |
| Run-root checkpoints every 20k | All agent/optimizer/RNG/replay checkpoints and validation agent snapshots are retained (`keep_only_latest_checkpoint=false`, `keep_only_best_and_final_checkpoints=false`). The final replay is also retained. |
| Existing run-root logs | TensorBoard events, `output.log`, resolved `config.yaml`, `eval.log`, selected-checkpoint `test.log` and `test_episodes.json`. |

Each task explicitly sets `queue.delete_checkpoints_after_completion=false`.
Remote backup still runs, but this setting overrides the scheduler's global
post-backup deletion of local checkpoints. Both local models/replay and their
remote backup remain available for analysis.

The initial diagnostic is untrained. AntNavigate prefill is 10k, so the first
trained diagnostic is at 10k, followed by 15k, 20k, etc. Diagnostic forwards use
FP32 evaluation mode with detached encoder targets and `autograd.grad`; they
restore RNG, module modes and existing parameter gradients. Diagnostic errors
therefore describe a separate measurement, while per-update records capture
the losses actually seen in training. Resuming discards records beyond the
restored committed cursor and reuses the original fixed probe.

## Preflight and scheduling

```bash
source tools/qrl_env.sh
.venv/bin/python tools/check_m_residual_dynamics.py --manifest configs/figure2_m_residual_min4_antnavigate_6seed.tsv --output runs/m_residual_min4_preflight
.venv/bin/python tools/check_m_residual_dynamics.py --manifest configs/figure2_m_residual_min32_antnavigate_6seed.tsv --output runs/m_residual_min32_preflight
```

This composes all six queue configurations, checks the paired differences and
old M preservation, then exercises both actual M-residual agents on CPU. Each
arm collects an isolated fixed probe and one real AntNavigate rollout, takes
one optimizer update, writes diagnostics, and evaluates two episodes. It
checks parameter count, shared initialization/probe hashes, both diagnostic
panels, zero versus nonzero effective MSE gradients, raw batch files, source
archives, and evaluation records. These preflight interactions are separate
from the queued training runs. Results are written under
the selected output directory. Omitting `--manifest` retains the original
Max4 preflight for reproducing the superseded setup.

The Min4 rows replace the six old Max4 rows in `runs/qrl_queue/tasks.tsv`;
the Min32 rows are appended. New jobs are submitted with `PENDING` status
and await normal scheduling. Queue mutation and preflight references are
recorded in `runs/m_residual_min_variants_submission/submission.json`.

If operating from a tool sandbox that cannot see the host training PIDs,
stop the six superseded Max4 processes from the host training shell:

```bash
.venv/bin/python tools/stop_m_residual_max4.py --apply
```

This verifies ownership, the exact output task ID, model preset and `max`
mode before SIGTERM. An absent PID with a final interruption record from its
current training attempt is reported as `ALREADY_EXITED`; stale `RUNNING`
status is then marked `PAUSED` with the previous PID and a supersession note.
Without exit evidence, an absent PID is reported as `ABSENT_UNVERIFIED`, and
the script sends no signals and changes no statuses. An absent PID alone
does not establish process namespace isolation. All result files are preserved;
the script never signals Min4/Min32 or other tasks.

On 2026-09-10 the running scheduler's existing GPU process cleanup terminated
the six removed Max4 jobs between 19:47 and 19:48. Their attempt logs record
SIGTERM interruptions. The old rows were already absent, so their status files
retained stale `RUNNING` values until reconciled by this helper. The stop report
is saved as `runs/m_residual_min_variants_submission/stop_result.json`.

For analysis, report all three seeds per arm, including zero-success counts,
individual error trajectories and selected-checkpoint test outcomes. Smaller
raw MSE should be interpreted alongside target variance and policy success.

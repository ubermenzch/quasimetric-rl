# Original-Paper Task Validation

This suite runs the **existing local baseline implementations** on
tasks appearing in their own papers. It is a one-training-seed diagnostic, not
an exact reproduction claim, a convergence certificate, or new NQRL paper data.
The initial preparation did not start full-budget jobs. The runtime queue now
contains completed runs as well as the follow-ups recorded below.

## Coverage Update (2026-09-10)

The four baseline papers have **22 online, state-observation main tasks**:
TD-InfoNCE 4, CRL 5, GCSL 5, and C-Learning 8. Of these, 11 have completed
local runs, one CRL run has been repaired and requeued, and nine additions have
passed preflight. The executable manifest now covers all 21 planned tasks.
C-Learning Pen is skipped by user request. Four of the latest seven additions
load released environment wrappers; Jaco, Finger and Hand use documented
reconstructions of Appendix H.3 because their original wrappers were not released.
The current plan covers 21 tasks; the full inventory retains all 22 for
traceability. This is not yet a complete paper reproduction.
After appending the seven verified additions, the audit reports 11 DONE,
10 PENDING and one SKIPPED_USER_REQUEST. The scheduler was not started here.
The scope follows the existing four transplanted baselines and training seed
1000; QRL, NQRL and Scaling-CRL are not added to this diagnostic suite.

The executable superset is
[`baseline_paper_validation_main_available_1seed.tsv`](../configs/baseline_paper_validation_main_available_1seed.tsv).
It retains all 12 original task IDs and adds:

| Algorithm | New task | Steps | Validation interval |
| --- | --- | ---: | ---: |
| TD-InfoNCE | FetchReach | 500,000 | 10,000 |
| C-Learning | CLearningSawyerDrawer | 3,000,000 | 49,981 |
| CRL | CRLSawyerPush | 3,000,000 | 49,950 |
| CRL | CRLSawyerBin | 3,000,000 | 49,950 |
| GCSL | GCSLLunarLander | 200,000 | 2,000 |
| GCSL | GCSLClaw | 1,000,000 | 10,000 |
| C-Learning | CLearningJacoReach (reconstructed) | 3,000,000 | 50,000 |
| C-Learning | CLearningFinger (reconstructed) | 3,000,000 | 50,000 |
| C-Learning | CLearningHand (reconstructed) | 3,000,000 | 50,000 |

All use the existing 50-validation/200-test episode protocol. FetchReach's
budget follows `td-infonce-official/lp_td_infonce.py`. Drawer uses the original
151-step loader and `SawyerDrawer.reset(arm_goal_type='goal')` from the pinned
C-Learning release. Its local loss remains pure TD; the paper used a TD/MC
mixture for Drawer as well as Push. This difference must remain explicit in
result comparisons.

All 21 executable configurations passed real environment collection, three
optimizer updates, finite-gradient checks, serialization and evaluation after
the resource fix. The seven additions have 127,104 to 273,466 trainable
parameters; no 22M model preset is used. Reports and the full main-task inventory can be refreshed
with:

```bash
source tools/qrl_env.sh
.venv/bin/python tools/check_baseline_paper_validation.py --suite paper-main
.venv/bin/python tools/generate_baseline_paper_validation_tasks.py --suite paper-main --append-to runs/qrl_queue/tasks.tsv
.venv/bin/python tools/audit_baseline_paper_validation.py
```

The audit writes `analysis/baseline_paper_validation/coverage.json`, including
each completed run's final validation and selected-test metrics. It reports
`planned_tasks`, `skipped_tasks` and `all_planned_complete` separately;
`all_complete` still requires every original task to be DONE. Runtime DONE
and scientific agreement with a paper are distinct: use final distance for
the GCSL/C-Learning tasks, and do not substitute any-time success for final
success. The GCSL Door run, for example, has any-time success 1.0 while its
selected-test mean final angle error is about 0.289 radians.

### Failed CRL Ant U-maze

`paper_validation_v1_CRL_CRLAntUMaze_s1000` failed after the 650,300-step
training cycle with `OSError: [Errno 24] Too many open files`. D4RL's
`MazeEnv.__init__` calls `tempfile.mkstemp` and discards its open descriptor.
Creating and closing five real environments reproduced five leaked XML file
descriptors. The training loop also omitted `close()` on collection environments.

The local adapter now closes the discarded descriptor and removes the generated
XML after model loading, including constructor failures. The shim is limited
to D4RL's maze module during construction; installed and reference source files
are unchanged. Training now closes prefill and cycle environments in `finally`
blocks before yielding optimization batches. One hundred real constructions
and closes showed no XML descriptor growth; lifecycle regression tests cover
both normal collection and errors.

The committed checkpoint at **600,600 environment steps / 590,800 updates**
passed model, optimizer and replay restore plus a finite update and rollout.
The same task ID has been returned to PENDING for exact resume from that
checkpoint, retaining its 10M-step total budget and existing results. It has
not completed its remaining training. The scheduler was not restarted here;
its configured result directory is outside this session's writable workspace.

### Remaining State Tasks

The machine-readable backlog is
[`baseline_paper_validation_remaining.json`](../configs/baseline_paper_validation_remaining.json).
The seven adapters are recorded under `resolved_tasks`; only `CLearningPen`
remains under `tasks`, marked `SKIPPED_USER_REQUEST`. The audit counts executable
tasks once and reports the three reconstructed protocols separately.
Exact protocol choices, sources and remaining scientific uncertainties are in
[`baseline_paper_validation_protocols.json`](../configs/baseline_paper_validation_protocols.json).
Each new run saves these details, installed versions and (for Hand) original
and modified XML hashes in `paper_validation_protocol.json`.

Prepare downloadable dependencies without root privileges:

```bash
sh tools/setup_baseline_paper_dependencies.sh
```

The script uses the existing `.venv` and compiler tools, installs a binary
[SWIG wheel](https://pypi.org/project/swig/4.3.1/), protects installed package
versions with pip constraints, checks out the separate CRL MetaWorld revision,
installs pinned ROBEL/Box2D/mjrl, and skips Pen data preparation and its simulator
check. It checks the remaining base simulators and saves a log under
`logs/baseline_paper_dependencies/`. `--dry-run` prints the commands without
downloads or writes; `--skip-checks` omits only simulator smoke checks.
The user completed dependency preparation on 2026-09-10. Box2D import,
ROBEL DClaw reset/step, D4RL Door reset/step, and DMC imports all passed;
the separate CRL MetaWorld checkout matches the required revision. The successful
log is `logs/baseline_paper_dependencies/setup-20260910-131948-1953929.log`.
Installation or smoke failures still stop with a nonzero exit.
Dependency checks alone do not validate task semantics; use the 21-task preflight.

ROBEL 0.1.2 declares Gym `<0.14` and mujoco-py `<2.1`, which conflict with
this repository's Gym 0.18.0 and mujoco-py 2.1.2.14. The setup script installs
Box2D/transforms3d normally under the existing constraints, then installs only
ROBEL with `--no-deps`. Its other declared runtime dependencies are the existing
NumPy and the explicitly installed transforms3d. This is a deliberate metadata
exception for validating the local port, not an exact original simulator stack.
The final DClaw creation/reset/step smoke check remains mandatory for claiming
runtime compatibility; `pip check` may still report the old ROBEL version bounds.
After the initial resolver failure, rerun the same script to continue; the
already installed SWIG is reused.

If setup has already completed package installation and MetaWorld checkout,
resume just the remaining simulator checks with
`sh tools/setup_baseline_paper_dependencies.sh --checks-only`.
The previous `--from-data` option remains an alias for this command.

If the Door check reports `UnregisteredEnv: door-human-v0`, the pinned mjrl
can be missing its undeclared `six` dependency. This was confirmed locally:
importing `mjrl.envs.mujoco_env` directly raised `ModuleNotFoundError: six`,
while D4RL suppressed that import error before registering the hand tasks.
The setup script now installs `six==1.16.0` in step 2 and imports mjrl explicitly
in the Door check so dependency errors remain visible. Rerun the script without
`--checks-only` to install this missing package; existing packages and the
MetaWorld checkout are reused. A diagnostic process temporarily using pip's
bundled six 1.16.0 passed Door reset and ten finite simulation steps. The user's
subsequent full setup installed six and passed the regular Door smoke check.
The Hand adapter now derives a separate model in memory; the installed Door
model and assets remain intact.

On 2026-09-10, both original Pen data endpoints redirected from
`rail.eecs.berkeley.edu` to `iris.eecs.berkeley.edu` and returned HTTP 404.
No alternate source for the exact original file was verified. The user elected
to omit C-Learning Pen, so the setup script no longer downloads or validates its
data, accepts no Pen URL override, and does not check its base simulator.
Its original goal-wrapper work is also dropped from the current plan.

| Algorithm | Added main tasks | Implemented protocol |
| --- | --- | --- |
| CRL | Sawyer Push, Sawyer Bin | Original wrappers with MetaWorld `a0009ed9a208ff9864a5c1368c04c273bb20dd06`; temporary module isolation preserves C-Learning's older classes. Horizon 150. |
| GCSL | Lunar Lander | Original modified LunarEnv, frame skip 2, native four-way categorical actions, horizon 50. |
| GCSL | Claw | Original ClawEnv and ROBEL; nine independent three-bin action heads, sine/cosine goals, wrapped angular distance, horizon 50. |
| C-Learning | Jaco Reach | Reconstructed with manipulation `reach_site_features`, joint-position feature goals from the first of two resets; backend default horizon 250. |
| C-Learning | Finger | Reconstructed with `finger/turn_easy`, spinner-position goals from the first of two resets; backend default horizon 1000. |
| C-Learning | Hand | Remove the frame/door/latch XML subtree, retain hand actuators/dynamics, use 30 hand-only observation coordinates; goal after 50 random actions, reset for a 50-step episode. |
| C-Learning | Pen (skipped) | Excluded by user request after original data retrieval failed; no further download or adapter work planned. |

The exact DMC variants, Jaco angle encoding and modified Hand observation
layout are not specified by the paper or pinned public release. These are
explicit reconstruction choices, so matching curves would support local
implementation validation but would not establish exact reproduction.
The three C-Learning additions use the released trainer's 3M-iteration default
as a validation budget; their original per-task figure budgets remain unverified.
Their discount/odds clipping pairs are the Appendix H.3 values: Jaco `.9/2`,
Finger `.99/50`, Hand `.3/2`. Diagnostic success thresholds for these three
are local conventions; final L2 goal distance is the relevant comparison metric.

### Other Paper Experiments

The 22-task count covers the online state main comparisons only. A broader
request for every experiment also requires the following, which are not
counted as completed or queued:

| Paper | Additional experiment families | Local implementation gap |
| --- | --- | --- |
| [TD-InfoNCE](https://arxiv.org/html/2310.20141v2) | Four image Fetch tasks, stochastic Fetch variants, six offline AntMaze datasets, tabular prediction/planning studies and ablations. | Local baseline agents accept vector observations; offline baseline training and the separate tabular/analysis protocols need implementation. |
| [CRL](https://arxiv.org/html/2206.07568) | Five image tasks (Fetch Reach/Push, Sawyer Push/Bin, Point Spiral11x11), six offline AntMaze tasks, representation transfer and ablations. | Image encoders and an offline baseline trainer are absent. State-based Sawyer adapters are now available. |
| [GCSL](https://arxiv.org/html/1912.06088) | Image observations, FetchReach comparison, relabeling/data-collection ablations and initialization with demonstrations. | Requires image support, original demonstrations and experiment-specific protocols. Native discrete actions are now supported. |
| [C-Learning](https://arxiv.org/html/2011.08909) | Tabular experiments, fixed-policy future-state prediction and goal-relabeling-ratio studies. | These require separate data generation, prediction evaluation and algorithm variants beyond the current online baseline runner. |

The task sets above were checked against the papers' main experiments and
environment appendices, as well as the pinned reference code. Experiments
requiring changes to the local algorithms remain explicitly outstanding.

## Scope and Queue

| Existing algorithm | Original-paper task | Local environment | Budget | Horizon |
| --- | --- | --- | ---: | ---: |
| TD-InfoNCE | Fetch Push | gcrl/FetchPush | 1,000,000 | 50 |
| TD-InfoNCE | Fetch Pick and Place | gcrl/FetchPickAndPlace | 1,000,000 | 50 |
| TD-InfoNCE | Fetch Slide | gcrl/FetchSlide | 1,000,000 | 50 |
| CRL (NCE, 2022) | Fetch Reach | gcrl/FetchReach | 200,000 | 50 |
| CRL (NCE, 2022) | Fetch Push | gcrl/FetchPush | 3,000,000 | 50 |
| CRL (NCE, 2022) | Online Ant U-maze | paper_validation/CRLAntUMaze | 10,000,000 | 700 |
| GCSL | Four Rooms | paper_validation/GCSLFourRooms | 200,000 | 50 |
| GCSL | Sawyer Pushing | paper_validation/GCSLSawyerPush | 1,000,000 | 50 |
| GCSL | Door Opening | paper_validation/GCSLSawyerDoor | 1,000,000 | 50 |
| C-Learning | Sawyer Reach | paper_validation/CLearningSawyerReach | 3,000,000 | 50 |
| C-Learning | Sawyer Push | paper_validation/CLearningSawyerPush | 3,000,000 | 151 |
| C-Learning | Sawyer Window | paper_validation/CLearningSawyerWindow | 3,000,000 | 151 |

Every task uses training seed **1000**. There is no QRL or NQRL task in this
suite. Budgets are the configured validation budgets, not an assertion that all
paper figure axes or stopping rules are identical. TD-InfoNCE's executable
recommends 1M for these Fetch tasks; GCSL's environment defaults specify 200k
for FourRooms and 1M for Push/Door. The long CRL navigation budget is intentional.
Episode-granular collection can exceed a requested budget by less than one
episode. C-Learning uses the released loader's 151-step limit for Push/Window;
the paper describes these as 150-step tasks.

The candidate manifest is
`configs/baseline_paper_validation_4alg_3task_1seed.tsv`.
Only tasks with a current successful preflight are appended to
`runs/qrl_queue/tasks.tsv`. Task IDs start with `paper_validation_v1_`; old task
IDs, statuses, checkpoints, and paper results are not overwritten. Appending is
idempotent and rejects an existing ID with different settings.

## Quick CRL Diagnostic

`configs/crl_quick_validation_1task_100k.tsv` adds one independent task,
`crl_quick_v1_CRL_CRLPointSmall_s1000`, without replacing any original-suite task.
It loads `PointEnv(walls='Small')` directly from the pinned CRL reference's
`contrastive_rl/point_env.py`. This is the upstream `point_Small` environment:
a 4-by-4 open map, two state coordinates, two continuous actions, and a 50-step
horizon. Its NumPy dynamics need no MuJoCo simulation. It is an additional
implementation diagnostic, not a claim to reproduce a paper figure.

The budget is **100,000 environment steps**, 1/100 of CRLAntUMaze's budget.
Checkpoints are evaluated every 5,000 steps on the same 50 validation seeds;
the selected checkpoint is tested on 200 disjoint seeds. Training seed 1000,
batch size 256, the two 256-wide hidden layers, learning rates, 10,000-step
replay prefill, and one update per environment step retain existing CRL
defaults. Consequently the first checkpoint is after the replay prefill.

The upstream reward reports distance below 2.0, which is too permissive for
diagnosing learning on this small map. Dynamics and rewards are retained;
the adapter's diagnostic `succ_rate` and `final_success_rate` use distance
below **0.25**. Check the final success rate and `final_goal_distance` as well
as any-time success. `epi_return` still reflects the upstream 2.0 threshold.
The stricter success metric is not an extra reward or a change to CRL's loss.
Learning should improve final distance and success relative to the initial
policy; a high any-time success rate alone does not establish correctness.

Generate, preflight, and append only this task with:

```bash
.venv/bin/python tools/generate_baseline_paper_validation_tasks.py --suite crl-quick
.venv/bin/python tools/check_baseline_paper_validation.py --suite crl-quick
.venv/bin/python tools/generate_baseline_paper_validation_tasks.py --suite crl-quick --append-to runs/qrl_queue/tasks.tsv
```

The preflight report is `runs/baseline_paper_validation/crl_quick_preflight.json`.
The existing scheduler reloads the queue automatically. These commands do not
start another scheduler or stop the existing AntUMaze task.

## What Was and Was Not Changed

GCSL now accepts native `Discrete` actions: categorical log likelihood for
training and integer argmax actions for collection/evaluation. Box policies keep
their existing implementation. Claw selects the existing factorized categorical
policy through a task override. Other baseline losses, initialization, target
updates and replay sampling remain as before. The new C-Learning discount/odds
overrides apply only to Jaco, Finger and Hand. Existing 14 task rows are retained.

The repository already had uncommitted baseline changes before this task; those
were preserved. The initialization and hybrid-loss edits briefly attempted
during investigation were withdrawn following the user's scope clarification.

The additions are environment registration, upstream environment adapters,
task-only conditioning-coordinate configuration, evaluation distances,
preflight checks, and task generation. `interaction.baseline_goal_dims` is
accepted only for the new `paper_validation` environments and never for QRL.
Existing environments retain their prior coordinate resolution and metrics.

The task configurations use the baseline defaults already in this repository,
without a model-size preset: TD-InfoNCE has four 512-wide hidden layers, CRL and
C-Learning have two 256-wide layers, and GCSL uses 400/300. Batch size is 256.
Baseline learning rates, exploration/prefill rules, and update/data ratios are
not overridden. The task-specific action and discount/odds overrides are listed
above. Collection cycles use one complete episode. This remains the
local trainer, not the upstream distributed Acme or TF-Agents training driver.

## Environment Provenance

The source repositories below are checked against their pinned Git revisions
and must have no tracked modifications. They are read, not edited or installed.
The default root is `/home/zhangcheng/gcrl-reference-repos`; override with
`QRL_PAPER_REFERENCE_ROOT` when moving the suite.

| Repository | Revision | Relevant source |
| --- | --- | --- |
| td-infonce-official | 18f4e7e5872da9c3653f57661d01b4fbce85b50e | envs/fetch_envs.py; lp_td_infonce.py; td_infonce/learning.py |
| google-research-gcrl | ec7c3d346277b737bc2decffcd1b533d4b7ec105 | contrastive_rl/ant_env.py; env_utils.py; lp_contrastive.py; c_learning/ |
| gcsl-official | cfae5609cee79e5a2228fb7653451023c41a64cb | gcsl/envs/; gcsl/algo/; dependencies/room_world; dependencies/multiworld |

GCSL uses the original MuJoCo FourRoom layout and Sawyer models/assets, not
Gymnasium PointMaze or MetaWorld replacements. The adapter reproduces GCSL's
separate goal-sampling reset and rollout reset. Push conditions on hand and puck
XY (four coordinates), while its evaluation distance uses puck XY. Door observes
hand XYZ and door angle, conditions on the door angle, and evaluates angle error.
Success thresholds are the upstream GCSL environment defaults (.08 for rooms,
.05 for Push/Door).

CRL's online Ant task directly loads the upstream D4RL-based wrapper, with
`non_zero_reset=True`, the original map, goal sampling, and 700-step horizon.
Its policy conditions on XY, matching `end_index=2` in the upstream executable.
It is not Gymnasium `AntMaze_UMaze-v5` and not an offline D4RL benchmark run.

The loader skips unused eager simulator imports. Plotting is imported lazily;
it is not required for state-based training. C-Learning's Gin decorators are
removed only from the loaded environment wrapper so that its environment
arguments can be supplied explicitly without importing TensorFlow/Gin. Simulator
method bodies are not rewritten.

## C-Learning Dependency and Compatibility

The original C-Learning README pins MetaWorld to
`33f3b90495be99f02a61da501d7d661e6bc675c5`. The user supplied this checkout at
`/home/zhangcheng/gcrl-reference-repos/metaworld-c-learning`; its revision and
clean working tree were verified. All three Sawyer tasks now pass the real
environment, collection, optimization, serialization, and evaluation preflight.
They have been appended to `runs/qrl_queue/tasks.tsv`, bringing this suite to
12 queued tasks (three per baseline); no scheduler was started.

When moving the checkout, export `QRL_C_LEARNING_METAWORLD` pointing to it before
both preflight and scheduler startup. A current MetaWorld release is not an
equivalent replacement. The loader verifies the revision; an archive needs to
preserve the Git checkout, not just the working files.

The old MetaWorld package automatically registers unrelated environments using
the removed Gym `tags` argument. The task loader skips only the package-level
registration entry point through namespace packages and imports the original
Sawyer classes directly. No MetaWorld source, simulator dynamics, installed Gym
version, or baseline algorithm was changed for this compatibility adjustment.

The existing local C-Learning algorithm is pure TD. The paper's Sawyer Push
experiment uses mixed TD/MC positives (Appendix F); that algorithm change is
explicitly outside the user's requested scope. Thus this candidate runs the
existing pure-TD baseline on the original **task**, not the paper's exact
Sawyer Push algorithm variant. The existing odds clipping/default optimizer
settings for those existing Sawyer tasks are also retained. The upstream training entry explicitly specifies
Glorot-uniform critic kernels, whereas the local critic keeps its existing
initialization; this observation is recorded, not fixed, and is not evidence
that it caused previous poor results.

## Evaluation and Interpretation

Each checkpoint evaluates 50 fixed validation episode seeds starting at
1,000,000. The local trainer's selected checkpoint is tested on 200 disjoint
episode seeds starting at 2,000,000. One training seed does not mean one
evaluation episode. Existing checkpoint selection remains unchanged.

For new environments, `eval.log`, `test.log`, and TensorBoard additionally record:

- `initial_goal_distance`: goal distance at s[0].
- `final_goal_distance`: goal distance at s[T], after the final action.
- `minimum_goal_distance` and `mean_goal_distance`: over s[1] through s[T].
- `final_success_rate`: success at s[T], distinct from any-time `succ_rate`.
- For GCSL only, `reference_final_goal_distance` and
  `reference_final_success_rate` use s[T-1], matching the original code's stored
  pre-action state sequence and evaluation routine.

Claw uses the original wrapped angular distance, including across the +/-pi
boundary. Other tasks use L2 over their evaluation coordinates.
Door and Claw distances are in radians; positional distances are in simulator coordinate
units. GCSL's internal reward/diagnostic info can refer to the base environment's
second-reset goal, just as in its original wrapper. Those internal rewards are
not learning inputs to GCSL and must not be used to compare commanded-goal
performance. Use the adapter's explicit goal distances instead.

Use the full validation curves and the budget-end checkpoint for comparison
with original-paper learning curves. A best-selected checkpoint test is a
different statistic. C-Learning paper metrics are distance-based; the .05
success threshold (.01 for Finger) is supplementary, not a claimed original-paper metric.
Single-seed success or failure alone cannot establish or refute implementation
correctness, nor identify a cause of the existing main-table results.

## Checks and Commands

The preflight parses each real task's Hydra configuration, collects two complete
episodes, performs three updates with the queued model and loss, checks
finite parameters/gradients and bounded actions, reloads a serialized model,
and evaluates two episodes. These smoke measurements are not experimental
results. The report records code hashes and the exact manifest; stale reports
cannot be used to append tasks after the implementation/configuration changes.

Regression checks cover native categorical training and replay serialization,
wrapped Claw distances, both DMC reset-goal protocols, physical door removal,
50-action Hand goal sampling, and coexisting MetaWorld versions. A separate
Lunar integration run completed 100 steps and resumed with its replay to 150
steps / 100 optimizer updates, preserving scalar int64 actions and writing
`paper_validation_protocol.json`. Its artifacts are under
`runs/baseline_paper_validation/integration/lunar_discrete_resume`; these short
run measurements are excluded from the scientific validation queue and audit.

```bash
cd /home/zhangcheng/quasimetric-rl
source tools/qrl_env.sh
.venv/bin/python -m unittest discover -s tests -p test_baseline_paper_validation.py
.venv/bin/python tools/check_baseline_paper_validation.py
.venv/bin/python tools/generate_baseline_paper_validation_tasks.py --append-to runs/qrl_queue/tasks.tsv
```

Preflight exits nonzero if any task is blocked. The append command may still be
run separately: it adds only the PASS subset. It never starts the scheduler.
Replay snapshots are enabled, as required by the scheduler's online-resume
contract. Only the latest full checkpoint is retained during training. The
long-budget tasks need more memory and checkpoint I/O than the existing short
main-table runs; their configured budgets are visible in the manifest above.

## Verified Papers

- [Contrastive Difference Predictive Coding](https://arxiv.org/html/2310.20141v2),
  experiments and Appendix D: state-based Fetch manipulation tasks.
- [Contrastive Learning as Goal-Conditioned Reinforcement Learning](https://arxiv.org/html/2206.07568),
  Section 5 and Appendix E: Fetch Reach/Push and online Ant navigation.
- [Learning to Reach Goals via Iterated Supervised Learning](https://arxiv.org/html/1912.06088),
  Section 5 and environment appendix: FourRooms, pushing, and door opening.
- [C-Learning: Learning to Achieve Goals via Recursive Classification](https://arxiv.org/html/2011.08909),
  Appendices F and H.3: Sawyer tasks, goal coordinates, distances, and the
  TD/MC mixture used for Sawyer Push.

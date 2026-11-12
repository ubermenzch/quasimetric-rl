# Model-size presets

Tasks select a QRL or CQRL network with one Hydra config-group override:

```text
+qrl_model_size=s
+qrl_model_size=m
+qrl_model_size=l
+qrl_model_size=xl
+qrl_model_size=xxl
+qrl_model_size=xxxl

+cqrl_model_size=s
+cqrl_model_size=m
+cqrl_model_size=m_residual
+cqrl_model_size=l
# `l_residual` remains available for checkpoints and manifests that record the
# explicit compatibility name.
+cqrl_model_size=l_residual
+cqrl_model_size=xl
+cqrl_model_size=xxl
+cqrl_model_size=xxxl

+td_infonce_model_size=m
+td_infonce_model_size=l
+crl_model_size=m
+crl_model_size=l
+scaling_crl_model_size=m
+scaling_crl_model_size=l
+scaling_crl_model_size=xl
+scaling_crl_model_size=xxl
+scaling_crl_model_size=xxxl
+gcsl_model_size=m
+gcsl_model_size=l
+gcsl_model_size=l_pusher
+gcsl_model_size=l_antnavigate
+c_learning_model_size=m
+c_learning_model_size=l
```

The selected level is recorded as `agent.model_size` in each run's
`config.yaml`. Edit the corresponding family YAML to change every future task
that selects that level. `+base_model_size=...` remains a
compatibility alias for `+qrl_model_size=...`.
Historical manifests using `+go_qrl_model_size=...` remain loadable as a
compatibility alias for `+cqrl_model_size=...`; new tasks should use CQRL only.

The four original reward-free baselines define M and L levels. These are
algorithm-specific trainable-parameter budgets, not a requirement that every
method use the same width or representation dimension. M preserves each
reference topology. At L, networks that would otherwise require extremely wide
layers are deepened to at most four hidden layers:

| Family | M hidden layers | L hidden layers | Representation | M range | L range |
| --- | --- | --- | ---: | ---: | ---: |
| TD-InfoNCE | `512 x 4` | `1192 x 4` | 16 | 3.994M-4.086M | 21.456M-21.671M |
| CRL | `1152 x 2` | `1544 x 4` | 64 | 4.170M-4.265M | 21.718M-21.845M |
| GCSL | `2304 -> 1728` | `2680 x 4` | n/a | 4.019M-4.197M | 21.603M-21.858M |
| C-Learning | `1184 x 2` | `1544 x 4` | n/a | 4.253M-4.412M | 21.526M-21.733M |

Counts include the actor and trainable critics/encoders. Original CRL uses a
fixed zero entropy coefficient, so it has no trainable entropy scalar. Counts
exclude frozen target-network copies, matching the trainable-parameter
convention used by QRL and CQRL. At M, TD-InfoNCE and C-Learning retain
roughly 7.2M and 7.1M resident parameters respectively when their frozen
targets are included. The YAML files are the source of truth for
future M- and L-level tasks. They override only the `hidden_sizes` capacity
field. Representation dimensions and every non-capacity setting come from the
corresponding reference default in `quasimetric_rl/modules/gcrl_baselines.py`
and are protected by regression tests. For L, `hidden_sizes` also raises CRL,
GCSL, and C-Learning from two to four hidden layers; M checkpoints are therefore
not shape-compatible with L. Selecting a size does not change an
optimizer, objective, batch size, replay setting, relabeling rule, or update
frequency. GCSL's output contains `3^action_dim` joint-action logits. The
general `l` preset applies to tasks with at most four action dimensions;
Pusher-v4 and AntNavigate-v4 use explicit parameter-matched presets:

| Environment | Override | Hidden sizes | Trainable parameters | CQRL-L mismatch |
| --- | --- | --- | ---: | ---: |
| Pusher-v4 | `+gcsl_model_size=l_pusher` | `2336 -> 2344 -> 2344 -> 2344` | 21,655,867 | -0.107% |
| AntNavigate-v4 | `+gcsl_model_size=l_antnavigate` | `1808 -> 1808 -> 1800 -> 1800` | 21,642,889 | -0.120% |

Both variants change only `hidden_sizes`; the GCSL objective, action
discretization, optimizer, and all training settings remain unchanged.

## Scaling-CRL scale

Scaling-CRL is a separate method family from CRL (2022). Its presets use the
actor and dual-encoder residual architecture from *1000 Layer Networks for
Self-Supervised RL*, rather than applying the original CRL loss to a larger
plain MLP. Actor, state-action encoder, and goal encoder are scaled jointly.
Each residual block contains four `Linear -> LayerNorm -> SiLU` units followed
by an identity addition; the input projection and 64-D output head sit outside
the reported logical depth.

The paper's main ten-environment sweep uses depths 4, 8, 16, 32, and 64. Its
256- and 1024-layer networks are a separate limits experiment on Humanoid
U-Maze, so they are not substituted into this multi-environment comparison.
L through XXXL use the paper's 8/16/32/64 sequence. Their near-constant widths
are solved against the nominal parameter budgets established for this scale:

| Level | Logical depth | Blocks | Width | Trainable range | Max target mismatch |
| --- | ---: | ---: | ---: | ---: | ---: |
| Scaling-CRL-M | 4 | 1 | 595 | 4.383M-4.410M | 0.283% |
| Scaling-CRL-L | 8 | 2 | 944 | 21.635M-21.678M | 0.143% |
| Scaling-CRL-XL | 16 | 4 | 916 | 40.581M-40.623M | 0.053% |
| Scaling-CRL-XXL | 32 | 8 | 901 | 78.364M-78.405M | 0.106% |
| Scaling-CRL-XXXL | 64 | 16 | 894 | 154.138M-154.179M | 0.051% |

Only `baselines.scaling_crl.hidden_sizes` appears in these YAML files. The
InfoNCE/L2 objective, LSE penalty, representation size, optimizer, entropy
learning, batch size, replay sampling, and update ratio remain owned by the
Scaling-CRL implementation. Exact per-task counts are regression-tested by
`scaling_crl_agent_parameter_count()`.

## Selection rule

The automatic QRL/CQRL selector chooses the smallest base level satisfying:

```text
latent_size / 2 > state_dim
```

| Level | Latent | Eligible state dimensions |
| --- | ---: | ---: |
| S | 128 | 1-63 |
| M | 256 | 64-127 |
| L | 512 | 128-255 |

For the currently registered online goal environments, automatic selection is:

| Level | Environments |
| --- | --- |
| S | FetchReach, FetchPush, FetchSlide, FetchPickAndPlace, Reacher-v4, Pusher-v4, DMC Reacher, DMC Manipulator |
| M | AntNavigate-v4 |
| L | None |

XL, XXL, and XXXL are explicit QRL/CQRL capacity-sweep levels and are never
chosen automatically.

## Shallow QRL and CQRL

QRL-S/M and CQRL-S/M retain their original plain ReLU MLPs and checkpoint
layouts:

| Level | State encoder | Latent | Projector | Dynamics | Actor |
| --- | --- | ---: | --- | --- | --- |
| QRL-S | `D -> 512 x 2 -> 128` | 128 | `128 -> 512 -> 2048` | `(128+A) -> 512 x 2 -> 128` | `2D -> 512 x 2 -> 2A` |
| QRL-M | `D -> 768 x 2 -> 256` | 256 | `256 -> 768 -> 2048` | `(256+A) -> 768 x 2 -> 256` | `2D -> 768 x 2 -> 2A` |

CQRL-S/M's independent goal/non-goal encoder parameter sum exactly matches
the corresponding unsplit reference encoder. Deeper constant-width branches
use the nearest realizable match. The target branch ratio is
`goal_dim:non_goal_dim`, clamped so the goal branch is never smaller than
`1:8`. The goal latent size is the effective goal share rounded upward.

## Explicit residual M alternative

`+cqrl_model_size=m_residual` selects `CQRL-M-Residual`. The original `m`
preset and automatic S/M/L selection retain their existing plain/ReLU layout.
This alternative uses the same residual/LayerNorm/SiLU construction as CQRL-L,
with one **two-layer** residual block per MLP and reference width 556. Input
projection and output Linear are additional layers outside the block; the
output remains unnormalized. Latent size is 256 and IQE remains 2048/64.

The automatic split encoder uses a 1% goal-branch budget tolerance because
integer widths have coarser parameter increments at M. It retains the 1:8
minimum goal/non-goal parameter ratio. The default tolerance for existing
presets remains 0.25%. On AntNavigate `(D=29,A=8,G=2)` the resolved modules are:

| Module | Input | Block width / depth | Output | Parameters |
| --- | ---: | --- | ---: | ---: |
| Goal encoder | 2 | 199 / 2 | 29 | 87,191 |
| Non-goal encoder | 27 | 527 / 2 | 227 | 694,286 |
| Projector | 256 | 556 / 2 | 2048 | 1,906,348 |
| Dynamics increment | 264 | 556 / 2 | 256 | 912,652 |
| Actor | 285 | 556 / 2 | 16 | 790,648 |
| IQE reduction | | | | 1 |
| **Total** | | | | **4,391,126** |

The original M has 4,396,305 parameters on this task (a 0.118% difference).
Select a fresh task ID when switching between these incompatible network
layouts. The twelve-run Min4/Min32 paired dynamics study is described in
[`docs/M_RESIDUAL_DYNAMICS.md`](../../docs/M_RESIDUAL_DYNAMICS.md).

## QRL and CQRL scale

QRL-L remains a four-layer plain ReLU MLP at width 1184. CQRL-L now uses the
four-layer residual MLP design at width 1024, matching the original residual
CQRL-L checkpoint family. Both use a latent size of 512. QRL/CQRL-XL and
above use the residual MLP design following
[1000 Layer Networks for Self-Supervised RL](https://arxiv.org/abs/2503.14858)
and its [official implementation](https://github.com/wang-kevin3290/scaling-crl).
One residual block contains four repetitions of
`Dense(width) -> LayerNorm -> SiLU`, followed by the identity addition. An
input projection and output head sit outside the blocks, and are not included
in the reported logical depth. Linear layers use the official code's
`variance_scaling(1/3, fan_in, uniform)` initialization and zero biases.
The final output is a bare Linear layer: no LayerNorm, activation, or L2
normalization is applied to the resulting latent or module output.

The state encoder or CQRL encoder branches, projector, latent dynamics, and
actor are all scaled together. The latent dynamics also keeps the existing outer
transition residual
`z_next = z + delta`; this is separate from the new internal residual blocks.
QRL retains its standard state encoder and raw `(state, goal)` actor input;
CQRL retains its split encoder and split-latent actor input.

| Level | Logical depth | Blocks | Width | Latent | Projector output | IQE |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| QRL/CQRL-M | 2 plain | 0 | 768 | 256 | 2048 | 2048 / 64 |
| QRL-L | 4 plain | 0 | 1184 | 512 | 2048 | 2048 / 64 |
| CQRL-L | 4 | 1 | 1024 | 512 | 2048 | 2048 / 64 |
| QRL/CQRL-XL | 8 | 2 | 1024 | 512 | 4096 | 4096 / 128 |
| QRL/CQRL-XXL | 16 | 4 | 1024 | 512 | 8192 | 8192 / 256 |
| QRL/CQRL-XXXL | 32 | 8 | 1024 | 512 | 16384 | 16384 / 512 |

M-to-L increases latent size, depth, and width. QRL-L is widened to 1184 while
CQRL-L uses the 1024-wide residual schedule; both remain near the nominal
21.7M L-level trainable-parameter budget. Residual-network engineering starts
at L for CQRL and at XL for QRL. From L upward, the IQE projection dimension
and component count follow the existing scale, and every IQE component remains
32-dimensional. CQRL's deep
constant-width branches cannot generally hit the unsplit reference encoder
budget exactly; the matcher selects the nearest realizable goal/non-goal widths
while preserving the branch budget ratio. Across the five L task shapes below,
total encoder mismatch is at most 0.224%.

The former two-layer QRL-L layout and any old CQRL-L plain (1160-wide)
checkpoints are incompatible with the current four-layer presets and must not
be resumed. QRL/CQRL-S/M remain checkpoint-compatible.

The explicit residual CQRL-L compatibility preset remains available:

```text
+cqrl_model_size=l_residual
```

It resolves to the same `1024 x 4` ResidualMLP, LayerNorm, SiLU, and parameter
layout as the default `l` preset. The old plain and current residual L
checkpoints are not shape-compatible because their internal module layouts
differ; use a new task ID/output directory after switching architectures.

## Deep QRL parameter counts

QRL-L is width-matched directly to the L trainable-parameter budget. QRL-XL and
above retain the residual width/depth/IQE schedule shared with CQRL.

| Task | QRL-L | QRL-XL | QRL-XXL | QRL-XXXL |
| --- | ---: | ---: | ---: | ---: |
| Manipulator | 21,853,803 | 40,099,851 | 77,950,987 | 153,653,259 |
| FetchSlide | 21,796,969 | 40,050,697 | 77,901,833 | 153,604,105 |
| Swimmer6 | 21,772,107 | 40,029,195 | 77,880,331 | 153,582,603 |
| Pusher-v4 | 21,789,871 | 40,044,559 | 77,895,695 | 153,597,967 |
| AntNavigate | 21,825,393 | 40,075,281 | 77,926,417 | 153,628,689 |

## Five-task parameter counts

Assumed dimensions are Manipulator `(D=40,A=5,G=2)`, FetchSlide `(25,4,3)`,
Swimmer6 `(17,5,2)`, Pusher-v4 `(20,7,3)`, and AntNavigate `(29,8,2)`.
Counts are trainable parameters. `Total` includes IQE's one trainable reduction
scalar in addition to the displayed modules.

| Task | Level | Goal enc. | Non-goal enc. | Projector | Dynamics | Actor | Total |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Manipulator | M | 91,025 | 727,919 | 1,772,288 | 988,672 | 817,930 | 4,397,835 |
| Manipulator | L | 530,427 | 4,240,295 | 6,833,152 | 5,263,872 | 4,802,570 | 21,670,317 |
| Manipulator | XL | 998,817 | 7,984,007 | 13,138,944 | 9,470,464 | 9,009,162 | 40,601,395 |
| Manipulator | XXL | 1,931,273 | 15,425,015 | 25,750,528 | 17,883,648 | 17,422,346 | 78,412,811 |
| Manipulator | XXXL | 3,796,941 | 30,370,935 | 50,973,696 | 34,710,016 | 34,248,714 | 154,100,303 |
| FetchSlide | M | 96,893 | 710,531 | 1,772,288 | 987,904 | 817,928 | 4,385,545 |
| FetchSlide | L | 571,198 | 4,187,102 | 6,833,152 | 5,262,848 | 4,805,640 | 21,659,941 |
| FetchSlide | XL | 1,075,842 | 7,884,040 | 13,138,944 | 9,469,440 | 9,012,232 | 40,580,499 |
| FetchSlide | XXL | 2,080,658 | 15,310,680 | 25,750,528 | 17,882,624 | 17,425,416 | 78,449,907 |
| FetchSlide | XXXL | 4,113,998 | 30,101,853 | 50,973,696 | 34,708,992 | 34,251,784 | 154,150,324 |
| Swimmer6 | M | 94,271 | 707,009 | 1,772,288 | 988,672 | 819,466 | 4,381,707 |
| Swimmer6 | L | 558,437 | 4,189,516 | 6,833,152 | 5,263,872 | 4,806,666 | 21,651,644 |
| Swimmer6 | XL | 1,051,783 | 7,910,071 | 13,138,944 | 9,470,464 | 9,013,258 | 40,584,521 |
| Swimmer6 | XXL | 2,045,473 | 15,336,226 | 25,750,528 | 17,883,648 | 17,426,442 | 78,442,318 |
| Swimmer6 | XXXL | 4,022,013 | 30,158,189 | 50,973,696 | 34,710,016 | 34,252,810 | 154,116,725 |
| Pusher-v4 | M | 120,534 | 683,050 | 1,772,288 | 990,208 | 828,686 | 4,394,767 |
| Pusher-v4 | L | 714,806 | 4,037,967 | 6,833,152 | 5,265,920 | 4,827,150 | 21,678,996 |
| Pusher-v4 | XL | 1,342,470 | 7,612,897 | 13,138,944 | 9,472,512 | 9,033,742 | 40,600,566 |
| Pusher-v4 | XXL | 2,612,477 | 14,764,170 | 25,750,528 | 17,885,696 | 17,446,926 | 78,459,798 |
| Pusher-v4 | XXXL | 5,140,247 | 29,036,931 | 50,973,696 | 34,712,064 | 34,273,294 | 154,136,233 |
| AntNavigate | M | 90,083 | 720,413 | 1,772,288 | 990,976 | 822,544 | 4,396,305 |
| AntNavigate | L | 530,427 | 4,229,647 | 6,833,152 | 5,266,944 | 4,808,720 | 21,668,891 |
| AntNavigate | XL | 998,817 | 7,973,370 | 13,138,944 | 9,473,536 | 9,015,312 | 40,599,980 |
| AntNavigate | XXL | 1,931,273 | 15,445,829 | 25,750,528 | 17,886,720 | 17,428,496 | 78,442,847 |
| AntNavigate | XXXL | 3,796,941 | 30,360,320 | 50,973,696 | 34,713,088 | 34,254,864 | 154,098,910 |

The resulting total ranges are 4.382M-4.398M (M), 21.652M-21.679M (L),
40.580M-40.601M (XL), 78.413M-78.460M (XXL), and
154.099M-154.150M (XXXL).

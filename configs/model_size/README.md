# Model-size presets

Tasks select a QRL or GO-QRL network with one Hydra config-group override:

```text
+qrl_model_size=s
+qrl_model_size=m
+qrl_model_size=l
+qrl_model_size=xl
+qrl_model_size=xxl
+qrl_model_size=xxxl

+go_qrl_model_size=s
+go_qrl_model_size=m
+go_qrl_model_size=l
+go_qrl_model_size=l_residual
+go_qrl_model_size=xl
+go_qrl_model_size=xxl
+go_qrl_model_size=xxxl

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
convention used by QRL and GO-QRL. At M, TD-InfoNCE and C-Learning retain
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

| Environment | Override | Hidden sizes | Trainable parameters | GO-QRL-L mismatch |
| --- | --- | --- | ---: | ---: |
| Pusher-v4 | `+gcsl_model_size=l_pusher` | `2336 -> 2344 -> 2344 -> 2344` | 21,655,867 | +0.033% |
| AntNavigate-v4 | `+gcsl_model_size=l_antnavigate` | `1808 -> 1808 -> 1800 -> 1800` | 21,642,889 | -0.002% |

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

The automatic QRL/GO-QRL selector chooses the smallest base level satisfying:

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

XL, XXL, and XXXL are explicit QRL/GO-QRL capacity-sweep levels and are never
chosen automatically.

## Shallow QRL and GO-QRL

QRL-S/M and GO-QRL-S/M retain their original plain ReLU MLPs and checkpoint
layouts:

| Level | State encoder | Latent | Projector | Dynamics | Actor |
| --- | --- | ---: | --- | --- | --- |
| QRL-S | `D -> 512 x 2 -> 128` | 128 | `128 -> 512 -> 2048` | `(128+A) -> 512 x 2 -> 128` | `2D -> 512 x 2 -> 2A` |
| QRL-M | `D -> 768 x 2 -> 256` | 256 | `256 -> 768 -> 2048` | `(256+A) -> 768 x 2 -> 256` | `2D -> 768 x 2 -> 2A` |

GO-QRL-S/M's independent goal/non-goal encoder parameter sum exactly matches
the corresponding unsplit reference encoder. Deeper constant-width branches
use the nearest realizable match. The target branch ratio is
`goal_dim:non_goal_dim`, clamped so the goal branch is never smaller than
`1:8`. The goal latent size is the effective goal share rounded upward.

## QRL and GO-QRL scale

QRL-L and GO-QRL-L use four-layer plain ReLU MLPs so the L comparison does not
confound either objective with residual connections, LayerNorm, or SiLU. Their
widths are 1184 and 1160 respectively, and both use a latent size of 512.
QRL/GO-QRL-XL and above use the residual MLP design following
[1000 Layer Networks for Self-Supervised RL](https://arxiv.org/abs/2503.14858)
and its [official implementation](https://github.com/wang-kevin3290/scaling-crl).
One residual block contains four repetitions of
`Dense(width) -> LayerNorm -> SiLU`, followed by the identity addition. An
input projection and output head sit outside the blocks, and are not included
in the reported logical depth. Linear layers use the official code's
`variance_scaling(1/3, fan_in, uniform)` initialization and zero biases.
The final output is a bare Linear layer: no LayerNorm, activation, or L2
normalization is applied to the resulting latent or module output.

The state encoder or GO-QRL encoder branches, projector, latent dynamics, and
actor are all scaled together. The latent dynamics also keeps the existing outer
transition residual
`z_next = z + delta`; this is separate from the new internal residual blocks.
QRL retains its standard state encoder and raw `(state, goal)` actor input;
GO-QRL retains its split encoder and split-latent actor input.

| Level | Logical depth | Blocks | Width | Latent | Projector output | IQE |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| QRL/GO-QRL-M | 2 plain | 0 | 768 | 256 | 2048 | 2048 / 64 |
| QRL-L | 4 plain | 0 | 1184 | 512 | 2048 | 2048 / 64 |
| GO-QRL-L | 4 plain | 0 | 1160 | 512 | 2048 | 2048 / 64 |
| QRL/GO-QRL-XL | 8 | 2 | 1024 | 512 | 4096 | 4096 / 128 |
| QRL/GO-QRL-XXL | 16 | 4 | 1024 | 512 | 8192 | 8192 / 256 |
| QRL/GO-QRL-XXXL | 32 | 8 | 1024 | 512 | 16384 | 16384 / 512 |

M-to-L increases latent size, depth, and width while retaining plain ReLU MLPs
without normalization. QRL-L is widened to 1184 and GO-QRL-L to 1160, placing
both near the nominal 21.7M L-level trainable-parameter budget.
Residual-network engineering starts at XL for both families. From
L upward, the IQE projection dimension and component count follow the existing
scale, and every IQE component remains 32-dimensional. GO-QRL's deep
constant-width branches cannot generally hit the unsplit reference encoder
budget exactly; the matcher selects the nearest realizable goal/non-goal widths
while preserving the branch budget ratio. Across the five L task shapes below,
total encoder mismatch is at most 0.122%.

Both the former two-layer QRL-L layout and the superseded residual QRL-L layout
are incompatible with this four-layer plain preset and must not be resumed.
The superseded residual GO-QRL-L layout is likewise incompatible with the new
plain layout. QRL/GO-QRL-S/M remain checkpoint-compatible.

The original residual GO-QRL-L is retained as an opt-in compatibility preset:

```text
+go_qrl_model_size=l_residual
```

It restores the original `1024 x 4` ResidualMLP, LayerNorm, SiLU, and parameter
counts without changing the default plain `l` preset. The plain and residual L
checkpoints are not shape-compatible because their internal module layouts
differ.

## Deep QRL parameter counts

QRL-L is width-matched directly to the L trainable-parameter budget. QRL-XL and
above retain the residual width/depth/IQE schedule shared with GO-QRL.

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
| Manipulator | L | 520,143 | 4,155,891 | 7,013,088 | 5,235,592 | 4,713,090 | 21,637,805 |
| Manipulator | XL | 998,817 | 7,984,007 | 13,138,944 | 9,470,464 | 9,009,162 | 40,601,395 |
| Manipulator | XXL | 1,931,273 | 15,425,015 | 25,750,528 | 17,883,648 | 17,422,346 | 78,412,811 |
| Manipulator | XXXL | 3,796,941 | 30,370,935 | 50,973,696 | 34,710,016 | 34,248,714 | 154,100,303 |
| FetchSlide | M | 96,893 | 710,531 | 1,772,288 | 987,904 | 817,928 | 4,385,545 |
| FetchSlide | L | 560,834 | 4,104,665 | 7,013,088 | 5,234,432 | 4,716,568 | 21,629,588 |
| FetchSlide | XL | 1,075,842 | 7,884,040 | 13,138,944 | 9,469,440 | 9,012,232 | 40,580,499 |
| FetchSlide | XXL | 2,080,658 | 15,310,680 | 25,750,528 | 17,882,624 | 17,425,416 | 78,449,907 |
| FetchSlide | XXXL | 4,113,998 | 30,101,853 | 50,973,696 | 34,708,992 | 34,251,784 | 154,150,324 |
| Swimmer6 | M | 94,271 | 707,009 | 1,772,288 | 988,672 | 819,466 | 4,381,707 |
| Swimmer6 | L | 547,101 | 4,105,139 | 7,013,088 | 5,235,592 | 4,717,730 | 21,618,651 |
| Swimmer6 | XL | 1,051,783 | 7,910,071 | 13,138,944 | 9,470,464 | 9,013,258 | 40,584,521 |
| Swimmer6 | XXL | 2,045,473 | 15,336,226 | 25,750,528 | 17,883,648 | 17,426,442 | 78,442,318 |
| Swimmer6 | XXXL | 4,022,013 | 30,158,189 | 50,973,696 | 34,710,016 | 34,252,810 | 154,116,725 |
| Pusher-v4 | M | 120,534 | 683,050 | 1,772,288 | 990,208 | 828,686 | 4,394,767 |
| Pusher-v4 | L | 699,356 | 3,957,510 | 7,013,088 | 5,237,912 | 4,740,934 | 21,648,801 |
| Pusher-v4 | XL | 1,342,470 | 7,612,897 | 13,138,944 | 9,472,512 | 9,033,742 | 40,600,566 |
| Pusher-v4 | XXL | 2,612,477 | 14,764,170 | 25,750,528 | 17,885,696 | 17,446,926 | 78,459,798 |
| Pusher-v4 | XXXL | 5,140,247 | 29,036,931 | 50,973,696 | 34,712,064 | 34,273,294 | 154,136,233 |
| AntNavigate | M | 90,083 | 720,413 | 1,772,288 | 990,976 | 822,544 | 4,396,305 |
| AntNavigate | L | 520,143 | 4,150,895 | 7,013,088 | 5,239,072 | 4,720,056 | 21,643,255 |
| AntNavigate | XL | 998,817 | 7,973,370 | 13,138,944 | 9,473,536 | 9,015,312 | 40,599,980 |
| AntNavigate | XXL | 1,931,273 | 15,445,829 | 25,750,528 | 17,886,720 | 17,428,496 | 78,442,847 |
| AntNavigate | XXXL | 3,796,941 | 30,360,320 | 50,973,696 | 34,713,088 | 34,254,864 | 154,098,910 |

The resulting total ranges are 4.382M-4.398M (M), 21.619M-21.649M (L),
40.580M-40.601M (XL), 78.413M-78.460M (XXL), and
154.099M-154.150M (XXXL).

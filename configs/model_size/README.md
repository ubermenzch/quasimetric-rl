# Model-size presets

Tasks select a QRL or GO-QRL network with one Hydra config-group override:

```text
+qrl_model_size=s
+qrl_model_size=m
+qrl_model_size=l

+go_qrl_model_size=s
+go_qrl_model_size=m
+go_qrl_model_size=l
+go_qrl_model_size=xl
+go_qrl_model_size=xxl
+go_qrl_model_size=xxxl

+td_infonce_model_size=m
+crl_model_size=m
+gcbc_model_size=m
+c_learning_model_size=m
```

The selected level is recorded as `agent.model_size` in each run's
`config.yaml`. Edit the corresponding YAML under `qrl/` or `go_qrl/` to change
every future task that selects that level. `+base_model_size=...` remains a
compatibility alias for `+qrl_model_size=...`.

The four reward-free baselines currently define an M level only. M is an
algorithm-specific trainable-parameter budget, not a requirement that every
method use the same depth, width, or representation dimension. This preserves
each method's structure while keeping all seven task shapes between 4.0M and
4.5M trainable parameters:

| Family | M hidden layers | Representation | Trainable range |
| --- | --- | ---: | ---: |
| TD-InfoNCE | `512 x 4` | 16 | 3.994M-4.030M |
| CRL | `1152 x 2` | 64 | 4.161M-4.214M |
| GCBC | `2304 -> 1728` | n/a | 4.019M-4.190M |
| C-Learning | `1184 x 2` | n/a | 4.253M-4.334M |

Counts include the actor, trainable critics/encoders, and CRL's trainable
entropy scalar. They exclude frozen target-network copies, matching the
trainable-parameter convention used by QRL and GO-QRL. TD-InfoNCE and
C-Learning retain roughly 7.2M and 7.1M resident parameters respectively when
their frozen targets are included. The YAML files are the source of truth for
future M-level tasks. They override hidden widths only. Representation
dimensions, layer count, and every non-capacity setting come from the
corresponding reference default in `quasimetric_rl/modules/gcrl_baselines.py`
and are protected by regression tests. Selecting M does not change an
optimizer, objective, batch size, replay setting, relabeling rule, or update
frequency.

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

XL, XXL, and XXXL are explicit GO-QRL capacity-sweep levels and are never
chosen automatically.

## QRL and legacy GO-QRL

QRL retains its original plain ReLU MLPs:

| Level | State encoder | Latent | Projector | Dynamics | Actor |
| --- | --- | ---: | --- | --- | --- |
| QRL-S | `D -> 512 x 2 -> 128` | 128 | `128 -> 512 -> 2048` | `(128+A) -> 512 x 2 -> 128` | `2D -> 512 x 2 -> 2A` |
| QRL-M | `D -> 768 x 2 -> 256` | 256 | `256 -> 768 -> 2048` | `(256+A) -> 768 x 2 -> 256` | `2D -> 768 x 2 -> 2A` |
| QRL-L | `D -> 1024 x 2 -> 512` | 512 | `512 -> 1024 -> 2048` | `(512+A) -> 1024 x 2 -> 512` | `2D -> 1024 x 2 -> 2A` |

GO-QRL-S and GO-QRL-M also retain their old plain MLPs and checkpoint layout.
Their independent goal/non-goal encoder parameter sum exactly matches the
corresponding QRL state encoder. The target branch ratio is
`goal_dim:non_goal_dim`, clamped so the goal branch is never smaller than
`1:8`. The goal latent size is the effective goal share rounded upward.

## Deep GO-QRL scale

GO-QRL-L and above follow the residual MLP design from
[1000 Layer Networks for Self-Supervised RL](https://arxiv.org/abs/2503.14858)
and its [official implementation](https://github.com/wang-kevin3290/scaling-crl).
One residual block contains four repetitions of
`Dense(width) -> LayerNorm -> SiLU`, followed by the identity addition. An
input projection and output head sit outside the blocks, and are not included
in the reported logical depth. Linear layers use the official code's
`variance_scaling(1/3, fan_in, uniform)` initialization and zero biases.
The final output is a bare Linear layer: no LayerNorm, activation, or L2
normalization is applied to the resulting latent or module output.

The encoder branches, projector, latent dynamics, and actor are all deepened.
The latent dynamics also keeps GO-QRL's existing outer transition residual
`z_next = z + delta`; this is separate from the new internal residual blocks.

| Level | Logical depth | Blocks | Width | Latent | Projector output | IQE |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| GO-QRL-M | 2 plain | 0 | 768 | 256 | 2048 | 2048 / 64 |
| GO-QRL-L | 4 | 1 | 1024 | 512 | 2048 | 2048 / 64 |
| GO-QRL-XL | 8 | 2 | 1024 | 512 | 4096 | 4096 / 128 |
| GO-QRL-XXL | 16 | 4 | 1024 | 512 | 8192 | 8192 / 256 |
| GO-QRL-XXXL | 32 | 8 | 1024 | 512 | 16384 | 16384 / 512 |

Thus M-to-L increases width, latent size, depth, and the residual-network
engineering recipe. From L upward, the logical depth doubles at every tier and
the IQE projection dimension and component count double with it. Every IQE
component remains 32-dimensional. This is a compound full-model scale rather
than a depth-only scale. Constant-width residual branches cannot generally hit
the unsplit encoder budget exactly; the matcher selects the nearest realizable
goal/non-goal widths while preserving the branch budget ratio. Across the five
task shapes below, total encoder mismatch is at most 0.224%.

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

# Model-size presets

Tasks select a QRL or GO-QRL network with one Hydra config-group override:

```text
+qrl_model_size=s
+qrl_model_size=m
+qrl_model_size=l

+go_qrl_model_size=s
+go_qrl_model_size=m
+go_qrl_model_size=l

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
dimensions, layer count, and every non-capacity setting come from the corresponding
reference default in `quasimetric_rl/modules/gcrl_baselines.py` and are
protected by regression tests. Selecting M does not change an optimizer,
objective, batch size, replay setting, relabeling rule, or update frequency.

## Selection rule

Both families choose the smallest level satisfying:

```text
latent_size / 2 > state_dim
```

| Level | Latent | Eligible state dimensions |
| --- | ---: | ---: |
| S | 128 | 1-63 |
| M | 256 | 64-127 |
| L | 512 | 128-255 |

## Shared architecture

Both families use one critic, residual MLP latent dynamics, and a fixed IQE
head with dimension 2048 and 64 components.

| Level | Projector | Latent dynamics | Actor hidden layers |
| --- | --- | --- | --- |
| S | `128 -> 512 -> 2048` | `(128+A) -> 512 -> 512 -> 128` | `512 -> 512` |
| M | `256 -> 768 -> 2048` | `(256+A) -> 768 -> 768 -> 256` | `768 -> 768` |
| L | `512 -> 1024 -> 2048` | `(512+A) -> 1024 -> 1024 -> 512` | `1024 -> 1024` |

QRL uses a Standard Encoder and raw-state Actor input:

| Level | State encoder | Actor |
| --- | --- | --- |
| QRL-S | `D -> 512 -> 512 -> 128` | `2D -> 512 -> 512 -> 2A` |
| QRL-M | `D -> 768 -> 768 -> 256` | `2D -> 768 -> 768 -> 2A` |
| QRL-L | `D -> 1024 -> 1024 -> 512` | `2D -> 1024 -> 1024 -> 2A` |

GO-QRL uses independent goal and non-goal encoders. Their parameter sum is
exactly the corresponding QRL state-encoder parameter count. The target
parameter ratio is `goal_dim:non_goal_dim`, clamped so the goal branch is never
smaller than `1:8`. Integer MLP widths use the nearest realizable ratio within
0.25% of the goal-branch budget. The goal latent size is the effective goal
share of the total latent size rounded upward, and the non-goal branch receives
the remaining latent dimensions. Its Actor receives the current full latent
plus the goal-branch latent.

For Maze2D (`D=4`, goal dimensions `G=2`, action dimensions `A=2`):

| Level | Goal encoder | Non-goal encoder | Actor input | Parameters |
| --- | --- | --- | ---: | ---: |
| GO-QRL-S | `2 -> 350 -> 396 -> 64` | `2 -> 404 -> 350 -> 64` | 192 | 2,206,469 |
| GO-QRL-M | `2 -> 518 -> 609 -> 128` | `2 -> 596 -> 543 -> 128` | 384 | 4,439,301 |
| GO-QRL-L | `2 -> 768 -> 768 -> 256` | `2 -> 768 -> 768 -> 256` | 768 | 8,146,949 |

The two GO-QRL encoder branches contain exactly 330,880, 791,296, and
1,579,520 parameters respectively, equal to QRL-S, QRL-M, and QRL-L state
encoders. GO-QRL's higher total count comes only from replacing QRL's `2D`
Actor input with `latent_size + goal_latent_size`.

For the currently registered online goal environments, automatic selection is:

| Level | Environments |
| --- | --- |
| S | FetchReach, FetchPush, FetchSlide, FetchPickAndPlace, Reacher-v4, Pusher-v4, DMC Reacher, DMC Manipulator |
| M | AntNavigate-v4 |
| L | None |

# GO-QRL online inner-loop ablations

The checked-in matrix isolates the number and direction of latent-goal inner
updates while keeping GO-QRL-M, the replay protocol, entropy objective, and all
other training settings fixed.

| Variant | Inner mode | Gradient steps |
| --- | --- | ---: |
| `GO-QRL+Inner0` | initial sampled completion | 0 |
| `GO-QRL+Min1` | minimize distance | 1 |
| `GO-QRL+Min8` | minimize distance | 8 |
| `GO-QRL+Max1` | maximize distance | 1 |
| `GO-QRL+Max8` | maximize distance | 8 |

`Inner0` is mode-independent in effect: it uses the initially sampled
non-goal latent and performs no gradient update. Its task configuration uses
`min` only to enter the same latent-completion path as the other variants.

## Environment matrix

`Inner0` runs on all seven validated environments. The full 1/8-step Min/Max
sweep runs on three representative environment families: FetchPush,
reacher_hard, and Reacher-v4. This gives 35 Inner0 tasks plus 60 gradient-step
tasks, for 95 tasks total at five seeds each.

The assignment keeps every five-seed environment/variant group on one server:

| Partition | Baseline workload | Ablation groups | Ablation tasks |
| --- | --- | --- | ---: |
| `local` | GCSL-M (35) | FetchPush all 5; reacher_hard all 5; Inner0 on FetchReach, FetchSlide, FetchPickAndPlace | 65 |
| `server_crl` | CRL-M (35) | Reacher-v4 Inner0, Min1, Min8 | 15 |
| `server_c_learning` | C-Learning-M (35) | Reacher-v4 Max1, Max8; reacher_easy Inner0 | 15 |

The source-of-truth manifests are:

```text
configs/go_qrl_inner_steps_ablation_95_7env_5seed_200k.tsv
configs/go_qrl_inner_steps_ablation_local65.tsv
configs/go_qrl_inner_steps_ablation_crl_server15.tsv
configs/go_qrl_inner_steps_ablation_c_learning_server15.tsv
```

Regenerate all four files with:

```bash
.venv/bin/python tools/generate_online_go_qrl_ablation_tasks.py \
  --write-all-partitions
```

To append a partition idempotently to a server's runtime queue:

```bash
.venv/bin/python tools/generate_online_go_qrl_ablation_tasks.py \
  --partition server_crl \
  --append-to runs/qrl_queue/tasks.tsv
```

All tasks train for 200k steps, retain 20k checkpoints, select the best model on
the validation seeds, and test that selected checkpoint on disjoint test seeds.

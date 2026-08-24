# GO-QRL inner-step ablation table

This is the full factorial ablation corresponding to the eight columns in the
comparison table. Every variant runs on every column with training seeds
`1000`-`1004`:

| Variant | Mode | Inner updates | Groups | Tasks |
| --- | --- | ---: | ---: | ---: |
| `GO-QRL+Inner0` | min | 0 | 8 | 40 |
| `GO-QRL+Min1` | min | 1 | 8 | 40 |
| `GO-QRL+Min8` | min | 8 | 8 | 40 |
| `GO-QRL+Max1` | max | 1 | 8 | 40 |
| `GO-QRL+Max8` | max | 8 | 8 | 40 |
| **Total** |  |  | **40** | **200** |

The environment columns use the following budgets and model sizes:

| Environment | Model | Training steps | Checkpoint steps |
| --- | --- | ---: | ---: |
| FetchPush | M | 100k | 20k |
| FetchSlide | L | 500k | 50k |
| FetchPickAndPlace | M | 200k | 20k |
| reacher_hard | M | 200k | 20k |
| reacher_hard | M | 100k | 20k |
| maze2d-large | M | 200k | 20k |
| Pusher-v4 | L | 500k | 50k |
| AntNavigate-v4 | L | 500k | 50k |

The source-of-truth task manifest is:

```text
configs/go_qrl_inner_steps_ablation_table_8env_5variant_5seed.tsv
configs/go_qrl_inner_steps_ablation_table_local135.tsv
configs/go_qrl_inner_steps_ablation_table_remote65.tsv
```

Regenerate all three files with:

```bash
.venv/bin/python tools/generate_online_go_qrl_ablation_tasks.py \
  --table --write-all-partitions
```

The five seeds in a table cell are indivisible. The `local_2x` partition has
27 cells/135 tasks and 38.0M cumulative environment steps; `remote_1x` has 13
cells/65 tasks and 19.5M cumulative environment steps. The partitions contain
10:5 L-500k cells and 17:8 M cells, respectively, and both cover every variant
and every environment column.

Append either partition idempotently with:

```bash
.venv/bin/python tools/generate_online_go_qrl_ablation_tasks.py \
  --table --partition local_2x --append-to runs/qrl_queue/tasks.tsv
```

The existing 95-task manifest and its partition files are unchanged. `Inner0`
uses the same `min` completion path as the other variants, but performs zero
latent-goal updates. The table suite pins batch size 256, joint training, and
the current GO-QRL hybrid latent-dynamics settings; only the latent-goal mode
and number of inner updates vary across rows.

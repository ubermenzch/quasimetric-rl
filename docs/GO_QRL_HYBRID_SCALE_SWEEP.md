# GO-QRL Hybrid model-scale sweep

This sweep contains 125 online runs:

```text
5 environments x 5 model scales x 5 training seeds = 125 runs
```

Every run trains for 500k environment steps with batch size 256. A checkpoint
and 500-episode validation evaluation are produced every 50k steps using
episode seeds 1000-1499. After training, the best validation checkpoint is
selected and evaluated for 1000 test episodes using seeds 1500-2499.

Hybrid means joint latent-dynamics training with an equally weighted IQE and
MSE loss (`distance=iqe_mse`, `iqe_weight=1`, `mse_weight=1`). The established
GO-QRL Max4 latent-goal policy settings are held fixed across model scales.

## Assignment

Five-seed `(environment, model scale)` groups are never split across servers.
The indivisible groups produce the closest count allocation to 2:1:1:

| Partition | Groups | Runs | Approximate parameter workload |
| --- | ---: | ---: | ---: |
| `server_2x` | 13 | 65 | 741 |
| `local_1x` | 6 | 30 | 378 |
| `server_1x` | 6 | 30 | 378 |

Each partition contains every environment and every model scale at least once.
The parameter workload is the sum of model parameter counts per five-seed
group and is approximately 1.96:1:1.

The scale presets use compound IQE growth from L upward: L is `2048/64`, XL
is `4096/128`, XXL is `8192/256`, and XXXL is `16384/512`. The IQE dimension
and component count grow together, keeping 32 dimensions per component.

| Partition | Assigned five-seed groups |
| --- | --- |
| `server_2x` | M: Swimmer6, Pusher-v4, AntNavigate; L: Manipulator, Pusher-v4, AntNavigate; XL: Manipulator, FetchSlide, AntNavigate; XXL: Swimmer6; XXXL: FetchSlide, Swimmer6, Pusher-v4 |
| `local_1x` | M: Manipulator; L: FetchSlide; XL: Swimmer6; XXL: Manipulator, Pusher-v4; XXXL: AntNavigate |
| `server_1x` | M: FetchSlide; L: Swimmer6; XL: Pusher-v4; XXL: FetchSlide, AntNavigate; XXXL: Manipulator |

Generate all source manifests:

```bash
.venv/bin/python tools/generate_go_qrl_hybrid_scale_tasks.py \
  --write-all-partitions
```

Append one partition idempotently to a runtime queue:

```bash
.venv/bin/python tools/generate_go_qrl_hybrid_scale_tasks.py \
  --partition local_1x \
  --append-to runs/qrl_queue/tasks.tsv
```

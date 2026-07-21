# Maze2D QRL architecture diagrams

This directory contains two editable QRL architecture diagrams:

- `qrl_1q_base_maze2d.svg`: a one-critic Maze2D Base architecture.
- `qrl_1q_split_latent_max8_maze2d.tex`: authoritative LaTeX/TikZ source for
  the existing one-critic split-latent Max8 architecture, including all critic,
  actor, and latent-completion routes.
- `qrl_1q_split_latent_max8_maze2d.svg`: directly viewable vector preview of
  the TikZ figure.

PNG previews with the same basenames are included for slides and quick viewing.

## Scope and configuration basis

Both diagrams use `maze2d-umaze-v1`, whose state is
`(x, y, vx, vy) in R^4` and whose action is in `R^2`.

The repository does not contain a completed run named `1q_Base` on Maze2D.
The Base diagram therefore uses the existing Maze2D Base network settings and
changes only `agent.num_critics` from 2 to 1. In particular, it keeps the raw
actor and the larger 1024-wide encoder, dynamics, actor, and projector MLPs.

The SplitLatentMax8 diagram uses the split-encoder dimensions from
`configs/qrl_tasks_latent_completion_offline.tsv` and the implemented paths in:

- `quasimetric_rl/modules/quasimetric_critic/models/encoder.py`
- `quasimetric_rl/modules/quasimetric_critic/models/latent_dynamics.py`
- `quasimetric_rl/modules/quasimetric_critic/losses/global_push.py`
- `quasimetric_rl/modules/quasimetric_critic/losses/local_constraint.py`
- `quasimetric_rl/modules/quasimetric_critic/losses/latent_dynamics.py`
- `quasimetric_rl/modules/actor/model.py`
- `quasimetric_rl/modules/actor/losses/min_dist.py`

The detailed figure separates the three critic losses: global push, local
constraint, and bidirectional IQE latent-dynamics loss. The completion loop
appears only in the actor-training objective. Policy
inference passes the full state latent `[E_G(s), E_N(s)]` and the goal branch
`E_G(g)` to the actor; it does not append a zero non-goal branch or run eight
Adam steps. Both split branches use RMSNorm without affine parameters. The
latent-completion loop initializes its non-goal variable from normalized
`E_N(g)` of that same sampled goal, normalizes every candidate before IQE, and
uses the per-sample best of steps 0 through 8. Critic training still encodes
complete states with both split-encoder branches.

An alternative bounded-residual experiment is defined in
`configs/qrl_tasks_split_latent_max8_bounded_residual_matched_1q_base_offline.tsv`.
It disables branch RMSNorm and searches in batch-standardized non-goal
coordinates. For the batch per-dimension standard deviation `sigma_N`, its
candidate is `h = h0 + sigma_N * r`; projected Adam enforces
`sqrt(mean(r^2)) <= rho` independently for every sample after each update. The
provided `BoundedResR1` tasks use `rho=1`, a standardized-residual learning rate
of `0.1`, seeds 1000 through 1009, and retain the per-sample best candidate
among `h0...h8`. Dimensions
whose batch standard deviation is zero use `latent_goal_eps` as the scale floor.

All mathematical expressions in the authoritative `.tex` source use LaTeX
math mode, including subscripts, superscripts, hats, gradients, and stop-gradient
operators. The SVG preview uses visually typeset Unicode equivalents so it can
be viewed without a TeX installation.

## Regeneration

Generate the dependency-free SVG files with the repository environment:

```bash
.venv/bin/python docs/architecture/generate_qrl_architectures.py
```

On this machine, render the PNG previews with the system Python bindings for
librsvg and Cairo:

```bash
python3 docs/architecture/render_qrl_architectures.py
```

Compile the authoritative TikZ figure on a machine with TeX Live, or upload it
to Overleaf:

```bash
cd docs/architecture
pdflatex qrl_1q_split_latent_max8_maze2d.tex
```

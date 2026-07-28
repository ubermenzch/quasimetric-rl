# Online GCRL baselines

The online trainer supports four reward-free goal-conditioned baselines on the
seven validated environments in `ONLINE_GOAL_ENVS.md`:

| CLI value | Method | Implementation source |
| --- | --- | --- |
| `td_infonce` | TD-InfoNCE | `chongyi-zheng/td_infonce`, commit `18f4e7e5` |
| `crl` | CRL | JaxGCRL CRL, commit `7c53a074`; original Google CRL also inspected at commit `ec7c3d34` |
| `gcbc` | GCSL/GCBC | `dibyaghosh/gcsl`, commit `cfae5609` |
| `c_learning` | C-Learning | `google-research/google-research/c_learning`, commit `ec7c3d34` |

These agents do not optimize the task reward. C-Learning constructs its binary
next-goal labels internally, and the other methods learn from state/action
trajectories and hindsight goals. Environment rewards are still recorded by the
shared replay and evaluation code for reporting only.

## Fidelity choices

TD-InfoNCE uses twin contrastive critics, the two-term discounted TD-InfoNCE
target, target-network soft updates, random actor goals, and the reference
online architecture and learning rates.

`crl` defaults to the stronger current JaxGCRL configuration: geometrically
sampled future goals, forward InfoNCE, negative Euclidean-norm energy,
log-sum-exp regularization, SiLU networks, and adaptive policy entropy. The
configured CRL discount is also used by replay's geometric future-goal sampler.
The network depth and width are configurable, so deeper CRL variants can be
added without changing the trainer. To reproduce the older vector-observation
CRL objective, override:

```bash
agent.baselines.crl.contrastive_loss=binary_nce \
agent.baselines.crl.energy=dot \
agent.baselines.crl.logsumexp_penalty=0 \
agent.baselines.crl.activation=relu \
agent.baselines.crl.entropy_coefficient=0 \
agent.baselines.crl.random_goal_fraction=0.5
```

`gcbc` keeps GCSL's uniform ordered state/future-goal pair sampler and
conditional negative log likelihood. The original repository discretizes
continuous actions; this implementation uses the repository's existing bounded
tanh-Gaussian continuous policy so every baseline shares the same physical
action interface. `gcsl` is accepted as an alias for `gcbc`.

C-Learning uses twin sigmoid classifiers, half next-state positives and half
shuffled random goals, clipped target odds, recursive classifier targets, and
Polyak target updates from the reference implementation.

## Run one experiment

```bash
source tools/qrl_env.sh
.venv/bin/python -m online.main \
  env.kind=gcrl \
  env.name=FetchPush \
  agent.algorithm=td_infonce \
  seed=1000 \
  interaction.total_env_steps=200000 \
  interaction.exploration_eps=0 \
  interaction.num_eval_episodes=200 \
  interaction.num_test_episodes=200 \
  interaction.validation_seed=1000 \
  interaction.test_seed=2000000 \
  eval_steps=null \
  save_steps=20000 \
  keep_only_latest_checkpoint=false \
  save_replay_buffer=true
```

Use `crl`, `gcbc`, or `c_learning` for the other methods. As with QRL and
GO-QRL, every 20k boundary is evaluated on the validation seeds and saved with
the replay and optimizer state. Training then reloads the checkpoint with the
best validation success, hitting time, and return tie-breakers, and evaluates
that model once on the disjoint test seeds. The selected artifact is
`selected_best_agent.pth`, with provenance in `best_checkpoint.json`.

## Seven-environment matrix

The checked-in task file contains four algorithms, seven environments, five
training seeds, and 200k environment steps, for 140 tasks total:

```bash
.venv/bin/python tools/generate_online_baseline_tasks.py
```

This writes `configs/gcrl_baselines_4alg_7env_5seed_200k.tsv`. Before submitting
the matrix, the real simulator/update smoke test can be rerun with:

```bash
source tools/qrl_env.sh
.venv/bin/python tools/smoke_test_gcrl_baselines.py
```

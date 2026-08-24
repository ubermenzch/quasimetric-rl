# Additional online goal environments

The DMC, Shadow Hand, Gymnasium maze, and Panda-Gym extension is catalogued in
[`EXTENDED_ONLINE_TASKS.md`](EXTENDED_ONLINE_TASKS.md). This document retains
the original online protocol and its previously generated experiment matrices.

For the extended DMC adapters, point-mass and dog fetch use fixed backend
world-frame targets, and ball-in-cup uses a fixed zero goal in the moving cup
frame. Stacker selects the box nearest the fixed episode target as its achieved
goal and uses repository-defined `reward >= 0.95` success semantics because
dm_control does not expose a native binary success signal for that task.

The online replay collector requires fixed-length episodes and dictionary
observations with same-shaped `observation`, `achieved_goal`, and
`desired_goal` tensors. The environments below provide that interface and can
be used by both QRL and vector-observation CQRL agents.

| Kind | Name | Horizon | Success dimensions | Backend |
| --- | --- | ---: | --- | --- |
| `gcrl` | `FetchReach` | 50 | `[0,1,2]` | Gym Fetch robotics |
| `gcrl` | `FetchPush` | 50 | `[3,4,5]` | Gym Fetch robotics |
| `gcrl` | `FetchSlide` | 50 | `[3,4,5]` | Gym Fetch robotics |
| `gcrl` | `FetchPickAndPlace` | 50 | `[3,4,5]` | Gym Fetch robotics |
| `gym_mujoco` | `Reacher-v4` | 50 | `[0,1]` | Pinned local Gym `Reacher-v2` backend |
| `gym_mujoco` | `Pusher-v4` | 100 | `[0,1,2]` | Pinned local Gym `Pusher-v2` backend |
| `gym_mujoco` | `AntNavigate-v4` | 1000 | `[0,1]` | Pinned local Gym `Ant-v3` backend |
| `dmc` | `reacher_easy` | 1000 | `[0,1]` | dm_control |
| `dmc` | `reacher_hard` | 1000 | `[0,1]` | dm_control |
| `dmc` | `swimmer6` | 1000 | `[0,1]` | dm_control |
| `dmc` | `swimmer15` | 1000 | `[0,1]` | dm_control |
| `dmc` | `quadruped_fetch` | 1000 | `[0,1]` | dm_control |
| `dmc` | `manipulator_bring_ball` | 1000 | `[0,1]` | dm_control |
| `dmc` | `manipulator_bring_peg` | 1000 | `[0,1,2,3]` | dm_control |
| `online_maze` | `maze2d-medium` | 600 | `[0,1]` | D4RL Maze2D v1 |
| `online_maze` | `maze2d-large` | 800 | `[0,1]` | D4RL Maze2D v1 |

These dimensions define task success and GO-QRL's task-goal branch. They are
not a universal conditioning projection for the reference baselines. In the
three Fetch object-manipulation tasks, TD-InfoNCE and C-Learning condition on the
full state, and GCSL conditions on gripper plus object position. Original
2022 CRL conditions on the full state in every validated environment; see
[`GCRL_BASELINES.md`](GCRL_BASELINES.md).

`AntNavigate-v4` is deliberately a separate name. It samples a reachable XY
navigation goal and does not optimize the standard Ant forward-velocity
reward. Its 29-dimensional state is `qpos + qvel`; backend contact-force cache
entries are excluded.

## QRL

```bash
source tools/qrl_env.sh
.venv/bin/python -m online.main \
  env.kind=dmc \
  env.name=reacher_easy \
  +qrl_model_size=s \
  interaction.total_env_steps=200000 \
  interaction.exploration_eps=0 \
  save_steps=20000 \
  keep_only_latest_checkpoint=false
```

## GO-QRL

The online entry point automatically obtains `encoder.goal_dims` from the
selected environment. The following runs GO-QRL+Max4+LN+RMSG with four steps
of RMS-normalized latent-goal gradient ascent:

```bash
source tools/qrl_env.sh
.venv/bin/python -m online.main \
  env.kind=dmc \
  env.name=reacher_easy \
  +go_qrl_model_size=m \
  agent.quasimetric_critic.model.encoder.branch_normalization=layernorm \
  agent.actor.losses.min_dist.latent_goal_mode=max \
  agent.actor.losses.min_dist.latent_goal_steps=4 \
  agent.actor.losses.min_dist.latent_goal_keep_best=true \
  agent.actor.losses.min_dist.latent_goal_optim=rmsg \
  agent.actor.losses.min_dist.latent_goal_search=direct \
  agent.actor.losses.min_dist.latent_goal_lr=0.01 \
  interaction.total_env_steps=200000 \
  interaction.exploration_eps=0 \
  interaction.num_eval_episodes=100 \
  interaction.num_test_episodes=100 \
  interaction.validation_seed=1000 \
  save_steps=20000 \
  save_replay_buffer=true \
  keep_only_latest_checkpoint=false
```

Use `agent.actor.losses.min_dist.latent_goal_mode=min` for
GO-QRL+Min4+LN+RMSG. Basic GO-QRL uses `branch_normalization=none` and
`latent_goal_optim=sgd`; no current scheme uses residual search. An explicit
`encoder.goal_dims` remains available as an override.

## Interaction schedule

The default online schedule is transition-based, so longer episodes do not
consume the entire training budget during random prefill. It resolves as:

| Horizon | Prefill episodes | Rollouts/cycle | Optimizations/cycle |
| ---: | ---: | ---: | ---: |
| 50 | 200 | 10 | 500 |
| 100 | 100 | 5 | 500 |
| 1000 | 10 | 1 | 1000 |

Validation and final test always use the same episode count. The already
running FetchReach tasks stay at 1000 episodes; the other 50/100-step
environments use 200, and the 1000-step environments use 100.

Every 20k-step checkpoint is evaluated with the actor distribution mean. Each
episode is reset independently from validation seed 1000 through 1999, 1199,
or 1099, depending on the environment.
The best checkpoint is selected by validation
success count, then by lower mean first-success step (failures are assigned
`horizon + 1`), then by higher mean episode return. Only that selected model is
evaluated on the independent test seed. The run writes checkpoint summaries to
`eval.log`, the final test summary to `test.log`, and the complete selection
record to `best_checkpoint.json`.

Set `keep_only_best_and_final_checkpoints=true` to apply compact checkpoint
retention throughout training. Each checkpoint-bound validation is compared
with the incumbent using the documented selection order. The fixed
`selected_best_agent.pth` file is created for the first validation and is
atomically replaced only when a later checkpoint is better; per-validation
`agent_checkpoint_*.pth` files are not created. Validation metrics for every
checkpoint remain in `eval.log`, TensorBoard, and resumable checkpoint metadata.
After validation selection and final testing complete, the run keeps only:

- `selected_best_agent.pth`, an inference/deployment checkpoint containing the
  selected agent weights and their validation provenance;
- `checkpoint_*_final.pth`, a full resumable checkpoint containing the final
  training weights, optimizer/loss state, RNG state, loop cursor, and replay
  buffer.

The final replay buffer is included whenever this flag is enabled, even if
`save_final_replay_buffer=false`. Logs, configuration, TensorBoard events, and
`best_checkpoint.json` are not removed. Cleanup runs before `COMPLETE` is
written, so interrupted runs retain a resumable final checkpoint. Use
`keep_only_latest_checkpoint=true` as well to retain only the latest resumable
checkpoint during training; otherwise compact retention still keeps all
periodic full checkpoints until successful final cleanup.

Set any of `interaction.num_prefill_episodes`,
`interaction.num_rollouts_per_cycle`, `interaction.num_samples_per_cycle`, or
`interaction.num_eval_episodes` to override the corresponding derived value.

Queue tasks still use `mode=online`. Since the queue defaults online tasks to
`gcrl`, add the effective kind at the end of `extra_args`, for example
`env.kind=dmc` or `env.kind=gym_mujoco`.

The complete 5-scheme x 5-seed x 11-environment M-size matrix is generated in
`configs/qrl_tasks_online_goal_envs_5variants_5seeds_200k.tsv` and appended to
the active queue by `tools/generate_online_goal_env_tasks.py`. The schemes are
Base, GO-QRL+Max4, GO-QRL+Min4, GO-QRL+Max4+LN+RMSG, and
GO-QRL+Min4+LN+RMSG.

TD-InfoNCE, CRL, GCSL, and C-Learning use the same online collection,
checkpoint validation, and best-checkpoint test protocol. Their implementation
and seven-environment task matrix are documented in
[`GCRL_BASELINES.md`](GCRL_BASELINES.md).

## Continue a completed run

Stop the queue scheduler, then continue any completed online task in place by
giving a new absolute total. For example, this resumes the training-end 200k
checkpoint (including replay, optimizer, scheduler progress, and RNG) to 300k:

```bash
tools/stop_qrl_queue.sh
.venv/bin/python tools/continue_online_task.py TASK_ID \
  --total-env-steps 300000 \
  --gpu 0
```

The continuation helper holds the queue lock for the whole run. Online training
removes the old `COMPLETE` marker only after validating the resume checkpoint,
and writes it again after the extended validation and test phases finish. The
same command can later target 400k or any larger absolute step count.

Install or repair the DMC and Gym MuJoCo dependencies in an existing project
environment with:

```bash
tools/setup_environment.sh --simulators-only
```

This also smoke-tests every DMC and Gym MuJoCo environment listed above. The
requirements intentionally match the source server: `dm-control==1.0.3` and
the Gym 0.18 legacy backends (`Reacher-v2`, `Pusher-v2`, and `Ant-v3`) behind
the task-facing names in this table.

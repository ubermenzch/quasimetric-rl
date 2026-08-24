# Curated extended online goal tasks

The extended suite contains 34 goal-conditioned online environments for QRL,
CQRL, and the supported baselines. Variants that differ mainly in difficulty
are deliberately pruned: easy is preferred to hard, Medium to Large, and
`stack_2` to `stack_4`. Different objects, control modes, and sensor modalities
remain because they test distinct partial-goal structures.

Every adapter returns fixed-length legacy Gym API episodes and same-shaped
`observation`, `achieved_goal`, and `desired_goal` vectors. Task coordinates
form a prefix of the full state and are registered in
`GOAL_SET_DIMS_REGISTRY`.

## Environment catalog

### dm_control (7)

| Name | State | Action | Goal | Horizon |
| --- | ---: | ---: | ---: | ---: |
| `point_mass_easy` | 4 | 2 | 2 | 1000 |
| `finger_turn_easy` | 9 | 2 | 2 | 1000 |
| `manipulator_insert_ball` | 40 | 5 | 2 | 1000 |
| `manipulator_insert_peg` | 40 | 5 | 4 | 1000 |
| `dog_fetch` | 210 | 38 | 3 | 1000 |
| `stacker_stack_2` | 49 | 5 | 2 | 1000 |
| `ball_in_cup_catch` | 8 | 2 | 2 | 1000 |

Use `env.kind=dmc`. The adapter exposes native task geometry as the goal and
retains the remaining Markov state. `point_mass_easy` and `dog_fetch` use the
fixed world-frame targets supplied by dm_control. For `dog_fetch`, the ball XYZ
goal prefix replaces the duplicated translation in the `ball_root` free joint
while retaining its quaternion and the rest of qpos, qvel, and activations.

`ball_in_cup_catch` uses ball displacement in the moving cup-target frame, so
the desired goal is zero. Stacker uses the XY position of the box nearest the
fixed target. Because stacker has no native binary success API, this repository
defines success as `reward >= 0.95`.

### Gymnasium-Robotics Shadow Hand (10)

| Name | State | Action | Goal | Horizon |
| --- | ---: | ---: | ---: | ---: |
| `HandReach-v2` | 63 | 20 | 15 | 50 |
| `HandManipulateBlockRotateZ-v1` | 66 | 20 | 9 | 100 |
| `HandManipulateEggRotate-v1` | 66 | 20 | 9 | 100 |
| `HandManipulatePenRotate-v1` | 66 | 20 | 9 | 100 |

Each of the three manipulation tasks also has both official sensor variants:

- `<stem>_BooleanTouchSensors-v1`
- `<stem>_ContinuousTouchSensors-v1`

These six touch tasks have state size 158, action size 20, goal size 9, and
horizon 100. They are retained because the 92 touch channels enlarge only the
non-goal state, directly testing whether CQRL handles a fixed goal geometry as
irrelevant state grows. They are not difficulty duplicates.

Use `env.kind=gymnasium_robotics`. Quaternion goals are mapped to a
sign-invariant 3x3 rotation matrix. The adapter removes fingertip coordinates
48--62 from HandReach and native quaternion coordinates 57--60 from each
rotation task. Object position and all touch channels remain non-goal state.
`HandReach-v2` never silently falls back to a later environment version.

### Gymnasium-Robotics mazes (5)

| Name | State | Action | Goal | Horizon |
| --- | ---: | ---: | ---: | ---: |
| `PointMaze_UMaze-v3` | 4 | 2 | 2 | 300 |
| `PointMaze_Open-v3` | 4 | 2 | 2 | 300 |
| `PointMaze_Medium-v3` | 4 | 2 | 2 | 600 |
| `AntMaze_UMaze-v5` | 107 | 8 | 2 | 700 |
| `AntMaze_BigMaze_DGR-v5` | 107 | 8 | 2 | 1000 |

`PointMaze_Large-v3` is omitted in favor of Medium. From the closely related
Big/Hardest fixed-goal, DG, and DGR AntMaze variants, the suite retains only
`AntMaze_BigMaze_DGR-v5`; it combines obstacles, randomized starts and goals,
and a large non-goal body state. The public name can fall back to the pinned
package's `AntMaze_Medium_Diverse_GR-v5` registry ID. Goals stay fixed within
an episode.

### Panda-Gym (12)

| Task | EE state/action | Joints state/action | Goal | Horizon |
| --- | ---: | ---: | ---: | ---: |
| Reach | 6 / 3 | 20 / 7 | 3 | 50 |
| Push | 18 / 3 | 32 / 7 | 3 | 50 |
| Slide | 18 / 3 | 32 / 7 | 3 | 50 |
| PickAndPlace | 19 / 4 | 37 / 8 | 3 | 50 |
| Stack | 31 / 4 | 49 / 8 | 6 | 100 |
| Flip | 20 / 4 | 38 / 8 | 4 | 50 |

Use `env.kind=panda_gym` with `Panda<Task>-v3` for end-effector control and
`Panda<Task>Joints-v3` for joint control. Both versions remain because control
mode changes the state/action interface rather than only difficulty. The
adapter moves achieved task coordinates to the goal prefix and removes their
native duplicates from non-goal state. Flip canonicalizes equivalent
quaternions and uses Panda-Gym's sign-invariant angular metric.
Panda-Gym 3.0.7's wheel omits the visual-only `colored_cube.png` used by Flip;
the adapter falls back to PyBullet's packaged `colors16.png` texture. This
preserves a multi-color orientation cue without changing task physics, vector
observations, goals, rewards, or success evaluation.

## Curated total

| Family | Environments |
| --- | ---: |
| dm_control | 7 |
| Shadow Hand without touch | 4 |
| Shadow Hand with touch | 6 |
| Maze | 5 |
| Panda end-effector control | 6 |
| Panda joint control | 6 |
| **Total** | **34** |

## Installation and verification

The pinned stack keeps legacy `gym==0.18.0` for existing environments and
installs Gymnasium separately for the new adapters:

```bash
tools/setup_environment.sh --simulators-only
source tools/qrl_env.sh
.venv/bin/python tools/verify_online_simulators.py
```

Individual families can be checked with `--suite dmc`,
`--suite gymnasium-robotics`, or `--suite panda-gym`. Optional simulator
packages are imported lazily, so configuration and unit tests remain usable
when they are absent.

## Training matrix

`tools/generate_extended_online_tasks.py` covers QRL, TD-InfoNCE, GCSL,
C-Learning, CRL, and CQRL/Inner0, Inner1, Inner4, and Inner8. Its default
protocol selects one primary capacity and budget per environment:

| Native horizon | Default capacity | Requested steps | Checkpoint interval |
| ---: | ---: | ---: | ---: |
| up to 100 | M | 100k | 20k |
| 101--999 | M | 200k | 20k |
| 1000 or longer | L | 500k | 50k |

The collector stores complete fixed-length episodes, so requested steps are
rounded up to a horizon multiple. For example, PointMaze with horizon 300 uses
200100 steps, while AntMaze UMaze with horizon 700 uses 200200 steps. Each row
sizes its replay allocation to its actual interaction budget.

The default matrix has `34 environments x 9 methods x 5 seeds = 1530` rows:

```bash
.venv/bin/python tools/generate_extended_online_tasks.py \
  --output configs/extended_online_9alg_34env_default_5seed.tsv
```

Filters support smaller pilots. For example:

```bash
.venv/bin/python tools/generate_extended_online_tasks.py \
  --families dmc maze \
  --algorithms qrl cqrl_inner0 cqrl_inner1 cqrl_inner4 cqrl_inner8 \
  --model-sizes m \
  --seeds 1000 1001 1002 1003 1004 \
  --output configs/extended_navigation_cqrl_m.tsv
```

GCSL's original joint categorical head has `3 ** action_dim` logits. For
action dimensions 7 and above, the generator uses independent three-bin heads
per action dimension and marks task IDs with `factorized`. This protocol
difference must be disclosed in comparisons.

The `params` column records the actual trainable count. M/L are capacity
protocols, not claims of identical integer parameter counts across methods.
Passing `--model-sizes m l` generates both capacities at the environment's
default interaction budget: `34 x 9 x 2 x 5 = 3060` rows.

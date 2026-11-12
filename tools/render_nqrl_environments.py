#!/usr/bin/env python3
"""Render real task scenes, including one scripted grasp-and-lift frame."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analyze_nqrl_paper_results import ENVIRONMENTS

OUTPUT = ROOT / "iclr2027/figures/environments"
WIDTH, HEIGHT = 768, 432


def legacy_frame(backend, *, lookat=None, distance=None, elevation=-45, azimuth=90, goal=None):
    import mujoco_py

    backend.render(mode="rgb_array", width=WIDTH, height=HEIGHT)
    viewer = backend._get_viewer("rgb_array")
    viewer.cam.type = mujoco_py.const.CAMERA_FREE
    viewer.cam.trackbodyid = -1
    if lookat is not None:
        viewer.cam.lookat[:] = lookat
    if distance is not None:
        viewer.cam.distance = distance
    viewer.cam.elevation = elevation
    viewer.cam.azimuth = azimuth
    if goal is not None:
        viewer.add_marker(
            pos=np.r_[goal, .035], size=np.array([.35, .35, .025]),
            rgba=np.array([.15, .75, .30, .9]), type=mujoco_py.const.GEOM_CYLINDER,
            label="",
        )
    viewer.render(WIDTH, HEIGHT, camera_id=-1)
    return viewer.read_pixels(WIDTH, HEIGHT, depth=False)[::-1]


def fetch_lift_frame_state(env, observation):
    """Move the native gripper through contact; never reposition simulator bodies."""
    start = np.asarray(observation["observation"])[3:6].copy()
    actions, stages = [], []
    for name, offset, gripper, steps in (
        ("approach", .10, 1., 12),
        ("descend", .005, 1., 12),
        ("grasp", .005, -1., 8),
        ("lift", .22, -1., 16),
    ):
        target = start + [0., 0., offset]
        for _ in range(steps):
            current = np.asarray(observation["observation"])[:3]
            action = np.r_[np.clip(8. * (target - current), -1., 1.), gripper]
            observation, _, done, _ = env.step(action)
            actions.append(action.tolist())
            if done:
                raise RuntimeError("Scripted capture exceeded the episode horizon")
        stages.append(dict(stage=name, end_step=len(actions)))
    final = np.asarray(observation["observation"])
    lift = float(final[5] - start[2])
    assert lift > .12, ("Object was not lifted", lift)
    assert np.linalg.norm(final[:3] - final[3:6]) < .055, "Object is not held"
    return observation, dict(
        controller="scripted proportional gripper control; not a trained policy",
        initial_object_position=start.tolist(), object_lift=lift,
        stages=stages, actions=actions,
    )


def render_task(task, seed):
    np.random.seed(seed)
    camera, annotations = {}, []
    scene = "reset state, not a learned-policy rollout"
    process = None
    if task.key.startswith("panda"):
        from quasimetric_rl.data.online.panda_gym import PandaGoalEnv
        env = PandaGoalEnv(task.paper_name)
        env.seed(seed)
        observation = env.reset()
        camera = dict(target_position=[-.18, 0, .18], distance=1.15,
                      yaw=45, pitch=-28)
        if task.paper_name == "PandaPickAndPlaceJoints-v3":
            camera.update(yaw=-55, pitch=-23)
        frame = env.backend.sim.render(width=WIDTH, height=HEIGHT, **camera)
        backend = env.backend_id
    elif task.key in ("antmaze", "pointmaze"):
        import gymnasium
        import gymnasium_robotics
        from quasimetric_rl.data.online.gymnasium_robotics import GymnasiumRoboticsGoalEnv, TASK_SPECS
        name = {"antmaze": "AntMaze_UMaze-v5", "pointmaze": "PointMaze_Medium-v3"}[task.key]
        native = gymnasium.make(name, render_mode="rgb_array", width=WIDTH, height=HEIGHT,
                               **TASK_SPECS[name]["backend_kwargs"])
        env = GymnasiumRoboticsGoalEnv(name, backend_env=native)
        env.seed(seed)
        observation = env.reset()
        sim = env.backend.ant_env if task.key == "antmaze" else env.backend.point_env
        env.backend.render()
        viewer = sim.mujoco_renderer.viewer
        sim.mujoco_renderer.camera_id = -1
        viewer.cam.type = 0
        viewer.cam.trackbodyid = -1
        viewer.cam.lookat[:] = [0, 0, 0]
        viewer.cam.distance = 28 if task.key == "antmaze" else 12.5
        viewer.cam.elevation = -90
        viewer.cam.azimuth = 90
        camera = dict(lookat=[0, 0, 0], distance=float(viewer.cam.distance),
                      elevation=-90, azimuth=90)
        frame = env.backend.render()
        backend = name
    elif task.key == "reacher":
        from quasimetric_rl.data.online.dmc import DMCGoalEnv
        env = DMCGoalEnv("reacher_hard")
        env.seed(seed)
        observation = env.reset()
        physics = env._env.physics
        physics.model.vis.global_.offwidth = WIDTH
        physics.model.vis.global_.offheight = HEIGHT
        frame = physics.render(height=HEIGHT, width=WIDTH, camera_id=0)
        backend, camera = "dm_control/reacher_hard", dict(camera_id=0)
    elif task.key == "maze2d":
        from quasimetric_rl.data.online.maze2d import OnlineMazeGoalEnv
        env = OnlineMazeGoalEnv("maze2d-large")
        env.seed(seed)
        observation = env.reset()
        camera = dict(lookat=[5, 6.5, 0], distance=21, elevation=-90, azimuth=90)
        target_id = env._env.model.site_name2id("target_site")
        env._env.model.site_rgba[target_id, 3] = 0
        frame = legacy_frame(env._env, **camera, goal=env._goal_values + 1)
        annotations.append("Goal cylinder replaces the native render-only marker at goal + model offset (1,1).")
        backend = env.backend_id
    elif task.key in ("antnavigate", "pusher"):
        from quasimetric_rl.data.online.gym_mujoco import GymMujocoGoalEnv
        env = GymMujocoGoalEnv(task.paper_name)
        env.seed(seed)
        observation = env.reset()
        backend = env.backend_id
        if task.key == "antnavigate":
            goal = env._goal_values.copy()
            center = (goal + observation["observation"][:2]) / 2
            camera = dict(lookat=[*center.tolist(), .1], distance=7, elevation=-65, azimuth=90)
            frame = legacy_frame(env.backend, **camera, goal=goal)
            annotations.append("Goal cylinder added to rendering only.")
        else:
            camera = dict(lookat=[.15, -.2, -.25], distance=2, elevation=-65, azimuth=90)
            frame = legacy_frame(env.backend, **camera)
    else:
        from quasimetric_rl.data.online.gcrl import create_env_from_spec
        env = create_env_from_spec(task.paper_name)
        env.seed(seed)
        observation = env.reset()
        camera = dict(lookat=[1.3, .75, .55], distance=1.8, elevation=-30, azimuth=135)
        if task.paper_name == "FetchPickAndPlace":
            observation, process = fetch_lift_frame_state(env, observation)
            scene = "scripted grasp-and-lift process frame, not a learned-policy rollout"
            camera.update(lookat=[1.3, .75, .58], elevation=-19, distance=1.65)
        frame = legacy_frame(env.unwrapped, **camera)
        backend = type(env.unwrapped).__name__

    frame = np.asarray(frame, dtype=np.uint8)
    assert frame.shape == (HEIGHT, WIDTH, 3), (task.key, frame.shape)
    assert float(frame.std()) > 8, (task.key, "Blank render")
    path = OUTPUT / f"{task.key}.png"
    Image.fromarray(frame).save(path)
    record = {
        "environment": task.paper_name, "key": task.key, "backend": backend,
        "seed": seed, "scene": scene,
        "image": str(path.relative_to(ROOT / "iclr2027")),
        "camera": camera, "annotations": annotations,
        "observation": np.asarray(observation["observation"]).tolist(),
        "desired_goal": np.asarray(observation["desired_goal"]).tolist(),
        "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "width": WIDTH, "height": HEIGHT,
    }
    if process is not None:
        record["scripted_process"] = process
    (OUTPUT / f"{task.key}.json").write_text(json.dumps(record, indent=2) + "\n")
    env.close()
    print(task.paper_name, backend, frame.shape, flush=True)
    return record


def refresh_audit():
    paths = [OUTPUT / f"{task.key}.json" for task in ENVIRONMENTS]
    if all(path.exists() for path in paths):
        records = [json.loads(path.read_text()) for path in paths]
        (ROOT / "iclr2027/data/environment_visualization_audit.json").write_text(
            json.dumps(records, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=[e.key for e in ENVIRONMENTS])
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.task:
        render_task(next(e for e in ENVIRONMENTS if e.key == args.task), args.seed)
        refresh_audit()
        return
    # Isolate the modern and legacy MuJoCo OpenGL contexts.
    for task in ENVIRONMENTS:
        subprocess.run([sys.executable, __file__, "--task", task.key, "--seed", str(args.seed)],
                       check=True, env=os.environ.copy())
    refresh_audit()
    print("Rendered all 14 experimental environments.", flush=True)


if __name__ == "__main__":
    main()

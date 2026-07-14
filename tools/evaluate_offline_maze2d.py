#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from d4rl_runtime import configure_d4rl_runtime

configure_d4rl_runtime(ROOT, require_library_paths=True)

import gym
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf, SCMode
from tqdm.auto import tqdm

import quasimetric_rl
from quasimetric_rl.data import Dataset
from quasimetric_rl.modules import QRLAgent, QRLConf


DEFAULT_RESULT_DIRS = (
    "online/results_queue/official_qrl_maze2d_umaze_s1000",
    "online/results_queue/official_qrl_maze2d_medium_s1000",
    "online/results_queue/official_qrl_maze2d_large_s1000",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained offline QRL maze2d checkpoints with policy rollouts.",
    )
    parser.add_argument(
        "result_dirs",
        nargs="*",
        default=list(DEFAULT_RESULT_DIRS),
        help="Result directories. Defaults to the three official_qrl_maze2d_*_s1000 dirs.",
    )
    parser.add_argument("--checkpoint", default="final", help="'final', 'latest', or an explicit checkpoint path.")
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--max-episode-steps", type=int, default=0, help="0 means env.max_episode_steps.")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--action-mode", choices=("mode", "mean", "sample"), default="mode")
    parser.add_argument(
        "--goal-mode",
        choices=("target_zero", "target_current", "target_reset"),
        default="target_zero",
        help=(
            "How to lift the 2D maze target into a full observation goal. "
            "target_zero sets non-xy dims to zero; target_current copies them from the current observation; "
            "target_reset copies them from the reset observation."
        ),
    )
    parser.add_argument("--success-radius", type=float, default=0.5)
    parser.add_argument("--out-dir", default="analysis/offline_eval")
    parser.add_argument("--prefix", default=None)
    return parser.parse_args()


def checkpoint_key(path: Path) -> tuple[int, int, int]:
    match = re.match(r"checkpoint_(\d+)_(\d+)(?:_final)?\.pth$", path.name)
    if match is None:
        return (-1, -1, 0)
    epoch, it = int(match.group(1)), int(match.group(2))
    is_final = int(path.name.endswith("_final.pth"))
    return (epoch, it, is_final)


def select_checkpoint(result_dir: Path, requested: str) -> Path:
    explicit = Path(requested)
    if requested not in ("final", "latest") and explicit.exists():
        return explicit

    ckpts = sorted(result_dir.glob("checkpoint_*.pth"), key=checkpoint_key)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint_*.pth found in {result_dir}")

    if requested == "final":
        finals = [path for path in ckpts if path.name.endswith("_final.pth")]
        if finals:
            return finals[-1]
    return ckpts[-1]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def make_agent(result_dir: Path, checkpoint: Path, device: torch.device) -> tuple[QRLAgent, Dataset, dict[str, Any]]:
    conf = load_yaml(result_dir / "config.yaml")
    env_conf = conf["env"]
    dataset = Dataset.Conf(
        kind=env_conf["kind"],
        name=env_conf["name"],
        future_observation_discount=env_conf.get("future_observation_discount", 0.99),
    ).make(dummy=True)
    agent_conf: QRLConf = OmegaConf.to_container(
        OmegaConf.merge(OmegaConf.structured(QRLConf()), conf["agent"]),
        structured_config_mode=SCMode.INSTANTIATE,
    )
    agent = agent_conf.make(
        env_spec=dataset.env_spec,
        total_optim_steps=int(conf.get("total_optim_steps", 1)),
        goal_set_dims=(
            dataset.goal_set_dims
            if agent_conf.goal_set_distance.enabled
            and agent_conf.goal_set_distance.losses.goal_dims is None
            else None
        ),
    )[0]
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    agent.load_state_dict(state["agent"])
    agent.to(device)
    agent.eval()
    return agent, dataset, conf


def reset_env(env: gym.Env, seed: int) -> np.ndarray:
    if hasattr(env, "seed"):
        env.seed(seed)
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    out = env.reset()
    if isinstance(out, tuple):
        out = out[0]
    return np.asarray(out, dtype=np.float32)


def step_env(env: gym.Env, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
    else:
        obs, reward, done, info = out
        terminated, truncated = bool(done), bool(info.get("TimeLimit.truncated", False))
    return np.asarray(obs, dtype=np.float32), float(reward), bool(terminated), bool(truncated), dict(info)


def get_target(env: gym.Env) -> np.ndarray:
    if hasattr(env, "get_target"):
        return np.asarray(env.get_target(), dtype=np.float32)
    for name in ("target_goal", "target", "_target", "goal"):
        if hasattr(env, name):
            return np.asarray(getattr(env, name), dtype=np.float32)
    raise AttributeError(f"Cannot find target on {type(env).__name__}")


def make_goal(obs: np.ndarray, reset_obs: np.ndarray, target_xy: np.ndarray, goal_mode: str) -> np.ndarray:
    if goal_mode == "target_current":
        goal = obs.copy()
    elif goal_mode == "target_reset":
        goal = reset_obs.copy()
    else:
        goal = np.zeros_like(obs, dtype=np.float32)
    goal[: target_xy.shape[0]] = target_xy
    return goal.astype(np.float32, copy=False)


def policy_action(
    agent: QRLAgent,
    obs: np.ndarray,
    goal: np.ndarray,
    action_space: gym.Space,
    device: torch.device,
    action_mode: str,
) -> np.ndarray:
    if agent.actor is None:
        raise RuntimeError("This checkpoint has no actor; policy rollout evaluation is not available.")
    with torch.inference_mode():
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        goal_t = torch.as_tensor(goal, dtype=torch.float32, device=device).unsqueeze(0)
        dist = agent.act(obs_t, goal_t)
        if action_mode == "mean":
            action_t = dist.mean
        elif action_mode == "sample":
            action_t = dist.sample()
        else:
            action_t = dist.mode
        action = action_t.squeeze(0).detach().cpu().numpy()

    if isinstance(action_space, gym.spaces.Box):
        action = np.clip(action, action_space.low, action_space.high)
    return action


def finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return value if math.isfinite(value) else None


def aggregate(values: list[float]) -> dict[str, float | None]:
    if not values:
        return dict(mean=None, std=None, min=None, max=None)
    arr = np.asarray(values, dtype=np.float64)
    return dict(
        mean=finite_or_none(float(arr.mean())),
        std=finite_or_none(float(arr.std())),
        min=finite_or_none(float(arr.min())),
        max=finite_or_none(float(arr.max())),
    )


def summarize(result: dict[str, Any], episodes: list[dict[str, Any]], elapsed_s: float) -> dict[str, Any]:
    summary = dict(result)
    summary.update(
        num_episodes=len(episodes),
        elapsed_s=elapsed_s,
        return_mean=aggregate([ep["episode_return"] for ep in episodes])["mean"],
        return_std=aggregate([ep["episode_return"] for ep in episodes])["std"],
        return_min=aggregate([ep["episode_return"] for ep in episodes])["min"],
        return_max=aggregate([ep["episode_return"] for ep in episodes])["max"],
        length_mean=aggregate([ep["episode_length"] for ep in episodes])["mean"],
        success_rate=aggregate([float(ep["success"]) for ep in episodes])["mean"],
        time_at_goal_mean=aggregate([ep["time_at_goal"] for ep in episodes])["mean"],
        first_success_step_mean=aggregate([ep["first_success_step"] for ep in episodes])["mean"],
        min_distance_mean=aggregate([ep["min_distance"] for ep in episodes])["mean"],
        final_distance_mean=aggregate([ep["final_distance"] for ep in episodes])["mean"],
    )
    success_steps = [ep["first_success_step"] for ep in episodes if ep["success"]]
    summary["first_success_step_success_only_mean"] = aggregate(success_steps)["mean"]
    norm_scores = [ep["normalized_score"] for ep in episodes if ep["normalized_score"] is not None]
    summary["normalized_score_mean"] = aggregate(norm_scores)["mean"]
    norm_scores_x100 = [ep["normalized_score_x100"] for ep in episodes if ep["normalized_score_x100"] is not None]
    summary["normalized_score_x100_mean"] = aggregate(norm_scores_x100)["mean"]
    return summary


def evaluate_one(
    result_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
    details_f,
) -> dict[str, Any]:
    checkpoint = select_checkpoint(result_dir, args.checkpoint)
    agent, dataset, conf = make_agent(result_dir, checkpoint, device)
    env = dataset.create_env()
    max_steps = args.max_episode_steps or int(getattr(env, "max_episode_steps", 1000))
    env_name = conf["env"]["name"]
    base_result = dict(
        task_id=result_dir.name,
        result_dir=str(result_dir),
        env_name=env_name,
        seed=conf.get("seed"),
        checkpoint=str(checkpoint),
        device=str(device),
        max_episode_steps=max_steps,
        action_mode=args.action_mode,
        goal_mode=args.goal_mode,
        success_radius=args.success_radius,
    )

    episodes: list[dict[str, Any]] = []
    start = time.time()
    iterator = tqdm(range(args.num_episodes), desc=f"eval {result_dir.name}", dynamic_ncols=True)
    for episode_idx in iterator:
        episode_seed = args.seed + episode_idx
        np.random.seed(episode_seed)
        random.seed(episode_seed)
        torch.manual_seed(episode_seed)
        obs = reset_env(env, episode_seed)
        reset_obs = obs.copy()
        target_xy = get_target(env)

        episode_return = 0.0
        rewards: list[float] = []
        distances = [float(np.linalg.norm(obs[: target_xy.shape[0]] - target_xy))]
        first_success_step = max_steps + 1
        terminated = False
        truncated = False

        for step in range(max_steps):
            goal = make_goal(obs, reset_obs, target_xy, args.goal_mode)
            action = policy_action(agent, obs, goal, env.action_space, device, args.action_mode)
            obs, reward, terminated, truncated, _info = step_env(env, action)
            reward = float(reward)
            episode_return += reward
            rewards.append(reward)

            distance = float(np.linalg.norm(obs[: target_xy.shape[0]] - target_xy))
            distances.append(distance)
            is_success_step = (distance <= args.success_radius) or (reward > 0)
            if is_success_step and first_success_step == max_steps + 1:
                first_success_step = step + 1
            if terminated or truncated:
                break

        success = first_success_step <= max_steps
        normalized_score = None
        if hasattr(env, "get_normalized_score"):
            normalized_score = float(env.get_normalized_score(episode_return))

        episode = dict(
            **base_result,
            episode_idx=episode_idx,
            episode_seed=episode_seed,
            episode_return=episode_return,
            normalized_score=normalized_score,
            normalized_score_x100=(None if normalized_score is None else normalized_score * 100.0),
            episode_length=len(rewards),
            success=success,
            time_at_goal=int(sum((d <= args.success_radius) or (r > 0) for d, r in zip(distances[1:], rewards))),
            first_success_step=first_success_step,
            min_distance=min(distances),
            final_distance=distances[-1],
            terminated=terminated,
            truncated=truncated,
        )
        episodes.append(episode)
        print(json.dumps(episode, sort_keys=True), file=details_f, flush=True)

    return summarize(base_result, episodes, time.time() - start)


def write_outputs(summaries: list[dict[str, Any]], details_path: Path, summary_json: Path, summary_tsv: Path) -> None:
    summary_json.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n")
    fieldnames = [
        "task_id",
        "env_name",
        "seed",
        "num_episodes",
        "return_mean",
        "return_std",
        "normalized_score_x100_mean",
        "success_rate",
        "time_at_goal_mean",
        "first_success_step_mean",
        "first_success_step_success_only_mean",
        "min_distance_mean",
        "final_distance_mean",
        "elapsed_s",
        "checkpoint",
    ]
    with summary_tsv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for summary in summaries:
            writer.writerow(summary)
    print(f"wrote details: {details_path}")
    print(f"wrote summary: {summary_json}")
    print(f"wrote summary tsv: {summary_tsv}")


def main() -> None:
    args = parse_args()
    if args.num_episodes <= 0:
        raise ValueError("--num-episodes must be positive")

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    prefix = args.prefix or f"offline_maze2d_eval_{stamp}"
    details_path = out_dir / f"{prefix}_episodes.jsonl"
    summary_json = out_dir / f"{prefix}_summary.json"
    summary_tsv = out_dir / f"{prefix}_summary.tsv"

    summaries: list[dict[str, Any]] = []
    with details_path.open("w") as details_f:
        for result_dir_arg in args.result_dirs:
            result_dir = Path(result_dir_arg)
            if not result_dir.is_absolute():
                result_dir = ROOT / result_dir
            summaries.append(evaluate_one(result_dir, args, device, details_f))

    write_outputs(summaries, details_path, summary_json, summary_tsv)
    print("")
    print("task_id\tenv\treturn_mean\tnorm_x100\tsuccess_rate\ttime_at_goal\tfirst_success")
    for summary in summaries:
        print(
            "\t".join(
                str(x)
                for x in [
                    summary["task_id"],
                    summary["env_name"],
                    f"{summary['return_mean']:.3f}" if summary["return_mean"] is not None else "NA",
                    f"{summary['normalized_score_x100_mean']:.3f}"
                    if summary["normalized_score_x100_mean"] is not None
                    else "NA",
                    f"{summary['success_rate']:.3f}" if summary["success_rate"] is not None else "NA",
                    f"{summary['time_at_goal_mean']:.3f}" if summary["time_at_goal_mean"] is not None else "NA",
                    f"{summary['first_success_step_mean']:.3f}"
                    if summary["first_success_step_mean"] is not None
                    else "NA",
                ]
            )
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.d4rl_runtime import configure_d4rl_runtime

configure_d4rl_runtime(
    ROOT,
    require_library_paths=True,
    reexec_if_library_path_changed=__name__ == "__main__",
)

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
    parser.add_argument(
        "--checkpoint",
        default="final",
        help="'final', 'latest', an Agent-checkpoint step, or an explicit path.",
    )
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of environments advanced together for batched policy inference.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help=(
            "Number of spawn-based CPU environment workers. "
            "0 keeps all environments in the evaluator process."
        ),
    )
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


def select_checkpoint(result_dir: Path, requested: str) -> Path:
    explicit = Path(requested)
    if requested not in ("final", "latest") and explicit.exists():
        return explicit

    if requested.isdigit():
        agent_checkpoint = result_dir / quasimetric_rl.utils.agent_checkpoint_filename(
            int(requested)
        )
        if agent_checkpoint.exists():
            return agent_checkpoint
        raise FileNotFoundError(
            f"No Agent checkpoint for optim_steps={int(requested)} in {result_dir}"
        )

    parsed_full_ckpts = [
        (key, path)
        for path in result_dir.glob("checkpoint_*.pth")
        if (key := quasimetric_rl.utils.full_checkpoint_key(path)) is not None
    ]
    full_ckpts = [path for _key, path in sorted(parsed_full_ckpts)]
    parsed_agent_ckpts = [
        (step, path)
        for path in result_dir.glob("agent_checkpoint_step*.pth")
        if (step := quasimetric_rl.utils.agent_checkpoint_step(path)) is not None
    ]
    agent_ckpts = [path for _step, path in sorted(parsed_agent_ckpts)]
    if not full_ckpts and not agent_ckpts:
        raise FileNotFoundError(f"No evaluation checkpoint found in {result_dir}")

    finals = [path for path in full_ckpts if path.name.endswith("_final.pth")]
    if requested in ("final", "latest") and finals:
        return finals[-1]
    if agent_ckpts:
        return agent_ckpts[-1]
    return full_ckpts[-1]


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


@dataclass
class EpisodeAccumulator:
    episode_idx: int
    episode_seed: int
    obs: np.ndarray
    reset_obs: np.ndarray
    target_xy: np.ndarray
    rewards: list[float] = field(default_factory=list)
    distances: list[float] = field(default_factory=list)
    episode_return: float = 0.0
    first_success_step: int = 0
    terminated: bool = False
    truncated: bool = False


@dataclass
class DatasetEnvFactory:
    kind: str
    name: str
    _dataset: Dataset | None = field(default=None, init=False, repr=False)

    def __call__(self) -> gym.Env:
        if self._dataset is None:
            self._dataset = Dataset.Conf(kind=self.kind, name=self.name).make(dummy=True)
        return self._dataset.create_env()


class LocalEnvPool:
    def __init__(
        self,
        slot_ids: list[int],
        env_factory: Callable[[], gym.Env],
        *,
        initial_env: gym.Env | None = None,
    ) -> None:
        self.envs: dict[int, gym.Env] = {}
        for slot_id in slot_ids:
            if slot_id == 0 and initial_env is not None:
                self.envs[slot_id] = initial_env
            else:
                self.envs[slot_id] = env_factory()

    def reset(self, requests: dict[int, int]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        results: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for slot_id, episode_seed in sorted(requests.items()):
            random.seed(episode_seed)
            np.random.seed(episode_seed)
            torch.manual_seed(episode_seed)
            env = self.envs[slot_id]
            obs = reset_env(env, episode_seed)
            results[slot_id] = (obs, get_target(env))
        return results

    def step(
        self, actions: dict[int, np.ndarray]
    ) -> dict[int, tuple[np.ndarray, float, bool, bool]]:
        results: dict[int, tuple[np.ndarray, float, bool, bool]] = {}
        for slot_id, action in sorted(actions.items()):
            env = self.envs[slot_id]
            if isinstance(env.action_space, gym.spaces.Box):
                action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, terminated, truncated, _info = step_env(env, action)
            results[slot_id] = (obs, reward, terminated, truncated)
        return results

    def close(self) -> None:
        for env in self.envs.values():
            close = getattr(env, "close", None)
            if close is not None:
                close()


def env_worker_main(
    connection: Any,
    slot_ids: list[int],
    env_factory: Callable[[], gym.Env],
) -> None:
    pool: LocalEnvPool | None = None
    try:
        pool = LocalEnvPool(slot_ids, env_factory)
        connection.send(("ready", None))
        while True:
            command, payload = connection.recv()
            if command == "close":
                connection.send(("ok", None))
                return
            if command == "reset":
                result = pool.reset(dict(payload))
            elif command == "step":
                result = pool.step(dict(payload))
            else:
                raise ValueError(f"Unknown environment worker command: {command!r}")
            connection.send(("ok", result))
    except BaseException:
        try:
            connection.send(("error", traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if pool is not None:
            pool.close()
        connection.close()


class ProcessEnvPool:
    def __init__(
        self,
        num_envs: int,
        num_workers: int,
        env_factory: Callable[[], gym.Env],
    ) -> None:
        context = mp.get_context("spawn")
        assignments = [list(range(worker_idx, num_envs, num_workers)) for worker_idx in range(num_workers)]
        self.worker_for_slot: dict[int, int] = {}
        self.connections: list[Any] = []
        self.processes: list[mp.Process] = []
        try:
            for worker_idx, slot_ids in enumerate(assignments):
                parent_connection, child_connection = context.Pipe()
                process = context.Process(
                    target=env_worker_main,
                    args=(child_connection, slot_ids, env_factory),
                    daemon=True,
                )
                process.start()
                child_connection.close()
                self.connections.append(parent_connection)
                self.processes.append(process)
                for slot_id in slot_ids:
                    self.worker_for_slot[slot_id] = worker_idx
            for worker_idx, connection in enumerate(self.connections):
                self._receive(worker_idx, connection, expected_status="ready")
        except BaseException:
            self.close(force=True)
            raise

    @staticmethod
    def _receive(worker_idx: int, connection: Any, *, expected_status: str = "ok") -> Any:
        try:
            status, payload = connection.recv()
        except EOFError as exc:
            raise RuntimeError(f"Environment worker {worker_idx} exited without a response") from exc
        if status == "error":
            raise RuntimeError(f"Environment worker {worker_idx} failed:\n{payload}")
        if status != expected_status:
            raise RuntimeError(
                f"Environment worker {worker_idx} returned {status!r}; expected {expected_status!r}"
            )
        return payload

    def _execute(self, command: str, payload: dict[int, Any]) -> dict[int, Any]:
        grouped: dict[int, list[tuple[int, Any]]] = {}
        for slot_id, value in sorted(payload.items()):
            grouped.setdefault(self.worker_for_slot[slot_id], []).append((slot_id, value))
        for worker_idx, items in grouped.items():
            self.connections[worker_idx].send((command, items))
        result: dict[int, Any] = {}
        for worker_idx in sorted(grouped):
            worker_result = self._receive(worker_idx, self.connections[worker_idx])
            result.update(worker_result)
        return result

    def reset(self, requests: dict[int, int]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        return self._execute("reset", requests)

    def step(
        self, actions: dict[int, np.ndarray]
    ) -> dict[int, tuple[np.ndarray, float, bool, bool]]:
        return self._execute("step", actions)

    def close(self, *, force: bool = False) -> None:
        if not force:
            for process, connection in zip(self.processes, self.connections):
                if process.is_alive():
                    try:
                        connection.send(("close", None))
                    except (BrokenPipeError, EOFError, OSError):
                        pass
            for worker_idx, (process, connection) in enumerate(zip(self.processes, self.connections)):
                if process.is_alive():
                    try:
                        self._receive(worker_idx, connection)
                    except (BrokenPipeError, EOFError, OSError, RuntimeError):
                        pass
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        for connection in self.connections:
            connection.close()


def policy_actions(
    agent: QRLAgent,
    observations: np.ndarray,
    goals: np.ndarray,
    action_space: gym.Space,
    device: torch.device,
    action_mode: str,
) -> np.ndarray:
    if agent.actor is None:
        raise RuntimeError("This checkpoint has no actor; policy rollout evaluation is not available.")
    with torch.inference_mode():
        obs_t = torch.as_tensor(observations, dtype=torch.float32, device=device)
        goal_t = torch.as_tensor(goals, dtype=torch.float32, device=device)
        dist = agent.act(obs_t, goal_t)
        if action_mode == "mean":
            action_t = dist.mean
        elif action_mode == "sample":
            action_t = dist.sample()
        else:
            action_t = dist.mode
        actions = action_t.detach().cpu().numpy()

    if isinstance(action_space, gym.spaces.Box):
        actions = np.clip(actions, action_space.low, action_space.high)
    return actions


def policy_action(
    agent: QRLAgent,
    obs: np.ndarray,
    goal: np.ndarray,
    action_space: gym.Space,
    device: torch.device,
    action_mode: str,
) -> np.ndarray:
    return policy_actions(
        agent,
        obs[None],
        goal[None],
        action_space,
        device,
        action_mode,
    )[0]


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


def start_episode(
    episode_idx: int,
    episode_seed: int,
    obs: np.ndarray,
    target_xy: np.ndarray,
    max_steps: int,
) -> EpisodeAccumulator:
    initial_distance = float(np.linalg.norm(obs[: target_xy.shape[0]] - target_xy))
    return EpisodeAccumulator(
        episode_idx=episode_idx,
        episode_seed=episode_seed,
        obs=obs,
        reset_obs=obs.copy(),
        target_xy=target_xy,
        distances=[initial_distance],
        first_success_step=max_steps + 1,
    )


def finish_episode(
    state: EpisodeAccumulator,
    base_result: dict[str, Any],
    normalizer_env: gym.Env,
    success_radius: float,
    max_steps: int,
) -> dict[str, Any]:
    success = state.first_success_step <= max_steps
    normalized_score = None
    if hasattr(normalizer_env, "get_normalized_score"):
        normalized_score = float(normalizer_env.get_normalized_score(state.episode_return))
    return dict(
        **base_result,
        episode_idx=state.episode_idx,
        episode_seed=state.episode_seed,
        episode_return=state.episode_return,
        normalized_score=normalized_score,
        normalized_score_x100=(None if normalized_score is None else normalized_score * 100.0),
        episode_length=len(state.rewards),
        success=success,
        time_at_goal=int(
            sum(
                (distance <= success_radius) or (reward > 0)
                for distance, reward in zip(state.distances[1:], state.rewards)
            )
        ),
        first_success_step=state.first_success_step,
        min_distance=min(state.distances),
        final_distance=state.distances[-1],
        terminated=state.terminated,
        truncated=state.truncated,
    )


def rollout_episodes(
    agent: QRLAgent,
    dataset: Dataset,
    args: argparse.Namespace,
    device: torch.device,
    details_f: Any,
    base_result: dict[str, Any],
) -> tuple[list[dict[str, Any]], float]:
    num_envs = min(args.num_envs, args.num_episodes)
    num_workers = min(args.num_workers, num_envs)
    probe_env = dataset.create_env()
    max_steps = base_result["max_episode_steps"]
    pool: LocalEnvPool | ProcessEnvPool | None = None
    probe_owned_by_pool = False
    try:
        if num_workers > 0:
            env_factory = DatasetEnvFactory(kind=dataset.kind, name=dataset.name)
            pool = ProcessEnvPool(num_envs, num_workers, env_factory)
        else:
            pool = LocalEnvPool(
                list(range(num_envs)),
                dataset.create_env,
                initial_env=probe_env,
            )
            probe_owned_by_pool = True

        action_space = probe_env.action_space
        active: dict[int, EpisodeAccumulator] = {}
        episode_records: list[dict[str, Any] | None] = [None] * args.num_episodes
        next_episode_idx = 0
        next_write_idx = 0
        torch.manual_seed(args.seed)

        def assign_episodes(slot_ids: list[int]) -> None:
            nonlocal next_episode_idx
            reset_requests: dict[int, int] = {}
            assigned_indices: dict[int, int] = {}
            for slot_id in sorted(slot_ids):
                if next_episode_idx >= args.num_episodes:
                    break
                episode_idx = next_episode_idx
                next_episode_idx += 1
                assigned_indices[slot_id] = episode_idx
                reset_requests[slot_id] = args.seed + episode_idx
            if not reset_requests:
                return
            reset_results = pool.reset(reset_requests)
            for slot_id, episode_idx in assigned_indices.items():
                obs, target_xy = reset_results[slot_id]
                active[slot_id] = start_episode(
                    episode_idx,
                    args.seed + episode_idx,
                    obs,
                    target_xy,
                    max_steps,
                )

        start = time.time()
        assign_episodes(list(range(num_envs)))
        with tqdm(
            total=args.num_episodes,
            desc=f"eval {base_result['task_id']}",
            dynamic_ncols=True,
        ) as progress:
            while active:
                slot_ids = sorted(active)
                observations = np.stack([active[slot_id].obs for slot_id in slot_ids])
                goals = np.stack(
                    [
                        make_goal(
                            active[slot_id].obs,
                            active[slot_id].reset_obs,
                            active[slot_id].target_xy,
                            args.goal_mode,
                        )
                        for slot_id in slot_ids
                    ]
                )
                action_batch = policy_actions(
                    agent,
                    observations,
                    goals,
                    action_space,
                    device,
                    args.action_mode,
                )
                step_results = pool.step(dict(zip(slot_ids, action_batch)))
                freed_slots: list[int] = []
                for slot_id in slot_ids:
                    state = active[slot_id]
                    obs, reward, terminated, truncated = step_results[slot_id]
                    state.obs = obs
                    state.episode_return += reward
                    state.rewards.append(reward)
                    state.terminated = terminated
                    state.truncated = truncated
                    distance = float(
                        np.linalg.norm(obs[: state.target_xy.shape[0]] - state.target_xy)
                    )
                    state.distances.append(distance)
                    if (
                        (distance <= args.success_radius or reward > 0)
                        and state.first_success_step == max_steps + 1
                    ):
                        state.first_success_step = len(state.rewards)

                    if terminated or truncated or len(state.rewards) >= max_steps:
                        episode_records[state.episode_idx] = finish_episode(
                            state,
                            base_result,
                            probe_env,
                            args.success_radius,
                            max_steps,
                        )
                        del active[slot_id]
                        freed_slots.append(slot_id)
                        progress.update(1)

                while (
                    next_write_idx < len(episode_records)
                    and episode_records[next_write_idx] is not None
                ):
                    print(
                        json.dumps(episode_records[next_write_idx], sort_keys=True),
                        file=details_f,
                        flush=True,
                    )
                    next_write_idx += 1
                assign_episodes(freed_slots)

        episodes = [episode for episode in episode_records if episode is not None]
        if len(episodes) != args.num_episodes:
            raise RuntimeError(
                f"Completed {len(episodes)} episodes; expected {args.num_episodes}"
            )
        return episodes, time.time() - start
    finally:
        if pool is not None:
            pool.close()
        if not probe_owned_by_pool:
            close = getattr(probe_env, "close", None)
            if close is not None:
                close()


def evaluate_one(
    result_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
    details_f,
) -> dict[str, Any]:
    checkpoint = select_checkpoint(result_dir, args.checkpoint)
    agent, dataset, conf = make_agent(result_dir, checkpoint, device)
    probe_env = dataset.create_env()
    try:
        max_steps = args.max_episode_steps or int(getattr(probe_env, "max_episode_steps", 1000))
    finally:
        close = getattr(probe_env, "close", None)
        if close is not None:
            close()
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
        num_envs=min(args.num_envs, args.num_episodes),
        num_workers=min(args.num_workers, args.num_envs, args.num_episodes),
    )
    episodes, elapsed_s = rollout_episodes(
        agent,
        dataset,
        args,
        device,
        details_f,
        base_result,
    )
    return summarize(base_result, episodes, elapsed_s)


def write_outputs(summaries: list[dict[str, Any]], details_path: Path, summary_json: Path, summary_tsv: Path) -> None:
    summary_json.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n")
    fieldnames = [
        "task_id",
        "env_name",
        "seed",
        "num_episodes",
        "num_envs",
        "num_workers",
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
    if args.num_envs <= 0:
        raise ValueError("--num-envs must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be nonnegative")
    if args.num_workers > args.num_envs:
        raise ValueError("--num-workers cannot exceed --num-envs")

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

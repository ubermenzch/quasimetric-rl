#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import math
import multiprocessing as mp
import os
import queue
import random
import re
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
    prefer_nvidia_egl_vendor=True,
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
ALL_CHECKPOINT_NUM_EPISODES = 1000
TRAINING_SEED_TOKEN_RE = re.compile(
    r"(?P<prefix>(?:^|[_/\-])s)(?P<seed>\d+)(?=$|[_/\-])"
)


@dataclass(frozen=True)
class EvaluationTask:
    result_dir: Path
    checkpoint: Path | None = None


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
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--checkpoint",
        default="final",
        help="'final', 'latest', an Agent-checkpoint step, or an explicit path.",
    )
    checkpoint_group.add_argument(
        "--all-checkpoints",
        action="store_true",
        help=(
            "Evaluate every archival checkpoint in each result directory. "
            f"This mode always runs {ALL_CHECKPOINT_NUM_EPISODES} episodes per checkpoint."
        ),
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=None,
        help=(
            "Episodes per evaluation (default: 100). In --all-checkpoints mode "
            f"this must be {ALL_CHECKPOINT_NUM_EPISODES}."
        ),
    )
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
    parser.add_argument(
        "--seed",
        type=int,
        default=1000,
        help=(
            "Base seed for evaluation episodes. The training seed is read from "
            "each result directory's config.yaml."
        ),
    )
    parser.add_argument(
        "--training-seeds",
        default=None,
        help=(
            "Comma-separated training seeds, for example '1000,1001,1002'. "
            "Each result directory must contain a {seed} placeholder or an sNNN seed token."
        ),
    )
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device used by the serial evaluator (default: cuda:0 when available).",
    )
    device_group.add_argument(
        "--gpus",
        default=None,
        help=(
            "Comma-separated CUDA device indices, for example '0,1,3'. "
            "Runs one evaluator per GPU and dynamically distributes evaluation tasks."
        ),
    )
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


def parse_gpu_ids(value: str) -> list[int]:
    gpu_ids: list[int] = []
    for token in value.replace(",", " ").split():
        if token.startswith("cuda:"):
            token = token.removeprefix("cuda:")
        try:
            gpu_id = int(token)
        except ValueError as exc:
            raise ValueError(f"Invalid GPU index {token!r} in --gpus") from exc
        if gpu_id < 0:
            raise ValueError("--gpus indices must be nonnegative")
        if gpu_id in gpu_ids:
            raise ValueError(f"Duplicate GPU index in --gpus: {gpu_id}")
        gpu_ids.append(gpu_id)
    if not gpu_ids:
        raise ValueError("--gpus must contain at least one CUDA device index")
    return gpu_ids


def parse_training_seeds(value: str) -> list[int]:
    training_seeds: list[int] = []
    for token in value.replace(",", " ").split():
        try:
            training_seed = int(token)
        except ValueError as exc:
            raise ValueError(
                f"Invalid training seed {token!r} in --training-seeds"
            ) from exc
        if training_seed < 0:
            raise ValueError("--training-seeds values must be nonnegative")
        if training_seed in training_seeds:
            raise ValueError(
                f"Duplicate training seed in --training-seeds: {training_seed}"
            )
        training_seeds.append(training_seed)
    if not training_seeds:
        raise ValueError("--training-seeds must contain at least one seed")
    return training_seeds


def expand_result_dirs_for_training_seeds(
    result_dir_args: list[str],
    training_seeds: list[int],
) -> list[str]:
    expanded: list[str] = []
    seen: set[str] = set()
    for result_dir_arg in result_dir_args:
        seed_matches = list(TRAINING_SEED_TOKEN_RE.finditer(result_dir_arg))
        if "{seed}" not in result_dir_arg and not seed_matches:
            raise ValueError(
                f"Result directory {result_dir_arg!r} has no {{seed}} placeholder or sNNN seed token"
            )
        for training_seed in training_seeds:
            if "{seed}" in result_dir_arg:
                expanded_arg = result_dir_arg.replace("{seed}", str(training_seed))
            else:
                seed_match = seed_matches[-1]
                expanded_arg = (
                    result_dir_arg[: seed_match.start("seed")]
                    + str(training_seed)
                    + result_dir_arg[seed_match.end("seed") :]
                )
            if expanded_arg not in seen:
                seen.add(expanded_arg)
                expanded.append(expanded_arg)
    return expanded


def available_cpu_cores() -> list[int]:
    if hasattr(os, "sched_getaffinity"):
        cores = sorted(os.sched_getaffinity(0))
        if cores:
            return cores
    return list(range(os.cpu_count() or 1))


def split_cpu_cores(cpu_cores: list[int], num_groups: int) -> list[list[int]]:
    if not cpu_cores:
        raise ValueError("Cannot allocate workers without any CPU cores")
    if num_groups <= 0:
        raise ValueError("num_groups must be positive")
    if num_groups > len(cpu_cores):
        return [[cpu_cores[group_idx % len(cpu_cores)]] for group_idx in range(num_groups)]

    quotient, remainder = divmod(len(cpu_cores), num_groups)
    groups: list[list[int]] = []
    start = 0
    for group_idx in range(num_groups):
        size = quotient + int(group_idx < remainder)
        groups.append(cpu_cores[start : start + size])
        start += size
    return groups


def format_cpu_cores(cpu_cores: list[int]) -> str:
    ranges: list[str] = []
    range_start = previous = cpu_cores[0]
    for core in cpu_cores[1:]:
        if core == previous + 1:
            previous = core
            continue
        ranges.append(str(range_start) if range_start == previous else f"{range_start}-{previous}")
        range_start = previous = core
    ranges.append(str(range_start) if range_start == previous else f"{range_start}-{previous}")
    return ",".join(ranges)


def resolve_num_episodes(requested: int | None, all_checkpoints: bool) -> int:
    if all_checkpoints:
        if requested not in (None, ALL_CHECKPOINT_NUM_EPISODES):
            raise ValueError(
                f"--all-checkpoints requires --num-episodes={ALL_CHECKPOINT_NUM_EPISODES}"
            )
        return ALL_CHECKPOINT_NUM_EPISODES
    return 100 if requested is None else requested


def find_evaluation_checkpoints(result_dir: Path) -> list[Path]:
    periodic_full: list[tuple[tuple[int, int, int], Path]] = []
    finals: list[tuple[tuple[int, int, int], Path]] = []
    for path in result_dir.glob("checkpoint_*.pth"):
        key = quasimetric_rl.utils.full_checkpoint_key(path)
        if key is None:
            continue
        (finals if key[2] else periodic_full).append((key, path))

    agent_checkpoints = sorted(
        (
            (step, path)
            for path in result_dir.glob("agent_checkpoint_step*.pth")
            if (step := quasimetric_rl.utils.agent_checkpoint_step(path)) is not None
        ),
        key=lambda item: item[0],
    )
    checkpoints = [path for _key, path in sorted(periodic_full)]
    checkpoints.extend(path for _step, path in agent_checkpoints)
    checkpoints.extend(path for _key, path in sorted(finals))
    if not checkpoints:
        raise FileNotFoundError(f"No evaluation checkpoint found in {result_dir}")
    return checkpoints


def make_evaluation_tasks(
    result_dirs: list[Path],
    *,
    all_checkpoints: bool,
) -> list[EvaluationTask]:
    if not all_checkpoints:
        return [EvaluationTask(result_dir) for result_dir in result_dirs]
    return [
        EvaluationTask(result_dir, checkpoint)
        for result_dir in result_dirs
        for checkpoint in find_evaluation_checkpoints(result_dir)
    ]


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


def validate_result_dir_training_seeds(
    result_dirs: list[Path],
    requested_seeds: list[int],
) -> None:
    requested = set(requested_seeds)
    found: set[int] = set()
    for result_dir in result_dirs:
        conf = load_yaml(result_dir / "config.yaml")
        if "seed" not in conf:
            raise ValueError(f"No training seed found in {result_dir / 'config.yaml'}")
        training_seed = int(conf["seed"])
        if training_seed not in requested:
            raise ValueError(
                f"Training seed mismatch for {result_dir}: config has {training_seed}, "
                f"requested {sorted(requested)}"
            )
        found.add(training_seed)
    missing = sorted(requested - found)
    if missing:
        raise ValueError(f"No result directory resolved for training seeds {missing}")


def make_agent(
    result_dir: Path,
    checkpoint: Path,
    device: torch.device,
) -> tuple[QRLAgent, Dataset, dict[str, Any], dict[str, Any]]:
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
    checkpoint_metadata = {
        key: state.get(key)
        for key in ("optim_steps", "epoch", "it", "checkpoint_kind")
    }
    agent_checkpoint_step = quasimetric_rl.utils.agent_checkpoint_step(checkpoint)
    if checkpoint_metadata["optim_steps"] is None:
        checkpoint_metadata["optim_steps"] = agent_checkpoint_step
    if checkpoint_metadata["checkpoint_kind"] is None:
        if agent_checkpoint_step is not None:
            checkpoint_metadata["checkpoint_kind"] = "agent"
        elif checkpoint.name.endswith("_final.pth"):
            checkpoint_metadata["checkpoint_kind"] = "final"
        else:
            checkpoint_metadata["checkpoint_kind"] = "full"
    return agent, dataset, conf, checkpoint_metadata


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
            position=getattr(args, "progress_position", 0),
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
    *,
    checkpoint_override: Path | None = None,
) -> dict[str, Any]:
    checkpoint = checkpoint_override or select_checkpoint(result_dir, args.checkpoint)
    agent, dataset, conf, checkpoint_metadata = make_agent(
        result_dir,
        checkpoint,
        device,
    )
    probe_env = dataset.create_env()
    try:
        max_steps = args.max_episode_steps or int(getattr(probe_env, "max_episode_steps", 1000))
    finally:
        close = getattr(probe_env, "close", None)
        if close is not None:
            close()
    env_name = conf["env"]["name"]
    run_id = result_dir.name
    task_id = (
        f"{run_id}@{checkpoint.stem}"
        if getattr(args, "all_checkpoints", False)
        else run_id
    )
    base_result = dict(
        task_id=task_id,
        run_id=run_id,
        result_dir=str(result_dir),
        env_name=env_name,
        seed=conf.get("seed"),
        checkpoint=str(checkpoint),
        checkpoint_name=checkpoint.name,
        checkpoint_optim_steps=checkpoint_metadata["optim_steps"],
        checkpoint_epoch=checkpoint_metadata["epoch"],
        checkpoint_iteration=checkpoint_metadata["it"],
        checkpoint_kind=checkpoint_metadata["checkpoint_kind"],
        device=str(device),
        max_episode_steps=max_steps,
        action_mode=args.action_mode,
        goal_mode=args.goal_mode,
        success_radius=args.success_radius,
        num_envs=min(args.num_envs, args.num_episodes),
        num_workers=min(args.num_workers, args.num_envs, args.num_episodes),
        worker_index=getattr(args, "worker_index", None),
        cpu_cores=getattr(args, "assigned_cpu_cores", None),
        num_cpu_cores=len(getattr(args, "assigned_cpu_cores", [])) or None,
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
        "run_id",
        "env_name",
        "seed",
        "checkpoint_name",
        "checkpoint_optim_steps",
        "checkpoint_epoch",
        "checkpoint_iteration",
        "checkpoint_kind",
        "num_episodes",
        "num_envs",
        "num_workers",
        "device",
        "num_cpu_cores",
        "cpu_cores",
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


def configure_evaluation_worker(cpu_cores: list[int]) -> None:
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, cpu_cores)

    thread_count = len(cpu_cores)
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = str(thread_count)
    torch.set_num_threads(thread_count)
    try:
        torch.set_num_interop_threads(max(1, min(thread_count, 4)))
    except RuntimeError:
        pass


def evaluation_worker_main(
    worker_index: int,
    gpu_id: int,
    cpu_cores: list[int],
    args: argparse.Namespace,
    task_queue: Any,
    result_queue: Any,
) -> None:
    try:
        configure_evaluation_worker(cpu_cores)
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)
    except Exception:
        result_queue.put(("worker_error", worker_index, traceback.format_exc()))
        return

    args.worker_index = worker_index
    args.progress_position = worker_index
    args.assigned_cpu_cores = cpu_cores
    while True:
        task = task_queue.get()
        if task is None:
            return
        task_index, evaluation_task = task
        details = io.StringIO()
        try:
            summary = evaluate_one(
                evaluation_task.result_dir,
                args,
                device,
                details,
                checkpoint_override=evaluation_task.checkpoint,
            )
            result_queue.put(
                ("result", task_index, summary, details.getvalue(), worker_index)
            )
        except Exception:
            result_queue.put(
                (
                    "task_error",
                    task_index,
                    str(evaluation_task.result_dir),
                    str(evaluation_task.checkpoint or args.checkpoint),
                    traceback.format_exc(),
                )
            )
        finally:
            torch.cuda.empty_cache()


def evaluate_in_parallel(
    evaluation_tasks: list[EvaluationTask],
    args: argparse.Namespace,
    gpu_ids: list[int],
) -> tuple[list[dict[str, Any]], list[str]]:
    worker_count = min(len(evaluation_tasks), len(gpu_ids))
    cpu_groups = split_cpu_cores(available_cpu_cores(), worker_count)
    context = mp.get_context("spawn")
    task_queue = context.Queue()
    result_queue = context.Queue()
    workers: list[mp.Process] = []

    for task_index, evaluation_task in enumerate(evaluation_tasks):
        task_queue.put((task_index, evaluation_task))
    for _ in range(worker_count):
        task_queue.put(None)

    print(
        f"running {len(evaluation_tasks)} evaluation task(s) on "
        f"{worker_count} GPU worker(s): {gpu_ids[:worker_count]}"
    )
    for worker_index in range(worker_count):
        print(
            f"worker {worker_index}: cuda:{gpu_ids[worker_index]}, "
            f"CPUs {format_cpu_cores(cpu_groups[worker_index])}"
        )
        process = context.Process(
            target=evaluation_worker_main,
            args=(
                worker_index,
                gpu_ids[worker_index],
                cpu_groups[worker_index],
                args,
                task_queue,
                result_queue,
            ),
            name=f"offline-eval-gpu-{gpu_ids[worker_index]}",
        )
        process.start()
        workers.append(process)

    results: dict[int, tuple[dict[str, Any], str]] = {}
    errors: list[str] = []
    failed_task_indices: set[int] = set()
    completed_tasks = 0
    try:
        while completed_tasks < len(evaluation_tasks):
            try:
                message = result_queue.get(timeout=0.5)
            except queue.Empty:
                if all(process.exitcode is not None for process in workers):
                    break
                continue

            message_type = message[0]
            if message_type == "result":
                _, task_index, summary, details, worker_index = message
                results[task_index] = (summary, details)
                completed_tasks += 1
                print(
                    f"completed [{completed_tasks}/{len(evaluation_tasks)}] "
                    f"{summary['task_id']} on cuda:{gpu_ids[worker_index]}"
                )
            elif message_type == "task_error":
                _, task_index, result_dir, checkpoint, error = message
                completed_tasks += 1
                failed_task_indices.add(task_index)
                errors.append(
                    f"evaluation task {task_index} ({result_dir}, checkpoint={checkpoint}) "
                    f"failed:\n{error}"
                )
            elif message_type == "worker_error":
                _, worker_index, error = message
                errors.append(f"evaluation worker {worker_index} failed to start:\n{error}")

        missing = sorted(set(range(len(evaluation_tasks))) - set(results))
        missing = [task_index for task_index in missing if task_index not in failed_task_indices]
        if missing:
            exit_codes = ", ".join(
                f"{process.name}={process.exitcode}" for process in workers
            )
            errors.append(
                f"evaluation workers exited before returning tasks {missing}; exit codes: {exit_codes}"
            )
        if errors:
            raise RuntimeError("\n\n".join(errors))
        return (
            [results[task_index][0] for task_index in range(len(evaluation_tasks))],
            [results[task_index][1] for task_index in range(len(evaluation_tasks))],
        )
    finally:
        for process in workers:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        task_queue.close()
        result_queue.close()


def main() -> None:
    args = parse_args()
    args.num_episodes = resolve_num_episodes(
        args.num_episodes,
        args.all_checkpoints,
    )
    if args.num_episodes <= 0:
        raise ValueError("--num-episodes must be positive")
    if args.num_envs <= 0:
        raise ValueError("--num-envs must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be nonnegative")
    if args.num_workers > args.num_envs:
        raise ValueError("--num-workers cannot exceed --num-envs")

    training_seeds: list[int] | None = None
    result_dir_args = args.result_dirs
    if args.training_seeds is not None:
        training_seeds = parse_training_seeds(args.training_seeds)
        result_dir_args = expand_result_dirs_for_training_seeds(
            result_dir_args,
            training_seeds,
        )

    result_dirs: list[Path] = []
    for result_dir_arg in result_dir_args:
        result_dir = Path(result_dir_arg)
        if not result_dir.is_absolute():
            result_dir = ROOT / result_dir
        result_dirs.append(result_dir)
    if training_seeds is not None:
        validate_result_dir_training_seeds(result_dirs, training_seeds)
        print(
            f"resolved {len(result_dirs)} result directory/directories for "
            f"training seeds {training_seeds}"
        )
    evaluation_tasks = make_evaluation_tasks(
        result_dirs,
        all_checkpoints=args.all_checkpoints,
    )
    if args.all_checkpoints:
        print(
            f"discovered {len(evaluation_tasks)} checkpoint(s) across "
            f"{len(result_dirs)} result directory/directories; "
            f"running {args.num_episodes} episodes per checkpoint"
        )

    gpu_ids: list[int] | None = None
    if args.gpus is not None:
        gpu_ids = parse_gpu_ids(args.gpus)
        if not torch.cuda.is_available():
            raise RuntimeError("--gpus requires CUDA, but torch.cuda.is_available() is false")
        device_count = torch.cuda.device_count()
        invalid_gpu_ids = [gpu_id for gpu_id in gpu_ids if gpu_id >= device_count]
        if invalid_gpu_ids:
            raise ValueError(
                f"GPU indices {invalid_gpu_ids} are unavailable; visible CUDA device count is {device_count}"
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    prefix = args.prefix or f"offline_maze2d_eval_{stamp}"
    details_path = out_dir / f"{prefix}_episodes.jsonl"
    summary_json = out_dir / f"{prefix}_summary.json"
    summary_tsv = out_dir / f"{prefix}_summary.tsv"

    summaries: list[dict[str, Any]] = []
    if gpu_ids is not None:
        summaries, task_details = evaluate_in_parallel(
            evaluation_tasks,
            args,
            gpu_ids,
        )
        with details_path.open("w") as details_f:
            for details in task_details:
                details_f.write(details)
    else:
        device = torch.device(args.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        with details_path.open("w") as details_f:
            for evaluation_task in evaluation_tasks:
                summaries.append(
                    evaluate_one(
                        evaluation_task.result_dir,
                        args,
                        device,
                        details_f,
                        checkpoint_override=evaluation_task.checkpoint,
                    )
                )

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

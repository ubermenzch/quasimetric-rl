#!/usr/bin/env python3
"""Permanently delete queue tasks selected by stable task identity."""

from __future__ import annotations

import argparse
import fcntl
import fnmatch
import json
import os
import shutil
import sys
import tempfile
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_qrl_queue import (
    Task,
    cfg,
    parse_config,
    pid_alive,
    read_status,
    read_tasks,
    resolve_path,
    timestamp,
)


KNOWN_STATES = (
    "PENDING",
    "RUNNING",
    "STALE",
    "DONE",
    "FAILED",
    "PAUSED",
    "MISSING",
)

CHECKPOINTS_DELETED_MARKER = "CHECKPOINTS_DELETED"


class DeletionError(RuntimeError):
    pass


@dataclass(frozen=True)
class QueuePaths:
    tasks_file: Path
    status_dir: Path
    results_root: Path
    log_dir: Path
    eta_sample_file: Path
    lock_file: Path


@dataclass(frozen=True)
class TaskEntry:
    task_id: str
    task: Task | None
    status: dict[str, str]

    @property
    def active(self) -> bool:
        return self.task is not None

    @property
    def state(self) -> str:
        state = self.status.get("state", "")
        if state == "RUNNING" and not pid_alive(self.status.get("pid", "")):
            return "STALE"
        if state:
            return state
        return "PENDING" if self.active else "MISSING"

    def field(self, task_name: str, status_name: str) -> str:
        if self.task is not None:
            return str(getattr(self.task, task_name))
        return self.status.get(status_name, "")


@dataclass(frozen=True)
class Selectors:
    exact_ids: frozenset[str]
    id_globs: tuple[str, ...]
    states: frozenset[str]
    env_names: frozenset[str]
    seeds: frozenset[str]
    modes: frozenset[str]
    orphaned: bool

    def specified(self) -> bool:
        return any((
            self.exact_ids,
            self.id_globs,
            self.states,
            self.env_names,
            self.seeds,
            self.modes,
            self.orphaned,
        ))


@dataclass(frozen=True)
class TaskArtifacts:
    entry: TaskEntry
    status_files: tuple[Path, ...]
    output_path: Path
    log_files: tuple[Path, ...]
    checkpoint_files: tuple[Path, ...]

    @property
    def checkpoint_bytes(self) -> int:
        total = 0
        for path in self.checkpoint_files:
            try:
                total += path.lstat().st_size
            except OSError:
                pass
        return total

    @property
    def checkpoints_deleted_marker(self) -> Path:
        return self.output_path / CHECKPOINTS_DELETED_MARKER


def resolve_override(value: str | None, config: dict[str, str], key: str, default: str) -> Path:
    return resolve_path(value if value is not None else cfg(config, key, default))


def queue_paths(args: argparse.Namespace) -> QueuePaths:
    config = parse_config(resolve_path(args.config))
    status_dir = resolve_override(args.status_dir, config, "STATUS_DIR", "runs/qrl_queue/status")
    return QueuePaths(
        tasks_file=resolve_override(
            args.tasks_file, config, "TASKS_FILE", "runs/qrl_queue/tasks.tsv"
        ),
        status_dir=status_dir,
        results_root=resolve_override(
            args.results_root, config, "RESULTS_ROOT", "../qrl-assets/results/queue"
        ),
        log_dir=resolve_override(args.log_dir, config, "LOG_DIR", "logs/qrl_queue"),
        eta_sample_file=resolve_override(
            args.eta_sample_file,
            config,
            "ETA_SAMPLE_FILE",
            str(status_dir / "watch_eta_samples.json"),
        ),
        lock_file=resolve_override(
            args.lock_file, config, "LOCK_FILE", "runs/qrl_queue/qrl_queue.lock"
        ),
    )


def validate_task_id(task_id: str) -> None:
    if not task_id or task_id in {".", ".."} or Path(task_id).name != task_id:
        raise DeletionError(f"unsafe task_id: {task_id!r}")


def ids_from_files(paths: Sequence[str]) -> set[str]:
    task_ids: set[str] = set()
    for value in paths:
        path = resolve_path(value)
        if not path.is_file():
            raise DeletionError(f"task ID file does not exist: {path}")
        for raw_line in path.read_text(errors="replace").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            task_ids.add(line.split("\t", 1)[0].strip())
    return task_ids


def build_selectors(args: argparse.Namespace) -> Selectors:
    exact_ids = set(args.task_id)
    exact_ids.update(ids_from_files(args.task_id_file))
    for task_id in exact_ids:
        validate_task_id(task_id)
    selectors = Selectors(
        exact_ids=frozenset(exact_ids),
        id_globs=tuple(args.task_id_glob),
        states=frozenset(args.state),
        env_names=frozenset(args.env_name),
        seeds=frozenset(args.seed),
        modes=frozenset(args.mode),
        orphaned=args.orphaned,
    )
    if not selectors.specified():
        raise DeletionError("at least one stable task selector is required")
    return selectors


def task_owned_paths_exist(paths: QueuePaths, task_id: str) -> bool:
    status_file = paths.status_dir / f"{task_id}.status"
    if status_file.exists() or status_file.with_suffix(".status.tmp").exists():
        return True
    if (paths.results_root / task_id).exists():
        return True
    if not paths.log_dir.is_dir():
        return False
    prefix = f"{task_id}_"
    return any(
        path.is_file() and path.name.startswith(prefix) and path.name.endswith(".log")
        for path in paths.log_dir.iterdir()
    )


def load_entries(
    paths: QueuePaths,
    *,
    exact_ids: frozenset[str] = frozenset(),
    include_orphans: bool = False,
) -> list[TaskEntry]:
    tasks = read_tasks(paths.tasks_file)
    task_ids = [task.task_id for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise DeletionError(f"task IDs are not unique in {paths.tasks_file}")

    entries = {
        task.task_id: TaskEntry(
            task_id=task.task_id,
            task=task,
            status=read_status(paths.status_dir, task.task_id),
        )
        for task in tasks
    }
    orphan_ids: set[str] = set()
    if include_orphans and paths.status_dir.is_dir():
        orphan_ids.update(path.stem for path in paths.status_dir.glob("*.status"))
        orphan_ids.update(
            path.name.removesuffix(".status.tmp")
            for path in paths.status_dir.glob("*.status.tmp")
        )
    orphan_ids.update(exact_ids - entries.keys())
    for task_id in sorted(orphan_ids):
        validate_task_id(task_id)
        if task_id in entries or not task_owned_paths_exist(paths, task_id):
            continue
        entries[task_id] = TaskEntry(
            task_id=task_id,
            task=None,
            status=read_status(paths.status_dir, task_id),
        )
    return list(entries.values())


def matches(entry: TaskEntry, selectors: Selectors) -> bool:
    if selectors.exact_ids and entry.task_id not in selectors.exact_ids:
        return False
    if selectors.id_globs and not any(
        fnmatch.fnmatchcase(entry.task_id, pattern) for pattern in selectors.id_globs
    ):
        return False
    if selectors.states and entry.state not in selectors.states:
        return False
    if selectors.env_names and entry.field("env_name", "env_name") not in selectors.env_names:
        return False
    if selectors.seeds and entry.field("seed", "seed") not in selectors.seeds:
        return False
    if selectors.modes and entry.field("mode", "mode") not in selectors.modes:
        return False
    if selectors.orphaned and entry.active:
        return False
    return True


def select_entries(paths: QueuePaths, selectors: Selectors) -> list[TaskEntry]:
    entries = load_entries(
        paths,
        exact_ids=selectors.exact_ids,
        include_orphans=selectors.orphaned,
    )
    selected = [entry for entry in entries if matches(entry, selectors)]
    selected.sort(key=lambda entry: entry.task_id)
    return selected


def status_output_path(paths: QueuePaths, entry: TaskEntry) -> Path:
    value = entry.status.get("output_dir", "")
    output_path = Path(value) if value else paths.results_root / entry.task_id
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path = output_path.resolve()
    results_root = paths.results_root.resolve()
    if output_path.parent != results_root or output_path.name != entry.task_id:
        raise DeletionError(
            f"refusing output path outside configured results root for {entry.task_id}: "
            f"{output_path}"
        )
    return output_path


def task_log_files(paths: QueuePaths, task_id: str) -> tuple[Path, ...]:
    if not paths.log_dir.is_dir():
        return ()
    prefix = f"{task_id}_"
    return tuple(sorted(
        path
        for path in paths.log_dir.iterdir()
        if path.is_file() and path.name.startswith(prefix) and path.name.endswith(".log")
    ))


def task_checkpoint_files(output_path: Path) -> tuple[Path, ...]:
    if not output_path.is_dir():
        return ()
    return tuple(sorted(
        path
        for path in output_path.glob("*.pth")
        if path.is_file() or path.is_symlink()
    ))


def build_artifact_plan(paths: QueuePaths, entries: Sequence[TaskEntry]) -> list[TaskArtifacts]:
    plan = []
    for entry in entries:
        output_path = status_output_path(paths, entry)
        plan.append(
            TaskArtifacts(
                entry=entry,
                status_files=(
                    paths.status_dir / f"{entry.task_id}.status",
                    paths.status_dir / f"{entry.task_id}.status.tmp",
                ),
                output_path=output_path,
                log_files=task_log_files(paths, entry.task_id),
                checkpoint_files=task_checkpoint_files(output_path),
            )
        )
    return plan


def format_bytes(size: int) -> str:
    value = float(size)
    for suffix in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or suffix == "TiB":
            return f"{value:.1f}{suffix}"
        value /= 1024.0
    raise AssertionError("unreachable")


def print_plan(plan: Sequence[TaskArtifacts], *, checkpoints_only: bool = False) -> None:
    print(f"Matched tasks: {len(plan)}")
    for item in plan:
        entry = item.entry
        checkpoint_details = ""
        if checkpoints_only:
            checkpoint_details = (
                f"\tcheckpoints={len(item.checkpoint_files)}"
                f"\tcheckpoint_bytes={format_bytes(item.checkpoint_bytes)}"
                f"\tmarked={str(item.checkpoints_deleted_marker.is_file()).lower()}"
            )
        print(
            f"{entry.task_id}\tstate={entry.state}\tactive={str(entry.active).lower()}"
            f"\tenv={entry.field('env_name', 'env_name')}\tseed={entry.field('seed', 'seed')}"
            f"\toutput={str(item.output_path) if item.output_path.exists() else '-'}"
            f"{checkpoint_details}"
            f"\tlogs={len(item.log_files)}"
            f"\tstatus={str(item.status_files[0]) if item.status_files[0].exists() else '-'}"
        )


@contextmanager
def exclusive_queue_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise DeletionError(
                "queue scheduler is active; stop it before deleting active tasks"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def remove_tasks_from_table(tasks_file: Path, task_ids: set[str]) -> int:
    raw_lines = tasks_file.read_text().splitlines(keepends=True)
    kept: list[str] = []
    removed = 0
    for raw_line in raw_lines:
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            kept.append(raw_line)
            continue
        task_id = raw_line.split("\t", 1)[0]
        if task_id in task_ids:
            removed += 1
        else:
            kept.append(raw_line)
    if removed == 0:
        return 0

    tasks_file.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{tasks_file.name}.", suffix=".tmp", dir=tasks_file.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write("".join(kept))
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, tasks_file.stat().st_mode)
        os.replace(temporary, tasks_file)
    finally:
        if temporary.exists():
            temporary.unlink()
    return removed


def task_table_archives(tasks_file: Path) -> tuple[Path, ...]:
    pattern = f"{tasks_file.name}.before_*"
    return tuple(sorted(
        path
        for path in tasks_file.parent.glob(pattern)
        if path.is_file() and not path.is_symlink()
    ))


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def write_checkpoints_deleted_marker(
    item: TaskArtifacts,
    checkpoint_bytes: int,
) -> None:
    payload = {
        "version": 1,
        "task_id": item.entry.task_id,
        "deleted_at": timestamp(),
        "checkpoint_count": len(item.checkpoint_files),
        "checkpoint_bytes": checkpoint_bytes,
    }
    marker = item.checkpoints_deleted_marker
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{marker.name}.", suffix=".tmp", dir=marker.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, marker)
    finally:
        if temporary.exists():
            temporary.unlink()


def remove_eta_history(path: Path, task_ids: set[str]) -> None:
    if not path.is_file():
        return
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise DeletionError(f"cannot update watcher ETA cache {path}: {exc}") from exc
    tasks = payload.get("tasks") if isinstance(payload, dict) else None
    if not isinstance(tasks, dict) or not task_ids.intersection(tasks):
        return
    payload["tasks"] = {
        task_id: record for task_id, record in tasks.items() if task_id not in task_ids
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, path.stat().st_mode)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def execute_deletion(paths: QueuePaths, selected: Sequence[TaskEntry]) -> None:
    if any(entry.state == "RUNNING" for entry in selected):
        running = ", ".join(entry.task_id for entry in selected if entry.state == "RUNNING")
        raise DeletionError(f"refusing to delete RUNNING task(s): {running}")

    selected_ids = {entry.task_id for entry in selected}
    lock = exclusive_queue_lock(paths.lock_file) if any(entry.active for entry in selected) else nullcontext()
    with lock:
        current = {
            entry.task_id: entry
            for entry in load_entries(
                paths,
                exact_ids=frozenset(selected_ids),
                include_orphans=True,
            )
            if entry.task_id in selected_ids
        }
        if set(current) != selected_ids:
            missing = ", ".join(sorted(selected_ids - current.keys()))
            raise DeletionError(f"task set changed before deletion; missing: {missing}")
        if any(entry.state == "RUNNING" for entry in current.values()):
            raise DeletionError("a selected task started running before deletion")

        plan = build_artifact_plan(paths, list(current.values()))
        remove_tasks_from_table(paths.tasks_file, selected_ids)
        for archive in task_table_archives(paths.tasks_file):
            remove_tasks_from_table(archive, selected_ids)
        remove_eta_history(paths.eta_sample_file, selected_ids)
        for item in plan:
            for log_file in item.log_files:
                remove_path(log_file)
            remove_path(item.output_path)
            for status_file in item.status_files:
                remove_path(status_file)


def execute_checkpoint_deletion(
    paths: QueuePaths,
    selected: Sequence[TaskEntry],
) -> int:
    not_done = [entry.task_id for entry in selected if entry.state != "DONE"]
    if not_done:
        raise DeletionError(
            "checkpoint-only deletion requires DONE task(s): " + ", ".join(not_done)
        )

    selected_ids = {entry.task_id for entry in selected}
    lock = (
        exclusive_queue_lock(paths.lock_file)
        if any(entry.active for entry in selected)
        else nullcontext()
    )
    with lock:
        current = {
            entry.task_id: entry
            for entry in load_entries(
                paths,
                exact_ids=frozenset(selected_ids),
                include_orphans=True,
            )
            if entry.task_id in selected_ids
        }
        if set(current) != selected_ids:
            missing = ", ".join(sorted(selected_ids - current.keys()))
            raise DeletionError(
                f"task set changed before checkpoint deletion; missing: {missing}"
            )
        no_longer_done = [
            entry.task_id for entry in current.values() if entry.state != "DONE"
        ]
        if no_longer_done:
            raise DeletionError(
                "task state changed before checkpoint deletion: "
                + ", ".join(no_longer_done)
            )

        plan = build_artifact_plan(paths, list(current.values()))
        missing_outputs = [
            item.entry.task_id for item in plan if not item.output_path.is_dir()
        ]
        if missing_outputs:
            raise DeletionError(
                "checkpoint-only deletion requires result directories: "
                + ", ".join(missing_outputs)
            )

        reclaimed = sum(item.checkpoint_bytes for item in plan)
        for item in plan:
            if (
                not item.checkpoint_files
                and item.checkpoints_deleted_marker.is_file()
            ):
                continue
            checkpoint_bytes = item.checkpoint_bytes
            for checkpoint in item.checkpoint_files:
                remove_path(checkpoint)
            write_checkpoints_deleted_marker(item, checkpoint_bytes)
        return reclaimed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Permanently delete QRL queue tasks, or only their checkpoints, "
            "by stable task_id or metadata. Without --yes, only a deletion "
            "plan is printed."
        )
    )
    parser.add_argument("--config", default="configs/qrl_queue.env")
    parser.add_argument("--tasks-file")
    parser.add_argument("--status-dir")
    parser.add_argument("--results-root")
    parser.add_argument("--log-dir")
    parser.add_argument("--eta-sample-file")
    parser.add_argument("--lock-file")
    parser.add_argument("--task-id", action="append", default=[], help="Exact stable task_id")
    parser.add_argument(
        "--task-id-file",
        action="append",
        default=[],
        help="File containing task IDs, or a task TSV whose first column is task_id",
    )
    parser.add_argument(
        "--task-id-glob", action="append", default=[], help="Shell-style task_id pattern"
    )
    parser.add_argument("--state", action="append", choices=KNOWN_STATES, default=[])
    parser.add_argument("--env-name", action="append", default=[])
    parser.add_argument("--seed", action="append", default=[])
    parser.add_argument("--mode", action="append", choices=("online", "offline"), default=[])
    parser.add_argument(
        "--orphaned",
        action="store_true",
        help="Only match status records whose task_id is absent from the active task table",
    )
    parser.add_argument(
        "--checkpoints-only",
        action="store_true",
        help=(
            "Delete every top-level *.pth file from matched DONE task result "
            "directories, preserving task records, logs, and result summaries"
        ),
    )
    parser.add_argument("--yes", action="store_true", help="Execute permanent deletion")
    parser.add_argument(
        "--expect",
        type=int,
        help="Required exact match count when --yes is used",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    paths = queue_paths(args)
    selectors = build_selectors(args)
    selected = select_entries(paths, selectors)
    plan = build_artifact_plan(paths, selected)
    print_plan(plan, checkpoints_only=args.checkpoints_only)

    if not args.yes:
        target = "their checkpoints" if args.checkpoints_only else "these tasks"
        print(f"Dry run only. Re-run with --yes --expect N to delete {target}.")
        return 0
    if args.expect is None:
        raise DeletionError("--yes requires --expect N")
    if args.expect != len(selected):
        raise DeletionError(
            f"expected {args.expect} matched tasks, but selector matched {len(selected)}"
        )
    if args.checkpoints_only:
        reclaimed = execute_checkpoint_deletion(paths, selected)
        print(
            f"Deleted checkpoints from {len(selected)} task(s); "
            f"reclaimed {format_bytes(reclaimed)}."
        )
    else:
        execute_deletion(paths, selected)
        print(f"Permanently deleted {len(selected)} task(s).")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except DeletionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Apply a live CPU-affinity limit to this user's QRL GPU training jobs."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BACKUP_DIR = ROOT / "logs/qrl_queue"


@dataclass
class GpuProcess:
    pid: int
    gpu_indices: tuple[int, ...]
    pci_bus_ids: tuple[str, ...]
    command: str
    start_time: str


def run_nvidia_smi(*query_args: str) -> list[list[str]]:
    result = subprocess.run(
        ["nvidia-smi", *query_args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"nvidia-smi failed: {detail}")
    return [
        [field.strip() for field in row]
        for row in csv.reader(result.stdout.splitlines())
        if row and any(field.strip() for field in row)
    ]


def normalize_pci_bus_id(value: str) -> str:
    match = re.fullmatch(
        r"(?:[0-9A-Fa-f]{4})?([0-9A-Fa-f]{4}):"
        r"([0-9A-Fa-f]{2}):([0-9A-Fa-f]{2}\.[0-7])",
        value.strip(),
    )
    if match is None:
        raise ValueError(f"Invalid PCI bus id: {value!r}")
    return ":".join(part.lower() for part in match.groups())


def parse_cpu_list(value: str) -> list[int]:
    cpus: list[int] = []
    for part in value.strip().split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(item) for item in part.split("-", 1))
            if end < start:
                raise ValueError(f"Invalid CPU range: {part!r}")
            cpus.extend(range(start, end + 1))
        else:
            cpus.append(int(part))
    return sorted(set(cpus))


def format_cpu_list(cpus: Iterable[int]) -> str:
    values = sorted(set(cpus))
    if not values:
        return ""
    ranges: list[str] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def process_command(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return " ".join(part.decode(errors="replace") for part in raw.split(b"\0") if part)


def process_start_time(pid: int) -> str:
    try:
        suffix = Path(f"/proc/{pid}/stat").read_text().rpartition(") ")[2]
        return suffix.split()[19]
    except (OSError, IndexError):
        return ""


def is_current_user_process(pid: int) -> bool:
    try:
        return Path(f"/proc/{pid}").stat().st_uid == os.geteuid()
    except OSError:
        return False


def is_qrl_training_command(command: str) -> bool:
    tokens = command.split()
    for index, token in enumerate(tokens[:-1]):
        if token == "-m" and tokens[index + 1] in {"online.main", "offline.main"}:
            return True
    return any(
        token.endswith(("/online/main.py", "/offline/main.py"))
        for token in tokens
    )


def discover_qrl_gpu_processes() -> list[GpuProcess]:
    gpu_rows = run_nvidia_smi(
        "--query-gpu=index,uuid,pci.bus_id", "--format=csv,noheader,nounits"
    )
    gpu_by_uuid: dict[str, tuple[int, str]] = {}
    for row in gpu_rows:
        if len(row) != 3:
            raise RuntimeError(f"Unexpected nvidia-smi GPU row: {row!r}")
        gpu_by_uuid[row[1]] = (int(row[0]), normalize_pci_bus_id(row[2]))

    process_rows = run_nvidia_smi(
        "--query-compute-apps=gpu_uuid,pid,process_name",
        "--format=csv,noheader,nounits",
    )
    gpu_processes: dict[int, set[tuple[int, str]]] = defaultdict(set)
    for row in process_rows:
        if len(row) != 3 or row[0] not in gpu_by_uuid:
            continue
        try:
            pid = int(row[1])
        except ValueError:
            continue
        gpu_processes[pid].add(gpu_by_uuid[row[0]])

    discovered: list[GpuProcess] = []
    for pid, gpu_records in gpu_processes.items():
        if not is_current_user_process(pid):
            continue
        command = process_command(pid)
        if not is_qrl_training_command(command):
            continue
        records = sorted(gpu_records)
        discovered.append(GpuProcess(
            pid=pid,
            gpu_indices=tuple(record[0] for record in records),
            pci_bus_ids=tuple(record[1] for record in records),
            command=command,
            start_time=process_start_time(pid),
        ))
    return sorted(discovered, key=lambda process: (process.gpu_indices, process.pid))


def available_cpus() -> list[int]:
    if hasattr(os, "sched_getaffinity"):
        cpus = sorted(os.sched_getaffinity(0))
        if cpus:
            return cpus
    return list(range(os.cpu_count() or 1))


def gpu_numa_node(process: GpuProcess) -> int | None:
    nodes = set()
    for pci_bus_id in process.pci_bus_ids:
        try:
            node = int(Path(
                f"/sys/bus/pci/devices/{pci_bus_id}/numa_node"
            ).read_text().strip())
        except (OSError, ValueError):
            continue
        if node >= 0:
            nodes.add(node)
    return next(iter(nodes)) if len(nodes) == 1 else None


def node_cpus(node: int | None, fallback: list[int]) -> list[int]:
    if node is None:
        return fallback
    try:
        cpus = parse_cpu_list(
            Path(f"/sys/devices/system/node/node{node}/cpulist").read_text()
        )
    except OSError:
        return fallback
    allowed = set(fallback)
    result = [cpu for cpu in cpus if cpu in allowed]
    return result or fallback


def physical_core_siblings(cpus: list[int]) -> list[list[int]]:
    allowed = set(cpus)
    cores: dict[tuple[str, str], list[int]] = defaultdict(list)
    for cpu in cpus:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            package = (topology / "physical_package_id").read_text().strip()
            core = (topology / "core_id").read_text().strip()
        except OSError:
            package, core = "", str(cpu)
        cores[package, core].append(cpu)
    return [
        sorted(cpu for cpu in siblings if cpu in allowed)
        for _, siblings in sorted(cores.items(), key=lambda item: min(item[1]))
    ]


def pack_affinity_groups(
    core_siblings: list[list[int]], max_logical_cpus: int
) -> list[list[int]]:
    if max_logical_cpus <= 0:
        raise ValueError("max_logical_cpus must be positive")
    groups: list[list[int]] = []
    current: list[int] = []
    for siblings in core_siblings:
        for start in range(0, len(siblings), max_logical_cpus):
            chunk = siblings[start:start + max_logical_cpus]
            if current and len(current) + len(chunk) > max_logical_cpus:
                groups.append(sorted(current))
                current = []
            current.extend(chunk)
            if len(current) == max_logical_cpus:
                groups.append(sorted(current))
                current = []
    if current:
        groups.append(sorted(current))
    return groups


def plan_affinities(
    processes: list[GpuProcess], max_logical_cpus: int
) -> dict[int, list[int]]:
    host_cpus = available_cpus()
    by_node: dict[int | None, list[GpuProcess]] = defaultdict(list)
    for process in processes:
        by_node[gpu_numa_node(process)].append(process)

    assignments: dict[int, list[int]] = {}
    for node, node_processes in sorted(
        by_node.items(), key=lambda item: (-1 if item[0] is None else item[0])
    ):
        cpus = node_cpus(node, host_cpus)
        groups = pack_affinity_groups(
            physical_core_siblings(cpus), max_logical_cpus
        )
        if not groups:
            raise RuntimeError(f"No CPUs available for NUMA node {node}")
        for index, process in enumerate(node_processes):
            assignments[process.pid] = groups[index % len(groups)]
    return assignments


def process_thread_ids(pid: int) -> list[int]:
    try:
        return sorted(
            int(path.name) for path in Path(f"/proc/{pid}/task").iterdir()
            if path.name.isdigit()
        )
    except OSError:
        return []


def affinity_backup(
    processes: list[GpuProcess], assignments: dict[int, list[int]]
) -> dict[str, object]:
    records = []
    for process in processes:
        thread_affinities = {}
        for tid in process_thread_ids(process.pid):
            try:
                thread_affinities[str(tid)] = sorted(os.sched_getaffinity(tid))
            except (OSError, ProcessLookupError):
                continue
        records.append({
            "pid": process.pid,
            "start_time": process.start_time,
            "gpus": list(process.gpu_indices),
            "command": process.command,
            "assigned_cpus": assignments[process.pid],
            "thread_affinities": thread_affinities,
        })
    return {
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "processes": records,
    }


def write_backup(backup: dict[str, object], backup_dir: Path) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = backup_dir / f"cpu_affinity_backup_{stamp}.json"
    path.write_text(json.dumps(backup, indent=2) + "\n")
    return path


def apply_affinity(pid: int, cpus: list[int]) -> tuple[int, list[str]]:
    errors: list[str] = []
    updated: set[int] = set()
    # Repeat because a library can create another worker while the first pass
    # is walking /proc/PID/task.
    for _ in range(3):
        tids = process_thread_ids(pid)
        pending = [tid for tid in tids if tid not in updated]
        if not pending:
            break
        for tid in pending:
            try:
                os.sched_setaffinity(tid, cpus)
                updated.add(tid)
            except (OSError, ProcessLookupError) as exc:
                errors.append(f"tid={tid}: {exc}")
    return len(updated), errors


def verify_affinity(pid: int, expected: list[int]) -> list[str]:
    expected_set = set(expected)
    issues = []
    for tid in process_thread_ids(pid):
        try:
            actual = set(os.sched_getaffinity(tid))
        except (OSError, ProcessLookupError):
            continue
        if actual != expected_set:
            issues.append(
                f"tid={tid} actual={format_cpu_list(actual)} "
                f"expected={format_cpu_list(expected_set)}"
            )
    return issues


def restore_affinities(path: Path) -> int:
    backup = json.loads(path.read_text())
    failures = 0
    for record in backup.get("processes", []):
        pid = int(record["pid"])
        if process_start_time(pid) != record.get("start_time"):
            print(f"SKIP pid={pid}: process exited or PID was reused")
            continue
        restored = 0
        for tid_value, cpus in record.get("thread_affinities", {}).items():
            try:
                os.sched_setaffinity(int(tid_value), cpus)
                restored += 1
            except (OSError, ProcessLookupError) as exc:
                print(f"FAILED pid={pid} tid={tid_value}: {exc}", file=sys.stderr)
                failures += 1
        print(f"RESTORED pid={pid} threads={restored}")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--restore", type=Path)
    parser.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUP_DIR)
    args = parser.parse_args()

    if args.restore is not None:
        if args.apply:
            parser.error("--apply and --restore are mutually exclusive")
        return restore_affinities(args.restore)
    if args.threads <= 0:
        parser.error("--threads must be positive")

    processes = discover_qrl_gpu_processes()
    if not processes:
        print("No current-user QRL GPU training processes found.")
        return 0
    assignments = plan_affinities(processes, args.threads)
    for process in processes:
        node = gpu_numa_node(process)
        print(
            f"pid={process.pid} gpu={','.join(map(str, process.gpu_indices))} "
            f"numa={node if node is not None else 'unknown'} "
            f"cpus={format_cpu_list(assignments[process.pid])} "
            f"command={process.command}"
        )

    if not args.apply:
        print("Preview only; rerun with --apply to change live affinities.")
        return 0

    backup_path = write_backup(
        affinity_backup(processes, assignments), args.backup_dir
    )
    print(f"Affinity backup: {backup_path}")
    failures = 0
    for process in processes:
        if process_start_time(process.pid) != process.start_time:
            print(
                f"SKIP pid={process.pid}: process exited or PID was reused",
                file=sys.stderr,
            )
            continue
        updated, errors = apply_affinity(
            process.pid, assignments[process.pid]
        )
        issues = verify_affinity(process.pid, assignments[process.pid])
        if errors or issues:
            failures += 1
            for issue in [*errors, *issues[:10]]:
                print(f"FAILED pid={process.pid}: {issue}", file=sys.stderr)
        else:
            print(
                f"APPLIED pid={process.pid} threads={updated} "
                f"cpus={format_cpu_list(assignments[process.pid])}"
            )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

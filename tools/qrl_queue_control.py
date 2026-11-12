"""Start or stop the queue using Python process and lock operations."""
from __future__ import annotations
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

try:
    from .watch_qrl_queue import ROOT, cfg, parse_config, resolve_path
except ImportError:
    from watch_qrl_queue import ROOT, cfg, parse_config, resolve_path


def lock_is_held(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open("r") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle, fcntl.LOCK_UN)
    return False


def lock_pid(path: Path) -> int | None:
    try:
        for line in path.read_text().splitlines():
            if line.startswith("pid=") and line[4:].isdigit():
                return int(line[4:])
    except OSError:
        pass
    return None


def scheduler_pid(pid: int) -> bool:
    if pid <= 1:
        return False
    try:
        proc = Path(f"/proc/{pid}")
        arguments = (proc / "cmdline").read_bytes().split(b"\0")
        return proc.stat().st_uid == os.getuid() and any(
            argument == b"tools/run_qrl_queue.py" or argument.endswith(b"/tools/run_qrl_queue.py")
            for argument in arguments
        )
    except OSError:
        return False


def scheduler_pids(path: Path) -> list[int]:
    candidates = {lock_pid(path)}
    st = path.stat()
    for line in Path("/proc/locks").read_text().splitlines():
        fields = line.split()
        if len(fields) < 6 or fields[1] != "FLOCK":
            continue
        try:
            major, minor, inode = fields[5].split(":")
            if (int(major, 16), int(minor, 16), int(inode)) == (
                os.major(st.st_dev), os.minor(st.st_dev), st.st_ino
            ):
                candidates.add(int(fields[4]))
        except ValueError:
            continue
    return sorted(pid for pid in candidates if pid is not None and scheduler_pid(pid))


def stop(path: Path, with_jobs: bool = False) -> int:
    if not lock_is_held(path):
        print(f"QRL queue scheduler is not running; lock is free: {path}")
        return 0
    pids = scheduler_pids(path)
    if not pids:
        raise RuntimeError(f"Queue lock is held, but its scheduler could not be identified: {path}")
    if with_jobs:
        groups = {os.getpgid(pid) for pid in pids}
        if any(group <= 1 or group == os.getpgrp() for group in groups):
            raise RuntimeError("Refusing to signal the current or an unsafe process group")
        for group in groups:
            os.killpg(group, signal.SIGTERM)
    else:
        for pid in pids:
            os.kill(pid, signal.SIGTERM)
            print(f"Sent SIGTERM to QRL scheduler PID {pid}; training jobs remain active.")
    deadline = time.monotonic() + 10
    while lock_is_held(path) and time.monotonic() < deadline:
        time.sleep(0.1)
    if lock_is_held(path):
        raise RuntimeError(f"Queue lock is still held after 10 seconds: {path}")
    print("QRL queue scheduler stopped; queue lock released.")
    return 0


def start(config_path: Path, arguments: list[str]) -> int:
    config = parse_config(config_path)
    lock = resolve_path(os.environ.get("QRL_QUEUE_LOCK_FILE") or cfg(config, "LOCK_FILE", "runs/qrl_queue/qrl_queue.lock"))
    if lock_is_held(lock):
        raise RuntimeError(f"Another QRL queue scheduler is already active: {lock}")
    log_dir = resolve_path(cfg(config, "LOG_DIR", "logs/qrl_queue"))
    output_dir = resolve_path(cfg(config, "RESULTS_ROOT", "../qrl-assets/results/queue"))
    log_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(os.environ.get("QRL_QUEUE_RUNNER_LOG", str(log_dir / "runner.log")))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    python = os.environ.get("QRL_PYTHON_BIN", cfg(config, "QRL_PYTHON_BIN", str(sys.executable)))
    if "/" in python:
        python = str(resolve_path(python))
    env = dict(os.environ, LOCK_FILE=str(lock))
    command = [python, "tools/run_qrl_queue.py", "--config", str(config_path), *arguments]
    with log_path.open("ab") as log:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        code = proc.poll()
        if code is not None:
            if code == 0:
                print(f"QRL queue runner exited cleanly; log: {log_path}")
                return 0
            with log_path.open("rb") as log:
                log.seek(max(0, log_path.stat().st_size - 3000))
                detail = log.read().decode(errors="replace")
            raise RuntimeError(f"Queue runner exited with {code}; log: {log_path}\n{detail}")
        if lock_pid(lock) == proc.pid and lock_is_held(lock):
            print(f"QRL queue runner started in the background (PID {proc.pid}); log: {log_path}")
            print(f"Results: {output_dir}")
            return 0
        time.sleep(0.1)
    proc.terminate()
    raise RuntimeError(f"Queue runner did not acquire its lock within 30 seconds; log: {log_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("start", "stop"))
    parser.add_argument("--config", default="configs/qrl_queue.env")
    parser.add_argument("--lock-file")
    parser.add_argument("--with-jobs", action="store_true")
    args, rest = parser.parse_known_args()
    if args.action == "stop":
        if rest:
            parser.error(f"Unrecognized stop arguments: {rest}")
        path = resolve_path(args.lock_file or os.environ.get("QRL_QUEUE_LOCK_FILE", "runs/qrl_queue/qrl_queue.lock"))
        return stop(path, args.with_jobs)
    return start(resolve_path(args.config), rest)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1)

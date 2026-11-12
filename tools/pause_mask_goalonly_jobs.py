#!/usr/bin/env python3
"""Pause the selected rebuttal jobs gracefully and retain their logs."""
from __future__ import annotations
import argparse, os, signal, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qrl_queue_control as control
import run_qrl_queue as q
import watch_qrl_queue as w

VARIANTS = ("QRL-Mask", "QRL-GoalOnly")
ENVS = {"FetchPickAndPlace", "maze2d-large", "reacher_hard"}
SEEDS = {"1000", "1001", "1002"}
EXPECTED = len(VARIANTS) * len(ENVS) * len(SEEDS)

def selected(t):
    return (t.mode == "online" and t.env_name in ENVS and t.seed in SEEDS
            and any(t.task_id.startswith(f"iclr_rebuttal_v1_{v}-L_") for v in VARIANTS))

def matches(pid, task):
    proc = Path(f"/proc/{pid}")
    try:
        if proc.stat().st_uid != os.getuid():
            return False
        args = set(x for x in (proc / "cmdline").read_bytes().split(b"\0") if x)
    except OSError:
        return False
    required = {b"-m", b"online.main", f"output_folder={task.task_id}".encode(),
                f"seed={task.seed}".encode(), f"env.name={task.env_name}".encode()}
    return required <= args

def running(pid, task):
    if not matches(pid, task):
        return False
    try:
        data = Path(f"/proc/{pid}/stat").read_text()
        return data[data.rfind(")") + 2] not in "ZX"
    except OSError:
        return False

def extras(status, reason):
    fixed = {"state", "updated_at", "task_id", "task_fingerprint", "mode",
             "env_name", "seed", "steps", "params", "extra_args", "pid", "gpu",
             "finished_at"}
    extra = {k:v for k,v in status.items() if k not in fixed}
    extra["previous_pid"] = status.get("pid") or status.get("previous_pid", "")
    extra["previous_gpu"] = status.get("gpu") or status.get("previous_gpu", "")
    extra["error"] = reason
    return extra

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/qrl_queue.env")
    args = parser.parse_args()
    config_path = w.resolve_path(args.config)
    cfg = q.parse_config(config_path)
    tasks = q.read_tasks(q.resolve_path(q.cfg(cfg, "TASKS_FILE", "runs/qrl_queue/tasks.tsv")))
    status_dir = q.resolve_path(q.cfg(cfg, "STATUS_DIR", "runs/qrl_queue/status"))
    results = q.resolve_path(q.cfg(cfg, "RESULTS_ROOT", "../qrl-assets/results/queue"))
    lock_path = Path(os.environ.get("QRL_QUEUE_LOCK_FILE")
                     or q.cfg(cfg, "LOCK_FILE", "runs/qrl_queue/qrl_queue.lock"))
    if not lock_path.is_absolute():
        lock_path = q.ROOT / lock_path

    targets = [t for t in tasks if selected(t)]
    if len(targets) != EXPECTED or len({t.task_id for t in targets}) != EXPECTED:
        raise RuntimeError(f"Expected {EXPECTED} unique selected tasks; found {len(targets)}. Nothing changed.")
    for t in targets:
        s = q.read_status(status_dir, t.task_id)
        if s.get("task_fingerprint") != q.task_fingerprint(t):
            raise RuntimeError(f"Task fingerprint mismatch: {t.task_id}. Nothing changed.")

    scheduler_active = control.lock_is_held(lock_path)
    if scheduler_active:
        control.stop(lock_path, with_jobs=False)
    lock = q.QueueLock(lock_path)
    if not lock.acquire():
        if scheduler_active:
            control.start(config_path, [])
        raise RuntimeError("Could not acquire queue lock; no jobs were signaled.")

    keep_scheduler_stopped = False
    try:
        signals, pauses, completed = [], [], []
        for t in targets:
            s = q.read_status(status_dir, t.task_id)
            if s.get("task_fingerprint") != q.task_fingerprint(t):
                raise RuntimeError(f"Task changed after scheduler stop: {t.task_id}.")
            state = s.get("state", "PENDING")
            if state == "RUNNING":
                expected = (results / t.task_id).resolve()
                actual = Path(s.get("output_dir") or expected).resolve()
                if actual != expected:
                    raise RuntimeError(f"Unexpected output path for {t.task_id}.")
                if q.output_finished_for_task(actual, t):
                    completed.append((t, s))
                    continue
                try:
                    pid = int(s.get("pid", "0"))
                except ValueError:
                    pid = 0
                if pid > 1 and matches(pid, t):
                    signals.append((t, pid, s))
                else:
                    pauses.append((t, s, "user_requested_pause_process_already_stopped"))
            elif state == "PENDING":
                pauses.append((t, s, "user_requested_pause_preserve_logs"))
            elif state not in {"DONE", "PAUSED", "EARLY_STOPPED", "FAILED"}:
                raise RuntimeError(f"Unexpected state {state!r}: {t.task_id}.")

        for t, pid, _ in signals:
            if not matches(pid, t):
                raise RuntimeError(f"PID identity changed for {t.task_id}; no signals sent.")

        keep_scheduler_stopped = True
        for t, pid, _ in signals:
            os.kill(pid, signal.SIGTERM)
            print(f"SIGTERM {t.task_id} pid={pid}", flush=True)
        deadline = time.monotonic() + 60
        left = [(t, p) for t, p, _ in signals]
        while left and time.monotonic() < deadline:
            left = [(t, p) for t, p in left if running(p, t)]
            if left:
                time.sleep(0.25)
        if left:
            detail = ", ".join(f"{t.task_id}({p})" for t, p in left)
            raise RuntimeError(f"Still running after SIGTERM: {detail}. Scheduler remains stopped.")

        for t, s, reason in pauses:
            q.write_status(status_dir, t, "PAUSED", extras(s, reason))
        for t, _, s in signals:
            out = Path(s.get("output_dir") or results / t.task_id)
            state = "DONE" if q.output_finished_for_task(out, t) else "PAUSED"
            reason = "completed_during_pause" if state == "DONE" else "user_requested_pause_preserve_logs"
            q.write_status(status_dir, t, state, extras(s, reason))
        for t, s in completed:
            q.write_status(status_dir, t, "DONE", extras(s, "completed_before_pause"))
        keep_scheduler_stopped = False
    finally:
        lock.release()
        if scheduler_active and not keep_scheduler_stopped:
            control.start(config_path, [])

    paused = sum(q.read_status(status_dir, t.task_id).get("state") == "PAUSED" for t in targets)
    done = sum(q.read_status(status_dir, t.task_id).get("state") == "DONE" for t in targets)
    print(f"Paused {paused}; DONE {done}; existing logs retained.")
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1)

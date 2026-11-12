"""Read validation curves for the explicitly listed QRL reproduction runs."""

from __future__ import annotations

import json
import math
from pathlib import Path


def validation_at_or_after(path: Path, *, target_steps: int):
    """Find a completed validation at or after the requested step limit."""
    if target_steps <= 0:
        raise ValueError("target_steps must be positive")
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return None
    evaluations = {}
    for line in lines:
        try:
            item = json.loads(line)
            if not isinstance(item, dict) or item.get("split", "validation") != "validation":
                continue
            step = int(item["env_steps"])
            success = float(item["succ_rate"])
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            continue
        if step >= 0 and math.isfinite(success) and 0.0 <= success <= 1.0:
            evaluations[step] = success
    eligible = [step for step in evaluations if step >= target_steps]
    if not eligible:
        return None
    trigger = min(eligible)
    latest = max(evaluations)
    return {
        "target_env_steps": target_steps,
        "trigger_validation_env_steps": trigger,
        "trigger_validation_success": evaluations[trigger],
        "stop_env_steps": latest,
        "last_validation_success": evaluations[latest],
        "validation_count": len(evaluations),
    }


def plateau_from_eval_log(path: Path, *, min_steps: int, patience_steps: int):
    """Return the evidence for a plateau, or None if the rule is not met.

    Each strict increase in the best validation success resets the patience
    counter.  Original QRL's eval.log has no split field; our implementation
    labels validation and test explicitly.  Incomplete trailing JSON is ignored.
    """
    if min_steps < 0 or patience_steps <= 0:
        raise ValueError("invalid plateau parameters")
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return None

    evaluations = {}
    for line in lines:
        try:
            item = json.loads(line)
            if not isinstance(item, dict) or item.get("split", "validation") != "validation":
                continue
            step = int(item["env_steps"])
            success = float(item["succ_rate"])
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            continue
        if step >= 0 and math.isfinite(success) and 0.0 <= success <= 1.0:
            evaluations[step] = success
    if len(evaluations) < 2:
        return None

    best = -1.0
    best_step = 0
    for step, success in sorted(evaluations.items()):
        if success > best + 1e-12:
            best = success
            best_step = step
    last_step = max(evaluations)
    if last_step < min_steps or last_step - best_step < patience_steps:
        return None
    return {
        "stop_env_steps": last_step,
        "last_validation_success": evaluations[last_step],
        "best_validation_success": best,
        "best_validation_env_steps": best_step,
        "validation_count": len(evaluations),
        "min_env_steps": min_steps,
        "patience_env_steps": patience_steps,
    }

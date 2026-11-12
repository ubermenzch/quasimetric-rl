#!/usr/bin/env python3
"""Export the completed baseline sanity checks cited by the NQRL draft."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "iclr2027/data/baseline_validation.csv"
CASES = (
    ("CRL", "FetchReach", "paper_validation_v1_CRL_FetchReach_s1000", "succ_rate"),
    ("GCSL", "FourRooms", "paper_validation_v1_GCSL_GCSLFourRooms_s1000", "reference_final_goal_distance"),
    ("CRL", "Point Small", "crl_quick_v1_CRL_CRLPointSmall_s1000", "final_goal_distance"),
)


def records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize() -> list[dict]:
    summaries = []
    for algorithm, task, task_id, metric in CASES:
        status = dict(
            line.split("=", 1)
            for line in (ROOT / "runs/qrl_queue/status" / f"{task_id}.status").read_text().splitlines()
            if "=" in line
        )
        assert status["state"] == "DONE", f"Unfinished sanity check: {task_id}"
        directory = Path(status["output_dir"])
        evaluations = records(directory / "eval.log")
        tests = records(directory / "test.log")
        assert len(tests) == 1
        first, last = evaluations[0], evaluations[-1]
        assert last["env_steps"] == int(status["steps"])
        summaries.append({
            "algorithm": algorithm,
            "task": task,
            "training_seed": int(status["seed"]),
            "metric": metric,
            "first_env_steps": first["env_steps"],
            "last_env_steps": last["env_steps"],
            "first_value": first[metric],
            "last_value": last[metric],
            "validation_episodes": last["num_episodes"],
            "selected_test_success": tests[0]["succ_rate"],
            "test_episodes": tests[0]["num_episodes"],
            "first_reference_final_success": first.get("reference_final_success_rate", ""),
            "last_reference_final_success": last.get("reference_final_success_rate", ""),
        })
    return summaries


def main() -> None:
    rows = summarize()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} completed sanity checks to {OUTPUT}")


if __name__ == "__main__":
    main()

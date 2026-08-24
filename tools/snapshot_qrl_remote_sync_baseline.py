#!/usr/bin/env python3
"""Record result directories that must remain local-only for remote syncing."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_qrl_queue import parse_config, resolve_path, results_root  # noqa: E402


def snapshot(config_path: Path, output: Path) -> int:
    config = parse_config(config_path)
    source = results_root(config)
    if not source.is_dir():
        raise FileNotFoundError(f"Result root does not exist: {source}")
    task_ids = sorted(path.name for path in source.iterdir() if path.is_dir())
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing baseline: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        "# QRL remote-sync local-only baseline.\n",
        f"# Created: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"# Source: {source}\n",
        *[f"{task_id}\n" for task_id in task_ids],
    ]
    temp = output.with_suffix(output.suffix + ".tmp")
    temp.write_text("".join(payload))
    temp.replace(output)
    print(f"Recorded {len(task_ids)} existing result directories in {output}")
    return len(task_ids)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/qrl_queue.env")
    parser.add_argument(
        "--output",
        default="runs/qrl_queue/remote_sync_225/local_only_baseline.txt",
    )
    args = parser.parse_args()
    try:
        snapshot(resolve_path(args.config), resolve_path(args.output))
    except (FileExistsError, FileNotFoundError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

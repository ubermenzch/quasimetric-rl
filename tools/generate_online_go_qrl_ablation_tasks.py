#!/usr/bin/env python3
"""Compatibility entry point for the renamed CQRL ablation generator."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.generate_cqrl_ablation_tasks import *  # noqa: F401,F403
from tools.generate_cqrl_ablation_tasks import main


if __name__ == '__main__':
    raise SystemExit(main())

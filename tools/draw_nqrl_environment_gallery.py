#!/usr/bin/env python3
"""Compose the fourteen archived simulator frames into one 4x4 gallery."""

import json
import hashlib
from pathlib import Path

import nqrl_latex_figure as canvas
from reportlab.lib.utils import ImageReader

from nqrl_figure_style import FONT, register_fonts, task_title

ROOT = Path(__file__).resolve().parents[1] / "iclr2027"

DESCRIPTIONS = {
    "antmaze": "Torso XY position",
    "antnavigate": "Torso XY position",
    "fetchpick": "Object XYZ position",
    "fetchpush": "Object XYZ position",
    "fetchslide": "Object XYZ position",
    "maze2d": "Point XY position",
    "pandaflip": "Object orientation; end-effector control",
    "pandapick": "Object XYZ position; end-effector control",
    "pandapickj": "Object XYZ position; joint control",
    "pandapushj": "Object XYZ position; joint control",
    "pandaslide": "Object XYZ position; end-effector control",
    "pointmaze": "Point XY position",
    "pusher": "Object XYZ position",
    "reacher": "Fingertip XY position",
}


def main():
    register_fonts()
    records = json.loads((ROOT / "data/environment_visualization_audit.json").read_text())
    assert len(records) == 14 and {r["key"] for r in records} == set(DESCRIPTIONS)
    assert [r['environment'] for r in records] == sorted(
        (r['environment'] for r in records), key=str.casefold)
    pdf = canvas.Canvas(str(ROOT / 'figures/task_environments_grid.pdf'),
                        pagesize=(576, 424), invariant=1, initialFontName=FONT)
    pdf.setTitle('Fourteen simulation environments')
    for i, record in enumerate(records):
        source = ROOT / record['image']
        assert hashlib.sha256(source.read_bytes()).hexdigest() == record['image_sha256']
        left = 8 + (i % 4) * 142
        top = 417 - (i // 4) * 104
        task_title(pdf, left, top, 133, record['environment'])
        pdf.drawImage(ImageReader(str(source)),
                      left, top - 99, 133, 133 * 432 / 768)
    pdf.save()
    print('Composed all 14 archived environments in a 4x4 grid; two cells remain empty.')


if __name__ == "__main__":
    main()

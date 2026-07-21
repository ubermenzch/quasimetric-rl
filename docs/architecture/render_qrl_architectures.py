#!/usr/bin/env python3
"""Render the generated QRL SVG diagrams to PNG with librsvg and Cairo."""

from __future__ import annotations

import warnings
from pathlib import Path

import cairo
import gi


gi.require_version("Rsvg", "2.0")
from gi.repository import Rsvg  # noqa: E402


ROOT = Path(__file__).resolve().parent
DIAGRAMS = {
    "qrl_1q_base_maze2d": (2000, 1080),
    "qrl_1q_split_latent_max8_maze2d": (2400, 1680),
}


def render(name: str, width: int, height: int) -> Path:
    source = ROOT / f"{name}.svg"
    output = ROOT / f"{name}.png"
    handle = Rsvg.Handle.new_from_file(str(source))
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        handle.render_cairo(cairo.Context(surface))
    surface.write_to_png(str(output))
    return output


def main() -> None:
    for name, dimensions in DIAGRAMS.items():
        print(render(name, *dimensions))


if __name__ == "__main__":
    main()

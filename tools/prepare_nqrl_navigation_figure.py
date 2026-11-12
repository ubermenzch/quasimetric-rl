#!/usr/bin/env python3
"""Crop the title band from the author's navigation illustration."""

from pathlib import Path

from PIL import Image


FIGURES = Path(__file__).resolve().parents[1] / "iclr2027/figures"
CROP_TOP = 280


def main():
    source = FIGURES / "navigation_goal_concept.png"
    output = FIGURES / "navigation_goal_concept_cropped.png"
    with Image.open(source) as image:
        assert image.size == (4054, 1663), "Recheck the title boundary for new artwork"
        image.crop((0, CROP_TOP, image.width, image.height)).save(output)
    print(f"Cropped the title band without resampling: {output}")


if __name__ == "__main__":
    main()

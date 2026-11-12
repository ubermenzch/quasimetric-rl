#!/usr/bin/env python3
"""Draw the policy interface and conditional completion as a vector figure."""

from math import atan2, cos, sin
from pathlib import Path

from reportlab.lib.colors import HexColor
import nqrl_latex_figure as canvas
from nqrl_latex_figure import math_source

from nqrl_figure_style import (
    BOLD, CONTEXT, CONTEXT_EDGE, GOAL, GOAL_EDGE, GRID, INK, MATH,
    MUTED, label, register_fonts,
)


OUTPUT = Path(__file__).resolve().parents[1] / "iclr2027/figures/nqrl_architecture.pdf"


def main():
    register_fonts()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(OUTPUT), pagesize=(560, 304), invariant=1)
    pdf.setTitle("NQRL: policy inputs and nearest conditional completion")
    pdf.setAuthor("Anonymous")

    def math_text(x, y, width, text, size=13):
        pdf.latex_label(x+width/2,y,math_source(text),size,
                        align='center',math=True,center_y=True,color=INK)

    def block(x, y, width, height, text, fill, edge, size=13):
        pdf.setFillColor(fill)
        pdf.setStrokeColor(edge)
        pdf.setLineWidth(.8)
        pdf.roundRect(x, y, width, height, 2, fill=1, stroke=1)
        math_text(x + 3, y + height / 2 + 1, width - 6, text, size)

    def latent(x, y, width, text_goal, text_context):
        for offset, text, fill, edge in (
            (0, text_goal, GOAL, GOAL_EDGE),
            (width / 2, text_context, CONTEXT, CONTEXT_EDGE),
        ):
            pdf.setFillColor(fill)
            pdf.setStrokeColor(edge)
            pdf.setLineWidth(.8)
            pdf.rect(x + offset, y, width / 2, 29, fill=1, stroke=1)
            math_text(x + offset + 2, y + 15, width / 2 - 4, text, 12)

    def arrow(points, *, dashed=False, color=MUTED):
        pdf.setStrokeColor(color)
        pdf.setFillColor(color)
        pdf.setLineWidth(.9)
        pdf.setDash(3, 2) if dashed else pdf.setDash()
        path = pdf.beginPath()
        path.moveTo(*points[0])
        for point in points[1:]:
            path.lineTo(*point)
        pdf.drawPath(path)
        pdf.setDash()
        x, y = points[-1]
        px, py = points[-2]
        angle = atan2(y - py, x - px)
        head = pdf.beginPath()
        head.moveTo(x, y)
        for offset in (2.65, -2.65):
            head.lineTo(x + 4.5 * cos(angle + offset), y + 4.5 * sin(angle + offset))
        head.close()
        pdf.drawPath(head, fill=1, stroke=0)

    label(pdf, 12, 287, "(a) Policy inputs and action", 12, font=BOLD)
    for x, text, fill, edge in (
        (354, "Goal", GOAL, GOAL_EDGE),
        (445, "Non-goal", CONTEXT, CONTEXT_EDGE),
    ):
        pdf.setFillColor(fill)
        pdf.setStrokeColor(edge)
        pdf.rect(x, 287, 7, 7, fill=1, stroke=1)
        label(pdf, x + 12, 287, text, 10)

    label(pdf, 13, 254, "Current state", 10)
    math_text(12, 227, 58, "<i>s</i>", 18)
    label(pdf, 120, 260, "Encoders", 10, color=MUTED)
    block(113, 222, 67, 29, "<i>e</i><sub>g</sub>", GOAL, GOAL_EDGE)
    block(113, 179, 67, 29, "<i>e</i><sub>n</sub>", CONTEXT, CONTEXT_EDGE)
    arrow([(64, 228), (84, 228), (84, 237), (113, 237)])
    arrow([(84, 228), (84, 194), (113, 194)])
    label(pdf, 221, 251, "Full state encoding", 10, color=MUTED)
    latent(221, 211, 86, "<i>z</i><sub>g</sub>(<i>s</i>)", "<i>z</i><sub>n</sub>(<i>s</i>)")
    arrow([(180, 237), (196, 237), (196, 226), (221, 226)])
    arrow([(180, 194), (285.5, 194), (285.5, 211)])

    label(pdf, 13, 166, "Goal conditions", 10)
    math_text(12, 143, 58, "<i>g</i>", 18)
    block(113, 128, 67, 29, "<i>e</i><sub>g</sub>", GOAL, GOAL_EDGE)
    block(221, 128, 86, 29, "<i>e</i><sub>g</sub>(<i>g</i>)", GOAL, GOAL_EDGE)
    arrow([(64, 143), (113, 143)])
    arrow([(180, 143), (221, 143)])
    label(pdf, 221, 166, "Goal encoding", 10, color=MUTED)

    label(pdf, 374, 251, "Policy", 10, color=MUTED)
    block(366, 209, 79, 33, "\u03c0",
          HexColor("#ECE7F2"), HexColor("#806796"), size=16)
    arrow([(307, 226), (366, 226)])
    arrow([(307, 143), (391, 143), (391, 209)])
    arrow([(445, 226), (495, 226)])
    label(pdf, 504, 251, "Action", 10, color=MUTED)
    math_text(495, 226, 49, "<i>a</i>", 18)

    pdf.setStrokeColor(GRID)
    pdf.setLineWidth(.7)
    pdf.line(12, 114, 548, 114)
    label(pdf, 12, 97, "(b) Nearest conditional completion", 12, font=BOLD)
    label(pdf, 378, 97, "During policy training", 10, color=MUTED)
    math_text(10, 53, 66, "<i>E</i>(<i>s</i>), <i>a</i>", 12)
    block(97, 37, 51, 32, "<i>F</i><sub>\u03d5</sub>",
          HexColor("#F1F3F5"), HexColor("#A2ADB7"), size=15)
    arrow([(77, 53), (97, 53)])
    arrow([(148, 53), (191, 53)])
    label(pdf, 196, 78, "Fixed prediction", 10, color=MUTED)
    latent(191, 38, 89, "<i>\u1e91</i><sub>g</sub><sup>+</sup>", "<i>\u1e91</i><sub>n</sub><sup>+</sup>")
    arrow([(280, 53), (329, 53)])
    label(pdf, 326, 78, "Distance", 10, color=MUTED)
    block(329, 38, 55, 30, "<i>d</i><sub>\u03b8</sub>",
          HexColor("#F1F3F5"), HexColor("#A2ADB7"), size=15)
    pdf.setStrokeColor(INK)
    pdf.setLineWidth(.6)
    pdf.line(350, 59, 357, 59)
    latent(446, 38, 102, "<i>e</i><sub>g</sub>(<i>g</i>)", "<i>h</i>")
    label(pdf, 471.5, 78, "Fixed", 10, color=GOAL_EDGE, align="center")
    label(pdf, 522.5, 78, "Searched", 10, color=CONTEXT_EDGE, align="center")
    arrow([(446, 53), (384, 53)])
    arrow([(356, 38), (356, 13), (522.5, 13), (522.5, 38)],
          dashed=True, color=CONTEXT_EDGE)
    label(pdf, 429, 18, "Gradient descent on h", 10,
          color=CONTEXT_EDGE, align="center")
    pdf.save()
    print(OUTPUT)


if __name__ == "__main__":
    main()

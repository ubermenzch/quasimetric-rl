#!/usr/bin/env python3
"""Generate publication-ready SVG diagrams for the Maze2D QRL variants.

The renderer intentionally uses only the Python standard library.  This keeps
the diagrams reproducible on training machines without a TeX or Graphviz
installation while retaining editable vector output.
"""

from __future__ import annotations

import argparse
import html
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parent


COLORS = {
    "ink": "#17212B",
    "muted": "#5B6773",
    "line": "#83909C",
    "faint": "#DCE2E7",
    "paper": "#FFFFFF",
    "panel": "#F7F9FA",
    "input": "#E8EDF1",
    "input_side": "#CAD3DB",
    "encoder": "#A9D7E8",
    "encoder_side": "#68AEC9",
    "actor": "#AFD6B2",
    "actor_side": "#71AF77",
    "dynamics": "#F2C27E",
    "dynamics_side": "#D9993E",
    "metric": "#D6C4E9",
    "metric_side": "#9B78BF",
    "optimizer": "#F2A6A0",
    "optimizer_side": "#D8645C",
    "red": "#C6463D",
    "green": "#347A3D",
    "blue": "#26799B",
    "orange": "#A96512",
    "purple": "#70459A",
}


class SVG:
    def __init__(self, width: int, height: int, title: str):
        self.width = width
        self.height = height
        self.parts: list[str] = []
        self.parts.append(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
            f'aria-label="{html.escape(title)}">'
        )
        self.parts.append(
            """
<defs>
  <filter id="shadow" x="-20%" y="-20%" width="150%" height="160%">
    <feDropShadow dx="0" dy="3" stdDeviation="4" flood-color="#17212B" flood-opacity="0.12"/>
  </filter>
  <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="#17212B"/>
  </marker>
  <marker id="arrow-red" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="#C6463D"/>
  </marker>
  <style>
    text { font-family: Inter, "Noto Sans", "DejaVu Sans", Arial, sans-serif; fill: #17212B; }
    .title { font-size: 30px; font-weight: 700; }
    .subtitle { font-size: 15px; fill: #5B6773; }
    .section { font-size: 15px; font-weight: 700; letter-spacing: 0; }
    .block-title { font-size: 16px; font-weight: 700; }
    .block-sub { font-size: 13px; fill: #263440; }
    .small { font-size: 12px; fill: #5B6773; }
    .tiny { font-size: 11px; fill: #5B6773; }
    .math { font-family: "DejaVu Sans Mono", monospace; font-size: 13px; }
  </style>
</defs>
"""
        )
        self.rect(0, 0, width, height, fill=COLORS["paper"], stroke="none")

    def add(self, value: str) -> None:
        self.parts.append(value)

    def rect(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        fill: str = "none",
        stroke: str = COLORS["ink"],
        stroke_width: float = 1.5,
        radius: float = 6,
        dash: str | None = None,
        opacity: float | None = None,
        css_class: str | None = None,
    ) -> None:
        attrs = [
            f'x="{x}"', f'y="{y}"', f'width="{width}"', f'height="{height}"',
            f'rx="{radius}"', f'fill="{fill}"', f'stroke="{stroke}"',
            f'stroke-width="{stroke_width}"',
        ]
        if dash:
            attrs.append(f'stroke-dasharray="{dash}"')
        if opacity is not None:
            attrs.append(f'opacity="{opacity}"')
        if css_class:
            attrs.append(f'class="{css_class}"')
        self.add(f'<rect {" ".join(attrs)}/>')

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        color: str = COLORS["ink"],
        width: float = 2,
        dash: str | None = None,
        arrow: bool = False,
        red_arrow: bool = False,
    ) -> None:
        attrs = [
            f'x1="{x1}"', f'y1="{y1}"', f'x2="{x2}"', f'y2="{y2}"',
            f'stroke="{color}"', f'stroke-width="{width}"', 'fill="none"',
        ]
        if dash:
            attrs.append(f'stroke-dasharray="{dash}"')
        if arrow:
            attrs.append(f'marker-end="url(#{"arrow-red" if red_arrow else "arrow"})"')
        self.add(f'<line {" ".join(attrs)}/>')

    def path(
        self,
        points: Sequence[tuple[float, float]],
        *,
        color: str = COLORS["ink"],
        width: float = 2,
        dash: str | None = None,
        arrow: bool = True,
        red_arrow: bool = False,
    ) -> None:
        coords = " ".join(f"{x},{y}" for x, y in points)
        attrs = [
            f'points="{coords}"', f'stroke="{color}"',
            f'stroke-width="{width}"', 'stroke-linejoin="round"',
            'stroke-linecap="round"', 'fill="none"',
        ]
        if dash:
            attrs.append(f'stroke-dasharray="{dash}"')
        if arrow:
            attrs.append(f'marker-end="url(#{"arrow-red" if red_arrow else "arrow"})"')
        self.add(f'<polyline {" ".join(attrs)}/>')

    def polygon(self, points: Sequence[tuple[float, float]], *, fill: str, stroke: str) -> None:
        coords = " ".join(f"{x},{y}" for x, y in points)
        self.add(
            f'<polygon points="{coords}" fill="{fill}" stroke="{stroke}" '
            'stroke-width="1.5" stroke-linejoin="round"/>'
        )

    def circle(self, x: float, y: float, radius: float, *, fill: str, stroke: str = COLORS["ink"]) -> None:
        self.add(
            f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="1.5"/>'
        )

    def text(
        self,
        x: float,
        y: float,
        value: str,
        *,
        css_class: str = "block-sub",
        anchor: str = "start",
        fill: str | None = None,
        weight: int | None = None,
    ) -> None:
        attrs = [f'x="{x}"', f'y="{y}"', f'class="{css_class}"', f'text-anchor="{anchor}"']
        if fill:
            attrs.append(f'style="fill:{fill}"')
        if weight:
            attrs.append(f'font-weight="{weight}"')
        self.add(f'<text {" ".join(attrs)}>{html.escape(value)}</text>')

    def multiline(
        self,
        x: float,
        y: float,
        lines: Iterable[str],
        *,
        css_class: str = "block-sub",
        anchor: str = "middle",
        line_height: int = 19,
    ) -> None:
        escaped = [html.escape(line) for line in lines]
        self.add(f'<text x="{x}" y="{y}" class="{css_class}" text-anchor="{anchor}">')
        for index, line in enumerate(escaped):
            dy = 0 if index == 0 else line_height
            self.add(f'<tspan x="{x}" dy="{dy}">{line}</tspan>')
        self.add('</text>')

    def finish(self) -> str:
        return "\n".join([*self.parts, "</svg>", ""])


def header(svg: SVG, title: str, subtitle: str, badge: str) -> None:
    svg.text(60, 58, title, css_class="title")
    svg.text(60, 86, subtitle, css_class="subtitle")
    badge_width = max(116, len(badge) * 7.5 + 28)
    svg.rect(svg.width - badge_width - 60, 34, badge_width, 34, fill=COLORS["ink"], stroke="none", radius=6)
    svg.text(svg.width - badge_width / 2 - 60, 57, badge, css_class="block-sub", anchor="middle", fill="#FFFFFF", weight=700)
    svg.line(60, 108, svg.width - 60, 108, color=COLORS["faint"], width=1.5)


def lane(svg: SVG, x: float, y: float, width: float, height: float, title: str, color: str) -> None:
    svg.rect(x, y, width, height, fill=COLORS["panel"], stroke=COLORS["faint"], radius=6)
    svg.rect(x, y, 8, height, fill=color, stroke="none", radius=4)
    svg.text(x + 22, y + 27, title, css_class="section", fill=color)


def block(
    svg: SVG,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    lines: Sequence[str],
    *,
    fill: str,
    side: str,
    depth: float = 12,
) -> None:
    # A restrained PlotNeuralNet-style extrusion: dimensions remain readable,
    # while the block still reads as a learned layer stack.
    svg.add('<g>')
    svg.polygon(
        [(x, y), (x + depth, y - depth), (x + width + depth, y - depth), (x + width, y)],
        fill="#FFFFFF",
        stroke=side,
    )
    svg.polygon(
        [(x + width, y), (x + width + depth, y - depth),
         (x + width + depth, y + height - depth), (x + width, y + height)],
        fill=side,
        stroke=side,
    )
    svg.rect(x, y, width, height, fill=fill, stroke=side, radius=4)
    svg.add('</g>')
    svg.text(x + width / 2, y + 30, title, css_class="block-title", anchor="middle")
    svg.multiline(x + width / 2, y + 55, lines, css_class="block-sub", anchor="middle", line_height=18)


def flat_block(
    svg: SVG,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    lines: Sequence[str],
    *,
    fill: str,
    stroke: str,
    align: str = "middle",
    title_color: str | None = None,
) -> None:
    """A compact 2-D block for route-heavy system diagrams."""
    svg.rect(x, y, width, height, fill=fill, stroke=stroke, stroke_width=1.6, radius=6)
    svg.rect(x, y, width, 6, fill=stroke, stroke="none", radius=3)
    text_x = x + width / 2 if align == "middle" else x + 16
    anchor = "middle" if align == "middle" else "start"
    svg.text(
        text_x,
        y + 30,
        title,
        css_class="block-title",
        anchor=anchor,
        fill=title_color or COLORS["ink"],
    )
    svg.multiline(
        text_x,
        y + 55,
        lines,
        css_class="block-sub",
        anchor=anchor,
        line_height=18,
    )


def pill(svg: SVG, x: float, y: float, width: float, title: str, subtitle: str, *, fill: str) -> None:
    svg.rect(x, y, width, 58, fill=fill, stroke=COLORS["line"], radius=8)
    svg.text(x + width / 2, y + 24, title, css_class="block-title", anchor="middle")
    svg.text(x + width / 2, y + 44, subtitle, css_class="small", anchor="middle")


def concat(svg: SVG, x: float, y: float, label: str) -> None:
    svg.circle(x, y, 18, fill="#FFFFFF", stroke=COLORS["ink"])
    svg.text(x, y + 5, "||", css_class="math", anchor="middle", weight=700)
    svg.text(x, y + 35, label, css_class="tiny", anchor="middle")


def shared_link(
    svg: SVG,
    x: float,
    y1: float,
    y2: float,
    label: str,
    *,
    label_y: float | None = None,
) -> None:
    svg.line(x, y1, x, y2, color=COLORS["blue"], width=2, dash="5 5")
    center_y = (y1 + y2) / 2 if label_y is None else label_y
    svg.rect(x - 43, center_y - 13, 86, 26, fill="#FFFFFF", stroke=COLORS["blue"], radius=5)
    svg.text(x, center_y + 4, label, css_class="tiny", anchor="middle", fill=COLORS["blue"], weight=700)


def legend(svg: SVG, y: float, *, include_red: bool = False) -> None:
    svg.line(70, y, 108, y, width=2, arrow=True)
    svg.text(120, y + 5, "forward / differentiable path", css_class="small")
    svg.line(370, y, 408, y, width=2, dash="5 5")
    svg.text(420, y + 5, "shared parameters", css_class="small")
    if include_red:
        svg.line(625, y, 663, y, color=COLORS["red"], width=2.5, arrow=True, red_arrow=True)
        svg.text(675, y + 5, "inner-loop gradient ascent", css_class="small", fill=COLORS["red"])


def generate_base() -> str:
    svg = SVG(2000, 1080, "1Q Base architecture on Maze2D")
    header(
        svg,
        "1Q Base",
        "Maze2D-umaze-v1  |  raw-input actor  |  one quasimetric critic",
        "MAZE2D / 1 CRITIC",
    )
    lane(svg, 45, 132, 1910, 260, "POLICY FORWARD", COLORS["green"])
    lane(svg, 45, 418, 1910, 474, "ONE QUASIMETRIC CRITIC", COLORS["orange"])

    # Policy connections are drawn before blocks so they sit behind them.
    svg.path([(200, 205), (300, 205), (300, 235), (348, 235)])
    svg.path([(200, 325), (300, 325), (300, 235), (348, 235)])
    svg.line(384, 235, 452, 235, arrow=True)
    svg.line(750, 235, 812, 235, arrow=True)
    svg.line(992, 235, 1052, 235, arrow=True)
    svg.path([(1142, 264), (1142, 477), (736, 477), (736, 527)], arrow=True)

    pill(svg, 70, 176, 130, "state s_t", "R^4: x,y,vx,vy", fill=COLORS["input"])
    pill(svg, 70, 296, 130, "goal g", "R^4", fill=COLORS["input"])
    concat(svg, 366, 235, "raw concat: R^8")
    block(
        svg, 452, 176, 298, 118, "Actor pi",
        ["8 -> 1024 x 4 -> 4", "ReLU; zero-init output"],
        fill=COLORS["actor"], side=COLORS["actor_side"],
    )
    block(
        svg, 812, 186, 180, 98, "Tanh Normal",
        ["mean + raw std", "bounded action"],
        fill="#D8EBD9", side=COLORS["actor_side"], depth=9,
    )
    pill(svg, 1052, 206, 180, "action a_t", "R^2", fill="#EDF6EE")
    svg.text(1255, 226, "Actor consumes raw (s_t, g), not critic latents.", css_class="small")

    # Critic / actor-loss flow.
    svg.path([(200, 205), (242, 205), (242, 545), (350, 545)])
    svg.path([(200, 325), (258, 325), (258, 650), (350, 650)])
    svg.line(650, 545, 718, 545, arrow=True)
    svg.line(650, 650, 718, 650, arrow=True)
    svg.line(754, 545, 812, 545, arrow=True)
    svg.path([(754, 650), (1195, 650), (1195, 686), (1280, 686)], arrow=True)
    svg.path([(1082, 545), (1168, 545), (1168, 580), (1280, 580)], arrow=True)
    svg.line(1530, 633, 1594, 633, arrow=True)
    svg.line(1764, 633, 1820, 633, arrow=True)

    block(
        svg, 350, 487, 300, 210, "Shared encoder E_theta",
        ["4 -> 1024 -> 1024", "-> 1024 -> 256", "same weights for s_t and g"],
        fill=COLORS["encoder"], side=COLORS["encoder_side"],
    )
    svg.text(366, 550, "s_t", css_class="small", anchor="middle")
    svg.text(366, 655, "g", css_class="small", anchor="middle")
    concat(svg, 736, 545, "[z_s, a_t]: R^258")
    svg.text(736, 676, "z_g: R^256", css_class="small", anchor="middle")
    block(
        svg, 812, 512, 270, 126, "Residual dynamics F_phi",
        ["[z_s, a_t]: 258", "258 -> 1024 x 3 -> 256", "z_hat = z_s + Delta z"],
        fill=COLORS["dynamics"], side=COLORS["dynamics_side"],
    )
    svg.text(1148, 535, "z_hat_(t+1)", css_class="math", anchor="middle")
    block(
        svg, 1280, 548, 250, 176, "Shared projector P_psi",
        ["256 -> 1024 -> 1024", "-> 2048", "applied to z_hat and z_g"],
        fill=COLORS["metric"], side=COLORS["metric_side"],
    )
    block(
        svg, 1594, 582, 170, 102, "IQE head",
        ["dim 2048", "64 components"],
        fill="#E7DCF1", side=COLORS["metric_side"], depth=9,
    )
    pill(svg, 1820, 604, 100, "d", "scalar", fill="#F0EBF6")

    svg.rect(300, 752, 1380, 94, fill="#FFFFFF", stroke=COLORS["faint"], radius=6)
    svg.text(330, 780, "Actor objective", css_class="section", fill=COLORS["green"])
    svg.text(480, 780, "min  d(F(E(s_t), pi(s_t,g)), E(g))", css_class="math")
    svg.text(330, 814, "Critic training", css_class="section", fill=COLORS["orange"])
    svg.text(480, 814, "global push + local transition constraint + latent-dynamics loss", css_class="block-sub")

    legend(svg, 935)
    svg.text(
        70, 990,
        "Configuration basis: Maze2D Base network settings from the existing 2Q run, with num_critics changed to 1.",
        css_class="small",
    )
    svg.text(
        70, 1017,
        "BC weight = 0; adaptive entropy regularization = off; goal-as-future-state = off; joint training.",
        css_class="small",
    )
    return svg.finish()


def generate_split_max8() -> str:
    svg = SVG(2400, 1680, "Detailed 1Q GO-QRL-Max8 architecture on Maze2D")
    header(
        svg,
        "1Q GO-QRL-Max8",
        "Maze2D-umaze-v1  |  one critic  |  full critic and actor training routes",
        "LATEX SOURCE + PREVIEW",
    )
    lane(svg, 40, 128, 2320, 700, "A. CRITIC TRAINING", COLORS["blue"])
    lane(svg, 40, 850, 2320, 350, "B. ACTOR FORWARD", COLORS["green"])
    lane(svg, 40, 1222, 2320, 395, "C. MAX8 LATENT GOAL COMPLETION", COLORS["red"])

    # Critic inputs and shared split encoder.
    pill(svg, 70, 210, 210, "current state  sₜ", "(xₜ,yₜ,vˣₜ,vʸₜ) in R⁴", fill=COLORS["input"])
    pill(svg, 70, 392, 210, "dataset action  aₜ", "R²", fill=COLORS["input"])
    pill(svg, 70, 574, 210, "next state  sₜ₊₁", "complete state in R⁴", fill=COLORS["input"])
    flat_block(
        svg, 330, 215, 340, 410, "Shared Split Encoder  Eθ",
        [
            "Position branch  Eᴳ",
            "2 → 384 → 384 → 64",
            "RMSNorm (no affine)",
            "",
            "Motion branch  Eᴺ",
            "2 → 384 → 384 → 64",
            "RMSNorm (no affine)",
            "",
            "Concatenate: 64 + 64 = 128",
        ],
        fill="#DCEFF6", stroke=COLORS["blue"],
    )
    pill(svg, 720, 250, 240, "current-state latent  zₜ", "[zₜᴳ ; zₜᴺ] in R¹²⁸", fill="#EAF5F8")
    pill(svg, 720, 540, 240, "next-state latent  zₜ₊₁", "[zₜ₊₁ᴳ ; zₜ₊₁ᴺ] in R¹²⁸", fill="#EAF5F8")
    svg.path([(280, 239), (305, 239), (305, 290), (330, 290)], arrow=True)
    svg.path([(280, 603), (305, 603), (305, 550), (330, 550)], arrow=True)
    svg.path([(670, 290), (690, 290), (690, 279), (720, 279)], arrow=True)
    svg.path([(670, 550), (690, 550), (690, 569), (720, 569)], arrow=True)

    # Local constraint route.
    flat_block(
        svg, 1030, 175, 255, 100, "Adjacent pair",
        ["(zₜ, zₜ₊₁)"], fill="#FFFFFF", stroke=COLORS["line"],
    )
    flat_block(
        svg, 1335, 165, 280, 120, "Shared  Pψ + IQE",
        ["128 → 512 → 2048", "IQE: 64 components", "distance  dψ(zₜ, zₜ₊₁)"],
        fill="#EEE6F5", stroke=COLORS["purple"],
    )
    flat_block(
        svg, 1665, 155, 365, 140, "Local constraint  Llocal",
        ["Adjacent distance ≤ step cost 1", "λ ( mean ReLU(d − 1)² − 0.25² )"],
        fill="#FFF4DB", stroke=COLORS["orange"], align="start",
    )

    # Global push route.
    flat_block(
        svg, 1030, 340, 255, 100, "Batch-random pair",
        ["(zₜ, roll(zₜ₊₁, 1))"], fill="#FFFFFF", stroke=COLORS["line"],
    )
    flat_block(
        svg, 1335, 330, 280, 120, "Shared  Pψ + IQE",
        ["same weights; IQE 64 components", "distance to random target"],
        fill="#EEE6F5", stroke=COLORS["purple"],
    )
    flat_block(
        svg, 1665, 320, 365, 140, "Global push  Lpush",
        ["Push unrelated pairs apart", "mean softplus(15 − d; β = 0.1)"],
        fill="#FFF4DB", stroke=COLORS["orange"], align="start",
    )

    # Latent-dynamics route.
    flat_block(
        svg, 1020, 540, 285, 125, "Residual dynamics  Fφ",
        ["[zₜ, aₜ]: 130 → 512 → 512 → 128", "prediction  ẑₜ₊₁ = zₜ + Δz"],
        fill="#FBE7C9", stroke=COLORS["orange"],
    )
    flat_block(
        svg, 1350, 550, 220, 105, "Prediction pair",
        ["(ẑₜ₊₁, zₜ₊₁)"], fill="#FFFFFF", stroke=COLORS["line"],
    )
    flat_block(
        svg, 1615, 540, 280, 125, "Shared  Pψ + IQE",
        ["IQE: 64 components", "bidirectional distances", "d(ẑ,z′) and d(z′,ẑ)"],
        fill="#EEE6F5", stroke=COLORS["purple"],
    )
    flat_block(
        svg, 1940, 530, 360, 145, "Dynamics loss  Ldyn",
        ["0.1 × mean of both squared", "directional IQE distances"],
        fill="#FFF4DB", stroke=COLORS["orange"], align="start",
    )

    # Joint update summary.
    flat_block(
        svg, 2075, 175, 225, 280, "Joint critic update",
        ["Lcritic =", "Lpush + Llocal + Ldyn", "", "Updates encoder,", "projector, IQE,", "and dynamics together"],
        fill="#F8EAD4", stroke=COLORS["orange"],
    )

    # Critic routing. Two separate trunks keep the pair construction explicit.
    svg.path([(960, 279), (990, 279), (990, 225), (1030, 225)], arrow=True)
    svg.path([(960, 569), (1000, 569), (1000, 245), (1030, 245)], arrow=True)
    svg.path([(960, 279), (980, 279), (980, 380), (1030, 380)], arrow=True)
    svg.path([(960, 569), (995, 569), (995, 420), (1030, 420)], arrow=True)
    svg.path([(960, 279), (985, 279), (985, 590), (1020, 590)], arrow=True)
    svg.path([(280, 421), (300, 421), (300, 710), (985, 710), (985, 630), (1020, 630)], arrow=True)
    svg.line(1285, 225, 1335, 225, arrow=True)
    svg.line(1615, 225, 1665, 225, arrow=True)
    svg.line(1285, 390, 1335, 390, arrow=True)
    svg.line(1615, 390, 1665, 390, arrow=True)
    svg.line(1305, 602, 1350, 602, arrow=True)
    svg.path([(960, 569), (1325, 569), (1325, 625), (1350, 625)], arrow=True)
    svg.line(1570, 602, 1615, 602, arrow=True)
    svg.line(1895, 602, 1940, 602, arrow=True)
    svg.line(2030, 225, 2075, 250, arrow=True)
    svg.line(2030, 390, 2075, 350, arrow=True)
    svg.path([(2300, 602), (2330, 602), (2330, 430), (2300, 430)], arrow=True)
    svg.line(1475, 285, 1475, 330, color=COLORS["purple"], width=2, dash="5 5")
    svg.path([(1475, 450), (1475, 490), (1755, 490), (1755, 540)], color=COLORS["purple"], width=2, dash="5 5", arrow=False)
    svg.text(1490, 310, "same Pψ + IQE weights", css_class="tiny", fill=COLORS["purple"])

    # Actor forward route.
    pill(svg, 70, 910, 200, "sampled state  s", "complete Maze2D state", fill=COLORS["input"])
    pill(svg, 70, 1055, 200, "random goal  g", "roll(next states, 1)", fill=COLORS["input"])
    flat_block(
        svg, 330, 885, 290, 105, "Shared Eᴳ and Eᴺ",
        ["encode all four coordinates"], fill="#DCEFF6", stroke=COLORS["blue"],
    )
    flat_block(
        svg, 330, 1025, 290, 115, "Shared Eᴳ only",
        ["encode goal (xᵍ,yᵍ)", "goal branch output in R⁶⁴"], fill="#DCEFF6", stroke=COLORS["blue"],
    )
    pill(svg, 680, 910, 225, "state latent  zₛ", "[zₛᴳ ; zₛᴺ] in R¹²⁸", fill="#EAF5F8")
    pill(svg, 680, 1055, 225, "actor goal  zᵍᴳ", "goal branch in R⁶⁴", fill="#EAF5F8")
    flat_block(
        svg, 975, 955, 230, 100, "Concatenate",
        ["[zₛ ; zᵍᴳ] in R¹⁹²"], fill="#FFFFFF", stroke=COLORS["line"],
    )
    flat_block(
        svg, 1260, 945, 265, 120, "Latent actor  πω",
        ["192 → 512 → 512 → 4"], fill="#DCEEDC", stroke=COLORS["green"],
    )
    flat_block(
        svg, 1580, 955, 190, 100, "Tanh Normal",
        ["sample action  a in R²"], fill="#DCEEDC", stroke=COLORS["green"],
    )
    flat_block(
        svg, 1830, 945, 280, 120, "Shared dynamics  Fφ",
        ["[zₛ, a] → predicted next latent"], fill="#FBE7C9", stroke=COLORS["orange"],
    )
    pill(svg, 2170, 976, 160, "prediction  ẑₛ′", "R¹²⁸", fill="#FBE7C9")
    svg.line(270, 939, 330, 939, arrow=True)
    svg.line(270, 1084, 330, 1084, arrow=True)
    svg.line(620, 939, 680, 939, arrow=True)
    svg.line(620, 1084, 680, 1084, arrow=True)
    svg.path([(905, 939), (940, 939), (940, 985), (975, 985)], arrow=True)
    svg.path([(905, 1084), (940, 1084), (940, 1025), (975, 1025)], arrow=True)
    svg.line(1205, 1005, 1260, 1005, arrow=True)
    svg.line(1525, 1005, 1580, 1005, arrow=True)
    svg.line(1770, 1005, 1830, 1005, arrow=True)
    svg.path([(905, 939), (925, 939), (925, 1160), (1800, 1160), (1800, 1040), (1830, 1040)], arrow=True)
    svg.line(2110, 1005, 2170, 1005, arrow=True)
    svg.text(2115, 1090, "Critic frozen for actor update.", css_class="small")
    svg.text(2115, 1115, "Gradients reach Fφ, action, and πω.", css_class="small")

    # Max8 inner loop and outer actor loss.
    pill(svg, 70, 1295, 240, "fixed target position  zᵍᴳ", "RMSNorm(Eᴳ(xᵍ,yᵍ))", fill="#EAF5F8")
    pill(svg, 70, 1455, 240, "sampled motion  h₀", "RMSNorm(Eᴺ(vₓᵍ,vᵧᵍ))", fill="#FCE8E6")
    flat_block(
        svg, 380, 1340, 290, 120, "Candidate completed goal",
        ["h̄ₖ = RMSNorm(hₖ)", "z̃ᵍ⁽ᵏ⁾ = [zᵍᴳ ; h̄ₖ] in R¹²⁸"], fill="#FCE8E6", stroke=COLORS["red"],
    )
    flat_block(
        svg, 735, 1328, 335, 145, "Shared Pψ + IQE",
        ["dₖ = dψ(stopgrad(ẑₛ′), z̃ᵍ⁽ᵏ⁾)", "prediction is fixed inside loop"],
        fill="#EEE6F5", stroke=COLORS["purple"],
    )
    flat_block(
        svg, 1140, 1328, 330, 145, "Adam gradient ascent",
        ["gradient passes through RMSNorm", "hₖ₊₁ = AdamAsc(hₖ, ∇ₕdₖ)", "k = 0,…,7;  lr = 0.01"],
        fill="#F9D8D5", stroke=COLORS["red"],
    )
    pill(svg, 1535, 1371, 235, "best goal  zᵍ⋆", "best of h̄₀,…,h̄₈", fill="#FCE8E6")
    flat_block(
        svg, 1830, 1328, 315, 145, "Final shared metric",
        ["d final = dψ(ẑₛ′, stopgrad(zᵍ⋆))", "gradient restored through prediction"],
        fill="#EEE6F5", stroke=COLORS["purple"],
    )
    flat_block(
        svg, 2200, 1340, 130, 120, "Actor loss",
        ["min  mean", "of d final"], fill="#DCEEDC", stroke=COLORS["green"],
    )
    svg.path([(310, 1324), (340, 1324), (340, 1375), (380, 1375)], arrow=True)
    svg.path([(310, 1484), (350, 1484), (350, 1430), (380, 1430)], arrow=True)
    svg.line(670, 1400, 735, 1400, arrow=True)
    svg.line(1070, 1400, 1140, 1400, color=COLORS["red"], width=2.7, arrow=True, red_arrow=True)
    svg.path([(1305, 1473), (1305, 1545), (190, 1545), (190, 1513)], color=COLORS["red"], width=2.7, arrow=True, red_arrow=True)
    svg.text(750, 1570, "normalize every candidate; retain the per-sample best of steps 0 through 8", css_class="small", anchor="middle", fill=COLORS["red"])
    svg.line(1470, 1400, 1535, 1400, color=COLORS["red"], width=2.7, arrow=True, red_arrow=True)
    svg.line(1770, 1400, 1830, 1400, arrow=True)
    svg.line(2145, 1400, 2200, 1400, arrow=True)
    svg.path([(2250, 1034), (2250, 1260), (900, 1260), (900, 1328)], arrow=True)
    svg.path([(2250, 1034), (2280, 1034), (2280, 1295), (1990, 1295), (1990, 1328)], arrow=True)
    svg.text(1455, 1248, "The .tex file is the authoritative source; all mathematical labels use LaTeX math mode.", css_class="small", anchor="middle")
    return svg.finish()


def write_diagrams(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    diagrams = {
        "qrl_1q_base_maze2d.svg": generate_base(),
        "qrl_1q_split_latent_max8_maze2d.svg": generate_split_max8(),
    }
    for filename, content in diagrams.items():
        path = output_dir / filename
        path.write_text(content, encoding="utf-8")
        print(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT,
        help="Directory for generated SVG files (default: script directory).",
    )
    args = parser.parse_args()
    write_diagrams(args.output_dir.resolve())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Draw the shared NQRL network architecture and information flow."""

import json
from math import atan2, cos, sin
from pathlib import Path

from reportlab.lib.colors import HexColor, white
import nqrl_latex_figure as canvas
from nqrl_latex_figure import math_source

from analyze_nqrl_paper_results import PALETTE

from nqrl_figure_style import (
    BOLD, FONT, MATH, INK, MUTED, GRID, GOAL, GOAL_EDGE, CONTEXT, CONTEXT_EDGE,
    label, register_fonts,
)

ROOT = Path(__file__).resolve().parents[1] / "iclr2027"
NEUTRAL = white
POLICY = white
POLICY_EDGE = HexColor(PALETTE['NQRL'])
PAD = HexColor("#F7DEDB")
PAD_EDGE = HexColor("#BC6158")


class Diagram:
    def __init__(self, name, height, title, *, width=560):
        self.pdf = canvas.Canvas(str(ROOT / "figures" / name),
                                 pagesize=(width, height), invariant=1,
                                 initialFontName=FONT)
        self.pdf.setTitle(title)
        self.pdf.setAuthor("Anonymous")

    def text(self, x, y, text, size=10, **kwargs):
        if text == '+':
            self.pdf.latex_label(x,y,'$+$',size,align='center',math=True,color=INK)
            return
        label(self.pdf, x, y, text, size, **kwargs)

    def math(self, x, y, width, text, size=13):
        self.pdf.latex_label(x+width/2,y,math_source(text),size,
                             align='center',math=True,center_y=True,color=INK)

    def box(self, x, y, w, h, text, *, fill=NEUTRAL, edge=MUTED, size=12, math=True):
        p = self.pdf
        p.setFillColor(fill)
        p.setStrokeColor(edge)
        p.setLineWidth(1.4)
        p.rect(x, y, w, h, fill=1, stroke=1)
        if math:
            self.math(x+2, y+h/2+1, w-4, text, size)
        else:
            lines = text.split("\n")
            for i, line in enumerate(lines):
                self.text(x+w/2, y+h/2+(len(lines)-1)*6-i*12-3,
                          line, size, align="center")

    def split(self, x, y, w, left, right, *, padded=False, size=12):
        self.box(x, y, w/2, 28, left, fill=GOAL, edge=GOAL_EDGE, size=size)
        self.box(x+w/2, y, w/2, 28, right,
                 fill=PAD if padded else CONTEXT,
                 edge=PAD_EDGE if padded else CONTEXT_EDGE, size=size)

    def arrow(self, points, *, color=MUTED, dashed=False, width=2.0):
        assert 2 <= len(points) <= 4, 'Arrows may have at most two bends'
        p = self.pdf
        p.setStrokeColor(color)
        p.setFillColor(color)
        p.setLineWidth(width)
        p.setLineJoin(1)
        p.setDash(3, 2) if dashed else p.setDash()
        path = p.beginPath()
        path.moveTo(*points[0])
        for point in points[1:]:
            path.lineTo(*point)
        p.drawPath(path)
        p.setDash()
        x,y = points[-1]
        px,py = points[-2]
        angle = atan2(y-py, x-px)
        path = p.beginPath()
        path.moveTo(x,y)
        for offset in (2.65, -2.65):
            path.lineTo(x+6*cos(angle+offset), y+6*sin(angle+offset))
        path.close()
        p.drawPath(path, fill=1, stroke=0)

    def rule(self, y):
        self.pdf.setStrokeColor(GRID)
        self.pdf.setLineWidth(.6)
        self.pdf.line(12, y, 548, y)

    def save(self):
        self.pdf.save()


def encoder_comparison():
    """Reacher wrapper: fingertip XY, two hinge angles, two angular velocities.

    Both panels encode the same current state. The NQRL output rectangles
    denote latent blocks, not scalar coordinates or specified latent widths.
    """
    d = Diagram('state_encoder_comparison.pdf', 144,
                'QRL and NQRL state encoding: reacher_hard example', width=460)
    d.text(14, 130, '(a) QRL', 12, font=BOLD)
    d.text(234, 130, '(b) NQRL', 12, font=BOLD)
    d.text(448, 130, 'reacher_hard', 10.5, align='right', color=MUTED)
    coordinates = ('<i>x</i>', '<i>y</i>', '\u03b8<sub>1</sub>', '\u03b8<sub>2</sub>',
                   '\u03c9<sub>1</sub>', '\u03c9<sub>2</sub>')
    for x in (18, 246):
        d.math(x, 116, 32, '<i>s</i>', 13)
        for i, value in enumerate(coordinates):
            goal = i < 2
            d.box(x, 92-i*16, 32, 16, value, size=12,
                  fill=GOAL if goal else CONTEXT,
                  edge=GOAL_EDGE if goal else CONTEXT_EDGE)

    # QRL maps all six current-state coordinates through one state encoder.
    d.box(76, 12, 54, 96, 'State\nencoder', math=False, size=11)
    d.arrow([(50, 60), (76, 60)])
    d.arrow([(130, 60), (156, 60)])
    d.box(156, 12, 38, 96, '<i>z</i>', size=14)

    d.pdf.setStrokeColor(GRID)
    d.pdf.setLineWidth(.6)
    d.pdf.line(220, 12, 220, 108)

    # The top two and bottom four state coordinates feed independent encoders.
    for y, h, name, latent, fill, edge in (
        (76, 32, 'Goal\nencoder', '<i>z</i><sub>g</sub>(<i>s</i>)', GOAL, GOAL_EDGE),
        (12, 64, 'Non-goal\nencoder', '<i>z</i><sub>n</sub>(<i>s</i>)', CONTEXT, CONTEXT_EDGE),
    ):
        center = y+h/2
        d.arrow([(278, center), (304, center)], color=edge)
        d.box(304, y, 58, h, name, fill=fill, edge=edge, math=False, size=11)
        d.arrow([(362, center), (388, center)], color=edge)
        d.box(388, y, 56, h, latent, fill=fill, edge=edge, size=12)
    d.math(388, 116, 56, '<i>E</i><sub>\u03b8</sub>(<i>s</i>)', 12)
    d.save()


def residual_backbone():
    """Render the implementation detail omitted from the main pipeline."""
    d = Diagram("nqrl_residual_mlp.pdf", 96, "NQRL residual MLP backbone")
    p = d.pdf
    d.math(4,48,28,"<i>x</i>",14)
    d.box(43,26,81,44,"Linear\nLayerNorm\nSiLU",math=False,size=10.5)
    d.arrow([(32,48),(43,48)])
    for i in range(4):
        x=155+i*66
        d.box(x,26,58,44,"Linear\nLayerNorm\nSiLU",math=False,size=10.5)
        d.arrow([(124 if i==0 else x-8,48),(x,48)])
    d.arrow([(411,48),(430,48)])
    p.setFillColor(white)
    p.setStrokeColor(MUTED)
    p.circle(440,48,10,fill=1,stroke=1)
    d.text(440,44,"+",13,align="center")
    d.arrow([(139,48),(139,80),(440,80),(440,58)])
    d.text(283,86,"Residual block",10,color=MUTED,align="center")
    d.arrow([(450,48),(474,48)])
    d.box(474,34,72,28,"Linear",math=False,size=11)
    d.save()


def network_dimensions():
    audit = json.loads((ROOT / "data/network_architecture_audit.json").read_text())
    modules = (
        ("Goal encoder", "critics.0.encoder.goal_encoder"),
        ("Non-goal encoder", "critics.0.encoder.non_goal_encoder"),
        ("Policy", "actor.backbone"),
        ("Dynamics increment", "critics.0.latent_dynamics.backbone"),
        ("Quasimetric projector", "critics.0.quasimetric_model.projector"),
    )
    lines = [
        r"\begin{table}[!htb]",
        r"\caption{Resolved NQRL network dimensions on AntNavigate. Each MLP uses",
        r"one four-layer residual block, preceded by an input projection and",
        r"followed by an output layer (Figure~\ref{fig:network-detail}).",
        r"The policy's 16 outputs parameterize eight action means and scales.}",
        r"\label{tab:network-dimensions}",
        r"\centering\normalsize",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}cccc@{}}",
        r"\toprule",
        r"Module & Input & Hidden width & Output \\",
        r"\midrule",
    ]
    for name, key in modules:
        module = audit["modules"][key]
        assert module["type"] == "ResidualMLP"
        assert module["residual_blocks"] == 1 and module["block_depth"] == 4
        lines.append(f"{name} & {module['input_size']} & {module['width']} & "
                     f"{module['output_size']} " + r"\\")
    lines.extend((r"\bottomrule", r"\end{tabular*}", r"\end{table}"))
    (ROOT / "tables/network_dimensions.tex").write_text("\n".join(lines) + "\n")


def detailed():
    audit = json.loads((ROOT / "data/network_architecture_audit.json").read_text())
    assert (audit["goal_latent_size"],audit["non_goal_latent_size"]) == (57,455)
    for item in audit["modules"].values():
        if item["type"] != "ResidualMLP":
            continue
        assert item["residual_blocks"] == 1 and item["block_depth"] == 4
    d = Diagram("nqrl_network_22m.pdf", 366,
                "NQRL network architecture and information flow")
    p = d.pdf
    p.saveState()
    p.translate(0, -34)
    d.text(14,383,"(a) Policy computation",12,font=BOLD)
    for y, inp, name, fill, edge in (
        (357, '<i>q</i>(<i>s</i>)', 'Goal encoder', GOAL, GOAL_EDGE),
        (324, '<i>s</i><sub><i>I</i><sub>n</sub></sub>', 'Non-goal encoder', CONTEXT, CONTEXT_EDGE),
        (291, '<i>g</i>', 'Goal encoder', GOAL, GOAL_EDGE),
    ):
        d.math(8,y,49,inp,13)
        d.arrow([(58,y),(70,y)])
        d.box(70,y-13,103,26,name,fill=fill,edge=edge,math=False,size=11)
    # A dotted bracket denotes parameter sharing, not a data path.
    p.setStrokeColor(GOAL_EDGE)
    p.setLineWidth(1.4)
    p.setDash(1,2)
    p.lines([(173,357,186,357),(186,357,186,291),(186,291,173,291)])
    p.setDash()
    d.text(193,306,"shared",10,color=GOAL_EDGE)
    d.split(210,331,90,'<i>z</i><sub>g</sub>(<i>s</i>)','<i>z</i><sub>n</sub>(<i>s</i>)')
    d.math(210,373,90,'<i>E</i><sub>\u03b8</sub>(<i>s</i>)',13)
    d.arrow([(173,357),(201,357),(201,345),(210,345)])
    d.arrow([(173,324),(278,324),(278,331)])
    d.arrow([(300,345),(323,345)])
    d.box(323,329,78,32,'Policy \u03c0<sub>\u03c8</sub>',fill=POLICY,edge=POLICY_EDGE,size=13)
    d.arrow([(173,291),(362,291),(362,329)],color=GOAL_EDGE)
    d.math(247,304,96,'<i>e</i><sub>g,\u03b8</sub>(<i>g</i>)',13)
    d.arrow([(401,345),(429,345)],color=POLICY_EDGE)
    d.math(393,368,44,'\u03bc, \u03c1',13)
    d.box(429,329,108,32,'Gaussian + tanh',math=False,size=11,fill=POLICY,edge=POLICY_EDGE)
    d.arrow([(537,345),(549,345),(549,314)],color=POLICY_EDGE)
    d.math(537,303,22,'<i>a</i>',14)
    d.math(385,305,148,'\u03c3 = softplus(\u03c1) + \u03b5<sub>\u03c3</sub>',12)
    d.rule(273)
    p.restoreState()

    d.text(14,227,"(b) Completion search (training only)",12,font=BOLD)
    d.math(8,194,64,'<i>z</i>, <i>a</i>',14)
    d.arrow([(72,194),(92,194)])
    d.box(92,178,112,32,'Dynamics increment\n\u0394z',math=False,size=11)
    d.arrow([(204,194),(224,194)])
    p.setFillColor(white)
    p.setStrokeColor(MUTED)
    p.circle(234,194,10,fill=1,stroke=1)
    d.text(234,190,"+",13,align="center")
    d.arrow([(78,194),(78,218),(234,218),(234,204)])
    d.math(45,214,28,'<i>z</i>',12)
    d.arrow([(244,194),(314,194)])
    d.math(247,208,63,'<i>z&#770;</i>&nbsp;<sup>+</sup>',14)
    d.text(252,178,"next state",10.5,color=MUTED)
    d.split(92,132,142,'<i>e</i><sub>g,\u03b8</sub>(<i>g</i>)','<i>h</i>')
    d.text(128,168,"fixed",10.5,color=GOAL_EDGE,align="center")
    d.text(199,168,"optimized",10.5,color=CONTEXT_EDGE,align="center")
    d.arrow([(234,146),(314,146)])
    d.text(251,158,"target",10.5,color=MUTED)
    for y in (194,146):
        d.box(314,y-14,81,28,'<i>P</i><sub>\u03b8</sub>',size=14)
    p.setStrokeColor(MUTED)
    p.setLineWidth(1.4)
    p.setDash(1,2)
    p.line(354,160,354,180)
    p.setDash()
    d.text(362,168,"shared",10.5,color=MUTED)
    d.arrow([(395,194),(469,194),(469,186)])
    d.arrow([(395,146),(469,146),(469,152)])
    d.box(435,152,68,34,'IQE <i>D</i><sub>\u03b8</sub>',size=12)
    d.arrow([(490,152),(490,118),(199,118),(199,132)],
            color=CONTEXT_EDGE,dashed=True)
    d.math(421,129,82,'\u2207<sub>h</sub>',13)
    d.text(257,121,"completion search",10.5,color=CONTEXT_EDGE)
    d.rule(110)

    d.text(14,96,"(c) Residual MLP",12,font=BOLD)
    d.math(4,48,28,"<i>x</i>",14)
    d.box(43,26,81,44,"Linear\nLayerNorm\nSiLU",math=False,size=10.5)
    d.arrow([(32,48),(43,48)])
    for i in range(4):
        x=155+i*66
        d.box(x,26,58,44,"Linear\nLayerNorm\nSiLU",math=False,size=10.5)
        d.arrow([(124 if i==0 else x-8,48),(x,48)])
    d.arrow([(411,48),(430,48)])
    p.setFillColor(white)
    p.setStrokeColor(MUTED)
    p.circle(440,48,10,fill=1,stroke=1)
    d.text(440,44,"+",13,align="center")
    d.arrow([(139,48),(139,80),(440,80),(440,58)])
    d.text(283,86,"Residual block",10,color=MUTED,align="center")
    d.arrow([(450,48),(474,48)])
    d.box(474,34,72,28,"Linear",math=False,size=11)
    d.text(14,8,"Solid: forward flow",10,color=MUTED)
    d.text(187,8,"Dashed: target gradient",10,color=CONTEXT_EDGE)
    d.text(391,8,"Dotted: shared weights",10,color=MUTED)
    d.save()


if __name__ == "__main__":
    register_fonts()
    residual_backbone()
    network_dimensions()
    encoder_comparison()
    detailed()
    print("Rendered the state-encoder comparison and shared NQRL network architecture.")

"""Shared typography and colors for the paper's vector figures."""

from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics


FONT = "Times-Roman"
BOLD = "Times-Bold"
MATH = "Times-Italic"
SERIF = FONT
SERIF_BOLD = BOLD
INK = HexColor("#283039")
MUTED = HexColor("#65707A")
GRID = HexColor("#E4E7EA")
GOAL = HexColor("#D9EEEA")
GOAL_EDGE = HexColor("#338779")
CONTEXT = HexColor("#FAEDCA")
CONTEXT_EDGE = HexColor("#B18B3C")

METHOD_DASHES = {
    "QRL": (4, 2),
    "TD-InfoNCE": (1.2, 1.6),
    "GCSL": (5, 2, 1, 2),
    "C-Learning": (7, 2),
    "CRL": (3, 1.5, 1, 1.5),
    "NQRL": (),
}


def task_title_lines(name):
    breaks = {
        "PandaPickAndPlaceJoints-v3": ("PandaPickAndPlace", "Joints-v3"),
        "PandaPickAndPlace-v3": ("PandaPickAndPlace", "-v3"),
        "PandaPushJoints-v3": ("PandaPushJoints-v3",),
        "PointMaze Medium-v3": ("PointMaze", "Medium-v3"),
        "AntMaze UMaze-v5": ("AntMaze UMaze-v5",),
    }
    return breaks.get(name, (name,))


def task_title(pdf, x, top, width, name, size=11.5):
    lines = task_title_lines(name)
    for i, text in enumerate(lines):
        assert pdfmetrics.stringWidth(text, BOLD, size) <= width, name
        label(pdf, x + width / 2, top - (5 if len(lines) == 2 else 10) - i * 11,
              text, size, font=BOLD, align="center")


def register_fonts():
    """Use Times AFM metrics for spacing; final glyphs are rendered by LaTeX."""
    for name in (FONT, BOLD, MATH):
        pdfmetrics.getFont(name)


def label(pdf, x, y, text, size=9, *, font=FONT, color=INK, align="left"):
    if hasattr(pdf, 'latex_label'):
        pdf.latex_label(x,y,text,size,font=font,color=color,align=align)
        return
    pdf.setFillColor(color)
    pdf.setFont(font, size)
    draw = {"left": pdf.drawString, "center": pdf.drawCentredString,
            "right": pdf.drawRightString}[align]
    draw(x, y, text)

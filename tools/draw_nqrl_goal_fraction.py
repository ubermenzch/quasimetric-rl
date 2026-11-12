"""Render the approved Figure 4(a,b) curves from archived statistics."""
import csv
from pathlib import Path
from reportlab.lib import colors
import nqrl_latex_figure as canvas
from nqrl_figure_style import label, FONT, BOLD, register_fonts

PAPER = Path(__file__).resolve().parents[1] / 'iclr2027'
W, H = 560, 184
INK = colors.HexColor('#26313B')
TICK = colors.HexColor('#58636D')
GRID = colors.HexColor('#E7EBEF')
AXIS = colors.HexColor('#A3ADB6')
SERIES = ('#636D76', '#008CB8')
FILES = ['goal_fraction_success_curves.csv', 'goal_fraction_success_tasks.csv',
         'goal_fraction_gain_curves.csv', 'goal_fraction_gain_tasks.csv']


def read(name):
    with (PAPER / 'data' / name).open() as handle:
        return list(csv.DictReader(handle))


def stroke(pdf, points, color, width=1):
    pdf.setStrokeColor(colors.HexColor(color) if isinstance(color, str) else color)
    pdf.setLineWidth(width)
    pdf.setLineJoin(1)
    pdf.setLineCap(1)
    path = pdf.beginPath()
    for i, (x, y) in enumerate(points):
        (path.moveTo if i == 0 else path.lineTo)(x, y)
    pdf.drawPath(path)


def dot(pdf, x, y, color, *, hollow=False, radius=2.55):
    pdf.setStrokeColor(color if hollow else colors.white)
    pdf.setFillColor(colors.white if hollow else color)
    pdf.setLineWidth(1.15 if hollow else .65)
    pdf.circle(x, y, radius, stroke=1, fill=1)


def render_goal_fraction(output_path):
    """Render approved styling without changing source data or aggregation."""
    register_fonts()
    data = [read(name) for name in FILES]
    pdf = canvas.Canvas(str(output_path), pagesize=(W, H),
                        invariant=1, initialFontName=FONT)
    pdf.setTitle('QRL success and AQRL gain: original observations and sample SD')
    specs = [
        ('(a) QRL success', 48, 0, 1, 'Success', (0, .25, .5, .75, 1),
         [r for r in data[0] if r['method'] == 'QRL'],
         [r for r in data[1] if r['method'] == 'QRL'], 'success_mean', 'success_std'),
        ('(b) AQRL gain over QRL', 330, -20, 100, 'Gain (pp)', (-20, 0, 25, 50, 75, 100),
         data[2], data[3], 'gain_mean_pp', 'gain_std_pp'),
    ]
    audit = []
    for index, (title, left, low, high, ylabel, ticks, rows, tasks, mk, sk) in enumerate(specs):
        color = colors.HexColor(SERIES[index])
        bottom, width, height = 58, 218, 102
        def xy(x, y):
            return left + (x - .45) / .55 * width, bottom + (y - low) / (high - low) * height
        points = sorted(rows, key=lambda r: float(r['non_goal_fraction']))
        assert len(points) == 11 and len(tasks) == 14
        assert all(int(r['seeds']) == 5 for r in points)
        label(pdf, left + width / 2, 169, title, 14, font=BOLD, color=INK, align='center')
        for tick in ticks:
            x, y = xy(.45, tick)
            stroke(pdf, [(x, y), (left + width, y)], GRID, .45)
            label(pdf, left - 7, y - 3.8, f'{tick:g}', 12.5, color=TICK, align='right')
        for tick in (.5, .6, .7, .8, .9, 1):
            x, y = xy(tick, low)
            stroke(pdf, [(x, y), (x, y - 2.8)], AXIS, .6)
            label(pdf, x, bottom - 15, f'{tick*100:.0f}', 12.5, color=TICK, align='center')
        stroke(pdf, [(left, bottom + height), (left, bottom), (left + width, bottom)], AXIS, .65)
        pdf.saveState()
        pdf.translate(left - 34, bottom + height / 2)
        pdf.rotate(90)
        label(pdf, 0, 0, ylabel, 13.5, color=INK, align='center')
        pdf.restoreState()
        if index:
            pdf.setDash(3, 2.5)
            stroke(pdf, [xy(.45, 0), xy(1, 0)], '#929DA7', .8)
            pdf.setDash()
        # Clip only the uncertainty ribbon to the original axis limits.
        pdf.saveState()
        clip = pdf.beginPath()
        clip.rect(left, bottom, width, height)
        pdf.clipPath(clip, stroke=0)
        band = pdf.beginPath()
        for i, row in enumerate(points):
            (band.moveTo if i == 0 else band.lineTo)(
                *xy(float(row['non_goal_fraction']), float(row[mk]) + float(row[sk])))
        for row in reversed(points):
            band.lineTo(*xy(float(row['non_goal_fraction']), float(row[mk]) - float(row[sk])))
        band.close()
        pdf.setFillColor(color)
        pdf.setFillAlpha(.15)
        pdf.drawPath(band, stroke=0, fill=1)
        pdf.restoreState()
        stroke(pdf, [xy(float(r['non_goal_fraction']), float(r[mk])) for r in points], color, 1.25)
        for row in points:
            dot(pdf, *xy(float(row['non_goal_fraction']), float(row[mk])), color)
        shared = {r['non_goal_fraction'] for r in points if int(r['task_count']) > 1}
        for row in tasks:
            if row['non_goal_fraction'] in shared:
                dot(pdf, *xy(float(row['non_goal_fraction']), float(row[mk])), color,
                    hollow=True, radius=2.7)
        audit.append(dict(panel=index, fraction_points=len(points), tasks=len(tasks),
                          hollow_task_points=sum(r['non_goal_fraction'] in shared for r in tasks),
                          observations=[{k: r[k] for k in ('non_goal_fraction', mk, sk)} for r in points]))
        label(pdf, left + width / 2, 24, 'Free coordinates (%)', 13.5, color=INK, align='center')
    # Shared legend below both per-panel x-axis labels.
    legend_color = colors.HexColor('#647583')
    pdf.setFillColor(legend_color)
    pdf.setFillAlpha(.15)
    pdf.rect(100, 4, 22, 7, stroke=0, fill=1)
    pdf.setFillAlpha(1)
    stroke(pdf, [(100, 7.5), (122, 7.5)], legend_color, 1.25)
    dot(pdf, 111, 7.5, legend_color, radius=2.4)
    label(pdf, 128, 4, 'Fraction mean ± SD', 11.5, color=TICK)
    dot(pdf, 291, 7.5, legend_color, hollow=True)
    label(pdf, 301, 4, 'Task mean at shared fractions', 11.5, color=TICK)
    pdf.save()
    return audit

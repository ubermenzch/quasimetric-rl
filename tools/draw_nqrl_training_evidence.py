#!/usr/bin/env python3
"""Render per-task learning curves and dynamics trajectories from archived logs."""

import argparse
import csv
import hashlib
import json
import math
import statistics
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

from reportlab.lib import colors
import nqrl_latex_figure as canvas
from reportlab.pdfbase.pdfmetrics import stringWidth
from nqrl_table_labels import task_label

from nqrl_figure_style import (
    BOLD, FONT, SERIF, SERIF_BOLD, GRID, INK, MUTED, METHOD_DASHES,
    label, register_fonts, task_title,
)

from analyze_nqrl_paper_results import (
    DATA_ROOT, ENVIRONMENTS, FIGURE_ROOT, METHODS, PALETTE, PAPER_ROOT, SEEDS,
    artifact_dir, interpolate, trapz, validation_curve,
)


def read_csv(name):
    with (DATA_ROOT / name).open() as handle:
        return list(csv.DictReader(handle))


def write_csv(name, rows):
    with (DATA_ROOT / name).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def rounded(value):
    return str(Decimal(str(round(value, 12))).quantize(Decimal('.001'), rounding=ROUND_HALF_UP))


def axes(pdf, box, xmax, ymin, ymax, ylabel, ticks, *, log=False, font_size=8,
         xmin=0, font=FONT, xlabel='Environment steps (k)', xticks=None, xformat=None):
    left, bottom, width, height = box
    transform = math.log10 if log else lambda v: v
    def xy(x, y):
        return (left + (x - xmin) / (xmax - xmin) * width,
                bottom + (transform(y) - transform(ymin)) / (transform(ymax) - transform(ymin)) * height)
    for tick in ticks:
        x, y = xy(xmin, tick)
        pdf.setStrokeColor(GRID)
        pdf.setLineWidth(.4)
        pdf.line(x, y, left + width, y)
        if log:
            exponent = str(int(math.log10(tick)))
            exponent_width = stringWidth(exponent, font, font_size * .7)
            label(pdf, left - 5 - exponent_width, y - 3, '10', font_size,
                  align='right', font=font)
            label(pdf, left - 5 - exponent_width, y + 1.5, exponent, font_size * .7, font=font)
        else:
            label(pdf, left - 5, y - 3, f'{tick:g}', font_size, align='right', font=font)
    if xticks is None:
        xticks = [xmin + fraction * (xmax - xmin) for fraction in (0, .2, .4, .6, .8, 1)]
    for step in xticks:
        x, _ = xy(step, ymin)
        text = xformat(step) if xformat else f'{step / 1000:.0f}'
        label(pdf, x, bottom - 13, text, font_size, align='center', font=font)
    label(pdf, left + width / 2, bottom - 29, xlabel, font_size,
          align='center', font=font)
    pdf.saveState()
    pdf.translate(left - 34, bottom + height / 2)
    pdf.rotate(90)
    label(pdf, 0, 0, ylabel, font_size, align='center', font=font)
    pdf.restoreState()
    pdf.setStrokeColor(MUTED)
    pdf.setLineWidth(.6)
    pdf.line(left, bottom, left + width, bottom)
    pdf.line(left, bottom, left, bottom + height)
    return xy


def line(pdf, points, color, width=1.3):
    pdf.setStrokeColor(colors.HexColor(color))
    pdf.setLineWidth(width)
    pdf.setLineJoin(1)
    path = pdf.beginPath()
    for i, (x, y) in enumerate(points):
        (path.moveTo if i == 0 else path.lineTo)(x, y)
    pdf.drawPath(path)


def task_grid(*, representative=False):
    rows = read_csv('per_task_learning_curves.csv')
    selected_keys = ('antnavigate', 'fetchpush', 'maze2d', 'pandapickj')
    environments = ([next(e for e in ENVIRONMENTS if e.key == key)
                     for key in selected_keys] if representative else ENVIRONMENTS)
    page_height = 175 if representative else 540
    display = lambda method: 'AQRL' if method == 'NQRL' else method
    # Distinct hues remain identifiable when dashed lines overlap or shrink.
    # Keep line styles as a second cue in both the overview and full task grid.
    plot_colors = {
        'QRL': '#303030',
        'TD-InfoNCE': '#0072B2',
        'GCSL': '#E69F00',
        'C-Learning': '#CC79A7',
        'CRL': '#009E73',
        'NQRL': '#D55E00',
    }
    stem = 'representative_learning_curves' if representative else 'task_success_grid'
    pdf = canvas.Canvas(str(FIGURE_ROOT / f'{stem}.pdf'),
                        pagesize=(576, page_height), invariant=1, initialFontName=FONT)
    pdf.setTitle('Validation success: five-seed mean and sample SD')
    sizes = [28 + stringWidth(display(method), FONT, 11.5) for method in METHODS]
    left = (576 - sum(sizes) - 10 * (len(METHODS) - 1)) / 2
    for method, size in zip(METHODS, sizes):
        pdf.setDash(METHOD_DASHES[method])
        line(pdf, [(left, 12), (left + 19, 12)], plot_colors[method],
             1.8 if method == 'NQRL' else 1.15)
        pdf.setDash()
        label(pdf, left + 24, 9, display(method), 11.5,
              font=BOLD if method == 'NQRL' else FONT)
        left += size + 10
    panels = []
    for index, env in enumerate(environments):
        col, row = index % 4, index // 4
        left, bottom, width, height = (37 + col * 135,
                                      56 if representative else 440 - row * 130,
                                      115, 75 if representative else 60)
        task_title(pdf, left - 6, bottom + height + 22, width + 12, env.paper_name)
        def xy(step, success):
            return left + step / env.budget * width, bottom + success * height
        for value in (0, .5, 1):
            _, y = xy(0, value)
            pdf.setStrokeColor(GRID)
            pdf.setLineWidth(.45)
            pdf.line(left, y, left + width, y)
            if col == 0:
                label(pdf, left - 5, y - 3, f'{value:g}', 11.5, align='right')
        for fraction in (0, .5, 1):
            x, _ = xy(fraction * env.budget, 0)
            label(pdf, x, bottom - 13, f'{fraction * env.budget / 1000:.0f}',
                  11.5, align='left' if fraction == 0 else 'right' if fraction == 1 else 'center')
        groups = {
            method: [r for r in rows if r['environment'] == env.paper_name and r['method'] == method]
            for method in METHODS
        }
        pdf.saveState()
        clip = pdf.beginPath()
        clip.rect(left, bottom, width, height)
        pdf.clipPath(clip, stroke=0)
        # Bands use the archived sample SD; the physical success range clips only the display.
        for method in METHODS:
            group = groups[method]
            assert len(group) >= 2 and all(int(r['seeds']) == 5 for r in group)
            points = [(float(r['env_steps']), float(r['success_mean']), float(r['success_std']))
                      for r in group]
            assert [p[0] for p in points] == sorted({p[0] for p in points})
            band = pdf.beginPath()
            for i, (step, mean, sd) in enumerate(points):
                (band.moveTo if i == 0 else band.lineTo)(*xy(step, mean + sd))
            for step, mean, sd in reversed(points):
                band.lineTo(*xy(step, mean - sd))
            band.close()
            pdf.setFillColor(colors.HexColor(plot_colors[method]))
            pdf.setFillAlpha(.12)
            pdf.drawPath(band, stroke=0, fill=1)
        pdf.setFillAlpha(1)
        for method in METHODS:
            pdf.setDash(METHOD_DASHES[method])
            line(pdf, [xy(float(r['env_steps']), float(r['success_mean']))
                       for r in groups[method]], plot_colors[method],
                 1.65 if method == 'NQRL' else 1.05)
        pdf.restoreState()
        pdf.setStrokeColor(MUTED)
        pdf.setLineWidth(.5)
        pdf.line(left, bottom, left, bottom + height)
        pdf.line(left, bottom, left + width, bottom)
        label(pdf, left + width / 2, bottom - 29, 'Environment steps (k)', 10.5, align='center')
        panels.append({'environment': env.paper_name, 'row': row, 'column': col,
                       'budget': env.budget, 'ylim': [0, 1],
                       'points': {method: len(group) for method, group in groups.items()}})
    pdf.saveState()
    pdf.translate(11, 93.5 if representative else 275)
    pdf.rotate(90)
    label(pdf, 0, 0, 'Validation success', 12, align='center')
    pdf.restoreState()

    pdf.save()
    audit = {
        'layout': [1, 4] if representative else [4, 4],
        'empty_cells': [] if representative else [[3, 2], [3, 3]], 'panels': panels,
        'center': 'arithmetic mean over five training seeds',
        'band': 'mean plus/minus sample standard deviation (ddof=1)',
        'band_display': 'clipped at the physical [0,1] success range; stored statistics unchanged',
        'alignment': 'linear checkpoint alignment within common observed support; no smoothing',
        'source_sha256': hashlib.sha256((DATA_ROOT / 'per_task_learning_curves.csv').read_bytes()).hexdigest(),
        'colors': plot_colors, 'line_styles': METHOD_DASHES,
    }
    audit_name = 'representative_learning_audit.json' if representative else 'learning_grid_audit.json'
    (DATA_ROOT / audit_name).write_text(json.dumps(audit, indent=2) + '\n')


def task_curves():
    raw, auc_runs, means, provenance = [], [], [], []
    reference = {(r['environment'], r['method'], int(r['seed'])): float(r['auc_20_100'])
                 for r in read_csv('learning_runs.csv')}
    figure_tex = []
    for index, env in enumerate(ENVIRONMENTS):
        pdf = canvas.Canvas(str(FIGURE_ROOT / f'learning_{env.key}.pdf'),
                            pagesize=(360, 232), invariant=1)
        pdf.setTitle(env.paper_name + ': validation success')
        label(pdf, 44, 217, env.paper_name, 11, font=BOLD)
        xy = axes(pdf, (44, 40, 302, 136), env.budget, 0, 1, 'Validation success',
                  (0, .2, .4, .6, .8, 1), font_size=9.5)
        for method_index, method in enumerate(METHODS):
            curves = []
            for seed in SEEDS:
                path = artifact_dir(env, method, seed) / 'eval.log'
                records, _ = validation_curve(path)
                points = [(float(r['env_steps']), float(r['succ_rate'])) for r in records]
                assert points[0][0] <= .2 * env.budget and points[-1][0] == env.budget
                curves.append(points)
                raw.extend({'environment': env.paper_name, 'method': method, 'seed': seed,
                            'env_steps': r['env_steps'], 'optim_steps': r.get('optim_steps', 0),
                            'validation_success': r['succ_rate']} for r in records)
                interval = [(.2 * env.budget, interpolate(points, .2 * env.budget))]
                interval += [(x, y) for x, y in points if .2 * env.budget < x < env.budget]
                interval.append((env.budget, points[-1][1]))
                auc = trapz((x for x, _ in interval), (y for _, y in interval)) / (.8 * env.budget)
                assert abs(auc - reference[env.paper_name, method, seed]) < 1e-12
                auc_runs.append({'environment': env.paper_name, 'method': method, 'seed': seed,
                                 'interval_start': .2 * env.budget, 'interval_end': env.budget,
                                 'normalized_auc': auc})
                provenance.append({'environment': env.paper_name, 'method': method, 'seed': seed,
                                   'eval_log_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            # Align checkpoints by interpolation only where all five seeds have observations.
            start = max(c[0][0] for c in curves)
            grid = sorted({x for c in curves for x, _ in c if start <= x <= env.budget})
            plot = []
            for step in grid:
                values = [interpolate(c, step) for c in curves]
                mean = statistics.mean(values)
                means.append({'environment': env.paper_name, 'method': method, 'env_steps': int(step),
                              'success_mean': mean, 'success_std': statistics.stdev(values), 'seeds': 5})
                plot.append(xy(step, mean))
            line(pdf, plot, PALETTE[method], 2 if method == 'NQRL' else 1.25)
            x = 44 + (method_index % 3) * 104
            y = 201 - (method_index // 3) * 13
            line(pdf, [(x, y), (x + 13, y)], PALETTE[method], 2 if method == 'NQRL' else 1.25)
            label(pdf, x + 17, y - 2.5, method, 9,
                  font=BOLD if method == 'NQRL' else FONT)
        pdf.save()
        if index % 2 == 0:
            figure_tex.extend([r'\begin{figure}[p]', r'\centering'])
        task_label = env.paper_name.replace('_', r'\_')
        figure_tex.extend([
            r'\begin{minipage}[t]{\textwidth}',
            rf'\includegraphics[width=\linewidth]{{figures/learning_{env.key}.pdf}}',
            rf'\caption{{{task_label}.}}',
            rf'\label{{fig:learning-{env.key}}}', r'\end{minipage}',
        ])
        if index % 2 == 1 or index == len(ENVIRONMENTS) - 1:
            figure_tex.extend([r'\end{figure}', r'\clearpage'])
        else:
            figure_tex.append(r'\par\vspace{1.2em}')
    (FIGURE_ROOT / 'task_learning_curves.tex').write_text('\n'.join(figure_tex) + '\n')
    summary = []
    for env in ENVIRONMENTS:
        for method in METHODS:
            values = [r['normalized_auc'] for r in auc_runs
                      if (r['environment'], r['method']) == (env.paper_name, method)]
            summary.append({'environment': env.paper_name, 'method': method,
                            'auc_mean': statistics.mean(values), 'auc_std': statistics.stdev(values),
                            'seeds': len(values), 'interval_start': .2 * env.budget, 'interval_end': env.budget})
    write_csv('per_task_auc_runs.csv', auc_runs)
    write_csv('per_task_auc.csv', summary)
    write_csv('validation_checkpoints.csv', raw)
    write_csv('per_task_learning_curves.csv', means)
    (DATA_ROOT / 'per_task_learning_audit.json').write_text(json.dumps({
        'runs': len(auc_runs), 'tasks': len(ENVIRONMENTS), 'methods': list(METHODS),
        'curve': 'One arithmetic seed-mean curve per method; linear checkpoint alignment without extrapolation or smoothing',
        'auc': 'Trapezoidal integral of each original seed curve over 20%-100% budget, divided by interval length; seed mean and sample SD',
        'validation_not_test': True, 'sources': provenance,
    }, indent=2) + '\n')
    table = [
        r'\begin{table}[htbp]',
        r'\caption{Normalized validation-success AUC (\%) by task over 20\%--100\% of its interaction budget.',
        r'Means $\pm$ sample SD over five training seeds. Bold marks the highest mean.',
        r'Task names are abbreviated as in Table~\ref{tab:main}.}',
        r'\label{tab:auc-by-task}', r'\centering\nqrltablestyle',
        r'\setlength{\tabcolsep}{5pt}', r'\renewcommand{\arraystretch}{1.12}',
        r'\nqrlpairsetup{$\mathbf{99.9}$}{$\mathbf{99.9}$}',
        r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}ccccccc@{}}', r'\toprule',
        r'Task & QRL & TD-InfoNCE & GCSL & C-Learning & CRL & \method{} (Ours) \\', r'\midrule',
    ]
    for env in ENVIRONMENTS:
        rows = [r for r in summary if r['environment'] == env.paper_name]
        best = max(r['auc_mean'] for r in rows)
        cells = []
        for row in rows:
            style = r'\mathbf' if row['auc_mean'] == best else ''
            cells.append(r'\nqrlpm{' + style + '}{' +
                         f"{100 * row['auc_mean']:.1f}" + '}{' +
                         f"{100 * row['auc_std']:.1f}" + '}')
        label = task_label(env.paper_name)
        table.append(label + ' & ' + ' & '.join(cells) + r' \\')
    table.extend([r'\bottomrule', r'\end{tabular*}', r'\end{table}'])
    (PAPER_ROOT / 'tables/validation_auc.tex').write_text('\n'.join(table) + '\n')


def collect_dynamics_validation():
    """Extract the exact checkpoints of the six auxiliary dynamics runs."""
    raw, summary, sources = [], [], []
    runs = read_csv('dynamics_mse_runs.csv')
    assert {(r['variant'], int(r['seed'])) for r in runs} == {
        (variant, seed) for variant in ('A01', 'A05') for seed in range(1000, 1003)}
    for run in runs:
        path = Path(run['run_dir']) / 'eval.log'
        records, duplicates = validation_curve(path)
        assert duplicates['duplicate_keys'] == 0
        assert [int(r['env_steps']) for r in records] == list(range(20000, 500001, 20000))
        for r in records:
            assert r['split'] == 'validation' and r['num_episodes'] == 100
            assert r['success_count'] / r['num_episodes'] == r['succ_rate']
            raw.append(dict(variant=run['variant'], seed=int(run['seed']),
                            env_steps=int(r['env_steps']), validation_success=r['succ_rate'],
                            num_episodes=r['num_episodes'], episode_seed=r['seed'],
                            episode_seed_end=r['seed_end']))
        sources.append(dict(variant=run['variant'], seed=int(run['seed']),
                            eval_log=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    for variant in ('A01', 'A05'):
        for step in range(20000, 500001, 20000):
            values = [r['validation_success'] for r in raw
                      if r['variant'] == variant and r['env_steps'] == step]
            assert len(values) == 3
            summary.append(dict(variant=variant, env_steps=step, success_mean=statistics.mean(values),
                                success_std=statistics.stdev(values), seeds=3))
    write_csv('dynamics_validation_runs.csv', raw)
    write_csv('dynamics_validation_curves.csv', summary)
    (DATA_ROOT / 'dynamics_validation_audit.json').write_text(json.dumps(dict(
        runs=6, checkpoints=len(raw), sources=sources,
        curve='Arithmetic mean and sample SD across three training seeds at identical validation checkpoints',
        evaluation='100 validation episodes at each checkpoint, every 20k environment interactions',
        alignment='Exact common checkpoints; no interpolation, smoothing or extrapolation',
    ), indent=2) + '\n')


def goal_fraction_success():
    """Display the approved QRL success and NQRL gain curves with sample SD."""
    from draw_nqrl_goal_fraction import render_goal_fraction
    render_goal_fraction(FIGURE_ROOT / 'goal_fraction_success.pdf')


def dynamics_curves(*, write_data=True):
    rows = read_csv('dynamics_mse_ant_trajectories.csv')
    summary = []
    late_rows = []
    pdf = canvas.Canvas(str(FIGURE_ROOT / 'dynamics_prediction.pdf'),
                        pagesize=(560, 300), invariant=1, initialFontName=SERIF)
    pdf.setTitle('AntNavigate dynamics: IQE plateau and growing latent prediction error')
    variant_colors = {'A01': '#AF4B42', 'A05': '#2F6F8C'}
    variant_names = {'A01': 'IQE-only', 'A05': 'IQE+MSE'}
    metric_styles = [('sq_dists', 'IQE loss', ()), ('mse', 'MSE loss', (4, 2.5))]
    error_legend = [(f'{variant_names[variant]}: {name}', color, dash)
                    for variant, color in variant_colors.items()
                    for _, name, dash in metric_styles]
    widths = [24 + stringWidth(name, SERIF, 11.5) for name, _, _ in error_legend]
    legend_width = sum(widths) + 12 * (len(widths)-1)
    assert legend_width <= 540, 'Error legend exceeds the figure width'
    x = (560-legend_width)/2
    for (name, color, dash), width in zip(error_legend, widths):
        pdf.setDash(dash)
        line(pdf, [(x, 285), (x + 18, 285)], color, 1.35)
        pdf.setDash()
        label(pdf, x + 24, 282, name, 11.5, font=SERIF)
        x += width + 12
    label(pdf, 56, 264, '(a) Training trajectories', 12, font=SERIF_BOLD)
    label(pdf, 397, 264, '(b) 400k-500k change', 12, font=SERIF_BOLD)
    error_box = (56, 158, 272, 94)
    success_box = (56, 35, 492, 60)
    pdf.setFillColor(colors.HexColor('#F5F5F5'))
    pdf.rect(56 + .8 * 272, 158, .2 * 272, 94, stroke=0, fill=1)
    xy = axes(pdf, error_box, 500000, .0001, 1e5,
              'Prediction error', (.0001, .01, 1, 100, 10000), log=True,
              font_size=11.5, font=SERIF)
    detail_xy = axes(pdf, (397, 158, 151, 94), 500000, -10, 40,
                     'Change from 400k (%)', (-10, 0, 20, 40),
                     font_size=11, xmin=400000, font=SERIF)
    series = []
    for metric, _, dash in metric_styles:
        for variant, color in variant_colors.items():
            means = []
            for step in range(20000, 500001, 20000):
                values = [float(r[metric]) for r in rows
                          if r['variant'] == variant and int(r['env_steps']) == step]
                assert len(values) == 3
                mean, sd = statistics.mean(values), statistics.stdev(values)
                summary.append({'variant': variant, 'metric': metric, 'window_end': step,
                                'mean': mean, 'sample_std': sd, 'seeds': 3})
                means.append((step, mean, sd))
            series.append((color, dash, means, variant, metric))
    pdf.saveState()
    clip = pdf.beginPath()
    clip.rect(*error_box)
    pdf.clipPath(clip, stroke=0)
    for color, _, means, _, _ in series:
        band = pdf.beginPath()
        for i, (step, mean, sd) in enumerate(means):
            assert mean - sd > 0, 'Nonpositive SD bound cannot be shown on a log axis'
            (band.moveTo if i == 0 else band.lineTo)(*xy(step, mean + sd))
        for step, mean, sd in reversed(means):
            band.lineTo(*xy(step, mean - sd))
        band.close()
        pdf.setFillColor(colors.HexColor(color))
        pdf.setFillAlpha(.18)
        pdf.drawPath(band, stroke=0, fill=1)
    pdf.restoreState()
    for color, dash, means, variant, metric in series:
        reference = next(mean for step, mean, _ in means if step == 400000)
        detail = []
        for step, mean, _ in means:
            if step < 400000:
                continue
            change = 100 * (mean / reference - 1)
            late_rows.append({'variant': variant, 'metric': metric, 'window_end': step,
                              'error_mean': mean, 'reference_mean_400k': reference,
                              'relative_change_pct': change, 'seeds': 3})
            detail.append(detail_xy(step, change))
        pdf.setDash(dash)
        line(pdf, [xy(step, mean) for step, mean, _ in means], color, 1.5)
        line(pdf, detail, color, 1.35)
        pdf.setDash()
        pdf.setStrokeColor(colors.HexColor(color))
        pdf.setFillColor(colors.white)
        pdf.setLineWidth(.9)
        for point in detail:
            pdf.circle(*point, 1.65, stroke=1, fill=1)
    label(pdf, 56, 108, '(c) Validation success', 12, font=SERIF_BOLD)
    success_dashes = {'A01': (4, 2.5), 'A05': ()}
    for x, (variant, color) in zip((329, 445), variant_colors.items()):
        pdf.setDash(success_dashes[variant])
        line(pdf, [(x, 111), (x+18, 111)], color, 1.7)
        pdf.setDash()
        label(pdf, x+24, 108, variant_names[variant], 11.5, font=SERIF)
    success_xy = axes(pdf, success_box, 500000, 0, 1,
                      'Success rate', (0, .5, 1), font_size=11.5, font=SERIF)
    success_rows = read_csv('dynamics_validation_curves.csv')
    pdf.saveState()
    clip = pdf.beginPath()
    clip.rect(*success_box)
    pdf.clipPath(clip, stroke=0)
    for variant, color in variant_colors.items():
        points = [(int(r['env_steps']), float(r['success_mean']), float(r['success_std']))
                  for r in success_rows if r['variant'] == variant]
        assert len(points) == 25
        band = pdf.beginPath()
        for i, (step, mean, sd) in enumerate(points):
            (band.moveTo if i == 0 else band.lineTo)(*success_xy(step, mean + sd))
        for step, mean, sd in reversed(points):
            band.lineTo(*success_xy(step, mean - sd))
        band.close()
        pdf.setFillColor(colors.HexColor(color))
        pdf.setFillAlpha(.18)
        pdf.drawPath(band, stroke=0, fill=1)
        pdf.setFillAlpha(1)
        pdf.setDash(success_dashes[variant])
        line(pdf, [success_xy(step, mean) for step, mean, _ in points], color, 1.7)
        pdf.setDash()
        pdf.setStrokeColor(colors.HexColor(color))
        pdf.setFillColor(colors.white)
        for step, mean, _ in points[::2]:
            pdf.circle(*success_xy(step, mean), 1.6, stroke=1, fill=1)
    pdf.restoreState()
    pdf.save()
    if write_data:
        write_csv('dynamics_trajectory_summary.csv', summary)
        write_csv('dynamics_late_window.csv', late_rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figures-only', action='store_true',
                        help='Render archived curve statistics without rewriting experimental CSVs.')
    parser.add_argument('--dynamics-only', action='store_true',
                        help='Render only the dynamics figure.')
    parser.add_argument('--collect-dynamics-validation', action='store_true',
                        help='Extract the six auxiliary validation logs before plotting.')
    parser.add_argument('--goal-fraction-only', action='store_true',
                        help='Render only QRL success and NQRL gain versus non-goal coordinate fraction.')
    args = parser.parse_args()
    register_fonts()
    if args.goal_fraction_only:
        if args.dynamics_only or args.collect_dynamics_validation:
            parser.error('--goal-fraction-only cannot be combined with dynamics flags')
        goal_fraction_success()
        print('Rendered QRL success and NQRL gain versus non-goal coordinate fraction.')
        raise SystemExit(0)
    if args.collect_dynamics_validation:
        collect_dynamics_validation()
    if not args.figures_only and not args.dynamics_only:
        task_curves()
    if not args.dynamics_only:
        task_grid()
        task_grid(representative=True)
        goal_fraction_success()
    dynamics_curves(write_data=not args.figures_only)
    print('Rendered the dynamics figure.' if args.dynamics_only else
          'Rendered the 14-task 4x4 mean/SD grid, non-goal coordinate fraction comparison, and dynamics figure.')

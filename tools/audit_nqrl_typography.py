"""Audit active figure fonts and actual LaTeX table fonts without editing the paper."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET

PAPER = Path(__file__).resolve().parents[1] / 'iclr2027'


def run(*args, cwd=PAPER):
    return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.STDOUT)


def main():
    source = (PAPER / 'nqrl_iclr2027.tex').read_text()
    active = re.sub(r'(?<!\\)%[^\n]*', '', source)
    figures = re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', active)
    tables = re.findall(r'\\input\{(tables/[^}]+)\}', active)
    report = {'figures': [], 'tables': [], 'passed': False}
    for name in figures:
        path = PAPER / name
        if path.suffix.lower() != '.pdf':
            report['figures'].append({'file': name, 'status': 'raster: skipped'})
            continue
        rows = run('pdffonts', str(path)).splitlines()[2:]
        fonts = []
        for row in rows:
            parts = row.split()
            font = parts[0].split('+')[-1]
            assert font.startswith(('NimbusRom', 'CM', 'MSAM', 'MSBM')), (name, row)
            assert parts[-5] == 'yes', (name, 'font not embedded', row)
            fonts.append(font)
        assert any(f.startswith('NimbusRom') for f in fonts), (name, fonts)
        if path.stem in ('nqrl_overview', 'nqrl_residual_mlp'):
            assert any(f.startswith('CMMI') for f in fonts), (name, 'missing native math')
        tree = ET.fromstring(run('pdftotext', '-bbox', str(path), '-'))
        ns = {'x': 'http://www.w3.org/1999/xhtml'}
        for page in tree.findall('.//x:page', ns):
            w, h = float(page.attrib['width']), float(page.attrib['height'])
            for word in page.findall('.//x:word', ns):
                x0, y0, x1, y1 = (float(word.attrib[k]) for k in ('xMin','yMin','xMax','yMax'))
                assert x0 >= -.1 and y0 >= -.1 and x1 <= w+.1 and y1 <= h+.1, (name, word.text)
        report['figures'].append({'file': name, 'fonts': sorted(set(fonts)),
                                  'embedded': True, 'text_inside_page': True, 'status': 'pass'})

    # Instrument a disposable copy at actual tabular/longtable entry, retaining
    # the full manuscript preamble, local size commands, and table contents.
    hook = r'''
\makeatletter
\let\nqrlfonttable\relax
\newcommand{\nqrlfontcheck}{%
  \ifx\nqrlfonttable\relax\else
    \typeout{NQRL-TABLE:\nqrlfonttable}
    \typeout{NQRL-TEXT:\f@family:\f@size:\fontname\font}
    \begingroup
    \setbox0=\hbox{$x+1$}
    \typeout{NQRL-MATH:\fontname\textfont0:\fontname\textfont1:\fontname\textfont2}
    \endgroup
  \fi}
\AddToHook{env/tabular/begin}{\nqrlfontcheck}
\AddToHook{env/tabular*/begin}{\nqrlfontcheck}
\AddToHook{env/longtable/begin}{\nqrlfontcheck}
\makeatother
'''
    instrumented = source.replace(r'\begin{document}', hook + r'\begin{document}', 1)
    for name in tables:
        table = (PAPER / name).read_text()
        assert not re.search(r'\\(?:texttt|textsf|ttfamily|sffamily|fontfamily|small|footnotesize|scriptsize|tiny|resizebox|scalebox)\b', table), name
        command = rf'\input{{{name}}}'
        instrumented = instrumented.replace(command, rf'\def\nqrlfonttable{{{name}}}' + command + r'\let\nqrlfonttable\relax')
    with tempfile.TemporaryDirectory(prefix='nqrl-typography-') as tmp:
        tmp = Path(tmp)
        for ext in ('aux', 'bbl', 'out'):
            if (PAPER / f'main.{ext}').exists():
                shutil.copy2(PAPER / f'main.{ext}', tmp / f'main.{ext}')
        (tmp / 'audit.tex').write_text(instrumented)
        run('pdflatex', '-interaction=nonstopmode', '-halt-on-error', '-jobname=main',
            f'-output-directory={tmp}', str(tmp / 'audit.tex'))
        log = (tmp / 'main.log').read_text()
    checks = re.findall(r'NQRL-TABLE:([^\n]+)\nNQRL-TEXT:([^\n]+)\nNQRL-MATH:([^\n]+)', log)
    assert set(name for name, _, _ in checks) == set(tables), checks
    for name in tables:
        entries = [(text, math) for table, text, math in checks if table == name]
        for text, math in entries:
            assert text == 'ptm:10:ptmr7t', (name, text)
            assert math == 'cmr10:cmmi10:cmsy10', (name, math)
        report['tables'].append({'file': name, 'text': entries[0][0], 'math': entries[0][1],
                                'sha256': hashlib.sha256((PAPER/name).read_bytes()).hexdigest(),
                                'status': 'pass'})
    report['passed'] = True
    (PAPER / 'data/typography_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    skipped = sum(item['status'] == 'raster: skipped' for item in report['figures'])
    print(f'PASS: {len(figures)-skipped} PDF figures, {len(tables)} table sources; {skipped} raster figures skipped.')


if __name__ == '__main__':
    main()

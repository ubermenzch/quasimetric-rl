"""Keep ReportLab graphics, typeset figure labels with the manuscript's LaTeX.

The graphics PDF contains no text fonts. A generated TikZ document overlays
embedded Nimbus Roman text and native LaTeX math at the original coordinates.
No experimental values, graphics paths, image pixels, or plot scales change.
"""
from html import unescape
from os.path import relpath
from pathlib import Path
import json
import re
import subprocess

from reportlab.pdfgen.canvas import Canvas as GraphicsCanvas

PAPER = Path(__file__).resolve().parents[1] / 'iclr2027'


def escape_text(value):
    replacements = {'\\': r'\textbackslash{}', '{': r'\{', '}': r'\}',
                    '$': r'\$', '&': r'\&', '#': r'\#', '%': r'\%',
                    '_': r'\_', '^': r'\textasciicircum{}', '~': r'\textasciitilde{}',
                    '−': '-', '–': '--', '—': '---', '±': r'$\pm$'}
    return ''.join(replacements.get(c,c) for c in str(value))


def math_source(value):
    """Translate the existing diagram's inline markup into real TeX math."""
    value=unescape(value).replace('\u00a0',' ')
    value=value.replace('z\u0302',r'\hat{z}').replace('ẑ',r'\hat{z}')
    value=value.replace('d\u0304',r'\bar{d}')
    value=re.sub(r'</?i>','',value)
    value=value.replace('<sub>','_{').replace('</sub>','}')
    value=value.replace('<sup>','^{').replace('</sup>','}')
    symbols={'θ':r'\theta ', 'π':r'\pi ', 'ψ':r'\psi ', 'φ':r'\varphi ',
             'ϕ':r'\phi ', 'σ':r'\sigma ', 'ρ':r'\rho ', 'ε':r'\varepsilon ',
             'ω':r'\omega ', 'μ':r'\mu ', '∇':r'\nabla ', 'Δ':r'\Delta ',
             'η':r'\eta ', '←':r'\leftarrow ', '−':'-'}
    value=''.join(symbols.get(c,c) for c in value)
    value=re.sub(r'\bsoftplus\b',lambda _:r'\operatorname{softplus}',value)
    for word in ('Policy','IQE'):
        value=value.replace(word,rf'\text{{{word}}}\;')
    assert '<' not in value and '>' not in value,value
    return '$'+value+'$'


class Canvas(GraphicsCanvas):
    def __init__(self,filename,*args,**kwargs):
        self.output=Path(filename).resolve()
        self.source_dir=self.output.parent/'latex_sources'/self.output.stem
        self.source_dir.mkdir(parents=True,exist_ok=True)
        self.labels=[]
        super().__init__(str(self.source_dir/'graphics.pdf'),*args,**kwargs)

    def _make_preamble(self):
        # Avoid ReportLab's otherwise-unused, unembedded initial font resource.
        assert self.bottomup, 'The paper figure generators use bottom-up coordinates'
        self._preamble='1 0 0 1 0 0 cm'

    def latex_label(self,x,y,value,size=9,*,font='Times-Roman',color=None,
                    align='left',math=False,center_y=False):
        content=value if math else escape_text(value)
        rgb=(color.red,color.green,color.blue) if color is not None else (0,0,0)
        self.labels.append(dict(x=x,y=y,text=str(value),latex=content,size=size,
            bold='Bold' in font,italic='Italic' in font,align=align,center_y=center_y,
            color=rgb,matrix=self._currentMatrix,math=math))

    def save(self):
        super().save()
        w,h=self._pagesize
        relative_math=Path(relpath(PAPER/'math_commands.tex',self.source_dir))
        lines=[r'\documentclass[tikz,border=0pt]{standalone}',
               r'\usepackage{times}',rf'\input{{{relative_math.as_posix()}}}',
               r'\usepackage{amssymb}',r'\pdfinfoomitdate=1',r'\pdftrailerid{}',
               r'\pdfsuppressptexinfo=15',r'\pdfinfo{/Author (Anonymous)}',
               r'\begin{document}',r'\begin{tikzpicture}[x=1bp,y=1bp]',
               rf'\path[use as bounding box] (0,0) rectangle ({w},{h});',
               rf'\node[anchor=south west,inner sep=0pt,outer sep=0pt] at (0,0) {{\includegraphics[width={w}bp,height={h}bp]{{graphics.pdf}}}};']
        for i,item in enumerate(self.labels):
            a,b,c,d,e,f=item['matrix']
            rgb=','.join(f'{v:.6f}' for v in item['color'])
            lines.append(rf'\definecolor{{label{i}}}{{rgb}}{{{rgb}}}')
            anchor=({'left':'west','center':'center','right':'east'} if item['center_y']
                    else {'left':'base west','center':'base','right':'base east'})[item['align']]
            font=rf'\fontsize{{{item["size"]}bp}}{{{item["size"]*1.2}bp}}\selectfont'
            if item['bold']:font+=r'\bfseries'
            if item['italic'] and not item['math']:font+=r'\itshape'
            matrix=','.join(f'{v:.9f}' for v in (a,b,c,d))
            lines.append(rf'\begin{{scope}}[cm={{{matrix},({e:.9f},{f:.9f})}},transform shape]')
            lines.append(rf'\node[anchor={anchor},inner sep=0pt,outer sep=0pt,text=label{i},font={{{font}}}] at ({item["x"]},{item["y"]}) {{{item["latex"]}}};')
            lines.append(r'\end{scope}')
        lines.extend([r'\end{tikzpicture}',r'\end{document}'])
        source=self.source_dir/'labels.tex'
        source.write_text('\n'.join(lines)+'\n')
        (self.source_dir/'labels.json').write_text(json.dumps(self.labels,indent=2)+'\n')
        build=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error',source.name],
            cwd=self.source_dir,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        if build.returncode:
            raise RuntimeError(build.stdout[-6000:])
        log=(self.source_dir/'labels.log').read_text()
        assert not re.search(r'Missing character:|Overfull|undefined|^!',log,re.M),log
        self.output.write_bytes((self.source_dir/'labels.pdf').read_bytes())

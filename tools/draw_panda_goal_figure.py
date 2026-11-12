"""Build a standalone vector-only Panda version of the Figure 2 concept.

Run after project_panda_vector_meshes.py, using system python3 (ReportLab).
The preview does not replace the manuscript's accepted Figure 2.
"""
from pathlib import Path
import json
import subprocess
import numpy as np
from reportlab.pdfgen.canvas import Canvas

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'iclr2027/figures/panda_vector_preview'


class VectorCanvas(Canvas):
    def _make_preamble(self):
        self._preamble = '1 0 0 1 0 0 cm'


def icons():
    metadata = json.loads((OUT/'poses.json').read_text())
    x0,y0,x1,y1 = metadata['projection_bounds']
    scale = 160/(y1-y0)
    width,height = (x1-x0)*scale+6,166
    for name in ['current']+[f'goal{i}' for i in range(1,7)]:
        data = np.load(OUT/f'{name}.npz')
        triangles = (data['triangles']-[x0,y0])*scale+3
        c = VectorCanvas(str(OUT/f'{name}.pdf'),pagesize=(width,height),pageCompression=1)
        c.setTitle('Panda visual mesh: '+name)
        c.setAuthor('Anonymous')
        for triangle,rgb in zip(triangles,data['colors']):
            c.setFillColorRGB(*rgb)
            # A hairline of the same face color prevents antialiasing seams.
            c.setStrokeColorRGB(*rgb)
            c.setLineWidth(.035)
            path=c.beginPath();path.moveTo(*triangle[0])
            path.lineTo(*triangle[1]);path.lineTo(*triangle[2]);path.close()
            c.drawPath(path,stroke=1,fill=1)
        if name!='current':
            gx,gy=(data['goal']-[x0,y0])*scale+3
            c.setStrokeColorRGB(.08,.48,.44);c.setLineWidth(1.2)
            c.line(gx-3,gy,gx+3,gy);c.line(gx,gy-3,gx,gy+3)
        c.save()
        print('Vector icon:',name,flush=True)


def layout():
    tex = r'''\documentclass[tikz,border=2pt]{standalone}
\usepackage{times,amsmath,amssymb}
\usetikzlibrary{arrows.meta,shapes.geometric}
\pdfinfo{/Title (Panda position-goal illustration: QRL versus NQRL) /Author (Anonymous)}
\definecolor{teal}{HTML}{197D73}
\definecolor{goalblue}{HTML}{B5CCE9}
\definecolor{unknown}{HTML}{E7E8EA}
\definecolor{gold}{HTML}{F5D774}
\definecolor{grayline}{HTML}{A3A7AB}
\begin{document}
\begin{tikzpicture}[x=1cm,y=1cm,line join=round,line cap=round,
 font=\fontsize{9}{10.5}\selectfont,
 >={Stealth[length=2mm,width=1.5mm]}]
\path[use as bounding box] (0,0) rectangle (13.1,8.05);
% The panel outline echoes the existing Figure 2 while leaving robots unboxed.
\draw[black!65,line width=.65pt,rounded corners=7pt] (.04,4.23) rectangle (8.06,7.94);
\draw[black!65,line width=.65pt,rounded corners=7pt] (.04,.32) rectangle (8.06,4.04);
\draw[black!75,line width=.85pt,dash pattern=on 3pt off 2.2pt,rounded corners=9pt]
 (8.30,.32) rectangle (13.05,7.94);
\node[anchor=west,font=\fontsize{10}{12}\selectfont\bfseries] at (.30,7.65) {(a) QRL};
\node[anchor=west,font=\fontsize{10}{12}\selectfont\bfseries] at (.30,3.75) {(b) NQRL};
\node[align=center,font=\fontsize{9}{10.5}\selectfont\bfseries] at (10.68,7.55)
 {Conditional goal set};
\node[align=center,font=\fontsize{8}{9}\selectfont] at (10.68,7.13)
 {Same end-effector position\\$(x^*,y^*,z^*)$};
% Robot-only vector projections: shared camera, scale and target marker.
\node[anchor=south west,inner sep=0pt] at (.38,4.55) {\includegraphics[height=1.85cm]{current.pdf}};
\node[anchor=south west,inner sep=0pt] at (5.72,4.55) {\includegraphics[height=1.85cm]{goal4.pdf}};
\node[anchor=south west,inner sep=0pt] at (.38,.60) {\includegraphics[height=2.03cm]{current.pdf}};
% Shared image bounds and placement give every pose the exact same base.
% Isolated transparency groups apply 70% transparency to each whole arm,
% rather than separately blending its many mesh faces.
\foreach \pose in {6,5,4,3,2}{
 \begin{scope}[transparency group,opacity=.30]
  \node[anchor=south west,inner sep=0pt] at (8.60,1.58)
   {\includegraphics[height=4.40cm]{goal\pose.pdf}};
 \end{scope}
}
% Selected completion is drawn last, fully opaque.
\node[anchor=south west,inner sep=0pt] at (8.60,1.58)
 {\includegraphics[height=4.40cm]{goal1.pdf}};
\node[align=center,font=\fontsize{8}{9}\selectfont] at (10.68,6.42)
 {One Panda, multiple states};
\node[star,star points=5,star point ratio=2.25,draw=black!65,fill=gold,
 minimum size=3.8mm,inner sep=0pt,line width=.4pt] at (11.50,5.77) {};
\node[align=center,font=\fontsize{8}{9}\selectfont] at (10.68,1.12)
 {Solid: selected completion\\Faint: alternative completions};
\node[anchor=west,font=\fontsize{8}{9}\selectfont] at (.41,4.42) {Current state};
\node[font=\fontsize{8}{9}\selectfont] at (6.60,4.42) {One completion};
\node[anchor=west,font=\fontsize{8}{9}\selectfont] at (.41,.48) {Current state};
% A complete state goal chooses one completion; partial goals leave it free.
\draw[grayline,->,line width=1.1pt,dash pattern=on 3pt off 2pt]
 (2.16,5.34) -- (5.56,5.34);
\node[fill=white,inner sep=1pt] at (3.91,5.65) {$d(s',g_{\mathrm{full}})$};
\draw[black,->,line width=1.1pt,dash pattern=on 3.2pt off 2pt]
 (2.33,1.64) -- (7.65,1.64) .. controls (8.40,1.64) and (8.40,2.85) .. (9.15,3.20);
\node[fill=white,inner sep=1pt] at (5.57,1.88) {$d(s',g_{\mathrm{full}})$};
\node[fill=gold,draw=black!65,line width=.55pt,rounded corners=3pt,
 align=center,inner sep=4pt] at (5.38,.86) {Approximate nearest\\completion};
\draw[black!70,line width=.6pt] (7.13,.92) -- (9.32,1.93);
% Input vectors are schematic; ellipses stand for omitted state coordinates.
'''
    def vector(x,y,filled):
        parts=[]
        labels=[r'$x^*$',r'$y^*$',r'$z^*$', '$0$' if filled else '?',
                '$0$' if filled else '?',r'$\cdots$']
        for i,label in enumerate(labels):
            a=x+i*.43
            parts.append(r'\draw[fill='+('goalblue' if i<3 else 'unknown')+
                         f',line width=.65pt] ({a:.3f},{y:.3f}) rectangle ({a+.43:.3f},{y+.40:.3f});')
            parts.append(f'\\node[inner sep=0pt] at ({a+.215:.3f},{y+.20:.3f}) {{{label}}};')
        return '\n'.join(parts)
    tex+=vector(.38,6.65,False)+'\n'+vector(5.02,6.65,True)+'\n'+vector(.38,2.74,False)
    tex+=r'''
\node[anchor=west,font=\fontsize{8}{9}\selectfont] at (.38,7.22) {Partial goal};
\node[anchor=west,font=\fontsize{8}{9}\selectfont] at (5.02,7.22) {Zero-filled goal};
\draw[grayline,->,line width=2pt] (3.10,6.85) -- (4.87,6.85);
\node[align=center,font=\fontsize{8}{9}\selectfont] at (3.98,7.31) {Zero-fill other\\coordinates};
\node[anchor=west,font=\fontsize{8}{9}\selectfont] at (.38,3.31) {Partial goal};
\draw[teal,->,line width=1.1pt] (3.14,2.94) -- (7.85,2.94);
\node[fill=white,inner sep=1.5pt,align=center,font=\fontsize{8}{9}\selectfont]
 at (5.54,3.38) {Fix goal encoding;\\optimize non-goal latent};
\node[anchor=west,font=\fontsize{7.3}{8.5}\selectfont] at (.10,.08)
 {Schematic state inputs and Panda poses; not policy rollouts.};
\end{tikzpicture}
\end{document}
'''
    path=OUT/'figure2_panda_vector.tex';path.write_text(tex)
    proc=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error',path.name],
                        cwd=OUT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    if proc.returncode:
        raise RuntimeError(proc.stdout[-5000:])
    (OUT.parent/'figure2_panda_vector.pdf').write_bytes(path.with_suffix('.pdf').read_bytes())
    print('Generated',OUT.parent/'figure2_panda_vector.pdf')


if __name__=='__main__':
    icons()
    layout()

"""Generate the illustrative NCC landscape as native TikZ paths and LaTeX text.

Run: python3 tools/draw_ncc_search_vector.py
The analytic surface is a schematic, not experimental data or a fitted critic.
"""
from pathlib import Path
import math
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'iclr2027/figures/ncc_vector'


def cost(u, v):
    return (.18 + 1.6 * math.exp(-((u-.7)**2+(v-3)**2)/.7)
            + 1.1 * math.exp(-((u-3.2)**2+(v-3.3)**2)/.6)
            + .65 * math.exp(-((u-3.6)**2+(v-.5)**2)/.7)
            + .10 * math.sin(1.7*u) * math.cos(v))


def project(u, v, z):
    return 3.35 + .96*u - .80*v, .55 + .34*u + .39*v + .85*z


def point(p):
    return f'({p[0]:.5f},{p[1]:.5f})'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tex = [r'''\documentclass[tikz,border=2pt]{standalone}
\usepackage{times}
\input{../../math_commands.tex}
\usepackage{amssymb}
\usetikzlibrary{arrows.meta,decorations.pathreplacing}
\pdfinfo{/Title (Nearest conditional completion: vector schematic) /Author (Anonymous)}
\definecolor{teal}{HTML}{137E83}
\definecolor{gold}{HTML}{B67D18}
\definecolor{tealbg}{HTML}{C9E7E6}
\definecolor{goldbg}{HTML}{F7E4B9}
\definecolor{rule}{HTML}{879096}
\begin{document}
\begin{tikzpicture}[x=1cm,y=1cm,line cap=round,line join=round,
 font=\fontsize{9}{10.5}\selectfont,
 >={Stealth[length=2.1mm,width=1.55mm]}]
\path[use as bounding box] (0,.35) rectangle (13.1,5.05);
''']
    # Back-to-front, individually colored vector quadrilaterals.
    n = 36
    cells = sorted(((i, j) for i in range(n) for j in range(n)),
                   key=lambda ij: sum(ij), reverse=True)
    for i, j in cells:
        u, v, step = 4*i/n, 4*j/n, 4/n
        t = max(0., min(1., cost(u+step/2, v+step/2)/1.9))
        lo, mid, hi = (91, 157, 178), (185, 220, 206), (252, 220, 139)
        a, b, f = (lo, mid, t*2) if t < .5 else (mid, hi, (t-.5)*2)
        rgb = [round(x+(y-x)*f) for x, y in zip(a, b)]
        vertices = [(u,v), (u+step,v), (u+step,v+step), (u,v+step)]
        path = ' -- '.join(point(project(x,y,cost(x,y))) for x,y in vertices)
        tex.append(r'\definecolor{cell}{RGB}{'+','.join(map(str,rgb))+'}\n'
                   +r'\path[fill=cell,draw=teal!32!white,line width=.12pt] '
                   +path+' -- cycle;')
    # Coordinate axes meet below the foreground edge of the surface.
    origin = project(0, 0, 0)
    for u, v, label, anchor in [(4.15, 0, 'h_1', 'north'), (0, 4.15, 'h_2', 'north east')]:
        end = project(u, v, 0)
        tex.append(r'\draw[->,line width=.7pt] '+point(origin)+' -- '+point(end)+';')
        mid = ((origin[0]+end[0])/2, (origin[1]+end[1])/2-.12)
        tex.append(r'\node[anchor='+anchor+'] at '+point(mid)+' {$'+label+'$};')
    base = project(0,4,0)
    tex.append(r'\draw[->,line width=.7pt] '+point(base)+' -- (0.15,4.40);')
    tex.append(r'\node[anchor=west] at (0.23,4.35) {$J(h)$};')
    # A schematic highlight follows the surface; it is not a feasibility boundary.
    outline = []
    for k in range(101):
        a = 2*math.pi*k/100
        u, v = 1.80+1.03*math.cos(a), 2.08+.81*math.sin(a)
        outline.append(project(u,v,cost(u,v)+.025))
    tex.append(r'\draw[teal,line width=.95pt,dash pattern=on 3pt off 2.2pt] '
               +' -- '.join(map(point,outline))+';')
    # An actual descent path on the illustrative analytic surface.
    u, v = 1.25, 2.55
    trajectory = [project(u,v,cost(u,v)+.04)]
    values = [cost(u,v)]
    for _ in range(9):
        e = 1e-5
        du = (cost(u+e,v)-cost(u-e,v))/(2*e)
        dv = (cost(u,v+e)-cost(u,v-e))/(2*e)
        u, v = u-.085*du, v-.085*dv
        values.append(cost(u,v))
        trajectory.append(project(u,v,cost(u,v)+.04))
    assert all(b < a for a,b in zip(values,values[1:])), values
    start, end = trajectory[0], trajectory[-1]
    # Magnification leaders originate at the same selected representation.
    tex.append(r'\draw[rule,line width=.55pt] '+point(start)+' -- (7.70,4.18);')
    tex.append(r'\draw[rule,line width=.55pt] '+point(start)+' -- (7.70,1.05);')
    tex.append(r'\draw[gold,->,line width=1.5pt] '+' -- '.join(map(point,trajectory))+';')
    tex.append(r'\filldraw[fill=teal,draw=black,line width=.7pt] '+point(start)+' circle (2pt);')
    tex.append(r'\node[anchor=west,align=left,fill=white,fill opacity=.88,text opacity=1,inner sep=2pt] at '
               +point((5.0,2.68))+r' {NCC search\\$-\nabla_h J(h)$};')
    tex.append(r'''
\node[align=center] at (3.55,4.72) {Conditional target set $\mathcal Z_g$\\(schematic view)};
\draw[teal,->,line width=.8pt] (3.55,4.34) -- (3.545,2.901);
% Fixed goal and optimized non-goal components; no right-pointing update arrow.
\draw[rule,line width=.75pt,rounded corners=3pt,fill=white] (7.70,1.05) rectangle (13.05,4.18);
\node[font=\fontsize{10}{12}\selectfont\bfseries] at (10.375,3.85) {Target latent};
\node at (9.05,3.35) {Goal encoding};
\node at (11.70,3.35) {Non-goal $h$};
\node at (9.05,3.00) {Fixed};
\node at (11.70,3.00) {Gradient update};
\draw[teal,line width=.8pt,decorate,decoration={brace,amplitude=3pt}]
 (7.95,2.70) -- (10.22,2.70);
\draw[gold,line width=.8pt,decorate,decoration={brace,amplitude=3pt}]
 (10.40,2.70) -- (12.80,2.70);
''')
    for x0, x1, color, bg in [(7.95,10.30,'teal','tealbg'), (10.30,12.80,'gold','goldbg')]:
        tex.append(r'\draw[fill='+bg+',draw='+color+r',line width=.8pt] '
                   +point((x0,2.15))+' rectangle '+point((x1,2.62))+';')
        for frac in [.25,.75]:
            x = x0+(x1-x0)*frac
            tex.append(r'\draw['+color+',line width=.6pt] '+point((x,2.15))+' -- '+point((x,2.62))+';')
        tex.append(r'\node at '+point(((x0+x1)/2,2.385))+r' {$\cdots$};')
    tex.append(r'''
\node at (10.375,1.78) {$z=[e_{g,\theta}(g);h]$};
\node at (10.375,1.32) {$h\leftarrow h-\eta\nabla_h J(h)$};
\end{tikzpicture}
\end{document}
''')
    source = OUT / 'ncc_search_vector.tex'
    source.write_text('\n'.join(tex))
    subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error',source.name],
                   cwd=OUT, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (OUT.parent/'ncc_search_vector.pdf').write_bytes(source.with_suffix('.pdf').read_bytes())
    print('Generated', OUT.parent/'ncc_search_vector.pdf')
    print('Illustrative descent:', values[0], '->', values[-1])


if __name__ == '__main__':
    main()

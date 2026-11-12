"""Combine Panda Figure 2 with a vector NCC search landscape.

Uses the model-only vector PDFs in panda_vector_preview.  Initial and
selected poses are illustrative target completions, not physical rollouts.
"""
from pathlib import Path
import math
import subprocess
from draw_ncc_search_vector import cost, project

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'iclr2027/figures/panda_vector_preview'


def point(p):
    return f'({p[0]:.5f},{p[1]:.5f})'


def landscape_point(u,v,z):
    x,y=project(u,v,z)
    return .40+.78*x,1.00+.78*y


def main():
    tex=[r'''\documentclass[tikz,border=2pt]{standalone}
\usepackage{times,amsmath,amssymb}
\usetikzlibrary{arrows.meta,shapes.geometric,decorations.pathreplacing}
\pdfinfo{/Title (Panda goal completion and NCC: vector schematic) /Author (Anonymous)}
\definecolor{teal}{HTML}{197D73}
\definecolor{goalblue}{HTML}{B5CCE9}
\definecolor{unknown}{HTML}{E7E8EA}
\definecolor{gold}{HTML}{F5D774}
\definecolor{goldline}{HTML}{B17B16}
\definecolor{grayline}{HTML}{9A9FA4}
\begin{document}
\begin{tikzpicture}[x=1cm,y=1cm,line join=round,line cap=round,
 font=\fontsize{9}{10.5}\selectfont,
 >={Stealth[length=2mm,width=1.5mm]}]
\path[use as bounding box] (0,0) rectangle (13.1,11.98);
\draw[black!65,line width=.65pt,rounded corners=7pt] (.04,8.35) rectangle (13.05,11.88);
\draw[black!65,line width=.65pt,rounded corners=7pt] (.04,.32) rectangle (8.06,8.10);
\draw[black!75,line width=.85pt,dash pattern=on 3pt off 2.2pt,rounded corners=9pt]
 (8.30,.32) rectangle (13.05,8.10);
\node[anchor=west,font=\fontsize{10}{12}\selectfont\bfseries] at (.30,11.59) {(a) QRL};
\node[anchor=west,font=\fontsize{10}{12}\selectfont\bfseries] at (.30,7.81) {(b) NQRL with NCC};
\node[anchor=south west,inner sep=0pt] at (.47,8.69)
 {\includegraphics[height=1.90cm]{current.pdf}};
\node[anchor=south west,inner sep=0pt] at (10.64,8.69)
 {\includegraphics[height=1.90cm]{goal4.pdf}};
\node[anchor=west,font=\fontsize{8}{9}\selectfont] at (.52,8.53) {Current state};
\node[font=\fontsize{8}{9}\selectfont] at (11.47,8.53) {One completion};
\draw[grayline,->,line width=1.1pt,dash pattern=on 3pt off 2pt]
 (2.41,9.48) -- (10.44,9.48);
\node[fill=white,inner sep=1pt] at (6.40,9.76) {$d(s',g_{\mathrm{full}})$};
\draw[grayline,->,line width=1.7pt] (3.30,10.96) -- (9.98,10.96);
\node[font=\fontsize{8}{9}\selectfont] at (6.50,11.24) {Zero-fill unspecified coordinates};
\node[anchor=west,font=\fontsize{8}{9}\selectfont] at (.47,11.31) {Partial goal};
\node[anchor=west,font=\fontsize{8}{9}\selectfont] at (10.08,11.31) {Zero-filled goal};
% Initial completion is a replay-target completion with the same XYZ goal.
\node[anchor=west,font=\fontsize{8}{9}\selectfont] at (.40,7.49) {Partial goal};
\begin{scope}[transparency group,opacity=.30]
 \node[anchor=south west,inner sep=0pt] at (.53,4.98)
  {\includegraphics[height=1.83cm]{goal6.pdf}};
\end{scope}
\node[font=\fontsize{8}{9}\selectfont] at (1.49,4.81) {Initial completion $h_0$};
% The original Figure 3 target-latent explanation sits above the surface.
\draw[grayline,fill=white,line width=.65pt,rounded corners=3pt]
 (3.33,5.27) rectangle (7.79,7.43);
\node[font=\fontsize{9}{10.5}\selectfont\bfseries] at (5.56,7.15) {Target latent};
\node[font=\fontsize{8}{9}\selectfont] at (4.47,6.78) {Goal encoding};
\node[font=\fontsize{8}{9}\selectfont] at (6.66,6.78) {Non-goal $h$};
\node[font=\fontsize{8}{9}\selectfont] at (4.47,6.46) {Fixed};
\node[font=\fontsize{8}{9}\selectfont] at (6.66,6.46) {Optimize};
\draw[fill=goalblue!65,draw=teal,line width=.7pt] (3.55,5.95) rectangle (5.56,6.27);
\draw[fill=gold!60,draw=goldline,line width=.7pt] (5.56,5.95) rectangle (7.57,6.27);
\node at (4.55,6.11) {$e_{g,\theta}(g)$};
\node at (6.56,6.11) {$h$};
\node[font=\fontsize{8}{9}\selectfont] at (5.56,5.60) {$h\leftarrow h-\eta\nabla_hJ(h)$};
\draw[teal,->,line width=.8pt] (3.02,7.14) -- (3.30,7.14);
\node[font=\fontsize{9}{10.5}\selectfont\bfseries] at (4.14,4.52)
 {NCC search in latent space};
% Shared base and shared XYZ endpoint for the six candidate poses.
\node[align=center,font=\fontsize{9}{10.5}\selectfont\bfseries] at (10.68,7.69)
 {Conditional goal set};
\node[align=center,font=\fontsize{8}{9}\selectfont] at (10.68,7.22)
 {Same end-effector position\\$(x^*,y^*,z^*)$};
\foreach \pose in {6,5,4,3,2}{
 \begin{scope}[transparency group,opacity=.30]
  \node[anchor=south west,inner sep=0pt] at (8.50,1.70)
   {\includegraphics[height=4.60cm]{goal\pose.pdf}};
 \end{scope}
}
\node[anchor=south west,inner sep=0pt] at (8.50,1.70)
 {\includegraphics[height=4.60cm]{goal1.pdf}};
\node[align=center,font=\fontsize{8}{9}\selectfont] at (10.68,1.14)
 {Solid: selected completion\\Faint: alternative completions};
''']
    def vector(x,y,filled):
        labels=[r'$x^*$',r'$y^*$',r'$z^*$', '$0$' if filled else '?',
                '$0$' if filled else '?',r'$\cdots$']
        for i,label in enumerate(labels):
            a=x+i*.43
            tex.append(r'\draw[fill='+('goalblue' if i<3 else 'unknown')+
                f',line width=.65pt] ({a:.3f},{y:.3f}) rectangle ({a+.43:.3f},{y+.40:.3f});')
            tex.append(f'\\node[inner sep=0pt] at ({a+.215:.3f},{y+.20:.3f}) {{{label}}};')
    vector(.47,10.72,False);vector(10.08,10.72,True);vector(.40,6.92,False)
    # Completely vector landscape, with back-to-front geometry ordering.
    n=32
    for i,j in sorted(((i,j) for i in range(n) for j in range(n)),key=lambda ij:sum(ij),reverse=True):
        u,v,step=4*i/n,4*j/n,4/n
        t=max(0,min(1,cost(u+step/2,v+step/2)/1.9))
        lo,mid,hi=(91,157,178),(185,220,206),(252,220,139)
        a,b,f=(lo,mid,t*2) if t<.5 else (mid,hi,(t-.5)*2)
        rgb=[round(x+(y-x)*f) for x,y in zip(a,b)]
        path=' -- '.join(point(landscape_point(x,y,cost(x,y)))
                        for x,y in [(u,v),(u+step,v),(u+step,v+step),(u,v+step)])
        tex.append(r'\definecolor{cell}{RGB}{'+','.join(map(str,rgb))+'}')
        tex.append(r'\path[fill=cell,draw=teal!32!white,line width=.12pt] '+path+' -- cycle;')
    origin=landscape_point(0,0,0)
    for u,v,label,anchor in [(4.15,0,'h_1','north'),(0,4.15,'h_2','north east')]:
        end=landscape_point(u,v,0)
        tex.append(r'\draw[->,line width=.7pt] '+point(origin)+' -- '+point(end)+';')
        mid=((origin[0]+end[0])/2,(origin[1]+end[1])/2-.10)
        tex.append(r'\node[anchor='+anchor+'] at '+point(mid)+' {$'+label+'$};')
    tex.append(r'\draw[->,line width=.7pt] '+point(landscape_point(0,4,0))+' -- (.517,4.43);')
    tex.append(r'\node[anchor=west] at (.61,4.36) {$J(h)$};')
    # Start/end positions on the schematic surface are paired with arm poses.
    u,v=1.25,2.55
    trajectory=[landscape_point(u,v,cost(u,v)+.04)]
    values=[cost(u,v)]
    for _ in range(9):
        e=1e-5
        du=(cost(u+e,v)-cost(u-e,v))/(2*e)
        dv=(cost(u,v+e)-cost(u,v-e))/(2*e)
        u,v=u-.085*du,v-.085*dv
        trajectory.append(landscape_point(u,v,cost(u,v)+.04));values.append(cost(u,v))
    assert all(b<a for a,b in zip(values,values[1:]))
    start,end=trajectory[0],trajectory[-1]
    # A representation mapping, not an arm-motion trajectory.
    tex.append(r'\draw[grayline,line width=.7pt,dash pattern=on 2pt off 1.5pt] '
               '(1.65,4.57) .. controls (1.53,4.14) and (1.58,3.77) .. '+point(start)+';')
    tex.append(r'\draw[goldline,-{Stealth[length=2.6mm,width=2mm]},shorten >=3pt,line width=1.5pt] '
               +' -- '.join(map(point,trajectory))+';')
    tex.append(r'\filldraw[fill=teal,draw=black,line width=.6pt] '+point(start)+' circle (2.3pt);')
    tex.append(r'\node[anchor=east,fill=white,inner sep=1pt] at '
               +point((start[0]-.13,start[1]+.12))+r' {$h_0$};')
    # The selected endpoint maps to an actual joint on the opaque arm.
    selected_arm=(10.3605,5.1734)
    tex.append(r'\draw[goldline,line width=.9pt,dash pattern=on 2.4pt off 1.8pt] '
               +point(end)+' .. controls (5.00,2.40) and (7.55,3.07) .. '+point(selected_arm)+';')
    tex.append(r'\filldraw[fill=gold,draw=black,line width=.7pt] '+point(end)+' circle (2.5pt);')
    tex.append(r'\node[anchor=north,fill=white,inner sep=1pt] at '
               +point((end[0]-.10,end[1]-.13))+r' {$\widehat h_K$};')
    tex.append(r'\filldraw[fill=gold,draw=black,line width=.7pt] '+point(selected_arm)+' circle (2.5pt);')
    tex.append(r'''
\node[star,star points=5,star point ratio=2.25,draw=black!65,fill=gold,
 minimum size=3.8mm,inner sep=0pt,line width=.4pt] at (10.70,5.50) {};
\node[align=center,fill=gold!65,draw=goldline,line width=.5pt,rounded corners=2pt,
 inner sep=3pt,font=\fontsize{8}{9}\selectfont] at (6.46,2.26)
 {Selected\\completion};
\node[font=\fontsize{8.5}{10}\selectfont] at (4.05,.71)
 {$J(h)=\bar d_\theta\!\left(\hat z^+,[e_{g,\theta}(g);h]\right)$};
\node[anchor=west,font=\fontsize{7.3}{8.5}\selectfont] at (.10,.08)
 {Schematic latent search; Panda poses illustrate target completions.};
\end{tikzpicture}
\end{document}
''')
    source=OUT/'figure2_panda_ncc_vector.tex';source.write_text('\n'.join(tex))
    proc=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error',source.name],
                        cwd=OUT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    if proc.returncode:
        raise RuntimeError(proc.stdout[-6000:])
    (OUT.parent/'figure2_panda_ncc_vector.pdf').write_bytes(source.with_suffix('.pdf').read_bytes())
    print('Generated',OUT.parent/'figure2_panda_ncc_vector.pdf')


if __name__=='__main__':
    main()

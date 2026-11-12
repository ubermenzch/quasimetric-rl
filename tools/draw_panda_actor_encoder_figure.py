"""Wide, native-vector QRL/NQRL review figure; never installs into the paper.

Current-state arrays make the shared/split encoder interfaces explicit.
Existing Panda mesh projections and the analytic NCC surface remain vector paths.
"""
from pathlib import Path
from math import hypot, sqrt
import json
import re
import subprocess
from draw_ncc_search_vector import cost, project

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'iclr2027/figures/panda_vector_preview'


def pt(x, y):
    return f'({x:.4f},{y:.4f})'


def main(*, draft=False):
    t = [r'''\documentclass[tikz,border=3pt]{standalone}
\usepackage{times,amsmath,amssymb}
\usetikzlibrary{arrows.meta,shapes.geometric}
\pdfinfo{/Title (QRL and NQRL: state encoding and target completion) /Author (Anonymous)}
\definecolor{teal}{HTML}{197D73}
\definecolor{goalblue}{HTML}{C0D4EB}
\definecolor{nongold}{HTML}{F8E7B9}
\definecolor{gold}{HTML}{F0CC65}
\definecolor{goldline}{HTML}{AE791A}
\definecolor{grayline}{HTML}{969FA6}
\definecolor{actorbg}{HTML}{FFFFFF}
\definecolor{coral}{HTML}{B75D58}
\begin{document}
\begin{tikzpicture}[x=1cm,y=1cm,line cap=round,line join=round,
 font=\fontsize{12}{13}\selectfont,
 >={Stealth[length=2.4mm,width=1.8mm]},
 block/.style={draw=grayline,line width=1pt,rounded corners=3pt,
  fill=white,align=center,inner sep=2pt},
 flow/.style={->,line width=1.15pt},
 note/.style={font=\fontsize{11}{12}\selectfont,align=center}]
\path[use as bounding box] (.45,.45) rectangle (25.11,12.55);
\draw[grayline,line width=.9pt] (8.25,.55) -- (8.25,12.47);
\node[anchor=west,font=\fontsize{15}{17}\selectfont\bfseries] at (.65,12.25) {(a) QRL};
\node[anchor=west,font=\fontsize{15}{17}\selectfont\bfseries] at (8.77,12.25) {(b) NQRL};
''']
    wires = []

    def node(name,x,y,text,w=None,h=None,extra=''):
        options='block'+(',minimum width='+str(w)+'cm' if w else '')+(',minimum height='+str(h)+'cm' if h else '')+(','+extra if extra else '')
        t.append(r'\node['+options+'] ('+name+') at '+pt(x,y)+' {'+text+'};')

    def text(x,y,s,options=''):
        t.append(r'\node['+options+'] at '+pt(x,y)+' {'+s+'};')

    def path(s,options='flow'):
        t.append(r'\draw['+options+'] '+s+';')

    def flow(points,style='flow',name=''):
        # A bent wire must have at least 5 mm of straight run before its arrowhead.
        if len(points)>2 and '->' not in style and style.startswith('flow'):
            a,b=points[-2:]
            assert hypot(b[0]-a[0],b[1]-a[1]) >= .50, (name,points)
        assert all(abs(a[0]-b[0]) < 1e-7 or abs(a[1]-b[1]) < 1e-7
                   for a,b in zip(points,points[1:])), (name,'diagonal segment')
        wires.append(dict(name=name,points=points,style=style))
        path(' -- '.join(pt(*p) for p in points),style)

    def vector(name,x,y,labels,kind='state',cell=.42):
        for i,label in enumerate(labels):
            fill = ('grayline!18' if kind=='joint' else
                    'goalblue' if kind=='goal' or (kind=='state' and i<3) else 'nongold')
            a=x+i*cell
            t.append(r'\draw[fill='+fill+',line width=.8pt] '+pt(a,y)+' rectangle '+pt(a+cell,y+.44)+';')
            text(a+cell/2,y+.22,label,r'inner sep=0pt,font=\fontsize{11}{12}\selectfont')
        return x+cell*len(labels),y+.22

    def arm(filename,x,y,height,opacity=1):
        if opacity<1:
            t.append(r'\begin{scope}[transparency group,opacity='+str(opacity)+']')
        t.append(r'\node[anchor=south west,inner sep=0pt] at '+pt(x,y)
                 +r' {\includegraphics[height='+str(height)+'cm]{'+filename+r'.pdf}};')
        if opacity<1:
            t.append(r'\end{scope}')

    # s_1,...,s_m are unspecified coordinates, not necessarily joint angles.
    state=[r'$x$',r'$y$',r'$z$',r'$s_1$',r'$\cdots$',r'$s_m$']
    goal=[r'$x^*$',r'$y^*$',r'$z^*$']
    latent=[r'$\cdot$',r'$\cdot$',r'$\cdot$']
    # QRL occupies the narrower left panel: direct actor input plus one encoder.
    text(.95,11.41,r'Task goal $g$','anchor=west')
    vector('qg',.95,10.49,goal+['?','?',r'$\cdots$'])
    vector('qfull',4.88,10.49,goal+[r'$0$',r'$0$',r'$\cdots$'])
    flow([(3.50,10.71),(4.85,10.71)],'flow,grayline',name='q-zero-fill')
    text(4.15,11.18,'Zero-fill','note')
    arm('current',1.52,7.92,2.15)
    text(2.87,7.43,r'Current state $s$')
    vector('qs',1.61,6.53,state)
    node('qa',6.14,6.75,r'\textbf{Actor}\\$\pi$',2.30,1.05,extra='draw=coral,fill=coral!5')
    # Align the zero-filled state directly above its actor input: no bends.
    flow([(6.14,10.46),(6.14,7.35)],name='q-goal-actor')
    flow([(4.27,6.75),(4.92,6.75)],name='q-state-actor')
    flow([(6.14,6.15),(6.14,5.35)],'flow,coral',name='q-action')
    text(6.55,5.53,r'$a$')
    # Before/after overlays share a base; the new pose is translucent.
    arm('qrl_moved',4.7,2.65,2.62,.30)
    arm('current',4.7,2.65,2.62)
    pose_metadata=json.loads((OUT/'poses.json').read_text())
    motion=json.loads((OUT/'qrl_motion.json').read_text())
    bx0,by0,bx1,by1=pose_metadata['projection_bounds']
    arm_scale=160/(by1-by0)*2.62/166
    def motion_point(key):
        x,y=motion[key]
        return (4.7+(x-bx0)*arm_scale+3*2.62/166,
                2.65+(y-by0)*arm_scale+3*2.62/166)
    tcp=motion_point('initial_tcp_projected')
    next_tcp=motion_point('moved_tcp_projected')
    t.append(r'\fill[coral] '+pt(*tcp)+' circle (1.8pt);')
    # A shallow upward bow: both controls sit above the endpoint chord.
    dx,dy=next_tcp[0]-tcp[0],next_tcp[1]-tcp[1]
    bow=.34
    control1=(tcp[0]+dx/3,tcp[1]+dy/3+bow)
    control2=(tcp[0]+2*dx/3,tcp[1]+2*dy/3+bow)
    path(pt(*tcp)+' .. controls '+pt(*control1)+' and '
         +pt(*control2)+' .. '+pt(*next_tcp),
         'coral,->,shorten <=2pt,shorten >=2pt,line width=1.5pt')
    node('qe',2.87,4.68,r'State encoder\\$E_\theta$',3.37,1.05,extra='fill=goalblue!20')
    flow([(2.87,6.39),(2.87,5.28)],name='q-state-encoder')
    vector('qz',1.61,2.97,latent+latent,kind='joint')
    flow([(2.87,4.08),(2.87,3.55)],name='q-encoded-state')
    text(2.87,2.62,r'$z_s$')
    text(1.10,1.90,r'\textcolor{goalblue}{\rule{.34cm}{.18cm}} Goal coordinates','anchor=west,note')
    text(1.10,1.30,r'\textcolor{nongold}{\rule{.34cm}{.18cm}} Non-goal coordinates','anchor=west,note')
    # Compact layout uses translated columns, without scaling shapes or text.
    # GNG has its own module boundary, with openings at outgoing v and z_s wires.
    path('(19.85,5.88) -- (8.82,5.88) -- (8.82,11.84) -- (19.85,11.84)',
         'grayline,line width=.8pt,dash pattern=on 3pt off 2pt,rounded corners=5pt')
    for ya,yb in [(5.88,7.31),(7.59,10.73),(11.01,11.84)]:
        path(pt(19.85,ya)+' -- '+pt(19.85,yb),
             'grayline,line width=.8pt,dash pattern=on 3pt off 2pt')
    text(15.36,11.84,r'\textbf{GNG}','fill=white,inner sep=3pt')
    # NQRL: the desired goal and current goal coordinates share the goal encoder.
    text(9.14,11.41,r'Task goal $g$','anchor=west')
    vector('ng',9.14,10.65,goal,kind='goal')
    node('ngg',15.3,9.56,r'Goal encoder\\$e_{g,\theta}$',3.13,3.64,extra='fill=goalblue!30')
    flow([(10.54,10.87),(13.65,10.87)],'flow,teal',name='n-task-goal-encoder')
    vector('nv',17.7,10.65,latent,kind='goal')
    flow([(16.95,10.87),(17.56,10.87)],'flow,teal')
    text(18.33,11.40,r'$v$')
    node('na',22.46,10.80,r'\textbf{Actor}\\$\pi_\psi$',2.40,1.08,extra='draw=coral,fill=coral!5')
    flow([(19.1,10.87),(21.19,10.87)],'flow,teal',name='n-goal-actor')
    arm('current',8.85,6.42,2.78)
    text(11.0,8.70,r'Current state $s$','anchor=west')
    vector('ns',11.1,7.45,state)
    node('nen',15.3,6.65,r'Non-goal encoder\\$e_{n,\theta}$',3.13,1.02,extra='fill=nongold!50')
    flow([(11.73,8.03),(11.73,8.25),(13.65,8.25)],'flow,teal',name='n-goal-split')
    flow([(12.99,7.31),(12.99,6.65),(13.65,6.65)],'flow,goldline',name='n-nongoal-split')
    vector('nzg',17.7,8.03,latent,kind='goal')
    vector('nzn',17.7,6.43,latent,kind='nongoal')
    flow([(16.95,8.25),(17.56,8.25)],'flow,teal')
    flow([(16.95,6.65),(17.56,6.65)],'flow,goldline')
    text(18.33,8.86,r'$z_g(s)$')
    text(18.33,6.12,r'$z_n(s)$')
    flow([(19.1,8.25),(19.55,8.25),(19.55,7.45)],'teal,line width=1.15pt')
    flow([(19.1,6.65),(19.55,6.65),(19.55,7.45)],'goldline,line width=1.15pt')
    t.append(r'\fill[grayline] (19.55,7.45) circle (1.6pt);')
    text(18.76,7.42,r'$z_s$')
    flow([(19.55,7.45),(20.22,7.45),(20.22,10.56),(21.19,10.56)],name='n-state-actor')
    node('nf',22.46,8.92,'Dynamics',2.35,1.05,extra='fill=goalblue!18')
    flow([(22.46,10.19),(22.46,9.52)],'flow,coral',name='n-action')
    text(22.93,9.88,r'$a$')
    flow([(20.22,8.92),(21.22,8.92)],name='n-state-dynamics')
    vector('npred',21.2,7.08,latent+latent)
    flow([(22.46,8.32),(22.46,7.66)],name='n-prediction')
    text(21.82,6.72,r'$\hat z^+$')
    # NCC source is a NON-GOAL latent vector, not a robot or a full state.
    path('(8.82,5.60) -- (8.82,.73) -- (24.6,.73) -- (24.6,4.81)',
         'grayline,line width=.8pt,dash pattern=on 3pt off 2pt,rounded corners=5pt')
    path('(24.6,5.13) -- (24.6,5.60) -- (22.63,5.60)',
         'grayline,line width=.8pt,dash pattern=on 3pt off 2pt,rounded corners=5pt')
    path('(22.29,5.60) -- (8.82,5.60)',
         'grayline,line width=.8pt,dash pattern=on 3pt off 2pt')
    text(15.36,5.60,r'\textbf{NCC} (training only)','fill=actorbg,inner sep=3pt')
    node('ncc-loss',21.56,4.97,r'\textbf{Loss}\\$J(h)=\bar d_\theta(\hat z^+,[v;h])$',
         5.25,1.05,extra='fill=goalblue!14')
    flow([(22.46,6.95),(22.46,5.56)],name='prediction-ncc-loss')
    flow([(24.26,4.97),(24.79,4.97),(24.79,10.80),(23.73,10.80)],
         'flow,coral,dash pattern=on 2.5pt off 1.8pt',name='ncc-loss-actor-update')
    text(24.06,6.12,'update','note')
    # The full initial completion y is distinct from its encoded non-goal h0.
    text(10.71,5.10,r'Initial target $y$','note')
    arm('goal6',9.15,1.05,3.02,.30)
    vector('initial-y',9.47,4.32,goal+[r'$s_1$',r'$\cdots$',r'$s_m$'])
    vector('ncc-h0',11.34,1.60,latent,kind='nongoal')
    path('(10.73,4.24) -- (10.73,4.16) -- (11.99,4.16) -- (11.99,4.24)',
         'goldline,line width=.9pt')
    flow([(11.36,4.04),(11.36,3.82),(11.97,3.82),(11.97,2.18)],
         'flow,goldline',name='initial-target-nongoal-encoding')
    text(11.31,3.40,r'$e_{n,\theta}$','note')
    text(11.97,1.13,r'$h_0=z_n(y)$','note')
    text(21.56,4.15,'Conditional goal set',r'font=\bfseries')
    # Three alternatives with separated projected elbows: right, left, and low.
    for i in [5,3,2]:
        arm('goal'+str(i),19.91,1.04,3.00,.30)
    arm('goal1',19.91,1.04,3.00)
    # A taller analytic landscape makes the drop in J visually explicit.
    def landscape(u,v):
        return 1.7*cost(u,v)
    def xy(u,v,z):
        x,_=project(u,v,z)
        return 12.45+.82*x,.95+.65*(.55+.22*u+.26*v+1.25*z)
    n=40
    for i,j in sorted(((i,j) for i in range(n) for j in range(n)),key=lambda ij:sum(ij),reverse=True):
        u,v,step=4*i/n,4*j/n,4/n
        f=max(0,min(1,landscape(u+step/2,v+step/2)/3.0))
        lo,mid,hi=(72,143,173),(173,211,187),(252,204,97)
        a,b,w=(lo,mid,f*2) if f<.5 else (mid,hi,(f-.5)*2)
        rgb=[round(x+(y-x)*w) for x,y in zip(a,b)]
        vertices=[xy(x,y,landscape(x,y)) for x,y in [(u,v),(u+step,v),(u+step,v+step),(u,v+step)]]
        t.append(r'\definecolor{cell}{RGB}{'+','.join(map(str,rgb))+'}')
        t.append(r'\path[fill=cell,draw=teal!30!white,line width=.10pt] '
                 +' -- '.join(pt(*p) for p in vertices)+' -- cycle;')
    # Actual descent on the schematic surface: both J and screen height decrease.
    u,v=1.05,2.72
    trajectory=[xy(u,v,landscape(u,v)+.035)]
    values=[landscape(u,v)]
    for _ in range(12):
        eps=1e-5
        du=(landscape(u+eps,v)-landscape(u-eps,v))/(2*eps)
        dv=(landscape(u,v+eps)-landscape(u,v-eps))/(2*eps)
        u,v=u-.065*du,v-.065*dv
        values.append(landscape(u,v))
        trajectory.append(xy(u,v,landscape(u,v)+.035))
    assert all(b<a for a,b in zip(values,values[1:]))
    assert all(b[1]<a[1] for a,b in zip(trajectory,trajectory[1:]))
    start,end=trajectory[0],trajectory[-1]
    assert start[1]-end[1]>1.3
    flow([(12.74,1.82),(start[0],1.82),start],
         'grayline,line width=.8pt,dash pattern=on 2pt off 1.5pt',name='h0-search-point')
    path('plot[smooth,tension=.65] coordinates {'+' '.join(pt(*p) for p in trajectory)+'}',
         'goldline,->,shorten <=3pt,shorten >=3pt,line width=1.65pt')
    (OUT/'ncc_descent_geometry.json').write_text(json.dumps(dict(
        points=trajectory,costs=values,note='Gradient descent on an analytic schematic, not a trained critic.'),indent=2)+'\n')
    # Paired ellipses are illustrative magnification outlines, not constraints.
    small=(end[0],end[1],.60,.38)
    large=(21.56,2.42,2.48,1.55)
    for sign in [1,-1]:
        a=(small[0]+small[2]/sqrt(2),small[1]+sign*small[3]/sqrt(2))
        b=(large[0]-large[2]/sqrt(2),large[1]+sign*large[3]/sqrt(2))
        path(pt(*a)+' -- '+pt(*b),'grayline,line width=.65pt')
    for cx,cy,rx,ry in [small,large]:
        path(pt(cx,cy)+f' ellipse[x radius={rx}cm,y radius={ry}cm]',
             'teal,line width=.9pt,dash pattern=on 2.6pt off 2pt')
    # The loss determines the cost being reduced during the search over h.
    flow([(18.86,4.97),(small[0],4.97),(small[0],small[1]+small[3]+.10)],
         'flow,goldline',name='loss-guides-search')
    text(17.65,5.24,r'Fix $v$; search $h$.','note')
    # Landmark is the selected arm's elbow, using the common icon transform.
    metadata=json.loads((OUT/'poses.json').read_text())
    x0,y0,x1,y1=metadata['projection_bounds']
    ex,ey=metadata['landmarks']['goal1']['elbow']
    scale=160/(y1-y0)*3.00/166
    selected=(19.91+(ex-x0)*scale+3*3.00/166,
              1.04+(ey-y0)*scale+3*3.00/166)
    for p,c in [(start,'teal'),(end,'gold'),(selected,'gold')]:
        t.append(r'\filldraw[fill='+c+',draw=black,line width=.65pt] '+pt(*p)+' circle (2.4pt);')
    text(start[0]-.16,start[1]+.10,r'$h_0$','anchor=east,fill=white,inner sep=1pt')
    text(end[0]-.03,end[1]-.20,r'$\widehat h_K$','anchor=north,fill=white,inner sep=1pt')
    text(selected[0]+.32,selected[1]+.27,'',
         'star,star points=5,star point ratio=2.25,draw=black!65,fill=gold,minimum size=3.7mm,inner sep=0pt')
    t.extend([r'\end{tikzpicture}',r'\end{document}'])
    source=OUT/'figure_actor_encoder_panda_vector.tex'
    source.write_text('\n'.join(t)+'\n')
    (OUT/'actor_encoder_geometry.json').write_text(json.dumps(wires,indent=2)+'\n')
    result=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error',source.name],cwd=OUT,
                          stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    if result.returncode:
        raise RuntimeError(result.stdout[-6000:])
    log=source.with_suffix('.log').read_text()
    assert not re.search(r'Overfull|Missing character:|undefined|^!',log,re.M)
    (OUT.parent/'figure_actor_encoder_panda_vector.pdf').write_bytes(source.with_suffix('.pdf').read_bytes())
    # User previews are immutable numbered files; never overwrite older PNGs.
    version_numbers = []
    for preview in OUT.parent.glob('figure_actor_encoder_panda_vector_v*.png'):
        match = re.fullmatch(r'figure_actor_encoder_panda_vector_v(\d+)\.png', preview.name)
        if match:
            version_numbers.append(int(match.group(1)))
    version = max(version_numbers, default=0) + 1
    preview_stem = (Path('/tmp/panda_actor_encoder_draft') if draft else
                    OUT.parent / f'figure_actor_encoder_panda_vector_v{version:02d}')
    if not draft:
        assert not preview_stem.with_suffix('.png').exists()
    subprocess.run(['pdftoppm','-scale-to','2880','-singlefile','-png',
                    str(OUT.parent/'figure_actor_encoder_panda_vector.pdf'),
                    str(preview_stem)],check=True)
    print('Generated new preview',preview_stem.with_suffix('.png'))



if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--draft',action='store_true',help='Render only a temporary PNG for review.')
    main(draft=parser.parse_args().draft)

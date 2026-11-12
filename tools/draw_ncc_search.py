"""Draw the NCC schematic as editable SVG and render it with librsvg/Cairo.

Run with system Python: python3 tools/draw_ncc_search.py
Only the renderer needs PyGObject, librsvg and Pycairo; geometry uses stdlib.
The analytic landscape is illustrative, not a fitted critic or experiment.
"""
from pathlib import Path
import math
import html

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'iclr2027' / 'figures'
INK, TEAL, GOLD = '#283039', '#338779', '#b17b22'


def cost(x, y):
    return (1.5 + 1.6 * math.exp(-((x+1.4)**2+(y+.9)**2)/.55)
            + 1.3 * math.exp(-((x-1.3)**2+(y+1.1)**2)/.65)
            + .9 * math.exp(-((x+1.1)**2+(y-1.5)**2)/.5)
            - 1.15 * math.exp(-((x-.9)**2+(y-.8)**2)/.8)
            + .18 * math.sin(2*x+y))


def grad(x, y):
    e = 1e-5
    return ((cost(x+e,y)-cost(x-e,y))/(2*e),
            (cost(x,y+e)-cost(x,y-e))/(2*e))


def project(x, y, z):
    return (310+72*x-46*y, 307+21*x+25*y-55*z)


def main():
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="460" viewBox="0 0 1080 460">',
             '<rect width="1080" height="460" fill="white"/>',
             '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L6,3 L0,6 Z" fill="'+GOLD+'"/></marker></defs>']

    def text(x,y,s,size=19,color=INK,anchor='start',italic=False):
        content = html.escape(s)
        for source, sub in [('Z_g','g'),('e_g,θ','g,θ')]:
            content = content.replace(source, source[0] + f'<tspan baseline-shift="sub" font-size="70%">{sub}</tspan>')
        parts.append(f'<text x="{x}" y="{y}" fill="{color}" font-family="Liberation Serif,serif" font-size="{size}" text-anchor="{anchor}" font-style="{"italic" if italic else "normal"}">{content}</text>')

    def path(points, color, width=1, extra=''):
        d = 'M'+' L'.join(f'{x:.3f},{y:.3f}' for x,y in points)
        parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" {extra}/>')

    # Back-to-front projected mesh, fully vector in both SVG and PDF.
    n = 65
    cells = [(i,j) for i in range(n) for j in range(n)]
    cells.sort(key=lambda ij: sum(ij))
    for i,j in cells:
        x,y = -2.4+i*4.8/n, -2.4+j*4.8/n
        step=4.8/n
        vertices=[(x,y),(x+step,y),(x+step,y+step),(x,y+step)]
        z=cost(x+step/2,y+step/2)
        t=max(0,min(1,(z-.3)/2.8))
        low,mid,high=(102,158,177),(196,222,208),(239,215,149)
        a,b,f=(low,mid,t*2) if t<.5 else (mid,high,(t-.5)*2)
        rgb=tuple(round(v+(w-v)*f) for v,w in zip(a,b))
        coords=' '.join(f'{u:.2f},{v:.2f}' for u,v in [project(xx,yy,cost(xx,yy)) for xx,yy in vertices])
        parts.append(f'<polygon points="{coords}" fill="rgb{rgb}" stroke="#63938d" stroke-opacity=".32" stroke-width=".35"/>')

    # Schematic circular highlight: a projected 2D circle, not a constraint.
    parts.append('<ellipse cx="335" cy="258" rx="126" ry="78" transform="rotate(6 335 258)" fill="none" stroke="'+TEAL+'" stroke-width="2.6" stroke-dasharray="9 6"/>')
    text(300,44,'Conditional target set  Z_g',23,TEAL,'middle')
    text(300,67,'Schematic 2D view',16,'#65707a','middle')
    path([(321,77),(335,179)],TEAL,1.3)
    text(32,108,'Cost J(h)',20,INK,italic=True)
    text(156,382,'h₂',21,INK,italic=True)
    text(461,375,'h₁',21,INK,italic=True)
    path([(65,314),(312,386),(558,313)],'#919b9f',1)

    x,y=.7,-.2
    start=project(x,y,cost(x,y))
    trajectory=[start]
    costs=[cost(x,y)]
    for _ in range(15):
        gx,gy=grad(x,y)
        x,y=x-.06*gx,y-.06*gy
        costs.append(cost(x,y))
        trajectory.append(project(x,y,cost(x,y)+.025))
    assert all(b<a for a,b in zip(costs,costs[1:])), costs
    # Two magnification lines originate immediately beside the candidate.
    path([(start[0]+8,start[1]-4),(655,105)],'#90979b',1.2)
    path([(start[0]+8,start[1]+4),(655,333)],'#90979b',1.2)
    path(trajectory,GOLD,3.3,'marker-end="url(#arrow)"')
    parts.append(f'<circle cx="{start[0]}" cy="{start[1]}" r="6" fill="{INK}" stroke="white" stroke-width="1.6"/>')
    text(start[0]-12,start[1]-11,'zₖ',20,INK,'end',True)
    end=trajectory[-1]
    text(479,278,'NCC search',21,GOLD)
    text(479,301,'−∇ₕ J(hₖ)',21,GOLD,italic=True)

    parts.append('<rect x="655" y="105" width="404" height="228" rx="9" fill="white" stroke="#b8bec1" stroke-width="1.2"/>')
    text(857,138,'Target latent  zₖ = [e_g,θ(g); hₖ]',22,INK,'middle')
    text(760,177,'Goal encoding',19,TEAL,'middle')
    text(944,177,'Non-goal latent',19,GOLD,'middle')
    for i in range(10):
        color,edge=('#d9eeea',TEAL) if i<4 else ('#faedca',GOLD)
        parts.append(f'<rect x="{677+36*i}" y="194" width="36" height="44" fill="{color}" stroke="{edge}" stroke-width="1.1"/>')
    text(749,266,'Fixed',19,TEAL,'middle')
    text(929,266,'Gradient update',19,GOLD,'middle')
    path([(844,285),(1015,285)],GOLD,2,'marker-end="url(#arrow)"')
    text(857,314,'hₖ₊₁ = hₖ − η ∇ₕ J(hₖ)',21,INK,'middle',True)
    text(540,418,'J(h) = d̄θ(ẑ⁺, [e_g,θ(g); h])',23,INK,'middle',True)
    text(540,446,'Goal encoding, predicted next latent, and network parameters stay fixed during NCC.',17,'#65707a','middle')
    parts.append('</svg>')
    OUT.mkdir(parents=True,exist_ok=True)
    svg=OUT/'ncc_search.svg'
    svg.write_text('\n'.join(parts),encoding='utf-8')
    import gi
    gi.require_version('Rsvg','2.0')
    from gi.repository import Rsvg
    import cairo
    handle=Rsvg.Handle.new_from_file(str(svg))
    rect=Rsvg.Rectangle()
    rect.x,rect.y,rect.width,rect.height=0,0,1080,460
    pdf=cairo.PDFSurface(str(OUT/'ncc_search.pdf'),540,230)
    ctx=cairo.Context(pdf)
    ctx.scale(.5,.5)
    handle.render_document(ctx,rect)
    pdf.finish()
    png=cairo.ImageSurface(cairo.FORMAT_ARGB32,2160,920)
    ctx=cairo.Context(png)
    ctx.scale(2,2)
    handle.render_document(ctx,rect)
    png.write_to_png(str(OUT/'ncc_search.png'))
    print(f'Saved SVG, PDF, PNG to {OUT}; illustrated descent: {costs[0]:.4f} -> {costs[-1]:.4f}')


if __name__ == '__main__':
    main()

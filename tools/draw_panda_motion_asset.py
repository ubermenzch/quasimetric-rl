"""Create one illustrative moved Panda pose without changing existing pose assets.

.venv/bin/python tools/draw_panda_motion_asset.py
python3 tools/draw_panda_motion_asset.py --pdf
The before/after pair is an illustration, not a trained-policy rollout.
"""
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'iclr2027/figures/panda_vector_preview'


def project_pose():
    import numpy as np
    import pybullet as p
    import pybullet_data
    from project_panda_vector_meshes import mesh
    p.connect(p.DIRECT)
    robot=p.loadURDF(str(Path(pybullet_data.getDataPath())/'franka_panda/panda.urdf'),useFixedBase=True)
    joints=[.90,.20,-.50,-2.00,.20,1.80,.70]
    for j,q in enumerate(joints): p.resetJointState(robot,j,q)
    for j in [9,10]: p.resetJointState(robot,j,.025)
    view=np.array([1.6,-2.7,1.25]); view/=np.linalg.norm(view)
    right=np.cross([0,0,1],view); right/=np.linalg.norm(right); up=np.cross(view,right)
    light=np.array([-.3,-.7,1.5]); light/=np.linalg.norm(light)
    triangles,colors,depths=[],[],[]
    visuals=p.getVisualShapeData(robot)
    loaded={v[4]:mesh(v[4].decode()) for v in visuals}
    for visual in visuals:
        link=visual[1]
        if link==-1: pos,orn=(0,0,0),(0,0,0,1)
        else:
            state=p.getLinkState(robot,link,computeForwardKinematics=True)
            pos,orn=state[4],state[5]
        pos,orn=p.multiplyTransforms(pos,orn,visual[5],visual[6])
        rot=np.asarray(p.getMatrixFromQuaternion(orn)).reshape(3,3)
        verts,basecolor=loaded[visual[4]]
        world=(verts*np.array(visual[3]))@rot.T+np.asarray(pos)
        normals=np.cross(world[:,1]-world[:,0],world[:,2]-world[:,0])
        norms=np.linalg.norm(normals,axis=1)
        normals/=np.maximum(norms[:,None],1e-10)
        visible=(normals@view>.001)&(norms>1e-10)
        world,normals,basecolor=world[visible],normals[visible],basecolor[visible]
        rgb=np.clip(basecolor*(.56+.44*np.maximum(0,normals@light))[:,None],.025,.97)
        xy=np.stack((world@right,world@up),axis=-1)
        triangles.append(xy);colors.append(rgb);depths.append(world.mean(axis=1)@view)
    xy,rgb,z=np.concatenate(triangles),np.concatenate(colors),np.concatenate(depths)
    order=np.argsort(z)
    bounds=json.loads((OUT/'poses.json').read_text())['projection_bounds']
    coords=xy.reshape(-1,2)
    assert np.all(coords.min(axis=0)>=np.array(bounds[:2])-.01)
    assert coords[:,1].max()<=bounds[3]+.01
    # Extend the right margin if needed; retain the shared origin and vertical scale.
    assert coords[:,0].max()<.80
    tcp=np.array(p.getLinkState(robot,11,computeForwardKinematics=True)[4])
    np.savez_compressed(OUT/'qrl_moved.npz',triangles=xy[order],colors=rgb[order])
    (OUT/'qrl_motion.json').write_text(json.dumps(dict(
        initial_joints=[.20,-.65,.10,-2.10,.10,1.50,.70],moved_joints=joints,
        initial_tcp_projected=[.3422985698873776,.4949339386216763],
        moved_tcp_projected=[float(tcp@right),float(tcp@up)],
        note='Illustrative before/after pose; same base and camera, not a policy rollout.'),indent=2)+'\n')
    p.disconnect()
    print('Created qrl_moved mesh and motion metadata.')


def render_pdf():
    import numpy as np
    from draw_panda_goal_figure import VectorCanvas
    x0,y0,x1,y1=json.loads((OUT/'poses.json').read_text())['projection_bounds']
    scale=160/(y1-y0)
    data=np.load(OUT/'qrl_moved.npz')
    x1=max(x1,float(data['triangles'][...,0].max()))
    c=VectorCanvas(str(OUT/'qrl_moved.pdf'),pagesize=((x1-x0)*scale+6,166),pageCompression=1)
    c.setTitle('Panda moved pose: illustrative action');c.setAuthor('Anonymous')
    for triangle,rgb in zip((data['triangles']-[x0,y0])*scale+3,data['colors']):
        c.setFillColorRGB(*rgb);c.setStrokeColorRGB(*rgb);c.setLineWidth(.035)
        path=c.beginPath();path.moveTo(*triangle[0]);path.lineTo(*triangle[1]);path.lineTo(*triangle[2]);path.close()
        c.drawPath(path,stroke=1,fill=1)
    c.save()
    print('Created vector qrl_moved.pdf.')


if __name__=='__main__':
    render_pdf() if '--pdf' in sys.argv else project_pose()

"""Project the actual PyBullet Panda visual meshes into shaded vector triangles.

Run with .venv/bin/python; the output is geometry, never a screenshot.
Poses illustrate a position-only end-effector goal, not learned behavior.
"""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image
import pybullet as p
import pybullet_data

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'iclr2027/figures/panda_vector_preview'


def mesh(filename):
    path = Path(filename)
    verts, uv, faces, tex = [], [], [], []
    for line in path.read_text().splitlines():
        bits = line.split()
        if not bits:
            continue
        if bits[0] == 'v':
            verts.append(list(map(float, bits[1:4])))
        elif bits[0] == 'vt':
            uv.append(list(map(float, bits[1:3])))
        elif bits[0] == 'f':
            items = [s.split('/') for s in bits[1:]]
            for i in range(1, len(items)-1):
                tri = [items[0], items[i], items[i+1]]
                faces.append([int(s[0])-1 for s in tri])
                tex.append([int(s[1])-1 if len(s)>1 and s[1] else 0 for s in tri])
    vertices = np.asarray(verts)[np.asarray(faces)]
    texture = path.parent/'colors.png'
    if not texture.exists():
        texture = path.parent.parent/'visual/colors.png'
    colors = np.full((len(faces),3),.83)
    if uv and texture.exists():
        arr = np.asarray(Image.open(texture).convert('RGB'))/255.
        coords = np.asarray(uv)[np.asarray(tex)].mean(axis=1)
        x = np.clip((coords[:,0]*(arr.shape[1]-1)).astype(int),0,arr.shape[1]-1)
        y = np.clip(((1-coords[:,1])*(arr.shape[0]-1)).astype(int),0,arr.shape[0]-1)
        colors = arr[y,x]
    return vertices, colors


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    p.connect(p.DIRECT)
    root = Path(pybullet_data.getDataPath())/'franka_panda'
    robot = p.loadURDF(str(root/'panda.urdf'),useFixedBase=True)
    visuals = p.getVisualShapeData(robot)
    loaded = {v[4]: mesh(v[4].decode()) for v in visuals}
    current = np.array([.20,-.65,.10,-2.10,.1,1.50,.70])
    target = np.array([.42, .0, .37])
    # Orthographic projection, shared across all icons: no table or environment.
    view = np.array([1.6,-2.7,1.25]);view/=np.linalg.norm(view)
    right = np.cross([0,0,1],view);right/=np.linalg.norm(right)
    up = np.cross(view,right)
    poses = [('current',current)]
    records = []
    # Keep the chosen pose fixed; vary other configurations far more strongly.
    selected = np.array([-1.3041328232888618,-.9491128299697216,.9984503356413735,
                         -2.329687287191897,.7554541900645894,1.6521883009130662,
                         1.439838123886289])
    def install(q):
        for j,a in enumerate(q):
            p.resetJointState(robot,j,float(a))
        p.getLinkState(robot,11,computeForwardKinematics=True)
    def pose_feature():
        positions=np.array([p.getLinkState(robot,j,computeForwardKinematics=True)[4]
                            for j in [2,3,4,5,6,8]])
        return np.stack((positions@right,positions@up),axis=-1).ravel()
    install(selected)
    candidates=[(selected,pose_feature())]
    rng=np.random.default_rng(2819)
    lower=np.array([p.getJointInfo(robot,j)[8] for j in range(7)])
    upper=np.array([p.getJointInfo(robot,j)[9] for j in range(7)])
    for _ in range(220):
        initial=rng.uniform([-2.5,-1.35,-2.5,-2.85,-2.5,.15,-2.5],
                            [2.5,1.35,2.5,-.45,2.5,3.35,2.5])
        install(initial)
        # Position-only goals intentionally leave end-effector orientation free.
        for _ in range(4):
            sol=p.calculateInverseKinematics(robot,11,target,maxNumIterations=250,
                                              residualThreshold=1e-7)
            q=np.array(sol[:7]);install(q)
        error=np.linalg.norm(np.array(p.getLinkState(robot,11)[4])-target)
        if error>.001 or np.any(q<lower) or np.any(q>upper):
            continue
        positions=np.array([p.getLinkState(robot,j)[4] for j in [2,3,4,5,6,8]])
        if positions[:,2].min()<.09:
            continue
        # Reject visible non-neighbor self-penetrations in collision geometry.
        links=[-1,0,1,2,3,4,5,6,8]
        collision=False
        for ia,a in enumerate(links):
            for b in links[ia+1:]:
                if abs(a-b)<=2:
                    continue
                hits=p.getClosestPoints(robot,robot,0.,linkIndexA=a,linkIndexB=b)
                if any(h[8]<-.003 for h in hits):
                    collision=True;break
            if collision:break
        if not collision:
            candidates.append((q.copy(),pose_feature()))
    assert len(candidates)>=6,len(candidates)
    # Farthest-point sampling favors distinct projected elbows and wrist poses.
    chosen=[0]
    while len(chosen)<6:
        remaining=[i for i in range(len(candidates)) if i not in chosen]
        pick=max(remaining,key=lambda i:min(np.linalg.norm(candidates[i][1]-candidates[j][1])
                                            for j in chosen))
        chosen.append(pick)
    print('Distinct-pose candidates:',len(candidates),flush=True)
    for k,index in enumerate(chosen):
        q=candidates[index][0];install(q)
        reached=np.array(p.getLinkState(robot,11)[4]);error=float(np.linalg.norm(reached-target))
        poses.append((f'goal{k+1}',q))
        records.append(dict(name=f'goal{k+1}',joints=q.tolist(),end_effector=reached.tolist(),error=error))
    light = np.array([-.3,-.7,1.5]);light/=np.linalg.norm(light)
    all_projected = []
    landmarks = {}
    for name, qs in poses:
        for j,q in enumerate(qs):
            p.resetJointState(robot,j,float(q))
        for j in [9,10]:
            p.resetJointState(robot,j,.025)
        elbow=np.array(p.getLinkState(robot,3,computeForwardKinematics=True)[4])
        landmarks[name]=dict(elbow=[float(elbow@right),float(elbow@up)])
        triangles,colors,depths = [],[],[]
        for visual in visuals:
            link = visual[1]
            if link == -1:
                pos,orn = (0,0,0),(0,0,0,1)
            else:
                state = p.getLinkState(robot,link,computeForwardKinematics=True)
                pos,orn = state[4],state[5]
            pos,orn = p.multiplyTransforms(pos,orn,visual[5],visual[6])
            rot = np.asarray(p.getMatrixFromQuaternion(orn)).reshape(3,3)
            verts,basecolor = loaded[visual[4]]
            world = (verts*np.array(visual[3]))@rot.T + np.asarray(pos)
            normals = np.cross(world[:,1]-world[:,0],world[:,2]-world[:,0])
            norms = np.linalg.norm(normals,axis=1)
            normals /= np.maximum(norms[:,None],1e-10)
            visible = (normals@view > .001) & (norms>1e-10)
            world,normals,basecolor = world[visible],normals[visible],basecolor[visible]
            shade = .56+.44*np.maximum(0,normals@light)
            rgb = np.clip(basecolor*shade[:,None],.025,.97)
            # Keep the white housing neutral and the black joint covers distinct.
            xy = np.stack((world@right,world@up),axis=-1)
            triangles.append(xy);colors.append(rgb);depths.append(world.mean(axis=1)@view)
        xy,rgb,z = np.concatenate(triangles),np.concatenate(colors),np.concatenate(depths)
        order=np.argsort(z)
        xy,rgb=xy[order],rgb[order]
        all_projected.append(xy.reshape(-1,2))
        goalxy = np.array([target@right,target@up])
        np.savez_compressed(OUT/f'{name}.npz',triangles=xy,colors=rgb,goal=goalxy)
        print(name,len(xy),'visible triangles')
    coords=np.concatenate(all_projected)
    bounds=[*coords.min(axis=0).tolist(),*coords.max(axis=0).tolist()]
    (OUT/'poses.json').write_text(json.dumps(dict(
        source=str(root/'panda.urdf'),goal=target.tolist(),projection_bounds=bounds,
        note='Analytic IK poses for a position-only goal; not trained-policy rollouts. '
             'The QRL illustration is not a physical decoding of a zero-filled vector.',
        poses=records,landmarks=landmarks),indent=2)+'\n')
    p.disconnect()


if __name__=='__main__':
    main()

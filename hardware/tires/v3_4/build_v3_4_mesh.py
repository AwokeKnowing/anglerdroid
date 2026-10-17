#!/usr/bin/env python3
"""v3.4 robust mesh build (manifold3d): v3.2 STL + lofted root fillet fills + mirrored outboard lip/tabs."""
import numpy as np, trimesh, manifold3d as mf
import gen_softtire_v3_4 as g4
P = g4.P
def M(tm):
    tm = trimesh.Trimesh(tm.vertices, tm.faces, process=True); tm.merge_vertices(digits_vertex=4); tm.fix_normals(); m = mf.Manifold(mf.Mesh(vert_properties=np.asarray(tm.vertices, np.float32), tri_verts=np.asarray(tm.faces, np.uint32))); print("M", m.status(), tm.is_watertight); return m
def _unused(tm): return mf.Manifold(mf.Mesh(vert_properties=np.asarray(tm.vertices, np.float32), tri_verts=np.asarray(tm.faces, np.uint32)))
def T(m): x = m.to_mesh(); return trimesh.Trimesh(x.vert_properties[:, :3], x.tri_verts)
base = M(trimesh.load('softtire_v3_2_tire_95a.stl'))
fills = []
for s in g4.fillet_fill():
    v, f = s.tessellate(0.02, 0.1); fills.append(M(trimesh.Trimesh([(p.x, p.y, p.z) for p in v], f)))
body = mf.Manifold.batch_boolean([base] + fills, mf.OpType.Add)
slab = mf.Manifold.cube([400, 400, P['zbed'] - P['z0'] + 0.05]).translate([-200, -200, P['z0'] - 0.05])
lip = (body ^ slab).mirror([0, 0, 1]).translate([0, 0, 2 * g4.ZM])
body = body + lip
parts = body.decompose(); print('components', len(parts), 'dropped vol', round(sum(p.volume() for p in parts) - max(p.volume() for p in parts), 3))
body = max(parts, key=lambda p: p.volume()); x = body.to_mesh()
t = trimesh.Trimesh(x.vert_properties[:, :3].astype(np.float64), x.tri_verts, process=False)
t.export('softtire_v3_4_tire_95a.stl'); print('watertight', t.is_watertight, 'bodies', t.body_count, 'vol cm3', round(t.volume / 1000, 2), t.bounds.round(2).tolist())

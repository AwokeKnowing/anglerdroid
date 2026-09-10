
import math
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import warp as wp
wp.init()
from house_mesh import load_indexed, wall_segments_xy, VisualMeshCaster

stl = Path("/home/james/gits/awokeknowing/anglerdroid/sim/casitahouse-walls-fixed.stl")
verts, faces = load_indexed(stl)
print("verts", verts.shape, "z", float(verts[:,2].min()), float(verts[:,2].max()))
segs = wall_segments_xy(verts, faces)
mins, maxs = verts.min(0), verts.max(0)
spawn = np.array([float(mins[0]) + 1.2, float(mins[1]) + 1.2])
print("spawn", spawn, "house", mins[:2], maxs[:2])

a = segs[:, 0]
b = segs[:, 1]
ab = b - a
ab2 = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-18)
ap = spawn - a
tt = np.clip(np.einsum("ij,ij->i", ap, ab) / ab2, 0.0, 1.0)
closest = a + tt[:, None] * ab
d = np.linalg.norm(closest - spawn, axis=1)
j = int(np.argmin(d))
delta = closest[j] - spawn
yaw = math.atan2(float(delta[1]), float(delta[0]))
print("nearest wall", float(d[j]), "yawdeg", math.degrees(yaw), "hit", closest[j])

# place axle so nose (x=+0.15) is 0.5 m from wall along +x
# wall point, heading toward wall
n = delta / (np.linalg.norm(delta) + 1e-12)
# front face at 0.50 from wall: axle = wall_point - n * (0.50 + 0.15)
axle = closest[j] - n * (0.50 + 0.15)
print("axle for 0.5m front", axle, "front_clear along +x", 0.50)

caster = VisualMeshCaster(verts, faces, "cuda:0")

def rotmat_to_quat(R):
    m = np.asarray(R, dtype=np.float64)
    t = float(np.trace(m))
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    q /= np.linalg.norm(q)
    return q

def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])

def quat_rot(q, v):
    x, y, z, w = [float(t) for t in q]
    qv = np.array([x, y, z])
    vv = np.asarray(v, dtype=np.float64)
    uv = np.cross(qv, vv)
    uuv = np.cross(qv, uv)
    return vv + 2.0 * (w * uv + uuv)

# body yaw quat about z
cy, sy = math.cos(yaw), math.sin(yaw)
# identity yaw: chassis +x is world +x if yaw=0. quat from yaw:
qz = np.array([0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5)])
r_rs1 = np.column_stack((
    np.array([-1.0, 0.0, 0.0]),
    np.array([0.0, -1.0, 0.0]),
    np.array([0.0, 0.0, 1.0]),
))
q_rs1 = rotmat_to_quat(r_rs1)
q1 = quat_mul(qz, q_rs1)
q1 /= np.linalg.norm(q1)
origin = np.array([axle[0], axle[1], 0.86])
look = quat_rot(q1, np.array([0.0, 0.0, -1.0]))
print("look", look, "origin", origin)

h, w = 60, 80
vfov = math.radians(58.0)
aspect = w / h
th = math.tan(vfov * 0.5)
py, px = np.mgrid[0:h, 0:w]
u = (px.astype(np.float32) + 0.5) / float(w) - 0.5
v = (py.astype(np.float32) + 0.5) / float(h) - 0.5
dx = (u * (2.0 * th * aspect)).astype(np.float32)
dy = (-v * (2.0 * th)).astype(np.float32)
depth = caster.forward_depth(origin, q1, dx, dy)
hit = np.isfinite(depth) & (depth > 1e-3)
print("depth hit", int(hit.sum()), "/", depth.size, "min", float(depth[hit].min()) if hit.any() else None,
      "p50", float(np.median(depth[hit])) if hit.any() else None,
      "max", float(depth[hit].max()) if hit.any() else None)
short = hit & (depth < 0.78)
print("short<0.78", int(short.sum()), "min short", float(depth[short].min()) if short.any() else None)

# save a simple png
from PIL import Image
img = np.zeros((h, w, 3), dtype=np.uint8)
if hit.any():
    d = np.clip(depth, 0, 1.5)
    # floor ~0.86 beige, wall shorter red
    img[hit] = (210, 190, 150)
    img[short] = (220, 50, 40)
Image.fromarray(img).resize((320, 240), Image.NEAREST).save("/tmp/rs1_diag_depth.png")
print("wrote /tmp/rs1_diag_depth.png")

# also: horizontal +x ray from nose to wall
nose = axle + n * 0.15
print("nose", nose)

# manual mesh ray along +x at body height
@wp.kernel
def one_ray(mesh_id: wp.uint64, o: wp.vec3, d: wp.vec3, out: wp.array(dtype=wp.float32)):
    q = wp.mesh_query_ray(mesh_id, o, d, 4.0)
    if q.result:
        out[0] = q.t
    else:
        out[0] = -1.0

out = wp.zeros(1, dtype=wp.float32, device="cuda:0")
for z in (0.2, 0.5, 0.8):
    origin_r = wp.vec3(float(nose[0]), float(nose[1]), z)
    direction = wp.vec3(float(n[0]), float(n[1]), 0.0)
    wp.launch(one_ray, dim=1, inputs=[caster.mesh.id, origin_r, direction], outputs=[out], device="cuda:0")
    print("plusx ray z", z, "t", float(out.numpy()[0]))

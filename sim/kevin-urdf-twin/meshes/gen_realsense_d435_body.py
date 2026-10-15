#!/usr/bin/env python3
"""Lofted stadium body for RealSense D435/D435i (Kevin twin). ≤80 verts."""
import math, struct
from pathlib import Path

DEPTH = 0.025
HALF_D = DEPTH / 2
ARC_SEGS = 8
STATIONS = [
    (-HALF_D,          0.0774, 0.0192),
    (-HALF_D + 0.0045, 0.0900, 0.0250),
    (+0.0005,          0.0900, 0.0250),
    (+HALF_D,          0.0845, 0.0195),
]

def stadium_ring(W, H, n_arc=ARC_SEGS):
    flat = W / 2.0 - H / 2.0
    r = H / 2.0
    pts = []
    for i in range(n_arc):
        a = math.pi / 2 - math.pi * i / n_arc
        pts.append((flat + r * math.cos(a), r * math.sin(a)))
    for i in range(n_arc):
        a = -math.pi / 2 - math.pi * i / n_arc
        pts.append((-flat + r * math.cos(a), r * math.sin(a)))
    return pts

def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
def sub(a, b):
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])
def nrm(v):
    L = math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]) or 1.0
    return (v[0]/L, v[1]/L, v[2]/L)

def build():
    rings = [[(x, y, z) for (y, z) in stadium_ring(W, H)] for x, W, H in STATIONS]
    N = len(rings[0])
    verts = [p for ring in rings for p in ring]
    i_front_c = len(verts); verts.append((STATIONS[-1][0], 0.0, 0.0))
    i_back_c = len(verts); verts.append((STATIONS[0][0], 0.0, 0.0))
    faces = []
    for ri in range(len(rings) - 1):
        ba, bb = ri * N, (ri + 1) * N
        for i in range(N):
            j = (i + 1) % N
            faces.append((ba + i, bb + i, bb + j))
            faces.append((ba + i, bb + j, ba + j))
    base_f = (len(rings) - 1) * N
    for i in range(N):
        j = (i + 1) % N
        faces.append((i_front_c, base_f + j, base_f + i))
        faces.append((i_back_c, i, j))

    def face_normal(i0, i1, i2):
        v0, v1, v2 = verts[i0], verts[i1], verts[i2]
        return nrm(cross(sub(v1, v0), sub(v2, v0)))

    tip = max(range(N), key=lambda i: rings[1][i][1])
    fn = face_normal(N + tip, 2 * N + tip, 2 * N + ((tip + 1) % N))
    if fn[1] < 0:
        n_loft = (len(rings) - 1) * N * 2
        faces = [((f[0], f[2], f[1]) if k < n_loft else f) for k, f in enumerate(faces)]
    return verts, faces, face_normal

def write_stl(path, verts, faces, face_normal):
    buf = bytearray()
    buf += b'RealSense D435 body'.ljust(80, b'\0')
    buf += struct.pack('<I', len(faces))
    for i0, i1, i2 in faces:
        n = face_normal(i0, i1, i2)
        buf += struct.pack('<3f', *n)
        for idx in (i0, i1, i2):
            buf += struct.pack('<3f', *verts[idx])
        buf += struct.pack('<H', 0)
    Path(path).write_bytes(buf)

if __name__ == '__main__':
    verts, faces, face_normal = build()
    out = Path(__file__).resolve().parent
    write_stl(out / 'realsense_d435_body.stl', verts, faces, face_normal)
    write_stl(out / 'realsense_d435i_body.stl', verts, faces, face_normal)
    print(len(verts), 'verts', len(faces), 'faces')

#!/usr/bin/env python3
"""front_caster_hood.stl — restored v157–v159 geometry.
200° arc + chord, D-profile end walls (no half-disk side covers), short neck stub.
"""
import math, struct
from pathlib import Path

AXLE_X = -0.01700
AXLE_Z = -0.03595
R_OUT = 0.02305
R_IN  = 0.02005
HALF_Y = 0.01375
ARC_DEG = 200.0
# Center arc on +Z so hood is level (parallel to ground) about the axle
ARC_START = math.pi/2 - math.radians(ARC_DEG/2)  # -10°
ARC_SPAN  = math.radians(ARC_DEG)                 # to +190°
N_ARC = 57
NECK_R = 0.00700
NECK_Z_TOP = 0.00500
NECK_Z_BOT = AXLE_Z + R_OUT
N_NECK = 20

def cross(a,b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
def sub(a,b):
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])
def nrm(v):
    L=math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]) or 1.0
    return (v[0]/L,v[1]/L,v[2]/L)

def build():
    verts=[]; faces=[]
    def V(p):
        verts.append(tuple(p)); return len(verts)-1
    def tri(a,b,c): faces.append((a,b,c))
    def quad(a,b,c,d):
        tri(a,b,c); tri(a,c,d)
    def arc_xz(r, i, n=N_ARC):
        a = ARC_START + ARC_SPAN * i / n
        return (AXLE_X + r*math.cos(a), AXLE_Z + r*math.sin(a))
    outer_L=[]; outer_R=[]; inner_L=[]; inner_R=[]
    for i in range(N_ARC+1):
        xo,zo = arc_xz(R_OUT, i)
        xi,zi = arc_xz(R_IN, i)
        outer_L.append(V((xo,-HALF_Y,zo)))
        outer_R.append(V((xo,+HALF_Y,zo)))
        inner_L.append(V((xi,-HALF_Y,zi)))
        inner_R.append(V((xi,+HALF_Y,zi)))
    for i in range(N_ARC):
        quad(outer_L[i], outer_R[i], outer_R[i+1], outer_L[i+1])
    for i in range(N_ARC):
        quad(inner_L[i], inner_L[i+1], inner_R[i+1], inner_R[i])
    for i in range(N_ARC):
        quad(outer_L[i], outer_L[i+1], inner_L[i+1], inner_L[i])
        quad(outer_R[i], inner_R[i], inner_R[i+1], outer_R[i+1])
    quad(outer_L[0], outer_L[N_ARC], outer_R[N_ARC], outer_R[0])
    quad(inner_L[0], inner_R[0], inner_R[N_ARC], inner_L[N_ARC])
    quad(outer_L[0], inner_L[0], inner_L[N_ARC], outer_L[N_ARC])
    quad(outer_R[0], outer_R[N_ARC], inner_R[N_ARC], inner_R[0])
    nb=[]; nt=[]
    for i in range(N_NECK):
        a=2*math.pi*i/N_NECK
        x=NECK_R*math.cos(a); y=NECK_R*math.sin(a)
        nb.append(V((x,y,NECK_Z_BOT)))
        nt.append(V((x,y,NECK_Z_TOP)))
    for i in range(N_NECK):
        j=(i+1)%N_NECK
        quad(nb[i], nb[j], nt[j], nt[i])
    ct=V((0.,0.,NECK_Z_TOP))
    for i in range(N_NECK):
        tri(ct, nt[i], nt[(i+1)%N_NECK])
    cb=V((0.,0.,NECK_Z_BOT))
    for i in range(N_NECK):
        tri(cb, nb[(i+1)%N_NECK], nb[i])
    def face_normal(i0,i1,i2):
        v0,v1,v2=verts[i0],verts[i1],verts[i2]
        return nrm(cross(sub(v1,v0),sub(v2,v0)))
    return verts, faces, face_normal

def write_stl(path, verts, faces, face_normal):
    buf=bytearray()
    buf += b'front_caster_hood v159-restored'.ljust(80,b'\0')
    buf += struct.pack('<I', len(faces))
    for i0,i1,i2 in faces:
        n=face_normal(i0,i1,i2)
        buf += struct.pack('<3f',*n)
        for idx in (i0,i1,i2):
            buf += struct.pack('<3f',*verts[idx])
        buf += struct.pack('<H',0)
    Path(path).write_bytes(buf)

if __name__ == '__main__':
    verts, faces, face_normal = build()
    out = Path(__file__).resolve().parent / 'front_caster_hood.stl'
    write_stl(out, verts, faces, face_normal)
    print(f'wrote {out} tris={len(faces)}')

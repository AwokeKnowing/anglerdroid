"""Synthetic mast frames of the casita model from known poses (Lambert-shaded ray cast, twin lens)."""
import math, numpy as np, trimesh
from mapbuild import load
from lineloc import Camera, se2_cam

class Renderer:
    def __init__(self, stl, ceiling=2.28):
        m = load(stl); V, F = m.vertices, m.faces; lo, hi = V.min(0) - 1, V.max(0) + 1; n = len(V)
        q = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
        V = np.vstack([V, np.c_[q, np.zeros(4)], np.c_[q, np.full(4, ceiling)]])
        F = np.vstack([F, [[n, n + 1, n + 2], [n, n + 2, n + 3]], [[n + 4, n + 6, n + 5], [n + 4, n + 7, n + 6]]])
        self.mesh = trimesh.Trimesh(V, F, process=False); self.N = self.mesh.face_normals; self.nw = len(m.faces)
        self.alb = np.r_[np.full(len(m.faces), 0.75), [0.45, 0.45], [0.92, 0.92]]

    def render(self, cam, pose, W=640, H=480, light=(0.35, 0.55, 0.76), seed=0):
        u, v = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
        xy = cam.undistort(np.c_[u.ravel(), v.ravel()]); rc = np.c_[xy, np.ones(len(xy))]
        c, R = se2_cam(pose, cam); d = rc @ (R @ cam.R_bc).T; d /= np.linalg.norm(d, axis=1, keepdims=True)
        _, ri, ti = self.mesh.ray.intersects_location(np.repeat(c[None], len(d), 0), d, multiple_hits=False)
        img = np.zeros(len(d)); L = np.asarray(light) / np.linalg.norm(light)
        nrm = self.N[ti]; nrm = nrm * np.sign(-(nrm * d[ri]).sum(1, keepdims=True))      # face the camera
        img[ri] = self.alb[ti] * (0.35 + 0.65 * np.clip(nrm @ L, 0, 1))
        img = img.reshape(H, W) * 255 + np.random.default_rng(seed).normal(0, 2.0, (H, W))
        return np.clip(img, 0, 255).astype(np.uint8)

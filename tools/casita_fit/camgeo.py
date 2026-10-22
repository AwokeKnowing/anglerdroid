"""Camera + casita model geometry for the mast-RGB <-> casita line fit.

World = casita frame of sim/casitahouse-walls-fixed.stl after the same normalisation emu/envs.env_casita uses
(min corner of the bounding box at the origin, metres): x east, y north, z up, floor z=0, wall tops z=2.28.
Camera pose = (x, y, z, yaw, pitch, roll): yaw = heading of the optical axis from +x (CCW, rad); pitch + = nose DOWN;
roll + = clockwise in the image (right side down).  Lens = Brown radial k1, k2 on normalised coords (the twin's
emu.realism.mast_truth model), principal point free or fixed.
"""
from __future__ import annotations
import numpy as np, trimesh


def load_casita(stl, drop_slivers=True):
    m = trimesh.load(stl, force="mesh")
    sc = 0.001 if (m.bounds[1] - m.bounds[0]).max() > 100 else 1.0
    m.apply_scale(sc); m.apply_translation(-m.bounds[0])
    if drop_slivers:   # zero-area faces left by the STL repair: invisible in a renderer, but rays hit them
        T = m.triangles; e = np.linalg.norm(T - np.roll(T, 1, axis=1), axis=2).max(1)
        keep = m.area_faces >= 1e-3 * e ** 2
        m = trimesh.Trimesh(m.vertices, m.faces[keep], process=False)
    return m


def R_world_to_cam(yaw, pitch, roll):
    """Rows = camera axes (x right, y down, z forward) in world coords."""
    f = np.array([np.cos(yaw), np.sin(yaw), 0.0])            # level forward
    r = np.array([np.sin(yaw), -np.cos(yaw), 0.0])           # level right
    d = np.array([0.0, 0.0, -1.0])                           # down
    # pitch nose-down about the right axis: forward tilts toward down
    cp, sp = np.cos(pitch), np.sin(pitch)
    f2 = cp * f + sp * d; d2 = -sp * f + cp * d
    # roll clockwise in the image (as seen by the camera) about forward: right axis tilts toward down
    cr, sr = np.cos(roll), np.sin(roll)
    r3 = cr * r + sr * d2; d3 = -sr * r + cr * d2
    return np.stack([r3, d3, f2])


class Cam:
    def __init__(self, p, w, h):
        # p: dict x y z yaw pitch roll f cx cy k1 k2
        self.p = p; self.w = w; self.h = h
        self.R = R_world_to_cam(p["yaw"], p["pitch"], p["roll"]); self.t = np.array([p["x"], p["y"], p["z"]])

    def to_cam(self, X):
        return (np.asarray(X) - self.t) @ self.R.T

    def distort(self, xn, yn):
        r2 = xn * xn + yn * yn; s = 1 + self.p["k1"] * r2 + self.p["k2"] * r2 * r2
        return xn * s, yn * s

    def undistort(self, xd, yd, it=20):
        xn, yn = xd.copy(), yd.copy()
        for _ in range(it):
            r2 = xn * xn + yn * yn; s = 1 + self.p["k1"] * r2 + self.p["k2"] * r2 * r2
            xn, yn = xd / s, yd / s
        return xn, yn

    def r_valid(self):
        """Largest undistorted normalised radius we trust: inside the image corner (+15 %) and below the radius
        where the radial polynomial stops being monotonic (beyond it barrel distortion folds points back in)."""
        rc = np.hypot(max(self.p["cx"], self.w - 1 - self.p["cx"]), max(self.p["cy"], self.h - 1 - self.p["cy"])) / self.p["f"]
        xn, _ = self.undistort(np.array([rc]), np.array([0.0])); r = float(xn[0]) * 1.15
        rr = np.linspace(0, 4 * r, 4000); d = 1 + 3 * self.p["k1"] * rr ** 2 + 5 * self.p["k2"] * rr ** 4
        bad = np.nonzero(d <= 0)[0]
        return min(r, rr[bad[0]] * 0.95) if len(bad) else r

    def project(self, X, valid=False):
        Pc = self.to_cam(X); z = Pc[..., 2]
        xn, yn = Pc[..., 0] / z, Pc[..., 1] / z
        xd, yd = self.distort(xn, yn)
        uv = np.stack([self.p["f"] * xd + self.p["cx"], self.p["f"] * yd + self.p["cy"]], -1)
        if valid:
            return uv, (z > 0.05) & (np.hypot(xn, yn) <= self.r_valid())
        return uv, z

    def pix_to_norm(self, uv):
        uv = np.asarray(uv, float)
        return self.undistort((uv[..., 0] - self.p["cx"]) / self.p["f"], (uv[..., 1] - self.p["cy"]) / self.p["f"])

    def rays(self, step=1):
        u, v = np.meshgrid(np.arange(0, self.w, step) + 0.5 * step - 0.5, np.arange(0, self.h, step) + 0.5 * step - 0.5)
        xn, yn = self.pix_to_norm(np.stack([u, v], -1))
        d = np.stack([xn, yn, np.ones_like(xn)], -1) @ self.R          # cam -> world directions
        return d / np.linalg.norm(d, axis=-1, keepdims=True), u.shape


def raycast(mesh, cam, step=1, with_floor=True):
    """Per-pixel (depth along ray, face id; -1 = floor, -2 = ceiling plane z = top, -3 = nothing)."""
    D, shp = cam.rays(step); D = D.reshape(-1, 3); O = np.repeat(cam.t[None], len(D), 0)
    loc, ri, ti = mesh.ray.intersects_location(O, D, multiple_hits=False)
    dist = np.full(len(D), np.inf); fid = np.full(len(D), -3)
    dd = np.linalg.norm(loc - O[ri], axis=1); dist[ri] = dd; fid[ri] = ti
    if with_floor:
        top = mesh.bounds[1][2]
        for zp, code in ((0.0, -1), (top, -2)):
            with np.errstate(divide="ignore", invalid="ignore"):
                tp = (zp - cam.t[2]) / D[:, 2]
            P = cam.t + tp[:, None] * D
            inside = (tp > 0) & (P[:, 0] >= 0) & (P[:, 0] <= mesh.bounds[1][0]) & (P[:, 1] >= 0) & (P[:, 1] <= mesh.bounds[1][1])
            better = inside & (tp < dist); dist[better] = tp[better]; fid[better] = code
    return dist.reshape(shp), fid.reshape(shp)


def segment_visible(mesh, cam, A, B, n=60, tol=0.03):
    """Sample a 3D segment, keep samples whose line of sight to the camera is not blocked by the walls."""
    s = np.linspace(0, 1, n)[:, None]; P = A + s * (B - A)
    d = P - cam.t; L = np.linalg.norm(d, axis=1); D = d / L[:, None]
    loc, ri, ti = mesh.ray.intersects_location(np.repeat(cam.t[None], n, 0), D, multiple_hits=False)
    hit = np.full(n, np.inf); hit[ri] = np.linalg.norm(loc - cam.t, axis=1)
    return P, (hit >= L - tol) & (cam.to_cam(P)[:, 2] > 0.05)

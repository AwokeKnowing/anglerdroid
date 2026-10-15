"""Mono 'RGB -> depth' layer for the RS1 (top-down mast) colour camera.  PROTOTYPE, map-only.

A small relative-depth net (Depth-Anything-V2-Small, Apache-2.0; or Distill-Any-Depth-Small, MIT) runs async at
~5-15 Hz.  Each frame its relative output is fitted per frame to the valid stereo pixels (least squares, inverse
depth = a*rel + b, one robust re-weight), giving metric depth where stereo has none (too close / textureless).
Pixels whose fitted depth puts a surface inside the robot's height band (0.10-0.95 m above floor) are returned as
robot-frame-camera points for a SEPARATE 'mono' obstacle layer (decays after `ttl`).  It never feeds the stereo
safety rule: no-depth-as-obstacle stays in nav/perceive.py (Ego._too_close) and safety.

Alignment (prototype): RS1 colour and depth share the optical axis (1.5 cm apart, colour FOV 42 deg vs depth
58 deg vertical), so the net output is resampled onto the depth grid's central window by the FOV ratio."""
from __future__ import annotations
import threading, time, math
import numpy as np
import cv2

MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


class MonoDepth:
    def __init__(self, fovy_d=58.0, fovy_c=42.0, size=308, floor_z=0.95, band=(0.10, 0.95), ttl=0.6, model=MODEL):
        import torch
        from transformers import AutoModelForDepthEstimation
        self.torch = torch
        self.net = AutoModelForDepthEstimation.from_pretrained(model).cuda().half().eval()
        self.size, self.floor_z, self.band, self.ttl = size, floor_z, band, ttl
        self.k = math.tan(math.radians(fovy_c) / 2) / math.tan(math.radians(fovy_d) / 2)   # colour/depth FOV ratio
        self.mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1).half()
        self.std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1).half()
        self._job = None; self._cv = threading.Condition(); self.pts = np.zeros((0, 3), np.float32); self.t = 0.0
        self.stats = dict(n=0, ms=0.0, fit_rmse=0.0)
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, t, color, verts, shape):
        with self._cv: self._job = (t, color, verts, shape); self._cv.notify()

    def points(self, now):
        return self.pts if now - self.t < self.ttl else np.zeros((0, 3), np.float32)

    def infer(self, color):
        th = self.torch
        x = th.from_numpy(cv2.resize(color, (self.size, self.size))).cuda().half().permute(2, 0, 1)[None] / 255.0
        with th.no_grad():
            r = self.net(pixel_values=(x - self.mean) / self.std).predicted_depth[0].float().cpu().numpy()
        return r                                        # relative inverse depth (bigger = closer)

    def process(self, t, color, verts, shape):
        t0 = time.perf_counter(); H, W = shape
        rel = self.infer(color)
        V = np.asarray(verts, np.float32).reshape(H, W, 3)
        # colour covers the central k-fraction of the depth image
        h, w = int(H * self.k), int(W * self.k); y0, x0 = (H - h) // 2, (W - w) // 2
        Vc = V[y0:y0 + h, x0:x0 + w]; R = cv2.resize(rel, (w, h), interpolation=cv2.INTER_LINEAR)
        z = Vc[..., 2]; ok = z > 0.01
        if ok.sum() < 200: return None
        A = np.c_[R[ok], np.ones(ok.sum())]; y = 1.0 / z[ok]
        p = np.linalg.lstsq(A, y, rcond=None)[0]
        res = np.abs(A @ p - y); wgt = 1.0 / (1.0 + (res / (np.median(res) + 1e-6)) ** 2)   # one robust re-weight
        p = np.linalg.lstsq(A * wgt[:, None], y * wgt, rcond=None)[0]
        inv = np.clip(R * p[0] + p[1], 1e-3, None); zm = 1.0 / inv
        self.stats["fit_rmse"] = float(np.sqrt(np.mean((zm[ok] - z[ok]) ** 2)))
        hgt = self.floor_z - zm
        sel = (hgt > self.band[0]) & (hgt < self.band[1]) & (zm > 0.02)
        # rays for the central window from valid stereo returns (nearest valid in the column/row is fine at 1 cm)
        ry = np.where(ok[..., None], Vc[..., :2] / np.maximum(z, 1e-3)[..., None], np.nan)
        gx, gy = np.meshgrid(np.arange(w), np.arange(h)); okr = ok.ravel()
        if okr.sum() > 50:            # fill unknown rays by a linear fit ray = c0 + c1*col/row (pinhole)
            cx = np.polyfit(gx.ravel()[okr], ry[..., 0].ravel()[okr], 1); cy = np.polyfit(gy.ravel()[okr], ry[..., 1].ravel()[okr], 1)
            ry = np.dstack([np.polyval(cx, gx), np.polyval(cy, gy)])
        P = np.c_[(ry[..., 0] * zm)[sel], (ry[..., 1] * zm)[sel], zm[sel]].astype(np.float32)
        self.stats.update(n=int(sel.sum()), ms=(time.perf_counter() - t0) * 1e3)
        return P

    def _loop(self):
        while True:
            with self._cv:
                while self._job is None: self._cv.wait()
                job, self._job = self._job, None
            try:
                P = self.process(*job)
                if P is not None: self.pts, self.t = P, job[0]
            except Exception as e:                                  # prototype: never take the loop down
                self.stats["err"] = repr(e)[:120]

#!/usr/bin/env python3
"""Tiny OpenCV window for the GPU vec-env. Not on the hot path."""
from __future__ import annotations

import os
import time

import cv2
import numpy as np

try:
    from . import kernels as K
except ImportError:
    import kernels as K


def _depth_color(d):
    x = np.clip(np.asarray(d, dtype=np.float32) / K.MAX_RAY, 0.0, 1.0)
    u8 = (x * 255.0).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)


def _topdown(snap, w=420, h=520):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (22, 24, 28)
    fx0, fx1 = K.FLOOR_X0, K.FLOOR_X1
    fy0, fy1 = K.FLOOR_Y0, K.FLOOR_Y1

    def xy(x, y):
        px = int((float(x) - fx0) / max(fx1 - fx0, 1e-3) * (w - 8) + 4)
        py = int((1.0 - (float(y) - fy0) / max(fy1 - fy0, 1e-3)) * (h - 8) + 4)
        return px, py

    boxes = snap["boxes"]
    for b in boxes:
        hx = float(b[3])
        if hx <= 0.04:
            continue
        cx, cy, hy = float(b[0]), float(b[1]), float(b[4])
        kind = int(b[7])
        color = (180, 190, 200) if kind <= 1 else ((90, 140, 200) if kind == 2 else (80, 180, 80))
        p0 = xy(cx - hx, cy - hy)
        p1 = xy(cx + hx, cy + hy)
        cv2.rectangle(img, p0, p1, color, 1)
    poly = snap["poly"]
    pts = [xy(p[0], p[1]) for p in poly]
    for a, b in zip(pts, pts[1:]):
        cv2.line(img, a, b, (40, 220, 255), 1)
    rx, ry = xy(snap["x"], snap["y"])
    yaw = float(snap["yaw"])
    cv2.circle(img, (rx, ry), 5, (0, 255, 255), -1)
    cv2.line(
        img, (rx, ry),
        (int(rx + 14 * np.cos(yaw)), int(ry - 14 * np.sin(yaw))),
        (0, 255, 255), 2,
    )
    return img


class GpuDash:
    def __init__(self, title="GPU insect VEC"):
        os.environ.setdefault("DISPLAY", ":1")
        self.title = title
        self.closed = False
        self._last = 0.0
        self.win_hist = []
        self.chord_hist = []
        self.sps_hist = []
        try:
            cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.title, 980, 640)
        except Exception as e:
            print("gpu dash skip:", e, flush=True)
            self.closed = True

    def present(self, env, sps=0.0, note="", min_dt=0.12):
        if self.closed:
            return
        now = time.time()
        if now - self._last < min_dt:
            return
        self._last = now
        try:
            snap = env.env0_cpu()
            st = env.stats()
            self.win_hist.append(st["win_frac"])
            self.chord_hist.append(min(1.0, st["mean_chord"] / 4.0) if st["mean_chord"] else 0.0)
            self.sps_hist.append(sps / 50000.0)
            self.win_hist = self.win_hist[-240:]
            self.chord_hist = self.chord_hist[-240:]
            self.sps_hist = self.sps_hist[-240:]
            d = _depth_color(snap["depth"])
            d = cv2.resize(d, (320, 192), interpolation=cv2.INTER_NEAREST)
            ego = (np.clip(snap["img"][1], 0, 1) * 255).astype(np.uint8)
            ego = cv2.applyColorMap(cv2.resize(ego, (320, 192)), cv2.COLORMAP_BONE)
            vis = (np.clip(snap["img"][2], 0, 1) * 255).astype(np.uint8)
            vis = cv2.applyColorMap(cv2.resize(vis, (320, 192)), cv2.COLORMAP_HOT)
            sensors = np.concatenate([d, ego, vis], axis=1)
            top = _topdown(snap)
            top = cv2.resize(top, (sensors.shape[1] // 2, 360))
            chart = np.zeros((360, sensors.shape[1] - top.shape[1], 3), dtype=np.uint8)
            chart[:] = (18, 20, 24)
            self._spark(chart, self.win_hist, (220, 220, 80), "win")
            self._spark(chart, self.chord_hist, (80, 200, 255), "chord")
            self._spark(chart, self.sps_hist, (80, 255, 120), "sps")
            hud = np.concatenate([top, chart], axis=1)
            if hud.shape[1] != sensors.shape[1]:
                hud = cv2.resize(hud, (sensors.shape[1], hud.shape[0]))
            frame = np.concatenate([sensors, hud], axis=0)
            txt = "N=%d  sps=%.0f  win=%.2f  chord=%.2f  cells=%.1f  path=%.1f  %s" % (
                env.n, sps, st["win_frac"], st["mean_chord"], st["mean_cells"],
                st["mean_path"], note,
            )
            cv2.putText(frame, txt, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
            cv2.imshow(self.title, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                self.closed = True
        except Exception as e:
            print("gpu dash:", type(e).__name__, e, flush=True)
            self.closed = True

    def _spark(self, img, hist, color, label):
        h, w = img.shape[:2]
        if label == "win":
            y0, y1 = 20, h // 3 - 8
        elif label == "chord":
            y0, y1 = h // 3 + 4, 2 * h // 3 - 8
        else:
            y0, y1 = 2 * h // 3 + 4, h - 12
        cv2.putText(img, label, (8, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        if len(hist) < 2:
            return
        xs = np.linspace(8, w - 8, len(hist))
        ys = y1 - np.clip(np.asarray(hist), 0, 1) * (y1 - y0 - 20)
        pts = np.stack([xs, ys], axis=1).astype(np.int32)
        cv2.polylines(img, [pts], False, color, 1, cv2.LINE_AA)

    def close(self):
        if not self.closed:
            try:
                cv2.destroyWindow(self.title)
            except Exception:
                pass
        self.closed = True

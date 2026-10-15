"""Unstuck policy: compact TinyUnstuck net, else a tiny yaw-search fallback.

Runs only while plus-x / safety is throttled. InsectPlanner holds this until
plus-x stays clear, then the 5 Hz heuristic takes over again.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

DEFAULT_CKPT = Path.home() / ".kevin" / "unstuck.pt"
REPO_CKPT = Path(__file__).resolve().parents[1] / "sim" / "unstuck" / "ckpt.pt"
_SIM = Path(__file__).resolve().parents[1] / "sim"
if str(_SIM) not in sys.path:
    sys.path.insert(0, str(_SIM))

IMG_H, IMG_W = 48, 80
BACK_M, FWD_M, LAT_M = 0.80, 2.40, 0.96


def crop_kevin_ego(height, obs=None, rcx=None, rcy=None, px_size=0.01):
    """Downsample a 1 cm ego heightmap into the 48×80 unstuck crop."""
    hmap = np.asarray(height if height is not None else obs, dtype=np.float32)
    if hmap.ndim != 2:
        return np.zeros((IMG_H, IMG_W), dtype=np.float32)
    hh, ww = hmap.shape
    if rcx is None or rcy is None:
        try:
            from robot_config import RCX as KCX, RCY as KCY
            rcx, rcy = float(KCX), float(KCY)
        except Exception:
            rcx, rcy = 81.0, 119.0
    c0 = int(round(rcx - BACK_M / px_size))
    c1 = int(round(rcx + FWD_M / px_size))
    r0 = int(round(rcy - LAT_M / px_size))
    r1 = int(round(rcy + LAT_M / px_size))
    pad = np.zeros((max(1, r1 - r0), max(1, c1 - c0)), dtype=np.float32)
    src_r0, dst_r0 = max(0, r0), max(0, -r0)
    src_r1 = min(hh, r1)
    dst_r1 = dst_r0 + (src_r1 - max(0, r0))
    src_c0, dst_c0 = max(0, c0), max(0, -c0)
    src_c1 = min(ww, c1)
    dst_c1 = dst_c0 + (src_c1 - max(0, c0))
    if src_r1 > src_r0 and src_c1 > src_c0:
        sl = hmap[src_r0:src_r1, src_c0:src_c1]
        occ = (sl >= 5.0).astype(np.float32)
        if obs is not None:
            o = np.asarray(obs, dtype=np.float32)
            if o.shape == hmap.shape:
                occ = np.maximum(
                    occ, (o[src_r0:src_r1, src_c0:src_c1] > 100).astype(np.float32))
        pad[dst_r0:dst_r1, dst_c0:dst_c1] = occ
    ys = (np.linspace(0, pad.shape[0] - 1, IMG_H)).astype(np.int32)
    xs = (np.linspace(0, pad.shape[1] - 1, IMG_W)).astype(np.int32)
    return pad[ys[:, None], xs[None, :]].astype(np.float32)


def _look_px(height, obs, col0, row0, yaw_img, dist_m, px_size=0.01):
    """Occupancy along a heading in ego pixels. yaw_img=0 is +x (right / forward)."""
    h, w = height.shape[:2]
    c, s = float(np.cos(yaw_img)), float(np.sin(yaw_img))
    n = max(2, int(round(dist_m / px_size)))
    for i in range(1, n + 1):
        col = int(round(col0 + c * i))
        row = int(round(row0 - s * i))
        if row < 0 or row >= h or col < 0 or col >= w:
            return i * px_size
        hv = float(height[row, col])
        ov = float(obs[row, col]) if obs is not None else 0.0
        if hv >= 5.0 or ov > 0:
            return i * px_size
    return dist_m


class UnstuckPolicy:
    """Cheap (v,w) while pinned. Neural if ckpt loads, else scripted."""

    def __init__(self, ckpt=None, hp=None):
        self.hp = hp or {}
        self.max_v = float(self.hp.get("max_v", 0.28))
        self.max_w = float(self.hp.get("max_w", 0.80))
        self.v_rev = float(self.hp.get("v_rev", -0.10))
        self.net = None
        self.hx = None
        self.device = "cpu"
        path = ckpt or os.environ.get("KEVIN_UNSTUCK_CKPT") or ""
        candidates = []
        if path:
            candidates.append(Path(path))
        candidates.extend((REPO_CKPT, DEFAULT_CKPT))
        for p in candidates:
            if p.is_file():
                self._try_load(p)
                if self.net is not None:
                    break
        kind = "neural" if self.net is not None else "scripted"
        print("UnstuckPolicy: %s" % kind, flush=True)

    def _try_load(self, path: Path):
        try:
            import torch
            from unstuck_net import TinyUnstuck
            ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
            net = TinyUnstuck()
            sd = ckpt.get("net", ckpt.get("model", ckpt))
            net.load_state_dict(sd, strict=False)
            net.eval()
            self.net = net
            print("UnstuckPolicy: loaded %s (%d params)" % (path, net.n_params()), flush=True)
        except Exception as e:
            print("UnstuckPolicy: ckpt skip %s (%s)" % (path, e), flush=True)
            self.net = None

    def reset(self):
        self.hx = None

    def scripted(self, plus_x, left_px, right_px, exe_v, exe_w, goal_bearing=0.0):
        """One heuristic: turn toward the more open side (tiny goal tie-break)."""
        stop = float(self.hp.get("plus_x_stop", 0.28))
        if plus_x < stop:
            v = float(self.v_rev) * 0.5 if (abs(exe_v) < 0.04 and plus_x < 0.16) else 0.0
            if abs(left_px - right_px) < 0.12 and abs(goal_bearing) > 0.15:
                side = 1.0 if goal_bearing > 0.0 else -1.0
            else:
                side = 1.0 if left_px >= right_px else -1.0
            return v, side * 0.62
        v = min(0.14, self.max_v)
        w = 0.22 if left_px > right_px + 0.05 else (-0.22 if right_px > left_px + 0.05 else 0.0)
        return v, w

    def _clip_reach(self, v, w, exe_v, exe_w):
        try:
            from insect_planner import reachable_vw
            v_lo, v_hi, w_lo, w_hi = reachable_vw(exe_v, exe_w, self.hp)
            return float(np.clip(v, v_lo, v_hi)), float(np.clip(w, w_lo, w_hi))
        except Exception:
            return (
                float(np.clip(v, self.v_rev, self.max_v)),
                float(np.clip(w, -self.max_w, self.max_w)),
            )

    def act(self, plus_x, left_px, right_px, exe_v, exe_w, img=None, vec=None,
            goal_bearing=0.0):
        if self.net is not None and img is not None and vec is not None:
            try:
                import torch
                with torch.no_grad():
                    im = torch.from_numpy(np.asarray(img, dtype=np.float32))
                    if im.dim() == 2:
                        im = im.unsqueeze(0)
                    im = im.unsqueeze(0)
                    ve = torch.from_numpy(np.asarray(vec, dtype=np.float32)).reshape(1, -1)
                    act, _, _, self.hx = self.net.act(im, ve, self.hx, deterministic=True)
                    a = act.squeeze(0).cpu().numpy()
                v = float(np.clip(a[0] * self.max_v, self.v_rev, self.max_v))
                w = float(np.clip(a[1] * self.max_w, -self.max_w, self.max_w))
                v, w = self._clip_reach(v, w, exe_v, exe_w)
                return v, w, "neural"
            except Exception as e:
                print("UnstuckPolicy: infer skip %s" % e, flush=True)
        v, w = self.scripted(plus_x, left_px, right_px, exe_v, exe_w, goal_bearing)
        v, w = self._clip_reach(v, w, exe_v, exe_w)
        return v, w, "scripted"

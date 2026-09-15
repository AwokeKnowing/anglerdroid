#!/usr/bin/env python3
"""Clip / HUD frames for rl_nav. Compose stays off the train hot path."""
from __future__ import annotations

import json
import math
import os
import queue
import shutil
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

try:
    from . import kernels as K
except ImportError:
    import kernels as K

# Occupancy HUD in Kevin bag tones (BGR): unknown / clear / obstacle.
# SELF is stamped after nearest-neighbor upscale. Policy tensor stores SELF
# as 0.0 (same as UNKNOWN); if we colorize then bilinear-scale, the hull
# mixes green/black/blue and the robot looks grey / white / blue.
_EGO_LUT = np.array(
    (
        (22, 18, 18),
        (90, 170, 40),
        (40, 50, 210),
    ),
    dtype=np.uint8,
)
_SELF_BGR = np.array((220, 90, 40), dtype=np.uint8)
_THROT_BGR = np.array((0, 210, 255), dtype=np.uint8)
# Same hull as kernels.in_self_body (meters, axle frame).
_SELF_LX0, _SELF_LX1 = -0.16, 0.20
_SELF_LY = 0.22


def _shape_pts(cx, cy, hx, hy, yaw, n, w, h):
    c, s = math.cos(yaw), math.sin(yaw)
    pts = []
    for k in range(int(n)):
        a = 2.0 * math.pi * k / float(n)
        lx, ly = hx * math.cos(a), hy * math.sin(a)
        pts.append(_xy(cx + c * lx - s * ly, cy + s * lx + c * ly, w, h))
    return np.array(pts, dtype=np.int32)


def _xy(x, y, w, h):
    fx0, fx1 = K.FLOOR_X0, K.FLOOR_X1
    fy0, fy1 = K.FLOOR_Y0, K.FLOOR_Y1
    px = int((float(x) - fx0) / max(fx1 - fx0, 1e-3) * (w - 16) + 8)
    py = int((1.0 - (float(y) - fy0) / max(fy1 - fy0, 1e-3)) * (h - 16) + 8)
    return px, py


def _occupancy(snap, w, h):
    img = np.full((h, w, 3), 22, dtype=np.uint8)
    seen = snap.get("seen")
    valid = snap.get("valid")
    if seen is None or valid is None:
        return img
    ny, nx = int(K.SEC_NY), int(K.SEC_NX)
    n = ny * nx
    v = np.asarray(valid[:n]).reshape(ny, nx)
    s = np.asarray(seen[:n]).reshape(ny, nx)
    g = np.full((ny, nx, 3), 22, dtype=np.uint8)
    walk = v != 0
    g[walk & (s == 0)] = (28, 28, 32)
    g[walk & (s != 0)] = (38, 88, 52)
    inner = cv2.resize(g[::-1], (max(w - 16, 1), max(h - 16, 1)), interpolation=cv2.INTER_NEAREST)
    img[8:8 + inner.shape[0], 8:8 + inner.shape[1]] = inner
    return img


def _draw_boxes(img, boxes, w, h):
    for b in boxes:
        hx = float(b[3])
        if hx < 1e-4:
            continue
        cx, cy, hy = float(b[0]), float(b[1]), float(b[4])
        hx, hy = max(hx, 0.06), max(hy, 0.06)
        yaw = float(b[6])
        kind = int(b[7])
        fill = {
            1: (95, 98, 105),
            2: (58, 72, 128),
            3: (110, 78, 118),
            4: (52, 92, 140),
            5: (40, 88, 128),
            6: (70, 110, 90),
            7: (45, 100, 155),
            8: (80, 70, 140),
        }.get(kind, (90, 110, 150))
        if kind == 7:
            pts = _shape_pts(cx, cy, hx, hy, yaw, 24, w, h)
            cv2.fillConvexPoly(img, pts, fill)
        elif kind >= 8:
            n = int(round(float(b[2])))
            n = min(7, max(5, n))
            pts = _shape_pts(cx, cy, hx, hy, yaw, n, w, h)
            cv2.fillConvexPoly(img, pts, fill)
        elif abs(yaw) > 1e-3:
            c, s = math.cos(yaw), math.sin(yaw)
            corners = [
                _xy(cx + c * sx - s * sy, cy + s * sx + c * sy, w, h)
                for sx, sy in ((hx, hy), (hx, -hy), (-hx, -hy), (-hx, hy))
            ]
            cv2.fillConvexPoly(img, np.array(corners, dtype=np.int32), fill)
        else:
            p0 = _xy(cx - hx, cy + hy, w, h)
            p1 = _xy(cx + hx, cy - hy, w, h)
            x0, x1 = min(p0[0], p1[0]), max(p0[0], p1[0])
            y0, y1 = min(p0[1], p1[1]), max(p0[1], p1[1])
            cv2.rectangle(img, (x0, y0), (x1, y1), fill, -1)
    return img


def _topdown(snap, w=400, h=500, trail=None, furn=None):
    img = _occupancy(snap, w, h)
    if furn is not None:
        m = furn.any(axis=2)
        img[m] = furn[m]
    else:
        _draw_boxes(img, snap["boxes"], w, h)
    for i in range(int(snap["n_movers"])):
        cv2.circle(img, _xy(snap["mx"][i], snap["my"][i], w, h), 7, (80, 160, 255), 2)
    if trail:
        pts = [_xy(p[0], p[1], w, h) for p in trail]
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, a, b, (50, 190, 230), 2)
    rx, ry = _xy(snap["x"], snap["y"], w, h)
    yaw = float(snap["yaw"])
    cv2.circle(img, (rx, ry), 7, (0, 255, 255), -1)
    cv2.line(
        img, (rx, ry),
        (int(rx + 18 * np.cos(yaw)), int(ry - 18 * np.sin(yaw))),
        (0, 255, 255), 2,
    )
    return img


def _sec_center(s):
    ix = int(s) % K.SEC_NX
    iy = int(s) // K.SEC_NX
    cx = K.FLOOR_X0 + (ix + 0.5) * ((K.FLOOR_X1 - K.FLOOR_X0) / float(K.SEC_NX))
    cy = K.FLOOR_Y0 + (iy + 0.5) * ((K.FLOOR_Y1 - K.FLOOR_Y0) / float(K.SEC_NY))
    return cx, cy


def _ego_occupancy(ego):
    x = np.clip(np.asarray(ego, dtype=np.float32), 0.0, 1.0)
    if x.ndim == 3:
        x = x[0]
    idx = np.zeros(x.shape, dtype=np.int32)
    idx = np.where(x >= 0.85, 2, idx)
    idx = np.where((x >= 0.25) & (x < 0.85), 1, idx)
    return _EGO_LUT[idx].copy()


def _fill_body(bgr, lx0, lx1, ly0, ly1, color):
    h, w = bgr.shape[:2]
    dx = K.EGO_X1 - K.EGO_X0
    dy = K.EGO_Y1 - K.EGO_Y0
    x0 = int(np.floor((lx0 - K.EGO_X0) / dx * w))
    x1 = int(np.ceil((lx1 - K.EGO_X0) / dx * w))
    y0 = int(np.floor((K.EGO_Y1 - ly1) / dy * h))
    y1 = int(np.ceil((K.EGO_Y1 - ly0) / dy * h))
    bgr[max(0, y0):min(h, y1), max(0, x0):min(w, x1)] = color


def _stamp_self(bgr):
    """Solid SELF rect in pixel space. Call after nearest upscale, not before."""
    _fill_body(bgr, _SELF_LX0, _SELF_LX1, -_SELF_LY, _SELF_LY, _SELF_BGR)
    return bgr


def _stamp_throttle(bgr, snap):
    """Amber on the hull sector that is the throttle source (policy sees 0.20)."""
    if float(snap.get("throt_f", 0.0)) > 0.08:
        _fill_body(bgr, 0.02, 0.20, -_SELF_LY, _SELF_LY, _THROT_BGR)
    if float(snap.get("throt_b", 0.0)) > 0.08:
        _fill_body(bgr, _SELF_LX0, 0.02, -_SELF_LY, _SELF_LY, _THROT_BGR)
    if float(snap.get("throt_l", 0.0)) > 0.08:
        _fill_body(bgr, _SELF_LX0, _SELF_LX1, 0.10, _SELF_LY, _THROT_BGR)
    if float(snap.get("throt_r", 0.0)) > 0.08:
        _fill_body(bgr, _SELF_LX0, _SELF_LX1, -_SELF_LY, -0.10, _THROT_BGR)
    return bgr


def _ego_color(ego, snap=None):
    rgb = _ego_occupancy(ego)
    _stamp_self(rgb)
    if snap is not None:
        _stamp_throttle(rgb, snap)
    return rgb


def _end_info(snap, ended=False):
    hits = int(snap.get("hits", 0) or 0)
    nf = int(snap.get("n_free", 0) or 0)
    if nf < 1:
        nf = 1
    cut = int(snap.get("cut", 0) or 0)
    how = {1: "TIMEOUT", 2: "CRASH", 3: "TIMEOUT"}.get(cut, "RUNNING")
    if ended and how == "RUNNING":
        how = "HOP"
    return {
        "how": how,
        "cut": cut,
        "hits": hits,
        "n_free": nf,
        "cover": float(hits) / float(nf),
        "path": float(snap.get("path_m", 0.0) or 0.0),
        "t": float(snap.get("t_sim", 0.0) or 0.0),
        "plus_x": float(snap.get("plus_x", 0.0) or 0.0),
        "n_movers": int(snap.get("n_movers", 0) or 0),
    }


def _last_snap(*seqs):
    for seq in seqs:
        if seq:
            return seq[-1]
    return None


def _stamp_end(hud, info, ended=False):
    """Big how-it-ended bar. Packs stamp the FINAL cut on every frame."""
    how = info["how"]
    if ended and how == "RUNNING":
        how = "HOP"
    col = {
        "CRASH": (40, 50, 230),
        "TIMEOUT": (70, 190, 80),
        "HOP": (80, 180, 230),
        "RUNNING": (170, 170, 170),
    }.get(how, (170, 170, 170))
    h, w = hud.shape[:2]
    y0 = h - 58
    cv2.rectangle(hud, (0, y0), (w, h), (12, 12, 12), -1)
    cv2.rectangle(hud, (0, y0), (8, h), col, -1)
    cv2.putText(hud, how, (18, y0 + 38), cv2.FONT_HERSHEY_SIMPLEX, 1.05, col, 3, cv2.LINE_AA)
    cover = "cover %d/%d (%.0f%%)" % (info["hits"], info["n_free"], 100.0 * info["cover"])
    line2 = "%s   path %.1fm   t %.0fs" % (cover, info["path"], info["t"])
    if how == "CRASH":
        line2 += "   gap %.2fm   pets %d" % (info["plus_x"], info["n_movers"])
    cv2.putText(hud, line2, (210, y0 + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    return hud


# Dual HUD: overhead 400×500 @ (8,8), ego 320×240 @ (_EGO_X,_EGO_Y).
# Kevin look is RS1 house rectangle ∪ RS2 cone; UL of the ego panel is
# outside that silhouette (black). Speed chip lives there.
_HUD_W, _HUD_H = 760, 560
_EGO_X, _EGO_Y = 428, 8
_EGO_W, _EGO_H = 320, 240


def compose_hud(snap, st=None, note="", sps=0.0, trail=None, furn=None, end=None, ended=False):
    top = _topdown(snap, trail=trail, furn=furn)
    # Integer 4× nearest (80×60 → 320×240). Non-integer + AREA is why SELF
    # went grey/white: policy stores 0.0, stamp is blue, mix with green.
    ego = cv2.resize(_ego_occupancy(snap["ego"]), (_EGO_W, _EGO_H), interpolation=cv2.INTER_NEAREST)
    _stamp_self(ego)
    _stamp_throttle(ego, snap)
    hud = np.full((_HUD_H, _HUD_W, 3), 16, dtype=np.uint8)
    hud[8:8 + top.shape[0], 8:8 + top.shape[1]] = top
    hud[_EGO_Y:_EGO_Y + ego.shape[0], _EGO_X:_EGO_X + ego.shape[1]] = ego
    info = end or _end_info(snap)
    _stamp_end(hud, info, ended=ended)
    if note:
        cv2.putText(hud, note, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return hud


def save_gif(frames, path, duration_ms=90):
    from PIL import Image
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
    pal = rgb[0].quantize(colors=256, dither=Image.Dither.NONE)
    imgs = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in rgb]
    imgs[0].save(
        path,
        save_all=True,
        append_images=imgs[1:],
        duration=int(duration_ms),
        loop=0,
        optimize=False,
    )
    return path


def save_mp4(frames, path, fps=12):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    if not vw.isOpened():
        raise RuntimeError("VideoWriter failed for %s" % path)
    for f in frames:
        vw.write(f)
    vw.release()
    return path


def save_contact(frames, path, cols=4, rows=3, tw=320, th=280):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = cols * rows
    last = max(len(frames) - 1, 0)
    idxs = [int(i * last / max(n - 1, 1)) for i in range(n)]
    tiles = [cv2.resize(frames[i], (tw, th), interpolation=cv2.INTER_NEAREST) for i in idxs]
    row_imgs = [np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)]
    cv2.imwrite(str(path), np.vstack(row_imgs))
    return path


def frames_from_snaps(snaps, st=None, note="", sps=0.0):
    trail = []
    frames = []
    last = _last_snap(snaps)
    end = _end_info(last, ended=True) if last is not None else None
    for snap in snaps:
        if trail:
            dx = snap["x"] - trail[-1][0]
            dy = snap["y"] - trail[-1][1]
            if dx * dx + dy * dy > 4.0:
                trail = []
        trail.append((snap["x"], snap["y"]))
        if len(trail) > 80:
            trail = trail[-80:]
        frames.append(compose_hud(
            snap, st=st, note=note, sps=sps, trail=trail, end=end, ended=True,
        ))
    return frames


def snaps_to_gif(snaps, path, st=None, note="", sps=0.0):
    return save_gif(frames_from_snaps(snaps, st=st, note=note, sps=sps), path)


def snaps_to_clip(snaps, stem, st=None, note="", sps=0.0):
    """Write PNG contact + MP4. GIF preview in Cursor is broken (webview SW)."""
    stem = Path(stem)
    frames = frames_from_snaps(snaps, st=st, note=note, sps=sps)
    png = save_contact(frames, stem.with_suffix(".png"))
    mp4 = save_mp4(frames, stem.with_suffix(".mp4"))
    # Half-res preview for the canvas. Full-HUD 300-frame GIF was ~2–3 min
    # and could overlap a second compose (OOM). Preview is 60 × ~380 px.
    prev = [
        cv2.resize(f, (f.shape[1] // 2, f.shape[0] // 2), interpolation=cv2.INTER_NEAREST)
        for f in frames
    ]
    gif = save_gif(prev, stem.with_suffix(".gif"), duration_ms=100)
    last = stem.with_name(stem.name + "_last.png")
    cv2.imwrite(str(last), frames[-1])
    return {"png": png, "mp4": mp4, "gif": gif, "last": last, "n": len(frames)}


class RlDash:
    def __init__(self, title="rl_nav"):
        os.environ.setdefault("DISPLAY", ":1")
        self.title = title
        self.closed = False
        self._last = 0.0
        try:
            cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.title, 980, 640)
        except Exception as e:
            print("rl dash skip:", e, flush=True)
            self.closed = True

    def present(self, env, sps=0.0, note="", min_dt=0.12):
        if self.closed:
            return
        now = time.time()
        if now - self._last < min_dt:
            return
        self._last = now
        snap = env.env0_cpu()
        hud = compose_hud(snap, st=env.stats(), note=note, sps=sps)
        cv2.imshow(self.title, hud)
        if cv2.waitKey(1) & 0xFF == 27:
            self.closed = True

    def close(self):
        if not self.closed:
            try:
                cv2.destroyWindow(self.title)
            except Exception:
                pass
            self.closed = True


# Full-episode pack: first 5 s wall @ 1×, middle @ 15×, last 5 s wall @ 1×, 25 fps.
_HEAD_S = 5.0
_TAIL_S = 5.0
_PLAY_FPS = 25
_SIM_HZ = 5
_DUP_1X = _PLAY_FPS // _SIM_HZ
_MID_STRIDE = 3  # 15× at 25 fps from 5 Hz
_HEAD_N = int(_HEAD_S * _SIM_HZ)
_TAIL_N = int(_TAIL_S * _SIM_HZ)


def _stamp_speed(bgr, label):
    # 75% of the old 210×62 / 150×62 top-right chip, parked in the black
    # UL of the ego panel (outside house rectangle ∪ cone).
    fast = label.startswith("15")
    bar = (0, 80, 255) if fast else (60, 200, 80)
    tw = 158 if fast else 112
    th = 47
    x0 = _EGO_X + 6
    y0 = _EGO_Y + 6
    cv2.rectangle(bgr, (x0, y0), (x0 + tw, y0 + th), (12, 12, 12), -1)
    cv2.rectangle(bgr, (x0, y0), (x0 + tw, y0 + th), bar, 2)
    cv2.putText(
        bgr, label, (x0 + 8, y0 + 34),
        cv2.FONT_HERSHEY_SIMPLEX, 1.09, bar, 2, cv2.LINE_AA,
    )
    return bgr


def _ffmpeg_exe():
    here = Path(__file__).resolve().parent / ".bin" / "ffmpeg"
    cands = [
        shutil.which("ffmpeg"),
        str(here) if here.is_file() else None,
        "/usr/bin/ffmpeg",
    ]
    try:
        import imageio_ffmpeg
        cands.append(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        pass
    for ff in cands:
        if ff and os.path.isfile(ff) and os.access(ff, os.X_OK):
            return ff
    return None


def _open_mp4(path, w, h, fps):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.mp4")
    ff = _ffmpeg_exe()
    if ff:
        cmd = [
            ff, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", "%dx%d" % (w, h), "-r", str(int(fps)), "-i", "-",
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(tmp),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        return ("ff", proc, tmp, path)
    for four in ("avc1", "H264", "mp4v"):
        vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*four), float(fps), (w, h))
        if vw.isOpened():
            return ("cv", vw, tmp, path)
    raise RuntimeError("no mp4 writer for %s" % path)


def _write_mp4_frame(hnd, bgr):
    kind, obj, _tmp, _path = hnd
    if kind == "ff":
        obj.stdin.write(np.ascontiguousarray(bgr).tobytes())
    else:
        obj.write(bgr)


def _close_mp4(hnd):
    kind, obj, tmp, path = hnd
    if kind == "ff":
        obj.stdin.close()
        obj.wait()
    else:
        obj.release()
    if tmp.is_file():
        os.replace(tmp, path)
    return path


def _compose_pack(head, mid, tail, st, sps, note, ended=True):
    trail = []
    furn = None
    seed = (head or mid or tail or [None])[0]
    if seed is not None:
        furn = np.zeros((500, 400, 3), dtype=np.uint8)
        _draw_boxes(furn, seed["boxes"], 400, 500)
    last = _last_snap(tail, mid, head)
    end = _end_info(last, ended=ended) if last is not None else None

    def one(snap, label, dup):
        nonlocal trail
        if trail:
            dx = snap["x"] - trail[-1][0]
            dy = snap["y"] - trail[-1][1]
            if dx * dx + dy * dy > 4.0:
                trail = []
        trail.append((snap["x"], snap["y"]))
        if len(trail) > 80:
            trail = trail[-80:]
        hud = compose_hud(
            snap, st=st, note=note, sps=sps, trail=trail, furn=furn,
            end=end, ended=ended,
        )
        _stamp_speed(hud, label)
        return hud, dup

    out = []
    short = len(mid) == 0
    for s in head:
        out.append(one(s, "1x", _DUP_1X))
    for s in mid:
        out.append(one(s, "15x", 1))
    for s in tail:
        out.append(one(s, "1x", _DUP_1X))
    if short and not tail and head:
        # tiny episode: already all 1x in head
        pass
    return out


class EpisodeRecorder:
    """Grab env0 ticks (pre-reset). Encode off-thread when the episode ends."""

    def __init__(self, out_dir):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.head = []
        self.mid = []
        self.ring = deque(maxlen=_TAIL_N)
        self.n = 0
        self.step = 0
        self.q = queue.Queue(maxsize=1)
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _submit(self, pack):
        try:
            self.q.put_nowait(pack)
            return
        except queue.Full:
            pass
        try:
            self.q.get_nowait()
        except queue.Empty:
            pass
        try:
            self.q.put_nowait(pack)
        except queue.Full:
            pass

    def on_tick(self, snap, done, step=0):
        self.step = int(step)
        self.n += 1
        if len(self.head) < _HEAD_N:
            self.head.append(snap)
        else:
            self.ring.append(snap)
            if (self.n - _HEAD_N) % _MID_STRIDE == 0:
                self.mid.append(snap)
        if not done:
            return
        tail = list(self.ring)
        t_end = 0.0
        if tail:
            t_end = float(tail[-1].get("t_sim", 0.0))
        elif self.head:
            t_end = float(self.head[-1].get("t_sim", 0.0))
        t_cut = t_end - _TAIL_S
        last = tail[-1] if tail else (self.head[-1] if self.head else None)
        how = _end_info(last, ended=True)["how"] if last is not None else ""
        if how not in ("CRASH", "TIMEOUT"):
            self.head, self.mid, self.ring, self.n = [], [], deque(maxlen=_TAIL_N), 0
            return
        if t_end < (_HEAD_S + _TAIL_S) - 0.05:
            pack = (list(self.head) + tail, [], [], self.step)
        else:
            mid = [s for s in self.mid if float(s.get("t_sim", 0.0)) < t_cut]
            pack = (list(self.head), mid, tail, self.step)
        self.head, self.mid, self.ring, self.n = [], [], deque(maxlen=_TAIL_N), 0
        self._submit(pack)

    def _worker(self):
        while True:
            head, mid, tail, step = self.q.get()
            try:
                self._encode(head, mid, tail, step)
            except Exception as e:
                print("episode encode skip", e, flush=True)

    def _encode(self, head, mid, tail, step):
        t0 = time.time()
        st = {}
        sps = 0.0
        note = "env0 ep 1x / 15x / 1x"
        items = _compose_pack(head, mid, tail, st, sps, note, ended=True)
        if not items:
            return
        hud0 = items[0][0]
        h, w = hud0.shape[:2]
        w -= w % 2
        h -= h % 2
        ep_path = self.out / "clips" / ("ep_s%d.mp4" % int(step))
        ep_w = _open_mp4(ep_path, w, h, _PLAY_FPS)
        last = hud0
        n_ep = 0
        for hud, dup in items:
            frame = hud[:h, :w]
            last = frame
            for _ in range(int(dup)):
                _write_mp4_frame(ep_w, frame)
                n_ep += 1
        _close_mp4(ep_w)
        frame_p = self.out / "last_frame.png"
        if frame_p.is_symlink() or frame_p.exists():
            frame_p.unlink()
        cv2.imwrite(str(frame_p), last)
        dest = self.out / "last_ep.mp4"
        try:
            if dest.is_symlink() or dest.exists():
                dest.unlink()
            dest.symlink_to(ep_path.resolve())
        except OSError:
            shutil.copy2(ep_path, dest)
        clip_dir = self.out / "clips"
        keep = sorted(clip_dir.glob("ep_s*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        for old in keep[4:]:
            old.unlink(missing_ok=True)
        t_sim = 0.0
        last = _last_snap(tail, mid, head)
        if last is not None:
            t_sim = float(last.get("t_sim", 0.0))
        info = _end_info(last, ended=True) if last is not None else {}
        meta = {
            "step": int(step),
            "t_sim": t_sim,
            "head": len(head),
            "mid": len(mid),
            "tail": len(tail),
            "frames": n_ep,
            "play_s": n_ep / float(_PLAY_FPS),
            "ep_mp4": "last_ep.mp4",
            "mtime": time.time(),
            "enc_s": time.time() - t0,
            "how": info.get("how", ""),
            "cut": info.get("cut", 0),
            "cover": info.get("cover", 0.0),
            "hits": info.get("hits", 0),
            "n_free": info.get("n_free", 0),
            "path": info.get("path", 0.0),
            "plus_x": info.get("plus_x", 0.0),
            "n_movers": info.get("n_movers", 0),
        }
        (self.out / "watch_meta.json").write_text(json.dumps(meta))
        print(
            "episode mp4 %s cover=%.2f t=%.1fs path=%.1fm gap=%.2f pets=%d "
            "head=%d mid=%d tail=%d vid_frames=%d enc=%.2fs"
            % (
                info.get("how", "?"), info.get("cover", 0.0), t_sim,
                info.get("path", 0.0), info.get("plus_x", 0.0),
                info.get("n_movers", 0), len(head), len(mid), len(tail),
                n_ep, time.time() - t0,
            ),
            flush=True,
        )

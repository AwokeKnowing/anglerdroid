#!/usr/bin/env python3
"""On-screen LIVE house + progress HUD. CPU blit only."""
from __future__ import annotations

import json
import os
import time
from collections import deque
from pathlib import Path

import numpy as np

from visit_wander import FLOOR_X0, FLOOR_X1, FLOOR_Y0, FLOOR_Y1
from house_view import HouseView

_LEDGER = Path(__file__).resolve().parent / "explore" / "ledger.jsonl"


def _colorize_depth(d):
    x = np.clip(np.asarray(d, dtype=np.float32), 0.0, 1.0)
    rgb = np.zeros(x.shape + (3,), dtype=np.uint8)
    rgb[..., 0] = (40 + 200 * (1.0 - x)).astype(np.uint8)
    rgb[..., 1] = (30 + 180 * x).astype(np.uint8)
    rgb[..., 2] = (80 + 140 * (1.0 - np.abs(x - 0.45) * 2)).astype(np.uint8)
    rgb[x < 0.02] = (18, 18, 22)
    return rgb


def _paste(dst, src, x, y):
    h, w = src.shape[:2]
    H, W = dst.shape[:2]
    if x >= W or y >= H or x + w <= 0 or y + h <= 0:
        return
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    sx0, sy0 = x0 - x, y0 - y
    dst[y0:y1, x0:x1] = src[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)]


def _rolling(a, w=25):
    x = np.asarray(a, dtype=np.float32)
    if x.size == 0:
        return x
    if x.size < w:
        c = np.cumsum(x)
        return c / np.arange(1.0, x.size + 1.0, dtype=np.float32)
    out = np.empty_like(x)
    c = np.cumsum(x)
    out[w - 1:] = (c[w - 1:] - np.concatenate(([0.0], c[:-w]))) / float(w)
    out[:w - 1] = c[:w - 1] / np.arange(1.0, w, dtype=np.float32)
    return out


def _load_ledger(path=_LEDGER, max_rows=8000):
    if not path.is_file():
        return []
    lines = deque(maxlen=int(max_rows))
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    lines.append(line)
    except OSError:
        return []
    rows = []
    for line in lines:
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def _improve_chart(rows, w, h, mark_t=None):
    """Rolling-25 win / chord / path / crash — loss-style improvement panel."""
    import cv2
    box = np.zeros((h, w, 3), dtype=np.uint8)
    box[:] = (20, 22, 28)
    cv2.rectangle(box, (0, 0), (w - 1, h - 1), (55, 58, 66), 1)
    cv2.putText(
        box, "improve  roll25  (win / chord / path / crash)",
        (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 190), 1, cv2.LINE_AA,
    )
    if len(rows) < 2:
        cv2.putText(box, "waiting for episodes…", (14, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 110), 1)
        return box
    wins = _rolling([float(r.get("win_frac") or 0) for r in rows])
    chords = _rolling([float(r.get("mean_chord") or 0) for r in rows])
    paths = _rolling([float(r.get("path_m") or 0) for r in rows])
    crash = _rolling([1.0 if str(r.get("cut") or "") == "crash" else 0.0 for r in rows])
    n = int(wins.size)
    # downsample for blit
    max_pts = min(n, w - 56)
    if n > max_pts:
        idx = np.linspace(0, n - 1, max_pts).astype(np.int32)
        wins, chords, paths, crash = wins[idx], chords[idx], paths[idx], crash[idx]
    pad_l, pad_t, pad_b = 46, 28, 22
    iw, ih = w - pad_l - 10, h - pad_t - pad_b
    for g in range(5):
        gy = pad_t + int(ih * g / 4.0)
        cv2.line(box, (pad_l, gy), (w - 8, gy), (38, 40, 48), 1)
    if mark_t:
        xs_mark = None
        for i, r in enumerate(rows):
            if str(r.get("t") or "") >= mark_t:
                xs_mark = i
                break
        if xs_mark is not None and n > 1:
            frac = float(xs_mark) / float(max(n - 1, 1))
            mx = pad_l + int(frac * (iw - 1))
            cv2.line(box, (mx, pad_t), (mx, pad_t + ih), (90, 90, 70), 1, cv2.LINE_AA)
            cv2.putText(box, "20h", (max(pad_l, mx - 12), pad_t + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (160, 160, 120), 1)
    series = (
        ("win", wins, (255, 210, 70), 1.0),
        ("chord", chords, (80, 200, 255), 1.5),
        ("path", paths, (90, 200, 110), 8.0),
        ("crash", crash, (70, 90, 255), 1.0),
    )
    for name, vals, rgb, ymax in series:
        a = np.clip(np.asarray(vals, dtype=np.float32) / max(float(ymax), 1e-6), 0.0, 1.0)
        xs = pad_l + (np.linspace(0, iw - 1, a.size)).astype(np.int32)
        ys = (pad_t + ih - 1 - a * (ih - 1)).astype(np.int32)
        for i in range(1, a.size):
            cv2.line(box, (int(xs[i - 1]), int(ys[i - 1])),
                     (int(xs[i]), int(ys[i])), rgb, 1, cv2.LINE_AA)
    lx = pad_l
    for name, vals, rgb, ymax in series:
        last = float(vals[-1]) if len(vals) else 0.0
        lab = "%s %.2f" % (name, last)
        cv2.putText(box, lab, (lx, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, rgb, 1, cv2.LINE_AA)
        lx += 8 * len(lab) + 10
    cv2.putText(box, "n=%d" % n, (w - 70, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                (140, 140, 130), 1)
    return box


def _cov_img(cov, robot, people, trail, door_open, w=420, h=520, curve=None,
             beads=None, got=None, cont_ray=None, alts=None):
    ch, cw = cov.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    vis[:] = (22, 24, 28)
    floor = np.zeros((ch, cw, 3), dtype=np.uint8)
    floor[:] = (36, 40, 46)
    floor[cov > 0] = (70, 170, 90)
    import cv2
    scaled = cv2.resize(floor, (w - 16, h - 70), interpolation=cv2.INTER_NEAREST)
    vis[50:50 + scaled.shape[0], 8:8 + scaled.shape[1]] = scaled
    def xy_to_px(x, y):
        px = 8 + int((float(x) - FLOOR_X0) / (FLOOR_X1 - FLOOR_X0) * (w - 16))
        py = 50 + int((FLOOR_Y1 - float(y)) / (FLOOR_Y1 - FLOOR_Y0) * (h - 70))
        return px, py
    if trail:
        for a, b in zip(trail[:-1], trail[1:]):
            p0, p1 = xy_to_px(*a), xy_to_px(*b)
            cv2.line(vis, p0, p1, (240, 200, 60), 1, cv2.LINE_AA)
    for alt in alts or []:
        if alt is None or len(alt) < 2:
            continue
        pts = [xy_to_px(float(p[0]), float(p[1])) for p in alt]
        for p0, p1 in zip(pts[:-1], pts[1:]):
            cv2.line(vis, p0, p1, (50, 90, 110), 1, cv2.LINE_AA)
    if curve is not None and len(curve) >= 2:
        pts = [xy_to_px(float(p[0]), float(p[1])) for p in curve]
        for p0, p1 in zip(pts[:-1], pts[1:]):
            cv2.line(vis, p0, p1, (80, 220, 255), 2, cv2.LINE_AA)
    if beads is not None:
        for i, b in enumerate(beads):
            col = (90, 255, 140) if got is not None and int(got[i]) else (70, 90, 255)
            cv2.circle(vis, xy_to_px(float(b[0]), float(b[1])), 5, col, -1)
    if cont_ray is not None and len(cont_ray) == 2:
        p0, p1 = xy_to_px(*cont_ray[0]), xy_to_px(*cont_ray[1])
        cv2.arrowedLine(vis, p0, p1, (80, 90, 255), 2, tipLength=0.18)
    if robot is not None:
        px, py = xy_to_px(robot[0], robot[1])
        cv2.circle(vis, (px, py), 6, (60, 140, 255), -1)
        c, s = float(np.cos(robot[2])), float(np.sin(robot[2]))
        cv2.line(vis, (px, py), (int(px + 14 * c), int(py - 14 * s)), (255, 230, 80), 2)
    for p in people or []:
        px, py = xy_to_px(p["xy"][0], p["xy"][1])
        talking = bool(p.get("talking") or p.get("chat_left", 0) > 0)
        col = (80, 220, 120) if talking else (180, 110, 110)
        cv2.circle(vis, (px, py), 7, col, -1)
    title = "coverage  door=%s" % ("OPEN" if door_open else "shut")
    cv2.putText(vis, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 220), 1, cv2.LINE_AA)
    return vis


class ExploreDash:
    def __init__(self, wall_boxes, clutter=None, people=None):
        os.environ.setdefault("DISPLAY", ":1")
        floor = (FLOOR_X0, FLOOR_X1, FLOOR_Y0, FLOOR_Y1)
        self.live = HouseView(
            wall_boxes, floor, people=people, clutter=clutter,
            caption="Kevin insect LIVE", size=(820, 540), location=(20, 40),
        )
        import pyglet
        self._pyglet = pyglet
        self.w, self.h = 1280, 780
        self.hud = pyglet.window.Window(
            width=self.w, height=self.h, caption="insect explore HUD", vsync=False,
        )
        try:
            self.hud.set_location(860, 30)
        except Exception:
            pass
        self.last = 0.0
        self.rew = deque(maxlen=360)
        self.nov = deque(maxlen=360)
        self.closed = False
        self._sprite = None
        self._ledger_rows = _load_ledger()
        self._ledger_mtime = 0.0
        self._ledger_check = 0.0
        self._run_mark = "2026-09-11T23:17"
        print("explore dash LIVE+HUD DISPLAY=%s" % os.environ.get("DISPLAY", ""), flush=True)

    def close(self):
        self.closed = True
        try:
            self.live.close()
        except Exception:
            pass
        try:
            self.hud.close()
        except Exception:
            pass

    def present(self, env, note="", rt=0.0, extras=None, min_dt=1.0 / 12.0):
        if self.closed:
            return
        now = time.monotonic()
        if now - self.last < min_dt:
            return
        self.last = now
        extras = extras or {}
        try:
            self.live.present(
                (env.x, env.y), env.yaw,
                trail=env.trail[-80:],
                people=env.people_draw(),
                chat=env.chat,
                note=note,
                rt=rt,
                min_dt=0.0,
            )
        except Exception as e:
            print("live skip:", type(e).__name__, e, flush=True)
        info = env.last_info or {}
        self.rew.append(float(info.get("reward", 0.0)))
        self.nov.append(float(extras.get("novelty", 0.0)))
        now_w = time.time()
        if now_w - self._ledger_check > 2.0:
            self._ledger_check = now_w
            try:
                mt = _LEDGER.stat().st_mtime
            except OSError:
                mt = 0.0
            if mt > self._ledger_mtime:
                self._ledger_rows = _load_ledger()
                self._ledger_mtime = mt
        canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        canvas[:] = (16, 17, 22)
        depth = _colorize_depth(getattr(env, "last_depth", np.zeros((48, 80))))
        import cv2
        depth = cv2.resize(depth, (400, 240), interpolation=cv2.INTER_NEAREST)
        ego = _colorize_depth(getattr(env, "last_ego", np.zeros((48, 64))))
        ego = cv2.resize(ego, (320, 240), interpolation=cv2.INTER_NEAREST)
        _paste(canvas, depth, 16, 70)
        _paste(canvas, ego, 430, 70)
        cov = _cov_img(
            env.cov, (env.x, env.y, env.yaw), env.people_draw(),
            env.trail[-120:], env.door_open,
            curve=getattr(env, "curve_poly", None),
            beads=getattr(env, "curve_beads", None),
            got=getattr(env, "curve_got", None),
            cont_ray=getattr(env, "curve_cont_ray", None),
            alts=getattr(env, "prim_alts", None),
            w=420, h=400,
        )
        _paste(canvas, cov, 770, 50)

        def spark(vals, x, y, w, h, rgb):
            box = np.zeros((h, w, 3), dtype=np.uint8)
            box[:] = (28, 30, 36)
            if len(vals) >= 2:
                a = np.array(vals, dtype=np.float32)
                lo, hi = float(a.min()), float(a.max())
                if hi - lo < 1e-6:
                    hi = lo + 1.0
                xs = np.linspace(0, w - 1, len(a)).astype(np.int32)
                ys = (h - 2 - (a - lo) / (hi - lo) * (h - 4)).astype(np.int32)
                for i in range(1, len(xs)):
                    cv2.line(box, (int(xs[i - 1]), int(ys[i - 1])), (int(xs[i]), int(ys[i])), rgb, 1, cv2.LINE_AA)
            _paste(canvas, box, x, y)

        spark(self.rew, 16, 318, 360, 72, (90, 200, 255))
        spark(self.nov, 390, 318, 360, 72, (255, 180, 70))
        chart = _improve_chart(self._ledger_rows, 740, 210, mark_t=self._run_mark)
        _paste(canvas, chart, 16, 400)
        vis = getattr(env, "visit", None)
        if vis is not None:
            v = np.clip(np.asarray(vis, dtype=np.float32) / 18.0, 0.0, 1.0)
            heat = np.stack(
                [(40 + 210 * v), (30 + 90 * (1.0 - v)), (20 + 50 * (1.0 - v))],
                axis=-1,
            ).astype(np.uint8)
            heat = cv2.resize(heat, (160, 160), interpolation=cv2.INTER_NEAREST)
            cv2.drawMarker(heat, (80, 80), (255, 240, 80), cv2.MARKER_CROSS, 10, 1)
            yaw = float(getattr(env, "yaw", 0.0))
            hx = int(80 + 36 * np.cos(yaw))
            hy = int(80 - 36 * np.sin(yaw))
            cv2.arrowedLine(heat, (80, 80), (hx, hy), (255, 240, 80), 1, tipLength=0.25)
            _paste(canvas, heat, 770, 460)
            cv2.putText(
                canvas, "visit 3.2m world", (770, 454),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 170), 1,
            )
        lines = [
            "INSECT EXPLORE  rt=%.1fx  %s" % (rt, note),
            "ep=%s step=%s t=%.0fs  xy=(%.2f,%.2f) yaw=%.0f  plus_x=%.2f  door=%s"
            % (env.ep, env.steps, env.t_sim, env.x, env.y, np.degrees(env.yaw),
               float(info.get("plus_x", 0)), "OPEN" if env.door_open else "shut"),
            "rooms=%s nrm=%s cells=%s path=%.1fm  vw %s path5=%.2f throt=%.2f hot=%.2f v=%.2f w=%.2f ms=%.1f win=%s/%s"
            % (info.get("rooms", ""), info.get("n_rooms", 0), info.get("cov_cells", 0),
               info.get("path_m", 0), info.get("last_seg", "") or "run",
               info.get("curve_chord", 0),
               info.get("curve_cont", 0),
               info.get("prim_hot", 0),
               info.get("v", 0), info.get("w", 0),
               info.get("prim_ms", 0),
               info.get("curve_wins", 0),
               int(info.get("curve_wins", 0)) + int(info.get("curve_losses", 0))),
            extras.get("status", ""),
            extras.get("eval", ""),
        ]
        y = 16
        for i, line in enumerate(lines):
            if not line:
                continue
            cv2.putText(
                canvas, str(line)[:140], (16, y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48 if i else 0.58,
                (235, 235, 225) if i else (255, 230, 120), 1, cv2.LINE_AA,
            )
            y += 22
        cv2.putText(canvas, "RS2 depth", (16, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 170), 1)
        cv2.putText(canvas, "ego height", (430, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 170), 1)
        cv2.putText(canvas, "reward", (16, 312), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 180, 210), 1)
        cv2.putText(canvas, "novelty", (390, 312), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (210, 170, 90), 1)
        # pyglet ImageData is bottom-up
        rgb = np.ascontiguousarray(canvas[::-1])
        img = self._pyglet.image.ImageData(self.w, self.h, "RGB", rgb.tobytes(), pitch=self.w * 3)
        try:
            self.hud.switch_to()
            self.hud.clear()
            self.hud.dispatch_events()
            img.blit(0, 0)
            self.hud.flip()
        except Exception as e:
            print("hud skip:", type(e).__name__, e, flush=True)
            self.closed = True

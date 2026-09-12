"""Cheap 3D casita view. No Warp, no Newton log_state, no CUDA sync.

High bird's-eye of the whole house so Kevin, walls, and people stay readable.
"""
from __future__ import annotations

import math
import os
import time
from collections import deque

import numpy as np


_VERT = """
#version 330
uniform WindowBlock {
    mat4 projection;
    mat4 view;
} window;
in vec3 position;
in vec4 colors;
out vec4 v_color;
void main() {
    gl_Position = window.projection * window.view * vec4(position, 1.0);
    v_color = colors;
}
"""

_FRAG = """
#version 330
in vec4 v_color;
out vec4 fragColor;
void main() {
    fragColor = v_color;
}
"""


def _box_mesh(cx, cy, cz, hx, hy, hz, yaw=0.0, rgba=(0.85, 0.80, 0.70, 1.0)):
    """12 triangles, world XY, Z up."""
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    corners = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                lx, ly, lz = sx * hx, sy * hy, sz * hz
                wx = cx + c * lx - s * ly
                wy = cy + s * lx + c * ly
                wz = cz + lz
                corners.append((wx, wy, wz))
    # index: x y z as bit 0=x, 1=y, 2=z → pack (sx+1)/2 etc. easier with list
    # corners order: sx,sy,sz nested z fastest
    def idx(ix, iy, iz):
        return (ix * 4) + (iy * 2) + iz

    faces = (
        (idx(0, 0, 0), idx(1, 0, 0), idx(1, 1, 0), idx(0, 0, 0), idx(1, 1, 0), idx(0, 1, 0)),  # z-
        (idx(0, 0, 1), idx(0, 1, 1), idx(1, 1, 1), idx(0, 0, 1), idx(1, 1, 1), idx(1, 0, 1)),  # z+
        (idx(0, 0, 0), idx(0, 0, 1), idx(1, 0, 1), idx(0, 0, 0), idx(1, 0, 1), idx(1, 0, 0)),  # y-
        (idx(0, 1, 0), idx(1, 1, 0), idx(1, 1, 1), idx(0, 1, 0), idx(1, 1, 1), idx(0, 1, 1)),  # y+
        (idx(0, 0, 0), idx(0, 1, 0), idx(0, 1, 1), idx(0, 0, 0), idx(0, 1, 1), idx(0, 0, 1)),  # x-
        (idx(1, 0, 0), idx(1, 0, 1), idx(1, 1, 1), idx(1, 0, 0), idx(1, 1, 1), idx(1, 1, 0)),  # x+
    )
    pos = []
    col = []
    r, g, b, a = rgba
    # shade faces a bit so walls read as 3D from overhead
    shade = (0.72, 0.78, 0.85, 0.90, 0.65, 1.0)
    for fi, tri in enumerate(faces):
        sh = shade[fi]
        for vi in tri:
            pos.extend(corners[vi])
            col.extend((r * sh, g * sh, b * sh, a))
    return pos, col


def _disk_mesh(cx, cy, z, radius, rgba, n=18):
    pos = []
    col = []
    r, g, b, a = rgba
    for i in range(n):
        a0 = 2.0 * math.pi * i / n
        a1 = 2.0 * math.pi * (i + 1) / n
        pos.extend((cx, cy, z, cx + radius * math.cos(a0), cy + radius * math.sin(a0), z,
                    cx + radius * math.cos(a1), cy + radius * math.sin(a1), z))
        col.extend((r, g, b, a) * 3)
    return pos, col


class HouseView:
    """Whole-casita bird's-eye. Present is CPU-only."""

    def __init__(
        self,
        wall_boxes,
        floor_aabb,
        people=None,
        clutter=None,
        caption="Kevin LIVE",
        size=(800, 520),
        location=None,
    ):
        os.environ.setdefault("DISPLAY", ":1")
        import pyglet
        from pyglet import gl
        from pyglet.graphics.shader import Shader, ShaderProgram
        from pyglet.math import Mat4, Vec3

        self._pyglet = pyglet
        self._gl = gl
        self._Mat4 = Mat4
        self._Vec3 = Vec3
        self.closed = False
        self.last = 0.0
        self.w, self.h = int(size[0]), int(size[1])
        self.win = pyglet.window.Window(
            width=self.w, height=self.h, caption=caption, vsync=False,
        )
        self.win.set_minimum_size(360, 240)
        if location is not None:
            try:
                self.win.set_location(int(location[0]), int(location[1]))
            except Exception:
                pass
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glClearColor(0.10, 0.12, 0.16, 1.0)
        self.prog = ShaderProgram(Shader(_VERT, "vertex"), Shader(_FRAG, "fragment"))
        self.batch = pyglet.graphics.Batch()
        self._people = list(people or [])
        self._labels = []
        self._rt = 0.0
        self._note = ""
        self._chat = ""
        self._status = pyglet.text.Label(
            caption, font_name="DejaVu Sans", font_size=11,
            x=8, y=self.h - 18, anchor_x="left", anchor_y="top",
            color=(230, 230, 220, 255),
        )
        self._bubble = pyglet.text.Label(
            "", font_name="DejaVu Sans", font_size=13,
            x=self.w * 0.5, y=self.h - 44, anchor_x="center", anchor_y="top",
            color=(255, 230, 120, 255),
        )

        x0, x1, y0, y1 = floor_aabb
        pos, col = [], []
        zf = -0.02
        pos.extend((x0, y0, zf, x1, y0, zf, x1, y1, zf, x0, y0, zf, x1, y1, zf, x0, y1, zf))
        col.extend((0.18, 0.20, 0.22, 1.0) * 6)
        for b in wall_boxes or []:
            p, c = _box_mesh(
                float(b["cx"]), float(b["cy"]), float(b.get("cz", float(b.get("hz", 1.1)))),
                float(b["hx"]), float(b["hy"]), float(b.get("hz", 1.1)),
                yaw=float(b.get("yaw", 0.0)),
                rgba=(0.82, 0.76, 0.62, 1.0),
            )
            pos.extend(p)
            col.extend(c)
        for b in clutter or []:
            rgba = tuple(b.get("rgba") or (0.62, 0.40, 0.22, 1.0))
            if len(rgba) == 3:
                rgba = (rgba[0], rgba[1], rgba[2], 1.0)
            p, c = _box_mesh(
                float(b["cx"]), float(b["cy"]), float(b.get("cz", float(b.get("hz", 0.35)))),
                float(b["hx"]), float(b["hy"]), float(b.get("hz", 0.35)),
                yaw=float(b.get("yaw", 0.0)),
                rgba=rgba,
            )
            pos.extend(p)
            col.extend(c)
        n = len(pos) // 3
        self.static = self.prog.vertex_list(
            n, gl.GL_TRIANGLES, batch=self.batch,
            position=("f", pos),
            colors=("f", col),
        )
        self._dyn = None
        self._n_dyn = 0
        print(
            "house view %dx%d %s overlays=off DISPLAY=%s"
            % (self.w, self.h, caption, os.environ.get("DISPLAY", "")),
            flush=True,
        )

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.win.close()
        except Exception:
            pass

    def _set_dyn(self, pos, col):
        n = len(pos) // 3
        gl = self._gl
        if self._dyn is None or n != self._n_dyn:
            if self._dyn is not None:
                try:
                    self._dyn.delete()
                except Exception:
                    pass
            self._dyn = self.prog.vertex_list(
                n, gl.GL_TRIANGLES, batch=self.batch,
                position=("f", pos),
                colors=("f", col),
            )
            self._n_dyn = n
        else:
            self._dyn.position[:] = pos
            self._dyn.colors[:] = col

    def present(self, pos_xy, yaw, trail=None, people=None, chat="", note="", rt=0.0, min_dt=1.0 / 20.0):
        if self.closed:
            return
        now = time.monotonic()
        if now - self.last < min_dt:
            return
        self.last = now
        try:
            self.win.switch_to()
        except Exception:
            self.close()
            return
        px, py = float(pos_xy[0]), float(pos_xy[1])
        pos, col = [], []
        # robot body
        p, c = _box_mesh(px, py, 0.12, 0.165, 0.165, 0.08, yaw=float(yaw), rgba=(0.20, 0.55, 0.95, 1.0))
        pos.extend(p)
        col.extend(c)
        # nose
        c_, s_ = math.cos(float(yaw)), math.sin(float(yaw))
        nx, ny = px + 0.22 * c_, py + 0.22 * s_
        p, c = _box_mesh(nx, ny, 0.16, 0.04, 0.05, 0.04, yaw=float(yaw), rgba=(0.95, 0.85, 0.20, 1.0))
        pos.extend(p)
        col.extend(c)
        for pe in (people if people is not None else self._people):
            mood = bool(pe.get("mood", True))
            rgba = (0.35, 0.85, 0.45, 1.0) if mood else (0.75, 0.45, 0.45, 1.0)
            p, c = _disk_mesh(float(pe["xy"][0]), float(pe["xy"][1]), 0.04, float(pe.get("r", 0.28)), rgba)
            pos.extend(p)
            col.extend(c)
            p, c = _box_mesh(
                float(pe["xy"][0]), float(pe["xy"][1]), 0.55,
                0.12, 0.12, 0.45, yaw=0.0, rgba=rgba,
            )
            pos.extend(p)
            col.extend(c)
        if trail:
            for i in range(1, len(trail)):
                a = trail[i - 1]
                b = trail[i]
                # thin quad along segment
                dx, dy = float(b[0] - a[0]), float(b[1] - a[1])
                L = math.hypot(dx, dy) or 1e-6
                tx, ty = -dy / L * 0.03, dx / L * 0.03
                z = 0.03
                pos.extend((
                    a[0] + tx, a[1] + ty, z, a[0] - tx, a[1] - ty, z, b[0] - tx, b[1] - ty, z,
                    a[0] + tx, a[1] + ty, z, b[0] - tx, b[1] - ty, z, b[0] + tx, b[1] + ty, z,
                ))
                col.extend((0.95, 0.82, 0.25, 0.9) * 6)
        self._set_dyn(pos, col)

        Mat4, Vec3 = self._Mat4, self._Vec3
        aspect = self.win.width / max(1, self.win.height)
        self.win.projection = Mat4.perspective_projection(aspect, 0.2, 80.0, fov=50)
        # slight south bias so walls have a face; still almost nadir
        self.win.view = Mat4.look_at(
            Vec3(0.12, -2.6, 21.0),
            Vec3(0.12, 0.4, 0.0),
            Vec3(0.0, 0.0, 1.0),
        )
        self.win.clear()
        self.win.dispatch_events()
        self.batch.draw()
        self._status.text = "rt=%.1fx  (%.2f, %.2f)  %s" % (float(rt), px, py, note or "")
        self._status.y = self.win.height - 18
        self._status.draw()
        self._bubble.text = chat or ""
        self._bubble.x = self.win.width * 0.5
        self._bubble.y = self.win.height - 40
        self._bubble.draw()
        self.win.flip()


class SampleReplay:
    """1× wall-clock playback of the last 20 s of sim. Never sleeps the sim."""

    SPAN = 20.0
    REC_DT = 0.10

    def __init__(self, wall_boxes, floor_aabb, people=None, clutter=None):
        self.view = HouseView(
            wall_boxes,
            floor_aabb,
            people=people,
            clutter=clutter,
            caption="Kevin 1x (last 20s sim)",
            size=(720, 460),
            location=(860, 50),
        )
        self.buf = deque()
        self._last_rec = -1e9
        self.clip = []
        self.clip_wall0 = None
        self.closed = False

    def close(self):
        self.closed = True
        try:
            self.view.close()
        except Exception:
            pass

    def record(self, t_sim, pos, yaw, chat, note, trail):
        if self.closed:
            return
        t_sim = float(t_sim)
        if t_sim - self._last_rec < self.REC_DT:
            return
        self._last_rec = t_sim
        tr = list(trail[-24:]) if trail else []
        self.buf.append((
            t_sim, float(pos[0]), float(pos[1]), float(yaw),
            chat or "", note or "", tr,
        ))
        cut = t_sim - 45.0
        while self.buf and self.buf[0][0] < cut:
            self.buf.popleft()

    def present(self, min_dt=1.0 / 15.0):
        if self.closed:
            return
        now = time.monotonic()
        need = (
            not self.clip
            or self.clip_wall0 is None
            or (now - self.clip_wall0) >= self.SPAN
        )
        if need and len(self.buf) >= 8:
            t_end = self.buf[-1][0]
            t_start = t_end - self.SPAN
            self.clip = [s for s in self.buf if s[0] >= t_start]
            self.clip_wall0 = now
            if self.clip:
                print(
                    "1x sample clip sim=%.1f..%.1f n=%d"
                    % (self.clip[0][0], self.clip[-1][0], len(self.clip)),
                    flush=True,
                )
        if not self.clip:
            if self.buf:
                s = self.buf[-1]
                self.view.present(
                    (s[1], s[2]), s[3], trail=s[6], chat=s[4],
                    note="recording 1x sample…", rt=1.0, min_dt=min_dt,
                )
            return
        elapsed = now - self.clip_wall0
        t0 = self.clip[0][0]
        t_play = t0 + elapsed
        s = self.clip[-1]
        for q in self.clip:
            if q[0] >= t_play:
                s = q
                break
        remain = max(0.0, self.SPAN - elapsed)
        self.view.present(
            (s[1], s[2]), s[3], trail=s[6], chat=s[4],
            note="1x sample t=%.1fs next=%.0fs  %s" % (s[0], remain, s[5]),
            rt=1.0, min_dt=min_dt,
        )

"""Cheap doorway wander. Visit grid is fused-pose xy, not chassis truth.

The first room is one open cell. Wall-follow loops it. This picks a heading
through a constriction (doorway) into low-visit floor, then drives. A wall in
the forward window forces v=0 until yaw opens that window. Yaw sign is the
turn that increases box clearance.
"""
from __future__ import annotations

import heapq
import math
from collections import deque

import numpy as np

# House floor AABB from the repaired STL.
FLOOR_X0, FLOOR_X1 = -3.55, 3.75
FLOOR_Y0, FLOOR_Y1 = -7.45, 7.45
CELL = 0.32

# Rooms are slabs between the interior partitions, for the coverage report.
ROOMS = (
    ("sw", -3.55, -1.05, -7.45, -2.65),
    ("south", -1.05, 3.75, -7.45, -3.55),
    ("west_mid", -3.55, -1.00, -2.65, 1.15),
    ("center", -1.00, 1.25, -3.55, 1.70),
    ("east_s", 1.25, 3.75, -3.55, 0.05),
    ("east_mid", 1.15, 3.75, 0.05, 1.75),
    ("north_w", -3.55, 0.15, 1.15, 5.50),
    ("north_e", 0.15, 3.75, 1.75, 5.50),
    ("far_north", -3.55, 3.75, 5.50, 7.45),
)


def room_of(x, y):
    for name, x0, x1, y0, y1 in ROOMS:
        if x0 <= x < x1 and y0 <= y < y1:
            return name
    return "other"


def _norm(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def _point_in_boxes(px, py, boxes, pad: float) -> bool:
    for b in boxes:
        yaw = float(b["yaw"])
        cb, sb = math.cos(yaw), math.sin(yaw)
        dx = float(px) - float(b["cx"])
        dy = float(py) - float(b["cy"])
        lx = cb * dx + sb * dy
        ly = -sb * dx + cb * dy
        if abs(lx) <= float(b["hx"]) + pad and abs(ly) <= float(b["hy"]) + pad:
            return True
    return False


def _sample_clear(px, py, yaw, boxes, pad: float = 0.01) -> float:
    """Min signed clearance of body samples to wall boxes. Negative if inside."""
    c, s = math.cos(yaw), math.sin(yaw)
    samples = (
        (0.15, 0.0), (0.15, 0.13), (0.15, -0.13),
        (0.05, 0.16), (0.05, -0.16),
        (-0.05, 0.16), (-0.05, -0.16),
        (-0.20, 0.13), (-0.20, -0.13), (-0.22, 0.0),
    )
    best = 9.0
    for lx, ly in samples:
        wx = px + c * lx - s * ly
        wy = py + s * lx + c * ly
        d = 9.0
        inside = False
        for b in boxes:
            yawb = float(b["yaw"])
            cb, sb = math.cos(yawb), math.sin(yawb)
            dx = wx - float(b["cx"])
            dy = wy - float(b["cy"])
            bx = cb * dx + sb * dy
            by = -sb * dx + cb * dy
            hx = float(b["hx"]) + pad
            hy = float(b["hy"]) + pad
            # outside distance to the rectangle
            ox = abs(bx) - hx
            oy = abs(by) - hy
            if ox <= 0.0 and oy <= 0.0:
                inside = True
                d = min(d, -min(-ox, -oy))
                break
            dd = math.hypot(max(ox, 0.0), max(oy, 0.0))
            if dd < d:
                d = dd
        if inside:
            best = min(best, -0.05)
        else:
            best = min(best, d)
    return best


def _yaw_sign_toward(pos, yaw, target, boxes) -> int:
    """Turn toward target. Only reject a side that actually sweeps into a box."""
    err = _norm(float(target) - float(yaw))
    if abs(err) < 0.05:
        return 1 if err >= 0.0 else -1
    prefer = 1 if err > 0.0 else -1
    px, py = float(pos[0]), float(pos[1])
    cur = _sample_clear(px, py, yaw, boxes)

    def score(sg):
        ny = yaw + sg * 0.30
        clr = _sample_clear(px, py, ny, boxes)
        # Inside a box, or a turn that closes a nearby wall, is illegal.
        if clr < 0.02:
            return -100.0 + clr
        if clr < 0.10 and clr < cur - 0.02:
            return -40.0 + clr
        toward = 1.2 if sg == prefer else 0.0
        return clr + toward

    left = score(1)
    right = score(-1)
    if left < -20.0 and right < -20.0:
        return 0
    if prefer == 1 and left >= -20.0:
        return 1
    if prefer == -1 and right >= -20.0:
        return -1
    return 1 if left >= right else -1


class VisitWander:
    def __init__(self, boxes):
        self.boxes = list(boxes or [])
        self.cs = CELL
        self.x0, self.x1 = FLOOR_X0, FLOOR_X1
        self.y0, self.y1 = FLOOR_Y0, FLOOR_Y1
        self.nx = int(math.ceil((self.x1 - self.x0) / self.cs))
        self.ny = int(math.ceil((self.y1 - self.y0) / self.cs))
        self.blocked = np.zeros((self.ny, self.nx), dtype=bool)
        self.free = np.zeros((self.ny, self.nx), dtype=bool)
        self.portal = np.zeros((self.ny, self.nx), dtype=bool)
        # Robot half-width so the planned path does not shave a box.
        pad = 0.17
        for iy in range(self.ny):
            cy = self.y0 + (iy + 0.5) * self.cs
            for ix in range(self.nx):
                cx = self.x0 + (ix + 0.5) * self.cs
                hit = _point_in_boxes(cx, cy, self.boxes, pad)
                self.blocked[iy, ix] = hit
                self.free[iy, ix] = not hit
        self._mark_portals()
        self.vis = np.zeros((self.ny, self.nx), dtype=np.float32)
        self.stamps = 0
        self.recoveries = 0
        self.stuck_recoveries = 0
        self.goal = None
        self.heading = None
        self.path = None
        self.landing = None
        self.plan_t = -99.0
        self.commit_xy = None
        self._last_portal = None
        self.recent = deque(maxlen=80)
        self.recent_t = deque(maxlen=80)
        self.fail_until = {}
        self.plan_note = "boot"
        self.n_free = int(self.free.sum())
        self.origin = None
        self.left_first = False
        self.wps = [
            (-2.30, -4.50),  # north in SW; avoid east open-floor furniture pin
            (-1.15, -3.60),  # approach pad: door-x, south of SW portal
            (-1.15, -3.05),  # doorway open cell (portal)
            (-0.50, -2.00),  # through into center / west_mid
            (0.20, -1.40),   # center
            (1.40, 0.30),
            (-1.50, 2.10),
            (0.20, 5.90),
        ]
        self.wp_i = 0
        self.spin_s = 0.0
        self.bad_heading = None
        self.escape_left = 0.0
        self.escape_sign = 1
        self.escape_phase = ""  # reverse | yaw | surge
        self._door_commit = False
        self.no_progress_s = 0.0
        self._last_prog_xy = None
        self.tip_s = 0.0
        self._wp_best_dist = None
        self._wp_still_s = 0.0
        self.wp_skips = 0
        self._wp_skip_cool = -99.0

    def _mark_portals(self):
        """A free cell squeezed by walls on both sides is a doorway."""
        reach = 3  # 3 * 0.32 m ~ 1.0 m; catches ~0.7-1.8 m gaps
        ny, nx = self.ny, self.nx
        blk = self.blocked

        def side(ix, iy, dx, dy):
            for k in range(1, reach + 1):
                x2, y2 = ix + dx * k, iy + dy * k
                if not (0 <= x2 < nx and 0 <= y2 < ny) or blk[y2, x2]:
                    return True
            return False

        for iy in range(ny):
            for ix in range(nx):
                if not self.free[iy, ix]:
                    continue
                ew = side(-1 if False else 1, iy, 1, 0) and side(ix, iy, -1, 0)
                # fix accidental call — rewrite cleanly below
        # rewrite properly
        self.portal[:] = False
        for iy in range(ny):
            for ix in range(nx):
                if not self.free[iy, ix]:
                    continue
                squeeze_x = side(ix, iy, 1, 0) and side(ix, iy, -1, 0)
                squeeze_y = side(ix, iy, 0, 1) and side(ix, iy, 0, -1)
                if squeeze_x or squeeze_y:
                    self.portal[iy, ix] = True

    def _idx(self, x, y):
        ix = int((float(x) - self.x0) / self.cs)
        iy = int((float(y) - self.y0) / self.cs)
        if 0 <= ix < self.nx and 0 <= iy < self.ny:
            return ix, iy
        return None

    def _clamp_idx(self, x, y):
        ix = int((float(x) - self.x0) / self.cs)
        iy = int((float(y) - self.y0) / self.cs)
        ix = max(0, min(self.nx - 1, ix))
        iy = max(0, min(self.ny - 1, iy))
        return ix, iy

    def _cell_xy(self, ix, iy):
        return self.x0 + (ix + 0.5) * self.cs, self.y0 + (iy + 0.5) * self.cs

    def nearest_free(self, x, y):
        hit = self._idx(x, y)
        if hit and self.free[hit[1], hit[0]]:
            return hit
        best = None
        best_d = 1e9
        # local search
        if hit:
            ix0, iy0 = hit
        else:
            ix0 = int(np.clip((float(x) - self.x0) / self.cs, 0, self.nx - 1))
            iy0 = int(np.clip((float(y) - self.y0) / self.cs, 0, self.ny - 1))
        for r in range(1, 8):
            for iy in range(max(0, iy0 - r), min(self.ny, iy0 + r + 1)):
                for ix in range(max(0, ix0 - r), min(self.nx, ix0 + r + 1)):
                    if not self.free[iy, ix]:
                        continue
                    cx, cy = self._cell_xy(ix, iy)
                    d = (cx - x) ** 2 + (cy - y) ** 2
                    if d < best_d:
                        best_d = d
                        best = (ix, iy)
            if best is not None:
                return best
        return best


    def _approach_pad(self, gx, gy):
        """Free cell ~0.5 m south of doorway — approach before crossing.

        Pad MUST sit in the open cell aligned with the door opening (door-x
        column). A lateral pad (e.g. x=-0.8 for a portal near x=-1.2) aims
        along the partition face into the wall-parallel corridor and never
        crosses. Prefer dx=0; allow at most ~0.20 m lateral. Never return a
        portal/jamb cell. Geometric fallback stays on door-x (no east snap).
        """
        gx, gy = float(gx), float(gy)
        gix, giy = self._clamp_idx(gx, gy)
        best = None
        best_score = 1e9

        def _consider(dx_lo, dx_hi, lat_max, lat_w):
            nonlocal best, best_score
            for dy in range(1, 6):  # cells south (lower y / lower iy)
                for dx in range(dx_lo, dx_hi):
                    x2, y2 = gix + dx, giy - dy
                    if not (0 <= x2 < self.nx and 0 <= y2 < self.ny):
                        continue
                    if not self.free[y2, x2]:
                        continue
                    if self.portal[y2, x2]:
                        continue
                    cx, cy = self._cell_xy(x2, y2)
                    south = gy - cy
                    lat = abs(cx - gx)
                    if south < 0.28 or south > 1.00:
                        continue
                    if lat > lat_max:
                        continue
                    score = abs(south - 0.55) + lat_w * lat
                    if score < best_score:
                        best_score = score
                        best = (cx, cy)

        # Pass 1: strict door column (aligned with opening).
        _consider(0, 1, 0.12, 6.0)
        if best is not None:
            return best
        # Pass 2: tiny lateral only if door column is blocked/portal.
        _consider(-1, 2, 0.20, 5.0)
        if best is not None:
            return best
        # Geometric fallback: stay on door x — never nearest_free east/west snap.
        return (gx, gy - 0.55)

    def _nearest_portal_xy(self, cx, cy, max_r=2.2):
        """Closest portal cell center within max_r, or None."""
        best = None
        best_d = 1e9
        ix0, iy0 = self._clamp_idx(cx, cy)
        reach = max(1, int(max_r / max(self.cs, 1e-6)) + 1)
        for dy in range(-reach, reach + 1):
            for dx in range(-reach, reach + 1):
                x2, y2 = ix0 + dx, iy0 + dy
                if not (0 <= x2 < self.nx and 0 <= y2 < self.ny):
                    continue
                if not self.portal[y2, x2]:
                    continue
                px, py = self._cell_xy(x2, y2)
                d = math.hypot(px - cx, py - cy)
                if d < best_d and d <= max_r:
                    best_d = d
                    best = (px, py)
        return best

    def _door_normal_heading(self, cx, cy, portal_xy):
        """Heading through the portal along its opening normal (not corridor).

        For the SW partition door the opening is N/S: prefer +Y when the
        robot is south of the jamb (and -Y if north). Lateral aim is capped
        so a false-open east corridor cannot steal the surge.
        """
        px, py = float(portal_xy[0]), float(portal_xy[1])
        # Past-portal aim point so surge clears the jamb instead of stopping on it.
        if cy <= py:
            aim = (px, py + 0.55)
            preferred = 0.5 * math.pi  # +Y
        else:
            aim = (px, py - 0.55)
            preferred = -0.5 * math.pi
        raw = math.atan2(aim[1] - cy, aim[0] - cx)
        # If nearly on door-x, lock to door normal; else blend toward normal.
        if abs(cx - px) <= 0.28:
            return preferred
        # Cap lateral component: reject headings more than ~35 deg off normal.
        if abs(_norm(raw - preferred)) <= 0.60:
            return raw
        return preferred

    def stamp(self, x, y):
        hit = self._idx(x, y)
        if hit is None:
            return
        ix, iy = hit
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                x2, y2 = ix + dx, iy + dy
                if 0 <= x2 < self.nx and 0 <= y2 < self.ny and self.free[y2, x2]:
                    w = 1.0 if dx == 0 and dy == 0 else 0.35
                    self.vis[y2, x2] += w
        self.stamps += 1

    def _neighbors(self, ix, iy, allow_portal=True):
        out = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                x2, y2 = ix + dx, iy + dy
                if not (0 <= x2 < self.nx and 0 <= y2 < self.ny):
                    continue
                if self.blocked[y2, x2]:
                    continue
                if (not allow_portal) and self.portal[y2, x2]:
                    continue
                if dx != 0 and dy != 0:
                    if self.blocked[iy, x2] or self.blocked[y2, ix]:
                        continue
                step = 1.41421356 if dx and dy else 1.0
                out.append((x2, y2, step))
        return out

    def _flood_room(self, start):
        """Wide cells reachable without crossing a doorway."""
        room = set()
        if start is None or not self.free[start[1], start[0]]:
            return room
        q = deque([start])
        room.add(start)
        while q:
            ix, iy = q.popleft()
            for x2, y2, _ in self._neighbors(ix, iy, allow_portal=False):
                if (x2, y2) in room:
                    continue
                room.add((x2, y2))
                q.append((x2, y2))
        return room

    def _portals_touching(self, room):
        found = []
        seen = set()
        for ix, iy in room:
            for x2, y2, _ in self._neighbors(ix, iy, allow_portal=True):
                if self.portal[y2, x2] and (x2, y2) not in room and (x2, y2) not in seen:
                    seen.add((x2, y2))
                    found.append((x2, y2))
        return found

    def _landings(self, portal, room):
        """Free cells about a metre past the door, not the jamb itself."""
        seeds = []
        ix, iy = portal
        for x2, y2, _ in self._neighbors(ix, iy, allow_portal=True):
            if (x2, y2) in room or self.blocked[y2, x2]:
                continue
            seeds.append((x2, y2))
        if not seeds:
            return []
        seen = set(seeds)
        q = deque(seeds)
        far = []
        px, py = self._cell_xy(*portal)
        while q:
            a, b = q.popleft()
            cx, cy = self._cell_xy(a, b)
            dist = math.hypot(cx - px, cy - py)
            if dist >= 0.85 and not self.portal[b, a]:
                far.append((a, b))
            if dist > 1.7:
                continue
            for x2, y2, _ in self._neighbors(a, b, allow_portal=True):
                if (x2, y2) in room or (x2, y2) in seen:
                    continue
                seen.add((x2, y2))
                q.append((x2, y2))
        if far:
            return far
        return [s for s in seeds if not self.blocked[s[1], s[0]]]

    def _flood_outside(self, seeds, room, cap=500):
        """Unvisited mass beyond the doorway. Does not re-enter the room."""
        seen = set()
        q = deque()
        unk = 0
        for s in seeds:
            if s in room or not self.free[s[1], s[0]]:
                continue
            seen.add(s)
            q.append(s)
        while q and len(seen) < cap:
            ix, iy = q.popleft()
            if self.vis[iy, ix] < 0.8:
                unk += 1
            for x2, y2, _ in self._neighbors(ix, iy, allow_portal=True):
                if (x2, y2) in seen or (x2, y2) in room:
                    continue
                seen.add((x2, y2))
                q.append((x2, y2))
        return unk, len(seen)

    def _astar(self, start, goal, room_bias=None):
        if start is None or goal is None:
            return None
        if start == goal:
            return [start]
        openh = []
        heapq.heappush(openh, (0.0, 0.0, start))
        came = {start: None}
        gscore = {start: 0.0}
        gx, gy = goal
        while openh:
            _, g, cur = heapq.heappop(openh)
            if cur == goal:
                path = [cur]
                while came[cur] is not None:
                    cur = came[cur]
                    path.append(cur)
                path.reverse()
                return path
            if g > gscore.get(cur, 1e18) + 1e-9:
                continue
            ix, iy = cur
            for x2, y2, step in self._neighbors(ix, iy, allow_portal=True):
                ng = g + step * self.cs
                # mild extra cost through already-trampled cells so we do not
                # orbit the same trail, but do not forbid the only path.
                ng += 0.05 * min(6.0, float(self.vis[y2, x2]))
                nxt = (x2, y2)
                if ng + 1e-6 < gscore.get(nxt, 1e18):
                    gscore[nxt] = ng
                    came[nxt] = cur
                    h = math.hypot(x2 - gx, y2 - gy) * self.cs
                    heapq.heappush(openh, (ng + h, ng, nxt))
        return None

    def _heading_free(self, x, y, heading, max_r=2.2):
        step = 0.10
        r = 0.12
        while r <= max_r:
            px = x + math.cos(heading) * r
            py = y + math.sin(heading) * r
            if _point_in_boxes(px, py, self.boxes, 0.06):
                return r
            r += step
        return max_r

    def _lookahead_heading(self, x, y, path, landing):
        # Far from the door, hold the bearing to the landing. A nearby
        # path kink changes every yaw and he never translates.
        if landing is not None:
            dland = math.hypot(landing[0] - x, landing[1] - y)
            if dland > 1.4:
                return math.atan2(landing[1] - y, landing[0] - x)
        if path and len(path) >= 2:
            best = None
            for ix, iy in path[1:]:
                cx, cy = self._cell_xy(ix, iy)
                d = math.hypot(cx - x, cy - y)
                if d < 0.35:
                    continue
                best = (cx, cy)
                if d >= 0.90:
                    break
            if best is not None:
                return math.atan2(best[1] - y, best[0] - x)
        if landing is not None:
            return math.atan2(landing[1] - y, landing[0] - x)
        return 0.0

    def plan(self, x, y, t):
        start = self.nearest_free(x, y)
        if start is None:
            self.plan_note = "no free cell"
            return None
        room = self._flood_room(start)
        if self.origin is None:
            self.origin = (float(x), float(y))
        ox, oy = self.origin
        here = room_of(x, y)
        if here not in ("sw", "south", "other"):
            self.left_first = True

        # After the first doorway, sweep the room we just entered before
        # any door that would send us back toward the spawn cell.
        if self.left_first:
            frontier = None
            best_d = 1.0
            for ix, iy in room:
                if self.portal[iy, ix] or self.vis[iy, ix] > 0.5:
                    continue
                cx, cy = self._cell_xy(ix, iy)
                d = math.hypot(cx - x, cy - y)
                if d < 1.4 or d > 6.5:
                    continue
                d_home = math.hypot(cx - ox, cy - oy)
                d_now = math.hypot(x - ox, y - oy)
                if d_home + 0.3 < d_now:
                    continue
                if d > best_d:
                    best_d = d
                    frontier = (ix, iy, cx, cy, d)
            if frontier is not None:
                path = self._astar(start, (frontier[0], frontier[1]))
                if path:
                    self.path = path
                    self.landing = (frontier[2], frontier[3])
                    self.goal = self.landing
                    self.heading = self._lookahead_heading(x, y, path, self.landing)
                    self._last_portal = None
                    self.plan_note = "room-frontier d=%.1f" % frontier[4]
                    return self.heading

        portals = self._portals_touching(room)
        best = None
        for p in portals:
            key = (p[0], p[1])
            if self.fail_until.get(key, -1.0) > t:
                continue
            lands = self._landings(p, room)
            if not lands:
                continue
            unk, nout = self._flood_outside(lands, room)
            if nout < 3:
                continue
            # Aim well past the jamb so he drives through, not yaw at the gap.
            def land_key(s):
                cx, cy = self._cell_xy(*s)
                px, py = self._cell_xy(*p)
                return math.hypot(cx - px, cy - py) - 0.4 * float(self.vis[s[1], s[0]])
            land = max(lands, key=land_key)
            path = self._astar(start, land)
            if not path:
                continue
            dist = max(1, len(path) - 1) * self.cs
            lx, ly = self._cell_xy(*land)
            # Prefer a doorway into a lot of unseen floor, nearer door if tied.
            score = 3.0 * unk + 0.35 * nout - 0.40 * dist
            d_home = math.hypot(lx - ox, ly - oy)
            d_now = math.hypot(x - ox, y - oy)
            # Do not U-turn back into the first room just because it is unseen.
            if d_home + 0.6 < d_now:
                score -= 120.0
            if self.left_first and unk < 25:
                score -= 40.0
            if best is None or score > best[0]:
                best = (score, path, (lx, ly), p, unk, dist)
        if best is None:
            # Fallback: farthest low-visit free cell, still leave the stamped blob.
            goal = None
            best_s = -1e9
            for iy in range(self.ny):
                for ix in range(self.nx):
                    if not self.free[iy, ix]:
                        continue
                    if self.vis[iy, ix] > 1.5:
                        continue
                    cx, cy = self._cell_xy(ix, iy)
                    d = math.hypot(cx - x, cy - y)
                    if d < 1.2 or d > 8.0:
                        continue
                    s = d - 2.0 * float(self.vis[iy, ix])
                    if s > best_s:
                        best_s = s
                        goal = (ix, iy, cx, cy, d)
            if goal is None:
                self.plan_note = "no exit"
                self.goal = None
                return self.heading
            path = self._astar(start, (goal[0], goal[1]))
            if path and len(path) >= 2:
                self.path = path
                wx, wy = goal[2], goal[3]
            else:
                self.path = None
                wx, wy = goal[2], goal[3]
            self.landing = (wx, wy)
            self.goal = (wx, wy)
            self.heading = self._lookahead_heading(x, y, path, (wx, wy))
            self.plan_note = "frontier fallback d=%.1f" % goal[4]
            return self.heading

        _score, path, landing, portal, unk, dist = best
        self.path = path
        self.landing = landing
        self.goal = landing
        self.heading = self._lookahead_heading(x, y, path, landing)
        px, py = self._cell_xy(*portal)
        self.plan_note = "door (%.1f,%.1f) unk=%d dist=%.1f" % (px, py, unk, dist)
        self._last_portal = portal
        return self.heading

    def _open_heading(self, x, y, yaw):
        """Longest ray that increases distance from spawn. Leaves the first cell."""
        best_h, best = yaw, -1.0
        ox, oy = self.origin if self.origin is not None else (x, y)
        for k in range(16):
            h = -math.pi + (2.0 * math.pi) * k / 16.0
            free = self._heading_free(x, y, h, max_r=6.0)
            nx = x + math.cos(h) * min(2.5, max(free, 0.2))
            ny = y + math.sin(h) * min(2.5, max(free, 0.2))
            leave = math.hypot(nx - ox, ny - oy) - math.hypot(x - ox, y - oy)
            score = free + 1.4 * max(0.0, leave)
            if abs(_norm(h - yaw)) < 0.4:
                score += 0.3
            if score > best:
                best = score
                best_h = h
        return best_h, self._heading_free(x, y, best_h, max_r=6.0)

    def note_fail(self, t, hold=8.0):
        p = getattr(self, "_last_portal", None)
        if p is not None:
            self.fail_until[(p[0], p[1])] = t + hold
        self.stuck_recoveries += 1
        self.recoveries += 1
        self.plan_t = -99.0

    def command(
        self,
        fused_xy,
        chassis_xy,
        chassis_yaw,
        labels_blocked,
        front_m,
        dt,
        t,
        rs1_valid=True,
        chassis_stuck=False,
    ):
        x, y = float(fused_xy[0]), float(fused_xy[1])
        cx, cy = float(chassis_xy[0]), float(chassis_xy[1])
        yaw = float(chassis_yaw)
        self.stamp(x, y)
        # Door heading from the body that actually moves. Fused xy only
        # paints the visit grid (IMU+wheel), never the command pose.
        x, y = cx, cy
        # Progress tracker (chassis truth): tip / snag without abort.
        if self._last_prog_xy is None:
            self._last_prog_xy = (cx, cy)
        step = math.hypot(cx - self._last_prog_xy[0], cy - self._last_prog_xy[1])
        if step >= 0.04:
            self._last_prog_xy = (cx, cy)
            self.no_progress_s = 0.0
            self.tip_s = 0.0
        else:
            self.no_progress_s += float(dt)

        if chassis_stuck and self.escape_left <= 0.0:
            # Physics snag: reverse OFF contact first, then yaw to open
            # leave-spawn heading, then surge. Pure yaw is the tip trap.
            best_h, best_free = self._open_heading(cx, cy, yaw)
            self.heading = best_h
            self.landing = (
                cx + math.cos(best_h) * min(3.5, max(1.2, best_free * 0.7)),
                cy + math.sin(best_h) * min(3.5, max(1.2, best_free * 0.7)),
            )
            self.goal = self.landing
            self.path = None
            self.hold_until = float(t) + 6.0
            self.escape_left = 2.6
            self.escape_phase = "reverse"
            self.escape_sign = 1 if _norm(best_h - yaw) >= 0.0 else -1
            self.stuck_recoveries += 1
            self.recoveries += 1
            self.tip_s = 0.0
            self.no_progress_s = 0.0
            self.plan_note = "snag-escape free=%.1f" % best_free

        if self.escape_left > 0.0:
            self.escape_left = max(0.0, self.escape_left - float(dt))
            if self.escape_left <= 0.0:
                self._door_commit = False
            heading = self.heading if self.heading is not None else yaw
            err = _norm(heading - yaw)
            sign = self.escape_sign if self.escape_sign else (1 if err >= 0.0 else -1)
            phase = getattr(self, "escape_phase", "yaw") or "yaw"
            if phase == "reverse":
                if self.escape_left <= 1.9:
                    self.escape_phase = "yaw"
                else:
                    back = self._heading_free(cx, cy, yaw + math.pi, max_r=0.60)
                    if back >= 0.20:
                        return -0.12, 0.15 * sign, "creep"
                    self.escape_phase = "yaw"
                    phase = "yaw"
            if phase == "yaw":
                # Door commit: require tighter alignment before surging through.
                yaw_tol = 0.22 if getattr(self, "_door_commit", False) else 0.35
                if abs(err) <= yaw_tol:
                    self.escape_phase = "surge"
                    phase = "surge"
                else:
                    return 0.0, 0.95 * sign, "turn_left" if sign > 0 else "turn_right"
            # Widen surge through the portal when pad/commit latch is set.
            if getattr(self, "_door_commit", False):
                return 0.38, float(max(-0.28, min(0.28, 1.0 * err))), "forward"
            return 0.30, float(max(-0.35, min(0.35, 1.2 * err))), "forward"

        # Room-tour waypoints (east of furniture, then north through doors).
        if getattr(self, "wps", None):
            while self.wp_i < len(self.wps):
                wtx, wty = self.wps[self.wp_i]
                # Doorway approach wps: require being near AND not still far
                # south of the pad, else 0.75 m radius skips the approach and
                # the next pad can land along the partition face.
                if -4.2 <= wty <= -1.5:
                    reach = 0.48
                    if cy < wty - 0.35:
                        break
                else:
                    reach = 0.75
                if math.hypot(cx - wtx, cy - wty) >= reach:
                    break
                self.wp_i += 1
                self.recoveries += 1
                self.left_first = True
                self._wp_best_dist = None
                self._wp_still_s = 0.0
            if self.wp_i < len(self.wps):
                tx, ty = self.wps[self.wp_i]
                heading = math.atan2(ty - cy, tx - cx)
                # Doorway wps: approach the free cell in front of the door
                # first, then cross. Never aim along the partition face while
                # still south of the jamb (A* lookahead used to run E/W into
                # the wall). Snap-to-cell only when the snap stays on/near the
                # approach pad — not when it jumps onto the portal itself.
                if ty > -4.0 and ty < -1.5:
                    start = self.nearest_free(cx, cy)
                    goal = self.nearest_free(tx, ty)
                    # Door-x is authoritative. nearest_free can snap into the
                    # wall-parallel corridor (e.g. x=-0.8 for door x=-1.15);
                    # reject lateral snaps so pad stays in the opening column.
                    gx, gy = float(tx), float(ty)
                    goal_portal = False
                    if goal is not None:
                        gcx, gcy = self._cell_xy(*goal)
                        if abs(gcx - tx) <= 0.20:
                            gx, gy = gcx, gcy
                            goal_portal = bool(self.portal[goal[1], goal[0]])
                        else:
                            goal = None  # lateral snap — treat as no grid goal
                    # Portal/crossing wps: pad south of jamb. Approach wps
                    # (already south, e.g. y<=-3.35) ARE the pad — do not
                    # push further south into corridor cells.
                    if goal_portal or ty > -3.35:
                        ax, ay = self._approach_pad(gx, gy)
                    else:
                        ax, ay = (gx, gy)
                    # Final guard: never aim more than 0.20 m off door-x.
                    if abs(ax - tx) > 0.20:
                        ax, ay = float(tx), float(ay if abs(ay - ty) <= 1.0 else ty)
                    d_appr = math.hypot(cx - ax, cy - ay)
                    south_of_jamb = cy < (gy - 0.32)
                    if south_of_jamb and d_appr > 0.42:
                        used = False
                        if start is not None:
                            sx, sy = self._cell_xy(*start)
                            d_start = math.hypot(sx - cx, sy - cy)
                            # Accept snap only if it is not north of the approach
                            # pad (portal snap reintroduces along-jamb aim).
                            if d_start > 0.28 and sy <= (ay + 0.12):
                                heading = math.atan2(sy - cy, sx - cx)
                                self.path = None
                                self.plan_note = "wp%d-to-cell (%.1f,%.1f)" % (self.wp_i, sx, sy)
                                used = True
                        if not used:
                            # Aim at approach pad; if nose is jammed (along face),
                            # bias toward door-x at approach-y (more north).
                            heading = math.atan2(ay - cy, ax - cx)
                            nose_h = self._heading_free(cx, cy, heading, max_r=1.2)
                            if nose_h < 0.40:
                                heading_n = math.atan2(ay - cy, gx - cx)
                                if self._heading_free(cx, cy, heading_n, max_r=1.2) >= nose_h:
                                    heading = heading_n
                            self.path = None
                            self.plan_note = "wp%d-approach (%.1f,%.1f)" % (self.wp_i, ax, ay)
                    elif start is not None and goal is not None:
                        sx, sy = self._cell_xy(*start)
                        d_start = math.hypot(sx - cx, sy - cy)
                        if d_start > 0.28:
                            heading = math.atan2(sy - cy, sx - cx)
                            self.path = None
                            self.plan_note = "wp%d-to-cell (%.1f,%.1f)" % (self.wp_i, sx, sy)
                        else:
                            path = self._astar(start, goal)
                            if path and len(path) >= 2:
                                self.path = path
                                heading = self._lookahead_heading(cx, cy, path, (tx, ty))
                                # Guard: if A* lookahead is along the jamb with
                                # a blocked nose, fall back to portal/approach.
                                nose_h = self._heading_free(cx, cy, heading, max_r=1.0)
                                if nose_h < 0.32 and south_of_jamb:
                                    heading = math.atan2(gy - cy, gx - cx)
                                    nose2 = self._heading_free(cx, cy, heading, max_r=1.0)
                                    if nose2 < 0.32:
                                        heading = math.atan2(ay - cy, ax - cx)
                                    self.path = None
                                    self.plan_note = "wp%d-door-fix (%.1f,%.1f)" % (self.wp_i, gx, gy)
                                else:
                                    self.plan_note = "wp%d-door (%.1f,%.1f)" % (self.wp_i, tx, ty)
                            else:
                                self.plan_note = "wp%d (%.1f,%.1f)" % (self.wp_i, tx, ty)
                    else:
                        self.plan_note = "wp%d (%.1f,%.1f)" % (self.wp_i, tx, ty)
                else:
                    self.plan_note = "wp%d (%.1f,%.1f)" % (self.wp_i, tx, ty)
                self.heading = heading
                self.landing = (tx, ty)
                self.goal = (tx, ty)
                err = _norm(heading - yaw)
                sign = _yaw_sign_toward((cx, cy), yaw, heading, self.boxes)
                if sign == 0:
                    sign = 1 if err >= 0.0 else -1
                nose = self._heading_free(cx, cy, heading, max_r=1.2)
                front_close = front_m is not None and math.isfinite(front_m) and front_m < 0.36
                # Track progress toward the active waypoint.
                d_wp = math.hypot(cx - tx, cy - ty)
                if self._wp_best_dist is None or d_wp + 0.05 < float(self._wp_best_dist):
                    self._wp_best_dist = d_wp
                    self._wp_still_s = 0.0
                else:
                    self._wp_still_s += float(dt)

                # --- Physical door cross (pad harden already in place) ---
                # Commit reverse→surge through portal when within 0.4 m and yaw
                # is aligned to the door normal (not the east corridor). Also
                # widen surge once the approach pad is reached under partition-
                # shadow false-open (front_m large while still south of jamb).
                portal_xy = self._nearest_portal_xy(cx, cy, max_r=2.0)
                if portal_xy is None and -4.2 <= ty <= -1.5:
                    portal_xy = (float(tx), float(ty)) if ty > -3.35 else (float(tx), -3.13)
                door_x = float(tx) if -4.2 <= ty <= -1.5 else (
                    float(portal_xy[0]) if portal_xy is not None else cx
                )
                pad_xy = self._approach_pad(door_x, float(portal_xy[1]) if portal_xy else float(ty))
                d_pad = math.hypot(cx - pad_xy[0], cy - pad_xy[1])
                pad_reached = (
                    d_pad <= 0.55
                    and abs(cx - door_x) <= 0.28
                    and cy >= (pad_xy[1] - 0.35)
                )
                d_portal = (
                    math.hypot(cx - portal_xy[0], cy - portal_xy[1])
                    if portal_xy is not None else 99.0
                )
                if portal_xy is not None and self.escape_left <= 0.0:
                    door_n = self._door_normal_heading(cx, cy, portal_xy)
                    yaw_err_door = abs(_norm(door_n - yaw))
                    # Partition-shadow false-open: depth says clear but we are
                    # still south of the jamb on the door column.
                    false_open = (
                        pad_reached
                        and cy < (portal_xy[1] - 0.12)
                        and (front_m is None or (math.isfinite(front_m) and front_m > 1.2))
                    )
                    commit_close = d_portal <= 0.40 and yaw_err_door <= 0.45
                    commit_pad = (
                        pad_reached
                        and d_portal <= 0.95
                        and yaw_err_door <= 0.55
                        and (false_open or self.no_progress_s > 2.5 or self._wp_still_s > 3.0)
                    )
                    if commit_close or commit_pad:
                        self._door_commit = True
                        self.heading = door_n
                        self.landing = (
                            portal_xy[0],
                            portal_xy[1] + (0.70 if cy <= portal_xy[1] else -0.70),
                        )
                        self.goal = self.landing
                        self.path = None
                        self.escape_left = 3.6 if commit_pad or false_open else 3.0
                        self.escape_phase = "reverse"
                        self.escape_sign = 1 if _norm(door_n - yaw) >= 0.0 else -1
                        self.hold_until = float(t) + 5.5
                        self.tip_s = 0.0
                        self.no_progress_s = 0.0
                        self._wp_still_s = 0.0
                        self._wp_best_dist = None
                        self.stuck_recoveries += 1
                        self.recoveries += 1
                        self.plan_note = (
                            "door-commit d=%.2f yaw_err=%.2f pad=%s fo=%s"
                            % (d_portal, yaw_err_door, int(pad_reached), int(false_open))
                        )
                        # Already on/near pad: skip long reverse that walks
                        # south into the corridor pin; brief reverse only if
                        # nose is jammed against the jamb face.
                        nose_door = self._heading_free(cx, cy, door_n, max_r=0.80)
                        back = self._heading_free(cx, cy, yaw + math.pi, max_r=0.55)
                        if pad_reached and nose_door >= 0.28:
                            self.escape_phase = "yaw"
                            if yaw_err_door <= 0.22:
                                self.escape_phase = "surge"
                                return 0.38, float(max(-0.28, min(0.28, 1.0 * _norm(door_n - yaw)))), "forward"
                            return 0.0, 0.95 * self.escape_sign, "turn_left" if self.escape_sign > 0 else "turn_right"
                        if back >= 0.18 and (not pad_reached or nose_door < 0.28):
                            return -0.12, 0.20 * self.escape_sign, "creep"
                        self.escape_phase = "yaw"
                        return 0.0, 0.95 * self.escape_sign, "turn_left" if self.escape_sign > 0 else "turn_right"

                # Near a portal: never skip past the jamb — A* through the gap.
                near_portal = False
                try:
                    ix0, iy0 = self._clamp_idx(cx, cy)
                    for dy in range(-3, 4):
                        for dx in range(-3, 4):
                            x2, y2 = ix0 + dx, iy0 + dy
                            if 0 <= x2 < self.nx and 0 <= y2 < self.ny and self.portal[y2, x2]:
                                near_portal = True
                                break
                        if near_portal:
                            break
                except Exception:
                    near_portal = False

                # True open-floor / solver pin only (front wide open). Do not
                # treat doorway tip-backs as skip — that jumps past the jamb.
                open_floor_pin = (
                    (not front_close) and nose >= 1.0
                    and (front_m is None or (math.isfinite(front_m) and front_m > 2.5))
                    and (not near_portal)
                )
                skip_due = (
                    open_floor_pin
                    and (self.no_progress_s > 6.0 or self._wp_still_s > 8.0)
                    and float(t) >= float(getattr(self, "_wp_skip_cool", -99.0))
                )
                if near_portal and (self.no_progress_s > 5.0 or self._wp_still_s > 6.0):
                    # Doorway physical unstick: reverse+yaw, then A* to a
                    # landing on the far side of the nearest portal.
                    self.tip_s = 0.0
                    self.no_progress_s = 0.0
                    self._wp_still_s = 0.0
                    self._wp_best_dist = None
                    self.stuck_recoveries += 1
                    self.recoveries += 1
                    # Aim through portal into low-visit / next room.
                    self.plan(cx, cy, t)
                    if self.heading is None:
                        best_h, best_free = self._open_heading(cx, cy, yaw)
                        self.heading = best_h
                    else:
                        best_free = self._heading_free(cx, cy, self.heading, max_r=4.0)
                    best_h = float(self.heading)
                    # If planned ray is jammed: while south of jamb prefer the
                    # approach pad (never aim along the partition at a portal
                    # cell). Only lock onto the portal once near/aligned.
                    if best_free < 0.35:
                        best_p = None
                        best_d = 1e9
                        ix0, iy0 = self._clamp_idx(cx, cy)
                        for dy in range(-5, 6):
                            for dx in range(-5, 6):
                                x2, y2 = ix0 + dx, iy0 + dy
                                if not (0 <= x2 < self.nx and 0 <= y2 < self.ny):
                                    continue
                                if not self.portal[y2, x2]:
                                    continue
                                px, py = self._cell_xy(x2, y2)
                                d = math.hypot(px - cx, py - cy)
                                if 0.15 < d < best_d:
                                    best_d = d
                                    best_p = (px, py)
                        if best_p is not None and cy < (best_p[1] - 0.32):
                            ax, ay = self._approach_pad(best_p[0], best_p[1])
                            best_h = math.atan2(ay - cy, ax - cx)
                            nose_h = self._heading_free(cx, cy, best_h, max_r=2.0)
                            if nose_h < 0.35:
                                best_h = math.atan2(ay - cy, best_p[0] - cx)
                                nose_h = self._heading_free(cx, cy, best_h, max_r=2.0)
                            best_free = nose_h
                            self.heading = best_h
                            self.landing = (ax, ay)
                            self.goal = (ax, ay)
                        elif best_p is not None:
                            best_h = math.atan2(best_p[1] - cy, best_p[0] - cx)
                            best_free = self._heading_free(cx, cy, best_h, max_r=2.0)
                            self.heading = best_h
                            self.landing = best_p
                            self.goal = best_p
                    # Prefer door-normal heading over corridor false-open.
                    if portal_xy is not None:
                        door_n = self._door_normal_heading(cx, cy, portal_xy)
                        nose_n = self._heading_free(cx, cy, door_n, max_r=2.0)
                        if nose_n >= 0.20 or abs(cx - portal_xy[0]) <= 0.30:
                            best_h = door_n
                            best_free = nose_n
                            self.heading = best_h
                            self.landing = (
                                portal_xy[0],
                                portal_xy[1] + (0.70 if cy <= portal_xy[1] else -0.70),
                            )
                            self.goal = self.landing
                    self.escape_left = 3.4
                    self.escape_phase = "reverse"
                    self.escape_sign = 1 if _norm(best_h - yaw) >= 0.0 else -1
                    self.hold_until = float(t) + 5.0
                    # Latch widen-surge if pad already reached (false-open case).
                    if pad_reached:
                        self._door_commit = True
                        self.escape_left = 4.0
                    self.plan_note = "door-unstick free=%.1f %s" % (best_free, self.plan_note)
                    # Advance wp only if already past this portal band.
                    if self.wp_i < len(self.wps) and self.wps[self.wp_i][1] < -2.8:
                        # still south of jamb — keep approach wps, do not skip
                        pass
                    # Do not reverse further south of the approach pad — that
                    # re-enters the snag loop at y≈-4.0. Yaw/surge north instead.
                    south_of_pad = cy < (pad_xy[1] - 0.10)
                    back = self._heading_free(cx, cy, yaw + math.pi, max_r=0.60)
                    if south_of_pad:
                        self.escape_phase = "yaw"
                        if abs(_norm(best_h - yaw)) <= 0.30:
                            self.escape_phase = "surge"
                            return 0.34, float(max(-0.30, min(0.30, 1.1 * _norm(best_h - yaw)))), "forward"
                        return 0.0, 0.95 * self.escape_sign, "turn_left" if self.escape_sign > 0 else "turn_right"
                    if back >= 0.18:
                        return -0.14, 0.30 * self.escape_sign, "creep"
                    return 0.0, 0.95 * self.escape_sign, "turn_left" if self.escape_sign > 0 else "turn_right"

                if skip_due:
                    self.tip_s = 0.0
                    self.no_progress_s = 0.0
                    self._wp_still_s = 0.0
                    self._wp_best_dist = None
                    self.wp_skips += 1
                    self._wp_skip_cool = float(t) + 12.0
                    self.wp_i += 1
                    self.stuck_recoveries += 1
                    self.recoveries += 1
                    best_h, best_free = self._open_heading(cx, cy, yaw)
                    if self.wp_i < len(self.wps):
                        ntx, nty = self.wps[self.wp_i]
                        cand = math.atan2(nty - cy, ntx - cx)
                        if self._heading_free(cx, cy, cand, max_r=1.5) >= 0.6:
                            best_h = cand
                            best_free = self._heading_free(cx, cy, best_h, max_r=4.0)
                    self.heading = best_h
                    self.landing = (
                        cx + math.cos(best_h) * min(3.5, max(1.2, best_free * 0.7)),
                        cy + math.sin(best_h) * min(3.5, max(1.2, best_free * 0.7)),
                    )
                    self.goal = self.landing
                    self.path = None
                    self.escape_left = 2.8
                    self.escape_phase = "reverse"
                    self.escape_sign = 1 if _norm(best_h - yaw) >= 0.0 else -1
                    self.hold_until = float(t) + 4.0
                    self.plan_note = "wp-skip-pin ->wp%d free=%.1f" % (self.wp_i, best_free)
                    back = self._heading_free(cx, cy, yaw + math.pi, max_r=0.60)
                    if back >= 0.20:
                        return -0.14, 0.25 * self.escape_sign, "creep"
                    return 0.0, 0.95 * self.escape_sign, "turn_left" if self.escape_sign > 0 else "turn_right"

                # Tip / no-progress only if the nose is actually blocked.
                if (front_close or nose < 0.40) and (self.no_progress_s > 4.5 or self.tip_s > 5.0):
                    self.tip_s = 0.0
                    self.no_progress_s = 0.0
                    self._wp_still_s = 0.0
                    self._wp_best_dist = None
                    self.note_fail(t, hold=4.0)
                    best_h, best_free = self._open_heading(cx, cy, yaw)
                    self.heading = best_h
                    self.escape_left = 2.4
                    self.escape_phase = "reverse"
                    self.escape_sign = 1 if _norm(best_h - yaw) >= 0.0 else -1
                    self.plan_note = "wp-tip-recover free=%.1f" % best_free
                    return -0.12, 0.20 * self.escape_sign, "creep"
                if front_close or nose < 0.28:
                    if abs(err) > 0.25:
                        return 0.0, 0.90 * sign, "turn_left" if sign > 0 else "turn_right"
                    back = self._heading_free(cx, cy, yaw + math.pi, max_r=0.55)
                    if back >= 0.28:
                        return -0.10, 0.0, "creep"
                    return 0.0, 0.90 * sign, "turn_left" if sign > 0 else "turn_right"
                if abs(err) > 0.85:
                    return 0.0, 0.90 * sign, "turn_left" if sign > 0 else "turn_right"
                if abs(err) > 0.40:
                    return 0.14, float(np.clip(1.5 * err, -0.70, 0.70)), "creep"
                return 0.28, float(np.clip(1.2 * err, -0.35, 0.35)), "forward"

        self.recent.append((x, y))
        self.recent_t.append(float(t))

        # Loop in one cell only if we are not closing on the committed door.
        if len(self.recent) >= 24 and self.landing is not None:
            arr = np.asarray(self.recent, dtype=np.float64)
            span = float(np.hypot(arr[:, 0].max() - arr[:, 0].min(), arr[:, 1].max() - arr[:, 1].min()))
            age = float(t) - float(self.recent_t[0])
            d0 = math.hypot(arr[0, 0] - self.landing[0], arr[0, 1] - self.landing[1])
            d1 = math.hypot(x - self.landing[0], y - self.landing[1])
            if age >= 9.0 and span < 1.25 and d1 > d0 - 0.25:
                self.note_fail(t, hold=10.0)
                self.recent.clear()
                self.recent_t.clear()
                self.landing = None
                self.plan_note = "loop-escape " + self.plan_note

        crossed = False
        holding = float(t) < float(getattr(self, "hold_until", 0.0) or 0.0)
        if (not holding) and self.landing is not None and math.hypot(x - self.landing[0], y - self.landing[1]) < 0.70:
            crossed = True
        # Hold the door until we are through it. Do not hop between nearby gaps.
        need_plan = (not holding) and (self.heading is None or self.landing is None or crossed)
        if need_plan:
            if crossed:
                self.recoveries += 1
            self.plan(x, y, t)
            self.plan_t = float(t)
            if self.commit_xy is None:
                self.commit_xy = (x, y)
        elif self.path is not None and self.landing is not None:
            # Refresh the path slowly. Every-tick A* makes the heading hunt
            # and he spends the run yawing instead of crossing the door.
            if (float(t) - self.plan_t) > 1.2:
                start = self.nearest_free(x, y)
                if start is not None:
                    land_i = self.nearest_free(self.landing[0], self.landing[1])
                    if land_i is not None:
                        path = self._astar(start, land_i)
                        if path:
                            self.path = path
                self.plan_t = float(t)
            self.heading = self._lookahead_heading(x, y, self.path, self.landing)

        heading = self.heading if self.heading is not None else yaw
        # Early open-corridor only before a door is committed.
        dist_home = 99.0
        if self.origin is not None:
            dist_home = math.hypot(cx - self.origin[0], cy - self.origin[1])
        if dist_home >= 2.4:
            self.left_first = True
        if (
            (not getattr(self, "left_first", False))
            and self.landing is None
            and dist_home < 2.2
            and float(t) >= float(getattr(self, "hold_until", 0.0) or 0.0)
        ):
            heading, _fr = self._open_heading(cx, cy, yaw)
            self.heading = heading
            self.plan_note = "open-corridor"
        # Nose ray uses a slim pad so a doorway gap is not a wall.
        nose = self._heading_free(cx, cy, heading, max_r=1.2)
        err = _norm(heading - yaw)
        front_close = front_m is not None and math.isfinite(front_m) and front_m < 0.36
        nose_hit = nose < 0.30
        ahead_wall = bool(labels_blocked or front_close or nose_hit)

        sign = _yaw_sign_toward((cx, cy), yaw, heading, self.boxes)
        if sign == 0:
            back = self._heading_free(cx, cy, yaw + math.pi, max_r=0.50)
            if back >= 0.28:
                return -0.08, 0.0, "creep"
            sign = 1 if err >= 0.0 else -1

        turning = ahead_wall or abs(err) > 0.85
        if turning and abs(err) > 0.4:
            self.spin_s += float(dt)
            self.tip_s += float(dt)
        else:
            self.spin_s = 0.0
            if step >= 0.02:
                self.tip_s = 0.0

        # Turn-in-place / no-progress: reverse+replan, never abort here.
        if self.tip_s > 5.0 or (self.no_progress_s > 4.0 and abs(err) > 0.5):
            self.spin_s = 0.0
            self.tip_s = 0.0
            self.no_progress_s = 0.0
            self.bad_heading = heading
            self.landing = None
            self.note_fail(t, hold=8.0)
            best_h, best_free = self._open_heading(cx, cy, yaw)
            self.heading = best_h
            self.landing = (
                cx + math.cos(best_h) * min(3.5, max(1.2, best_free * 0.7)),
                cy + math.sin(best_h) * min(3.5, max(1.2, best_free * 0.7)),
            )
            self.goal = self.landing
            self.path = None
            self.hold_until = float(t) + 5.0
            self.escape_left = 2.4
            self.escape_phase = "reverse"
            self.escape_sign = 1 if _norm(best_h - yaw) >= 0.0 else -1
            self.plan_note = "tip-recover free=%.1f" % best_free
            back = self._heading_free(cx, cy, yaw + math.pi, max_r=0.60)
            if back >= 0.22:
                return -0.12, 0.20 * self.escape_sign, "creep"
            return 0.0, 0.95 * self.escape_sign, "turn_left" if self.escape_sign > 0 else "turn_right"

        if self.spin_s > 2.0:
            self.spin_s = 0.0
            self.bad_heading = heading
            self.landing = None
            self.note_fail(t, hold=6.0)
            back = self._heading_free(cx, cy, yaw + math.pi, max_r=0.55)
            if back >= 0.28:
                return -0.10, 0.0, "creep"
            sign = -sign

        if ahead_wall:
            # Latch one yaw so he does not tick-flip left/right at the jamb.
            latch = getattr(self, "yaw_latch", None)
            if latch is None or float(t) >= latch[1]:
                self.yaw_latch = (1 if sign >= 0 else -1, float(t) + 1.6)
            sign = self.yaw_latch[0]
            if abs(err) < 0.45 and not front_close and nose >= 0.32:
                return 0.18, 0.40 * sign, "creep"
            if front_close and front_m is not None and front_m < 0.40:
                back = self._heading_free(cx, cy, yaw + math.pi, max_r=0.50)
                if back >= 0.25 and float(getattr(self, "wall_back_t", 0.0)) < float(t):
                    self.wall_back_t = float(t) + 0.45
                    return -0.12, 0.0, "creep"
            return 0.0, 0.95 * sign, "turn_left" if sign > 0 else "turn_right"

        # Keep wheels turning; arc when moderately misaligned.
        if abs(err) > 1.35:
            return 0.0, 0.95 * sign, "turn_left" if sign > 0 else "turn_right"
        if abs(err) > 0.55 and nose >= 0.34:
            return 0.16, float(np.clip(1.6 * err, -0.75, 0.75)), "creep"
        w = float(np.clip(1.6 * err, -0.45, 0.45))
        return 0.28, w, "forward"

    def coverage(self, path_xy=None):
        visited = self.vis > 0.4
        n_vis = int((visited & self.free).sum())
        frac = (n_vis / float(self.n_free)) if self.n_free else 0.0
        rooms = {}
        src = path_xy if path_xy else []
        if src:
            for item in src:
                px, py = float(item[0]), float(item[1])
                name = room_of(px, py)
                rooms[name] = rooms.get(name, 0) + 1
        else:
            ys, xs = np.nonzero(visited & self.free)
            for iy, ix in zip(ys, xs):
                cx, cy = self._cell_xy(int(ix), int(iy))
                name = room_of(cx, cy)
                rooms[name] = rooms.get(name, 0) + 1
        # A room counts if the path actually lingered there, not a graze.
        touched = sorted(k for k, n in rooms.items() if n >= 4 and k != "other")
        return {
            "visit_cells": n_vis,
            "free_cells": self.n_free,
            "visit_fraction": frac,
            "rooms": touched,
            "room_counts": rooms,
            "stuck_recoveries": self.stuck_recoveries,
            "recoveries": self.recoveries,
            "wp_skips": int(getattr(self, "wp_skips", 0) or 0),
            "plan": self.plan_note,
        }

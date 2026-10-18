"""Test environments for the Jetson emulator, with ground truth.

Each env is a deterministic function of (name, seed) -> Env: textured static geoms (MuJoCo), named places
(map labels the nav stack can be told about), spawn pose, and a ground-truth occupancy grid (2 cm, robot
collision band 3-95 cm so table tops count, the mast is ~0.92 m tall).  Walls/furniture get procedural
textures so stereo VSLAM has features, like real rooms do.

  python -m emu.envs list | python -m emu.envs png <name> [seed]   (writes GT map preview)
"""
from __future__ import annotations
import math, os, hashlib
from dataclasses import dataclass, field
import numpy as np

TEX_DIR = os.path.join(os.environ.get("KEVIN_EMU_CACHE", os.path.expanduser("~/.cache/kevin_emu")), "tex")
GT_RES = 0.02
WALL_RES = 0.01        # mesh-wall collision raster (emu.world boxes); finer than GT_RES so the skin hugs the visual wall
# Mesh-wall slicing band: floor clearance up to Kevin's full height.  Twin (kevin_full.xml, all geoms, axle at
# 0.0857 m): top 1.006 m (mast head); above 0.20 m only the mast/RS1 column (x -0.20..+0.03 m from the axle).
# Mesh features above ROBOT_TOP_M cannot touch Kevin and are not walls.  Slices every SLICE_DZ: the old 4 slices
# (0.08/0.35/0.63/0.90 m) missed the casita island top (z 0.684-0.744 m) -> perceived but neither GT nor collision.
ROBOT_TOP_M = 1.02
SLICE_DZ = 0.02
BAND = (0.03, ROBOT_TOP_M)


@dataclass
class Geom:
    kind: str                  # box | cyl | mesh
    pos: tuple                 # centre x,y,z (m)
    size: tuple                # box half-sizes / cyl (r, half-h) / mesh scale
    yaw: float = 0.0
    mat: str = "plaster"
    mesh: str | None = None


@dataclass
class Env:
    name: str
    seed: int
    extent: tuple              # xmin, ymin, xmax, ymax
    geoms: list = field(default_factory=list)
    places: dict = field(default_factory=dict)       # label -> (x, y)
    start: tuple = (0.0, 0.0, 0.0)
    gt: np.ndarray | None = None                     # bool occupancy, row = y
    gt_origin: tuple = (0.0, 0.0)

    def gt_world_to_rc(self, x, y):
        return int((y - self.gt_origin[1]) / GT_RES), int((x - self.gt_origin[0]) / GT_RES)


# ---------------------------------------------------------------- textures
MATS = {   # name: (base rgb, noise amp, pattern)
    "plaster": ((0.86, 0.84, 0.80), 0.10, "noise"), "plaster2": ((0.78, 0.82, 0.86), 0.10, "noise"),
    "plaster3": ((0.88, 0.80, 0.70), 0.10, "noise"), "wood": ((0.55, 0.38, 0.22), 0.25, "grain"),
    "fabric": ((0.30, 0.36, 0.48), 0.20, "noise"), "fabric2": ((0.55, 0.25, 0.22), 0.20, "noise"),
    "metal": ((0.62, 0.64, 0.66), 0.08, "noise"), "white": ((0.93, 0.93, 0.92), 0.05, "noise"),
    "poster": ((0.5, 0.5, 0.5), 0.0, "poster"), "dark": ((0.15, 0.15, 0.17), 0.08, "noise"),
    "concrete": ((0.62, 0.61, 0.58), 0.12, "noise"), "grass": ((0.22, 0.42, 0.14), 0.30, "noise"),
    "tile": ((0.80, 0.80, 0.78), 0.06, "tile"), "carpet": ((0.45, 0.42, 0.38), 0.25, "noise"),
    "fur": ((0.62, 0.45, 0.25), 0.30, "noise"), "fur2": ((0.22, 0.24, 0.30), 0.25, "noise"),
}


def _noise(rng, n, octaves=5):
    out = np.zeros((n, n), np.float32)
    for o in range(octaves):
        k = 4 * 2 ** o
        g = rng.standard_normal((k + 1, k + 1)).astype(np.float32)
        x = np.linspace(0, k, n, endpoint=False); i = x.astype(int); f = x - i; f = f * f * (3 - 2 * f)
        a = g[i][:, i] * (1 - f)[None] + g[i][:, i + 1] * f[None]
        b = g[i + 1][:, i] * (1 - f)[None] + g[i + 1][:, i + 1] * f[None]
        out += (a * (1 - f)[:, None] + b * f[:, None]) / (2 ** o)
    return out / (np.abs(out).max() + 1e-6)


def ensure_textures():
    os.makedirs(TEX_DIR, exist_ok=True)
    import cv2
    for name, (rgb, amp, pat) in MATS.items():
        p = os.path.join(TEX_DIR, name + ".png")
        if os.path.exists(p): continue
        rng = np.random.default_rng(int(hashlib.md5(name.encode()).hexdigest()[:8], 16)); n = 512
        if pat == "poster":
            img = np.full((n, n, 3), 0.9, np.float32)
            for _ in range(40):
                x0, y0 = rng.integers(0, n - 40, 2); w, h = rng.integers(20, 160, 2)
                img[y0:y0 + h, x0:x0 + w] = rng.random(3)
            img += 0.05 * _noise(rng, n)[..., None]
        else:
            nz = _noise(rng, n)
            if pat == "grain":
                yy = np.linspace(0, 40, n)[:, None]; nz = 0.6 * np.sin(yy + 6 * nz) + 0.4 * nz
            img = np.clip(np.array(rgb, np.float32)[None, None] * (1 + amp * nz[..., None]), 0, 1)
            if pat == "tile":
                g = (np.arange(n) % 64 < 2); img[g] *= 0.6; img[:, g] *= 0.6
            sp = rng.random((n, n)) < 0.004        # small specks = corners for FAST/ORB
            img[sp] *= 0.4
        cv2.imwrite(p, (np.clip(img, 0, 1)[..., ::-1] * 255).astype(np.uint8))
    return TEX_DIR


# ---------------------------------------------------------------- builders
class B:
    def __init__(self, name, seed):
        self.e = Env(name, seed, (0, 0, 0, 0)); self.rng = np.random.default_rng(seed)

    def box(self, x, y, z, hx, hy, hz, yaw=0.0, mat="plaster"):
        self.e.geoms.append(Geom("box", (x, y, z), (hx, hy, hz), yaw, mat))

    def cyl(self, x, y, r, h, z0=0.0, mat="metal"):
        self.e.geoms.append(Geom("cyl", (x, y, z0 + h / 2), (r, h / 2), 0.0, mat))

    def wall(self, x0, y0, x1, y1, gaps=(), t=0.06, h=2.4, mat="plaster"):
        """Wall segment from (x0,y0) to (x1,y1) with door gaps [(s_from, s_to)] in metres along it."""
        L = math.hypot(x1 - x0, y1 - y0); ux, uy = (x1 - x0) / L, (y1 - y0) / L; yaw = math.atan2(uy, ux)
        cuts = [0.0]
        for a, b in sorted(gaps): cuts += [a, b]
        cuts.append(L)
        for a, b in zip(cuts[0::2], cuts[1::2]):
            if b - a < 1e-3: continue
            m = (a + b) / 2
            self.box(x0 + ux * m, y0 + uy * m, h / 2, (b - a) / 2, t / 2, h / 2, yaw, mat)
        for a, b in zip(cuts[1:-1:2], cuts[2:-1:2]):      # lintel over doors (above robot, visual only)
            m = (a + b) / 2
            # floor transition / door threshold strip across the gap (added physically by emu.world when the
            # floor model is on): centre, direction along the wall, length
            if not hasattr(self.e, "thresholds"): self.e.thresholds = []
            self.e.thresholds.append((x0 + ux * m, y0 + uy * m, yaw, b - a))
            self.box(x0 + ux * m, y0 + uy * m, 2.2, (b - a) / 2, t / 2, 0.2, yaw, mat)

    def room(self, x0, y0, x1, y1, mat="plaster", doors=None):
        doors = doors or {}
        self.wall(x0, y0, x1, y0, doors.get("s", ()), mat=mat); self.wall(x1, y0, x1, y1, doors.get("e", ()), mat=mat)
        self.wall(x1, y1, x0, y1, doors.get("n", ()), mat=mat); self.wall(x0, y1, x0, y0, doors.get("w", ()), mat=mat)

    def poster(self, x, y, yaw, w=0.8, h=0.6, z=1.2):
        self.box(x, y, z, 0.005, w / 2, h / 2, yaw, "poster")

    def table(self, x, y, w=1.2, d=0.8, h=0.75, yaw=0.0, mat="wood"):
        self.box(x, y, h - 0.02, w / 2, d / 2, 0.02, yaw, mat)
        c, s = math.cos(yaw), math.sin(yaw)
        for sx in (-1, 1):
            for sy in (-1, 1):
                lx, ly = sx * (w / 2 - 0.05), sy * (d / 2 - 0.05)
                self.box(x + c * lx - s * ly, y + s * lx + c * ly, (h - 0.04) / 2, 0.025, 0.025, (h - 0.04) / 2, yaw, mat)

    def chair(self, x, y, yaw=0.0):
        self.table(x, y, 0.42, 0.42, 0.45, yaw, "wood")
        c, s = math.cos(yaw), math.sin(yaw)
        self.box(x - c * 0.2, y - s * 0.2, 0.7, 0.02, 0.2, 0.25, yaw, "wood")

    def couch(self, x, y, yaw=0.0, w=2.0):
        c, s = math.cos(yaw), math.sin(yaw)
        self.box(x, y, 0.22, w / 2, 0.45, 0.22, yaw, "fabric")
        self.box(x - s * -0.38, y + c * -0.38, 0.45, w / 2, 0.08, 0.45, yaw, "fabric")   # back

    def bed(self, x, y, yaw=0.0):
        self.box(x, y, 0.27, 1.0, 0.8, 0.27, yaw, "fabric2")

    def shelf(self, x, y, yaw=0.0, w=0.9):
        self.box(x, y, 0.9, w / 2, 0.17, 0.9, yaw, "wood")

    def clutter(self, xmin, ymin, xmax, ymax, n, keep=()):
        k = 0
        while k < n:
            x, y = self.rng.uniform(xmin, xmax), self.rng.uniform(ymin, ymax)
            if any(math.hypot(x - a, y - b) < r for a, b, r in keep): continue
            t = self.rng.integers(0, 4)
            if t == 0: self.box(x, y, 0.12, *self.rng.uniform(0.08, 0.25, 2), 0.12, self.rng.uniform(0, 3.14), "dark")
            elif t == 1: self.cyl(x, y, self.rng.uniform(0.04, 0.15), self.rng.uniform(0.15, 0.9), mat="metal")
            elif t == 2: self.box(x, y, 0.025, *self.rng.uniform(0.15, 0.3, 2), 0.025, self.rng.uniform(0, 3.14), "fabric2")  # low (5 cm)
            else: self.box(x, y, 0.3, *self.rng.uniform(0.1, 0.2, 2), 0.3, self.rng.uniform(0, 3.14), "white")
            k += 1


# ---------------------------------------------------------------- environments
def env_room(seed):
    b = B("room", seed); W, H = 5.0, 4.0
    b.room(0, 0, W, H, mat="plaster2")
    b.couch(2.5, 3.4, 0.0); b.table(2.5, 2.2, 1.0, 0.6, 0.45, mat="wood"); b.shelf(0.25, 2.0, math.pi / 2)
    b.poster(W - 0.04, 2.0, math.pi); b.poster(0.04, 3.2, 0.0); b.poster(2.0, 0.04, math.pi / 2, w=1.2)
    b.cyl(4.5, 3.5, 0.18, 0.6, mat="white")       # plant pot
    b.e.places = {"couch": (2.5, 2.75), "door": (4.6, 0.5), "shelf": (0.75, 2.0), "center": (2.5, 1.2)}
    b.e.start = (1.0, 0.9, 0.0); b.e.extent = (-0.2, -0.2, W + 0.2, H + 0.2); return b.e


def env_apartment(seed):
    """2-bed apartment, 10 x 7 m: living, kitchen, hall, 2 bedrooms, bath. Doors 0.80-0.90 m."""
    b = B("apartment", seed); r = b.rng
    j = lambda: float(r.uniform(-0.05, 0.05))
    b.room(0, 0, 10, 7, mat="plaster")
    b.wall(6, 0, 6, 7, gaps=[(1.0 + j(), 1.9), (4.6, 5.45 + j())], mat="plaster2")       # living | bedrooms
    b.wall(6, 3.5, 10, 3.5, gaps=[(0.6, 1.45)], mat="plaster2")                         # bed1 | bed2
    b.wall(0, 4.5, 3.4, 4.5, gaps=[(2.3, 3.2)], mat="plaster3")                         # kitchen
    b.wall(3.4, 4.5, 3.4, 7, gaps=[(0.5, 1.3)], mat="plaster3")                         # bath
    # living room
    b.couch(3.0, 0.55, math.pi); b.table(3.0, 1.6, 1.1, 0.6, 0.42); b.shelf(5.75, 2.5, -math.pi / 2)
    b.table(1.3, 2.8, 1.4, 0.9); [b.chair(1.3 + dx, 2.8 + dy, a) for dx, dy, a in ((-0.45, 0.6, -math.pi / 2), (0.45, 0.6, -math.pi / 2), (-0.45, -0.6, math.pi / 2), (0.45, -0.6, math.pi / 2))]
    b.poster(0.04, 1.5, 0); b.poster(3.0, 0.04, math.pi / 2, w=1.4); b.poster(5.96, 3.6, math.pi)
    # kitchen: counters + island
    b.box(1.7, 6.7, 0.45, 1.6, 0.3, 0.45, 0, "white"); b.box(0.3, 5.6, 0.45, 0.3, 0.8, 0.45, 0, "white")
    b.box(1.7, 5.5, 0.45, 0.6, 0.35, 0.45, 0, "wood"); b.poster(1.7, 6.96, -math.pi / 2)
    # bath
    b.box(4.0, 6.6, 0.4, 0.4, 0.35, 0.4, 0, "white"); b.box(5.3, 6.3, 0.3, 0.6, 0.7, 0.3, 0, "white")
    # bedrooms
    b.bed(8.5, 1.4, 0); b.shelf(9.75, 3.0, math.pi / 2, 0.8); b.poster(9.96, 1.4, math.pi)
    b.bed(8.5, 5.6, 0); b.table(6.6, 6.5, 1.0, 0.5, 0.75); b.chair(6.6, 6.0, math.pi / 2); b.poster(8.0, 6.96, -math.pi / 2)
    b.clutter(0.6, 1.0, 5.5, 2.0, 4, keep=((3.0, 1.6, 0.8), (1.3, 2.8, 1.2), (4.8, 1.0, 0.7)))
    b.e.places = {"living room": (2.5, 2.0), "kitchen": (1.6, 4.9), "bathroom": (4.6, 5.6), "hall": (4.8, 4.0),
                  "bedroom 1": (7.5, 2.0), "bedroom 2": (7.4, 4.9), "couch": (3.0, 1.15), "dining table": (2.6, 2.8),
                  "front door": (5.5, 0.4)}
    b.e.start = (4.8, 1.0, math.pi / 2); b.e.extent = (-0.2, -0.2, 10.2, 7.2); return b.e


def env_corridor(seed):
    b = B("corridor", seed)
    b.wall(0, 0, 20, 0, gaps=[(4, 4.85), (11, 11.8)], mat="plaster2"); b.wall(0, 1.4, 20, 1.4, gaps=[(7, 7.9), (15, 15.9)], mat="plaster2")
    b.wall(0, 0, 0, 1.4); b.wall(20, 0, 20, 1.4)
    for i in range(0, 20, 3):             # (fix 2026-10-13: the x=7.5 poster hung in the office-b doorway at mast height)
        if not 6.6 < i + 1.5 < 8.3: b.poster(i + 1.5, 1.36, -math.pi / 2, w=0.6)
        b.poster(i + 0.2, 0.04, math.pi / 2, w=0.5)
    def side(x0, x1, y_c, y_far, mat):           # side room sharing the corridor wall (door already cut there)
        b.wall(x0, y_c, x0, y_far, mat=mat); b.wall(x1, y_c, x1, y_far, mat=mat); b.wall(x0, y_far, x1, y_far, mat=mat)
    side(3, 6, 0, -3.5, "plaster3"); side(6, 9, 1.4, 4.6, "plaster3"); side(10, 13, 0, -3.0, "plaster"); side(14, 17, 1.4, 4.4, "plaster")
    b.table(4.5, -2.5); b.shelf(8.6, 3.0, math.pi / 2); b.table(11.5, -2.0, 1.4, 0.7); b.couch(15.5, 3.8, math.pi)
    b.cyl(9.5, 0.3, 0.15, 0.9); b.box(13.0, 1.1, 0.4, 0.3, 0.2, 0.4, 0, "metal")            # bin + cart narrow the corridor
    b.e.places = {"west end": (0.8, 0.7), "east end": (19.2, 0.7), "office a": (4.5, -1.5), "office b": (7.5, 3.0),
                  "office c": (11.4, -1.0), "lounge": (15.5, 2.6)}
    b.e.start = (0.8, 0.7, 0.0); b.e.extent = (-0.2, -3.7, 20.2, 4.8); return b.e


def env_clutter(seed):
    b = B("clutter", seed); W, H = 7.0, 6.0
    b.room(0, 0, W, H, mat="plaster3")
    for k in range(6): b.poster(b.rng.uniform(0.5, W - 0.5), H - 0.04, -math.pi / 2)
    b.clutter(0.6, 0.6, W - 0.6, H - 0.6, 28, keep=((1.0, 1.0, 0.7), (6.0, 5.0, 0.6)))
    b.e.places = {"start corner": (1.0, 1.0), "far corner": (6.0, 5.0)}
    b.e.start = (1.0, 1.0, 0.6); b.e.extent = (-0.2, -0.2, W + 0.2, H + 0.2); return b.e


def env_office(seed):
    b = B("office", seed); W, H = 12.0, 8.0
    b.room(0, 0, W, H, mat="white", doors={"s": [(0.6, 1.5)]})
    for i in range(3):
        for k in range(3):
            x, y = 2.5 + 3.2 * i, 1.8 + 2.3 * k
            b.table(x, y, 1.6, 0.8); b.chair(x, y - 0.65, math.pi / 2); b.box(x + 0.9, y, 0.3, 0.2, 0.3, 0.3, 0, "metal")
    for k in range(5): b.poster(W - 0.04, 1 + 1.5 * k, math.pi); b.poster(0.04, 1 + 1.5 * k, 0)
    b.room(9.5, 6.0, 12, 8, mat="plaster2", doors={"w": [(0.4, 1.3)]})
    b.e.places = {"entrance": (1.0, 0.6), "meeting room": (10.8, 7.0), "desk 5": (5.7, 3.5), "corner": (11.3, 0.6)}
    b.e.start = (1.0, 0.6, math.pi / 2); b.e.extent = (-0.2, -0.2, W + 0.2, H + 0.2); return b.e


def env_maze(seed):
    """Perfect maze, 1.2 m cells: dead ends + loops-free -> tests escape and long-horizon planning."""
    b = B("maze", seed); n, m, c = 6, 5, 1.2; rng = b.rng
    E = {((i, j), (i + 1, j)) for i in range(n - 1) for j in range(m)} | {((i, j), (i, j + 1)) for i in range(n) for j in range(m - 1)}
    seen = {(0, 0)}; st = [(0, 0)]; open_ = set()
    while st:
        i, j = st[-1]; nb = [(i + a, j + d) for a, d in ((1, 0), (-1, 0), (0, 1), (0, -1)) if 0 <= i + a < n and 0 <= j + d < m and (i + a, j + d) not in seen]
        if not nb: st.pop(); continue
        q = nb[rng.integers(len(nb))]; open_.add(tuple(sorted([(i, j), q]))); seen.add(q); st.append(q)
    b.room(0, 0, n * c, m * c, mat="plaster2")
    mats = ["plaster", "plaster2", "plaster3", "wood"]
    for (p, q) in E:
        if tuple(sorted([p, q])) in open_: continue
        (i, j), (k, l) = p, q
        if k == i + 1: b.wall((i + 1) * c, j * c, (i + 1) * c, (j + 1) * c, mat=mats[(i + j) % 4], h=1.2)
        else: b.wall(i * c, (j + 1) * c, (i + 1) * c, (j + 1) * c, mat=mats[(i + 2 * j) % 4], h=1.2)
    b.e.places = {"entry": (0.6, 0.6), "exit": (n * c - 0.6, m * c - 0.6), "middle": (n * c / 2 - 0.6, m * c / 2 - 0.6)}
    b.e.start = (0.6, 0.6, 0.0); b.e.extent = (-0.2, -0.2, n * c + 0.2, m * c + 0.2); return b.e


def env_doors(seed):
    """Row of doorways 0.55..0.95 m: which ones Kevin (0.42 m wide, R240 planning) should and shouldn't take."""
    b = B("doors", seed); W = [0.55, 0.65, 0.75, 0.85, 0.95]
    b.room(0, 0, 3 * len(W) + 1, 5, mat="plaster")
    for k, w in enumerate(W):
        x = 2 + 3 * k; b.wall(x - 1.4, 2.5, x + 1.4, 2.5, gaps=[(1.4 - w / 2, 1.4 + w / 2)], mat="wood")
        if k < len(W) - 1: b.wall(x + 1.5, 0, x + 1.5, 2.5, mat="plaster3")
        b.poster(x, 4.96, -math.pi / 2); b.e.places["door %.2f" % w] = (x, 2.5)
    b.e.places.update({"north hall": (8, 4.0)})
    b.e.start = (2.0, 1.0, math.pi / 2); b.e.extent = (-0.2, -0.2, 3 * len(W) + 1.2, 5.2); return b.e


def env_casita(seed):
    """James's real house walls (sim/casitahouse-walls-fixed.stl), textured, with furniture placed in open space."""
    import trimesh
    stl = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sim", "casitahouse-walls-fixed.stl")
    m = trimesh.load(stl); ext = m.bounds; sz = ext[1] - ext[0]
    scale = 0.001 if sz.max() > 100 else 1.0
    b = B("casita", seed)
    b.e.geoms.append(Geom("mesh", (-ext[0][0] * scale, -ext[0][1] * scale, -ext[0][2] * scale), (scale, scale, scale), 0.0, "plaster", stl))
    W, H = sz[0] * scale, sz[1] * scale
    b.e.extent = (-0.2, -0.2, W + 0.2, H + 0.2)
    b.e._mesh = (m, scale, -ext[0] * scale)
    # ---- furnishing (2026-10-14, James): realistic sizes; rooms read off the wall STL section at 0.5 m
    #   living room (south, y 0-4.7) | front door = west wall gap y 3.1-4.8 | kitchen x 3.4-7, y 3-7.3 (around the
    #   angled island) | laundry + back door x 4.6-7, y 7.3-8.4 | hallway | bedroom 1 (NW) | bedroom 2 (NE)
    P = math.pi
    b.couch(2.6, 0.62, 0.0, w=2.2)                          # 3-seat couch against the south wall
    b.couch(0.62, 1.7, -P / 2, w=1.6)                       # loveseat along the west wall
    b.table(2.6, 1.75, 1.1, 0.55, 0.42, mat="wood")         # coffee table
    b.table(5.6, 6.2, 1.1, 0.75, 0.75, mat="wood")          # kitchen table + 4 chairs
    for x, y, yw in ((4.86, 6.2, 0.0), (6.34, 6.2, P), (5.6, 5.62, P / 2), (5.6, 6.78, -P / 2)): b.chair(x, y, yw)
    b.bed(1.4, 12.7)                                        # bedroom 1: queen bed 2.0 x 1.6
    b.table(2.7, 13.5, 1.2, 0.6, 0.75, mat="wood")          # computer desk + monitor + office chair
    b.box(2.7, 13.65, 0.95, 0.27, 0.03, 0.18, 0.0, "dark")
    b.chair(2.7, 12.95, P / 2)
    b.bed(5.6, 12.3)                                        # bedroom 2
    b.e.places = {"living room": (3.6, 2.2), "couch": (3.7, 1.6), "front door": (0.75, 4.0), "kitchen": (4.3, 5.9),
                  "kitchen table": (5.6, 4.95), "laundry": (5.2, 7.8), "back door": (5.95, 7.8), "hallway": (2.0, 7.0),
                  "bedroom": (1.9, 11.2), "bed": (1.4, 11.45), "desk": (2.9, 12.4), "bedroom 2": (5.0, 11.0)}
    b.e.start = (3.6, 2.2, P / 2)
    # floor: old lumpy carpet everywhere except these hard-floor zones (x0, y0, x1, y1); ~1 cm step at the edges
    b.e.hard_zones = [(3.4, 3.0, 7.0, 7.3), (4.6, 7.3, 7.0, 8.4), (0.0, 3.0, 1.3, 4.9)]
    b.e.dog = True                                          # wandering yorkie (emu.world)
    return b.e


# ---------------------------------------------------------------- outdoor (terrain = heightfield built by emu.world)
# env.terrain(xx, yy, rng) -> absolute ground z (m); env.ground = (friction, rolling, condim, rgb material);
# env.hazard(x, y) -> True where the robot centre means a wheel is over a drop (counted as a collision);
# env.blades = [(x, y, h)] visual-only grass blades (depth sees them, wheels do not).
def _joints(xx, yy, pitch=1.5, w=0.02, d=0.006):   # 2 cm grid: joint is one cell wide
    g = (np.abs(((xx + 1e3) % pitch) - pitch / 2) > pitch / 2 - w / 2) | (np.abs(((yy + 1e3) % pitch) - pitch / 2) > pitch / 2 - w / 2)
    return -d * g

def env_sidewalk(seed):
    """12 x 5 m street edge: concrete sidewalk y 0..2.4 (2 % cross-slope, expansion joints every 1.5 m, a raised
    slab lip), real 15 cm curb at y=0 with a 1.2 m curb-cut ramp at x 6..7.2, asphalt road y<0, posts + bin."""
    b = B("sidewalk", seed); rng = b.rng
    RX0, RX1 = 6.0, 7.2
    def terrain(xx, yy, r):
        z = np.where(yy >= 0, -0.01 * (2.4 - np.clip(yy, 0, 2.4)) + _joints(xx, yy), -0.15)          # 1 % fall to curb
        ramp = (xx >= RX0) & (xx <= RX1) & (yy >= 0) & (yy < 1.0)
        z = np.where(ramp, -0.15 + 0.136 * yy, z)                                                   # 1:7 curb cut
        lip = (xx > 2.95) & (xx < 4.5) & (yy > 0.4) & (yy < 2.4); z = np.where(lip, z + 0.008, z)  # heaved slab, 8 mm lip
        return np.where(yy < 0, z + 0.002 * r.standard_normal(z.shape), z)                          # asphalt texture
    b.e.terrain = terrain; b.e.ground = (0.9, 0.003, 3, "concrete")
    b.e.hazard = lambda x, y: (-0.05 < y < 0.12) and not (RX0 + 0.25 < x < RX1 - 0.25)
    b.e.hazard_line = [(0.0, 0.0, RX0, 0.0), (RX1, 0.0, 12.0, 0.0)]
    for x in (1.5, 4.5, 9.0): b.cyl(x, 2.2, 0.05, 1.0, mat="metal")            # sign posts at the back edge
    b.box(10.5, 1.8, 0.45, 0.3, 0.3, 0.45, mat="dark")                            # trash bin
    b.wall(0, 2.5, 12, 2.5, h=2.4, mat="plaster2")                                # building facade
    b.e.places = {"post": (1.5, 1.6), "ramp_top": (6.6, 1.3), "road": (6.6, -1.2), "bin": (10.5, 1.0), "lip": (3.7, 1.4)}
    b.e.start = (0.8, 1.2, 0.0); b.e.extent = (-0.2, -2.5, 12.2, 2.6); return b.e

def env_parking(seed):
    """10 x 8 m asphalt lot: 3 % slope, potholes 2-4 cm deep / 20-40 cm, gravel patch, parked-car boxes, poles."""
    b = B("parking", seed); rng = b.rng
    holes = [(rng.uniform(1, 9), rng.uniform(1, 7), rng.uniform(0.10, 0.20), rng.uniform(0.02, 0.04)) for _ in range(7)]
    def terrain(xx, yy, r):
        z = 0.03 * (xx - 5.0) * 0.5 + 0.004 * _smooth(r, xx.shape, 0.4 / (xx[0, 1] - xx[0, 0]))
        for (x, y, rad, dep) in holes:
            d = np.hypot(xx - x, yy - y); z = z - dep * np.clip(1.2 - d / rad, 0, 1).clip(0, 1) ** 0.5 * (d < rad * 1.2)
        g = (xx > 6.5) & (xx < 9.5) & (yy > 5.5) & (yy < 7.8)
        z = z + g * np.abs(r.normal(0, 0.005, z.shape)) * (r.random(z.shape) < 0.3)                # gravel 5-15 mm stones
        return z
    b.e.terrain = terrain; b.e.ground = (0.85, 0.004, 3, "dark")
    for x, y in ((2.0, 6.5), (5.0, 6.5)): b.box(x, y, 0.7, 0.9, 2.2 / 2, 0.7, mat="metal")       # parked cars
    for x, y in ((8.0, 2.0), (3.0, 3.5)): b.cyl(x, y, 0.08, 1.2, mat="white")
    b.e.places = {"lot_center": (5.0, 3.0), "gravel": (8.0, 6.6), "car_gap": (3.5, 6.0), "corner": (9.0, 1.0), "entry": (1.0, 1.0)}
    b.e.start = (1.0, 2.0, 0.0); b.e.extent = (-0.2, -0.2, 10.2, 8.2); return b.e

def env_lawn(seed):
    """8 x 6 m back yard lawn: bumps 2-4 cm over 0.3-1 m, 6 % slope, 5-9 cm grass blades (visual), low/variable
    friction, high rolling resistance; tree, bench, flower-bed edging (4 cm border)."""
    b = B("lawn", seed); rng = b.rng
    def terrain(xx, yy, r):
        z = 0.06 * yy * 0.5 + 0.02 * _smooth(r, xx.shape, 0.8 / (xx[0, 1] - xx[0, 0])) + 0.01 * _smooth(r, xx.shape, 0.25 / (xx[0, 1] - xx[0, 0]))
        return z
    b.e.terrain = terrain; b.e.ground = (0.55, 0.03, 6, "grass")
    b.e.blades = [(x, y, rng.uniform(0.04, 0.09)) for x, y in rng.uniform((0, 0), (8, 6), (900, 2))]   # 2500 single geoms -> cam 10 Hz
    b.cyl(6.0, 4.5, 0.18, 2.5, mat="wood")                                                         # tree trunk
    b.box(2.0, 5.3, 0.22, 0.8, 0.25, 0.22, mat="wood")                                             # bench
    b.box(4.0, 0.3, 0.02, 1.5, 0.03, 0.04, mat="white")                                            # bed edging
    b.e.places = {"tree": (5.3, 4.0), "bench": (2.0, 4.6), "middle": (4.0, 3.0), "far": (7.2, 1.0), "gate": (0.8, 0.8)}
    b.e.start = (1.0, 2.0, 0.0); b.e.extent = (-0.2, -0.2, 8.2, 6.2); return b.e

def _smooth(rng, shape, k):
    import cv2
    k = max(2, int(k)); g = rng.standard_normal((shape[0] // k + 3, shape[1] // k + 3)).astype(np.float32)
    z = cv2.resize(g, (shape[1] + 2 * k, shape[0] + 2 * k), interpolation=cv2.INTER_CUBIC)[k:k + shape[0], k:k + shape[1]]
    return z / (np.abs(z).max() + 1e-9)


def env_hall(seed):
    """Open space (A/B for gyro vs wheel heading): 14 x 10 m plain hall, 4 thin pillars, few features; walls often
    beyond scan range -> long drives and turns with weak scan/vision constraints."""
    b = B("hall", seed); W, H = 14.0, 10.0
    b.room(0, 0, W, H, mat="plaster")
    for x, y in ((4.5, 3.5), (9.5, 3.5), (4.5, 6.5), (9.5, 6.5)): b.cyl(x, y, 0.12, 2.0, mat="white")
    b.e.places = {"far east": (12.8, 1.5), "north east": (12.5, 8.6), "north west": (1.5, 8.5), "center": (7.0, 5.0), "south": (7.0, 1.2)}
    b.e.start = (1.2, 1.2, 0.0); b.e.extent = (-0.2, -0.2, W + 0.2, H + 0.2); return b.e


ENVS = {"room": env_room, "apartment": env_apartment, "corridor": env_corridor, "clutter": env_clutter,
        "office": env_office, "maze": env_maze, "doors": env_doors, "casita": env_casita,
        "sidewalk": env_sidewalk, "parking": env_parking, "lawn": env_lawn, "hall": env_hall}


# ---------------------------------------------------------------- ground truth
def _raster_box(g, occ, org):
    (x, y, z), (hx, hy, hz) = g.pos, g.size
    if z + hz < BAND[0] or z - hz > BAND[1]: return
    c, s = math.cos(g.yaw), math.sin(g.yaw); R = math.hypot(hx, hy)
    x0, x1 = int((x - R - org[0]) / GT_RES), int((x + R - org[0]) / GT_RES) + 1
    y0, y1 = int((y - R - org[1]) / GT_RES), int((y + R - org[1]) / GT_RES) + 1
    x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, occ.shape[1]), min(y1, occ.shape[0])
    if x1 <= x0 or y1 <= y0: return
    xx, yy = np.meshgrid(org[0] + (np.arange(x0, x1) + 0.5) * GT_RES, org[1] + (np.arange(y0, y1) + 0.5) * GT_RES)
    dx, dy = xx - x, yy - y; u, v = c * dx + s * dy, -s * dx + c * dy
    occ[y0:y1, x0:x1] |= (np.abs(u) <= hx + GT_RES / 2) & (np.abs(v) <= hy + GT_RES / 2)


def _raster_cyl(g, occ, org):
    (x, y, z), (r, hz) = g.pos, g.size
    if z + hz < BAND[0] or z - hz > BAND[1]: return
    yy, xx = np.mgrid[0:occ.shape[0], 0:occ.shape[1]]
    occ |= np.hypot(org[0] + (xx + 0.5) * GT_RES - x, org[1] + (yy + 0.5) * GT_RES - y) <= r + GT_RES / 2


def build(name: str, seed: int = 0) -> Env:
    e = ENVS[name](seed)
    x0, y0, x1, y1 = e.extent
    occ = np.zeros((int(math.ceil((y1 - y0) / GT_RES)), int(math.ceil((x1 - x0) / GT_RES))), bool)
    org = (x0, y0)
    for g in e.geoms:
        if g.kind == "box": _raster_box(g, occ, org)
        elif g.kind == "cyl": _raster_cyl(g, occ, org)
    for (hx0, hy0, hx1, hy1) in getattr(e, "hazard_line", []):      # curb edge = negative obstacle in GT
        _raster_box(Geom("box", ((hx0 + hx1) / 2, (hy0 + hy1) / 2, 0.3), ((hx1 - hx0) / 2, 0.05, 0.3)), occ, org)
    if hasattr(e, "_mesh"):
        m, s, off = e._mesh; occ0 = occ.copy(); e.wall_levels = []; wr = e.wall_res = WALL_RES
        wshape = (int(math.ceil((y1 - y0) / wr)), int(math.ceil((x1 - x0) / wr)))
        for zc in np.arange(BAND[0] + SLICE_DZ / 2, BAND[1], SLICE_DZ):     # slab [zc - dz/2, zc + dz/2] (world.py)
            lv = np.zeros(wshape, bool); e.wall_levels.append((float(zc), lv))
            # raw triangle/plane segments (Path3D.discrete chains open segments and drew phantom diagonals)
            import trimesh
            segs = trimesh.intersections.mesh_plane(m, plane_normal=[0, 0, 1], plane_origin=[0, 0, (zc - off[2]) / s])
            for seg in segs:
                P = seg[:, :2] * s + off[:2]
                for a, bb in zip(P[:-1], P[1:]):
                    n = int(np.hypot(*(bb - a)) / (wr / 2)) + 2
                    q = a + (bb - a) * np.linspace(0, 1, n)[:, None]
                    c = ((q - org) / GT_RES).astype(int); ok = (c[:, 0] >= 0) & (c[:, 1] >= 0) & (c[:, 0] < occ.shape[1]) & (c[:, 1] < occ.shape[0])
                    occ[c[ok, 1], c[ok, 0]] = True
                    c = ((q - org) / wr).astype(int); ok = (c[:, 0] >= 0) & (c[:, 1] >= 0) & (c[:, 0] < wshape[1]) & (c[:, 1] < wshape[0])
                    lv[c[ok, 1], c[ok, 0]] = True
        from scipy import ndimage
        e.wall_occ = ndimage.binary_dilation(occ & ~occ0, iterations=1)   # mesh walls only (all heights)
        # per-height slabs -> collision boxes (world.py): a counter overhang at 0.9 m must not become a floor-level wall
        # at WALL_RES and NOT dilated: the cells already contain the wall surface, so the box skin stands at most one cell
        # diagonal (1.4 cm) proud of the rendered wall.  The old 2 cm raster + 1-cell dilation stood 2-4.3 cm proud: an
        # invisible skin that ate the 3.5 cm safety margin (the casita wall "contacts" 2026-10-14..17 were all 2.5-3.7 cm
        # in front of the wall surface the cameras see).
        occ = ndimage.binary_dilation(occ, iterations=2)
        if not e.places: _auto_places(e, occ, org)
    e.gt, e.gt_origin = occ, org
    from scipy import ndimage
    clr = ndimage.distance_transform_edt(~occ) * GT_RES
    r, c = e.gt_world_to_rc(e.start[0], e.start[1])
    if clr[r, c] < 0.30:               # never spawn inside/against furniture: move to nearest point with 30 cm clearance
        ok = np.argwhere(clr >= 0.35); k = np.argmin(np.hypot(ok[:, 0] - r, ok[:, 1] - c))
        e.start = (org[0] + (ok[k, 1] + 0.5) * GT_RES, org[1] + (ok[k, 0] + 0.5) * GT_RES, e.start[2])
    e.clearance = clr
    return e


def _auto_places(e, occ, org):
    """For imported scenes: label the 5 largest free 'rooms' (distance-transform maxima) and pick a start."""
    from scipy import ndimage
    free = ~occ
    lab, n = ndimage.label(free)
    if n == 0: return
    sizes = ndimage.sum(free, lab, range(1, n + 1)); inside = 1 + int(np.argsort(sizes)[-2] if n > 1 else 0)   # [-1] = outside
    dt = ndimage.distance_transform_edt(lab == inside) * GT_RES
    pts = []
    d = dt.copy()
    for k in range(6):
        r, c = np.unravel_index(np.argmax(d), d.shape)
        if d[r, c] < 0.45: break
        pts.append((org[0] + (c + 0.5) * GT_RES, org[1] + (r + 0.5) * GT_RES))
        rr = int(max(1.5, d[r, c] * 1.5) / GT_RES); d[max(0, r - rr):r + rr, max(0, c - rr):c + rr] = 0
    e.places = {"room %d" % (i + 1): p for i, p in enumerate(pts)}
    if pts: e.start = (pts[0][0], pts[0][1], 0.0)


def gt_png(e: Env, path: str):
    import cv2
    img = np.where(e.gt, 30, 235).astype(np.uint8)[::-1].copy()
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    def px(x, y): return int((x - e.gt_origin[0]) / GT_RES), img.shape[0] - 1 - int((y - e.gt_origin[1]) / GT_RES)
    for k, (x, y) in e.places.items():
        cv2.circle(img, px(x, y), 6, (0, 0, 255), -1); cv2.putText(img, k, px(x + 0.1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 0, 0), 1)
    sx, sy, sth = e.start; p0 = px(sx, sy)
    cv2.arrowedLine(img, p0, (p0[0] + int(15 * math.cos(sth)), p0[1] - int(15 * math.sin(sth))), (0, 160, 0), 2)
    cv2.imwrite(path, img)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2 or sys.argv[1] == "list":
        print(" ".join(ENVS)); sys.exit()
    if sys.argv[1] == "png":
        names = sys.argv[2:3] or list(ENVS)
        for nm in names:
            e = build(nm, int(sys.argv[3]) if len(sys.argv) > 3 else 0)
            gt_png(e, "/tmp/env_%s.png" % nm); print(nm, e.gt.shape, len(e.geoms), "geoms", e.places)

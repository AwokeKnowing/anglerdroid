"""Render realism for the emulator's sensor cameras (KEVIN_EMU_REAL=1, default on).  Physics/collision are untouched:
everything added here is visual-only (contype/conaffinity 0, outside the GT raster) and the casita walls keep their
collision boxes from the wall STL section.

  textures   CC0 photo-scanned PBR albedo maps (ambientCG, https://ambientcg.com, CC0 1.0) fetched once into
             ~/.cache/kevin_emu/pbr_src by fetch_pbr(); per-material tint (painted walls per room), repeat, specular.
             Missing downloads fall back to the procedural textures (envs.ensure_textures), never fail the sim.
  walls      casita wall STL -> OBJ with per-face planar UVs (dominant normal axis, metres), split by room so each
             room gets its own paint; skirting boards; a ceiling (group 1: hidden in the third-person viewer).
  decor      visual-only household objects: framed pictures, wall shelves with books, lamps, plants, kitchen
             counters/backsplash/cabinets/fridge along walls (outside Kevin's band), rugs (flush), TV, curtains.
  lights     sun (directional, shadows) entering through the window/door openings, warm ceiling lamps per room,
             weak headlight as the ambient (indirect) term.
  CamFX      camera pipeline applied to rendered RGB/IR: lens distortion (Brown k1,k2 via remap from a wider pinhole
             render), auto-exposure (EMA on mean luminance), vignetting, motion blur along the image-x axis from the
             yaw rate x exposure time, shot + read noise in linear space, gamma, 8-bit quantisation.
Mast camera (USB16MP01 / IMX298, listed "118 deg no distortion"): a lens spec FOV is normally DIAGONAL, so the twin
truth is DFOV 118 deg -> HFOV 106.2 deg at 4:3 with a small residual barrel (k1 -0.05).  This is TRUTH (withheld,
written to /dev/shm/kevin_emu_truth.json for eval tools); the robot keeps its own belief (KEVIN_MAST_HFOV)."""
from __future__ import annotations
import glob, io, json, math, os, zipfile
import numpy as np

CACHE = os.path.join(os.environ.get("KEVIN_EMU_CACHE", os.path.expanduser("~/.cache/kevin_emu")))
TEX = os.environ.get("KEVIN_EMU_TEX", "v2")       # texture set: v2 = feature-realistic (emu/texture_v2.py, default 2026-10-20); v1 = tinted scans (2026-10-17)
SRC = os.path.join(CACHE, "pbr_src"); OUT = os.path.join(CACHE, "tex_real" if TEX == "v1" else "tex_real_" + TEX)
REAL = os.environ.get("KEVIN_EMU_REAL", "1") == "1"
SHADOW = int(os.environ.get("KEVIN_EMU_SHADOW", "4096"))      # sun shadow-map size; 0 disables shadows (live-rate knob)
#  material: (ambientCG id, tint rgb (None = as scanned), texrepeat per metre, specular, shininess)
PBR = {
    "plaster":  ("PaintedPlaster017", (0.93, 0.91, 0.86), 0.7, 0.05, 0.1),   # warm white
    "plaster2": ("Plaster001", (0.80, 0.86, 0.92), 0.7, 0.05, 0.1),          # pale blue
    "plaster3": ("Plaster007", (0.93, 0.86, 0.72), 0.7, 0.05, 0.1),          # light ochre
    "plaster4": ("PaintedPlaster017", (0.78, 0.85, 0.76), 0.7, 0.05, 0.1),   # sage
    "ceiling":  ("Plaster002", (0.95, 0.95, 0.94), 0.5, 0.02, 0.05),
    "wood":     ("Wood051", None, 1.5, 0.25, 0.4), "woodfloor": ("WoodFloor051", None, 0.6, 0.3, 0.5),
    "fabric":   ("Fabric026", (0.45, 0.50, 0.60), 2.0, 0.02, 0.05), "fabric2": ("Fabric061", (0.75, 0.45, 0.40), 2.0, 0.02, 0.05),
    "metal":    ("Metal049A", None, 2.0, 0.6, 0.7), "white": ("Plaster002", (0.97, 0.97, 0.96), 1.0, 0.3, 0.5),
    "dark":     ("Leather037", (0.25, 0.25, 0.27), 2.0, 0.2, 0.3), "concrete": ("Concrete034", None, 0.5, 0.05, 0.1),
    "tile":     ("Tiles107", None, 1.0, 0.45, 0.6), "carpet": ("Carpet016", None, 1.2, 0.0, 0.0),
    "marble":   ("Marble012", None, 1.0, 0.5, 0.7), "wallpaper": ("Wallpaper001A", None, 1.0, 0.05, 0.1),
}
EXTRA_MATS = ["plaster4", "ceiling", "woodfloor", "marble", "wallpaper", "art0", "art1", "art2", "art3", "book", "leaf", "lampshade", "screen"] + (["rug0", "rug1", "rug2", "outlet", "switch", "poster0", "poster1", "poster2", "grille", "clock", "smoke", "blinds", "cabinet"] if TEX != "v1" else [])


def fetch_pbr(ids=None):
    """One-time download of the CC0 ambientCG 1K-JPG sets (albedo used; roughness/normal kept for a PBR renderer)."""
    import urllib.request
    os.makedirs(SRC, exist_ok=True)
    for i in ids or sorted({v[0] for v in PBR.values()}):
        if os.path.isdir(os.path.join(SRC, i)): continue
        req = urllib.request.Request("https://ambientcg.com/get?file=%s_1K-JPG.zip" % i, headers={"User-Agent": "Mozilla/5.0"})
        z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(req, timeout=60).read())); os.makedirs(os.path.join(SRC, i))
        for n in z.namelist():
            if any(k in n for k in ("Color", "Roughness", "NormalGL")): z.extract(n, os.path.join(SRC, i))


def _art(rng, n=512):
    """Framed picture: smooth colour fields + strokes (a photo-like low-frequency image with edges), dark frame, mat."""
    import cv2
    img = np.zeros((n, n, 3), np.float32)
    for c in range(3):
        g = rng.random((6, 6)).astype(np.float32); img[..., c] = cv2.resize(g, (n, n), interpolation=cv2.INTER_CUBIC)
    for _ in range(int(rng.integers(3, 9))):
        p = rng.integers(40, n - 40, 4); cv2.line(img, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), rng.random(3).tolist(), int(rng.integers(3, 18)))
    for _ in range(int(rng.integers(1, 4))):
        cv2.circle(img, tuple(int(v) for v in rng.integers(60, n - 60, 2)), int(rng.integers(15, 70)), rng.random(3).tolist(), -1)
    img = np.clip(img, 0, 1); img[:28] = img[-28:] = 0.12; img[:, :28] = img[:, -28:] = 0.12
    img[28:50, 28:-28] = img[-50:-28, 28:-28] = 0.92; img[28:-28, 28:50] = img[28:-28, -50:-28] = 0.92
    return img


def prepare_textures(proc_dir):
    """-> {material: png path} for every material compose() may use.  Real (tinted ambientCG) where available."""
    import cv2
    os.makedirs(OUT, exist_ok=True); paths = {}
    for name, (aid, tint, *_r) in PBR.items():
        p = os.path.join(OUT, name + ".png")
        if not os.path.exists(p):
            src = glob.glob(os.path.join(SRC, aid, "*_Color.jpg"))
            if not src: continue
            im = cv2.imread(src[0]).astype(np.float32) / 255.0
            im = cv2.resize(im, (1024, 1024), interpolation=cv2.INTER_AREA)
            if tint is not None:          # keep the scan's detail, set its mean colour to the paint/fabric tint
                im = im / (im.reshape(-1, 3).mean(0) + 1e-6) * np.array(tint[::-1], np.float32)
            if TEX != "v1":               # v2: normal bake, wear, real-scale detail (emu/texture_v2.py)
                from emu import texture_v2
                nf = glob.glob(os.path.join(SRC, aid, "*_NormalGL.jpg"))
                nrm = cv2.resize(cv2.imread(nf[0]), (1024, 1024), interpolation=cv2.INTER_AREA)[..., ::-1].astype(np.float32) / 127.5 - 1 if nf else None
                im = texture_v2.make(name, im, nrm, PBR[name][2], seed=sum(map(ord, name)))
            cv2.imwrite(p, (np.clip(im, 0, 1) * 255).astype(np.uint8))
        paths[name] = p
    if TEX != "v1":                       # v2: patterned rugs (were flat rgba)
        from emu import texture_v2
        for k, pal in enumerate(texture_v2.RUG_PALETTES):
            p = os.path.join(OUT, "rug%d.png" % k)
            if not os.path.exists(p): cv2.imwrite(p, (np.clip(texture_v2.rug(1024, np.random.default_rng(k), [c[::-1] for c in pal]), 0, 1) * 255).astype(np.uint8))
            paths["rug%d" % k] = p
        T = None
        for nm in texture_v2.ITEMS:
            p = os.path.join(OUT, nm + ".png")
            if not os.path.exists(p):
                T = T or texture_v2.item_textures(); cv2.imwrite(p, (np.clip(T[nm][..., ::-1], 0, 1) * 255).astype(np.uint8))
            paths[nm] = p
    rng = np.random.default_rng(42)
    for k in range(4):
        p = os.path.join(OUT, "art%d.png" % k)
        if not os.path.exists(p): cv2.imwrite(p, (_art(rng)[..., ::-1] * 255).astype(np.uint8))
        paths["art%d" % k] = p
    simple = {"book": None, "leaf": (0.20, 0.42, 0.16), "lampshade": (0.95, 0.90, 0.78), "screen": (0.05, 0.05, 0.06)}
    for name, rgb in simple.items():
        p = os.path.join(OUT, name + ".png")
        if not os.path.exists(p):
            n = 256; r = np.random.default_rng(len(name))
            if name == "book":   # book spines: vertical colour bands with title marks
                im = np.zeros((n, n, 3), np.float32); x = 0
                while x < n:
                    w = int(r.integers(8, 26)); im[:, x:x + w] = r.random(3) * 0.7 + 0.15; im[n // 3:n // 3 + 6, x + 2:x + w - 2] = 0.9; x += w
            else: im = np.array(rgb, np.float32)[None, None] * (1 + 0.08 * r.standard_normal((n, n, 1)).astype(np.float32))
            cv2.imwrite(p, (np.clip(im, 0, 1)[..., ::-1] * 255).astype(np.uint8))
        paths[name] = p
    for name in os.listdir(proc_dir):                          # procedural fallbacks for anything not covered
        paths.setdefault(name[:-4], os.path.join(proc_dir, name))
    return paths


def mat_attrs(name):
    if name in PBR:
        _, _, rep, spec, shin = PBR[name]; return dict(texrepeat="%g %g" % (rep, rep), texuniform="true", specular="%g" % spec, shininess="%g" % shin)
    if name.startswith("art") or name.startswith("rug") or name.startswith("poster") or name in ("book", "screen", "outlet", "switch", "grille", "clock", "smoke", "blinds", "cabinet"):
        return dict(texrepeat="1 1", texuniform="false", specular="0.3" if name == "screen" else "0.1", shininess="0.6" if name == "screen" else "0.2")
    return dict(texrepeat="2 2", texuniform="true", specular="0.1", shininess="0.1")


# ------------------------------------------------------------------ casita walls with UVs, per-room paint
def wall_mat_attrs(name):
    """Material for the casita wall OBJs, whose UVs are already in metres: texuniform off (with it on MuJoCo rescales
    the mesh UVs and the paint stretched ~20x along the wall: horizontal streaks, visible once v2 gave paint detail)."""
    a = mat_attrs(name); a["texuniform"] = "false"; return a
CASITA_ROOMS = [  # (x0, y0, x1, y1, material) in world metres; first match wins; default = plaster
    (3.4, 3.0, 7.2, 7.3, "plaster3"),    # kitchen
    (4.6, 7.3, 7.2, 8.5, "white"),       # laundry
    (-0.5, 9.0, 3.6, 15.5, "plaster2"),  # bedroom 1 (NW)
    (3.6, 9.0, 7.6, 15.5, "plaster4" if TEX == "v1" else "wallpaper"),   # bedroom 2 (NE); v2: patterned wallpaper
]


def casita_wall_objs(stl, scale, off):
    """Wall STL -> one OBJ per paint region, per-face planar UVs in metres (u along the wall, v = height)."""
    import trimesh
    d = os.path.join(CACHE, "casita_walls"); os.makedirs(d, exist_ok=True)
    tag = "%d_%g" % (int(os.path.getmtime(stl)), scale) + ("" if TEX == "v1" else "_" + TEX)
    done = [p for p in sorted(glob.glob(os.path.join(d, tag + "_*.obj"))) if "_" not in os.path.basename(p)[len(tag) + 1:-4]]   # not another set's
    if done: return [(p, os.path.basename(p)[len(tag) + 1:-4]) for p in done]
    m = trimesh.load(stl); V = m.vertices * scale + off; F = m.faces; N = m.face_normals
    C = V[F].mean(1) + 0.06 * N                   # sample just in front of the face: inner side picks its room
    reg = np.full(len(F), "plaster", object)
    for x0, y0, x1, y1, mat in CASITA_ROOMS[::-1]:
        inside = (C[:, 0] >= x0) & (C[:, 0] < x1) & (C[:, 1] >= y0) & (C[:, 1] < y1); reg[inside] = mat
    out = []
    for mat in sorted(set(reg)):
        idx = np.nonzero(reg == mat)[0]; lines = []; vt = []; k = 0
        for fi in idx:
            n = np.abs(N[fi]); ax = int(np.argmax(n)); P = V[F[fi]]
            if ax == 2: uv = P[:, [0, 1]]            # horizontal faces (wall tops / sills)
            elif ax == 0: uv = P[:, [1, 2]]          # wall facing x: u = y, v = z
            else: uv = P[:, [0, 2]]                  # wall facing y: u = x, v = z
            for j in range(3): lines.append("v %.5f %.5f %.5f" % tuple(P[j])); vt.append("vt %.5f %.5f" % tuple(uv[j]))
            lines.append(("f %d/%d %d/%d %d/%d") % (k + 1, k + 1, k + 2, k + 2, k + 3, k + 3)); k += 3
        p = os.path.join(d, "%s_%s.obj" % (tag, mat))
        open(p, "w").write("\n".join([l for l in lines if l[0] == "v"] + vt + [l for l in lines if l[0] == "f"]) + "\n")
        out.append((p, mat))
    return out


def merge_decor(decor):
    """v2: the ~160 visual-only boxes with unit-mapped textures (outlets, switches, posters, photos, clocks, books,
    grilles, blinds) -> one OBJ per (material, group), 24 verts/box with each face carrying the full texture (as MuJoCo
    maps texuniform=false boxes).  Same geometry, ~10 draw calls instead of ~160 (colour-render cost).  {(mat, grp): obj}"""
    import hashlib
    G = {}
    for kind, pos, size, yaw, mat, col in decor:
        if kind != "box" or col is True or mat == "blinds" or mat_attrs(mat)["texuniform"] != "false": continue
        G.setdefault((mat, "1" if col == "ceiling" else "0"), []).append((pos, size, yaw))
    d = os.path.join(CACHE, "decor_" + TEX); os.makedirs(d, exist_ok=True); out = {}
    for key, boxes in G.items():
        if len(boxes) < 2: continue
        V, T, F = [], [], []
        for pos, h, yaw in boxes:
            c = np.array(pos, float); E = [np.array([math.cos(yaw), math.sin(yaw), 0.0]), np.array([-math.sin(yaw), math.cos(yaw), 0.0]), np.array([0.0, 0.0, 1.0])]
            for a in range(3):
                for sg in (-1, 1):
                    b1, b2, f = ((1, 2, sg), (0, 2, -sg), (0, 1, sg))[a]   # u horizontal / v up on side faces, t1 x t2 = outward
                    t1 = E[b1] * h[b1] * f; t2 = E[b2] * h[b2]
                    fc = c + sg * h[a] * E[a]; k = len(V)
                    for u, v in ((0, 0), (1, 0), (1, 1), (0, 1)):
                        V.append(fc + (2 * u - 1) * t1 + (2 * v - 1) * t2); T.append((1 - u, v))   # MuJoCo's box map runs u the other way
                    F += [(k, k + 1, k + 2), (k, k + 2, k + 3)]
        txt = "\n".join(["v %.5f %.5f %.5f" % tuple(p) for p in V] + ["vt %g %g" % t for t in T] +
                         ["f %d/%d %d/%d %d/%d" % (a + 1, a + 1, b + 1, b + 1, c2 + 1, c2 + 1) for a, b, c2 in F]) + "\n"
        p = os.path.join(d, "%s_%s_%s.obj" % (key[0], key[1], hashlib.md5(txt.encode()).hexdigest()[:10]))
        if not os.path.exists(p): open(p, "w").write(txt)
        out[key] = p
    return out


# ------------------------------------------------------------------ decor (visual-only), lights
def _wall_hit(env, x, y, z, ang):
    """First wall-STL hit from (x, y, z) along heading ang (world, m) -> (hx, hy) or None."""
    m, s, off = env._mesh
    o = (np.array([[x, y, z]]) - off) / s; d = np.array([[math.cos(ang), math.sin(ang), 0.0]])
    loc, _, _ = m.ray.intersects_location(o, d, multiple_hits=True)
    if len(loc) == 0: return None
    W = loc * s + off; k = int(np.argmin(np.hypot(W[:, 0] - x, W[:, 1] - y))); return float(W[k, 0]), float(W[k, 1])


def casita_decor(env):
    """[(kind, pos, size, yaw, mat, collide)].  Wall items are snapped onto the wall STL by ray casting from a point
    inside the room (so they sit 3 mm proud of the real wall face); collide=True items (plant pots, lamp base) are
    real obstacles: envs.build adds them to the geoms, so they are in the GT map and in MuJoCo collision."""
    P = math.pi; D = []
    def wall_box(x, y, ang, z, depth, w, h, mat, maxd=None):
        hit = _wall_hit(env, x, y, z, ang)
        if hit is None or (maxd is not None and math.hypot(hit[0] - x, hit[1] - y) > maxd): return
        if maxd is not None and z - h / 2 < 1.3:   # v2 items in the depth cameras' band: the wall must continue past both edges
            d0 = math.hypot(hit[0] - x, hit[1] - y); px, py = -math.sin(ang), math.cos(ang)
            for sg in (-1, 1):
                for zz in (z - h / 2, z + h / 2):
                    o = (w / 2 + 0.03) * sg; hh = _wall_hit(env, x + px * o, y + py * o, zz, ang)
                    if hh is None or abs(math.hypot(hh[0] - x - px * o, hh[1] - y - py * o) - d0) > 0.03: return
        cx, cy = hit[0] - math.cos(ang) * (depth + 0.003), hit[1] - math.sin(ang) * (depth + 0.003)
        D.append(("box", (cx, cy, z), (depth, w / 2, h / 2), ang, mat, False))
    # framed pictures (1.6 cm deep frames), z = centre height
    arts = [(2.7, 2.0, -P / 2, 1.55, 1.0, 0.7), (1.5, 2.5, P, 1.5, 0.6, 0.8), (2.0, 7.0, P, 1.5, 0.7, 0.5), (2.0, 6.0, 0.0, 1.5, 0.9, 0.6),
            (2.5, 11.2, P, 1.5, 0.5, 0.7), (2.0, 11.0, -P / 2, 1.5, 0.8, 0.6), (5.0, 11.0, 0.0, 1.5, 0.9, 0.6), (5.0, 11.0, -P / 2, 1.5, 0.6, 0.45),
            (2.0, 2.5, P / 2, 1.5, 0.6, 0.6)]
    for k, (x, y, a, z, w, h) in enumerate(arts): wall_box(x, y, a, z, 0.008, w, h, "art%d" % (k % 4))
    wall_box(3.0, 2.0, 0.0, 1.25, 0.02, 1.12, 0.66, "screen")                      # TV on the living room east wall
    for (x, y, a) in ((2.7, 2.0, -P / 2), (2.5, 11.2, P)):                         # wall shelves with books
        wall_box(x, y, a, 1.84, 0.12, 1.1, 0.024, "wood"); wall_box(x, y, a, 1.96, 0.09, 1.0, 0.2, "book")
    for (x, y, a, w) in ((5.6, 6.6, 0.0, 1.8), (5.0, 6.8, P / 2, 1.6)):           # kitchen: upper cabinets + tiled band
        wall_box(x, y, a, 1.82, 0.17, w, 0.66, "white" if TEX == "v1" else "cabinet"); wall_box(x, y, a, 1.2, 0.004, w, 0.6, "tile")
    # floor-standing (collide): plant pots + floor lamp base in free corners; foliage / shade / pole visual only
    for (x, y) in ((1.2, 0.42), (0.45, 10.45), (6.7, 10.35)):
        D.append(("cylinder", (x, y, 0.16), (0.14, 0.16), 0.0, "concrete", True))
        D.append(("sphere", (x, y, 0.52), (0.24,), 0.0, "leaf", False)); D.append(("sphere", (x + 0.07, y - 0.05, 0.78), (0.15,), 0.0, "leaf", False))
    D.append(("cylinder", (0.45, 0.45, 0.02), (0.15, 0.02), 0.0, "metal", True))
    D.append(("cylinder", (0.45, 0.45, 0.82), (0.012, 0.8), 0.0, "metal", False)); D.append(("cylinder", (0.45, 0.45, 1.66), (0.18, 0.13), 0.0, "lampshade", False))
    # objects on furniture tops (coffee table 0.42, kitchen table 0.75, desk 0.75)
    D += [("box", (2.4, 1.70, 0.44), (0.12, 0.09, 0.02), 0.3, "book", False), ("cylinder", (2.8, 1.65, 0.47), (0.04, 0.05), 0.0, "white", False),
          ("cylinder", (2.95, 1.9, 0.55), (0.05, 0.13), 0.0, "white", False), ("cylinder", (2.95, 1.9, 0.75), (0.12, 0.08), 0.0, "lampshade", False),
          ("cylinder", (5.6, 6.2, 0.78), (0.15, 0.03), 0.0, "white", False), ("sphere", (5.6, 6.2, 0.88), (0.1,), 0.0, "leaf", False),
          ("cylinder", (5.3, 6.0, 0.8), (0.04, 0.05), 0.0, "fabric2", False), ("box", (2.4, 13.4, 0.77), (0.2, 0.14, 0.012), 0.2, "dark", False)]
    if TEX != "v1":     # v2: everyday wall/ceiling fixtures (outlets, switches, posters, clocks, light fixtures, vents)
        from emu import texture_v2
        texture_v2.casita_items(env, wall_box, D)
    # ceiling (group 1 -> hidden in the third-person viewer), just above the 2.28 m wall tops
    x0, y0, x1, y1 = env.extent
    D.append(("box", ((x0 + x1) / 2, (y0 + y1) / 2, 2.30), ((x1 - x0) / 2 + 0.3, (y1 - y0) / 2 + 0.3, 0.01), 0.0, "ceiling", "ceiling"))
    return D


def casita_lights(env):
    """[(attrs)] MuJoCo <light> attributes: sun + ceiling lamps per room."""
    L = [dict(name="sun", directional="true", castshadow="true" if SHADOW > 0 else "false", pos="-6 -4 8", dir="0.55 0.35 -0.76", diffuse="0.75 0.70 0.62", specular="0.2 0.2 0.2")]
    for nm, (x, y) in (("lamp_living", (2.8, 2.2)), ("lamp_kitchen", (5.3, 5.2)), ("lamp_hall", (2.0, 7.2)), ("lamp_bed1", (1.8, 11.8)), ("lamp_bed2", (5.4, 11.8)), ("lamp_laundry", (5.6, 7.9))):
        L.append(dict(name=nm, pos="%g %g 2.15" % (x, y), dir="0 0 -1", diffuse="0.42 0.38 0.30", specular="0.05 0.05 0.05",
                      attenuation="0.6 0.25 0.08", cutoff="80", exponent="2", castshadow="false"))   # sun-only shadows: 30 Hz budget
    return L


# ------------------------------------------------------------------ camera pipeline
class CamFX:
    """Sensor pipeline on a rendered 8-bit image.  dist=(k1, k2): output is the distorted image of a pinhole render
    made at `render_K` (wider when barrel).  Call with yaw_rate (rad/s) for motion blur."""
    def __init__(self, w, h, K, dist=None, render_K=None, exposure_s=0.012, ae=True, ae_target=0.40, read_e=3.0, full_well=4000.0,
                 vignette=0.25, gray=False, seed=0, nbank=6):
        import cv2
        self.w, self.h, self.K = w, h, np.asarray(K, np.float64); self.exp = exposure_s; self.ae = ae; self.target = ae_target
        self.gain = 1.0; self.read_e = read_e; self.fw = full_well; self.gray = gray
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r2 = ((xx - K[0][2]) / K[0][0]) ** 2 + ((yy - K[1][2]) / K[1][1]) ** 2
        self.vig = (1.0 / (1.0 + vignette * r2) ** 2).astype(np.float32)[..., None]
        self.map = None
        if dist is not None and any(dist):
            pts = np.stack([xx.ravel(), yy.ravel()], 1).astype(np.float64)[:, None, :]
            und = cv2.undistortPoints(pts, self.K, np.array([dist[0], dist[1], 0, 0], np.float64))[:, 0, :]   # normalized pinhole coords
            RK = np.asarray(render_K, np.float64)
            self.map = ((und[:, 0] * RK[0, 0] + RK[0, 2]).reshape(h, w).astype(np.float32), (und[:, 1] * RK[1, 1] + RK[1, 2]).reshape(h, w).astype(np.float32))
        self.lut = ((np.arange(256) / 255.0) ** 2.2).astype(np.float32)
        self.glut = (np.clip(np.arange(4096) / 4095.0, 0, 1) ** (1 / 2.2) * 255 + 0.5).astype(np.uint8)
        c = 1 if gray else 3
        if not gray: self.vig = np.repeat(self.vig, 3, axis=2)
        else: self.vig = self.vig[..., 0]
        self.vig8 = (self.vig ** (1 / 2.2)).astype(np.float32) if vignette > 0 else None   # vignetting in gamma domain
        r = np.random.default_rng(seed)        # noise banks twice the size: a random window per frame (no copy)
        self.bank = [r.standard_normal((2 * h, 2 * w) + ((c,) if c > 1 else ())).astype(np.float32) for _ in range(nbank)]; self.r = r

    @staticmethod
    def render_K_for(w, h, K, dist, margin=1.02):
        """Pinhole intrinsics (same size) whose FOV covers every undistorted ray of the distorted image."""
        import cv2
        u = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1], [w / 2, 0], [0, h / 2]], np.float64)[:, None, :]
        und = cv2.undistortPoints(u, np.asarray(K, np.float64), np.array([dist[0], dist[1], 0, 0], np.float64))[:, 0, :]
        sx = np.abs(und[:, 0]).max() * margin; sy = np.abs(und[:, 1]).max() * margin
        s = max(sx / ((w / 2) / K[0][0]), sy / ((h / 2) / K[1][1]))
        return [[K[0][0] / s, 0, (w - 1) / 2], [0, K[1][1] / s, (h - 1) / 2], [0, 0, 1]]

    def __call__(self, img, yaw_rate=0.0):
        """uint8 HxWx3 (or HxW gray) render -> uint8 sensor image.  Works in the 8-bit (gamma) domain with LUTs that are
        exact per grey level: exposure gain is applied in linear light (LUT), the shot+read noise SD for each output
        level is the linear-domain SD mapped through the gamma slope (LUT), vignetting is a fixed gain map.  ~3 ms."""
        import cv2
        if self.map is not None: img = cv2.remap(img, self.map[0], self.map[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        L = abs(yaw_rate) * self.exp * self.K[0, 0]                            # px smear during the exposure
        if L >= 1.0: img = cv2.blur(img, (int(round(L)) | 1, 1))
        if self.ae:                                                            # AE: damped step toward the target mean
            m = float(np.mean(cv2.mean(img[::8, ::8])[:1 if self.gray else 3]))
            m = float(self.lut[min(255, int(m))]) * self.gain + 1e-4
            self.gain = float(np.clip(self.gain * (self.target / m) ** 0.25, 0.25, 8.0))
        g = round(self.gain, 2)
        if g != getattr(self, "_g", None):                                     # gain LUT + noise-SD LUT for this gain
            lin = np.clip(self.lut * g, 0, 1); v = lin ** (1 / 2.2)
            self._glut8 = (v * 255 + 0.5).astype(np.uint8)
            e = lin * self.fw; sd_lin = np.sqrt(e + self.read_e ** 2 * max(g, 1.0)) / self.fw
            slope = (1 / 2.2) * np.maximum(lin, 1e-3) ** (1 / 2.2 - 1)           # d(gamma)/d(linear)
            self._sdlut = (np.minimum(sd_lin * slope, 0.1) * 255).astype(np.float32); self._g = g
        out = cv2.LUT(img, self._glut8)
        out = cv2.multiply(out, self.vig8, dtype=cv2.CV_8U) if self.vig8 is not None else out
        sd = cv2.LUT(out, self._sdlut)
        oy, ox = int(self.r.integers(self.h)), int(self.r.integers(self.w))
        nz = self.bank[int(self.r.integers(len(self.bank)))][oy:oy + self.h, ox:ox + self.w]
        return cv2.add(out, cv2.multiply(sd, nz), dtype=cv2.CV_8U)


def mast_truth(w=640, h=480):
    """Twin truth for the mast camera: DFOV 118 deg (spec), 4:3 -> HFOV, residual barrel."""
    dfov = float(os.environ.get("KEVIN_MAST_TRUE_DFOV", "118")); k1, k2 = [float(v) for v in os.environ.get("KEVIN_MAST_TRUE_DIST", "-0.05,0.008").split(",")]
    diag = math.hypot(w, h); f = (diag / 2) / math.tan(math.radians(dfov) / 2)
    hfov = math.degrees(2 * math.atan((w / 2) / f))
    return dict(dfov_deg=dfov, hfov_deg=hfov, fx=f, fy=f, cx=(w - 1) / 2, cy=(h - 1) / 2, k1=k1, k2=k2, w=w, h=h)

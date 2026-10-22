"""Texture set v2 (KEVIN_EMU_TEX=v2): feature-realistic albedo for the classic (non-PBR) MuJoCo renderer.

James 2026-10-20 (t617): the forward mast RGB must see as many trackable points as a real home.  Measured (emu/tools/
texprobe.py, FAST-20 32 px cell coverage, mast views along the recorded drives): v1 0.16 vs real-home reference frames
(TUM RGB-D fr1_room 0.46, fr2_pioneer_360 0.31).  Causes: near-flat scans (Wallpaper001A std 1.8, Metal049A 1.6,
Leather037 0.8 grey levels), relief that only exists in the normal maps (MuJoCo classic ignores them), untextured rugs,
no wear.  v2, per material, all tileable (periodic FFT noise, wrap-around drawing) so no tile seams become edges:
  * normal-map bake into the albedo (standard bake for renderers without normal mapping; key light 45 deg elevation,
    shading normalised to mean 1 -> the material's mean colour/tint is unchanged);
  * colour / brightness mottling at 10-60 cm (paint and floor wear, sun fading);
  * real-scale detail 5-30 cm where real surfaces have it: scuffs and marks on paint, beige cut-pile carpet (fibre scan) + faint marks,
    upholstery seams, brushed metal + scratches, leather creases/stitching, patterned wallpaper (damask), patterned rugs.
Geometry, collision and depth are untouched (albedo only)."""
import math
import numpy as np


def _noise(n, rng, scale_px, amp):
    """Periodic band-limited noise, std = amp, features ~scale_px (FFT -> tiles seamlessly)."""
    f = np.fft.fftfreq(n)[:, None] ** 2 + np.fft.fftfreq(n)[None, :] ** 2
    k = np.exp(-f * (scale_px / 2.0) ** 2 * (2 * math.pi) ** 2 / 2)
    z = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal((n, n))) * k))
    return (z / (z.std() + 1e-9) * amp).astype(np.float32)


def _blur_wrap(m, sx, sy=None):
    """Gaussian blur with periodic boundary (pad wrap -> blur -> crop)."""
    import cv2
    sy = sx if sy is None else sy; p = int(3 * max(sx, sy)) + 1; n = m.shape[0]; p = min(p, n - 1)
    b = cv2.GaussianBlur(np.pad(m, p, mode="wrap"), (0, 0), sigmaX=sx, sigmaY=sy); return b[p:p + n, p:p + n]


def _wrap_draw(n, draw):
    """Call draw(dx, dy) for the 9 periodic copies so marks crossing the tile edge wrap around."""
    for dx in (-n, 0, n):
        for dy in (-n, 0, n): draw(dx, dy)


def _marks(n, ppm, rng, per_m2, size_m, contrast, elong=(1.0, 3.0), soft=True):
    """Multiplicative map of sparse dark marks (scuffs, smudges, dots): density per m^2, size range m, contrast range."""
    import cv2
    m = np.zeros((n, n), np.float32); area = (n / ppm) ** 2
    for _ in range(rng.poisson(per_m2 * area)):
        cx, cy = rng.uniform(0, n, 2); s = rng.uniform(*size_m) * ppm / 2; e = rng.uniform(*elong); a = rng.uniform(0, 180); c = rng.uniform(*contrast)
        ax = (max(1, int(s)), max(1, int(s / e)))
        _wrap_draw(n, lambda dx, dy: cv2.ellipse(m, (int(cx + dx), int(cy + dy)), ax, a, 0, 360, float(c), -1, cv2.LINE_AA))
    if soft: m = _blur_wrap(m, max(0.8, 0.004 * ppm))
    return 1.0 - np.clip(m, 0, 0.9)


def knockdown(n, ppm, rng, amp, per_m2=120, size_m=(0.02, 0.08)):
    """Relief shading of flattened splatter blobs: height map of soft blobs -> directional derivative (light from
    +x+y) -> brightness variation with std ~amp."""
    import cv2
    h = np.zeros((n, n), np.float32); area = (n / ppm) ** 2
    for _ in range(rng.poisson(per_m2 * area)):
        cx, cy = rng.uniform(0, n, 2); r = rng.uniform(*size_m) * ppm / 2; e = rng.uniform(1.0, 1.4); a = rng.uniform(0, 180)
        _wrap_draw(n, lambda dx, dy: cv2.ellipse(h, (int(cx + dx), int(cy + dy)), (max(1, int(r)), max(1, int(r / e))), a, 0, 360, 1.0, -1, cv2.LINE_AA))
    h = _blur_wrap(np.minimum(h, 1.0), max(0.7, 0.003 * ppm))
    g = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) + (np.roll(h, -1, 0) - np.roll(h, 1, 0))
    return (g / (g.std() + 1e-9) * amp).astype(np.float32)


def bake(im, nrm, strength=1.0):
    """Lambert shading of the normal map (key light az 30, el 45) blended with ambient; mean 1."""
    if nrm is None: return im
    l = np.array([math.cos(math.radians(45)) * math.cos(math.radians(30)), math.cos(math.radians(45)) * math.sin(math.radians(30)), math.sin(math.radians(45))], np.float32)
    n = nrm / (np.linalg.norm(nrm, axis=-1, keepdims=True) + 1e-6); sh = 0.35 + 0.65 * np.clip(n @ l, 0, 1); sh = sh / sh.mean()
    return im * (1 + strength * (sh - 1))[..., None]


def _lines(n, ppm, rng, count_per_m, length_m, width_px, contrast, angle=None):
    import cv2
    m = np.zeros((n, n), np.float32); L = n / ppm
    for _ in range(rng.poisson(count_per_m * L * L)):
        x, y = rng.uniform(0, n, 2); ln = rng.uniform(*length_m) * ppm; a = rng.uniform(0, math.pi) if angle is None else angle + rng.normal(0, 0.05)
        x2, y2 = x + ln * math.cos(a), y + ln * math.sin(a); c = rng.uniform(*contrast)
        _wrap_draw(n, lambda dx, dy: cv2.line(m, (int(x + dx), int(y + dy)), (int(x2 + dx), int(y2 + dy)), float(c), width_px, cv2.LINE_AA))
    return 1.0 - np.clip(m, 0, 0.9)


def damask(n, ppm, base, ink, rng):
    """Patterned wallpaper: offset lattice of ornaments (ellipse pairs + stem + dots), period 1/6 x 1/5 m."""
    import cv2
    im = np.ones((n, n, 3), np.float32) * np.array(base, np.float32)
    px, py = n // 6, n // 5; col = tuple(float(v) for v in ink)
    for j in range(5):
        for i in range(6):
            cx = int((i + 0.5 * (j % 2)) * px) % n; cy = int(j * py + py / 2)
            def d(dx, dy, cx=cx, cy=cy):
                c = (cx + dx, cy + dy)
                cv2.ellipse(im, (c[0] - px // 7, c[1]), (px // 6, py // 4), 25, 0, 360, col, 2, cv2.LINE_AA)
                cv2.ellipse(im, (c[0] + px // 7, c[1]), (px // 6, py // 4), -25, 0, 360, col, 2, cv2.LINE_AA)
                cv2.line(im, (c[0], c[1] - py // 3), (c[0], c[1] + py // 3), col, 2, cv2.LINE_AA)
                cv2.circle(im, (c[0], c[1] - py // 3), max(2, px // 22), col, -1, cv2.LINE_AA)
                cv2.circle(im, (c[0] + px // 2, c[1] + py // 2 - py), max(2, px // 30), col, -1, cv2.LINE_AA)
            _wrap_draw(n, d)
    return im


def rug(n, rng, palette):
    """Patterned rug (stretched over the rug top): border bands, central medallion, field of small motifs."""
    import cv2
    p = [np.array(c, np.float32) for c in palette]; im = np.ones((n, n, 3), np.float32) * p[0]
    b = n // 12
    for k, w in enumerate((b, b // 2, b // 3)):
        o = sum((b, b // 2, b // 3)[:k]); cv2.rectangle(im, (o, o), (n - 1 - o, n - 1 - o), tuple(float(v) for v in p[1 + k % 2]), w)
    step = n // 10
    for y in range(2 * b, n - 2 * b, step):
        for x in range(2 * b, n - 2 * b, step):
            pts = np.array([[x, y - step // 3], [x + step // 3, y], [x, y + step // 3], [x - step // 3, y]], np.int32)
            cv2.fillPoly(im, [pts], tuple(float(v) for v in p[2]), cv2.LINE_AA); cv2.circle(im, (x, y), step // 10, tuple(float(v) for v in p[3]), -1, cv2.LINE_AA)
    c = n // 2; cv2.ellipse(im, (c, c), (n // 4, n // 6), 0, 0, 360, tuple(float(v) for v in p[1]), -1, cv2.LINE_AA)
    cv2.ellipse(im, (c, c), (n // 6, n // 10), 0, 0, 360, tuple(float(v) for v in p[3]), -1, cv2.LINE_AA)
    im *= (1 + _noise(n, rng, 3, 0.06))[..., None]           # pile
    return im


RUG_PALETTES = [((0.45, 0.12, 0.10), (0.85, 0.75, 0.55), (0.12, 0.15, 0.30), (0.80, 0.55, 0.20)),   # red / cream / navy / gold
                ((0.18, 0.25, 0.35), (0.80, 0.80, 0.75), (0.55, 0.30, 0.20), (0.90, 0.85, 0.70)),   # navy / ivory / rust
                ((0.35, 0.40, 0.25), (0.85, 0.80, 0.65), (0.30, 0.15, 0.10), (0.70, 0.60, 0.40))]   # olive / sand / brown


def make(name, im, nrm, rep, seed):
    """im: albedo (n,n,3) BGR float 0-1 (already tinted); nrm: tangent normal map (n,n,3) or None; rep: texrepeat per
    metre (tile = 1/rep m).  -> v2 albedo."""
    n = im.shape[0]; ppm = n * rep; rng = np.random.default_rng(seed); one = np.ones((n, n), np.float32)
    paint = name.startswith("plaster") or name in ("white", "wallpaper")
    if name == "wallpaper":     # patterned (damask) instead of the near-flat scan (std 1.8): cream + muted green ink
        grain = im.mean(-1, keepdims=True) / (im.mean() + 1e-6)
        im = damask(n, ppm, (0.80, 0.86, 0.88), (0.52, 0.62, 0.55), rng) * grain
    if name == "ceiling":       # knockdown texture (common sprayed US ceiling): flattened 2-8 cm splatters, relief lit from the side
        im = bake(im, nrm, 0.5) * (1 + _noise(n, rng, 0.25 * ppm, 0.02) + knockdown(n, ppm, rng, 0.08, per_m2=260, size_m=(0.02, 0.06)))[..., None]
        im *= _marks(n, ppm, rng, 0.6, (0.01, 0.04), (0.05, 0.15))[..., None]
    elif paint:
        mot = _noise(n, rng, 0.3 * ppm, 0.025) + _noise(n, rng, 0.08 * ppm, 0.012)
        im = bake(im, nrm, 0.3) * (1 + mot)[..., None]
        im *= (_marks(n, ppm, rng, 3.0, (0.01, 0.05), (0.10, 0.30), (1.5, 5.0))                 # scuffs, smudges
               * _marks(n, ppm, rng, 2.0, (0.004, 0.008), (0.35, 0.6), (1.0, 1.3), soft=False)   # nail / screw holes
               * _lines(n, ppm, rng, 0.8, (0.05, 0.25), 2, (0.10, 0.25)))[..., None]              # scrapes
    elif name == "carpet":      # beige cut pile (James t630): the scan's fibre structure (luminance + baked normals) on a
        # beige base, scan chroma kept at 25 % (fibre heather), low-frequency shading / traffic mottle (tileable, subtle)
        lum = im.mean(-1); L = (lum - lum.mean()) / (lum.std() + 1e-6)
        sh = bake(np.ones_like(im), nrm, 1.0)[..., 0] if nrm is not None else one; sh = (sh - sh.mean()) / (sh.std() + 1e-6)
        chroma = im - lum[..., None]; chroma = (chroma - chroma.reshape(-1, 3).mean(0)) * 0.25
        beige = np.array((0.588, 0.698, 0.769), np.float32)       # BGR of sRGB (196, 178, 150)
        im = beige * (1 + 0.10 * L + 0.06 * sh + _noise(n, rng, 0.12 * ppm, 0.008) + _noise(n, rng, 0.03 * ppm, 0.01))[..., None] + chroma
        im *= _marks(n, ppm, rng, 0.5, (0.08, 0.3), (0.04, 0.10), (1.0, 2.0))[..., None]     # faint traffic / spill marks
    elif name in ("fabric", "fabric2"):     # upholstery: weave check 2.5 cm + piping seams every tile + wear
        y, x = np.mgrid[0:n, 0:n] / ppm
        chk = ((np.floor(x / 0.025) + np.floor(y / 0.025)) % 2) * 0.07
        seam = (np.minimum(x % (1 / rep), (1 / rep) - x % (1 / rep)) * ppm < 3) | (np.minimum(y % (1 / rep), (1 / rep) - y % (1 / rep)) * ppm < 3)
        im = bake(im, nrm, 1.0) * (1 - chk - 0.35 * seam + _noise(n, rng, 0.15 * ppm, 0.05))[..., None]
    elif name == "metal":       # brushed (anisotropic) + scratches
        import cv2
        br = _blur_wrap(np.random.default_rng(seed).standard_normal((n, n)).astype(np.float32), 40, 0.8)
        im = im * (1 + br / (br.std() + 1e-6) * 0.07)[..., None] * _lines(n, ppm, rng, 40, (0.01, 0.06), 1, (0.15, 0.35))[..., None]
    elif name == "dark":        # leather: creases + stitched seam per tile
        im = bake(im, nrm, 1.0) * (1 + _noise(n, rng, 0.05 * ppm, 0.12))[..., None] * _lines(n, ppm, rng, 25, (0.02, 0.08), 2, (0.3, 0.5))[..., None]
        im[n // 16:n // 16 + 3, ::12] = im[n // 16:n // 16 + 3, ::12] * 2.5 + 0.1        # stitch dashes
    elif name in ("wood", "woodfloor", "tile", "marble", "concrete"):
        im = bake(im, nrm, 1.0) * (1 + _noise(n, rng, 0.3 * ppm, 0.04))[..., None]
        if name in ("woodfloor", "tile"): im *= _marks(n, ppm, rng, 1.0, (0.01, 0.06), (0.08, 0.25), (1.0, 4.0))[..., None]
    return np.clip(im, 0, 1)


# ------------------------------------------------------------------ small household items (decor v2 textures)
def item_textures(n=256, seed=7):
    """{name: BGR float image} for wall/ceiling fixtures: outlet, switch, poster0-2, grille, clock, smoke."""
    import cv2
    rng = np.random.default_rng(seed); T = {}
    def plate():
        im = np.ones((n, n, 3), np.float32) * 0.93; cv2.rectangle(im, (2, 2), (n - 3, n - 3), (0.75, 0.75, 0.75), 3); return im
    o = plate()
    for cy in (n // 3, 2 * n // 3):
        for dx in (-n // 10, n // 10): cv2.rectangle(o, (n // 2 + dx - 4, cy - n // 12), (n // 2 + dx + 4, cy + n // 12), (0.12, 0.12, 0.12), -1)
        cv2.circle(o, (n // 2, cy + n // 8), 6, (0.15, 0.15, 0.15), -1)
    T["outlet"] = o
    s = plate(); cv2.rectangle(s, (n // 3, n // 4), (2 * n // 3, 3 * n // 4), (0.82, 0.82, 0.82), -1); cv2.rectangle(s, (n // 3, n // 4), (2 * n // 3, 3 * n // 4), (0.45, 0.45, 0.45), 2)
    T["switch"] = s
    for k in range(3):        # posters / calendars: title band, photo block, text lines
        im = np.ones((n * 2, n * 2, 3), np.float32) * (0.92 - 0.1 * k); N = 2 * n
        cv2.rectangle(im, (0, 0), (N - 1, N // 7), tuple(float(v) for v in rng.uniform(0.1, 0.6, 3)), -1)
        ph = cv2.resize(rng.random((5, 5, 3)).astype(np.float32), (N - N // 5, N // 2), interpolation=cv2.INTER_CUBIC)
        for _ in range(6): cv2.circle(ph, tuple(int(v) for v in rng.integers(0, N // 2, 2)), int(rng.integers(8, 40)), tuple(float(v) for v in rng.random(3)), -1)
        im[N // 5:N // 5 + N // 2, N // 10:N // 10 + ph.shape[1]] = ph
        y = N // 5 + N // 2 + 12
        while y < N - 10:
            x = N // 10
            while x < N - N // 10:
                w = int(rng.integers(10, 40)); cv2.rectangle(im, (x, y), (min(x + w, N - N // 10), y + 5), (0.15, 0.15, 0.15), -1); x += w + 8
            y += 16
        cv2.rectangle(im, (0, 0), (N - 1, N - 1), (0.1, 0.1, 0.1), 6); T["poster%d" % k] = im
    g = np.ones((n, n, 3), np.float32) * 0.9
    for y in range(12, n - 12, 14): cv2.rectangle(g, (12, y), (n - 12, y + 6), (0.35, 0.35, 0.35), -1)
    cv2.rectangle(g, (4, 4), (n - 5, n - 5), (0.6, 0.6, 0.6), 4); T["grille"] = g
    c = np.ones((n, n, 3), np.float32) * 0.15; cv2.circle(c, (n // 2, n // 2), n // 2 - 8, (0.95, 0.95, 0.93), -1)
    for h in range(12):
        a = h / 12 * 2 * math.pi; r0, r1 = n // 2 - 30, n // 2 - 14
        cv2.line(c, (int(n / 2 + r0 * math.cos(a)), int(n / 2 + r0 * math.sin(a))), (int(n / 2 + r1 * math.cos(a)), int(n / 2 + r1 * math.sin(a))), (0.1, 0.1, 0.1), 5)
    cv2.line(c, (n // 2, n // 2), (n // 2 + 50, n // 2 - 60), (0.1, 0.1, 0.1), 6); cv2.line(c, (n // 2, n // 2), (n // 2 - 80, n // 2 + 10), (0.1, 0.1, 0.1), 4)
    T["clock"] = c
    sm = np.ones((n, n, 3), np.float32) * 0.95; cv2.circle(sm, (n // 2, n // 2), n // 5, (0.75, 0.75, 0.75), 3); cv2.circle(sm, (n // 2 + 20, n // 2), 8, (0.2, 0.6, 0.2), -1)
    T["smoke"] = sm
    b = np.ones((n * 2, n, 3), np.float32) * 0.97                       # venetian blind, half open: slats + light gaps
    for y in range(0, 2 * n, 40): b[y:y + 32] = (0.80, 0.78, 0.74); b[y + 29:y + 32] = (0.50, 0.48, 0.45)   # ~5 cm slats
    b[:, n // 4 - 1:n // 4 + 1] = b[:, 3 * n // 4 - 1:3 * n // 4 + 1] = (0.45, 0.43, 0.40); b[-24:] = (0.70, 0.68, 0.64)
    T["blinds"] = b
    cb = np.ones((n, n * 2, 3), np.float32) * 0.92                       # kitchen cabinet fronts: 4 doors, panels, knobs
    W2 = 2 * n
    for k in range(4):
        x0 = k * W2 // 4; cv2.rectangle(cb, (x0 + 4, 4), (x0 + W2 // 4 - 4, n - 5), (0.70, 0.70, 0.68), 3)
        cv2.rectangle(cb, (x0 + 22, 22), (x0 + W2 // 4 - 22, n - 23), (0.80, 0.80, 0.78), 2)
        kx = x0 + (W2 // 4 - 14 if k % 2 == 0 else 14); cv2.circle(cb, (kx, n * 3 // 4), 6, (0.25, 0.25, 0.25), -1)
    T["cabinet"] = cb
    return T


ITEMS = ["outlet", "switch", "poster0", "poster1", "poster2", "grille", "clock", "smoke", "blinds", "cabinet"]


def windows(env):
    """Window openings of the wall mesh: a sill (up-facing horizontal face group, z 0.25-1.5) under a header
    (down-facing group, z 1.55 .. ceiling - 2 cm) with overlapping footprint, wall-thin (<= 15 cm) -> [(lo_xy, hi_xy, z_sill, z_head)]."""
    import trimesh
    m0, sc, off = env._mesh; m = m0.copy(); m.merge_vertices(); V = m.vertices * sc + off; F = m.faces; N = m.face_normals
    zc = V[F].mean(1)[:, 2]; adj = m.face_adjacency
    def groups(sign, zlo, zhi):
        idx = np.nonzero((N[:, 2] * sign > 0.99) & (zc > zlo) & (zc < zhi))[0]
        if len(idx) == 0: return []
        comps = trimesh.graph.connected_components(adj[np.isin(adj, idx).all(1)], nodes=idx, min_len=1)
        return [(V[F[c]].reshape(-1, 3)[:, :2].min(0), V[F[c]].reshape(-1, 3)[:, :2].max(0), float(zc[c].mean())) for c in comps]
    out = []
    for a0, a1, zs in groups(1, 0.25, 1.5):
        for b0, b1, zh in groups(-1, 1.55, getattr(env, "ceiling_z", 2.29) - 0.02):
            lo, hi = np.maximum(a0, b0), np.minimum(a1, b1); d = hi - lo
            if d.min() > 0.02 and d.min() <= 0.15 and d.max() > 0.3 and not any(np.allclose(lo, o[0]) and np.allclose(hi, o[1]) for o in out):
                out.append((lo, hi, zs, zh))
    return out


def casita_items(env, wall_box, D, seed=11):
    """Everyday wall / ceiling fixtures (visual-only, <= 2 cm proud of the wall/ceiling): outlets at 0.30 m every
    few metres, light switches at 1.2 m, posters/calendars, a clock, ceiling light fixtures, smoke detectors, HVAC
    grilles.  Wall positions by ray casting from each place into 8 directions (first wall within 4 m)."""
    import os
    if os.environ.get("KEVIN_EMU_ITEMS") == "0": return
    rng = np.random.default_rng(seed); P = math.pi
    for lbl, (x, y) in sorted(env.places.items()):
        for k in range(8):
            a = k * P / 4 + rng.uniform(-0.2, 0.2)
            wall_box(x, y, a, 0.30, 0.002, 0.07, 0.115, "outlet", maxd=4.0)     # 7 mm proud (real cover plates ~5 mm)
            if k % 2 == 0: wall_box(x, y, a + 0.12, 1.20, 0.002, 0.07, 0.115, "switch", maxd=4.0)
            if k % 3 == 1: wall_box(x, y, a - 0.15, float(rng.uniform(1.35, 1.6)), 0.004, float(rng.uniform(0.3, 0.55)), float(rng.uniform(0.4, 0.7)), "poster%d" % (k % 3), maxd=4.0)
        wall_box(x, y, rng.uniform(0, 2 * P), 1.85, 0.02, 0.3, 0.3, "clock", maxd=4.0)
        a = rng.uniform(0, 2 * P); z = float(rng.uniform(1.55, 1.8))          # wall shelf (above Kevin's height) + books
        wall_box(x, y, a, z, 0.10, 0.9, 0.022, "wood", maxd=4.0); wall_box(x, y, a, z + 0.11, 0.08, 0.8, 0.19, "book", maxd=4.0)
        a = rng.uniform(0, 2 * P)                                                # cluster of small framed photos
        for j, (du, dz, w, h) in enumerate(((-0.25, 1.55, 0.2, 0.25), (0.0, 1.45, 0.25, 0.18), (0.22, 1.6, 0.18, 0.22), (0.05, 1.72, 0.15, 0.12))):
            wall_box(x, y, a + du / 2.5, dz, 0.012, w, h, "art%d" % (j % 4), maxd=4.0)
        cz = getattr(env, "ceiling_z", 2.29)                                   # ceiling underside (world_params)
        D.append(("cylinder", (x + 0.3, y + 0.3, cz - 0.025), (0.17, 0.025), 0.0, "lampshade", "ceiling"))   # flush-mount fixture
        D.append(("cylinder", (x - 0.6, y - 0.4, cz - 0.015), (0.06, 0.015), 0.0, "smoke", "ceiling"))
        D.append(("box", (x - 0.2, y + 0.9, cz - 0.003), (0.15, 0.25, 0.003), 0.0, "grille", "ceiling"))
    for lo, hi, zs, zh in windows(env):     # venetian blinds, lowered over the top ~55 % of every window
        c = (lo + hi) / 2; d = hi - lo; hgt = 0.55 * (zh - zs); along_x = d[0] > d[1]
        size = (d[0] / 2, 0.005, hgt / 2) if along_x else (0.005, d[1] / 2, hgt / 2)
        D.append(("box", (float(c[0]), float(c[1]), zh - hgt / 2), tuple(float(v) for v in size), 0.0, "blinds", False))

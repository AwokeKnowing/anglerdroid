"""Per-frame perception, reusing the robot's own code where it exists.

ego(): RS1 verts -> 320x240 ego labels (1 cm, axle at RCX,RCY, +col = forward, -row = left) via
       perception.ego_rs1.label_rs1_ego (the exact code the robot runs) + distance-to-obstacle field.
rs2_points(): RS2 depth -> base-frame obstacle points and floor points (for the map and for matching)."""
from __future__ import annotations
import os
import numpy as np
import cv2
import nav  # noqa
from robot_config import RCX, RCY, FRAME_W, FRAME_H, EGO_PX_SIZE
from perception.ego_rs1 import label_rs1_ego
from perception.labels import CLEAR, OBSTACLE, UNKNOWN, SELF
from nav.hw import RS2_R, RS2_T, AXLE_H

try:
    from vision import TD_X_OFFSET
except Exception:
    TD_X_OFFSET = -75


class Ego:
    def __init__(self):
        self.lab = np.zeros((FRAME_H, FRAME_W), np.uint8); self.hgt = np.zeros_like(self.lab)
        self.wl = np.zeros_like(self.lab); self.wh = np.zeros_like(self.lab)
        cols = (np.arange(FRAME_W) - RCX) * EGO_PX_SIZE; rows = (RCY - np.arange(FRAME_H)) * EGO_PX_SIZE
        self.X, self.Y = np.meshgrid(cols.astype(np.float32), rows.astype(np.float32))   # robot frame metres per pixel
        # pixels inside the robot's own footprint (+1.5 cm) can never be obstacles: the RS1 sees its own wheels / bumper
        # corners there (SELF mask in perception/ is offset by the TD_X_OFFSET question, see docs/fullstack.md)
        # FOOTPRINT_SAFE (SIL12 + deck corners) + 1 cm: the gate caster / deck corners sit on its rear hull edge and
        # were returned as obstacles at sd~0 (safety then saw permanent 'contact'); 1 cm still keeps door jambs
        from nav.safety import FOOTPRINT_SAFE, _edges
        n, c = _edges(FOOTPRINT_SAFE); P = np.stack([self.X.ravel(), self.Y.ravel()], 1)
        self.body = ((P @ n.T - c).max(1) < 0.010).reshape(self.X.shape)
        # wheels (twin: y=+-0.181, half-width 0.026, r 0.088) and the swivelling gate caster read as OBSTACLE 1-3 cm
        # outside the hull in some frames (below the 50 % self-mask threshold): the recurring "clearance 0.020"
        # phantom that pinned safety at 'limited' in open space (apartment kitchen shuttling)
        ax, ay = np.abs(self.X), np.abs(self.Y)
        self.body |= (ax < 0.095) & (ay > 0.15) & (ay < 0.222)   # was 0.245: hid maze walls at the wheel (14 contacts)
        self.body |= (self.X < -0.185) & (self.X > -0.30) & (ay < 0.175)   # caster + deck rear corners
        import os
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        f = os.path.join(d, "self_mask_sim.npy" if os.environ.get("KEVIN_HAL") == "sim" else "self_mask.npy")
        self.self_mask = np.load(f) if os.path.exists(f) else np.zeros_like(self.body)
        self.body |= self.self_mask
        # near ring (<=5 cm outside the hull): the RS1 depth at the robot's own silhouette throws flickering speckle
        # there; keep a ring pixel only if it is part of a blob (>=3 of 8 neighbours) and was OBSTACLE last frame too
        self.ring = ((P @ n.T - c).max(1) < 0.05).reshape(self.X.shape) & ~self.body
        self._prev = np.zeros_like(self.body)
        # overhangs (desk tops ~0.7 m) above the body: the body mask hid them, so once the deck was under a desk
        # edge the mast (only robot part above ~0.5 m, x=-0.16) was blind to it (office: 367 contacts, mast vs desk).
        # Inside the body keep OBSTACLE pixels >= 0.55 m high, except the mast/e-stop column itself.
        inside = ((P @ n.T - c).max(1) < -0.01).reshape(self.X.shape)       # strictly inside the hull (not the
        # +1 cm band / wheel+caster boxes, where door jambs sit when squeezing through a 0.55-0.65 m door)
        self.over_ok = inside & ~((self.X < -0.11) & (self.X > -0.22) & (np.abs(self.Y) < 0.06))

    # KEVIN_EGO_DEPTHCULL=1 (default off): same labels, computed in depth space.  The floor cutoff is applied to the
    # raw RS1 Z per pixel instead of to a levelled copy of the cloud: with the per-frame floor plane z = c0 x + c1 y
    # + c2 and the per-pixel ray (rx, ry) = (x/z, y/z) (measured rays, i.e. the exact decimated grid / pixel-centre
    # convention of the SDK cloud), levelled Z = z (1 - c0 rx - c1 ry), so a pixel is OBSTACLE iff
    # z < clip / (1 - c0 rx - c1 ry) and CLIFF iff z > (c2 + CLIFF_DZ) / (1 - c0 rx - c1 ry).  1/D is the per-pixel
    # floor-cutoff image; it is regenerated every frame from the tilt fit (one 45k-element pass).  Pixels inside a
    # 0.1 mm margin of either cutoff are re-decided with the original float expressions, so the result is
    # bit-identical to the transform path (tests/test_ego_depthcull.py); only the cell transform is shared by the
    # map and safety label passes and the levelled cloud copy is never built.
    DEPTHCULL = os.environ.get("KEVIN_EGO_DEPTHCULL", "0") == "1"   # TC_Z (0.80) < clip is assumed: synthetic points are always OBSTACLE
    DC_MARGIN = 1e-4

    def update(self, verts):
        if self.DEPTHCULL:
            r = self._update_dc(verts)
            if r is not None: return r
        v2 = self._too_close(verts)
        self.lab_map = None
        if self.too_close_px or self.cliff_px:
            # synthetic (no-depth / drop-off) obstacles are for the SAFETY layer only: the map gets the measured
            # labels, otherwise every 0.8 m 'shadow' is painted into slam2 and comes back as remembered phantoms
            label_rs1_ego(verts, labels_out=self.lab, height_out=self.hgt, work_labels=self.wl, work_height=self.wh, x_offset=int(TD_X_OFFSET))
            self.lab_map = self.lab.copy()
        label_rs1_ego(v2, labels_out=self.lab, height_out=self.hgt, work_labels=self.wl, work_height=self.wh, x_offset=int(TD_X_OFFSET))
        return self.lab

    # RS1 'too close' rule: a real D435 returns 0 inside min-Z (~0.175 m from the lens), so an overhang at mast
    # height (desk edge, shelf) shows up as a hole, not as points.  Per-pixel rays are learned from valid returns;
    # a no-depth blob that is large (>= TC_MIN_PX), outside the stereo-occlusion band (left columns), and present in
    # TC_FRAMES consecutive frames is re-inserted along those rays at TC_Z (its floor shadow) -> OBSTACLE.
    # Floor holes / speckle are small or flicker and are rejected by size + persistence.
    TC_MIN_PX, TC_FRAMES, TC_Z, TC_BAND = 150, 3, 0.80, 0.08   # TC_Z: drop the blob as its floor 'shadow' (rays at 0.80 m -> OBSTACLE h~11 cm): the occluder is somewhere on those rays, so the cells under it are not provably free
    def _too_close(self, verts):
        v = np.asarray(verts, np.float32); self.cliff_px = 0
        if v.ndim != 2 or len(v) == 0: return verts
        z = v[:, 2]; ok = z > 0.01
        self._init_rays(v)
        self._ray[ok] = v[ok, :2] / z[ok, None]                # measured rays override (exact frame convention)
        return self._too_close_rest(v, ok)

    def _init_rays(self, v):
        if not hasattr(self, "_shape"):
            self._shape = next(((h, len(v) // h) for h in (160, 120, 240, 480, 180) if len(v) % h == 0 and 1.2 < len(v) / h / h < 2.0), None)
        if not hasattr(self, "_ray") or len(self._ray) != len(v):
            self._ray = np.full((len(v), 2), np.nan, np.float32); self._inv_n = np.zeros(len(v), np.uint8); self.too_close_px = 0
            I = getattr(self, "intr", None)
            if I and self._shape:          # rays from intrinsics: no prior valid return needed (hand over the lens at boot)
                H, W = self._shape; f = I["width"] / W
                uu, vv = np.meshgrid((np.arange(W) + 0.5) * f, (np.arange(H) + 0.5) * f)
                self._ray[:] = np.c_[((uu - I["ppx"]) / I["fx"]).ravel(), ((vv - I["ppy"]) / I["fy"]).ravel()]
            # legit-invalid pixels: decimation border (4 px), stereo-occlusion band (left 8 %)
            self._legit = np.zeros(len(v), bool)
            if self._shape:
                H, W = self._shape; L = np.zeros((H, W), bool); L[:4] = L[-4:] = True; L[:, :max(4, int(self.TC_BAND * W))] = True; L[:, -4:] = True
                if os.environ.get("KEVIN_HAL") == "sim" and (H, W) == (160, 282):
                    L[68:92, 254:] = True      # sim twin: a robot part inside min-Z at the right image edge (always 0, ~240 px)
                self._legit = L.ravel()

    def _too_close_sel(self, ok):
        """too-close bookkeeping; returns the bool mask of re-inserted pixels (or None)."""
        inv = ~ok & ~self._legit & np.isfinite(self._ray[:, 0])
        self._inv_n = np.where(inv, np.minimum(self._inv_n + 1, 255), 0).astype(np.uint8)
        # whole-view occlusion (hand / cloth over the lens, lens fogged): immobilise, independent of any map
        self.invalid_frac = float(inv.sum()) / max(1, int((~self._legit).sum()))
        self.occluded = self.invalid_frac > 0.5
        cand = self._inv_n >= self.TC_FRAMES
        self.too_close_px = 0
        if cand.sum() >= self.TC_MIN_PX and self._shape is not None:
            H, W = self._shape; C = cand.reshape(H, W).copy()
            n, lab, st, _ = cv2.connectedComponentsWithStats(C.astype(np.uint8), connectivity=8)
            big = np.zeros(n, bool); big[1:] = st[1:, cv2.CC_STAT_AREA] >= self.TC_MIN_PX
            sel = big[lab].ravel()
            if sel.any():
                self.too_close_px = int(sel.sum()); return sel
        return None

    def _too_close_rest(self, v, ok):
        sel = self._too_close_sel(ok)
        if sel is not None:
            v = v.copy(); v[sel, :2] = self._ray[sel] * self.TC_Z; v[sel, 2] = self.TC_Z
        return self._cliff(v, ok)

    # Negative obstacles (curb edge, stair, drop-off): RS1 looks straight down, so ground that is >= CLIFF_DZ farther
    # than the local floor plane is a drop.  The floor plane is fitted per frame (robust, 2 passes) so body
    # pitch/roll on slopes and carpet/threshold bumps are not mistaken for drops; potholes < CLIFF_DZ are drivable.
    # Persistent (CLIFF_FRAMES) blobs >= CLIFF_MIN_PX are re-inserted along their rays at TC_Z -> OBSTACLE.
    # CLIFF_Z0 = expected RS1 depth to the floor (fallback plane + fit window), resolved: robot.json cliff_z0_ref_m 0.946 + the
    # wheels' ride_dz (2026-10-21 soft tyres -> 0.961; SAFETY PARAMETER, James approval a4224de).
    import robot_params as _RP
    CLIFF_Z0, CLIFF_DZ, CLIFF_FRAMES, CLIFF_MIN_PX, LEVEL = float(_RP.V["cliff_z0_m"]), 0.05, 2, 40, True
    def _cliff(self, v, ok):
        self.cliff_px = 0
        if self._shape is None: return v
        z = v[:, 2]; coef = self._fit_floor(v, ok)
        dz = z - (v[:, 0] * coef[0] + v[:, 1] * coef[1] + coef[2])
        self.floor_tilt = (float(coef[0]), float(coef[1]))
        return self._cliff_level(v, ok, z, coef, dz)

    def _fit_floor(self, v, ok, z=None):
        """robust per-frame floor plane z = c0 x + c1 y + c2 (RS1 camera frame) on <= ~3000 floor px."""
        z = v[:, 2] if z is None else z; fl = ok & (np.abs(z - self.CLIFF_Z0) < 0.06)
        coef = np.array([0.0, 0.0, self.CLIFF_Z0], np.float32)
        if fl.sum() > 500:
            fi = np.flatnonzero(fl); fi = fi[::max(1, len(fi) // 3000)]      # plane fit on <= ~3000 floor px (safety core budget)
            A = np.c_[v[fi, 0], v[fi, 1], np.ones(len(fi), np.float32)]; zf = z[fi]
            for _ in range(2):
                coef = np.linalg.lstsq(A, zf, rcond=None)[0]; r = np.abs(A @ coef - zf); k = r < max(0.01, np.percentile(r, 70))
                A, zf = A[k], zf[k]
            if abs(coef[0]) > 0.2 or abs(coef[1]) > 0.2: coef = np.array([0.0, 0.0, self.CLIFF_Z0], np.float32)   # > ~11 deg: implausible fit
        self.floor_coef = coef
        return coef

    def _cliff_level(self, v, ok, z, coef, dz):
        if self.LEVEL and (coef[0] or coef[1]):     # level the cloud: body pitch/roll on slopes/bumps must not tip the
            v = v.copy(); v[ok, 2] -= v[ok, 0] * coef[0] + v[ok, 1] * coef[1]   # floor into OBSTACLE / below the clip
            z = v[:, 2]
        c = ok & (dz > self.CLIFF_DZ)
        if not hasattr(self, "_cl_n") or len(self._cl_n) != len(v): self._cl_n = np.zeros(len(v), np.uint8)
        self._cl_n = np.where(c, np.minimum(self._cl_n + 1, 255), 0).astype(np.uint8)
        cand = self._cl_n >= self.CLIFF_FRAMES
        if cand.sum() >= self.CLIFF_MIN_PX:
            H, W = self._shape
            n, lab, st, _ = cv2.connectedComponentsWithStats(cand.reshape(H, W).astype(np.uint8), connectivity=8)
            big = np.zeros(n, bool); big[1:] = st[1:, cv2.CC_STAT_AREA] >= self.CLIFF_MIN_PX
            sel = big[lab].ravel()
            if sel.any():
                v = v.copy(); r = v[sel, :2] / z[sel, None]
                v[sel, :2] = r * self.TC_Z; v[sel, 2] = self.TC_Z; self.cliff_px = int(sel.sum())
        return v

    def _update_dc(self, verts):
        """Depth-space cull path of update() (KEVIN_EGO_DEPTHCULL=1); None -> caller falls back to the transform path."""
        import perception.ego_rs1 as E
        v = np.asarray(verts, np.float32); self.cliff_px = 0
        if v.ndim != 2 or len(v) == 0: return None
        self._init_rays(v)
        if self._shape is None: return None
        x, y, z = (np.ascontiguousarray(v[:, k]) for k in range(3))          # column copies: contiguous vector ops
        ok = z > 0.01
        R = self._ray
        rx, ry = np.divide(x, z, out=np.zeros_like(x), where=ok), np.divide(y, z, out=np.zeros_like(y), where=ok)   # == R[ok] = v[ok,:2] / z[ok,None]
        R[ok, 0] = rx[ok]; R[ok, 1] = ry[ok]
        rx, ry = np.ascontiguousarray(R[:, 0]), np.ascontiguousarray(R[:, 1])
        tc = self._too_close_sel(ok)
        coef = self._fit_floor(v, ok, z)
        self.floor_tilt = (float(coef[0]), float(coef[1]))
        clip = float(E.TD_FLOOR_CLIP); m = self.DC_MARGIN
        # per-pixel cutoff image for this tilt: 1/D, D = 1 - c0 rx - c1 ry  (rays: exact decimated pixel grid)
        D = 1.0 - (np.float32(coef[0]) * rx + np.float32(coef[1]) * ry)
        with np.errstate(invalid="ignore", divide="ignore"):
            zD = z * D                                                       # ~levelled Z (only used against margins)
        lev = bool(self.LEVEL and (coef[0] or coef[1]))
        def zlev(i):                                                         # exact levelled Z, original float ops
            zz = z[i].copy()
            if lev: zz -= x[i] * coef[0] + y[i] * coef[1]
            return zz
        # cliff: exact dz on the candidates only (dz > CLIFF_DZ  <=>  zD > c2 + CLIFF_DZ)
        ci = np.flatnonzero(ok & (zD > float(coef[2]) + self.CLIFF_DZ - m))
        dzc = z[ci] - (x[ci] * coef[0] + y[ci] * coef[1] + coef[2])
        c = np.zeros(len(v), bool); c[ci[dzc > self.CLIFF_DZ]] = True
        if not hasattr(self, "_cl_n") or len(self._cl_n) != len(v): self._cl_n = np.zeros(len(v), np.uint8)
        self._cl_n = np.where(c, np.minimum(self._cl_n + 1, 255), 0).astype(np.uint8)
        cand = self._cl_n >= self.CLIFF_FRAMES; csel = None
        if cand.sum() >= self.CLIFF_MIN_PX:
            H, W = self._shape
            n, lab, st, _ = cv2.connectedComponentsWithStats(cand.reshape(H, W).astype(np.uint8), connectivity=8)
            big = np.zeros(n, bool); big[1:] = st[1:, cv2.CC_STAT_AREA] >= self.CLIFF_MIN_PX
            sel = big[lab].ravel()
            if sel.any(): csel = sel; self.cliff_px = int(sel.sum())
        # one cell transform for both label passes (xy are not changed by levelling)
        flat, inb = E.cell_index_rs1(x, y)
        def hgt(zz): return np.clip(((clip - zz) * 100.0).astype(np.int32), 1, 100).astype(np.uint8)
        kw = dict(labels_out=self.lab, height_out=self.hgt, work_labels=self.wl, work_height=self.wh, x_offset=int(TD_X_OFFSET))
        self.lab_map = None
        if self.too_close_px or self.cliff_px:          # map pass: measured, unlevelled verts (as the transform path)
            vm = ok & inb; ob = vm & ~(z >= clip)
            E.label_rs1_ego_cells(flat[vm & ~ob], flat[ob], hgt(z[ob]), **kw)
            self.lab_map = self.lab.copy()
        # safety pass: levelled verts + synthetic too-close / cliff points
        meas = ok & inb if csel is None else ok & inb & ~csel
        oi = np.flatnonzero(meas & (zD < clip + m))                          # obstacle candidates
        zo = zlev(oi); vo = zo > 0.01; fo = zo >= clip
        clear_mask = meas.copy(); clear_mask[oi[~fo | ~vo]] = False          # non-candidates are floor by construction
        isob = vo & ~fo
        fl_obs = [flat[oi[isob]]]; h_obs = [hgt(zo[isob])]
        for sel, cliff in ((tc, False), (csel, True)):                     # synthetic points: z = TC_Z < clip -> OBSTACLE
            if sel is None: continue
            si = np.flatnonzero(sel)
            xy = (v[si, :2] / zlev(si)[:, None] * self.TC_Z) if cliff else (R[si] * self.TC_Z)
            xy = xy.astype(np.float32); zs = np.full(len(si), self.TC_Z, np.float32)
            fs, ib = E.cell_index_rs1(xy[:, 0], xy[:, 1])
            fl_obs.append(fs[ib]); h_obs.append(hgt(zs[ib]))
        E.label_rs1_ego_cells(flat[clear_mask], np.concatenate(fl_obs), np.concatenate(h_obs), **kw)
        self.dc_cand = len(oi) + len(ci)
        return self.lab

    def valid_fraction(self):
        if getattr(self, "occluded", False): return 0.0             # safety: no_depth -> immobilised
        known = (self.lab == CLEAR) | (self.lab == OBSTACLE)
        return float(known.mean())

    def obstacle_xy(self, max_n=None):
        m = (self.lab == OBSTACLE) & ~self.body
        m = self._grass(m)
        if m.any():
            nb = cv2.filter2D(m.astype(np.uint8), -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT) - m
            keep = (nb >= 3) & cv2.dilate(self._prev.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            self._prev = m.copy(); m = m & (~self.ring | keep | (self.hgt >= 55))   # overhang edges are thin lines: keep
        else: self._prev = m
        # overhang above the hull: label_rs1_ego marks the robot's own boxes SELF ('SELF wins'), which also hid a desk
        # top sitting over the rear deck (office: mast vs desk edge, 204 contacts). Nothing of the robot except the
        # mast column reaches 55 cm there (open floor: 0 such px), so SELF/OBSTACLE >= 55 cm inside the hull = overhang.
        m = m | (self.over_ok & ((self.lab == OBSTACLE) | (self.lab == SELF)) & (self.hgt >= 55))
        self.obs_mask = m                                              # safety-grade obstacles this frame (planner obstacle layer)
        mm = m if getattr(self, "lab_map", None) is None else (m & (self.lab_map == OBSTACLE))
        self.map_obs = np.stack([self.X[mm], self.Y[mm]], 1)
        self.map_obs_h = self.hgt[mm].astype(np.float32) * 0.01                # cm -> m above floor (carving height band)
        P = np.stack([self.X[m], self.Y[m]], 1)
        if max_n and len(P) > max_n:
            P = P[np.random.default_rng(0).choice(len(P), max_n, replace=False)]
        return P

    # Grass / vegetation traversability: blades give sparse, speckled low returns (2-9 cm) mixed with ground, a
    # real low obstacle (curb edge, edging, toy, shoe) is a solid blob.  Low (< GRASS_H cm) obstacle pixels whose
    # 7x7 neighbourhood is < GRASS_FILL obstacle are dropped -- only when the view is grassy (>= GRASS_FRAC of the
    # known ground is such speckle), so indoor scenes are untouched.  Tall or solid things always stay.
    GRASS_H, GRASS_FILL, GRASS_FRAC = 12, 0.55, 0.015
    def _grass(self, m):
        self.grass_px = 0
        low = m & (self.hgt < self.GRASS_H)
        if not low.any(): return m
        fill = cv2.boxFilter(m.astype(np.float32), -1, (7, 7), normalize=True, borderType=cv2.BORDER_CONSTANT)
        sp = low & (fill < self.GRASS_FILL)
        known = int(((self.lab == CLEAR) | (self.lab == OBSTACLE)).sum())
        if sp.sum() < self.GRASS_FRAC * max(known, 1): return m
        self.grass_px = int(sp.sum())
        return m & ~sp

    def clear_xy(self, stride=4):
        m = self.lab[::stride, ::stride] == CLEAR
        return np.stack([self.X[::stride, ::stride][m], self.Y[::stride, ::stride][m]], 1)

    def rays(self, origin, stride=4):
        """RS1 returns for free-space carving: CLEAR floor pixels (h = 0) and measured OBSTACLE pixels at their height,
        as rays from the RS1 lens (origin, z = height above floor).  UNKNOWN pixels carve nothing."""
        lab = self.lab[::stride, ::stride]; cl = lab == CLEAR; ob = (lab == OBSTACLE) & ~self.body[::stride, ::stride]
        m = cl | ob
        E = np.stack([self.X[::stride, ::stride][m], self.Y[::stride, ::stride][m],
                      np.where(cl, 0.0, self.hgt[::stride, ::stride] * 0.01)[m].astype(np.float32)], 1)
        return (np.asarray(origin, np.float32), E, cl[m])


class Rs2Cloud:
    """RS2 depth -> base frame points. Obstacles: 3..95 cm above floor; floor: |h| < 2 cm."""
    def __init__(self, intr, stride=4, max_range=4.0):
        w, h = intr.get("width", 848), intr.get("height", 480)
        u, v = np.meshgrid(np.arange(0, w, stride, dtype=np.float32), np.arange(0, h, stride, dtype=np.float32))
        self.rx = ((u - intr.get("ppx", w / 2)) / intr.get("fx", 430.0)).ravel()
        self.ry = ((v - intr.get("ppy", h / 2)) / intr.get("fy", 430.0)).ravel()
        self.s = stride; self.max_range = max_range
        self.R, self.T = RS2_R.astype(np.float32), RS2_T.astype(np.float32)     # replaced live by nav.autocal

    def set_pose(self, T, R):
        self.T, self.R = np.asarray(T, np.float32), np.asarray(R, np.float32)

    def points(self, depth_u16, h_max=0.95, return_h=False):
        z = depth_u16[::self.s, ::self.s].ravel().astype(np.float32) * 1e-3
        ok = (z > 0.2) & (z < self.max_range)
        z = z[ok]
        P = np.stack([self.rx[ok] * z, self.ry[ok] * z, z], 1) @ self.R.T + self.T
        hgt = P[:, 2] + AXLE_H                       # height above floor
        hmin = 0.03 + 0.012 * z * z                  # D435 depth error grows ~z^2: keep noisy far floor out of 'obstacle'
        mo = (hgt > hmin) & (hgt < h_max)
        obs = P[mo, :2]          # safety core: h_max 1.0 (all below Kevin's top -> 2-D)
        flo = P[np.abs(hgt) < 0.02, :2]
        if return_h: return obs, hgt[mo], flo
        return obs, flo

    def rays(self, depth_u16, sub=2):
        """RS2 returns for free-space carving (every sub-th stride pixel): rays from the lens, z = height above floor.
        No-return pixels are unknown and carve nothing; returns beyond max_range still prove the ray free up to there
        (carve_rays caps the carved length and scales the noise margin with the true range)."""
        hh, ww = depth_u16[::self.s, ::self.s].shape
        z = depth_u16[::self.s, ::self.s][::sub, ::sub].ravel().astype(np.float32) * 1e-3
        rx = self.rx.reshape(hh, ww)[::sub, ::sub].ravel(); ry = self.ry.reshape(hh, ww)[::sub, ::sub].ravel()
        ok = (z > 0.2) & (z < 12.0); z = z[ok]
        P = np.stack([rx[ok] * z, ry[ok] * z, z], 1) @ self.R.T + self.T
        P[:, 2] += AXLE_H; o = self.T.copy(); o[2] += AXLE_H
        return (o, P, np.abs(P[:, 2]) < 0.02)

    def wall_z4(self, depth_u16):
        """Wall directions for the Manhattan heading (bench/slam/kslam): surface normals from neighbouring depth
        pixels (stride grid), rotated to the base frame; vertical surfaces 12 cm..2.2 m above the floor.  Returns
        (z4, n): z4 = mean exp(4i * normal angle) in the base frame, n = wall pixels used."""
        z = depth_u16[::self.s, ::self.s].astype(np.float32) * 1e-3; hh, ww = z.shape
        Pc = np.stack([self.rx.reshape(hh, ww) * z, self.ry.reshape(hh, ww) * z, z], -1)
        valid = (z > 0.3) & (z < 4.0)
        dx = Pc[1:-1, 2:] - Pc[1:-1, :-2]; dy = Pc[2:, 1:-1] - Pc[:-2, 1:-1]; n = np.cross(dx, dy); nn = np.linalg.norm(n, axis=-1)
        ok = valid[1:-1, 1:-1] & valid[1:-1, 2:] & valid[1:-1, :-2] & valid[2:, 1:-1] & valid[:-2, 1:-1] & (nn > 1e-9)
        ok &= (np.linalg.norm(dx, axis=-1) < 0.15) & (np.linalg.norm(dy, axis=-1) < 0.15)
        nb = (n[ok] / nn[ok][:, None]) @ self.R.T; hgt = (Pc[1:-1, 1:-1][ok] @ self.R.T + self.T)[:, 2] + AXLE_H
        sel = (np.abs(nb[:, 2]) < 0.15) & (hgt > 0.12) & (hgt < 2.2)
        k = int(sel.sum())
        if k == 0: return 0j, 0
        return complex(np.exp(4j * np.arctan2(nb[sel, 1], nb[sel, 0])).mean()), k

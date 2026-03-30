"""slam.py – Pluggable SLAM backend with pose-graph optimisation.

Provides:
  SlamBackend   – abstract interface (swap in network SLAM, ORB-SLAM, etc.)
  PoseGraphSLAM – concrete implementation using:
      • Posed keyframes (20 cm / 10°)
      • Rotation-invariant radial descriptors for place recognition
      • Scan-matching verification for loop closures
      • Gauss-Newton 2D pose-graph optimisation (scipy sparse)
      • In-memory map with 200–500 MB budget (no database)

The SLAM backend *owns* a GlobalMap and exposes the same update/render/project
API so it can be dropped in wherever GlobalMap was used.
"""

import math
import time

import cv2
import numpy as np

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

from globalmap import (
    GlobalMap, MAP_W, MAP_H, PX_SIZE, ORIGIN_X, ORIGIN_Y, UNKNOWN_VAL,
)

# ── Tunables ────────────────────────────────────────────────────────
KF_DIST = 0.20          # metres between keyframes
KF_ANGLE = math.radians(10)  # radians between keyframes

LOOP_MIN_FRAMES = 30    # skip the N most recent keyframes when searching
LOOP_MIN_DIST = 1.5     # metres – candidates must be at least this far (graph-wise)
LOOP_SCORE_THRESH = 0.40  # radial-descriptor cosine similarity
LOOP_MATCH_THRESH = 0.25  # scan-match overlap ratio

DESC_N_RINGS = 20       # rings in the rotation-invariant descriptor
DESC_MAX_R = 120        # pixel radius for descriptor
THUMB_SZ = 48           # scan-match thumbnail size

ODOM_INFO = np.diag([100.0, 100.0, 200.0]).astype(np.float64)
LOOP_INFO = np.diag([40.0, 40.0, 80.0]).astype(np.float64)

MEM_HIGH = 500 * 1024 * 1024   # 500 MB
MEM_LOW  = 200 * 1024 * 1024   # 200 MB
MEM_CHECK_EVERY = 50            # keyframes between memory checks

OPTIM_ITERS = 15


# ── Helpers ─────────────────────────────────────────────────────────

def _norm_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def _radial_descriptor(obs, cx, cy, n_rings=DESC_N_RINGS, max_r=DESC_MAX_R):
    """Rotation-invariant descriptor: mean normalised obstacle height per ring.

    obs values: 0=free/unobserved, 1-100=obstacle height in cm.
    Each ring's descriptor value = mean(obs) / 100 within the annulus.
    """
    h, w = obs.shape
    yy, xx = np.ogrid[0:h, 0:w]
    dist_sq = (xx - cx).astype(np.float32) ** 2 + (yy - cy).astype(np.float32) ** 2
    ring_width = float(max_r) / n_rings
    desc = np.zeros(n_rings, dtype=np.float32)
    for i in range(n_rings):
        r0_sq = (i * ring_width) ** 2
        r1_sq = ((i + 1) * ring_width) ** 2
        mask = (dist_sq >= r0_sq) & (dist_sq < r1_sq)
        cnt = mask.sum()
        if cnt > 0:
            desc[i] = obs[mask].sum() / (cnt * 100.0)
    return desc


def _scan_match(obs_a, obs_b, n_angles=36, thumb_sz=THUMB_SZ):
    """Brute-force scan match: returns (dx_px, dy_px, dtheta, score).

    Tries n_angles rotations of obs_b, uses phase correlation for
    translation, returns the best (highest overlap) match.
    """
    sz = thumb_sz
    a = cv2.resize(obs_a, (sz, sz), interpolation=cv2.INTER_AREA).astype(np.float32) / 100.0
    b_full = cv2.resize(obs_b, (sz, sz), interpolation=cv2.INTER_AREA).astype(np.float32) / 100.0

    best_score = -1.0
    best = (0.0, 0.0, 0.0, 0.0)
    center = (sz / 2.0, sz / 2.0)

    for ai in range(n_angles):
        angle = ai * (360.0 / n_angles)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        b_rot = cv2.warpAffine(b_full, M, (sz, sz), borderValue=0.0)

        shift, resp = cv2.phaseCorrelate(a, b_rot)
        dx, dy = shift

        M2 = np.float32([[1, 0, dx], [0, 1, dy]])
        b_shifted = cv2.warpAffine(b_rot, M2, (sz, sz), borderValue=0.0)

        overlap = (a * b_shifted).sum()
        union = np.maximum(a, b_shifted).sum()
        score = overlap / max(union, 1e-6)

        if score > best_score:
            best_score = score
            best = (dx, dy, math.radians(-angle), score)

    return best


# ── Pose-graph solver (pure scipy sparse) ───────────────────────────

def _optimize_pose_graph(nodes, edges, n_iter=OPTIM_ITERS):
    """Gauss-Newton on 2D pose graph.  Returns dict {id: (x, y, theta)}.

    nodes : dict  {id: (x, y, theta)}
    edges : list  [(id_i, id_j, dx, dy, dtheta, info_3x3), ...]

    First node is anchored (fixed).
    """
    if len(nodes) < 2 or len(edges) == 0:
        return dict(nodes)

    ids = sorted(nodes.keys())
    n = len(ids)
    idx = {nid: k for k, nid in enumerate(ids)}

    x = np.zeros(3 * n, dtype=np.float64)
    for nid in ids:
        k = idx[nid]
        x[3 * k], x[3 * k + 1], x[3 * k + 2] = nodes[nid]

    for _ in range(n_iter):
        rows, cols, vals = [], [], []
        b = np.zeros(3 * n, dtype=np.float64)

        for (ni, nj, dxm, dym, dtm, omega) in edges:
            ii, jj = idx[ni], idx[nj]
            xi, yi, ti = x[3 * ii: 3 * ii + 3]
            xj, yj, tj = x[3 * jj: 3 * jj + 3]

            ct, st = math.cos(ti), math.sin(ti)
            dpx, dpy = xj - xi, yj - yi

            # error in local frame of node i
            e = np.array([
                ct * dpx + st * dpy - dxm,
                -st * dpx + ct * dpy - dym,
                _norm_angle(tj - ti - dtm),
            ], dtype=np.float64)

            # Jacobian A (de/dxi) and B (de/dxj)
            dR_dp = np.array([-st * dpx + ct * dpy, -ct * dpx - st * dpy])
            A = np.array([
                [-ct, -st, dR_dp[0]],
                [st, -ct, dR_dp[1]],
                [0.0, 0.0, -1.0],
            ], dtype=np.float64)
            B = np.array([
                [ct, st, 0.0],
                [-st, ct, 0.0],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)

            Hii = A.T @ omega @ A
            Hij = A.T @ omega @ B
            Hjj = B.T @ omega @ B
            bi = -A.T @ omega @ e
            bj = -B.T @ omega @ e

            for r in range(3):
                for c in range(3):
                    ri0, ci0 = 3 * ii + r, 3 * ii + c
                    rows.append(ri0); cols.append(ci0); vals.append(Hii[r, c])

                    ri0, cj0 = 3 * ii + r, 3 * jj + c
                    rows.append(ri0); cols.append(cj0); vals.append(Hij[r, c])
                    rows.append(cj0); cols.append(ri0); vals.append(Hij[r, c])

                    rj0, cj0 = 3 * jj + r, 3 * jj + c
                    rows.append(rj0); cols.append(cj0); vals.append(Hjj[r, c])

            b[3 * ii: 3 * ii + 3] += bi
            b[3 * jj: 3 * jj + 3] += bj

        # Anchor first node
        for k in range(3):
            rows.append(k); cols.append(k); vals.append(1e6)

        if _HAS_SCIPY:
            H = sp.coo_matrix((vals, (rows, cols)), shape=(3 * n, 3 * n)).tocsr()
            dx_sol = spla.spsolve(H, b)
        else:
            H = np.zeros((3 * n, 3 * n), dtype=np.float64)
            for r, c, v in zip(rows, cols, vals):
                H[r, c] += v
            dx_sol = np.linalg.solve(H, b)

        x += dx_sol
        for k in range(n):
            x[3 * k + 2] = _norm_angle(x[3 * k + 2])

        if np.max(np.abs(dx_sol)) < 1e-6:
            break

    return {nid: (x[3 * idx[nid]], x[3 * idx[nid] + 1], x[3 * idx[nid] + 2])
            for nid in ids}


# ── Keyframe ────────────────────────────────────────────────────────

class Keyframe:
    __slots__ = (
        'id', 'x', 'y', 'theta',
        'obs_roi', 'known_roi', 'roi_bounds',
        'descriptor', 'thumb',
        'cx', 'cy', 'px_size',
        'timestamp',
    )

    def approx_bytes(self):
        s = 128  # fixed overhead
        if self.obs_roi is not None:
            s += self.obs_roi.nbytes + self.known_roi.nbytes
        if self.thumb is not None:
            s += self.thumb.nbytes
        if self.descriptor is not None:
            s += self.descriptor.nbytes
        return s


# ── Abstract backend ────────────────────────────────────────────────

class PoseGraphSLAM:
    """Lightweight 2D pose-graph SLAM with in-memory keyframes."""

    def __init__(self):
        self._gmap = GlobalMap()
        self._keyframes: list[Keyframe] = []
        self._edges: list[tuple] = []       # (id_i, id_j, dx, dy, dtheta, info)
        self._next_id = 0
        self._last_kf_pose = None           # (x, y, theta) of most recent keyframe
        self._last_kf_id = -1

        # descriptor matrix for fast search (N × DESC_N_RINGS)
        self._desc_mat = np.empty((0, DESC_N_RINGS), dtype=np.float32)

        self._loop_count = 0
        self._kf_since_mem_check = 0

    # ── SlamBackend interface ───────────────────────────────────────

    def keyframe_check(self, obs_ego, known_ego, x, y, theta,
                       ego_cx, ego_cy, ego_px_size=0.01):
        """Check keyframing (GPU handles the actual map update)."""
        if self._should_keyframe(x, y, theta):
            self._create_keyframe(obs_ego, known_ego, x, y, theta,
                                  ego_cx, ego_cy, ego_px_size)

    # ── Keyframe management ─────────────────────────────────────────

    def _should_keyframe(self, x, y, theta):
        if self._last_kf_pose is None:
            return True
        lx, ly, lt = self._last_kf_pose
        dist = math.sqrt((x - lx) ** 2 + (y - ly) ** 2)
        dtheta = abs(_norm_angle(theta - lt))
        return dist >= KF_DIST or dtheta >= KF_ANGLE

    def _create_keyframe(self, obs, known, x, y, theta, cx, cy, px_size):
        kf = Keyframe()
        kf.id = self._next_id
        self._next_id += 1
        kf.x, kf.y, kf.theta = x, y, theta
        kf.cx, kf.cy, kf.px_size = cx, cy, px_size
        kf.timestamp = time.monotonic()

        # Store cropped ROI of observation
        rows = np.any(known > 0, axis=1)
        cols = np.any(known > 0, axis=0)
        if rows.any() and cols.any():
            r0, r1 = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1]) + 1
            c0, c1 = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1]) + 1
            kf.obs_roi = obs[r0:r1, c0:c1].copy()
            kf.known_roi = known[r0:r1, c0:c1].copy()
            kf.roi_bounds = (r0, r1, c0, c1)
        else:
            kf.obs_roi = None
            kf.known_roi = None
            kf.roi_bounds = None

        kf.descriptor = _radial_descriptor(obs, cx, cy)
        kf.thumb = cv2.resize(obs, (THUMB_SZ, THUMB_SZ),
                               interpolation=cv2.INTER_AREA)

        self._keyframes.append(kf)
        self._desc_mat = np.vstack([self._desc_mat, kf.descriptor[np.newaxis, :]])

        # Odometry edge to previous keyframe
        if self._last_kf_id >= 0:
            prev = self._keyframes[self._last_kf_id]
            self._add_odom_edge(prev, kf)

        self._last_kf_pose = (x, y, theta)
        self._last_kf_id = len(self._keyframes) - 1

        # Loop closure detection
        self._check_loop_closure(kf)

        # Memory management
        self._kf_since_mem_check += 1
        if self._kf_since_mem_check >= MEM_CHECK_EVERY:
            self._kf_since_mem_check = 0
            self._manage_memory()

    # ── Odometry edges ──────────────────────────────────────────────

    def _add_odom_edge(self, kf_i, kf_j):
        ct = math.cos(kf_i.theta)
        st = math.sin(kf_i.theta)
        dpx = kf_j.x - kf_i.x
        dpy = kf_j.y - kf_i.y
        local_dx = ct * dpx + st * dpy
        local_dy = -st * dpx + ct * dpy
        local_dt = _norm_angle(kf_j.theta - kf_i.theta)
        self._edges.append((kf_i.id, kf_j.id, local_dx, local_dy, local_dt,
                            ODOM_INFO))

    # ── Place recognition & loop closure ────────────────────────────

    def _check_loop_closure(self, kf):
        n = len(self._keyframes)
        if n <= LOOP_MIN_FRAMES + 1:
            return

        # Compare descriptor against all candidates (skip recent ones)
        n_cand = n - LOOP_MIN_FRAMES - 1
        cand_descs = self._desc_mat[:n_cand]
        norms = np.linalg.norm(cand_descs - cand_descs.mean(axis=1, keepdims=True),
                               axis=1)
        kf_norm = np.linalg.norm(kf.descriptor - kf.descriptor.mean())
        denom = norms * kf_norm
        denom[denom < 1e-8] = 1e-8
        centered_cand = cand_descs - cand_descs.mean(axis=1, keepdims=True)
        centered_kf = kf.descriptor - kf.descriptor.mean()
        cos_sims = (centered_cand @ centered_kf) / denom

        # Filter by score
        candidates = np.where(cos_sims >= LOOP_SCORE_THRESH)[0]
        if len(candidates) == 0:
            return

        # Sort by score (best first)
        order = np.argsort(-cos_sims[candidates])
        for ci in candidates[order][:5]:
            cand_kf = self._keyframes[ci]
            dx = kf.x - cand_kf.x
            dy = kf.y - cand_kf.y
            spatial_dist = math.sqrt(dx * dx + dy * dy)
            if spatial_dist < LOOP_MIN_DIST:
                continue

            # Scan-match verification
            if kf.thumb is None or cand_kf.thumb is None:
                continue
            sm_dx, sm_dy, sm_dt, sm_score = _scan_match(kf.thumb, cand_kf.thumb)
            if sm_score < LOOP_MATCH_THRESH:
                continue

            # Relative pose measurement for the loop edge
            ct_c = math.cos(cand_kf.theta)
            st_c = math.sin(cand_kf.theta)
            raw_dx = kf.x - cand_kf.x
            raw_dy = kf.y - cand_kf.y
            local_dx = ct_c * raw_dx + st_c * raw_dy
            local_dy = -st_c * raw_dx + ct_c * raw_dy
            local_dt = _norm_angle(kf.theta - cand_kf.theta)

            self._edges.append((cand_kf.id, kf.id,
                                local_dx, local_dy, local_dt, LOOP_INFO))
            self._loop_count += 1
            print("slam: loop closure #%d  kf%d↔kf%d  dist=%.2fm  "
                  "desc=%.2f  match=%.2f" %
                  (self._loop_count, cand_kf.id, kf.id,
                   spatial_dist, cos_sims[ci], sm_score))

            self._optimize_and_rebuild()
            return  # one loop closure per keyframe

    # ── Optimisation ────────────────────────────────────────────────

    def _optimize_and_rebuild(self):
        t0 = time.monotonic()

        nodes = {kf.id: (kf.x, kf.y, kf.theta) for kf in self._keyframes}
        corrected = _optimize_pose_graph(nodes, self._edges)

        max_shift = 0.0
        for kf in self._keyframes:
            nx, ny, nt = corrected[kf.id]
            dx = nx - kf.x
            dy = ny - kf.y
            max_shift = max(max_shift, math.sqrt(dx * dx + dy * dy))
            kf.x, kf.y, kf.theta = nx, ny, nt

        t1 = time.monotonic()

        if max_shift < 0.005:
            print("slam: optimisation converged, max_shift=%.4fm (%.1fms, skip rebuild)"
                  % (max_shift, (t1 - t0) * 1000))
            return

        # Rebuild map from corrected keyframes
        self._gmap._map[:] = UNKNOWN_VAL
        self._gmap._height_map[:] = 0
        self._gmap._count = 0

        for kf in self._keyframes:
            if kf.obs_roi is None:
                continue
            r0, r1, c0, c1 = kf.roi_bounds
            obs_full = np.zeros((max(r1, kf.obs_roi.shape[0] + r0),
                                 max(c1, kf.obs_roi.shape[1] + c0)),
                                dtype=np.uint8)
            known_full = np.zeros_like(obs_full)
            obs_full[r0:r0 + kf.obs_roi.shape[0],
                     c0:c0 + kf.obs_roi.shape[1]] = kf.obs_roi
            known_full[r0:r0 + kf.known_roi.shape[0],
                       c0:c0 + kf.known_roi.shape[1]] = kf.known_roi

            M = GlobalMap._forward_affine(kf.x, kf.y, kf.theta,
                                          kf.cx, kf.cy, kf.px_size)
            h_obs, w_obs = obs_full.shape
            # Encode: 0=unobserved, 1=free, 2-101=obstacle (height+1)
            ego_obs = np.zeros((h_obs, w_obs), dtype=np.uint8)
            free = (known_full > 0) & (obs_full == 0)
            ego_obs[free] = 1
            obs_px = obs_full > 0
            ego_obs[obs_px] = np.minimum(
                obs_full[obs_px].astype(np.uint16) + 1, 101).astype(np.uint8)

            corners = np.float32([[0, 0], [w_obs, 0], [w_obs, h_obs],
                                  [0, h_obs]]).reshape(1, 4, 2)
            gc = cv2.transform(corners, M).reshape(-1, 2)
            gr0 = max(0, int(np.floor(gc[:, 1].min())))
            gr1 = min(MAP_H, int(np.ceil(gc[:, 1].max())) + 1)
            gc0 = max(0, int(np.floor(gc[:, 0].min())))
            gc1 = min(MAP_W, int(np.ceil(gc[:, 0].max())) + 1)
            if gr1 <= gr0 or gc1 <= gc0:
                continue

            M_roi = M.copy()
            M_roi[0, 2] -= gc0
            M_roi[1, 2] -= gr0
            roi_w, roi_h = gc1 - gc0, gr1 - gr0
            warped = cv2.warpAffine(ego_obs, M_roi, (roi_w, roi_h),
                                    flags=cv2.INTER_NEAREST, borderValue=0)

            free_mask = warped == 1
            obs_mask = warped >= 2
            roi_map = self._gmap._map[gr0:gr1, gc0:gc1]
            m = roi_map.astype(np.int16)
            m[free_mask] = np.minimum(m[free_mask] + 64, 255)
            m[obs_mask] = np.maximum(m[obs_mask] - 64, 0)
            roi_map[:] = m.astype(np.uint8)

            # Accumulate heights (tallest wins)
            if obs_mask.any():
                hm_roi = self._gmap._height_map[gr0:gr1, gc0:gc1]
                obs_h = (warped[obs_mask].astype(np.uint16) - 1).astype(np.uint8)
                hm_roi[obs_mask] = np.maximum(hm_roi[obs_mask], obs_h)

        t2 = time.monotonic()
        print("slam: optimise=%.1fms  rebuild=%.1fms  max_shift=%.3fm  "
              "keyframes=%d  edges=%d  loops=%d" %
              ((t1 - t0) * 1000, (t2 - t1) * 1000, max_shift,
               len(self._keyframes), len(self._edges), self._loop_count))

    # ── Memory management ───────────────────────────────────────────

    def _estimate_memory(self):
        total = 0
        for kf in self._keyframes:
            total += kf.approx_bytes()
        total += self._gmap._map.nbytes
        total += self._gmap._height_map.nbytes
        total += self._gmap._out.nbytes
        total += self._desc_mat.nbytes
        total += len(self._edges) * 200  # rough edge overhead
        return total

    def _manage_memory(self):
        mem = self._estimate_memory()
        if mem < MEM_HIGH:
            return

        print("slam: memory %.1f MB > %.1f MB, pruning..."
              % (mem / 1e6, MEM_HIGH / 1e6))

        # Keep first, last, and loop-closure keyframes; thin the rest.
        loop_ids = set()
        for e in self._edges:
            if np.allclose(e[5], LOOP_INFO):
                loop_ids.add(e[0])
                loop_ids.add(e[1])

        protected = {self._keyframes[0].id, self._keyframes[-1].id} | loop_ids

        # Remove every other non-protected keyframe (thin the graph)
        keep = []
        removed_ids = set()
        for i, kf in enumerate(self._keyframes):
            if kf.id in protected or i % 2 == 0:
                keep.append(kf)
            else:
                removed_ids.add(kf.id)
                # free observation data
                kf.obs_roi = None
                kf.known_roi = None
                kf.thumb = None

        # Remove edges that reference removed keyframes, bridge gaps
        new_edges = []
        for e in self._edges:
            if e[0] not in removed_ids and e[1] not in removed_ids:
                new_edges.append(e)

        self._keyframes = keep
        self._edges = new_edges

        # Rebuild descriptor matrix
        if len(self._keyframes) > 0:
            descs = [kf.descriptor for kf in self._keyframes
                     if kf.descriptor is not None]
            self._desc_mat = np.vstack(descs) if descs else np.empty(
                (0, DESC_N_RINGS), dtype=np.float32)
        else:
            self._desc_mat = np.empty((0, DESC_N_RINGS), dtype=np.float32)

        mem_after = self._estimate_memory()
        print("slam: pruned %d → %d keyframes, %.1f MB → %.1f MB" %
              (len(self._keyframes) + len(removed_ids), len(self._keyframes),
               mem / 1e6, mem_after / 1e6))

    # ── Stats ───────────────────────────────────────────────────────

    def stats(self):
        return {
            'keyframes': len(self._keyframes),
            'edges': len(self._edges),
            'loop_closures': self._loop_count,
            'memory_mb': self._estimate_memory() / 1e6,
        }

"""globalmap.py – World-frame 2D occupancy grid with 3-frame consensus gating.

960×720 pixels at 2 cm/px = 19.2 m × 14.4 m coverage.
World frame: x-right, y-up, origin at robot's start position.

Confidence map (_map): each cell stores a byte (0–255):
  0   = strong obstacle evidence
  128 = unknown (initial)
  255 = strong free-space evidence

Height map (_height_map): each cell stores uint8 (0–100):
  0 = floor / no obstacle
  1–100 = obstacle height in cm above floor (median of 3-frame consensus)

Observations from the ego-frame cameras are warped into global space
at 30 fps.  Only pixels where the last 3 consecutive frames agree
(all free or all obstacle) are applied to the main map, preventing
transient noise from corrupting the persistent map.  On obstacle
consensus, the median height of the 3 frames is stored in the height map.
"""

import math
import time
import numpy as np
import cv2

MAP_W = 960
MAP_H = 720
PX_SIZE = 0.02        # metres per pixel (2 cm)
ORIGIN_X = MAP_W // 2  # pixel 480 = world x=0
ORIGIN_Y = MAP_H // 2  # pixel 360 = world y=0

UNKNOWN_VAL = 128
FREE_DISPLAY = 255
OBS_DISPLAY = 64
UNK_DISPLAY = 160

FREE_THRESH = 190
OBS_THRESH = 90

# One consensus (3 agreeing frames at 30fps = 100ms) should be enough
# to flip a cell.  128 takes unknown→free or unknown→obstacle in a
# single consensus; free↔obstacle requires 2 (200ms hysteresis).
EVIDENCE_UP = 128
EVIDENCE_DOWN = 128

CONSENSUS_N = 3

SPRITE_SZ = 192
_SC = SPRITE_SZ // 2  # 96 — sprite centre

# ── 3D follow-cam parameters ─────────────────────────────────────
CAM_BEHIND = 2.5     # metres behind robot
CAM_HEIGHT = 3.0     # metres above ground
CAM_LOOK_AHEAD = 1.0 # metres ahead of robot (look-at target)
CAM_FOV_DEG = 60     # vertical field of view (degrees)
LEFT_ZOOM = 0.5      # 2D map scale (0.5 = show 2× more area)


class GlobalMap:
    def __init__(self):
        t0 = time.monotonic()
        self._map = np.full((MAP_H, MAP_W), UNKNOWN_VAL, dtype=np.uint8)
        self._height_map = np.zeros((MAP_H, MAP_W), dtype=np.uint8)
        self._ring = [None] * CONSENSUS_N  # (r0,r1,c0,c1,signed_roi) per slot
        self._ring_idx = 0
        self._count = 0
        self._spr_rgba, self._spr_lab = self._make_sprite()
        self._render_times = []
        self._update_times = []
        self._out = np.empty((MAP_H, MAP_W, 3), dtype=np.uint8)
        t1 = time.monotonic()
        self._3d_rays, self._3d_f = self._precompute_3d_rays()
        t2 = time.monotonic()
        self._3d_wr = np.empty_like(self._3d_rays)
        self._robot_mesh = self._build_robot_mesh()
        t3 = time.monotonic()
        spr, sx, sy = self._prerender_robot_3d_sprite()
        self._robot_3d_spr = spr
        self._robot_3d_x0 = sx
        self._robot_3d_y0 = sy
        self._rows_f = np.repeat(
            np.arange(MAP_H, dtype=np.float32)[:, None], MAP_W // 2, axis=1)
        self._pitch_cos = math.sqrt(CAM_BEHIND**2 + CAM_LOOK_AHEAD**2) / max(
            math.sqrt(CAM_BEHIND**2 + CAM_LOOK_AHEAD**2 + CAM_HEIGHT**2), 1e-6)
        t4 = time.monotonic()
        # Self-test: verify map persistence
        self._map[0, 0] = 255
        assert self._map[0, 0] == 255, "MAP WRITE FAILED"
        roi = self._map[0:1, 0:1]
        roi[:] = 0
        assert self._map[0, 0] == 0, "MAP ROI WRITE FAILED (not a view!)"
        self._map[0, 0] = UNKNOWN_VAL  # restore
        print(f"GlobalMap: init {(t1-t0)*1e3:.0f}ms  rays {(t2-t1)*1e3:.0f}ms  "
              f"mesh {(t3-t2)*1e3:.0f}ms  sprite {(t4-t3)*1e3:.0f}ms  "
              f"robot_3d={spr.shape} at ({sx},{sy})")

    def update(self, obs_ego, known_ego, x, y, theta,
               ego_cx, ego_cy, ego_px_size=0.01):
        """Feed one frame of ego-space observations and robot pose.

        obs_ego values: 0=free, 1-100=obstacle height in cm.
        known_ego: 255 where sensor has data, 0 elsewhere.

        ROI-optimised: only warp/compare/update the bounding box of the
        observation footprint in global space (~200×200 vs 960×720).
        """
        t0 = time.monotonic()
        h, w = obs_ego.shape[:2]

        if self._count < 5:
            nk = int(np.count_nonzero(known_ego))
            no = int(np.count_nonzero(obs_ego))
            print("gmap.update[%d]: known_px=%d obs_px=%d ego=%dx%d "
                  "pose=(%.3f,%.3f,%.1f°) cx=%.1f cy=%.1f px=%.4f"
                  % (self._count, nk, no, w, h,
                     x, y, np.degrees(theta), ego_cx, ego_cy, ego_px_size))

        # Encode as: 0=unobserved, 1=free, 2-101=obstacle (height+1)
        ego_obs = np.zeros((h, w), dtype=np.uint8)
        free = (known_ego > 0) & (obs_ego == 0)
        ego_obs[free] = 1
        obs_px = obs_ego > 0
        ego_obs[obs_px] = np.minimum(obs_ego[obs_px].astype(np.uint16) + 1,
                                     101).astype(np.uint8)

        M_fwd = self._forward_affine(x, y, theta, ego_cx, ego_cy, ego_px_size)
        t1 = time.monotonic()

        # Compute ROI: transform ego corners → global pixel bounds
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(1, 4, 2)
        gc = cv2.transform(corners, M_fwd).reshape(-1, 2)
        r0 = max(0, int(np.floor(gc[:, 1].min())))
        r1 = min(MAP_H, int(np.ceil(gc[:, 1].max())) + 1)
        c0 = max(0, int(np.floor(gc[:, 0].min())))
        c1 = min(MAP_W, int(np.ceil(gc[:, 0].max())) + 1)
        if r1 <= r0 or c1 <= c0:
            if self._count % 30 == 0:
                print("gmap: observation outside map bounds! r=%d:%d c=%d:%d" %
                      (r0, r1, c0, c1))
            return

        # Warp only into the ROI-sized output (shift affine origin)
        M_roi = M_fwd.copy()
        M_roi[0, 2] -= c0
        M_roi[1, 2] -= r0
        roi_w, roi_h = c1 - c0, r1 - r0
        obs_roi = cv2.warpAffine(
            ego_obs, M_roi, (roi_w, roi_h),
            flags=cv2.INTER_NEAREST, borderValue=0)
        t2 = time.monotonic()

        # Encode as signed int8: free=+1, obs=-(height_cm), unobserved=0
        signed = np.zeros((roi_h, roi_w), dtype=np.int8)
        signed[obs_roi == 1] = 1
        obs_mask = obs_roi >= 2
        signed[obs_mask] = (1 - obs_roi[obs_mask].astype(np.int16)).astype(np.int8)

        self._ring[self._ring_idx] = (r0, r1, c0, c1, signed)
        self._ring_idx = (self._ring_idx + 1) % CONSENSUS_N
        self._count += 1

        if self._count < CONSENSUS_N:
            return
        t3 = time.monotonic()

        # Consensus within intersection of all 3 ROIs
        rois = self._ring
        ir0 = max(r[0] for r in rois)
        ir1 = min(r[1] for r in rois)
        ic0 = max(r[2] for r in rois)
        ic1 = min(r[3] for r in rois)
        if ir1 <= ir0 or ic1 <= ic0:
            self._count += 0  # no-op but marks this path was taken
            if self._count % 30 == 0:
                print("gmap: EMPTY consensus intersection! ROIs: %s"
                      % [(r[0],r[1],r[2],r[3]) for r in rois])
            return

        # Extract overlapping sub-regions from each ring entry
        slices = []
        for br0, br1, bc0, bc1, s in rois:
            slices.append(s[ir0 - br0:ir1 - br0, ic0 - bc0:ic1 - bc0])
        s0, s1, s2 = slices

        all_free = (s0 == 1) & (s1 == 1) & (s2 == 1)
        all_obs = (s0 < 0) & (s1 < 0) & (s2 < 0)

        roi_map = self._map[ir0:ir1, ic0:ic1]
        m = roi_map.astype(np.int16)
        m[all_free] = np.minimum(m[all_free] + EVIDENCE_UP, 255)
        m[all_obs] = np.maximum(m[all_obs] - EVIDENCE_DOWN, 0)
        roi_map[:] = m.astype(np.uint8)

        # Height map: median of the 3 frames' heights where obstacle confirmed
        roi_hm = self._height_map[ir0:ir1, ic0:ic1]
        if all_obs.any():
            h0 = np.abs(s0[all_obs]).astype(np.uint8)
            h1 = np.abs(s1[all_obs]).astype(np.uint8)
            h2 = np.abs(s2[all_obs]).astype(np.uint8)
            roi_hm[all_obs] = np.maximum(
                np.minimum(h0, h1),
                np.minimum(np.maximum(h0, h1), h2))
        if all_free.any():
            roi_hm[all_free] = 0
        t4 = time.monotonic()

        self._update_times.append((t1 - t0, t2 - t1, t3 - t2, t4 - t3))
        if len(self._update_times) % 30 == 0:
            nfree = int(np.count_nonzero(self._map > FREE_THRESH))
            nobs  = int(np.count_nonzero(self._map < OBS_THRESH))
            ntot  = MAP_H * MAP_W
            avg = np.mean(self._update_times[-30:], axis=0) * 1000
            n_af = int(all_free.sum())
            n_ao = int(all_obs.sum())
            print("gmap: free=%d obs=%d unk=%d | consensus free_px=%d obs_px=%d | "
                  "pose=(%.3f,%.3f,%.1f°) roi=%dx%d %.1fms"
                  % (nfree, nobs, ntot - nfree - nobs,
                     n_af, n_ao,
                     x, y, np.degrees(theta),
                     roi_h, roi_w, sum(avg)))

    def project_to_ego(self, x, y, theta,
                       ego_cx, ego_cy, ego_px_size, ego_h, ego_w):
        """Warp global map back to ego-centred coordinates for safety."""
        M_inv = self._inverse_affine(x, y, theta, ego_cx, ego_cy, ego_px_size)
        return cv2.warpAffine(
            self._map, M_inv, (ego_w, ego_h),
            flags=cv2.INTER_NEAREST, borderValue=UNKNOWN_VAL)

    @staticmethod
    def _precompute_3d_rays():
        """Precompute camera-space ray directions for the 3D follow-cam.

        Returns (rays, focal_length):
          rays — (MAP_H, MAP_W//2, 3) float32, normalised.
          focal_length — float, for projecting points to screen.
        """
        h, w = MAP_H, MAP_W // 2
        f = h / (2.0 * math.tan(math.radians(CAM_FOV_DEG) / 2.0))
        px, py = np.meshgrid(
            np.arange(w, dtype=np.float32),
            np.arange(h, dtype=np.float32))
        rays = np.stack([
            (px - w * 0.5) / f,
            -(py - h * 0.5) / f,
            -np.ones((h, w), dtype=np.float32),
        ], axis=2)
        rays /= np.linalg.norm(rays, axis=2, keepdims=True)
        return rays, f

    # ── 3D robot mesh ────────────────────────────────────────────

    @staticmethod
    def _build_robot_mesh():
        """Build hardcoded 3D robot mesh (origin = axle centre on ground, +X fwd).

        Physical dims: 30 cm wide frame, 28 cm front-to-back, 17.13 cm wheel
        diameter, 5 cm wheel width, 1 cm frame-wheel gap, ~90 cm mast from
        base, 15 cm camera arm.  Seeed J4012 Orin NX, two RealSense cameras.

        Returns (verts, faces, colors):
          verts  — (N, 3) float32 in metres
          faces  — list of int32 arrays (polygon vertex indices per face)
          colors — (M, 3) uint8, RGB per face
        """
        vl, fl, cl = [], [], []
        R = 0.0857

        def _box(x0, y0, z0, x1, y1, z1, rgb):
            n = len(vl)
            vl.extend([[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
                        [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]])
            for q in ([n,n+3,n+2,n+1],[n+4,n+5,n+6,n+7],
                       [n,n+1,n+5,n+4],[n+2,n+3,n+7,n+6],
                       [n,n+4,n+7,n+3],[n+1,n+2,n+6,n+5]):
                fl.append(np.array(q, dtype=np.int32))
                cl.append(rgb)

        def _wheel(cx, cz, y0, y1, radius, rgb, ns=10):
            n = len(vl)
            ang = np.linspace(0, 2*np.pi, ns, endpoint=False)
            for a in ang:
                vl.append([cx+radius*np.cos(a), y0, cz+radius*np.sin(a)])
            for a in ang:
                vl.append([cx+radius*np.cos(a), y1, cz+radius*np.sin(a)])
            for i in range(ns):
                j = (i+1) % ns
                fl.append(np.array([n+i, n+ns+i, n+ns+j, n+j], dtype=np.int32))
                cl.append(rgb)
            fl.append(np.arange(n, n+ns, dtype=np.int32))
            cl.append([min(c+12, 255) for c in rgb])
            fl.append(np.arange(n+2*ns-1, n+ns-1, -1, dtype=np.int32))
            cl.append([min(c+12, 255) for c in rgb])

        _box(-0.04, -0.15, R-0.015, 0.24, 0.15, R+0.01,
             [45, 45, 50])                                       # base plate
        _box(-0.04, -0.13, R+0.01,  0.10, 0.13, R+0.16,
             [30, 30, 35])                                       # electronics box
        _box( 0.10, -0.12, R+0.01,  0.22, 0.12, R+0.07,
             [35, 35, 40])                                       # front nose
        _wheel(0, R, -0.21, -0.16, R*0.97, [22, 22, 26])       # left wheel
        _wheel(0, R,  0.16,  0.21, R*0.97, [22, 22, 26])       # right wheel
        _box( 0.21, -0.015, 0.0,   0.25, 0.015, 0.04,
             [40, 40, 45])                                       # caster
        _box(-0.015,-0.015, R+0.16, 0.015, 0.015, 1.00,
             [100, 105, 115])                                    # mast
        _box( 0.015,-0.015, 0.97,   0.165, 0.015, 1.00,
             [85, 90, 100])                                      # camera arm
        _box( 0.10, -0.04,  1.00,   0.19,  0.04,  1.025,
             [30, 55, 80])                                       # RS1 top-down
        _box( 0.165,-0.03,  0.94,   0.20,  0.03,  0.97,
             [30, 55, 80])                                       # RS2 forward
        _box(-0.01, -0.04,  R+0.16, 0.09,  0.04,  R+0.185,
             [40, 55, 40])                                       # Jetson NX

        return (np.array(vl, dtype=np.float32), fl,
                np.array(cl, dtype=np.uint8))

    def _prerender_robot_3d_sprite(self):
        """Pre-render 3D robot from the follow-cam angle into an RGBA sprite.

        The follow-cam always sits at the same position relative to the robot
        so the rendered appearance is constant.  We bake it once into a tight-
        cropped RGBA patch and record its screen position in the 480×720 panel.

        Returns (sprite_rgba, x0, y0) where (x0, y0) is the top-left corner
        of the sprite in the right-panel coordinate system.
        """
        HALF = MAP_W // 2
        verts, faces, colors = self._robot_mesh

        cam = np.float32([-CAM_BEHIND, 0, CAM_HEIGHT])
        tgt = np.float32([CAM_LOOK_AHEAD, 0, 0])
        fwd = tgt - cam
        fwd /= np.linalg.norm(fwd)
        up = np.float32([0, 0, 1])
        rt = np.cross(fwd, up)
        rn = np.linalg.norm(rt)
        rt = rt / rn if rn > 1e-6 else np.float32([1, 0, 0])
        up_a = np.cross(rt, fwd)
        Rm = np.stack([rt, up_a, -fwd], axis=1).astype(np.float32)

        cv3 = (verts - cam) @ Rm
        f = self._3d_f
        ok = cv3[:, 2] < -0.01
        scr_x = np.where(ok, cv3[:, 0]*f/(-cv3[:, 2]) + HALF*0.5, -9999)
        scr_y = np.where(ok, -cv3[:, 1]*f/(-cv3[:, 2]) + MAP_H*0.5, -9999)

        light = np.float32([0.2, -0.3, -0.9])
        light /= np.linalg.norm(light)

        draw = []
        for i, fidx in enumerate(faces):
            fv = cv3[fidx]
            if (fv[:, 2] >= -0.01).any():
                continue
            e1, e2 = fv[1] - fv[0], fv[-1] - fv[0]
            nrm = np.cross(e1, e2)
            nl = np.linalg.norm(nrm)
            if nl < 1e-8:
                continue
            nrm /= nl
            if np.dot(nrm, fv.mean(axis=0)) > 0:
                continue
            brt = max(0.3, min(1.0, -np.dot(nrm, light)))
            pts = np.column_stack([scr_x[fidx].astype(np.int32),
                                   scr_y[fidx].astype(np.int32)])
            col = np.clip(colors[i].astype(np.float32) * brt,
                          0, 255).astype(np.uint8).tolist()
            draw.append((fv[:, 2].mean(), pts, col))

        draw.sort(key=lambda d: d[0])

        rgb = np.zeros((MAP_H, HALF, 3), dtype=np.uint8)
        amask = np.zeros((MAP_H, HALF), dtype=np.uint8)
        for _, pts, col in draw:
            cv2.fillConvexPoly(rgb, pts, col)
            cv2.fillConvexPoly(amask, pts, 255)
        for _, pts, col in draw:
            cv2.polylines(rgb, [pts], True,
                          [max(0, c - 20) for c in col], 1, cv2.LINE_AA)

        rows = np.any(amask > 0, axis=1)
        cmask = np.any(amask > 0, axis=0)
        if not rows.any():
            return np.zeros((1, 1, 4), dtype=np.uint8), 0, 0
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cmask)[0][[0, -1]]
        r0, c0 = max(0, r0-2), max(0, c0-2)
        r1 = min(MAP_H-1, r1+2) + 1
        c1 = min(HALF-1, c1+2) + 1
        canvas = np.dstack([rgb[r0:r1, c0:c1], amask[r0:r1, c0:c1]])
        return canvas.copy(), c0, r0

    def render(self, x, y, theta, trail_xy=None,
               fwd_scale=1.0, bwd_scale=1.0, ang_scale=1.0):
        """960×720 split view: left=zoomed-out 2D map, right=3D follow-cam."""
        HALF = MAP_W // 2
        rc = int(ORIGIN_X + x / PX_SIZE)
        rr = int(ORIGIN_Y - y / PX_SIZE)
        out = self._out

        # --- Left panel: zoomed-out 2D map centred on robot ---
        M_left = np.float64([
            [LEFT_ZOOM, 0, HALF * 0.5 - rc * LEFT_ZOOM],
            [0, LEFT_ZOOM, MAP_H * 0.5 - rr * LEFT_ZOOM],
        ])
        left_conf = cv2.warpAffine(
            self._map, M_left, (HALF, MAP_H),
            flags=cv2.INTER_NEAREST, borderValue=UNKNOWN_VAL)
        left_hm = cv2.warpAffine(
            self._height_map, M_left, (HALF, MAP_H),
            flags=cv2.INTER_NEAREST, borderValue=0)

        left_g = np.full((MAP_H, HALF), UNK_DISPLAY, dtype=np.uint8)
        left_g[left_conf > FREE_THRESH] = FREE_DISPLAY
        obs_m = left_conf < OBS_THRESH
        left_g[obs_m] = np.clip(
            OBS_DISPLAY - (left_hm[obs_m].astype(np.int16) >> 1),
            10, OBS_DISPLAY).astype(np.uint8)
        left = cv2.cvtColor(left_g, cv2.COLOR_GRAY2BGR)

        if trail_xy is not None and len(trail_xy) >= 2:
            tc = (ORIGIN_X + trail_xy[:, 0] / PX_SIZE).astype(np.float32)
            tr = (ORIGIN_Y - trail_xy[:, 1] / PX_SIZE).astype(np.float32)
            tc_z = (tc * LEFT_ZOOM + M_left[0, 2]).astype(np.int32)
            tr_z = (tr * LEFT_ZOOM + M_left[1, 2]).astype(np.int32)
            keep = (tc_z >= 0) & (tc_z < HALF) & (tr_z >= 0) & (tr_z < MAP_H)
            pts = np.column_stack([tc_z[keep], tr_z[keep]])
            if len(pts) >= 2:
                cv2.polylines(left, [pts.reshape(-1, 1, 2)], False,
                              (80, 140, 255), 1, cv2.LINE_AA)

        rob_zx = int(rc * LEFT_ZOOM + M_left[0, 2])
        rob_zy = int(rr * LEFT_ZOOM + M_left[1, 2])
        self._draw_robot(left, rob_zx, rob_zy, theta,
                         fwd_scale, bwd_scale, ang_scale,
                         zoom=LEFT_ZOOM * 0.3)
        out[:, :HALF] = left

        # --- Right panel: 3D follow-cam ---
        out[:, HALF:] = self._render_3d(x, y, theta, trail_xy)

        return out

    def _render_3d(self, x, y, theta, trail_xy=None):
        """3D follow-cam with heightmap terrain, pillar extrusion, and lighting."""
        HALF = MAP_W // 2

        ct, st = math.cos(theta), math.sin(theta)
        cam_pos = np.float32([
            x - CAM_BEHIND * ct,
            y - CAM_BEHIND * st,
            CAM_HEIGHT])
        target = np.float32([
            x + CAM_LOOK_AHEAD * ct,
            y + CAM_LOOK_AHEAD * st,
            0.0])

        fwd = target - cam_pos
        fwd /= np.linalg.norm(fwd)
        up = np.float32([0, 0, 1])
        right = np.cross(fwd, up)
        rn = np.linalg.norm(right)
        if rn < 1e-6:
            right = np.float32([1, 0, 0])
        else:
            right /= rn
        up_act = np.cross(right, fwd)
        R = np.stack([right, up_act, -fwd], axis=1).astype(np.float32)

        np.matmul(self._3d_rays, R.T, out=self._3d_wr)
        wr = self._3d_wr

        # ── ground-plane intersection ──
        denom = wr[:, :, 2].copy()
        denom[np.abs(denom) < 1e-6] = -1e-6
        t_gnd = -cam_pos[2] / denom
        valid = t_gnd > 0

        wx = cam_pos[0] + t_gnd * wr[:, :, 0]
        wy = cam_pos[1] + t_gnd * wr[:, :, 1]
        mc = (wx / PX_SIZE + ORIGIN_X).astype(np.float32)
        mr = (ORIGIN_Y - wy / PX_SIZE).astype(np.float32)

        # Sample confidence and height from ground-plane coords (2D map)
        conf = cv2.remap(self._map, mc, mr,
                         cv2.INTER_NEAREST, borderValue=UNKNOWN_VAL)
        h_f = cv2.remap(self._height_map, mc, mr,
                        cv2.INTER_NEAREST, borderValue=0)

        free_m = valid & (conf > FREE_THRESH)
        obs_m = valid & (conf < OBS_THRESH)
        h_float = h_f.astype(np.float32)

        # Height-adjusted depth (for pillar sizing only)
        z_surf = h_float * 0.01
        t2 = np.where(obs_m, (z_surf - cam_pos[2]) / denom, t_gnd)

        # ── base coloring (FSD palette) ──
        out = np.full((MAP_H, HALF, 3), 145, dtype=np.uint8)
        out[~valid] = [32, 35, 45]
        out[free_m] = [195, 202, 185]

        if obs_m.any():
            hn = h_float[obs_m]
            out[obs_m, 0] = np.clip(90 - hn * 0.5, 30, 90).astype(np.uint8)
            out[obs_m, 1] = np.clip(75 - hn * 0.3, 25, 75).astype(np.uint8)
            out[obs_m, 2] = np.clip(65 - hn * 0.2, 20, 65).astype(np.uint8)

        # ── edge lighting (height discontinuities) ──
        gx = cv2.Sobel(h_float, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(h_float, cv2.CV_32F, 0, 1, ksize=3)
        edge = np.sqrt(gx * gx + gy * gy)
        shade = np.clip(1.0 - edge * 0.08, 0.45, 1.0)[:, :, None]
        out = (out.astype(np.float32) * shade).astype(np.uint8)

        # ── pillar extrusion (side faces) ──
        depth = np.maximum(np.abs(t2), 0.3)
        MIN_PILLAR_PX = 3
        h_px = np.clip(
            h_float * 0.01 * self._3d_f * self._pitch_cos / depth,
            0, 60).astype(np.uint8)
        h_px[obs_m & (h_px < MIN_PILLAR_PX)] = MIN_PILLAR_PX
        h_px[~obs_m] = 0

        max_ext = int(h_px.max())
        if max_ext > 0:
            k_up = np.ones((max_ext + 1, 1), dtype=np.uint8)
            obs_up = cv2.dilate(obs_m.astype(np.uint8), k_up, anchor=(0, 0))

            pos_map = np.zeros((MAP_H, HALF), dtype=np.float32)
            pos_map[obs_m] = self._rows_f[obs_m]
            pos_up = cv2.dilate(pos_map, k_up, anchor=(0, 0))
            h_px_up = cv2.dilate(h_px, k_up, anchor=(0, 0))

            dist_from_obs = pos_up - self._rows_f
            side_m = ((obs_up > 0) & ~obs_m & valid
                      & (dist_from_obs > 0)
                      & (dist_from_obs <= h_px_up.astype(np.float32)))

            if side_m.any():
                ratio = dist_from_obs[side_m] / np.maximum(
                    h_px_up[side_m].astype(np.float32), 1.0)
                brt = 0.85 - ratio * 0.45
                out[side_m, 0] = np.clip(70 * brt, 15, 70).astype(np.uint8)
                out[side_m, 1] = np.clip(58 * brt, 12, 58).astype(np.uint8)
                out[side_m, 2] = np.clip(48 * brt, 8, 48).astype(np.uint8)

        # ── fog ──
        dist_val = np.abs(t2) * valid.astype(np.float32)
        fog = np.clip(dist_val / 12.0, 0, 0.75)[:, :, None]
        out = (out.astype(np.float32) * (1.0 - fog)
               + np.float32([145, 145, 150]) * fog).astype(np.uint8)

        # ── robot sprite ──
        spr = self._robot_3d_spr
        sh, sw = spr.shape[:2]
        sx0, sy0 = self._robot_3d_x0, self._robot_3d_y0
        ox0, oy0 = max(0, sx0), max(0, sy0)
        ox1, oy1 = min(HALF, sx0 + sw), min(MAP_H, sy0 + sh)
        if ox0 < ox1 and oy0 < oy1:
            px0, py0 = ox0 - sx0, oy0 - sy0
            patch = spr[py0:py0 + oy1 - oy0, px0:px0 + ox1 - ox0]
            a16 = patch[:, :, 3:4].astype(np.uint16)
            region = out[oy0:oy1, ox0:ox1]
            region[:] = ((region.astype(np.uint16) * (255 - a16) +
                          patch[:, :, :3].astype(np.uint16) * a16
                          ) >> 8).astype(np.uint8)

        # ── trail ──
        if trail_xy is not None and len(trail_xy) >= 2:
            trail_3d = np.column_stack([
                trail_xy, np.zeros(len(trail_xy), dtype=np.float64)])
            tc = (trail_3d - cam_pos.astype(np.float64)) @ R.astype(np.float64)
            in_front = tc[:, 2] < -0.1
            if in_front.any():
                tcf = tc[in_front]
                f = self._3d_f
                tu = (tcf[:, 0] * f / (-tcf[:, 2]) + HALF * 0.5).astype(np.int32)
                tv = (-tcf[:, 1] * f / (-tcf[:, 2]) + MAP_H * 0.5).astype(np.int32)
                keep = (tu >= 0) & (tu < HALF) & (tv >= 0) & (tv < MAP_H)
                pts = np.column_stack([tu[keep], tv[keep]])
                if len(pts) >= 2:
                    cv2.polylines(out, [pts.reshape(-1, 1, 2)], False,
                                  (80, 140, 255), 2, cv2.LINE_AA)

        return out

    # ── sprite-based robot rendering ──────────────────────────────

    @staticmethod
    def _make_sprite():
        """Pre-render robot sprite facing right at 192×192. Returns (rgba, labels).

        Labels: 0=background, 1=body, 2=track/caster, 3=arrow.
        Alpha: body/tracks=153 (60%), arrow=255 (100%).
        Drawn at 4× resolution; scaled down to 48×48 for global view.
        """
        S, C = SPRITE_SZ, _SC
        rgba = np.zeros((S, S, 4), dtype=np.uint8)
        lab = np.zeros((S, S), dtype=np.uint8)

        def _rect(x0, y0, x1, y1, a, lbl):
            cv2.rectangle(rgba, (x0, y0), (x1, y1), (0, 0, 0, a), -1)
            lab[y0:y1 + 1, x0:x1 + 1] = lbl

        _rect(C - 36, C - 28, C + 28, C + 28, 153, 1)    # body 65×57
        _rect(C - 16, C - 40, C + 16, C - 32, 153, 2)    # left track 33×9
        _rect(C - 16, C + 32, C + 16, C + 40, 153, 2)    # right track 33×9
        _rect(C - 44, C - 4, C - 36, C + 3, 153, 2)      # caster 9×8

        cv2.arrowedLine(rgba, (C - 20, C), (C + 24, C),
                        (255, 180, 0, 255), 2, tipLength=0.35)
        lab[rgba[:, :, 3] == 255] = 3

        return rgba, lab

    def _draw_robot(self, disp, rc, rr, theta,
                    fwd_scale, bwd_scale, ang_scale, zoom=1.0):
        """Tint, rotate, and alpha-composite the pre-rendered robot sprite."""
        S, C = SPRITE_SZ, _SC
        h, w = disp.shape[:2]
        spr = self._spr_rgba.copy()

        sf = min(fwd_scale, bwd_scale, ang_scale)
        body_rgb = [int(255 * (1 - sf)),
                    int(160 * sf + 60 * (1 - sf)),
                    int(255 * sf)]
        td = 1.0 - min(fwd_scale, bwd_scale)
        trk_rgb = [int(180 * td),
                   int(60 + 100 * (1 - td)),
                   int(180 * (1 - td))]

        spr[self._spr_lab == 1, :3] = body_rgb
        spr[self._spr_lab == 2, :3] = trk_rgb

        out_s = int(S * zoom)
        out_c = out_s // 2
        M = cv2.getRotationMatrix2D((float(C), float(C)),
                                    np.degrees(theta), zoom)
        M[0, 2] += out_c - C
        M[1, 2] += out_c - C
        rot = cv2.warpAffine(spr, M, (out_s, out_s),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0, 0, 0, 0))

        x0, y0 = rc - out_c, rr - out_c
        sx0, sy0 = max(0, -x0), max(0, -y0)
        dx0, dy0 = max(0, x0), max(0, y0)
        sw = min(out_s, w - x0) - sx0
        sh = min(out_s, h - y0) - sy0
        if sw <= 0 or sh <= 0:
            return

        patch = rot[sy0:sy0 + sh, sx0:sx0 + sw]
        a = patch[:, :, 3]
        mask = a > 0
        if not mask.any():
            return
        region = disp[dy0:dy0 + sh, dx0:dx0 + sw]
        am = a[mask].astype(np.uint16)[:, None]
        inv = np.uint16(255) - am
        region[mask] = ((region[mask].astype(np.uint16) * inv +
                         patch[:, :, :3][mask].astype(np.uint16) * am
                         ) >> 8).astype(np.uint8)

    # ── affine transforms ─────────────────────────────────────────

    @staticmethod
    def _forward_affine(x, y, theta, ego_cx, ego_cy, ego_px_size):
        """2×3 affine: ego pixel → global pixel (forward mapping)."""
        ct = np.cos(theta)
        st = np.sin(theta)
        s = ego_px_size / PX_SIZE

        return np.float64([
            [ s * ct,  s * st,
              ORIGIN_X + x / PX_SIZE - ego_cx * s * ct - ego_cy * s * st],
            [-s * st,  s * ct,
              ORIGIN_Y - y / PX_SIZE + ego_cx * s * st - ego_cy * s * ct],
        ])

    @staticmethod
    def _inverse_affine(x, y, theta, ego_cx, ego_cy, ego_px_size):
        """2×3 affine: global pixel → ego pixel (src→dst for warpAffine)."""
        ct = np.cos(theta)
        st = np.sin(theta)
        r = PX_SIZE / ego_px_size

        t_gx = ORIGIN_X + x / PX_SIZE
        t_gy = ORIGIN_Y - y / PX_SIZE

        return np.float64([
            [ r * ct, -r * st,  ego_cx - r * ct * t_gx + r * st * t_gy],
            [ r * st,  r * ct,  ego_cy - r * st * t_gx - r * ct * t_gy],
        ])

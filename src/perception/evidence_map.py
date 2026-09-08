"""World-frame accumulated drivable evidence with dynamic decay (CONTRACT step 3).

Holds clear_evidence / obstacle_evidence / last_seen on the same geometry as
GlobalMap (MAP_W×MAP_H @ PX_SIZE, origin at ORIGIN_X/Y). Ego labels (320×240
@ EGO_PX_SIZE, axle at RCX/RCY) are sparsely splatted via GlobalMap._forward_affine.

Rules (docs/perception/CONTRACT.md):
  - CLEAR boosts clear_evidence
  - OBSTACLE boosts obstacle_evidence (height-weighted); SELF/UNKNOWN never do
  - SELF must never increase obstacle_evidence
  - Obstacle evidence decays when not re-observed; clear decays slower and can reclaim

Expected cost on Orin CPU (sparse splat of labeled ego cells, no GPU):
  ~2–8 ms for a typical 320×240 ego frame with a few thousand CLEAR/OBSTACLE
  hits into the 960×720 world grid. Full-array decay is one float mul (~0.5 ms).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from globalmap import (
    GlobalMap, MAP_W, MAP_H, PX_SIZE,
)
from robot_config import EGO_PX_SIZE, RCX, RCY

from .labels import CLEAR, OBSTACLE

# Evidence boosts / decay (tuned for ~10 Hz ego updates)
CLEAR_BOOST = 1.0
OBS_BOOST_BASE = 1.0
OBS_BOOST_PER_CM = 0.02          # taller → stronger obstacle evidence
OBS_DECAY = 0.92                 # per update when not refreshed (~half-life ~8 frames)
CLEAR_DECAY = 0.995              # slow; floor memory lasts longer than movers
OBS_EVIDENCE_THRESH = 0.5
CLEAR_EVIDENCE_THRESH = 0.5
MAX_EVIDENCE = 20.0


class EvidenceMap:
    """Accumulated world-frame clear/obstacle evidence with dynamic decay."""

    def __init__(
        self,
        map_w: int = MAP_W,
        map_h: int = MAP_H,
        px_size: float = PX_SIZE,
        ego_cx: float = float(RCX),
        ego_cy: float = float(RCY),
        ego_px_size: float = float(EGO_PX_SIZE),
        obs_decay: float = OBS_DECAY,
        clear_decay: float = CLEAR_DECAY,
    ):
        self.map_w = int(map_w)
        self.map_h = int(map_h)
        self.px_size = float(px_size)
        self.ego_cx = float(ego_cx)
        self.ego_cy = float(ego_cy)
        self.ego_px_size = float(ego_px_size)
        self.obs_decay = float(obs_decay)
        self.clear_decay = float(clear_decay)

        self.clear_evidence = np.zeros((self.map_h, self.map_w), dtype=np.float32)
        self.obstacle_evidence = np.zeros((self.map_h, self.map_w), dtype=np.float32)
        self.last_seen = np.zeros((self.map_h, self.map_w), dtype=np.int32)
        self._frame_i = 0
        self._last_update_ms = 0.0
        # Scratch for refreshed-cell mask (reused; avoid per-frame alloc of full grid
        # by only marking sparse indices — see update()).
        self._touch_buf: Optional[np.ndarray] = None

    @property
    def frame_i(self) -> int:
        return self._frame_i

    @property
    def last_update_ms(self) -> float:
        return self._last_update_ms

    def reset(self) -> None:
        self.clear_evidence.fill(0)
        self.obstacle_evidence.fill(0)
        self.last_seen.fill(0)
        self._frame_i = 0

    def decay(self, *, touch_mask: Optional[np.ndarray] = None) -> None:
        """Decay evidence. If touch_mask given, skip cells refreshed this frame."""
        if touch_mask is None:
            self.obstacle_evidence *= self.obs_decay
            self.clear_evidence *= self.clear_decay
            return
        # Decay only untouched cells (vectorized)
        unt = ~touch_mask
        self.obstacle_evidence[unt] *= self.obs_decay
        self.clear_evidence[unt] *= self.clear_decay

    def update(
        self,
        ego_labels: np.ndarray,
        height_cm: np.ndarray,
        pose_xy_theta: Tuple[float, float, float],
        *,
        frame_i: Optional[int] = None,
    ) -> dict:
        """Warp ego labels into the world evidence grids.

        Parameters
        ----------
        ego_labels : (H,W) uint8 — UNKNOWN|SELF|CLEAR|OBSTACLE
        height_cm : (H,W) uint8 — obstacle height above floor (cm)
        pose_xy_theta : (x_m, y_m, theta_rad) robot pose in world frame
        frame_i : optional frame index; defaults to internal counter

        Returns
        -------
        metrics dict with n_clear, n_obs, update_ms, frame_i
        """
        import time
        t0 = time.monotonic()

        if frame_i is None:
            self._frame_i += 1
            fi = self._frame_i
        else:
            fi = int(frame_i)
            self._frame_i = fi

        x, y, theta = float(pose_xy_theta[0]), float(pose_xy_theta[1]), float(pose_xy_theta[2])
        labels = np.asarray(ego_labels)
        height = np.asarray(height_cm)
        if labels.shape != height.shape:
            raise ValueError("ego_labels and height_cm shape mismatch")

        # Sparse splat: only CLEAR / OBSTACLE (SELF and UNKNOWN contribute nothing)
        clear_yx = np.nonzero(labels == CLEAR)
        obs_yx = np.nonzero(labels == OBSTACLE)
        n_clear = int(clear_yx[0].size)
        n_obs = int(obs_yx[0].size)

        touch_rows: list = []
        touch_cols: list = []

        if n_clear or n_obs:
            M = GlobalMap._forward_affine(
                x, y, theta, self.ego_cx, self.ego_cy, self.ego_px_size)

            def _to_world(rows, cols):
                # M maps [col, row, 1] → [gx, gy]
                ones = np.ones(rows.size, dtype=np.float64)
                pts = np.vstack([cols.astype(np.float64), rows.astype(np.float64), ones])
                g = M @ pts
                gx = np.rint(g[0]).astype(np.int32)
                gy = np.rint(g[1]).astype(np.int32)
                ok = (gx >= 0) & (gx < self.map_w) & (gy >= 0) & (gy < self.map_h)
                return gy[ok], gx[ok], ok

            if n_clear:
                rows = clear_yx[0].astype(np.int32)
                cols = clear_yx[1].astype(np.int32)
                gy, gx, _ = _to_world(rows, cols)
                if gy.size:
                    np.add.at(self.clear_evidence, (gy, gx), CLEAR_BOOST)
                    np.minimum(self.clear_evidence, MAX_EVIDENCE, out=self.clear_evidence)
                    # Clear observation reclaim: pull obstacle evidence down so
                    # movers/chairs can free the cell without waiting for full decay.
                    self.obstacle_evidence[gy, gx] *= self.obs_decay
                    self.last_seen[gy, gx] = fi
                    touch_rows.append(gy)
                    touch_cols.append(gx)

            if n_obs:
                rows = obs_yx[0].astype(np.int32)
                cols = obs_yx[1].astype(np.int32)
                hvals = height[rows, cols].astype(np.float32)
                gy, gx, ok = _to_world(rows, cols)
                if gy.size:
                    boost = OBS_BOOST_BASE + OBS_BOOST_PER_CM * hvals[ok]
                    np.add.at(self.obstacle_evidence, (gy, gx), boost)
                    np.minimum(self.obstacle_evidence, MAX_EVIDENCE, out=self.obstacle_evidence)
                    self.last_seen[gy, gx] = fi
                    touch_rows.append(gy)
                    touch_cols.append(gx)

        # Decay cells not refreshed this frame (full-grid mul is cheap vs building mask)
        if touch_rows:
            tr = np.concatenate(touch_rows)
            tc = np.concatenate(touch_cols)
            # Mark touch on a bool scratch, decay elsewhere, then clear scratch
            if self._touch_buf is None or self._touch_buf.shape != self.clear_evidence.shape:
                self._touch_buf = np.zeros_like(self.clear_evidence, dtype=bool)
            else:
                self._touch_buf.fill(False)
            self._touch_buf[tr, tc] = True
            self.decay(touch_mask=self._touch_buf)
        else:
            self.decay()

        self._last_update_ms = (time.monotonic() - t0) * 1000.0
        return {
            "n_clear": n_clear,
            "n_obs": n_obs,
            "update_ms": self._last_update_ms,
            "frame_i": fi,
        }

    def drivable_mask(self) -> np.ndarray:
        """True where clear evidence dominates and obstacle has decayed away."""
        return (
            (self.clear_evidence >= CLEAR_EVIDENCE_THRESH)
            & (self.obstacle_evidence < OBS_EVIDENCE_THRESH)
        )

    def obstacle_mask(self) -> np.ndarray:
        """True where obstacle evidence is above threshold."""
        return self.obstacle_evidence >= OBS_EVIDENCE_THRESH

    def unknown_mask(self) -> np.ndarray:
        """True where neither clear nor obstacle evidence is trusted."""
        return (
            (self.clear_evidence < CLEAR_EVIDENCE_THRESH)
            & (self.obstacle_evidence < OBS_EVIDENCE_THRESH)
        )

    def to_obs_known(
        self,
        obs_out: Optional[np.ndarray] = None,
        known_out: Optional[np.ndarray] = None,
    ):
        """Honest dual (obs, known) for consumers.

        known=255 where clear or obstacle evidence exists; obs>0 only for
        obstacle_mask. Unknown (no evidence) → known=0, obs=0.
        """
        shape = self.clear_evidence.shape
        known = known_out if known_out is not None else np.zeros(shape, dtype=np.uint8)
        obs = obs_out if obs_out is not None else np.zeros(shape, dtype=np.uint8)
        known.fill(0)
        obs.fill(0)
        drv = self.drivable_mask()
        obst = self.obstacle_mask()
        known[drv | obst] = 255
        # Encode a soft height proxy from evidence magnitude (1–100)
        if np.any(obst):
            h = np.clip(
                (self.obstacle_evidence[obst] / OBS_BOOST_BASE).astype(np.int32),
                1, 100,
            ).astype(np.uint8)
            obs[obst] = h
        return obs, known

    def counts(self) -> dict:
        """Cell counts for logging: clear / obs / unk (mutually exclusive)."""
        obst = self.obstacle_mask()
        drv = self.drivable_mask()
        # Prefer obstacle when both (shouldn't happen often after reclaim rules)
        n_obs = int(np.count_nonzero(obst))
        n_clr = int(np.count_nonzero(drv & ~obst))
        n_unk = int(self.map_h * self.map_w - n_obs - n_clr)
        return {"clear": n_clr, "obs": n_obs, "unk": n_unk}

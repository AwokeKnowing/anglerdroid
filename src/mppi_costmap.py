"""mppi_costmap.py – LIVE NumPy MPPI local planner (mppi-costmap-v0).

Runs on Orin CPU with vectorized NumPy (no torch). Consumes ego-space
vis._persistent_obs (240x320) with robot_config RCX/RCY/EGO_PX_SIZE.
Wired behind LocalExecutive mailbox API (planner=mppi).

Run unit tests: python3 -m mppi_costmap
"""
from __future__ import annotations

import math
import time
import numpy as np
from robot_config import RCX, RCY, EGO_PX_SIZE, FRAME_W, FRAME_H

OBS_THRESH = 100
WANDER_REFRESH_S = 0.5


class MppiCostmapPlanner:
    """
    LIVE NumPy MPPI local planner for autonomous navigation (non-ROS).

    Accepts async map-frame (x,y) goals, computes velocity commands from
    ego-space 2D costmap (vis._persistent_obs). Non-blocking mailbox design.

    Ego frame: 320x240, robot at (RCX=81, RCY=119) facing RIGHT (+x image).
    World pose (x,y,theta): meters/radians, 0 yaw = +x world.
    """

    def __init__(self,
                 horizon_sec=0.8,
                 n_samples=160,
                 v_max=0.25,
                 w_max=0.8,
                 subgoal_dist=0.8,
                 min_clearance=0.20,
                 goal_tolerance=0.10,
                 mppi_temperature=0.5,
                 mppi_noise_sigma_v=0.08,
                 mppi_noise_sigma_w=0.35):
        self.horizon_sec = float(horizon_sec)
        self.n_samples = int(n_samples)
        self.dt = 0.033  # 30 Hz main loop
        self.n_steps = max(1, int(round(self.horizon_sec / self.dt)))

        self.v_max = float(v_max)
        self.w_max = float(w_max)
        self.subgoal_dist = float(subgoal_dist)
        self.min_clearance = float(min_clearance)
        self.goal_tolerance = float(goal_tolerance)

        self.temperature = float(mppi_temperature)
        self.noise_sigma_v = float(mppi_noise_sigma_v)
        self.noise_sigma_w = float(mppi_noise_sigma_w)

        # Cost weights
        self.w_obs = 80.0
        self.w_oob = 8.0
        self.w_goal = 12.0
        self.w_smooth = 0.02
        self.w_ctrl = 0.01

        self._goal = None          # (x, y) world meters
        self._wander_mode = False
        self._active = False
        self._last_wander_t = 0.0

        self._v_current = 0.0
        self._w_current = 0.0

        # Warm-start control sequence (n_steps, 2) = [v, w]
        self._u_prev = np.zeros((self.n_steps, 2), dtype=np.float32)

        self._debug = {
            'goal_x': 0.0,
            'goal_y': 0.0,
            'subgoal_x': 0.0,
            'subgoal_y': 0.0,
            'cmd_fwd': 0.0,
            'cmd_ang': 0.0,
            'best_cost': 0.0,
            'tick_ms': 0.0,
            'n_samples': self.n_samples,
            'n_steps': self.n_steps,
            'live': True,
        }

        print("MppiCostmapPlanner: LIVE NumPy MPPI (mppi-costmap-v0) — NOT stub")
        print("  MPPI params: horizon=%.2fs n_samples=%d n_steps=%d dt=%.3f" % (
            self.horizon_sec, self.n_samples, self.n_steps, self.dt))
        print("  Limits: v_max=%.2f m/s w_max=%.2f rad/s" % (self.v_max, self.w_max))
        print("  Temperature λ=%.2f, noise σ_v=%.2f σ_ω=%.2f" % (
            self.temperature, self.noise_sigma_v, self.noise_sigma_w))
        print("  Ego map: %dx%d RCX=%d RCY=%d EGO_PX_SIZE=%.3f m/px" % (
            FRAME_W, FRAME_H, RCX, RCY, EGO_PX_SIZE))

    def set_goal(self, x: float, y: float) -> None:
        self._goal = (float(x), float(y))
        self._wander_mode = False
        self._active = True
        print("MppiCostmapPlanner: goal set to (%.2f, %.2f) map frame" % (x, y))

    def set_wander_mode(self, enabled: bool) -> None:
        self._wander_mode = bool(enabled)
        if enabled:
            self._goal = None
            self._active = True
            self._last_wander_t = 0.0
            print("MppiCostmapPlanner: wander mode ENABLED")
        else:
            self._active = False
            print("MppiCostmapPlanner: wander mode DISABLED")

    def cancel(self) -> None:
        self._goal = None
        self._wander_mode = False
        self._active = False
        self._v_current = 0.0
        self._w_current = 0.0
        self._u_prev[:] = 0.0
        print("MppiCostmapPlanner: cancelled (inactive)")

    def is_active(self) -> bool:
        return self._active

    def get_debug_state(self) -> dict:
        return self._debug.copy()

    # ── Map helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _extract_obs_map(obs_map: np.ndarray) -> np.ndarray:
        """Return 2D uint8 ego obstacle map (H,W). Prefer already-cropped ego.

        Accepts:
          - (240,320) or any HxW 2D ego map → use as-is
          - (H,W,C) → max over channels
          - full atlas (H>=480, W>=640): NOT preferred for MPPI; if given,
            crop bottom-right 320x240 as last-resort fallback (VFH quadrant).
        """
        if obs_map is None:
            return np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)

        arr = np.asarray(obs_map)
        if arr.ndim == 3:
            # Prefer ego-sized maps; full atlas is RGB
            h, w = arr.shape[:2]
            if h >= 480 and w >= 640:
                # Last-resort: VFH atlas quadrant (should not be primary path)
                quad = arr[240:480, 320:640]
                arr = np.maximum(quad[:, :, 0], quad[:, :, 1])
            else:
                arr = np.max(arr, axis=2)
        elif arr.ndim != 2:
            return np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)

        # If somehow full-atlas grayscale, crop quadrant
        h, w = arr.shape[:2]
        if h >= 480 and w >= 640:
            arr = arr[240:480, 320:640]

        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    # ── Frame transforms ──────────────────────────────────────────────────

    @staticmethod
    def _world_to_ego(dx_world, dy_world, theta):
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        dx_ego = dx_world * cos_t + dy_world * sin_t
        dy_ego = -dx_world * sin_t + dy_world * cos_t
        return dx_ego, dy_ego

    @staticmethod
    def _ego_to_world(dx_ego, dy_ego, theta):
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        dx_world = dx_ego * cos_t - dy_ego * sin_t
        dy_world = dx_ego * sin_t + dy_ego * cos_t
        return dx_world, dy_world

    def _compute_subgoal(self, goal_world, pose):
        x, y, _theta = pose
        dx = goal_world[0] - x
        dy = goal_world[1] - y
        dist = math.hypot(dx, dy)
        if dist < self.subgoal_dist:
            return (goal_world[0], goal_world[1])
        scale = self.subgoal_dist / dist
        return (x + dx * scale, y + dy * scale)

    def _maybe_refresh_wander_goal(self, pose):
        """Periodically place a virtual goal ~subgoal_dist forward in world."""
        if not self._wander_mode:
            return
        now = time.monotonic()
        if self._goal is not None and (now - self._last_wander_t) < WANDER_REFRESH_S:
            return
        x, y, theta = pose
        gx = x + self.subgoal_dist * math.cos(theta)
        gy = y + self.subgoal_dist * math.sin(theta)
        self._goal = (gx, gy)
        self._last_wander_t = now

    # ── Vectorized MPPI core ──────────────────────────────────────────────

    def _mppi_sample_trajectories(self, u_prev, n_samples, n_steps):
        """Vectorized sample U ~ N(u_nominal, σ), clipped to limits."""
        u_prev = np.asarray(u_prev, dtype=np.float32)
        if u_prev.shape != (n_steps, 2):
            # Resize / pad warm-start if horizon changed
            new_u = np.zeros((n_steps, 2), dtype=np.float32)
            m = min(n_steps, u_prev.shape[0])
            new_u[:m] = u_prev[:m]
            u_prev = new_u

        noise = np.empty((n_samples, n_steps, 2), dtype=np.float32)
        noise[:, :, 0] = np.random.normal(
            0.0, self.noise_sigma_v, size=(n_samples, n_steps)).astype(np.float32)
        noise[:, :, 1] = np.random.normal(
            0.0, self.noise_sigma_w, size=(n_samples, n_steps)).astype(np.float32)

        U = u_prev[None, :, :] + noise
        # Small reverse ok if stuck; clip v to [-0.2*v_max, v_max]
        U[:, :, 0] = np.clip(U[:, :, 0], -self.v_max * 0.2, self.v_max)
        U[:, :, 1] = np.clip(U[:, :, 1], -self.w_max, self.w_max)
        return U

    def _mppi_rollout_batch(self, U, x0, obs_map, subgoal):
        """Fully vectorized unicycle rollout + cost on ego costmap.

        Integration is in world frame from x0; obstacle lookup converts each
        world state into ego pixels relative to x0 (robot at RCX,RCY facing +x).
        """
        n_samples, n_steps, _ = U.shape
        dt = self.dt
        v = U[:, :, 0]  # (N, T)
        w = U[:, :, 1]

        # Batched unicycle: start at x0 for all samples
        x = np.full(n_samples, x0[0], dtype=np.float32)
        y = np.full(n_samples, x0[1], dtype=np.float32)
        th = np.full(n_samples, x0[2], dtype=np.float32)

        X = np.empty((n_samples, n_steps, 3), dtype=np.float32)
        S = np.zeros(n_samples, dtype=np.float32)

        h, w_map = obs_map.shape[:2]
        # Flat view for fast gather
        obs_flat = obs_map.ravel()

        x0x, x0y, x0th = float(x0[0]), float(x0[1]), float(x0[2])
        cos0 = math.cos(x0th)
        sin0 = math.sin(x0th)
        inv_px = 1.0 / EGO_PX_SIZE

        sgx, sgy = float(subgoal[0]), float(subgoal[1])

        prev_v = np.zeros(n_samples, dtype=np.float32)
        prev_w = np.zeros(n_samples, dtype=np.float32)

        for k in range(n_steps):
            vk = v[:, k]
            wk = w[:, k]
            x = x + vk * np.cos(th) * dt
            y = y + vk * np.sin(th) * dt
            th = th + wk * dt
            # Keep theta bounded (cheap wrap)
            th = (th + np.pi) % (2.0 * np.pi) - np.pi

            X[:, k, 0] = x
            X[:, k, 1] = y
            X[:, k, 2] = th

            # World → ego relative to pose x0 (robot-centric map)
            dx_w = x - x0x
            dy_w = y - x0y
            dx_ego = dx_w * cos0 + dy_w * sin0
            dy_ego = -dx_w * sin0 + dy_w * cos0

            # Ego pixel: col = RCX + dx/px, row = RCY - dy/px (y up in ego meters)
            cols = (RCX + dx_ego * inv_px).astype(np.int32)
            rows = (RCY - dy_ego * inv_px).astype(np.int32)

            inb = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w_map)
            idx = rows * w_map + cols
            # Safe gather: clamp OOB indices then mask
            idx_safe = np.clip(idx, 0, h * w_map - 1)
            vals = obs_flat[idx_safe].astype(np.float32)

            obs_cost = np.zeros(n_samples, dtype=np.float32)
            hit = inb & (vals > OBS_THRESH)
            soft = inb & (~hit) & (vals > 50)
            obs_cost[hit] = self.w_obs
            obs_cost[soft] = (vals[soft] / 100.0) * (self.w_obs * 0.15)
            obs_cost[~inb] = self.w_oob

            # Smoothness / control effort
            dv = vk - prev_v
            dw = wk - prev_w
            smooth = self.w_smooth * (dv * dv + dw * dw) + self.w_ctrl * (vk * vk + wk * wk)

            S += obs_cost + smooth
            prev_v = vk
            prev_w = wk

        # Terminal goal distance
        dxg = sgx - X[:, -1, 0]
        dyg = sgy - X[:, -1, 1]
        S += self.w_goal * np.sqrt(dxg * dxg + dyg * dyg)

        return S, X

    def _mppi_compute_weights(self, S):
        S_norm = S - S.min()
        # Softmax with temperature; guard empty / inf
        lam = max(self.temperature, 1e-6)
        exp_S = np.exp(-S_norm / lam)
        z = exp_S.sum()
        if z < 1e-12 or not np.isfinite(z):
            w = np.ones_like(S) / float(len(S))
        else:
            w = exp_S / z
        return w

    def _mppi_weighted_control(self, U, w):
        u_star = np.sum(w[:, None] * U[:, 0, :], axis=0)
        return float(u_star[0]), float(u_star[1])

    def _shift_warm_start(self, U, w):
        """Warm-start: weighted mean sequence, shift by 1, pad last."""
        u_mean = np.sum(w[:, None, None] * U, axis=0)  # (T, 2)
        self._u_prev[:-1] = u_mean[1:]
        self._u_prev[-1] = u_mean[-1]

    # ── Main tick ─────────────────────────────────────────────────────────

    def tick(self, obs_map: np.ndarray, pose: tuple, dt: float):
        """Run LIVE MPPI when active. Returns {'fwd_mps','ang_rads'} or None."""
        if not self._active:
            return None

        t0 = time.perf_counter()
        x, y, theta = float(pose[0]), float(pose[1]), float(pose[2])
        pose_t = (x, y, theta)

        self._maybe_refresh_wander_goal(pose_t)

        if self._goal is None:
            return None

        dx = self._goal[0] - x
        dy = self._goal[1] - y
        dist = math.hypot(dx, dy)

        if (not self._wander_mode) and dist < self.goal_tolerance:
            print("MppiCostmapPlanner: goal reached (%.2f m), cancelling" % dist)
            self.cancel()
            return None

        subgoal = self._compute_subgoal(self._goal, pose_t)
        ego = self._extract_obs_map(obs_map)

        # If nearly stuck against obstacle ahead, bias reverse into warm-start
        ahead = ego[max(0, RCY - 8):min(ego.shape[0], RCY + 8),
                    max(0, RCX + 5):min(ego.shape[1], RCX + 35)]
        if ahead.size and ahead.max() > OBS_THRESH and abs(self._v_current) < 0.02:
            self._u_prev[:, 0] = -0.05

        U = self._mppi_sample_trajectories(self._u_prev, self.n_samples, self.n_steps)
        S, X = self._mppi_rollout_batch(U, pose_t, ego, subgoal)
        w = self._mppi_compute_weights(S)
        v_star, w_star = self._mppi_weighted_control(U, w)

        v_star = float(np.clip(v_star, -self.v_max * 0.2, self.v_max))
        w_star = float(np.clip(w_star, -self.w_max, self.w_max))

        self._shift_warm_start(U, w)
        self._v_current = v_star
        self._w_current = w_star

        tick_ms = (time.perf_counter() - t0) * 1000.0
        best_i = int(np.argmin(S))

        self._debug.update({
            'goal_x': self._goal[0],
            'goal_y': self._goal[1],
            'subgoal_x': subgoal[0],
            'subgoal_y': subgoal[1],
            'cmd_fwd': v_star,
            'cmd_ang': w_star,
            'best_cost': float(S[best_i]),
            'tick_ms': tick_ms,
            'n_samples': self.n_samples,
            'n_steps': self.n_steps,
            'live': True,
            'dist_goal': dist,
        })

        return {'fwd_mps': v_star, 'ang_rads': w_star}


# ── Unit test / standalone demo ───────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("MppiCostmapPlanner unit test — LIVE NumPy MPPI (not stub)")
    print("=" * 60)

    le = MppiCostmapPlanner(n_samples=160, horizon_sec=0.8)
    assert not le.is_active(), "Should start inactive"

    le.set_goal(2.0, 1.0)
    assert le.is_active(), "Should be active after set_goal"

    # Ego-space obs map (matches vis._persistent_obs)
    obs_map = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    # Sprinkle a soft obstacle band to the right of robot
    obs_map[RCY - 20:RCY + 20, RCX + 60:RCX + 80] = 150

    pose = (0.0, 0.0, 0.0)

    # Warm-up + latency samples
    latencies = []
    cmd = None
    for i in range(12):
        t0 = time.perf_counter()
        cmd = le.tick(obs_map, pose, 0.033)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        assert cmd is not None, "LIVE MPPI should return command"
        assert 'fwd_mps' in cmd and 'ang_rads' in cmd

    print("  LIVE MPPI cmd:", cmd)
    lat = np.array(latencies[2:])  # skip cold start
    print("  tick latency ms: mean=%.2f p95=%.2f max=%.2f (n=%d samples=%d steps=%d)" % (
        lat.mean(), np.percentile(lat, 95), lat.max(), len(lat), le.n_samples, le.n_steps))

    le.cancel()
    assert not le.is_active()

    # Wander should produce a command (virtual forward goal)
    le.set_wander_mode(True)
    assert le.is_active()
    cmd_w = le.tick(obs_map, pose, 0.033)
    assert cmd_w is not None, "Wander LIVE MPPI should return command"
    print("  Wander cmd:", cmd_w)
    le.cancel()

    dbg = le.get_debug_state()
    assert dbg.get('live') is True
    assert 'goal_x' in dbg

    # Sampling / weights shape checks
    print("\n  Testing vectorized MPPI primitives...")
    u_prev = np.zeros((le.n_steps, 2), dtype=np.float32)
    U = le._mppi_sample_trajectories(u_prev, n_samples=32, n_steps=le.n_steps)
    assert U.shape == (32, le.n_steps, 2)
    print("    ✓ _mppi_sample_trajectories (vectorized)")

    S, X = le._mppi_rollout_batch(U, (0.0, 0.0, 0.0), obs_map, (0.8, 0.0))
    assert S.shape == (32,) and X.shape == (32, le.n_steps, 3)
    print("    ✓ _mppi_rollout_batch (vectorized)")

    ww = le._mppi_compute_weights(S)
    assert abs(ww.sum() - 1.0) < 1e-5
    print("    ✓ _mppi_compute_weights")

    v_star, w_star = le._mppi_weighted_control(U, ww)
    print("    ✓ _mppi_weighted_control: v*=%.3f ω*=%.3f" % (v_star, w_star))

    # Also accept full-atlas shape via extract (fallback path)
    atlas = np.zeros((480, 640, 3), dtype=np.uint8)
    extracted = le._extract_obs_map(atlas)
    assert extracted.shape == (240, 320)

    print("\n✓ All unit tests passed — LIVE NumPy MPPI (mppi-costmap-v0)")
    print("  measured tick p95=%.2f ms" % float(np.percentile(lat, 95)))

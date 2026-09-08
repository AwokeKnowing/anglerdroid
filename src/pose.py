"""pose.py – Kalman-filtered 2D pose estimation.

Wheel odometry is the primary source, with visual odometry providing
corrections for systematic bias (carpet slip, uneven surfaces).

A configurable ANGULAR_SLIP_SCALE compensates for the fact that
tracked vehicles on soft surfaces (carpet) systematically under-report
rotation — typically 3–5% on medium-pile carpet.

Gating hierarchy (any failure → visual odom rejected, wheel odom used):
  1. Physical plausibility  — exceeds robot's max speed × dt?
  2. Confidence threshold   — ORB match quality too low?
  3. Stationary gate        — wheels stopped but vision claims motion?
  4. Agreement gate         — visual delta too far from wheel prediction?
  5. Kalman Mahalanobis     — fused innovation too large?

World frame: x=right, y=up, theta=0 facing +x, CCW positive.
Maintains a circular buffer of 900 world-space positions for trajectory.
"""

import math
import time
import cv2
import numpy as np

# ── History ────────────────────────────────────────────────────────
HISTORY_SIZE = 300

# ── Surface slip compensation ──────────────────────────────────────
# Tracked vehicles on carpet: treads slide on the pile and encoders
# over-report rotation (body turns less than treads move).
# <1.0 = reduce reported rotation.  Set 1.0 for hard floors.
# 15° over-report per revolution → 360/(360+15) ≈ 0.96.
ANGULAR_SLIP_SCALE = 0.92
LINEAR_SLIP_SCALE  = 1.0

# ── Robot physics (generous upper bounds) ──────────────────────────
MAX_SPEED_MPS = 1.0       # absolute max forward speed the robot can reach
MAX_OMEGA_RPS = 4.0       # absolute max angular velocity
PHYS_MARGIN   = 3.0       # multiplier on physics limit for gate

# ── Visual odometry gating ─────────────────────────────────────────
MIN_VIS_CONFIDENCE  = 0.10    # below this, discard visual entirely
STATIONARY_THRESH   = 0.01    # m/s and rad/s — wheels below this = stopped
AGREEMENT_FACTOR    = 8.0     # max ratio: |vis_delta| / |wheel_delta|
AGREEMENT_ABS_YAW   = 0.03    # rad — visual can disagree by at most this
AGREEMENT_ABS_FWD   = 0.01    # m   — even when wheels say zero

# ── IMU fusion ─────────────────────────────────────────────────
IMU_YAW_WEIGHT_BASE = 0.15    # base weight for IMU yaw rate (when visual ok)
IMU_YAW_WEIGHT_FALLBACK = 0.50  # weight when visual weak/lost
IMU_VIS_CONF_THRESH = 0.20    # below this, boost IMU weight

# ── Kalman noise ───────────────────────────────────────────────────
# Wheel odom process noise (1-sigma, scales with magnitude + floor).
# Higher Q_YAW_SCALE = trust wheels less for yaw → more visual correction.
Q_YAW_SCALE  = 0.10      # 10% of |dtheta| (was 5% — increased for carpet)
Q_FWD_SCALE  = 0.05      # 5% of |ds|
Q_YAW_FLOOR  = 0.0015    # rad
Q_FWD_FLOOR  = 0.0005    # m

# Visual odom measurement noise (base 1-sigma, scaled by 1/confidence).
# Lower R_YAW_BASE = trust visual more for yaw.
R_YAW_BASE = 0.002   # rad (was 0.003)
R_FWD_BASE = 0.002   # m

# Mahalanobis gate — chi-squared, 2 DOF, 95%
GATE_CHI2 = 5.991

# ── Minimap ────────────────────────────────────────────────────────
MINIMAP_SIZE = 50         # pixels
MINIMAP_SCALE = 0.10      # metres per pixel (10 cm)
MINIMAP_BG = (30, 30, 30)
MINIMAP_TRAIL = (80, 140, 255)    # blue
MINIMAP_ROBOT = (255, 200, 60)    # bright cyan-ish


class PoseEstimator:
    """Fused 2D pose with world-space history and minimap."""

    def __init__(self, wheelbase_m: float, wheel_radius_m: float):
        self._wb = wheelbase_m
        self._wr = wheel_radius_m

        self.x     = 0.0
        self.y     = 0.0
        self.theta = 0.0

        self._hx   = np.zeros(HISTORY_SIZE, dtype=np.float64)
        self._hy   = np.zeros(HISTORY_SIZE, dtype=np.float64)
        self._hlen = 0
        self._hidx = 0
        
        # Tracking loss / quality metrics
        self._visual_accepted = 0
        self._visual_rejected = 0
        self._wheel_only_frames = 0
        self._last_visual_time = 0.0
        self._excessive_disagreement_count = 0
        
        # IMU fusion tracking
        self._imu_used_count = 0
        self._imu_fallback_count = 0  # IMU with boosted weight (visual weak/rejected)
        
        # Stuck detection (CRITICAL SAFETY)
        self._commanded_forward_sum = 0.0  # Commanded displacement
        self._actual_forward_sum = 0.0     # Actual visual displacement
        self._stuck_check_start = 0.0      # When current window started
        self._stuck_count = 0              # Number of times stuck detected
        self._is_stuck = False             # Current stuck state
        self._last_stuck_log = 0.0         # Throttle stuck logging

    def reset(self):
        self.x = self.y = self.theta = 0.0
        self._hlen = self._hidx = 0
        self._visual_accepted = 0
        self._visual_rejected = 0
        self._wheel_only_frames = 0
        self._last_visual_time = 0.0
        self._excessive_disagreement_count = 0
        self._imu_used_count = 0
        self._imu_fallback_count = 0

    # ── main entry point ───────────────────────────────────────────

    def update(self, v_left_mps: float, v_right_mps: float,
               dt: float,
               vis_yaw: float, vis_fwd: float,
               vis_confidence: float = 0.0,
               using_encoder_feedback: bool = True,
               imu_yaw_rate: float = 0.0):
        """Fuse wheel + visual odom + IMU, integrate, record history.

        Args:
            v_left_mps, v_right_mps: Wheel velocities
            dt: Time delta
            vis_yaw, vis_fwd: Visual odometry
            vis_confidence: Visual confidence [0-1]
            using_encoder_feedback: True if wheel velocities from encoders, 
                                    False if from commanded velocity
            imu_yaw_rate: IMU gyro yaw rate (rad/s, body frame Z-axis)
        
        Returns (fused_yaw, fused_fwd) for map warping.
        """
        if dt <= 0:
            return 0.0, 0.0

        # ── 1. Wheel odometry with slip compensation ──
        v     = (v_left_mps + v_right_mps) * 0.5
        omega = (v_right_mps - v_left_mps) / self._wb
        dtheta_w = omega * dt * ANGULAR_SLIP_SCALE
        ds_w     = v * dt * LINEAR_SLIP_SCALE

        # ── 2. Gate visual odometry ──
        vis_ok, gate_reason = self._gate_visual(
            dtheta_w, ds_w, v, omega, dt,
            vis_yaw, vis_fwd, vis_confidence)

        # ── 3. Fuse or use wheel-only (IMU blended later) ──
        if vis_ok:
            r_scale = 1.0 / max(vis_confidence, 0.1)
            dtheta, ds = self._fuse(dtheta_w, ds_w,
                                    vis_yaw, vis_fwd, r_scale)
            self._visual_accepted += 1
            self._last_visual_time = time.time()
        else:
            # Visual rejected → wheel-only for now (IMU blended in step 3b below)
            dtheta, ds = dtheta_w, ds_w
            self._visual_rejected += 1
            self._wheel_only_frames += 1
            if gate_reason == 'agreement':
                self._excessive_disagreement_count += 1
        
        # ── 3b. Blend in IMU yaw rate (if available) ──
        # CRITICAL: IMU fusion happens REGARDLESS of visual acceptance.
        # When visual is weak/rejected, IMU weight is BOOSTED to compensate.
        # This ensures we use wheel+IMU fusion, NOT wheel-only.
        imu_used = False
        imu_weight_used = 0.0
        if abs(imu_yaw_rate) > 1e-6 and dt > 0:
            dtheta_imu = imu_yaw_rate * dt
            
            # Adaptive weight: boost IMU when visual confidence is low
            if vis_ok and vis_confidence >= IMU_VIS_CONF_THRESH:
                imu_weight = IMU_YAW_WEIGHT_BASE
            else:
                # Visual weak/rejected → BOOST IMU weight from 0.15 to 0.50
                imu_weight = IMU_YAW_WEIGHT_FALLBACK
                self._imu_fallback_count += 1
            
            # Complementary blend: dtheta_fused = (1-w)*dtheta + w*dtheta_imu
            dtheta = (1.0 - imu_weight) * dtheta + imu_weight * dtheta_imu
            imu_used = True
            imu_weight_used = imu_weight
            self._imu_used_count += 1

        # ── 4. Integrate into global pose ──
        self.theta += dtheta
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))
        self.x += ds * math.cos(self.theta)
        self.y += ds * math.sin(self.theta)

        # ── 5. Record to history ──
        self._hx[self._hidx] = self.x
        self._hy[self._hidx] = self.y
        self._hidx = (self._hidx + 1) % HISTORY_SIZE
        if self._hlen < HISTORY_SIZE:
            self._hlen += 1
        
        # ── 6. Stuck detection (CRITICAL SAFETY) ──
        # If using commanded velocity (not encoder feedback), we MUST check
        # if actual displacement matches commanded displacement.
        # Scenario: Robot stuck on bump, wheels spinning, commanded says moving
        # but visual odom shows zero displacement → STUCK
        self._update_stuck_detection(ds_w, vis_fwd, vis_ok, using_encoder_feedback, dt)

        return dtheta, ds

    # ── visual odometry gating ─────────────────────────────────────

    @staticmethod
    def _gate_visual(dtheta_w, ds_w, v, omega, dt,
                     vis_yaw, vis_fwd, vis_confidence):
        """Return (ok, reason) tuple. ok=True only if visual odom passes all gates.
        
        Reason codes: 'no_data', 'confidence', 'physics', 'stationary', 'agreement'
        """
        have_vis = abs(vis_yaw) > 1e-8 or abs(vis_fwd) > 1e-8
        if not have_vis:
            return False, 'no_data'

        # Gate 1: confidence too low
        if vis_confidence < MIN_VIS_CONFIDENCE:
            return False, 'confidence'

        # Gate 2: physical plausibility
        max_ds     = MAX_SPEED_MPS * dt * PHYS_MARGIN
        max_dtheta = MAX_OMEGA_RPS * dt * PHYS_MARGIN
        if abs(vis_fwd) > max_ds or abs(vis_yaw) > max_dtheta:
            return False, 'physics'

        # Gate 3: stationary — wheels say stopped, vision must agree
        wheel_still = (abs(v) < STATIONARY_THRESH and
                       abs(omega) < STATIONARY_THRESH)
        if wheel_still:
            if (abs(vis_fwd) > AGREEMENT_ABS_FWD or
                    abs(vis_yaw) > AGREEMENT_ABS_YAW):
                return False, 'stationary'

        # Gate 4: agreement — vision shouldn't wildly disagree with wheels
        if abs(dtheta_w) > 1e-6:
            if abs(vis_yaw) > AGREEMENT_FACTOR * abs(dtheta_w) + AGREEMENT_ABS_YAW:
                return False, 'agreement'
        else:
            if abs(vis_yaw) > AGREEMENT_ABS_YAW:
                return False, 'agreement'

        if abs(ds_w) > 1e-6:
            if abs(vis_fwd) > AGREEMENT_FACTOR * abs(ds_w) + AGREEMENT_ABS_FWD:
                return False, 'agreement'
        else:
            if abs(vis_fwd) > AGREEMENT_ABS_FWD:
                return False, 'agreement'

        return True, None

    # ── Kalman fusion ──────────────────────────────────────────────

    @staticmethod
    def _fuse(dtheta_w, ds_w, dtheta_v, ds_v, r_scale=1.0):
        """Per-frame Kalman: wheel = prior, visual = measurement.

        r_scale multiplies R (higher = trust visual less).
        """
        q0 = (Q_YAW_SCALE * abs(dtheta_w) + Q_YAW_FLOOR) ** 2
        q1 = (Q_FWD_SCALE * abs(ds_w)     + Q_FWD_FLOOR) ** 2

        r0 = (R_YAW_BASE * r_scale) ** 2
        r1 = (R_FWD_BASE * r_scale) ** 2

        y0 = dtheta_v - dtheta_w
        y1 = ds_v     - ds_w

        s0 = q0 + r0
        s1 = q1 + r1

        maha = (y0 * y0) / s0 + (y1 * y1) / s1
        if maha > GATE_CHI2:
            return dtheta_w, ds_w

        k0 = q0 / s0
        k1 = q1 / s1
        return dtheta_w + k0 * y0, ds_w + k1 * y1

    # ── Trajectory output ──────────────────────────────────────────

    def get_world_history(self):
        """(N, 2) float64 array of world [x, y], oldest → newest."""
        if self._hlen == 0:
            return np.empty((0, 2), dtype=np.float64)
        if self._hlen < HISTORY_SIZE:
            return np.column_stack([
                self._hx[:self._hlen], self._hy[:self._hlen]])
        idx = (np.arange(HISTORY_SIZE) + self._hidx) % HISTORY_SIZE
        return np.column_stack([self._hx[idx], self._hy[idx]])

    def world_to_ego_pixels(self, px_size: float, cx: float, cy: float):
        """Project world history to ego costmap pixels.

        Robot at (cx, cy) facing RIGHT.  px_size = metres/pixel.
        Returns (N, 2) int32 [col, row].
        """
        pts = self.get_world_history()
        if len(pts) == 0:
            return np.empty((0, 2), dtype=np.int32)

        dx = pts[:, 0] - self.x
        dy = pts[:, 1] - self.y
        ct = math.cos(self.theta)
        st = math.sin(self.theta)

        local_fwd  =  dx * ct + dy * st
        local_left = -dx * st + dy * ct

        col = cx + local_fwd  / px_size
        row = cy - local_left / px_size
        return np.column_stack([col, row]).astype(np.int32)

    # ── Minimap ────────────────────────────────────────────────────

    def render_minimap(self):
        """50×50 px global-coordinate minimap, robot = blue triangle.

        Centred on robot, 10 cm/px, north-up (initial heading = right).
        """
        sz  = MINIMAP_SIZE
        scl = MINIMAP_SCALE
        cx = cy = sz // 2

        img = np.full((sz, sz, 3), MINIMAP_BG, dtype=np.uint8)

        # --- trail ---
        pts = self.get_world_history()
        if len(pts) > 0:
            pcol = ((pts[:, 0] - self.x) / scl + cx).astype(np.int32)
            prow = (cy - (pts[:, 1] - self.y) / scl).astype(np.int32)
            mask = (pcol >= 0) & (pcol < sz) & (prow >= 0) & (prow < sz)
            img[prow[mask], pcol[mask]] = MINIMAP_TRAIL

        # --- robot triangle (heading arrow) ---
        fwd_x =  math.cos(self.theta)
        fwd_y = -math.sin(self.theta)   # pixel-y is inverted
        prp_x = -fwd_y
        prp_y =  fwd_x

        r = 3.5
        tip  = (int(cx + fwd_x * r),       int(cy + fwd_y * r))
        bl   = (int(cx - fwd_x * r * 0.6 + prp_x * r * 0.5),
                int(cy - fwd_y * r * 0.6 + prp_y * r * 0.5))
        br   = (int(cx - fwd_x * r * 0.6 - prp_x * r * 0.5),
                int(cy - fwd_y * r * 0.6 - prp_y * r * 0.5))
        tri  = np.array([tip, bl, br], dtype=np.int32)
        cv2.fillPoly(img, [tri], MINIMAP_ROBOT)

        # --- thin border ---
        cv2.rectangle(img, (0, 0), (sz - 1, sz - 1), (80, 80, 80), 1)

        return img
    
    # ── Tracking quality metrics ───────────────────────────────────
    
    def get_tracking_quality(self):
        """Return tracking quality metrics for diagnostics.
        
        Returns dict with:
          - visual_accept_rate: fraction of frames with accepted visual odom
          - wheel_only_rate: fraction of frames using wheel-only (before IMU blend)
          - time_since_visual: seconds since last visual correction
          - excessive_disagreement: count of agreement gate failures
          - imu_used_rate: fraction of frames where IMU contributed
          - imu_fallback_rate: fraction of frames where IMU weight was boosted
        """
        import time
        total = max(self._visual_accepted + self._visual_rejected, 1)
        imu_total = max(self._imu_used_count, 1)
        return {
            'visual_accept_rate': self._visual_accepted / total,
            'wheel_only_rate': self._wheel_only_frames / total,
            'time_since_visual': time.time() - self._last_visual_time if self._last_visual_time > 0 else float('inf'),
            'excessive_disagreement': self._excessive_disagreement_count,
            'visual_accepted': self._visual_accepted,
            'visual_rejected': self._visual_rejected,
            'imu_used_count': self._imu_used_count,
            'imu_fallback_count': self._imu_fallback_count,
            'imu_used_rate': self._imu_used_count / total,
            'imu_fallback_rate': self._imu_fallback_count / imu_total if self._imu_used_count > 0 else 0.0,
        }
    
    # ── Stuck detection (CRITICAL SAFETY) ──────────────────────────
    
    STUCK_WINDOW_S = 3.0          # Detection window (3 seconds)
    STUCK_CMD_THRESHOLD = 0.15    # Min commanded displacement to check (15cm)
    STUCK_RATIO_THRESHOLD = 0.3   # Max ratio: actual/commanded to be stuck
    
    def _update_stuck_detection(self, ds_commanded, ds_visual, visual_ok, 
                                using_encoder_feedback, dt):
        """Detect if robot is stuck (wheels spinning but not moving).
        
        CRITICAL SAFETY: When encoders fail and we fall back to commanded
        velocity, we MUST detect if actual displacement (from visual) is
        much less than commanded displacement.
        
        Scenario: Robot stuck on bump
        - Commanded: "move forward 0.5m"
        - Actual visual: displacement ~0.0m
        - Detection: STUCK after 3s of this mismatch
        
        Args:
            ds_commanded: Commanded forward displacement (from wheels/commanded vel)
            ds_visual: Visual forward displacement (from GPU odom)
            visual_ok: True if visual odom was accepted
            using_encoder_feedback: False when using commanded velocity fallback
            dt: Time delta
        """
        now = time.time()
        
        # Reset window if starting
        if self._stuck_check_start == 0.0:
            self._stuck_check_start = now
        
        # Accumulate commanded and actual displacement
        self._commanded_forward_sum += abs(ds_commanded)
        if visual_ok:
            self._actual_forward_sum += abs(ds_visual)
        
        # Check window duration
        window_duration = now - self._stuck_check_start
        
        if window_duration >= self.STUCK_WINDOW_S:
            # Time to evaluate
            commanded_total = self._commanded_forward_sum
            actual_total = self._actual_forward_sum
            
            # Only check if we commanded significant motion
            if commanded_total >= self.STUCK_CMD_THRESHOLD:
                ratio = actual_total / max(commanded_total, 1e-6)
                
                was_stuck = self._is_stuck
                
                if ratio < self.STUCK_RATIO_THRESHOLD:
                    # STUCK: actual displacement much less than commanded
                    if not self._is_stuck:
                        self._stuck_count += 1
                        self._is_stuck = True
                        print(f"🚨 STUCK DETECTED (count={self._stuck_count})")
                        print(f"   Commanded: {commanded_total:.3f}m, "
                              f"Actual: {actual_total:.3f}m, "
                              f"Ratio: {ratio:.2f}")
                        print(f"   Using encoders: {using_encoder_feedback}, "
                              f"Visual OK: {visual_ok}")
                        print(f"   ⚠️  Wheels spinning but no forward motion!")
                        self._last_stuck_log = now
                    elif now - self._last_stuck_log >= 2.0:
                        # Still stuck, periodic reminder
                        print(f"🚨 STILL STUCK (duration: {window_duration:.1f}s)")
                        self._last_stuck_log = now
                else:
                    # Not stuck
                    if self._is_stuck:
                        print(f"✓ UNSTUCK (was stuck {window_duration:.1f}s)")
                    self._is_stuck = False
            else:
                # Not commanding significant motion, reset stuck
                self._is_stuck = False
            
            # Reset window
            self._stuck_check_start = now
            self._commanded_forward_sum = 0.0
            self._actual_forward_sum = 0.0
    
    @property
    def is_stuck(self):
        """True if robot is currently detected as stuck (wheels spinning, no motion)."""
        return self._is_stuck
    
    @property
    def stuck_count(self):
        """Number of times stuck has been detected this session."""
        return self._stuck_count

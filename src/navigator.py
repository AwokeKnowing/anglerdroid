"""
navigator.py – Reactive obstacle-avoidance local planner.

Reads the obstacle map from the vision atlas and outputs twist commands.
Pure numpy — runs in ~1ms at 30fps in the main loop. No model, no network.

Gemini (high-level planner) sets a goal heading via set_goal().
This module steers toward that heading while avoiding obstacles.

The obstacle map is the bottom-right 320x240 quadrant of the 640x480 atlas.
  R channel = forward-camera obstacles
  G channel = top-down-camera obstacles
  B channel = unknown area (dim)
Robot faces RIGHT in the obstacle map. Crosshair marks robot center.

Algorithm: Vector Field Histogram (VFH-lite).
  1. Extract obstacle pixels → polar coordinates (angle, distance) from robot center
  2. Build angular histogram of obstacle density, weighted by proximity
  3. Find the clearest direction near the goal heading
  4. Set forward speed inversely proportional to nearest obstacle ahead
  5. Set turn rate to steer toward chosen direction
"""

import math
import numpy as np
import cv2

# Robot center in the 320x240 obstacle map quadrant
ROBOT_CX = 159
ROBOT_CY = 119

# Obstacle map is at atlas[240:480, 320:640]
QUAD_Y0, QUAD_X0 = 240, 320
QUAD_H, QUAD_W = 240, 320

# Scan parameters (in obstacle-map pixels)
MIN_RANGE = 8       # ignore closer than this (robot body / noise)
MAX_RANGE = 130     # max detection range
DANGER_RANGE = 40   # slow down when nearest obstacle < this
STOP_RANGE = 18     # full stop when nearest obstacle < this

# Angular histogram
N_BINS = 36         # 10° per bin
BIN_DEG = 360.0 / N_BINS

# Speed limits
MAX_FWD = 0.20      # m/s
MIN_FWD = 0.06      # below ~0.05 doesn't move on carpet
MAX_ANG = 0.6       # rad/s
TURN_GAIN = 0.015   # rad/s per degree of heading error

# Obstacle threshold (pixel intensity)
OBS_THRESH = 100

# State
_goal_heading = None   # type: float or None (degrees: 0=fwd, 90=left, -90=right)
_active = False

# Oscillation detection
_ang_history = []      # last N angular outputs
_OSCILLATION_WINDOW = 20  # frames to check (~0.67s at 30fps)
_OSCILLATION_THRESH = 12  # sign changes in window → oscillating

# Last computed debug state (for overlay drawing)
_dbg = None


def set_goal(heading_deg):
    # type: (float) -> None
    """Set goal heading in degrees. 0=forward, 90=left, -90=right. None to stop."""
    global _goal_heading, _active
    if heading_deg is None:
        _goal_heading = None
        _active = False
    else:
        _goal_heading = float(heading_deg)
        _active = True


def clear_goal():
    global _goal_heading, _active, _dbg, _ang_history
    _goal_heading = None
    _active = False
    _dbg = None
    _ang_history = []


def is_active():
    return _active


def get_goal():
    return _goal_heading


def _angle_to_image(angle_deg, radius):
    """Convert nav angle (0=right, +CCW) to image-space (x, y) offset from robot center."""
    rad = math.radians(angle_deg)
    return (int(radius * math.cos(rad)), int(-radius * math.sin(rad)))


def _detect_oscillation(angular):
    """Track angular sign changes. If too many in a short window, we're oscillating."""
    global _ang_history
    _ang_history.append(1 if angular > 0.01 else (-1 if angular < -0.01 else 0))
    if len(_ang_history) > _OSCILLATION_WINDOW:
        _ang_history = _ang_history[-_OSCILLATION_WINDOW:]
    if len(_ang_history) < _OSCILLATION_WINDOW:
        return False
    changes = sum(1 for i in range(1, len(_ang_history))
                  if _ang_history[i] != 0 and _ang_history[i - 1] != 0
                  and _ang_history[i] != _ang_history[i - 1])
    return changes >= _OSCILLATION_THRESH


def compute_twist(atlas):
    # type: (np.ndarray) -> tuple
    """Compute (forward_mps, angular_rads) from obstacle map. Returns None if inactive."""
    global _dbg

    if not _active or _goal_heading is None:
        _dbg = None
        return None

    h, w = atlas.shape[:2]
    if h < 480 or w < 640:
        return None

    goal = _goal_heading

    # For large heading angles (>90°), do a pure in-place rotation.
    # The obstacle map only covers the forward hemisphere, so VFH can't help here.
    # Just spin toward the goal heading, then normal VFH takes over once we're roughly aimed.
    if abs(goal) > 90.0:
        turn_dir = MAX_ANG if goal > 0 else -MAX_ANG
        _dbg = {
            "goal": goal, "best_angle": goal, "best_bin": 0,
            "forward": 0.0, "angular": turn_dir, "nearest_fwd": float(MAX_RANGE),
            "min_dist": np.full(N_BINS, float(MAX_RANGE), dtype=np.float32),
            "scores": np.zeros(N_BINS, dtype=np.float32),
            "mode": "spin",
        }
        return (0.0, turn_dir)

    quad = atlas[QUAD_Y0:QUAD_Y0 + QUAD_H, QUAD_X0:QUAD_X0 + QUAD_W]
    obs = np.maximum(quad[:, :, 0], quad[:, :, 1])

    ys, xs = np.nonzero(obs > OBS_THRESH)

    min_dist = np.full(N_BINS, float(MAX_RANGE), dtype=np.float32)
    scores = np.zeros(N_BINS, dtype=np.float32)

    if len(xs) > 0:
        dx = xs.astype(np.float32) - ROBOT_CX
        dy = ys.astype(np.float32) - ROBOT_CY
        dist = np.sqrt(dx * dx + dy * dy)
        angle = np.degrees(np.arctan2(-dy, dx))

        in_range = (dist >= MIN_RANGE) & (dist <= MAX_RANGE)
        dist = dist[in_range]
        angle = angle[in_range]

        if len(dist) > 0:
            bin_idx = np.floor((angle + 180.0) / BIN_DEG).astype(np.int32)
            bin_idx = np.clip(bin_idx, 0, N_BINS - 1)

            for b in range(N_BINS):
                mask = bin_idx == b
                if np.any(mask):
                    bd = dist[mask]
                    min_dist[b] = bd.min()

    for b in range(N_BINS):
        b_angle = b * BIN_DEG - 180.0
        diff = ((b_angle - goal + 180.0) % 360.0) - 180.0
        heading_cost = abs(diff) / 180.0

        if min_dist[b] < STOP_RANGE:
            clearance = 0.0
        elif min_dist[b] < DANGER_RANGE:
            clearance = (min_dist[b] - STOP_RANGE) / (DANGER_RANGE - STOP_RANGE)
        else:
            clearance = 1.0

        if abs(b_angle) > 120.0:
            scores[b] = -10.0
            continue

        scores[b] = clearance * 0.6 - heading_cost * 0.4

    best_bin = int(np.argmax(scores))
    best_angle = best_bin * BIN_DEG - 180.0

    fwd_bins = [b for b in range(N_BINS) if abs(b * BIN_DEG - 180.0) <= 30.0]
    nearest_fwd = min((min_dist[b] for b in fwd_bins), default=float(MAX_RANGE))

    if nearest_fwd < STOP_RANGE:
        forward = 0.0
    elif nearest_fwd < DANGER_RANGE:
        frac = (nearest_fwd - STOP_RANGE) / (DANGER_RANGE - STOP_RANGE)
        forward = MIN_FWD + frac * (MAX_FWD - MIN_FWD)
    else:
        forward = MAX_FWD

    angular = best_angle * TURN_GAIN
    angular = max(-MAX_ANG, min(MAX_ANG, angular))

    if scores[best_bin] < 0.0:
        forward = 0.0
        angular = 0.0

    # Auto-stop on oscillation (poorly tuned heading → jitter)
    if _detect_oscillation(angular):
        print("nav: oscillation detected — auto-stopping")
        clear_goal()
        return None

    _dbg = {
        "goal": goal,
        "best_angle": best_angle,
        "best_bin": best_bin,
        "forward": forward,
        "angular": angular,
        "nearest_fwd": nearest_fwd,
        "min_dist": min_dist.copy(),
        "scores": scores.copy(),
        "mode": "vfh",
    }

    return (forward, angular)


def draw_overlay(display):
    """Draw VFH debug overlay onto the display image (2x-scaled atlas, so 1280x960).
    Call after compute_twist(), before cv2.imshow()."""
    if _dbg is None:
        return

    # The obstacle-map quadrant in the 2x display starts at (640, 480) and is 640x480
    ox, oy = QUAD_X0 * 2, QUAD_Y0 * 2  # top-left of quadrant in display coords
    cx, cy = ox + ROBOT_CX * 2, oy + ROBOT_CY * 2  # robot center in display coords

    min_dist = _dbg["min_dist"]
    scores = _dbg["scores"]
    goal = _dbg["goal"]
    best_angle = _dbg["best_angle"]
    forward = _dbg["forward"]
    angular = _dbg["angular"]
    nearest_fwd = _dbg["nearest_fwd"]

    # Draw range rings
    cv2.circle(display, (cx, cy), STOP_RANGE * 2, (0, 0, 180), 1)
    cv2.circle(display, (cx, cy), DANGER_RANGE * 2, (0, 140, 180), 1)
    cv2.circle(display, (cx, cy), MAX_RANGE * 2, (80, 80, 80), 1)

    # Draw sector lines colored by score
    for b in range(N_BINS):
        b_angle = b * BIN_DEG - 180.0
        if abs(b_angle) > 120.0:
            continue

        sc = scores[b]
        d = min_dist[b]

        # Color: green = high score/clear, red = blocked, yellow = marginal
        if sc > 0.3:
            color = (0, 200, 0)
        elif sc > 0.0:
            color = (0, 200, 200)
        else:
            color = (0, 0, 200)

        # Line length proportional to min distance in that bin
        line_len = min(int(d * 2), MAX_RANGE * 2)
        dx_img, dy_img = _angle_to_image(b_angle, line_len)
        end = (cx + dx_img, cy + dy_img)
        cv2.line(display, (cx, cy), end, color, 1)

    # Draw goal heading (cyan, thick)
    gdx, gdy = _angle_to_image(goal, MAX_RANGE * 2)
    cv2.line(display, (cx, cy), (cx + gdx, cy + gdy), (255, 255, 0), 2)

    # Draw chosen heading (bright green, thick arrow)
    bdx, bdy = _angle_to_image(best_angle, int(nearest_fwd * 2))
    end_pt = (cx + bdx, cy + bdy)
    cv2.arrowedLine(display, (cx, cy), end_pt, (0, 255, 0), 2, tipLength=0.3)

    # Text overlay: speed, turn, heading info
    txt_x = ox + 4
    txt_y = oy + 16
    mode = _dbg.get("mode", "vfh")
    cv2.putText(display, "NAV[%s]: goal=%d best=%d" % (mode, int(goal), int(best_angle)),
                (txt_x, txt_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(display, "fwd=%.2f ang=%.2f near=%dpx" % (forward, angular, int(nearest_fwd)),
                (txt_x, txt_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Speed bar (bottom of quadrant)
    bar_y = oy + QUAD_H * 2 - 12
    bar_w = int((forward / MAX_FWD) * 120)
    cv2.rectangle(display, (txt_x, bar_y), (txt_x + bar_w, bar_y + 8), (0, 255, 0), -1)
    cv2.rectangle(display, (txt_x, bar_y), (txt_x + 120, bar_y + 8), (100, 100, 100), 1)

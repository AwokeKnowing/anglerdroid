"""Top-down enjoy-mode visualization for the lightweight sim."""

from __future__ import annotations

import math
import numpy as np

from sim.robot import (
    EGO_PX_SIZE, FOOT_X0, FOOT_X1, FOOT_Y0, FOOT_Y1, OBS_THRESH, RCX, RCY,
)


def render_world_frame(world_obs, robot, step, safety_scales, trail, goal=None):
    """Birds-eye RGB frame: map + footprint + heading + safety bars + trail."""
    h, w = world_obs.shape
    # Upscale 2x for watchability
    scale = 2
    frame = np.zeros((h * scale, w * scale, 3), dtype=np.uint8)

    # Floor / walls
    occ = world_obs >= OBS_THRESH
    base = np.zeros((h, w, 3), dtype=np.uint8)
    base[:] = (32, 36, 42)
    base[occ] = (210, 210, 220)
    # Nearest-neighbor upscale
    frame = np.repeat(np.repeat(base, scale, axis=0), scale, axis=1)

    def wxwy_to_px(x_m, y_m):
        px = int(round(x_m / EGO_PX_SIZE)) * scale
        py = int(round(y_m / EGO_PX_SIZE)) * scale
        return px, py

    # Trail
    if trail:
        for i, (tx, ty) in enumerate(trail[-200:]):
            px, py = wxwy_to_px(tx, ty)
            if 0 <= px < frame.shape[1] and 0 <= py < frame.shape[0]:
                t = i / max(1, len(trail[-200:]) - 1)
                frame[py, px] = (40, int(80 + 140 * t), int(200 * t))

    # Goal
    if goal is not None:
        gx, gy = goal
        px, py = wxwy_to_px(gx, gy)
        _disk(frame, px, py, 5, (80, 220, 120))

    # Robot pose in world: footprint corners from ego FOOT_* rotated
    cos_t = math.cos(robot.theta)
    sin_t = math.sin(robot.theta)
    corners = []
    for ex, ey in (
        (FOOT_X0, FOOT_Y0), (FOOT_X1, FOOT_Y0),
        (FOOT_X1, FOOT_Y1), (FOOT_X0, FOOT_Y1),
    ):
        dx = (ex - RCX) * EGO_PX_SIZE
        dy = (ey - RCY) * EGO_PX_SIZE
        wx = robot.x + dx * cos_t - dy * sin_t
        wy = robot.y + dx * sin_t + dy * cos_t
        corners.append(wxwy_to_px(wx, wy))
    _poly(frame, corners, (50, 180, 255))

    # Heading arrow
    hx = robot.x + 0.25 * cos_t
    hy = robot.y + 0.25 * sin_t
    x0, y0 = wxwy_to_px(robot.x, robot.y)
    x1, y1 = wxwy_to_px(hx, hy)
    _line(frame, x0, y0, x1, y1, (255, 220, 40), 2)
    _disk(frame, x0, y0, 3, (40, 255, 120))

    # Safety HUD bars (top)
    bar_h = 12
    fw = frame.shape[1]
    third = fw // 3
    frame[0:bar_h, 0:third] = (int(255 * safety_scales["fwd"]), 40, 40)
    frame[0:bar_h, third:2 * third] = (40, int(255 * safety_scales["bwd"]), 40)
    frame[0:bar_h, 2 * third:] = (40, 40, int(255 * safety_scales["ang"]))

    # Caption strip
    cap_h = 28
    canvas = np.zeros((frame.shape[0] + cap_h, frame.shape[1], 3), dtype=np.uint8)
    canvas[: frame.shape[0]] = frame
    canvas[frame.shape[0] :] = (18, 18, 22)
    # Simple bitmap text via pixels is ugly — encode numbers in colored ticks
    # Prefer OpenCV text if available
    try:
        import cv2
        msg = (
            f"t={step}  pose=({robot.x:.2f},{robot.y:.2f},{math.degrees(robot.theta):.0f}deg)  "
            f"v={robot.v:.2f} w={robot.w:.2f}  "
            f"safe f={safety_scales['fwd']:.2f} b={safety_scales['bwd']:.2f} a={safety_scales['ang']:.2f}"
        )
        cv2.putText(
            canvas, msg, (6, frame.shape[0] + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 230), 1, cv2.LINE_AA,
        )
        cv2.putText(
            canvas, "HUD: RED=fwd  GREEN=bwd  BLUE=ang  (bright=full authority)",
            (6, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (240, 240, 240), 1, cv2.LINE_AA,
        )
    except Exception:
        pass
    return canvas


def _disk(img, x, y, r, color):
    h, w = img.shape[:2]
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy <= r * r:
                xx, yy = x + dx, y + dy
                if 0 <= xx < w and 0 <= yy < h:
                    img[yy, xx] = color


def _line(img, x0, y0, x1, y1, color, thickness=1):
    try:
        import cv2
        cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), color, thickness, cv2.LINE_AA)
    except Exception:
        # Bresenham fallback
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        h, w = img.shape[:2]
        while True:
            if 0 <= x0 < w and 0 <= y0 < h:
                img[y0, x0] = color
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy


def _poly(img, pts, color):
    try:
        import cv2
        arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [arr], True, color, 2, cv2.LINE_AA)
    except Exception:
        for i in range(len(pts)):
            x0, y0 = pts[i]
            x1, y1 = pts[(i + 1) % len(pts)]
            _line(img, x0, y0, x1, y1, color, 1)

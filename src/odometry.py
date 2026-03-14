"""odometry.py – Lightweight 2D visual odometry from forward camera.

ORB feature matching between consecutive greyscale frames.
Returns yaw (radians) and forward translation (metres) per frame.
Pure OpenCV, no external deps.
"""

import cv2
import numpy as np

_orb = cv2.ORB_create(500)
_bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
_prev_kp = None
_prev_des = None

FX = 307.0  # D435 color focal length at 320×240 (approximate)


def update(gray, avg_depth=1.5):
    """Feed forward-camera greyscale frame (320×240).

    Returns (yaw_rad, forward_m).
      yaw_rad  – positive = robot turned left (CCW from above).
      forward_m – positive = robot moved forward.
    Both 0.0 on first frame or when tracking fails.
    """
    global _prev_kp, _prev_des

    kp, des = _orb.detectAndCompute(gray, None)

    if _prev_des is None or des is None or len(des) < 10:
        _prev_kp, _prev_des = kp, des
        return 0.0, 0.0

    matches = _bf.match(_prev_des, des)

    if len(matches) < 8:
        _prev_kp, _prev_des = kp, des
        return 0.0, 0.0

    matches = sorted(matches, key=lambda m: m.distance)[:60]

    pts_p = np.float32([_prev_kp[m.queryIdx].pt for m in matches])
    pts_c = np.float32([kp[m.trainIdx].pt for m in matches])

    M, inliers = cv2.estimateAffinePartial2D(pts_p, pts_c, method=cv2.RANSAC)

    _prev_kp, _prev_des = kp, des

    if M is None or inliers is None or np.sum(inliers) < 5:
        return 0.0, 0.0

    tx = M[0, 2]
    yaw = tx / FX

    scale = np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2)
    forward = (scale - 1.0) * avg_depth

    return yaw, forward


def reset():
    """Clear tracking state (e.g. after a teleop jump)."""
    global _prev_kp, _prev_des
    _prev_kp = _prev_des = None

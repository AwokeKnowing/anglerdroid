"""odometry.py – Lightweight 2D visual odometry from forward camera.

ORB feature matching between consecutive greyscale frames.
Returns yaw (radians), forward translation (metres), and a confidence
score (0–1) indicating match quality.  Pure OpenCV, no external deps.

Matching uses kNN (k=2) with Lowe's ratio test instead of cross-check,
which significantly improves match quality and outlier rejection.
"""

import cv2
import numpy as np

_orb = cv2.ORB_create(300)
_bf = cv2.BFMatcher(cv2.NORM_HAMMING)
_prev_kp = None
_prev_des = None

FX = 307.0  # D435 color focal length at 320×240 (approximate)

# Matching
_LOWE_RATIO = 0.78       # Lowe ratio test threshold (lower = stricter)
_MAX_MATCHES = 100        # keep best N after ratio test

# Confidence tuning
_INLIER_SATURATE = 30     # inlier count at which count-confidence saturates
_MAX_GOOD_DIST = 45.0     # mean match distance above this → zero dist-confidence


def update(gray, avg_depth=1.5):
    """Feed forward-camera greyscale frame (320×240).

    Returns (yaw_rad, forward_m, confidence).
      yaw_rad    – positive = robot turned left (CCW from above).
      forward_m  – positive = robot moved forward.
      confidence – 0.0 (no match / garbage) to 1.0 (strong match).
    All zeros on first frame or when tracking fails outright.
    """
    global _prev_kp, _prev_des

    kp, des = _orb.detectAndCompute(gray, None)

    if _prev_des is None or des is None or len(des) < 10:
        _prev_kp, _prev_des = kp, des
        return 0.0, 0.0, 0.0

    raw = _bf.knnMatch(_prev_des, des, k=2)

    good = []
    for pair in raw:
        if len(pair) == 2:
            m, n = pair
            if m.distance < _LOWE_RATIO * n.distance:
                good.append(m)

    if len(good) < 8:
        _prev_kp, _prev_des = kp, des
        return 0.0, 0.0, 0.0

    good = sorted(good, key=lambda m: m.distance)[:_MAX_MATCHES]

    pts_p = np.float32([_prev_kp[m.queryIdx].pt for m in good])
    pts_c = np.float32([kp[m.trainIdx].pt for m in good])

    M, inliers = cv2.estimateAffinePartial2D(pts_p, pts_c, method=cv2.RANSAC,
                                              ransacReprojThreshold=2.0)

    _prev_kp, _prev_des = kp, des

    if M is None or inliers is None:
        return 0.0, 0.0, 0.0

    n_inliers = int(np.sum(inliers))
    if n_inliers < 5:
        return 0.0, 0.0, 0.0

    # --- Extract motion ---
    tx = M[0, 2]
    yaw = tx / FX

    scale = np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2)
    forward = (scale - 1.0) * avg_depth

    # --- Confidence from match quality ---
    n_total = len(good)
    inlier_ratio = n_inliers / n_total if n_total > 0 else 0.0
    count_conf = min(1.0, n_inliers / _INLIER_SATURATE)

    inlier_mask = inliers.ravel().astype(bool)
    inlier_dists = np.array([good[i].distance for i in range(n_total)
                             if inlier_mask[i]], dtype=np.float32)
    mean_dist = float(np.mean(inlier_dists)) if len(inlier_dists) > 0 else _MAX_GOOD_DIST
    dist_conf = max(0.0, 1.0 - mean_dist / _MAX_GOOD_DIST)

    # Reprojection residual
    pts_p_h = np.hstack([pts_p, np.ones((len(pts_p), 1), dtype=np.float32)])
    predicted = pts_p_h @ M.T
    residuals = np.linalg.norm(predicted - pts_c, axis=1)
    mean_reproj = float(np.mean(residuals[inlier_mask]))
    reproj_conf = max(0.0, 1.0 - mean_reproj / 4.0)  # >4px → 0 (was 3px)

    confidence = count_conf * inlier_ratio * dist_conf * reproj_conf

    return yaw, forward, confidence


def reset():
    """Clear tracking state (e.g. after a teleop jump)."""
    global _prev_kp, _prev_des
    _prev_kp = _prev_des = None

"""Colour correction from the resolved config (robots/<robot>/robot.json cameras.rgb_card_gains): per-channel gains that
make the deck's WhiBal neutral card grey at the fixed white balance (scripts/color_card_eval.py --calibrate measures them).
Consumers call apply(); nothing here runs in the safety core."""
import numpy as np
import robot_params as _RP

GAINS = np.asarray(_RP.V.get("cam_rgb_card_gains", [1.0, 1.0, 1.0]), np.float32)


def apply(img_rgb, gains=None):
    """RGB uint8 -> RGB uint8 with the card gains (BGR input: pass gains reversed, or use apply_bgr)."""
    g = GAINS if gains is None else np.asarray(gains, np.float32)
    return np.clip(img_rgb.astype(np.float32) * g, 0, 255).astype(np.uint8)


def apply_bgr(img_bgr):
    return apply(img_bgr, GAINS[::-1])


def neutral_err_deg(rgb_mean):
    m = np.asarray(rgb_mean, np.float64)
    return float(np.degrees(np.arccos(min(1.0, m.sum() / (np.linalg.norm(m) * np.sqrt(3))))))

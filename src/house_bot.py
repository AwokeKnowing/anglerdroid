"""house_bot.py – Curious look-before-leap for Kevin.

CRITICAL: Divert EARLY. Once SafetyGuard pins forward (and often angular),
"turn left" is already too late. Steer while clearance still allows yaw.

Mast-aware: tall cells (>= MAST_CLEAR_CM) count as overhangs/tables.
"""

from __future__ import annotations

import math
import os
import threading
import time

import cv2
import numpy as np

import local_executive
import speech_io
from robot_config import MAST_CLEAR_CM, RCX, RCY, FOOT_X1

SNAP_DIR = os.path.expanduser("~/.kevin/snapshots")
LOOK_PERIOD_S = 1.0
NARRATE_EVERY_S = 50.0
OBS_THRESH = 100

# Divert while we can still turn — NOT after nose is pinned.
EARLY_FWD_SCALE = 0.55
LATE_FWD_SCALE = 0.18
EARLY_FREE = 0.72
EARLY_MAST = 0.06
MIN_ANG_TO_DIVERT = 0.30


class HouseBot:
    def __init__(self, vision, enabled=True):
        self.vision = vision
        self.enabled = enabled
        self._stop = False
        self._thread = None
        self._last_narrate = 0.0
        self._turn_bias = 1.0
        self._n_looks = 0
        self._last_decision = "init"
        os.makedirs(SNAP_DIR, exist_ok=True)

    def start(self):
        if not self.enabled:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("house_bot: started (EARLY divert @ %.1fs, voice=am_michael)" % LOOK_PERIOD_S)
        speech_io.speak("Hello. I am Kevin. I look ahead early so I can still turn.")

    def stop(self):
        self._stop = True

    def _snapshot(self):
        vis = self.vision
        ts = time.strftime("%H%M%S")
        paths = {}
        try:
            for name, idx in (("rgb", 0), ("rs1", 1), ("rs2", 2)):
                if vis.frames is None or idx >= len(vis.frames):
                    continue
                fr = vis.frames[idx]
                if fr is None or fr.size == 0:
                    continue
                p = os.path.join(SNAP_DIR, "%s_%s.jpg" % (ts, name))
                cv2.imwrite(p, fr[:, :, ::-1].copy(), [cv2.IMWRITE_JPEG_QUALITY, 75])
                paths[name] = p
            if vis.atlas is not None:
                p = os.path.join(SNAP_DIR, "%s_atlas.jpg" % ts)
                cv2.imwrite(p, vis.atlas[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 70])
                paths["atlas"] = p
        except Exception as e:
            print("house_bot: snapshot err %s" % e)
        return paths

    def _score_sectors(self, obs, height=None):
        if obs is None:
            return {
                "fwd_near": 0.5, "fwd_mid": 0.5, "left": 0.5, "right": 0.5,
                "mast_near": 0.0, "mast_mid": 0.0,
            }
        if obs.ndim == 3:
            obs = obs[:, :, 0]
        h, w = obs.shape[:2]
        blocked = (obs.astype(np.float32) >= OBS_THRESH).astype(np.float32)
        tall = None
        if height is not None:
            tall = (height.astype(np.float32) >= MAST_CLEAR_CM).astype(np.float32)
            blocked = np.clip(blocked + tall, 0, 1)

        def free_score(y0, y1, x0, x1):
            patch = blocked[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
            if patch.size == 0:
                return 0.0
            return 1.0 - float(patch.mean())

        def mast_score(y0, y1, x0, x1):
            if tall is None:
                return 0.0
            patch = tall[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
            if patch.size == 0:
                return 0.0
            return float(patch.mean())

        # Ego map: forward = +x (rightward from FOOT_X1), same as SafetyGuard.
        cx, cy = int(RCX), int(RCY)
        near_x0, near_x1 = FOOT_X1, min(w, FOOT_X1 + 28)
        mid_x0, mid_x1 = min(w, FOOT_X1 + 28), min(w, FOOT_X1 + 85)
        y0, y1 = max(0, cy - 12), min(h, cy + 12)

        return {
            "fwd_near": free_score(y0, y1, near_x0, near_x1),
            "fwd_mid": free_score(y0, y1, mid_x0, mid_x1),
            "left": free_score(max(0, cy - 40), max(0, cy - 8), FOOT_X1, min(w, FOOT_X1 + 50)),
            "right": free_score(min(h, cy + 8), min(h, cy + 40), FOOT_X1, min(w, FOOT_X1 + 50)),
            "mast_near": mast_score(y0, y1, near_x0, near_x1),
            "mast_mid": mast_score(y0, y1, mid_x0, mid_x1),
        }

    def _loop(self):
        time.sleep(3.0)
        while not self._stop:
            t0 = time.monotonic()
            try:
                self._tick()
            except Exception as e:
                print("house_bot: tick err %s" % e)
            time.sleep(max(0.15, LOOK_PERIOD_S - (time.monotonic() - t0)))

    def _pick_turn(self, scores):
        left = scores["left"] - 0.5 * scores.get("mast_mid", 0)
        right = scores["right"] - 0.5 * scores.get("mast_mid", 0)
        if abs(left - right) < 0.05:
            turn = "left" if self._turn_bias > 0 else "right"
            self._turn_bias *= -1.0
        elif left >= right:
            turn = "left"
        else:
            turn = "right"
        return turn

    def _set_turn_goal(self, turn, soft=True):
        pose = getattr(self.vision, "_pose", None)
        deg = (35.0 if soft else 75.0) * (1.0 if turn == "left" else -1.0)
        dist = 0.95 if soft else 0.65
        if pose is None:
            local_executive.set_wander()
            return deg
        th = pose.theta + math.radians(deg)
        local_executive.set_goal_xy(
            pose.x + dist * math.cos(th),
            pose.y + dist * math.sin(th),
        )
        return deg

    def _tick(self):
        vis = self.vision
        self._n_looks += 1
        paths = self._snapshot()
        scores = self._score_sectors(
            getattr(vis, "_persistent_obs", None),
            getattr(vis, "_persistent_height", None),
        )
        fwd_scale = float(getattr(vis, "safety_fwd_scale", 1.0) or 1.0)
        ang_scale = float(getattr(vis, "safety_ang_scale", 1.0) or 1.0)
        scores["safety_fwd"] = fwd_scale
        scores["safety_ang"] = ang_scale

        mast_ahead = max(scores.get("mast_near", 0), scores.get("mast_mid", 0))
        free_mid = scores.get("fwd_mid", 1.0)
        free_near = scores.get("fwd_near", 1.0)

        early = (
            ang_scale >= MIN_ANG_TO_DIVERT
            and (
                (fwd_scale < EARLY_FWD_SCALE and fwd_scale >= LATE_FWD_SCALE)
                or (free_mid < EARLY_FREE)
                or (mast_ahead > EARLY_MAST)
                or (free_near < 0.80 and free_mid < 0.85)
            )
        )
        late = (fwd_scale < LATE_FWD_SCALE) or (free_near < 0.40)

        decision = "wander"
        note = "ok fwd=%.2f mid=%.2f mast=%.2f ang=%.2f" % (
            fwd_scale, free_mid, mast_ahead, ang_scale)

        if early or late:
            soft = bool(early and not late)
            turn = self._pick_turn(scores)
            deg = self._set_turn_goal(turn, soft=soft)
            decision = ("early_" if soft else "late_") + turn
            note = (
                "%s divert %s deg=%.0f | fwd=%.2f mid=%.2f near=%.2f mast=%.2f ang=%.2f"
                % (("early" if soft else "LATE"), turn, deg,
                   fwd_scale, free_mid, free_near, mast_ahead, ang_scale)
            )
            if decision != self._last_decision and not speech_io.is_speaking():
                if mast_ahead > EARLY_MAST:
                    speech_io.speak("Tall obstacle ahead. Veering %s." % turn)
                elif soft:
                    speech_io.speak("Path tightening. Going %s." % turn)
                else:
                    speech_io.speak("Tight. Turning %s." % turn)
        else:
            if not local_executive.is_active():
                local_executive.set_wander()
            now = time.monotonic()
            if now - self._last_narrate > NARRATE_EVERY_S and not speech_io.is_speaking():
                self._last_narrate = now
                speech_io.speak("Looking ahead. Still clear.")

        self._last_decision = decision
        line = "house_bot: look#%d %s %s snaps=%s" % (
            self._n_looks, decision, note, ",".join(paths.keys()))
        print(line)
        try:
            with open(os.path.join(SNAP_DIR, "latest.txt"), "w") as f:
                f.write(line + "\n")
                f.write("scores=%s\n" % scores)
                for k, v in paths.items():
                    f.write("%s=%s\n" % (k, v))
        except Exception:
            pass

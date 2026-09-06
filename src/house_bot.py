"""house_bot.py – Curious look-before-leap for Kevin.

CRITICAL: Divert EARLY. Once SafetyGuard pins forward (and often angular),
"turn left" is already too late. Steer while clearance still allows yaw.

Mast-aware: tall cells (>= MAST_CLEAR_CM) count as overhangs/tables.
Stuck recovery: after several late looks, reverse+spin toward freer side.
"""

from __future__ import annotations

import math
import os
import threading
import time

import cv2
import numpy as np

import local_executive
import tools
import speech_io
from robot_config import MAST_CLEAR_CM, RCX, RCY, FOOT_X1
from safety import build_safety_occ, OBS_THRESH as SAFETY_OBS

SNAP_DIR = os.path.expanduser("~/.kevin/snapshots")
LOOK_PERIOD_S = 1.0
NARRATE_EVERY_S = 50.0
OBS_THRESH = 100

# Divert while we can still turn — NOT after nose is pinned.
EARLY_FWD_SCALE = 0.55
LATE_FWD_SCALE = 0.18
EARLY_FREE = 0.72
EARLY_MAST = 0.12
MIN_ANG_TO_DIVERT = 0.30
LATE_STUCK_LOOKS = 4          # consecutive late looks → reverse+spin
RECOVER_BACK_MPS = -0.18
RECOVER_SPIN_RAD = 0.95
RECOVER_SECS = 1.1
STICKY_BREAK_MARGIN = 0.25    # freer opposite breaks sticky early


class HouseBot:
    def __init__(self, vision, enabled=True):
        self.vision = vision
        self.enabled = enabled
        self._stop = False
        self._thread = None
        self._last_narrate = 0.0
        self._turn_bias = 1.0
        self._n_looks = 0
        self._escape_until = 0.0
        self._escape_turn = None
        self._late_streak = 0
        self._recover_until = 0.0
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
        # Same mast-inflated occ SafetyGuard uses (footprint cleared).
        occ = build_safety_occ(obs, height)
        blocked = (occ.astype(np.float32) >= 100).astype(np.float32)
        tall = None
        if height is not None:
            tall = (height.astype(np.float32) >= MAST_CLEAR_CM).astype(np.float32)

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

    def _set_turn_goal(self, turn, soft=True, recover=False, bwd_ok=True):
        pose = getattr(self.vision, "_pose", None)
        sign = 1.0 if turn == "left" else -1.0
        if soft:
            # Gentle veer while still moving — MPPI goal ~35° ahead.
            self._escape_until = 0.0
            self._escape_turn = None
            self._late_streak = 0
            deg = 35.0 * sign
            dist = 0.95
            if pose is None:
                local_executive.set_wander()
                return deg
            th = pose.theta + math.radians(deg)
            local_executive.set_goal_xy(
                pose.x + dist * math.cos(th),
                pose.y + dist * math.sin(th),
            )
            return deg

        # LATE: nose pinned — MPPI can't push through fwd_scale=0.
        # Sticky spin: hold one direction ~2.5s so we don't thrash left/right
        # and cancel MPPI mid-tick every look.
        now = time.monotonic()
        if (not recover) and now < self._escape_until and self._escape_turn:
            turn = self._escape_turn
            sign = 1.0 if turn == "left" else -1.0
        else:
            self._escape_turn = turn
            self._escape_until = now + (RECOVER_SECS + 0.4 if recover else 2.5)
            local_executive.clear()
            ang = (RECOVER_SPIN_RAD if recover else 0.85) * sign
            # Back up while yawing when stuck — pure spin often stays pinned.
            fwd = RECOVER_BACK_MPS if (recover and bwd_ok) else 0.0
            tools.twist_for(
                fwd, ang,
                duration_secs=(RECOVER_SECS if recover else 2.4),
                ramp_in_secs=0.12, ramp_out_secs=0.2,
            )
            if recover:
                self._recover_until = now + RECOVER_SECS + 0.3
                self._late_streak = 0
        deg = (120.0 if recover else 95.0) * sign
        return deg

    def _tick(self):
        vis = self.vision
        self._n_looks += 1
        paths = self._snapshot()
        scores = self._score_sectors(
            getattr(vis, "_persistent_obs", None),
            getattr(vis, "_persistent_height", None),
        )
        # Do NOT use `or 1.0` — real safety scale 0.0 is falsy and must stay 0.
        try:
            fwd_scale = float(vis.safety_fwd_scale)
        except Exception:
            fwd_scale = 1.0
        try:
            ang_scale = float(vis.safety_ang_scale)
        except Exception:
            ang_scale = 1.0
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
            now = time.monotonic()
            if soft:
                self._late_streak = 0
            else:
                self._late_streak += 1

            # Prefer freer side; break sticky early if opposite is clearly better.
            preferred = self._pick_turn(scores)
            sticky = (
                (not soft)
                and now < self._escape_until
                and self._escape_turn
                and now >= self._recover_until
            )
            if sticky:
                sticky_turn = self._escape_turn
                left_s = scores["left"] - 0.5 * scores.get("mast_mid", 0)
                right_s = scores["right"] - 0.5 * scores.get("mast_mid", 0)
                sticky_score = left_s if sticky_turn == "left" else right_s
                other_score = right_s if sticky_turn == "left" else left_s
                if other_score >= sticky_score + STICKY_BREAK_MARGIN:
                    turn = preferred
                    self._escape_until = 0.0  # force re-arm toward freer side
                else:
                    turn = sticky_turn
            else:
                turn = preferred

            try:
                bwd_ok = float(vis.safety_bwd_scale) > 0.25
            except Exception:
                bwd_ok = True
            recover = (not soft) and self._late_streak >= LATE_STUCK_LOOKS and now >= self._recover_until
            deg = self._set_turn_goal(turn, soft=soft, recover=recover, bwd_ok=bwd_ok)
            decision = ("early_" if soft else ("recover_" if recover else "late_")) + turn
            note = (
                "%s divert %s deg=%.0f | fwd=%.2f mid=%.2f near=%.2f mast=%.2f ang=%.2f streak=%d"
                % (("early" if soft else ("RECOVER" if recover else "LATE")), turn, deg,
                   fwd_scale, free_mid, free_near, mast_ahead, ang_scale, self._late_streak)
            )
            if decision != self._last_decision and not speech_io.is_speaking():
                if recover:
                    speech_io.speak("Stuck. Backing and turning %s." % turn)
                elif mast_ahead > EARLY_MAST:
                    speech_io.speak("Tall obstacle ahead. Veering %s." % turn)
                elif soft:
                    speech_io.speak("Path tightening. Going %s." % turn)
                else:
                    speech_io.speak("Tight. Turning %s." % turn)
        else:
            self._late_streak = 0
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

"""house_bot.py – Curious look-before-leap for Kevin.

CRITICAL: Divert EARLY. Once SafetyGuard pins forward (and often angular),
"turn left" is already too late. Steer while clearance still allows yaw.

Mast-aware: tall cells (>= MAST_CLEAR_CM) count as overhangs/tables.

Stuck recovery (no forward/back oscillation):
  1) BACK  – reverse out of the pinch
  2) SPIN  – ~180° toward freer/curious side
  3) COMMIT – keep driving the NEW way for several seconds
             (do not immediately nose back into the same pinch)
"""

from __future__ import annotations

import math
import os
import random
import threading
import time

import cv2
import numpy as np

import local_executive
import tools
import speech_io
import people_live as people_live_mod
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
LATE_STUCK_LOOKS = 3          # consecutive late looks → phased recover

# Phased recover: back → ~180 spin → commit other way
BACK_MPS = -0.22
BACK_SECS = 1.15
SPIN_RAD = 0.90               # before safety_ang scale (~0.3 → ~0.27 rad/s)
SPIN_SECS = 3.2               # ~180° at ~0.27 rad/s * 0.3 floor needs longer; aim big
COMMIT_SECS = 5.5
COMMIT_DIST_M = 1.25
STICKY_BREAK_MARGIN = 0.25
CURIOUS_FLIP_P = 0.35         # sometimes pick the other side on purpose


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
        self._last_decision = "init"
        # Phased recover state
        self._phase = None          # None | 'back' | 'spin' | 'commit'
        self._phase_until = 0.0
        self._phase_turn = None     # 'left' | 'right'
        self._commit_theta = None   # world heading to keep after turnaround
        os.makedirs(SNAP_DIR, exist_ok=True)

    def start(self):
        if not self.enabled:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("house_bot: started (EARLY divert + turnaround recover, voice=am_michael)")
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

    def _pick_turn(self, scores, curious=False):
        left = scores["left"] - 0.5 * scores.get("mast_mid", 0)
        right = scores["right"] - 0.5 * scores.get("mast_mid", 0)
        if abs(left - right) < 0.08:
            turn = "left" if self._turn_bias > 0 else "right"
            self._turn_bias *= -1.0
        elif left >= right:
            turn = "left"
        else:
            turn = "right"
        # Curiosity: sometimes leave the "optimal" side so we don't tunnel
        # into the same dead-end forever.
        if curious and random.random() < CURIOUS_FLIP_P:
            turn = "right" if turn == "left" else "left"
        return turn

    def _pose(self):
        return getattr(self.vision, "_pose", None)

    def _bwd_ok(self):
        try:
            return float(self.vision.safety_bwd_scale) > 0.20
        except Exception:
            return True

    def _start_back(self, turn):
        """Phase 1: reverse out of the pinch (little yaw)."""
        now = time.monotonic()
        self._phase = "back"
        self._phase_turn = turn
        self._phase_until = now + BACK_SECS
        self._escape_turn = turn
        self._escape_until = self._phase_until
        self._late_streak = 0
        local_executive.clear()
        sign = 1.0 if turn == "left" else -1.0
        fwd = BACK_MPS if self._bwd_ok() else 0.0
        # Mostly reverse; tiny yaw so we don't stay square to the wall.
        tools.twist_for(
            fwd, 0.25 * sign,
            duration_secs=BACK_SECS,
            ramp_in_secs=0.1, ramp_out_secs=0.15,
        )
        print("house_bot: PHASE back turn=%s fwd=%.2f for %.1fs" % (turn, fwd, BACK_SECS))

    def _start_spin(self):
        """Phase 2: big turnaround spin (~180° ambition)."""
        now = time.monotonic()
        turn = self._phase_turn or "left"
        sign = 1.0 if turn == "left" else -1.0
        self._phase = "spin"
        self._phase_until = now + SPIN_SECS
        self._escape_turn = turn
        self._escape_until = self._phase_until
        local_executive.clear()
        tools.twist_for(
            0.0, SPIN_RAD * sign,
            duration_secs=SPIN_SECS,
            ramp_in_secs=0.1, ramp_out_secs=0.2,
        )
        print("house_bot: PHASE spin turn=%s for %.1fs (turnaround)" % (turn, SPIN_SECS))

    def _start_commit(self):
        """Phase 3: drive the NEW way; don't immediately re-enter the pinch."""
        now = time.monotonic()
        pose = self._pose()
        self._phase = "commit"
        self._phase_until = now + COMMIT_SECS
        if pose is not None:
            # After spin, current theta IS the new "other way".
            self._commit_theta = float(pose.theta)
            th = self._commit_theta
            local_executive.set_goal_xy(
                pose.x + COMMIT_DIST_M * math.cos(th),
                pose.y + COMMIT_DIST_M * math.sin(th),
            )
        else:
            self._commit_theta = None
            local_executive.set_wander()
        print("house_bot: PHASE commit other-way for %.1fs theta=%s" % (
            COMMIT_SECS,
            ("%.1f°" % math.degrees(self._commit_theta)) if self._commit_theta is not None else "?",
        ))

    def _refresh_commit_goal(self):
        pose = self._pose()
        if pose is None or self._commit_theta is None:
            if not local_executive.is_active():
                local_executive.set_wander()
            return
        th = self._commit_theta
        local_executive.set_goal_xy(
            pose.x + COMMIT_DIST_M * math.cos(th),
            pose.y + COMMIT_DIST_M * math.sin(th),
        )

    def _advance_phase(self):
        """Step back→spin→commit→done when timers elapse."""
        now = time.monotonic()
        if self._phase is None:
            return False
        if now < self._phase_until:
            return True  # still in phase
        if self._phase == "back":
            self._start_spin()
            return True
        if self._phase == "spin":
            self._start_commit()
            return True
        if self._phase == "commit":
            print("house_bot: PHASE done — resume normal wander")
            self._phase = None
            self._phase_turn = None
            self._commit_theta = None
            self._escape_until = 0.0
            self._escape_turn = None
            return False
        return False

    def _set_soft_veer(self, turn):
        pose = self._pose()
        sign = 1.0 if turn == "left" else -1.0
        deg = 35.0 * sign
        if pose is None:
            local_executive.set_wander()
            return deg
        th = pose.theta + math.radians(deg)
        local_executive.set_goal_xy(
            pose.x + 0.95 * math.cos(th),
            pose.y + 0.95 * math.sin(th),
        )
        return deg

    def _set_late_spin(self, turn):
        """Short in-place spin while waiting for recover threshold."""
        now = time.monotonic()
        sign = 1.0 if turn == "left" else -1.0
        if now < self._escape_until and self._escape_turn:
            turn = self._escape_turn
            sign = 1.0 if turn == "left" else -1.0
        else:
            self._escape_turn = turn
            self._escape_until = now + 2.0
            local_executive.clear()
            tools.twist_for(
                0.0, 0.85 * sign,
                duration_secs=2.0,
                ramp_in_secs=0.12, ramp_out_secs=0.2,
            )
        return 95.0 * sign

    def _tick(self):
        vis = self.vision
        self._n_looks += 1
        paths = self._snapshot()
        scores = self._score_sectors(
            getattr(vis, "_persistent_obs", None),
            getattr(vis, "_persistent_height", None),
        )
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

        # ── Phased recover owns the robot until commit finishes ──
        in_phase = self._advance_phase() if self._phase else False
        if self._phase:
            in_phase = True
            turn = self._phase_turn or "left"
            if self._phase == "commit":
                # Keep going the new way; only abort if nose is HARD-pinned again
                # AND free space ahead is poor. Momentary fwd_scale=0 with free mid
                # (mast-inflation ghosts) must not throw away a turnaround commit.
                hard_repin = (
                    fwd_scale < LATE_FWD_SCALE
                    and free_mid < 0.50
                )
                if hard_repin:
                    self._late_streak += 1
                else:
                    self._late_streak = 0
                if hard_repin and self._late_streak >= LATE_STUCK_LOOKS:
                    preferred = self._pick_turn(scores, curious=True)
                    self._start_back(preferred)
                    decision = "recover_back_" + preferred
                    note = "RECOVER re-back %s | fwd=%.2f mid=%.2f" % (
                        preferred, fwd_scale, free_mid)
                else:
                    self._refresh_commit_goal()
                    decision = "commit_" + (turn or "fwd")
                    note = "COMMIT other-way %.1fs left | fwd=%.2f mid=%.2f" % (
                        max(0.0, self._phase_until - time.monotonic()),
                        fwd_scale, free_mid)
            else:
                decision = "recover_%s_%s" % (self._phase, turn)
                note = "RECOVER %s %s | fwd=%.2f bwd_ok=%s" % (
                    self._phase, turn, fwd_scale, self._bwd_ok())
            self._last_decision = decision
            line = "house_bot: look#%d %s %s snaps=%s" % (
                self._n_looks, decision, note, ",".join(paths.keys()))
            print(line)
            try:
                with open(os.path.join(SNAP_DIR, "latest.txt"), "w") as f:
                    f.write(line + "\n")
                    f.write("scores=%s\n" % scores)
            except Exception:
                pass
            return

        # ── Normal look-before-leap ──
        if early or late:
            soft = bool(early and not late)
            now = time.monotonic()
            if soft:
                self._late_streak = 0
            else:
                self._late_streak += 1

            preferred = self._pick_turn(scores, curious=False)
            sticky = (
                (not soft)
                and now < self._escape_until
                and self._escape_turn
            )
            if sticky:
                sticky_turn = self._escape_turn
                left_s = scores["left"] - 0.5 * scores.get("mast_mid", 0)
                right_s = scores["right"] - 0.5 * scores.get("mast_mid", 0)
                sticky_score = left_s if sticky_turn == "left" else right_s
                other_score = right_s if sticky_turn == "left" else left_s
                if other_score >= sticky_score + STICKY_BREAK_MARGIN:
                    turn = preferred
                    self._escape_until = 0.0
                else:
                    turn = sticky_turn
            else:
                turn = preferred

            if (not soft) and self._late_streak >= LATE_STUCK_LOOKS:
                # Curiosity on the turnaround direction
                turn = self._pick_turn(scores, curious=True)
                self._start_back(turn)
                decision = "recover_back_" + turn
                note = (
                    "RECOVER start back→spin→commit %s | fwd=%.2f mid=%.2f near=%.2f streak=%d"
                    % (turn, fwd_scale, free_mid, free_near, self._late_streak)
                )
                if decision != self._last_decision and not speech_io.is_speaking() and not getattr(people_live_mod.PeopleLive, "social_priority", False):
                    speech_io.speak("No room. Backing up, then the other way.")
            elif soft:
                deg = self._set_soft_veer(turn)
                decision = "early_" + turn
                note = (
                    "early divert %s deg=%.0f | fwd=%.2f mid=%.2f near=%.2f mast=%.2f ang=%.2f"
                    % (turn, deg, fwd_scale, free_mid, free_near, mast_ahead, ang_scale)
                )
                if decision != self._last_decision and not speech_io.is_speaking() and not getattr(people_live_mod.PeopleLive, "social_priority", False):
                    if mast_ahead > EARLY_MAST:
                        speech_io.speak("Tall obstacle ahead. Veering %s." % turn)
                    else:
                        speech_io.speak("Path tightening. Going %s." % turn)
            else:
                deg = self._set_late_spin(turn)
                decision = "late_" + turn
                note = (
                    "LATE divert %s deg=%.0f | fwd=%.2f mid=%.2f near=%.2f ang=%.2f streak=%d"
                    % (turn, deg, fwd_scale, free_mid, free_near, ang_scale, self._late_streak)
                )
                if decision != self._last_decision and not speech_io.is_speaking() and not getattr(people_live_mod.PeopleLive, "social_priority", False):
                    speech_io.speak("Tight. Turning %s." % turn)
        else:
            self._late_streak = 0
            if not local_executive.is_active():
                local_executive.set_wander()
            now = time.monotonic()
            # Yield the speaker when a face greet is queued.
            if getattr(people_live_mod.PeopleLive, "social_priority", False):
                pass
            elif now - self._last_narrate > NARRATE_EVERY_S and not speech_io.is_speaking():
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

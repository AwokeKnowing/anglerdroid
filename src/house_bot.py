"""house_bot.py – Curious look-before-leap layer for Kevin."""

from __future__ import annotations

import math
import os
import threading
import time

import cv2
import numpy as np

import local_executive
import speech_io

SNAP_DIR = os.path.expanduser("~/.kevin/snapshots")
LOOK_PERIOD_S = 2.0
NARRATE_EVERY_S = 45.0
FWD_BLOCKED = 0.12
OBS_THRESH = 100


class HouseBot:
    def __init__(self, vision, enabled=True):
        self.vision = vision
        self.enabled = enabled
        self._stop = False
        self._thread = None
        self._last_narrate = 0.0
        self._turn_bias = 1.0
        self._n_looks = 0
        os.makedirs(SNAP_DIR, exist_ok=True)

    def start(self):
        if not self.enabled:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("house_bot: started (look-before-leap @ %.1fs)" % LOOK_PERIOD_S)
        speech_io.speak("Hello. I am Kevin. I will look before I wander.")

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
                p = os.path.join(SNAP_DIR, f"{ts}_{name}.jpg")
                cv2.imwrite(p, fr[:, :, ::-1].copy(), [cv2.IMWRITE_JPEG_QUALITY, 75])
                paths[name] = p
            if vis.atlas is not None:
                p = os.path.join(SNAP_DIR, f"{ts}_atlas.jpg")
                cv2.imwrite(p, vis.atlas[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 70])
                paths["atlas"] = p
        except Exception as e:
            print("house_bot: snapshot err %s" % e)
        return paths

    def _score_sectors(self, obs):
        if obs is None:
            return {"fwd": 0.5, "left": 0.5, "right": 0.5}
        if obs.ndim == 3:
            obs = obs[:, :, 0]
        h, w = obs.shape[:2]
        blocked = (obs.astype(np.float32) >= OBS_THRESH).astype(np.float32)
        cy, cx = h * 2 // 3, w // 2

        def free_score(y0, y1, x0, x1):
            patch = blocked[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
            if patch.size == 0:
                return 0.0
            return 1.0 - float(patch.mean())

        return {
            "fwd": free_score(cy - h // 3, cy, cx - w // 10, cx + w // 10),
            "left": free_score(cy - h // 4, cy + h // 8, 0, cx - w // 8),
            "right": free_score(cy - h // 4, cy + h // 8, cx + w // 8, w),
        }

    def _loop(self):
        time.sleep(4.0)
        while not self._stop:
            t0 = time.monotonic()
            try:
                self._tick()
            except Exception as e:
                print("house_bot: tick err %s" % e)
            time.sleep(max(0.2, LOOK_PERIOD_S - (time.monotonic() - t0)))

    def _tick(self):
        vis = self.vision
        self._n_looks += 1
        paths = self._snapshot()
        scores = self._score_sectors(getattr(vis, "_persistent_obs", None))
        fwd_scale = float(getattr(vis, "safety_fwd_scale", 1.0) or 1.0)
        scores["safety_fwd"] = fwd_scale
        blocked = (fwd_scale < FWD_BLOCKED) or (scores["fwd"] < 0.35)
        if blocked:
            turn = "left" if scores["left"] >= scores["right"] else "right"
            hdg = 70.0 * self._turn_bias if turn == "left" else -70.0 * self._turn_bias
            self._turn_bias *= -1.0
            pose = getattr(vis, "_pose", None)
            if pose is not None:
                th = pose.theta + math.radians(hdg)
                local_executive.set_goal_xy(
                    pose.x + 0.7 * math.cos(th),
                    pose.y + 0.7 * math.sin(th),
                )
            else:
                local_executive.set_wander()
            decision = "turn_" + turn
            note = "blocked safety=%.2f free=%.2f → %s" % (fwd_scale, scores["fwd"], turn)
            if not speech_io.is_speaking():
                speech_io.speak("Oops, blocked. Turning %s." % turn)
        else:
            if not local_executive.is_active():
                local_executive.set_wander()
            decision = "wander"
            note = "clear safety=%.2f free=%.2f" % (fwd_scale, scores["fwd"])
            now = time.monotonic()
            if now - self._last_narrate > NARRATE_EVERY_S and not speech_io.is_speaking():
                self._last_narrate = now
                speech_io.speak("Looking around. Path looks open.")
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

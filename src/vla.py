"""
vla.py – Vision-Language-Action client.
Sends atlas frames to a remote inference server, maintains an action buffer.
The main loop consumes one action per FRAMES_PER_ACTION frames.
Python 3.8 compatible.
"""

import base64
import collections
import json
import threading
import time
from typing import Optional, Dict
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import cv2
import numpy as np

ACTIONS_PER_BATCH = 8
REFILL_THRESHOLD = 3
FRAMES_PER_ACTION = 10   # 333 ms at 30 fps
INFER_TIMEOUT = 15.0
MAX_FORWARD_MPS = 0.3
MAX_ANGULAR_RDS = 1.0
MAX_INSTRUCTION_SECS = 15.0  # auto-stop after this many seconds


class VLAClient:
    """HTTP client for a remote VLA inference server + local action buffer."""

    def __init__(self, endpoint_url):
        # type: (str) -> None
        self._endpoint = endpoint_url.rstrip("/")
        self._infer_url = self._endpoint + "/infer"

        self._instruction = ""
        self._instruction_lock = threading.Lock()

        self._actions = collections.deque()        # type: collections.deque
        self._actions_lock = threading.Lock()
        self._current_action = None                # type: Optional[Dict]
        self._action_frames_left = 0

        self._latest_atlas = None                  # type: Optional[np.ndarray]
        self._atlas_lock = threading.Lock()

        self._running = False
        self._thread = None                        # type: Optional[threading.Thread]
        self._requesting = False
        self._connected = False
        self._last_error = ""
        self._deadline = 0.0  # monotonic time when current instruction expires

    # ── lifecycle ─────────────────────────────────────────────

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self._check_health()
        print("vla: client started -> %s" % self._endpoint)

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        print("vla: client stopped")

    # ── main-loop interface ───────────────────────────────────

    def set_instruction(self, instruction):
        # type: (str) -> None
        with self._instruction_lock:
            changed = self._instruction != instruction
            self._instruction = instruction
            if instruction:
                self._deadline = time.monotonic() + MAX_INSTRUCTION_SECS
            else:
                self._deadline = 0.0
        if changed:
            with self._actions_lock:
                self._actions.clear()
                self._current_action = None
                self._action_frames_left = 0
            if instruction:
                print("vla: instruction = '%s' (%.0fs timeout)" % (
                    instruction[:80], MAX_INSTRUCTION_SECS))
            else:
                print("vla: instruction cleared (idle)")

    def get_instruction(self):
        # type: () -> str
        with self._instruction_lock:
            return self._instruction

    def update_atlas(self, atlas_rgb):
        # type: (np.ndarray) -> None
        with self._atlas_lock:
            if self._latest_atlas is None or self._latest_atlas.shape != atlas_rgb.shape:
                self._latest_atlas = atlas_rgb.copy()
            else:
                np.copyto(self._latest_atlas, atlas_rgb)

    def get_action(self):
        # type: () -> Optional[Dict]
        """Return current action dict or None if idle / buffer empty / timed out."""
        with self._instruction_lock:
            if not self._instruction:
                return None
            if self._deadline and time.monotonic() > self._deadline:
                expired_inst = self._instruction
                self._instruction = ""
                self._deadline = 0.0
                with self._actions_lock:
                    self._actions.clear()
                    self._current_action = None
                    self._action_frames_left = 0
                print("vla: timeout (%.0fs) — auto-stopped '%s'" % (
                    MAX_INSTRUCTION_SECS, expired_inst[:60]))
                return None

        with self._actions_lock:
            if self._action_frames_left > 0 and self._current_action is not None:
                self._action_frames_left -= 1
                return self._current_action

            if self._actions:
                self._current_action = self._actions.popleft()
                self._action_frames_left = FRAMES_PER_ACTION - 1
                return self._current_action

        return None

    def clear_buffer(self):
        """Discard pending actions (e.g. human took over gamepad)."""
        with self._actions_lock:
            self._actions.clear()
            self._current_action = None
            self._action_frames_left = 0

    def buffer_size(self):
        # type: () -> int
        with self._actions_lock:
            extra = 1 if self._action_frames_left > 0 else 0
            return len(self._actions) + extra

    def is_connected(self):
        # type: () -> bool
        return self._connected

    # ── background thread ─────────────────────────────────────

    def _loop(self):
        while self._running:
            try:
                with self._instruction_lock:
                    instruction = self._instruction
                    deadline = self._deadline

                if not instruction:
                    time.sleep(0.1)
                    continue

                if deadline and time.monotonic() > deadline:
                    time.sleep(0.1)
                    continue

                with self._actions_lock:
                    buf_len = len(self._actions)

                if buf_len <= REFILL_THRESHOLD and not self._requesting:
                    with self._atlas_lock:
                        atlas = self._latest_atlas
                    if atlas is not None:
                        self._request_actions(atlas, instruction)
                    else:
                        time.sleep(0.1)
                else:
                    time.sleep(0.05)
            except Exception as e:
                print("vla: loop error: %s" % e)
                time.sleep(1.0)

    def _request_actions(self, atlas, instruction):
        # type: (np.ndarray, str) -> None
        self._requesting = True
        try:
            _, buf = cv2.imencode(".jpg", atlas[:, :, ::-1],
                                  [cv2.IMWRITE_JPEG_QUALITY, 80])
            img_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

            body = json.dumps({
                "image": img_b64,
                "instruction": instruction,
            }).encode("utf-8")

            req = Request(self._infer_url, data=body,
                          headers={"Content-Type": "application/json"})

            t0 = time.time()
            resp = urlopen(req, timeout=INFER_TIMEOUT)
            result = json.loads(resp.read().decode("utf-8"))
            dt = time.time() - t0

            actions = result.get("actions", [])
            infer_ms = result.get("inference_ms", 0)

            with self._actions_lock:
                self._actions.clear()
                for a in actions:
                    fwd = max(-MAX_FORWARD_MPS, min(MAX_FORWARD_MPS,
                              float(a.get("forward_mps", 0))))
                    ang = max(-MAX_ANGULAR_RDS, min(MAX_ANGULAR_RDS,
                              float(a.get("angular_rads", 0))))
                    self._actions.append({
                        "forward_mps": fwd,
                        "angular_rads": ang,
                    })

            self._connected = True
            self._last_error = ""
            raw_text = result.get("raw_output", "")
            if raw_text:
                print("vla: model said: %s" % raw_text[:200])
            print("vla: %d actions (infer=%.0fms, rtt=%.0fms) first=%s" % (
                len(actions), infer_ms, dt * 1000,
                json.dumps(actions[0]) if actions else "none"))

        except (HTTPError, URLError) as e:
            self._connected = False
            msg = str(e)
            if msg != self._last_error:
                print("vla: server error: %s" % msg)
                self._last_error = msg
        except Exception as e:
            self._connected = False
            msg = str(e)
            if msg != self._last_error:
                print("vla: request error: %s" % msg)
                self._last_error = msg
        finally:
            self._requesting = False

    def _check_health(self):
        try:
            req = Request(self._endpoint + "/health")
            resp = urlopen(req, timeout=3.0)
            data = json.loads(resp.read().decode("utf-8"))
            self._connected = data.get("status") == "ok"
            if self._connected:
                print("vla: server healthy")
            else:
                print("vla: server responded but unhealthy: %s" % data)
        except Exception as e:
            self._connected = False
            print("vla: server not reachable (%s) — will retry on first instruction" % e)

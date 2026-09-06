"""people_live.py – Face greet + name-call / dirs (SPEECH ONLY).

Runs beside HouseBot. Never commands wheels. Uses ~/.kevin/faces gallery
and speech_io (Kokoro + faster-whisper). Goal hints from DirectionalHelp
are logged only — not applied to LocalExecutive.

RGB webcam is frames[0] (Vision rgb1). Upscaled before YuNet for 320x240.
"""

from __future__ import annotations

import os
import sys
import threading
import time

# faces/ lives at repo root alongside src/
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import speech_io

GREET_PERIOD_S = 0.6
LISTEN_PERIOD_S = 12.0
LISTEN_SECS = 2.5
HEARTBEAT_S = 30.0
MAX_PENDING = 4


class PeopleLive:
    """Social layer: RGB face greets + listen for name-call / dirs."""

    # HouseBot checks this to yield the speaker for greetings.
    social_priority = False
    social_hold_until = 0.0

    def __init__(self, vision, enabled=True):
        self.vision = vision
        self.enabled = enabled
        self._stop = False
        self._thread = None
        self._cm = None
        self._pb = None
        self._n_greet = 0
        self._n_hear = 0
        self._n_tick = 0
        self._n_rgb_miss = 0
        self._pending = []
        self._lock = threading.Lock()

    def start(self):
        if not self.enabled:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True, name="people_live")
        self._thread.start()
        print("people_live: started (RGB face greet + name-call; speech only)")

    def stop(self):
        self._stop = True

    def _ensure(self):
        if self._cm is not None:
            return True
        try:
            from faces.recognizer import FaceRecognizer
            from faces.conversation import ConversationManager
            from faces.people_behavior import PeopleBehaviorStub, GreetHours

            rec = FaceRecognizer()
            people = rec.list_people()
            print("people_live: gallery %s" % (people,))

            def _speak(text: str):
                # Never drop a greeting — queue if house_bot is talking.
                self._enqueue(text)

            self._cm = ConversationManager(
                recognizer=rec,
                speak_fn=_speak,
                volume=float(os.environ.get("KEVIN_SPEAK_VOL", "0.30")),
                greet_hours=GreetHours(start_hour=8, end_hour=22),
            )
            # Shorter cooldown so walking past again gets a hello sooner.
            self._cm.cooldown_seconds = 90.0
            self._pb = PeopleBehaviorStub(
                speak_fn=_speak,
                greet_hours=GreetHours(start_hour=8, end_hour=22),
                cooldown_seconds=90.0,
                volume=float(os.environ.get("KEVIN_SPEAK_VOL", "0.30")),
            )
            self._pb.enabled_on_hardware = True
            try:
                speech_io._ensure_kokoro()
                print("people_live: kokoro prewarmed")
            except Exception as e:
                print("people_live: kokoro prewarm skip: %s" % e)
            return True
        except Exception as e:
            print("people_live: init failed: %s" % e)
            return False

    def _enqueue(self, text: str):
        text = (text or "").strip()
        if not text:
            return
        with self._lock:
            PeopleLive.social_priority = True
            if self._pending and self._pending[-1] == text:
                return
            self._pending.append(text)
            if len(self._pending) > MAX_PENDING:
                self._pending = self._pending[-MAX_PENDING:]

    def _flush_pending(self):
        hold = time.monotonic() < float(getattr(PeopleLive, "social_hold_until", 0.0) or 0.0)
        if speech_io.is_speaking():
            PeopleLive.social_priority = True
            return
        with self._lock:
            if not self._pending:
                PeopleLive.social_priority = hold
                return
            text = self._pending.pop(0)
            PeopleLive.social_priority = bool(self._pending)
        try:
            speech_io.speak(text)
            print("people_live: spoke %r" % (text[:80],))
        except Exception as e:
            print("people_live: speak err %s" % e)

    def _rgb(self):
        """Safe copy of Vision RGB webcam (frames[0]), BGR for OpenCV."""
        vis = self.vision
        try:
            read = getattr(vis, "read", None)
            if callable(read):
                frames, _atlas, _ts = read()
                fr = frames[0] if frames else None
            else:
                frames = getattr(vis, "frames", None)
                fr = None if frames is None or len(frames) < 1 else frames[0]
            if fr is None or getattr(fr, "size", 0) == 0:
                return None
            import numpy as np
            import cv2

            img = np.asarray(fr)
            if img.ndim != 3 or img.shape[2] < 3:
                return None
            # Vision stores RGB; recognizer expects BGR.
            bgr = img[:, :, ::-1].copy()
            # 320x240 is tight for YuNet — 2x upsample helps detection.
            h, w = bgr.shape[:2]
            if max(h, w) < 480:
                bgr = cv2.resize(bgr, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
            return bgr
        except Exception as e:
            print("people_live: rgb err %s" % e)
            return None

    def _tick_faces(self):
        img = self._rgb()
        if img is None:
            self._n_rgb_miss += 1
            return
        try:
            out = self._cm.process_frame(img)
            g = out.get("greetings") or []
            u = out.get("unknowns") or []
            faces = out.get("faces") or []
            self._n_tick += 1
            if g or u or faces:
                self._n_greet += 1
                if g or u:
                    # Hold sociable window so we greet while still roughly facing them.
                    PeopleLive.social_priority = True
                    PeopleLive.social_hold_until = time.monotonic() + 2.5
                print(
                    "people_live: greet#%d faces=%d greetings=%s unknowns=%d rgb=%sx%s"
                    % (
                        self._n_greet,
                        len(faces),
                        [x.get("name") for x in g],
                        len(u),
                        img.shape[1],
                        img.shape[0],
                    )
                )
        except Exception as e:
            print("people_live: face tick err %s" % e)

    def _tick_listen(self):
        if speech_io.is_speaking() or PeopleLive.social_priority:
            return
        try:
            text = speech_io.listen_seconds(LISTEN_SECS)
            if not text:
                return
            self._n_hear += 1
            action = self._pb.on_transcript(text)
            if action is None:
                print("people_live: heard#%d %r (no action)" % (self._n_hear, text[:80]))
                return
            hint = action.goal_hint
            print(
                "people_live: heard#%d kind=%s utter=%r hint=%s"
                % (self._n_hear, action.kind, (action.utterance or "")[:80], hint)
            )
        except Exception as e:
            print("people_live: listen tick err %s" % e)

    def _loop(self):
        time.sleep(0.8)  # cameras settle (was 4s — greets came too late)
        if not self._ensure():
            return
        last_face = 0.0
        last_listen = time.monotonic()
        last_hb = 0.0
        while not self._stop:
            now = time.monotonic()
            try:
                self._flush_pending()
                if now - last_face >= GREET_PERIOD_S:
                    last_face = now
                    self._tick_faces()
                if now - last_listen >= LISTEN_PERIOD_S:
                    last_listen = now
                    self._tick_listen()
                if now - last_hb >= HEARTBEAT_S:
                    last_hb = now
                    print(
                        "people_live: heartbeat ticks=%d greets=%d rgb_miss=%d pending=%d"
                        % (self._n_tick, self._n_greet, self._n_rgb_miss, len(self._pending))
                    )
            except Exception as e:
                print("people_live: loop err %s" % e)
            time.sleep(0.25)

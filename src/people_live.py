"""people_live.py – Face greet + name-call / dirs (SPEECH ONLY).

Runs beside HouseBot. Never commands wheels. Uses ~/.kevin/faces gallery
and speech_io (Kokoro + faster-whisper). Goal hints from DirectionalHelp
are logged only — not applied to LocalExecutive.
"""

from __future__ import annotations

import os
import sys
import threading
time_mod = __import__("time")
time = time_mod

# faces/ lives at repo root alongside src/
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import speech_io

GREET_PERIOD_S = 2.5
LISTEN_PERIOD_S = 10.0
LISTEN_SECS = 2.5


class PeopleLive:
    def __init__(self, vision, enabled=True):
        self.vision = vision
        self.enabled = enabled
        self._stop = False
        self._thread = None
        self._cm = None
        self._pb = None
        self._n_greet = 0
        self._n_hear = 0

    def start(self):
        if not self.enabled:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True, name="people_live")
        self._thread.start()
        print("people_live: started (face greet + name-call; speech only)")

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
                if speech_io.is_speaking():
                    return
                speech_io.speak(text)

            self._cm = ConversationManager(
                recognizer=rec,
                speak_fn=_speak,
                volume=float(os.environ.get("KEVIN_SPEAK_VOL", "0.30")),
                greet_hours=GreetHours(start_hour=8, end_hour=22),
            )
            self._pb = PeopleBehaviorStub(
                speak_fn=_speak,
                greet_hours=GreetHours(start_hour=8, end_hour=22),
                cooldown_seconds=300.0,
                volume=float(os.environ.get("KEVIN_SPEAK_VOL", "0.30")),
            )
            self._pb.enabled_on_hardware = True
            return True
        except Exception as e:
            print("people_live: init failed: %s" % e)
            return False

    def _rgb(self):
        vis = self.vision
        try:
            frames = getattr(vis, "frames", None)
            if frames is None or len(frames) < 1:
                return None
            fr = frames[0]
            if fr is None or getattr(fr, "size", 0) == 0:
                return None
            # Vision RGB is typically RGB; OpenCV recognizer expects BGR.
            import numpy as np
            img = np.asarray(fr)
            if img.ndim != 3 or img.shape[2] < 3:
                return None
            return img[:, :, ::-1].copy()
        except Exception:
            return None

    def _tick_faces(self):
        if speech_io.is_speaking():
            return
        img = self._rgb()
        if img is None:
            return
        try:
            out = self._cm.process_frame(img)
            g = out.get("greetings") or []
            u = out.get("unknowns") or []
            if g or u:
                self._n_greet += 1
                print(
                    "people_live: greet#%d faces=%d greetings=%s unknowns=%d"
                    % (
                        self._n_greet,
                        len(out.get("faces") or []),
                        [x.get("name") for x in g],
                        len(u),
                    )
                )
        except Exception as e:
            print("people_live: face tick err %s" % e)

    def _tick_listen(self):
        if speech_io.is_speaking():
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
            # Speech-only: never apply goal_hint to LocalExecutive.
        except Exception as e:
            print("people_live: listen tick err %s" % e)

    def _loop(self):
        time.sleep(5.0)  # let cameras settle
        if not self._ensure():
            return
        last_face = 0.0
        last_listen = time.monotonic()
        while not self._stop:
            now = time.monotonic()
            try:
                if now - last_face >= GREET_PERIOD_S:
                    last_face = now
                    self._tick_faces()
                if now - last_listen >= LISTEN_PERIOD_S:
                    last_listen = now
                    self._tick_listen()
            except Exception as e:
                print("people_live: loop err %s" % e)
            time.sleep(0.35)

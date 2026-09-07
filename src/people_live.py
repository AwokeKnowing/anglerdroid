"""people_live.py – Face greet + social conversation FSM.

Runs beside HouseBot. Implements respectful approach/greet/converse/leave
behavior. Uses ~/.kevin/faces gallery, speech_io (Kokoro + faster-whisper),
and social_fsm for state management.

Goal hints are applied to LocalExecutive when drive is ARMED (~/.kevin/drive_arm).
When disarmed, speech-only behavior continues safely.

RGB webcam is frames[0] (Vision rgb1). Upscaled before YuNet for 320x240.
"""

from __future__ import annotations

import math
import os
import sys
import threading
import time

# faces/ lives at repo root alongside src/
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import local_executive
import speech_io
from social_fsm import SocialFSM

GREET_PERIOD_S = 0.6
LISTEN_PERIOD_S = 12.0
LISTEN_SECS = 2.5
HEARTBEAT_S = 30.0
MAX_PENDING = 4
DRIVE_ARM_FILE = os.path.expanduser("~/.kevin/drive_arm")

def _drive_armed() -> bool:
    """True only when ~/.kevin/drive_arm contents are exactly 'armed'."""
    try:
        with open(DRIVE_ARM_FILE, "r", encoding="utf-8") as f:
            return f.read().strip().lower() == "armed"
    except OSError:
        return False




def _robot_is_moving(speed_eps: float = 0.04) -> bool:
    """True if recent wheel speeds suggest translating or turning."""
    try:
        import tools
        wb = tools.get_wheelbase()
        if wb is not None:
            vl = abs(float(getattr(wb, "_last_sent_left", 0.0) or 0.0))
            vr = abs(float(getattr(wb, "_last_sent_right", 0.0) or 0.0))
            if max(vl, vr) > speed_eps:
                return True
    except Exception:
        pass
    return False


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
        self._fsm = None
        self._n_greet = 0
        self._n_hear = 0
        self._n_tick = 0
        self._n_rgb_miss = 0
        self._pending = []
        self._lock = threading.Lock()
        self._last_goal_apply = 0.0

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
            from faces.people_behavior import create_people_behavior, GreetHours

            # Try InsightFace first (auto-falls back to face_recognition/opencv)
            rec = FaceRecognizer(backend="insightface", model_pack="buffalo_l")
            people = rec.list_people()
            print("people_live: gallery %s backend=%s" % (people, rec.backend))

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
            
            # PeopleBehaviorStub with live enrollment
            self._pb = create_people_behavior(
                speak_fn=_speak,
                recognizer=rec,
                greet_start=8,
                greet_end=22,
                cooldown_seconds=90.0,
                volume=float(os.environ.get("KEVIN_SPEAK_VOL", "0.30")),
                enable_live_enrollment=True,
            )
            self._pb.enabled_on_hardware = True
            
            # Initialize social FSM
            armed = _drive_armed()
            self._fsm = SocialFSM(drive_armed=armed)
            print("people_live: social FSM init drive_armed=%s" % armed)
            try:
                speech_io._ensure_kokoro()
                print("people_live: kokoro prewarmed")
            except Exception as e:
                print("people_live: kokoro prewarm skip: %s" % e)
            return True
        except Exception as e:
            print("people_live: init failed: %s" % e)
            import traceback
            traceback.print_exc()
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

    def _apply_goal_hint(self, hint: dict, person: str = None):
        """Apply goal hint to LocalExecutive when drive is armed."""
        if hint is None:
            return
        
        # Latch file always exists; contents must be "armed".
        armed = _drive_armed()
        if self._fsm is not None:
            self._fsm.drive_armed = armed
        if not armed:
            print("people_live: goal hint %s skipped (drive disarmed)" % hint.get("type"))
            return
        
        now = time.monotonic()
        # Throttle goal updates
        if now - self._last_goal_apply < 0.5:
            return
        self._last_goal_apply = now
        
        hint_type = hint.get("type", "")
        
        try:
            if hint_type == "clear":
                local_executive.clear()
                print("people_live: cleared goals")
            
            elif hint_type == "resume_wander":
                local_executive.set_wander()
                print("people_live: resumed wander")
            
            elif hint_type == "stop":
                local_executive.clear()
                print("people_live: stopped for %s" % (person or "social"))
            
            elif hint_type == "approach_person":
                # Convert bearing and distance to world goal
                pose = getattr(self.vision, "_pose", None)
                if pose is None:
                    print("people_live: no pose for approach")
                    return
                bearing_deg = hint.get("bearing_deg", 0.0)
                target_dist = hint.get("target_distance_m", 1.35)
                
                # Bearing is relative to current heading
                target_theta = pose.theta + math.radians(bearing_deg)
                gx = pose.x + target_dist * math.cos(target_theta)
                gy = pose.y + target_dist * math.sin(target_theta)
                
                local_executive.set_goal_xy(gx, gy)
                print("people_live: approach %s at bearing %.0f° dist %.1fm" % (
                    person or "person", bearing_deg, target_dist))
            
            elif hint_type == "relative_bearing":
                # Directional help (kitchen, left, etc.)
                pose = getattr(self.vision, "_pose", None)
                if pose is None:
                    return
                bearing_deg = hint.get("bearing_deg", 0.0)
                distance_m = hint.get("distance_m", 1.0)
                
                target_theta = pose.theta + math.radians(bearing_deg)
                gx = pose.x + distance_m * math.cos(target_theta)
                gy = pose.y + distance_m * math.sin(target_theta)
                
                local_executive.set_goal_xy(gx, gy)
                label = hint.get("label", "target")
                print("people_live: relative_bearing %s %.0f° %.1fm" % (
                    label, bearing_deg, distance_m))
            
            elif hint_type in ("approach_speaker", "turn_to_speaker"):
                # Come-here / look-at: small step toward current heading (speaker assumed ahead).
                pose = getattr(self.vision, "_pose", None)
                if pose is None:
                    print("people_live: no pose for %s" % hint_type)
                    return
                distance_m = 0.6 if hint_type == "turn_to_speaker" else float(
                    hint.get("target_distance_m", 1.35))
                gx = pose.x + distance_m * math.cos(pose.theta)
                gy = pose.y + distance_m * math.sin(pose.theta)
                local_executive.set_goal_xy(gx, gy)
                print("people_live: %s dist %.1fm" % (hint_type, distance_m))

            elif hint_type == "retreat":
                # Back away
                pose = getattr(self.vision, "_pose", None)
                if pose is None:
                    return
                distance_m = hint.get("distance_m", 2.0)
                # Move backwards
                gx = pose.x - distance_m * math.cos(pose.theta)
                gy = pose.y - distance_m * math.sin(pose.theta)
                
                local_executive.set_goal_xy(gx, gy)
                print("people_live: retreat %.1fm" % distance_m)
            
        except Exception as e:
            print("people_live: goal apply err %s" % e)

    def _tick_faces(self):
        img = self._rgb()
        if img is None:
            self._n_rgb_miss += 1
            return
        try:
            # Get depth map if available (for distance estimation)
            depth_map = None
            try:
                # Vision may have depth from RGB-D camera
                if hasattr(self.vision, "frames") and len(self.vision.frames) > 0:
                    # For now, use box heuristic; depth integration is TODO
                    pass
            except Exception:
                pass
            
            # Recognize with threshold + margin (unknown = rejected by accept gate)
            faces = self._cm.recognizer.recognize(img, threshold=0.60, margin=0.05, log_scores=False)
            unknowns = [r for r in faces if r[0] == "unknown"]
            
            self._n_tick += 1
            
            # Keep FSM drive_armed latch fresh
            if self._fsm is not None:
                self._fsm.drive_armed = _drive_armed()
            
            now = time.monotonic()
            
            # Handle known faces via FSM (approach/greet/leave)
            fsm_actions = []
            for name, confidence, box in faces:
                if name != "unknown" and self._fsm is not None:
                    action = self._fsm.on_face_seen(name, confidence, box, now, depth_map)
                    if action:
                        fsm_actions.append(action)
            
            # Apply FSM actions (known people)
            for action in fsm_actions:
                action_type = action.get("action_type")
                utterance = action.get("utterance", "")
                goal_hint = action.get("goal_hint")
                person = action.get("person")
                
                if utterance:
                    self._enqueue(utterance)
                
                if goal_hint:
                    self._apply_goal_hint(goal_hint, person)
                
                if action_type in ("greet", "approach"):
                    self._n_greet += 1
                    PeopleLive.social_priority = True
                    PeopleLive.social_hold_until = now + 2.5
            
            # Handle unknown faces with live enrollment (PeopleBehaviorStub)
            # Never start/ask while driving or turning — face the person first, hold still.
            enrollment_actions = []
            moving = _robot_is_moving()
            if self._pb is not None and not moving:
                # Get landmarks for InsightFace if available
                try:
                    detections = self._cm.recognizer.detect_faces_with_landmarks(img)
                except Exception:
                    detections = [(box, None) for _, _, box in faces]
                
                # Match faces to detections by box similarity
                for name, confidence, box in faces:
                    # Find matching detection with landmarks
                    landmarks = None
                    for det_box, det_lm in detections:
                        # Rough box match (within 20px)
                        if abs(det_box[0] - box[0]) < 20 and abs(det_box[1] - box[1]) < 20:
                            landmarks = det_lm
                            break
                    
                    # Feed all faces (unknown and known) to enrollment manager
                    action = self._pb.on_face_seen(
                        name=name,
                        confidence=confidence,
                        box=box,
                        image=img,
                        landmarks=landmarks,
                        now=now,
                        speak=True
                    )
                    
                    if action and action.kind in ("enrollment_prompt", "enrollment_collecting",
                                                   "enrollment_complete", "enrollment_timeout",
                                                   "enrollment_name_received", "enrollment_failed"):
                        enrollment_actions.append(action)
                        print("people_live: enrollment %s for box=%s" % (action.kind, box))
                        if action.kind == "enrollment_prompt":
                            PeopleLive.social_priority = True
                            PeopleLive.social_hold_until = now + 18.0
                            try:
                                import local_executive
                                local_executive.cancel()
                            except Exception:
                                pass
                        elif action.kind in ("enrollment_complete", "enrollment_timeout", "enrollment_failed"):
                            PeopleLive.social_hold_until = now + 1.0

            if faces or fsm_actions or enrollment_actions:
                ids = ["%s:%.2f" % (n, c) for n, c, _b in faces]
                print(
                    "people_live: tick#%d faces=%d fsm_actions=%d enrollment_actions=%d unknowns=%d ids=%s rgb=%sx%s"
                    % (
                        self._n_tick,
                        len(faces),
                        len(fsm_actions),
                        len(enrollment_actions),
                        len(unknowns),
                        ids,
                        img.shape[1],
                        img.shape[0],
                    )
                )
        except Exception as e:
            print("people_live: face tick err %s" % e)
            import traceback
            traceback.print_exc()

    def _tick_listen(self):
        if speech_io.is_speaking() or PeopleLive.social_priority:
            return
        try:
            text = speech_io.listen_seconds(LISTEN_SECS)
            if not text:
                return
            self._n_hear += 1
            
            now = time.monotonic()
            
            # Check if enrollment session is active (handles name responses)
            if self._pb and self._pb.enrollment_manager and self._pb.enrollment_manager.is_session_active():
                session = self._pb.enrollment_manager.active_session
                if session.person_name is None:
                    # Waiting for name — feed transcript to enrollment manager
                    action = self._pb.on_transcript(text, now=now, speak=True, language="en")
                    if action and action.kind == "enrollment_name_received":
                        print("people_live: enrollment name received: %s" % action.meta.get("name"))
                        return
            
            # Wake/commands first (Designing for Exit: "go away" must win over empathy).
            action = self._pb.on_transcript(text, now=now, speak=True, language="en")
            if action is not None:
                hint = action.goal_hint
                kind = action.kind
                utterance = action.utterance or ""
                if hint and kind in ("command", "directional_help"):
                    self._apply_goal_hint(hint)
                # If dismissed, tell FSM to leave
                if kind == "command" and (hint or {}).get("dismissed") and self._fsm:
                    if self._fsm.current_person:
                        leave = self._fsm.on_dismiss(
                            self._fsm.current_person, now) if hasattr(self._fsm, "on_dismiss") else None
                        if leave and leave.get("utterance"):
                            # command ack already spoken by stub; skip duplicate
                            pass
                print(
                    "people_live: heard#%d kind=%s utter=%r hint=%s"
                    % (self._n_hear, kind, utterance[:80], hint)
                )
                return

            # Engagement / empathy while in WAIT_ENGAGE or CONVERSE
            if self._fsm and self._fsm.current_person:
                fsm_action = self._fsm.on_speech_heard(
                    self._fsm.current_person, text, now)
                if fsm_action:
                    utterance = fsm_action.get("utterance", "")
                    if utterance:
                        self._enqueue(utterance)
                    print("people_live: FSM heard#%d empathy=%r" % (
                        self._n_hear, utterance[:80] if utterance else "(silent)"))
                    return

            print("people_live: heard#%d %r (no action)" % (self._n_hear, text[:80]))
        except Exception as e:
            print("people_live: listen tick err %s" % e)
            import traceback
            traceback.print_exc()

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

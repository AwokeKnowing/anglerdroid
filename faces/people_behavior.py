"""Offline people-behavior stubs for Kevin (no live drive).

Provides:
  - GreetHours: only greet during configured local hours
  - NameCallResponder: react when someone says Kevin's name
  - DirectionalHelp: map simple help / direction asks to utterances
    (+ optional goal_hint dict for a future mid-layer — not wired)
  - LiveEnrollmentIntegration: handle unknown faces with interactive enrollment

These are code-only stubs. Do not enable on hardware until re-arm criteria
are met and the user asks.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

try:
    from faces.live_enrollment import LiveEnrollmentManager
    _HAS_LIVE_ENROLLMENT = True
except ImportError:
    _HAS_LIVE_ENROLLMENT = False
    LiveEnrollmentManager = None


DEFAULT_GREET_START = 8   # inclusive local hour
DEFAULT_GREET_END = 22    # exclusive local hour


@dataclass(frozen=True)
class GreetHours:
    """Window when proactive face greetings are allowed."""

    start_hour: int = DEFAULT_GREET_START
    end_hour: int = DEFAULT_GREET_END

    def __post_init__(self):
        if not (0 <= self.start_hour <= 23 and 0 <= self.end_hour <= 24):
            raise ValueError("hours must be in 0..24")
        if self.start_hour == self.end_hour:
            raise ValueError("start_hour and end_hour must differ")

    def allows(self, hour: Optional[int] = None) -> bool:
        """True if ``hour`` (local 0-23) is inside the greet window.

        Windows that wrap midnight (e.g. 22→8) are supported.
        """
        if hour is None:
            hour = time.localtime().tm_hour
        hour = int(hour) % 24
        if self.start_hour < self.end_hour:
            return self.start_hour <= hour < self.end_hour
        # wraps midnight
        return hour >= self.start_hour or hour < self.end_hour


@dataclass
class PeopleAction:
    """One social action the stub wants to take (speak only for now)."""

    kind: str  # greet | name_call | directional_help | unknown
    utterance: str
    goal_hint: Optional[Dict] = None  # never applied to drive from this module
    meta: Dict = field(default_factory=dict)


_WAKE_RE = re.compile(
    r"\b(?:hey\s+|ok\s+|okay\s+|yo\s+)?kevin\b",
    re.IGNORECASE,
)

_HELP_RE = re.compile(
    r"\b(?:help|where(?:'s| is)|how do i (?:get|find)|point me|which way|"
    r"take me|show me|directions?(?:\s+to)?)\b",
    re.IGNORECASE,
)

# Simple household landmarks → relative hint for later mid-layer (stub only).
_LANDMARKS: Dict[str, Dict] = {
    "kitchen": {"bearing_deg": 90.0, "label": "kitchen"},
    "living room": {"bearing_deg": 0.0, "label": "living room"},
    "livingroom": {"bearing_deg": 0.0, "label": "living room"},
    "bathroom": {"bearing_deg": -90.0, "label": "bathroom"},
    "bedroom": {"bearing_deg": 180.0, "label": "bedroom"},
    "front door": {"bearing_deg": 45.0, "label": "front door"},
    "door": {"bearing_deg": 45.0, "label": "door"},
    "couch": {"bearing_deg": -30.0, "label": "couch"},
    "charger": {"bearing_deg": 135.0, "label": "charger"},
    "plug": {"bearing_deg": 135.0, "label": "charger"},
}

_DIR_WORDS: Dict[str, float] = {
    "left": -90.0,
    "right": 90.0,
    "ahead": 0.0,
    "forward": 0.0,
    "straight": 0.0,
    "behind": 180.0,
    "back": 180.0,
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


class NameCallResponder:
    """Detect Kevin name-calls in ASR text and craft a short reply."""

    ACKS = (
        "Yes?",
        "Hey — I'm here.",
        "Listening.",
        "Kevin here.",
    )

    def match(self, transcript: str) -> Optional[str]:
        """Return wake payload (remainder after name) or None if no name-call."""
        text = _normalize(transcript)
        if not text:
            return None
        m = _WAKE_RE.search(text)
        if not m:
            return None
        remainder = text[m.end():].strip(" .,!?:;")
        return remainder

    def respond(self, transcript: str, ack_index: int = 0) -> Optional[PeopleAction]:
        rem = self.match(transcript)
        if rem is None:
            return None
        # If the remainder looks like a help ask, leave it for DirectionalHelp.
        if rem and _HELP_RE.search(rem):
            return PeopleAction(
                kind="name_call",
                utterance=self.ACKS[ack_index % len(self.ACKS)],
                meta={"remainder": rem, "defer_help": True},
            )
        if not rem:
            return PeopleAction(
                kind="name_call",
                utterance=self.ACKS[ack_index % len(self.ACKS)],
                meta={"remainder": ""},
            )
        return PeopleAction(
            kind="name_call",
            utterance=f"You said: {rem}. I heard you.",
            meta={"remainder": rem},
        )


class DirectionalHelp:
    """Parse simple directional / where-is asks. Returns speech + optional hint."""

    def parse_landmark(self, text: str) -> Optional[Tuple[str, Dict]]:
        norm = _normalize(text)
        # Longer keys first so "living room" beats "room" if we add room later.
        for key in sorted(_LANDMARKS.keys(), key=len, reverse=True):
            if key in norm:
                return key, dict(_LANDMARKS[key])
        return None

    def parse_direction(self, text: str) -> Optional[Tuple[str, float]]:
        norm = _normalize(text)
        for word, bearing in _DIR_WORDS.items():
            if re.search(rf"\b{re.escape(word)}\b", norm):
                return word, bearing
        return None

    def wants_help(self, transcript: str) -> bool:
        return bool(_HELP_RE.search(_normalize(transcript)))

    def parse_command(self, transcript: str) -> Optional[Tuple[str, Dict]]:
        """Parse Hey Kevin xyz commands beyond directional help.
        
        Returns:
            (command_type, goal_hint) or None if not a command
        """
        norm = _normalize(transcript)
        
        # Stop/halt commands
        if re.search(r"\b(stop|halt|freeze|hold|stay)\b", norm):
            return ("stop", {"type": "clear"})
        
        # Come here commands
        if re.search(r"\b(come (here|to me|over|closer)|approach)\b", norm):
            return ("come_here", {"type": "approach_speaker"})
        
        # Go away / retreat commands (Designing for Exit: immediate response)
        if re.search(r"\b(go away|leave me|back up|give (me )?space|retreat|not now|i'?m busy)\b", norm):
            return ("go_away", {"type": "retreat", "distance_m": 2.0, "dismissed": True})
        
        # Wander / explore commands
        if re.search(r"\b(wander|explore|look around|roam)\b", norm):
            return ("wander", {"type": "resume_wander"})
        
        # Look at / turn toward commands
        if re.search(r"\b(look at|turn (to|toward)|face)\b", norm):
            return ("look_at", {"type": "turn_to_speaker"})
        
        return None

    def respond(self, transcript: str) -> Optional[PeopleAction]:
        text = _normalize(transcript)
        if not text:
            return None
        # Strip leading name-call so "kevin where is the kitchen" still parses.
        text = _WAKE_RE.sub(" ", text)
        text = _normalize(text)
        if not (self.wants_help(text) or self.parse_landmark(text) or self.parse_direction(text)):
            return None

        lm = self.parse_landmark(text)
        if lm is not None:
            key, hint = lm
            label = hint["label"]
            bearing = float(hint["bearing_deg"])
            side = "ahead"
            if -135 <= bearing < -45:
                side = "on your left"
            elif 45 < bearing <= 135:
                side = "on your right"
            elif abs(bearing) >= 135:
                side = "behind you"
            utterance = f"The {label} is roughly {side}."
            return PeopleAction(
                kind="directional_help",
                utterance=utterance,
                goal_hint={
                    "type": "relative_bearing",
                    "bearing_deg": bearing,
                    "label": label,
                    "distance_m": 1.0,
                },
                meta={"landmark": key},
            )

        d = self.parse_direction(text)
        if d is not None:
            word, bearing = d
            utterance = f"Try looking {word}."
            return PeopleAction(
                kind="directional_help",
                utterance=utterance,
                goal_hint={
                    "type": "relative_bearing",
                    "bearing_deg": bearing,
                    "label": word,
                    "distance_m": 0.8,
                },
                meta={"direction": word},
            )

        return PeopleAction(
            kind="directional_help",
            utterance="I can point toward the kitchen, bathroom, bedroom, or front door.",
            goal_hint=None,
            meta={"fallback": True},
        )


class PeopleBehaviorStub:
    """Compose greet-hours + name-call + directional help + live enrollment. Speak only; no drive."""

    def __init__(
        self,
        speak_fn: Optional[Callable[[str], None]] = None,
        greet_hours: Optional[GreetHours] = None,
        cooldown_seconds: float = 300.0,
        volume: float = 0.1,
        recognizer = None,
        enable_live_enrollment: bool = True,
        enrollment_confidence_threshold: float = 0.90,
    ):
        self.speak_fn = speak_fn if speak_fn is not None else self._stub_speak
        self.greet_hours = greet_hours or GreetHours()
        self.cooldown_seconds = float(cooldown_seconds)
        self.volume = float(volume)
        self.name_call = NameCallResponder()
        self.directional = DirectionalHelp()
        self._last_greet: Dict[str, float] = {}
        self._last_name_call = 0.0
        self.enabled_on_hardware = False  # hard flag — never auto-arm
        
        # Live enrollment manager
        self.enrollment_manager = None
        if enable_live_enrollment and recognizer is not None and _HAS_LIVE_ENROLLMENT:
            self.enrollment_manager = LiveEnrollmentManager(
                recognizer=recognizer,
                speak_fn=self.speak_fn,
                confidence_threshold=enrollment_confidence_threshold,
                min_samples=3,
                max_samples=5,
                session_timeout=15.0,
                cooldown_seconds=cooldown_seconds,
                language="en"
            )
        
    def _stub_speak(self, text: str) -> None:
        print(f"🔊 [people @ {self.volume:.0%}]: {text}")

    def _speak(self, text: str) -> None:
        if text:
            self.speak_fn(text)

    def can_greet_now(self, hour: Optional[int] = None) -> bool:
        return self.greet_hours.allows(hour)

    def on_face_seen(
        self,
        name: str,
        confidence: float,
        box: Tuple[int, int, int, int],
        image = None,
        landmarks = None,
        *,
        hour: Optional[int] = None,
        now: Optional[float] = None,
        speak: bool = True,
    ) -> Optional[PeopleAction]:
        """Handle face detection with recognition result.
        
        If confidence is low (<90% by default), triggers interactive enrollment.
        If recognized with high confidence, greets if appropriate.
        
        Args:
            name: Recognized name or "unknown"
            confidence: Recognition confidence (0.0-1.0)
            box: Face bounding box (x, y, w, h)
            image: Optional BGR image for enrollment samples
            landmarks: Optional 5-point landmarks
            hour: Optional hour override
            now: Optional time override
            speak: Whether to speak greetings/prompts
        
        Returns:
            PeopleAction or None
        """
        t = time.time() if now is None else float(now)
        
        # Check if enrollment session is active
        if self.enrollment_manager is not None and self.enrollment_manager.is_session_active():
            session = self.enrollment_manager.active_session
            
            # Check timeout
            timed_out, leave_msg = self.enrollment_manager.check_timeout(now=t)
            if timed_out:
                return PeopleAction(
                    kind="enrollment_timeout",
                    utterance=leave_msg,
                    meta={"reason": "no_name_received"}
                )
            
            # Collecting samples?
            if session.person_name is not None and image is not None:
                collected, n_samples, n_needed = self.enrollment_manager.collect_sample(image, box, landmarks)
                
                if collected:
                    # Check if we have enough samples
                    if session.is_complete():
                        success, msg = self.enrollment_manager.finalize_enrollment()
                        if success:
                            return PeopleAction(
                                kind="enrollment_complete",
                                utterance=msg,
                                meta={"name": session.person_name, "samples": n_samples}
                            )
                        else:
                            return PeopleAction(
                                kind="enrollment_failed",
                                utterance=msg,
                                meta={"name": session.person_name}
                            )
                
                # Still collecting
                return PeopleAction(
                    kind="enrollment_collecting",
                    utterance="",
                    meta={"samples": n_samples, "needed": n_needed}
                )
            
            # Waiting for name from ASR (handled in on_transcript)
            return None
        
        # Unknown or low-confidence face?
        if name == "unknown" or confidence < (self.enrollment_manager.confidence_threshold if self.enrollment_manager else 0.90):
            if self.enrollment_manager is not None:
                # Should we start enrollment?
                if self.enrollment_manager.should_enroll(confidence, box, now=t):
                    prompt = self.enrollment_manager.start_session(box, language="en", now=t)
                    if speak:
                        pass  # Already spoken in start_session
                    return PeopleAction(
                        kind="enrollment_prompt",
                        utterance=prompt,
                        meta={"confidence": confidence, "box": box}
                    )
            
            # Don't pretend to know them, just log
            return PeopleAction(
                kind="face_unknown",
                utterance="",
                meta={"confidence": confidence, "threshold_failed": True}
            )
        
        # High-confidence recognition: proceed with greeting
        if not self.can_greet_now(hour):
            return PeopleAction(
                kind="greet",
                utterance="",
                meta={"skipped": "outside_greet_hours", "name": name},
            )
        
        last = self._last_greet.get(name, 0.0)
        if t - last < self.cooldown_seconds:
            return PeopleAction(
                kind="greet",
                utterance="",
                meta={"skipped": "cooldown", "name": name},
            )
        
        # Lightweight time-of-day phrase (mirrors conversation.py buckets).
        h = time.localtime().tm_hour if hour is None else int(hour)
        if 5 <= h < 12:
            utter = f"Good morning, {name}!"
        elif 12 <= h < 18:
            utter = f"Hi {name}!"
        else:
            utter = f"Good evening, {name}!"
        
        action = PeopleAction(kind="greet", utterance=utter, meta={"name": name, "confidence": confidence})
        self._last_greet[name] = t
        if speak and utter:
            self._speak(utter)
        return action

    def on_transcript(
        self,
        transcript: str,
        *,
        now: Optional[float] = None,
        speak: bool = True,
        language: str = "en",
    ) -> Optional[PeopleAction]:
        """Handle name-call and/or directional help from ASR text.
        
        Also handles name responses during enrollment sessions.
        
        Args:
            transcript: ASR text
            now: Optional time override
            speak: Whether to speak responses
            language: Detected language ("en" or "es")
        """
        text = (transcript or "").strip()
        if not text:
            return None
        t = time.time() if now is None else float(now)

        # Check if enrollment session is waiting for name
        if self.enrollment_manager is not None and self.enrollment_manager.is_session_active():
            session = self.enrollment_manager.active_session
            if session.person_name is None:
                # Extract name from transcript
                # Simple heuristic: remove common prefixes
                name = text
                for prefix in ["my name is", "i'm", "i am", "call me", "it's", "this is",
                              "me llamo", "soy", "mi nombre es"]:
                    if name.lower().startswith(prefix):
                        name = name[len(prefix):].strip()
                        break
                
                # Clean name (remove trailing punctuation, etc)
                name = re.sub(r'[^\w\s-]', '', name).strip()
                
                if name:
                    # Acknowledge name and prepare to collect samples
                    ack = self.enrollment_manager.on_name_received(name, language)
                    return PeopleAction(
                        kind="enrollment_name_received",
                        utterance=ack,
                        meta={"name": name, "language": language}
                    )

        name_action = self.name_call.respond(text)
        help_action = self.directional.respond(text)
        
        # Check for command (stop, come here, go away, wander, etc.)
        cmd = self.directional.parse_command(text)
        if cmd is not None:
            cmd_type, goal_hint = cmd
            # Commands should acknowledge (curious, earnest, playful tone)
            ack = {
                "stop": "Stopping!",
                "come_here": "Coming over!",
                "go_away": "Sorry! Moving away.",
                "wander": "Back to wandering!",
                "look_at": "Looking!",
            }.get(cmd_type, "Okay!")
            
            self._last_name_call = t
            if speak:
                self._speak(ack)
            return PeopleAction(
                kind="command",
                utterance=ack,
                goal_hint=goal_hint,
                meta={"command_type": cmd_type},
            )

        # Prefer directional help when both match (e.g. "Kevin, where's the kitchen?").
        if help_action is not None and (
            name_action is None or name_action.meta.get("defer_help") or help_action.kind == "directional_help"
        ):
            # Still require name-call OR an explicit help phrase.
            if name_action is not None or self.directional.wants_help(text):
                if t - self._last_name_call < 2.0 and name_action is not None:
                    # debounce rapid double fires after a wake
                    pass
                self._last_name_call = t
                if speak:
                    self._speak(help_action.utterance)
                return help_action

        if name_action is not None:
            if t - self._last_name_call < self.cooldown_seconds * 0.02:  # ~6s default
                return PeopleAction(
                    kind="name_call",
                    utterance="",
                    meta={"skipped": "cooldown"},
                )
            self._last_name_call = t
            if speak:
                self._speak(name_action.utterance)
            return name_action

        return None

    def reset_cooldowns(self) -> None:
        self._last_greet.clear()
        self._last_name_call = 0.0


def create_people_behavior(
    speak_fn: Optional[Callable[[str], None]] = None,
    greet_start: int = DEFAULT_GREET_START,
    greet_end: int = DEFAULT_GREET_END,
    cooldown_seconds: float = 300.0,
    volume: float = 0.1,
    recognizer = None,
    enable_live_enrollment: bool = True,
    enrollment_confidence_threshold: float = 0.90,
) -> PeopleBehaviorStub:
    """Factory for offline demos / tests. Does not touch main.py or hardware.
    
    Args:
        speak_fn: Speech function
        greet_start: Greet window start hour (0-23)
        greet_end: Greet window end hour (0-24)
        cooldown_seconds: Seconds between greets/prompts
        volume: Speech volume (0.0-1.0)
        recognizer: FaceRecognizer instance for live enrollment
        enable_live_enrollment: Enable interactive enrollment for unknown faces
        enrollment_confidence_threshold: User-facing confidence bar (0.90 = 90%)
    """
    return PeopleBehaviorStub(
        speak_fn=speak_fn,
        greet_hours=GreetHours(start_hour=greet_start, end_hour=greet_end),
        cooldown_seconds=cooldown_seconds,
        volume=volume,
        recognizer=recognizer,
        enable_live_enrollment=enable_live_enrollment,
        enrollment_confidence_threshold=enrollment_confidence_threshold,
    )

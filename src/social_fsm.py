"""social_fsm.py – Kevin's social conversation state machine.

Respectful companion behavior: approach → greet → listen → converse or leave.
Follows Hall's proxemics (conversational distance ~1.2–1.5 m / 4–5 ft).
Never spam, hover, or re-greet the same person without cooldown.

FSM States:
  IDLE_WANDER – wandering; no person noticed
  NOTICE      – person seen; decide if approach is warranted
  APPROACH    – moving toward person to conversational distance
  GREET       – stopped at distance; short greeting
  WAIT_ENGAGE – listening for engagement (15s timeout)
  CONVERSE    – active conversation; short empathetic turns
  LEAVE       – politely disengaging; resume wander

Safety: all goals are hints to LocalExecutive. SafetyGuard/keepouts remain active.
Drive may be DISARMED (~/.kevin/drive_arm). When disarmed, skip approach goals.
"""

from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

# Distance thresholds (meters)
CONVERSATIONAL_DIST_MIN = 1.2  # Hall's personal/conversational start
CONVERSATIONAL_DIST_MAX = 1.5  # comfort zone upper bound
INTIMATE_DIST = 0.5            # too close; do NOT enter unless invited
APPROACH_STOP_DIST = 1.35      # target stop distance (middle of range)

# Timing (seconds)
WAIT_ENGAGE_TIMEOUT = 15.0     # legacy alias; unengaged leave uses prompts
CONVERSE_BUDGET_S = 120.0      # max chat if they keep talking
CONVERSE_PROMPT_S = 12.0       # no reply this long → wrap and wander
CONVERSE_PROMPT_MAX = 0        # do not nag; one greeting is the try
CONVERSE_GONE_S = 25.0         # face lost this long → wrap up
CONVERSE_TURN_MAX = 8.0        # keep conversational turns short
COOLDOWN_NO_ENGAGE = 300.0     # 5 min after leave-without-chat
COOLDOWN_ENGAGED = 90.0        # 1.5 min after conversation
COOLDOWN_DISMISSED = 600.0     # 10 min after explicit "go away" (2x normal)
NOTICE_DECIDE_DELAY = 0.5      # brief observation before approach (Kendon initiation phase)

# Face box heuristic calibration (when depth unavailable)
# Assume typical face width ~20 cm at ~4 ft (1.2 m) → box_width ~X pixels at 320px wide
# These are rough; prefer depth when available.
FACE_BOX_CLOSE_PX = 80         # box width > this → closer than conversational
FACE_BOX_FAR_PX = 30           # box width < this → farther than conversational
FACE_BOX_GOOD_MIN = 40         # 40–70 px width ≈ 4–5 ft ballpark
FACE_BOX_GOOD_MAX = 70


class SocialState(Enum):
    """FSM states for social interaction."""
    IDLE_WANDER = auto()
    NOTICE = auto()
    APPROACH = auto()
    GREET = auto()
    WAIT_ENGAGE = auto()
    CONVERSE = auto()
    LEAVE = auto()


@dataclass
class PersonEncounter:
    """Tracks one person's interaction state."""
    name: str
    state: SocialState = SocialState.IDLE_WANDER
    notice_time: float = 0.0
    approach_time: float = 0.0
    greet_time: float = 0.0
    last_heard_time: float = 0.0
    last_spoke_time: float = 0.0
    last_seen_time: float = 0.0
    converse_t0: float = 0.0
    last_prompt_time: float = 0.0
    prompt_count: int = 0
    engage_count: int = 0
    converse_turns: int = 0
    estimated_distance: float = 999.0
    goal_hint: Optional[Dict] = None
    
    # Cooldown tracking
    cooldown_until: float = 0.0
    engaged_this_session: bool = False


# Spanish-preferred names (from conversation.py)
SPANISH_NAMES = {"nohemi", "karina"}

# Kevin's personality: curious, calm, earnest helper (Hero / Astro Boy)
# Clean wholesome humor: puns, gentle self-deprecation, playful observations
# Keep encouraging/empathizing; jokes secondary to kindness

# Empathy/encouragement templates (English)
# Keep short (1-2 sentences), earnest, optionally playful
EMPATHY_TEMPLATES = [
    "I hear you!",
    "That makes sense to me.",
    "You're doing great!",
    "I'm learning too. Keep going!",
    "That sounds tricky.",
    "I'm here if you need me!",
    "You've got this!",
    "That's really interesting!",
    "I think I understand.",
    "Hang in there, friend.",
]

EMPATHY_TOUGH = [
    "That sounds hard. You're doing your best!",
    "I'm still learning, but I think you're brave.",
    "Tough day? I'm here.",
    "Keep going. I believe in you!",
]

EMPATHY_HAPPY = [
    "That's wonderful! I'm happy for you!",
    "That sounds great!",
    "I love hearing good news!",
    "That made my sensors warm. Keep it up!",
]

# Spanish empathy templates (same earnest, gentle tone)
EMPATHY_TEMPLATES_ES = [
    "¡Te escucho!",
    "Tiene sentido para mí.",
    "¡Lo estás haciendo muy bien!",
    "Yo también estoy aprendiendo. ¡Sigue así!",
    "Eso suena difícil.",
    "¡Estoy aquí si me necesitas!",
    "¡Tú puedes!",
    "¡Qué interesante!",
    "Creo que entiendo.",
    "Ten paciencia, amigo.",
]

EMPATHY_TOUGH_ES = [
    "Eso suena difícil. ¡Lo estás haciendo bien!",
    "Todavía estoy aprendiendo, pero creo que eres valiente.",
    "¿Día difícil? Estoy aquí.",
    "Sigue adelante. ¡Creo en ti!",
]

EMPATHY_HAPPY_ES = [
    "¡Qué maravilla! ¡Me alegro por ti!",
    "¡Eso suena genial!",
    "¡Me encanta escuchar buenas noticias!",
    "Eso me calentó los sensores. ¡Sigue así!",
]

# Leave messages (polite + optional tiny clean joke, never guilt-trip)
LEAVE_MESSAGES = [
    "",  # silent leave is fine
    "I'll let you get back to it!",
    "Catch you later!",
    "I'll keep wandering. Call if you need me!",
    "Bye for now!",
    "Off to explore. My wheels are excited!",
]

LEAVE_MESSAGES_ES = [
    "",  # silent leave
    "¡Te dejo trabajar!",
    "¡Nos vemos!",
    "Voy a seguir explorando. ¡Llámame si me necesitas!",
    "¡Hasta luego!",
    "A explorar. ¡Mis ruedas están emocionadas!",
]

# After the greeting, if they stay quiet, try once or twice more.
CHAT_PROMPTS = [
    "What's been going on?",
    "I'm listening if you want to talk.",
    "Still there? I can hang for a couple minutes.",
]
CHAT_PROMPTS_ES = [
    "¿Qué has andado haciendo?",
    "Te escucho si quieres platicar.",
    "¿Sigues ahí? Puedo quedarme un par de minutos.",
]
WRAP_MESSAGES = [
    "That was a good couple of minutes. I'll wander — call if you need me!",
    "Good talking with you. I'll keep exploring!",
]
WRAP_MESSAGES_ES = [
    "Fueron dos minutos ricos. Voy a explorar. ¡Llámame si me necesitas!",
    "¡Qué gusto platicar! Sigo explorando.",
]


class SocialFSM:
    """Manages social conversation FSM for one person at a time.
    
    Not thread-safe; call from the people_live main loop.
    """
    
    def __init__(self, drive_armed: bool = False):
        """Initialize social FSM.
        
        Args:
            drive_armed: if False, skip approach goals (speech-only)
        """
        self.drive_armed = drive_armed
        self.encounters: Dict[str, PersonEncounter] = {}
        self.current_person: Optional[str] = None
        self._command_cooldown: float = 0.0

    def on_timer(self, now: float) -> Optional[Dict]:
        """Advance converse timers even if the face is off-camera this tick."""
        person = self.current_person
        if not person:
            return None
        enc = self.encounters.get(person)
        if enc is None:
            return None
        if enc.state == SocialState.GREET and now - enc.greet_time >= 0.5:
            self._begin_converse(enc, now)
        if enc.state == SocialState.WAIT_ENGAGE:
            self._begin_converse(enc, now)
        if enc.state == SocialState.CONVERSE:
            return self._converse_tick(person, enc, now)
        return None
        

    def on_dismiss(self, name: str, now: float) -> Optional[Dict]:
        """Immediate leave after explicit go-away / not-now (Designing for Exit)."""
        enc = self._get_or_create(name)
        enc.state = SocialState.LEAVE
        enc.engaged_this_session = False
        self._start_cooldown(name, COOLDOWN_DISMISSED, now)
        prefer_spanish = name.strip().lower() in SPANISH_NAMES
        leave_bank = LEAVE_MESSAGES_ES if prefer_spanish else LEAVE_MESSAGES
        utterance = random.choice(leave_bank)
        hint = {"type": "retreat", "distance_m": 2.0, "dismissed": True} if self.drive_armed else None
        enc.state = SocialState.IDLE_WANDER
        enc.goal_hint = {"type": "resume_wander"} if self.drive_armed else None
        self.current_person = None
        return {
            "action_type": "leave",
            "person": name,
            "utterance": utterance,
            "goal_hint": hint or enc.goal_hint,
        }

    def is_on_cooldown(self, name: str, now: float) -> bool:
        """Check if person is on cooldown."""
        enc = self.encounters.get(name)
        if enc is None:
            return False
        return now < enc.cooldown_until

    def in_conversation(self) -> bool:
        """True while greeting or trying the 2-minute chat."""
        person = self.current_person
        if not person:
            return False
        enc = self.encounters.get(person)
        if enc is None:
            return False
        return enc.state in (
            SocialState.GREET, SocialState.WAIT_ENGAGE, SocialState.CONVERSE,
        )
    
    def estimate_distance_from_box(self, box: Tuple[int, int, int, int]) -> float:
        """Heuristic distance estimation from face bounding box.
        
        Args:
            box: (x, y, w, h) in pixels
            
        Returns:
            estimated distance in meters (rough)
        """
        x, y, w, h = box
        # Use width as proxy for distance (assumes frontal face)
        if w > FACE_BOX_CLOSE_PX:
            return 0.8  # closer than conversational
        elif w < FACE_BOX_FAR_PX:
            return 3.0  # farther than conversational
        elif FACE_BOX_GOOD_MIN <= w <= FACE_BOX_GOOD_MAX:
            # Interpolate: 40px→1.5m, 70px→1.2m
            return 1.5 - (w - FACE_BOX_GOOD_MIN) * 0.3 / (FACE_BOX_GOOD_MAX - FACE_BOX_GOOD_MIN)
        elif w < FACE_BOX_GOOD_MIN:
            return 2.0
        else:
            return 1.0
    
    def estimate_distance_from_depth(
        self, 
        box: Tuple[int, int, int, int],
        depth_map: Optional[object] = None
    ) -> Optional[float]:
        """Estimate distance using depth map (preferred when available).
        
        Args:
            box: (x, y, w, h) face box in pixels
            depth_map: depth image (meters), same resolution as RGB
            
        Returns:
            distance in meters, or None if depth unavailable
        """
        if depth_map is None:
            return None
        try:
            import numpy as np
            x, y, w, h = box
            # Sample center region of face box
            cx, cy = x + w // 2, y + h // 2
            rx, ry = max(1, w // 4), max(1, h // 4)
            x0, x1 = max(0, cx - rx), min(depth_map.shape[1], cx + rx)
            y0, y1 = max(0, cy - ry), min(depth_map.shape[0], cy + ry)
            
            patch = depth_map[y0:y1, x0:x1]
            if patch.size == 0:
                return None
            
            # Median of valid (positive) depths
            valid = patch[patch > 0.1]
            if valid.size == 0:
                return None
            return float(np.median(valid))
        except Exception:
            return None
    
    def estimate_distance(
        self,
        box: Tuple[int, int, int, int],
        depth_map: Optional[object] = None
    ) -> float:
        """Estimate person distance; prefer depth, fallback to box heuristic."""
        d = self.estimate_distance_from_depth(box, depth_map)
        if d is not None and d > 0.1:
            return d
        return self.estimate_distance_from_box(box)
    
    def _start_cooldown(self, name: str, duration: float, now: float) -> None:
        """Start cooldown for person."""
        enc = self.encounters.get(name)
        if enc is None:
            return
        enc.cooldown_until = now + duration
    
    def on_face_seen(
        self,
        name: str,
        confidence: float,
        box: Tuple[int, int, int, int],
        now: float,
        depth_map: Optional[object] = None,
    ) -> Optional[Dict]:
        """Update FSM when a face is seen.
        
        Returns:
            action dict with keys: action_type, utterance, goal_hint, state
            or None if no action needed
        """
        if name == "unknown":
            return None  # conversation.py handles unknowns
        
        # Get or create encounter
        if name not in self.encounters:
            self.encounters[name] = PersonEncounter(name=name)
        enc = self.encounters[name]
        enc.last_seen_time = now
        
        # Estimate distance
        enc.estimated_distance = self.estimate_distance(box, depth_map)
        
        # Check cooldown
        if self.is_on_cooldown(name, now):
            return None
        
        # FSM transitions
        state = enc.state
        
        if state == SocialState.IDLE_WANDER:
            # Noticed person → NOTICE
            enc.state = SocialState.NOTICE
            enc.notice_time = now
            self.current_person = name
            return {
                "action_type": "notice",
                "utterance": "",
                "goal_hint": None,
                "state": "notice",
                "person": name,
            }
        
        elif state == SocialState.NOTICE:
            # Brief observation delay before approach
            if now - enc.notice_time >= NOTICE_DECIDE_DELAY:
                # Check if already at conversational distance
                if CONVERSATIONAL_DIST_MIN <= enc.estimated_distance <= CONVERSATIONAL_DIST_MAX:
                    enc.state = SocialState.GREET
                    enc.greet_time = now
                    return self._generate_greeting(name, now, enc)
                elif enc.estimated_distance < INTIMATE_DIST:
                    # Too close; wait for them to step back
                    enc.state = SocialState.IDLE_WANDER
                    self.current_person = None
                    self._start_cooldown(name, COOLDOWN_NO_ENGAGE, now)
                    return None
                else:
                    # Need to approach
                    enc.state = SocialState.APPROACH
                    enc.approach_time = now
                    return self._generate_approach_goal(name, box, enc, now)
            return None
        
        elif state == SocialState.APPROACH:
            # Check if we reached conversational distance
            if CONVERSATIONAL_DIST_MIN <= enc.estimated_distance <= CONVERSATIONAL_DIST_MAX:
                enc.state = SocialState.GREET
                enc.greet_time = now
                return self._generate_greeting(name, now, enc)
            elif enc.estimated_distance < CONVERSATIONAL_DIST_MIN:
                # Stop; we're close enough or too close
                enc.state = SocialState.GREET
                enc.greet_time = now
                return self._generate_greeting(name, now, enc)
            # Still approaching; update goal if distance changed significantly
            return self._update_approach_goal(name, box, enc, now)
        
        elif state == SocialState.GREET:
            # Greeting spoken; start the 2-minute conversation try.
            if now - enc.greet_time >= 0.5:
                self._begin_converse(enc, now)
            return None
        
        elif state == SocialState.WAIT_ENGAGE:
            # Legacy wait → same 2-minute try (prompt / budget / leave).
            self._begin_converse(enc, now)
            return self._converse_tick(name, enc, now)
        
        elif state == SocialState.CONVERSE:
            return self._converse_tick(name, enc, now)
        
        elif state == SocialState.LEAVE:
            # Already leaving; reset to idle
            enc.state = SocialState.IDLE_WANDER
            return None
        
        return None
    
    def on_speech_heard(
        self,
        name: Optional[str],
        transcript: str,
        now: float,
    ) -> Optional[Dict]:
        """Update FSM when speech is heard (engagement signal).
        
        Returns:
            action dict with empathy response or None
        """
        if not transcript.strip():
            return None
        
        # If we're waiting or conversing with someone, this is engagement
        person = self.current_person or name
        if person is None:
            return None
        
        enc = self.encounters.get(person)
        if enc is None:
            return None
        
        enc.last_heard_time = now

        if enc.state in (SocialState.WAIT_ENGAGE, SocialState.GREET):
            self._begin_converse(enc, now)
        if enc.state == SocialState.CONVERSE:
            enc.engaged_this_session = True
            enc.engage_count += 1
            enc.converse_turns += 1
            enc.prompt_count = 0
            # Always answer the first thing they say; later turns may just listen.
            if enc.converse_turns <= 1 or random.random() < 0.75:
                return self._generate_empathy_response(person, transcript, enc, now)
            return None

        return None
    
    def on_command(
        self,
        command: str,
        now: float,
    ) -> Optional[Dict]:
        """Handle "Hey Kevin, xyz" commands.
        
        Returns:
            action dict with acknowledgement and goal_hint
        """
        # Basic command cooldown to avoid spam
        if now < self._command_cooldown:
            return None
        self._command_cooldown = now + 2.0
        
        cmd = command.strip().lower()
        
        # Command acknowledgements: curious, calm, earnest helper tone
        # Movement commands
        if any(w in cmd for w in ["stop", "halt", "freeze", "hold"]):
            acks = ["Stopping!", "Okay, holding still.", "Wheels stopping now!"]
            return {
                "action_type": "command",
                "utterance": random.choice(acks),
                "goal_hint": {"type": "clear"},
                "command": "stop",
            }
        
        if any(w in cmd for w in ["come here", "come to me", "come over"]):
            acks = ["Coming over!", "On my way!", "Rolling over now!"]
            return {
                "action_type": "command",
                "utterance": random.choice(acks),
                "goal_hint": {"type": "approach_speaker"},
                "command": "come_here",
            }
        
        # Designing for Exit (WeRobot 2022): immediate response + long cooldown
        if any(w in cmd for w in ["go away", "leave me alone", "back up", "give me space", "not now", "i'm busy"]):
            # Set extended cooldown for current person if known
            if self.current_person and self.current_person in self.encounters:
                self._start_cooldown(self.current_person, COOLDOWN_DISMISSED, now)
            
            acks = ["Sorry! Moving away.", "Understood. Backing up.", "No problem. I'll give you space!"]
            return {
                "action_type": "command",
                "utterance": random.choice(acks),
                "goal_hint": {"type": "retreat", "distance_m": 2.0, "dismissed": True},
                "command": "go_away",
                "cooldown_extended": True,
            }
        
        if any(w in cmd for w in ["wander", "explore", "look around", "roam"]):
            acks = ["Back to wandering!", "Time to explore!", "Off I go!"]
            return {
                "action_type": "command",
                "utterance": random.choice(acks),
                "goal_hint": {"type": "resume_wander"},
                "command": "wander",
            }
        
        if "say hi to" in cmd or "greet" in cmd:
            return {
                "action_type": "command",
                "utterance": "I'll say hello if I see them!",
                "goal_hint": None,
                "command": "greet_request",
            }
        
        # Unknown command; acknowledge honestly (gentle self-deprecation)
        acks = [
            "I heard you, but I'm not sure how to do that yet.",
            "Hmm. I'm still learning that one!",
            "That's a new one for me. Still learning!",
        ]
        return {
            "action_type": "command",
            "utterance": random.choice(acks),
            "goal_hint": None,
            "command": "unknown",
        }
    
    def _begin_converse(self, enc: PersonEncounter, now: float) -> None:
        """Enter CONVERSE and start the 2-minute budget."""
        enc.state = SocialState.CONVERSE
        if enc.converse_t0 <= 0.0:
            enc.converse_t0 = enc.greet_time if enc.greet_time > 0.0 else now
        if enc.last_heard_time <= 0.0:
            enc.last_heard_time = enc.converse_t0
        if enc.last_prompt_time <= 0.0:
            enc.last_prompt_time = enc.converse_t0
        self.current_person = enc.name

    def _leave_chat(self, name: str, enc: PersonEncounter, now: float,
                    utterance: str) -> Dict:
        engaged = bool(enc.engaged_this_session)
        enc.state = SocialState.LEAVE
        self.current_person = None
        self._start_cooldown(
            name, COOLDOWN_ENGAGED if engaged else COOLDOWN_NO_ENGAGE, now)
        enc.state = SocialState.IDLE_WANDER
        enc.goal_hint = {"type": "resume_wander"} if self.drive_armed else None
        return {
            "action_type": "leave",
            "utterance": utterance,
            "goal_hint": {"type": "resume_wander"},
            "state": "leave",
            "person": name,
        }

    def _prompt_chat(self, name: str, enc: PersonEncounter, now: float) -> Dict:
        prefer_spanish = name.strip().lower() in SPANISH_NAMES
        bank = CHAT_PROMPTS_ES if prefer_spanish else CHAT_PROMPTS
        utterance = bank[enc.prompt_count % len(bank)]
        enc.prompt_count += 1
        enc.last_prompt_time = now
        enc.last_spoke_time = now
        return {
            "action_type": "prompt",
            "utterance": utterance,
            "goal_hint": {"type": "stop"},
            "state": "converse",
            "person": name,
        }

    def _converse_tick(self, name: str, enc: PersonEncounter, now: float):
        """Stay for up to 2 minutes; prompt if quiet; leave if ignored or done."""
        t0 = enc.converse_t0 or enc.greet_time or now
        elapsed = now - t0
        last_human = max(enc.last_heard_time, enc.converse_t0, enc.greet_time)
        last_kevin = max(enc.last_prompt_time, enc.last_spoke_time, t0)
        quiet = now - max(last_human, last_kevin)
        prefer_spanish = name.strip().lower() in SPANISH_NAMES

        if enc.last_seen_time > 0.0 and (now - enc.last_seen_time) >= CONVERSE_GONE_S:
            wrap = random.choice(WRAP_MESSAGES_ES if prefer_spanish else WRAP_MESSAGES)
            return self._leave_chat(name, enc, now, wrap)

        if elapsed >= CONVERSE_BUDGET_S:
            wrap = random.choice(WRAP_MESSAGES_ES if prefer_spanish else WRAP_MESSAGES)
            return self._leave_chat(name, enc, now, wrap)

        if not enc.engaged_this_session:
            if quiet >= CONVERSE_PROMPT_S:
                if enc.prompt_count >= CONVERSE_PROMPT_MAX:
                    leave_bank = LEAVE_MESSAGES_ES if prefer_spanish else LEAVE_MESSAGES
                    msg = random.choice([m for m in leave_bank if m] or leave_bank)
                    return self._leave_chat(name, enc, now, msg)
                return self._prompt_chat(name, enc, now)
            return None

        # They talked — hold the rest of the 2 minutes; nudge if it goes quiet.
        if quiet >= CONVERSE_PROMPT_S:
            if enc.prompt_count < CONVERSE_PROMPT_MAX:
                return self._prompt_chat(name, enc, now)
            wrap = "¡Fue un placer hablar contigo!" if prefer_spanish else "Good talking with you!"
            return self._leave_chat(name, enc, now, wrap)
        return None

    def _generate_greeting(
        self, 
        name: str, 
        now: float,
        enc: PersonEncounter,
    ) -> Dict:
        """Generate greeting utterance (curious, earnest, optionally playful)."""
        hour = time.localtime().tm_hour
        prefer_spanish = name.strip().lower() in SPANISH_NAMES
        
        # Invite a short chat — Kevin tries for ~2 minutes after a known face.
        if prefer_spanish:
            if 5 <= hour < 12:
                greetings = [
                    f"¡Buenos días, {name}! ¿Me das dos minutos para platicar?",
                    f"¡Hola {name}! Buenos días. ¿Platicamos un par de minutos?",
                ]
            elif 12 <= hour < 18:
                greetings = [
                    f"¡Hola, {name}! ¿Me das dos minutos para platicar?",
                    f"¡Qué tal, {name}! ¿Platicamos un ratito?",
                ]
            else:
                greetings = [
                    f"¡Buenas noches, {name}! ¿Me das dos minutos para platicar?",
                    f"¡Hola {name}! ¿Cómo te fue hoy? Tengo un par de minutos.",
                ]
        else:
            if 5 <= hour < 12:
                greetings = [
                    f"Good morning, {name}! Got two minutes to chat?",
                    f"Hi {name}! Nice morning. Want to talk for a couple minutes?",
                ]
            elif 12 <= hour < 18:
                greetings = [
                    f"Hi, {name}! Got two minutes to chat?",
                    f"Hey {name}! Can I steal a couple minutes?",
                    f"Hello, {name}! Want to talk for two minutes?",
                ]
            else:
                greetings = [
                    f"Good evening, {name}! Got two minutes to chat?",
                    f"Hi {name}! How was your day? I can hang for a couple minutes.",
                ]
        
        utterance = random.choice(greetings)
        enc.last_spoke_time = now
        return {
            "action_type": "greet",
            "utterance": utterance,
            "goal_hint": {"type": "stop"},
            "state": "greet",
            "person": name,
        }
    
    def _generate_empathy_response(
        self,
        name: str,
        transcript: str,
        enc: PersonEncounter,
        now: float,
    ) -> Dict:
        """Generate short empathetic response (curious, calm, earnest helper tone)."""
        prefer_spanish = name.strip().lower() in SPANISH_NAMES
        
        # Simple keyword-based response selection
        text = transcript.lower()
        
        # Negative/difficult keywords → supportive
        if any(w in text for w in ["hard", "difficult", "tough", "bad", "sad", "tired", "difícil", "cansado"]):
            bank = EMPATHY_TOUGH_ES if prefer_spanish else EMPATHY_TOUGH
            utterance = random.choice(bank)
        # Positive keywords → encouraging
        elif any(w in text for w in ["good", "great", "awesome", "happy", "bien", "genial", "feliz"]):
            bank = EMPATHY_HAPPY_ES if prefer_spanish else EMPATHY_HAPPY
            utterance = random.choice(bank)
        else:
            # Generic acknowledgement
            bank = EMPATHY_TEMPLATES_ES if prefer_spanish else EMPATHY_TEMPLATES
            utterance = random.choice(bank)
        
        enc.last_spoke_time = now
        return {
            "action_type": "empathy",
            "utterance": utterance,
            "goal_hint": None,
            "state": "converse",
            "person": name,
        }
    
    def _generate_approach_goal(
        self,
        name: str,
        box: Tuple[int, int, int, int],
        enc: PersonEncounter,
        now: float,
    ) -> Dict:
        """Generate goal hint to approach person to conversational distance."""
        if not self.drive_armed:
            # Drive disarmed; skip approach
            enc.state = SocialState.GREET
            enc.greet_time = now
            return self._generate_greeting(name, now, enc)
        
        x, y, w, h = box
        # Calculate bearing to face center (assume 320px wide frame)
        frame_cx = 160  # center of 320px frame
        face_cx = x + w // 2
        pixel_offset = face_cx - frame_cx
        # Rough bearing: ~60° FOV → ~0.2 deg/px
        bearing_deg = pixel_offset * 0.19
        
        # Target distance: aim for APPROACH_STOP_DIST, but don't get closer
        # than we already are (if somehow already within range)
        if enc.estimated_distance > CONVERSATIONAL_DIST_MAX:
            target_dist = APPROACH_STOP_DIST
        else:
            target_dist = max(CONVERSATIONAL_DIST_MIN, enc.estimated_distance - 0.2)
        
        goal_hint = {
            "type": "approach_person",
            "bearing_deg": bearing_deg,
            "target_distance_m": target_dist,
            "person": name,
        }
        enc.goal_hint = goal_hint
        
        return {
            "action_type": "approach",
            "utterance": "",  # silent approach; greet when arrived
            "goal_hint": goal_hint,
            "state": "approach",
            "person": name,
        }
    
    def _update_approach_goal(
        self,
        name: str,
        box: Tuple[int, int, int, int],
        enc: PersonEncounter,
        now: float,
    ) -> Optional[Dict]:
        """Update approach goal if person moved significantly."""
        # Only update if significant change in bearing or distance
        if enc.goal_hint is None:
            return None
        
        x, y, w, h = box
        frame_cx = 160
        face_cx = x + w // 2
        pixel_offset = face_cx - frame_cx
        new_bearing_deg = pixel_offset * 0.19
        
        old_bearing = enc.goal_hint.get("bearing_deg", 0.0)
        if abs(new_bearing_deg - old_bearing) > 15.0:  # 15° change
            return self._generate_approach_goal(name, box, enc, now)
        
        return None
    
    def reset_encounter(self, name: str) -> None:
        """Reset encounter state (for testing or explicit reset)."""
        if name in self.encounters:
            del self.encounters[name]
        if self.current_person == name:
            self.current_person = None

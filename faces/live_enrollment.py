"""Interactive live enrollment for unknown faces.

When Kevin sees an unknown face (below confidence threshold), he politely
asks for their name, collects live webcam samples, and enrolls them.

This module integrates with people_behavior.py and requires:
- ASR for name capture
- Camera feed for face crops
- Speech output for prompts/confirmation
"""

import time
import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable, List, Tuple
from pathlib import Path

try:
    import cv2
    _HAS_OPENCV = True
except ImportError:
    _HAS_OPENCV = False


@dataclass
class EnrollmentSession:
    """Active enrollment session state."""
    started_at: float
    prompted_at: float
    person_name: Optional[str] = None
    samples_collected: int = 0
    samples_needed: int = 3
    last_box: Optional[Tuple[int, int, int, int]] = None
    timeout_seconds: float = 15.0
    
    def is_timed_out(self, now: Optional[float] = None) -> bool:
        """Check if session has exceeded timeout."""
        t = time.time() if now is None else now
        return (t - self.prompted_at) > self.timeout_seconds
    
    def is_complete(self) -> bool:
        """Check if session collected enough samples."""
        return self.samples_collected >= self.samples_needed


class LiveEnrollmentManager:
    """Manages interactive enrollment sessions for unknown faces.
    
    Workflow:
    1. Unknown face detected (confidence < threshold)
    2. Ask for name (English by default, Spanish if they respond in Spanish)
    3. Collect 3-5 live face crops while they're facing robot
    4. Enroll with InsightFace backend
    5. Confirm: "Nice to meet you, {name}!"
    6. Cooldown to prevent re-prompting
    
    Kevin personality: curious, calm, clean humor
    """
    
    # Prompts (Kevin personality: curious, calm, helpful)
    NAME_PROMPTS_EN = [
        "Hi! I don't think we've met. What's your name?",
        "Hey there! I'm Kevin. What should I call you?",
        "Hello! I don't recognize you yet. What's your name?",
    ]
    
    NAME_PROMPTS_ES = [
        "¡Hola! No creo que nos hayamos conocido. ¿Cómo te llamas?",
        "¡Hola! Soy Kevin. ¿Cómo te llamas?",
        "¡Hola! No te reconozco todavía. ¿Cómo te llamas?",
    ]
    
    CONFIRM_EN = [
        "Nice to meet you, {name}! I'll remember you.",
        "Great to meet you, {name}! Got it.",
        "Pleasure to meet you, {name}! I'll recognize you next time.",
    ]
    
    CONFIRM_ES = [
        "¡Mucho gusto, {name}! Te recordaré.",
        "¡Encantado de conocerte, {name}! Entendido.",
        "¡Un placer conocerte, {name}! Te reconoceré la próxima vez.",
    ]
    
    TIMEOUT_LEAVE_EN = [
        "No worries! Let me know if you need anything.",
        "Okay! I'll be around if you need me.",
        "Alright! Feel free to say hi anytime.",
    ]
    
    TIMEOUT_LEAVE_ES = [
        "¡No hay problema! Avísame si necesitas algo.",
        "¡Está bien! Estaré por aquí si me necesitas.",
        "¡De acuerdo! Salúdame cuando quieras.",
    ]
    
    def __init__(
        self,
        recognizer,
        speak_fn: Callable[[str], None],
        min_samples: int = 3,
        max_samples: int = 5,
        session_timeout: float = 15.0,
        cooldown_seconds: float = 300.0,
        language: str = "en"
    ):
        """Initialize live enrollment manager.
        
        Args:
            recognizer: FaceRecognizer instance
            speak_fn: Function to speak text
            min_samples: Minimum samples to collect
            max_samples: Maximum samples to collect
            session_timeout: Seconds to wait for name response
            cooldown_seconds: Seconds before re-prompting same unknown face
            language: Default language ("en" or "es")
        """
        self.recognizer = recognizer
        self.speak = speak_fn
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.session_timeout = session_timeout
        self.cooldown_seconds = cooldown_seconds
        self.language = language
        
        # Session state
        self.active_session: Optional[EnrollmentSession] = None
        self.unknown_face_cooldowns = {}  # box_hash -> last_prompted_time
        self.prompt_index = 0
        
    def _box_hash(self, box: Tuple[int, int, int, int]) -> str:
        """Create simple hash of face box for cooldown tracking."""
        x, y, w, h = box
        # Quantize to 50px grid to handle small movements
        qx, qy = (x // 50) * 50, (y // 50) * 50
        qw, qh = (w // 20) * 20, (h // 20) * 20
        return f"{qx}_{qy}_{qw}_{qh}"
    
    def _should_prompt(self, box: Tuple[int, int, int, int], now: float) -> bool:
        """Check if we should prompt for this unknown face."""
        box_hash = self._box_hash(box)
        last_prompted = self.unknown_face_cooldowns.get(box_hash, 0.0)
        return (now - last_prompted) >= self.cooldown_seconds
    
    def _get_prompt(self, language: str = "en") -> str:
        """Get name prompt in specified language."""
        prompts = self.NAME_PROMPTS_EN if language == "en" else self.NAME_PROMPTS_ES
        prompt = prompts[self.prompt_index % len(prompts)]
        self.prompt_index += 1
        return prompt
    
    def _get_confirm(self, name: str, language: str = "en") -> str:
        """Get enrollment confirmation in specified language."""
        confirms = self.CONFIRM_EN if language == "en" else self.CONFIRM_ES
        confirm = confirms[self.prompt_index % len(confirms)]
        return confirm.format(name=name)
    
    def _get_timeout_leave(self, language: str = "en") -> str:
        """Get polite leave message for timeout."""
        leaves = self.TIMEOUT_LEAVE_EN if language == "en" else self.TIMEOUT_LEAVE_ES
        return leaves[self.prompt_index % len(leaves)]
    
    def should_enroll(self, name: str, box: Tuple[int, int, int, int], now: Optional[float] = None) -> bool:
        """Check if we should start enrollment for this unknown face.
        
        Enrollment triggers ONLY when recognition returns "unknown" (i.e., failed
        threshold + margin checks). Never triggers based on raw cosine similarity.
        
        Args:
            name: Recognition result ("unknown" or person name)
            box: Face bounding box
            now: Current time (optional)
        
        Returns:
            True if should start enrollment session
        """
        t = time.time() if now is None else now
        
        # Already in a session?
        if self.active_session is not None:
            return False
        
        # Face was recognized (accepted by threshold + margin)?
        if name != "unknown":
            return False
        
        # Cooldown active for this face?
        if not self._should_prompt(box, t):
            return False
        
        return True
    
    def start_session(self, box: Tuple[int, int, int, int], language: str = "en", now: Optional[float] = None) -> str:
        """Start enrollment session and return prompt.
        
        Args:
            box: Face bounding box
            language: Language for prompt
            now: Current time (optional)
        
        Returns:
            Prompt text to speak
        """
        t = time.time() if now is None else now
        
        # Mark cooldown
        box_hash = self._box_hash(box)
        self.unknown_face_cooldowns[box_hash] = t
        
        # Create session
        self.active_session = EnrollmentSession(
            started_at=t,
            prompted_at=t,
            samples_needed=self.min_samples,
            timeout_seconds=self.session_timeout,
            last_box=box
        )
        
        prompt = self._get_prompt(language)
        self.speak(prompt)
        return prompt
    
    def on_name_received(self, name: str, language: str = "en") -> str:
        """Handle name received from ASR.
        
        Args:
            name: Person's name from ASR
            language: Detected language
        
        Returns:
            Acknowledgment text
        """
        if self.active_session is None:
            return ""
        
        # Clean name
        name = name.strip().title()
        if not name:
            return ""
        
        self.active_session.person_name = name
        self.language = language  # Switch to their language
        
        # Acknowledge and prepare to collect samples
        ack = f"Thanks, {name}! Let me get a good look at you."
        if language == "es":
            ack = f"¡Gracias, {name}! Déjame verte bien."
        
        self.speak(ack)
        return ack
    
    def collect_sample(self, image: np.ndarray, box: Tuple[int, int, int, int], landmarks: Optional[np.ndarray] = None) -> Tuple[bool, int, int]:
        """Collect face sample during active session.
        
        Args:
            image: BGR image
            box: Face bounding box
            landmarks: Optional 5-point landmarks
        
        Returns:
            (sample_collected, samples_so_far, samples_needed)
        """
        if self.active_session is None or self.active_session.person_name is None:
            return False, 0, 0
        
        # Store box for tracking
        self.active_session.last_box = box
        
        # Extract and cache face crop (don't enroll yet, collect all first)
        if not hasattr(self.active_session, 'crops'):
            self.active_session.crops = []
            self.active_session.landmarks_list = []
        
        # Check if we should collect this sample (rate limiting)
        if len(self.active_session.crops) > 0:
            # Wait ~0.5s between samples for variety
            elapsed = time.time() - self.active_session.started_at
            expected_samples = int(elapsed / 0.5)
            if len(self.active_session.crops) >= expected_samples:
                return False, len(self.active_session.crops), self.active_session.samples_needed
        
        # Collect sample
        self.active_session.crops.append(image.copy())
        self.active_session.landmarks_list.append(landmarks)
        self.active_session.samples_collected = len(self.active_session.crops)
        
        return True, self.active_session.samples_collected, self.active_session.samples_needed
    
    def finalize_enrollment(self) -> Tuple[bool, str]:
        """Finalize enrollment with collected samples.
        
        Returns:
            (success, message)
        """
        if self.active_session is None or self.active_session.person_name is None:
            return False, "No active session"
        
        if not hasattr(self.active_session, 'crops') or len(self.active_session.crops) == 0:
            return False, "No samples collected"
        
        name = self.active_session.person_name
        
        # Enroll all collected samples
        total_enrolled = 0
        for img, landmarks in zip(self.active_session.crops, self.active_session.landmarks_list):
            # Use InsightFace backend for enrollment
            if self.recognizer.backend == "insightface":
                # Detect faces with landmarks
                detections = self.recognizer.detect_faces_with_landmarks(img)
                if detections:
                    box, lm = detections[0]  # Use largest/first face
                    emb = self.recognizer.extract_embedding(img, box, lm if lm is not None else landmarks)
                    if emb is not None:
                        if name not in self.recognizer.db:
                            self.recognizer.db[name] = {"embeddings": []}
                        self.recognizer.db[name]["embeddings"].append(emb)
                        total_enrolled += 1
                        
                        # Save face crop
                        person_dir = self.recognizer.gallery_path / name.replace(" ", "_").lower()
                        person_dir.mkdir(exist_ok=True)
                        crop, _ = self.recognizer.padded_crop(img, box, pad=0.25)
                        if crop is not None and crop.size > 0:
                            img_idx = len(list(person_dir.glob("*.jpg")))
                            img_path = person_dir / f"{img_idx:03d}.jpg"
                            cv2.imwrite(str(img_path), crop)
            else:
                # Fallback for other backends
                count = self.recognizer.enroll(name, img)
                total_enrolled += count
        
        if total_enrolled > 0:
            # Save database
            self.recognizer._save_database()
            
            # Confirm enrollment
            confirm = self._get_confirm(name, self.language)
            self.speak(confirm)
            
            # Clear session
            self.active_session = None
            
            return True, f"Enrolled {name} with {total_enrolled} samples"
        else:
            self.active_session = None
            return False, "Failed to extract embeddings"
    
    def check_timeout(self, now: Optional[float] = None) -> Tuple[bool, str]:
        """Check if active session timed out.
        
        Returns:
            (timed_out, leave_message)
        """
        if self.active_session is None:
            return False, ""
        
        t = time.time() if now is None else now
        
        if self.active_session.is_timed_out(t):
            # No name received, politely leave
            leave_msg = self._get_timeout_leave(self.language)
            self.speak(leave_msg)
            
            # Clear session
            self.active_session = None
            return True, leave_msg
        
        return False, ""
    
    def cancel_session(self) -> None:
        """Cancel active enrollment session."""
        self.active_session = None
    
    def is_session_active(self) -> bool:
        """Check if enrollment session is active."""
        return self.active_session is not None

"""Conversation glue for face recognition.

Integrates face recognition with greetings and name enrollment.
Uses existing TTS pattern from ui.py or provides stub for offline demos.
"""

import time
import random
from typing import Optional, Callable

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

from faces.recognizer import FaceRecognizer


GREETINGS = {
    "morning": [
        "Good morning, {name}!",
        "Morning, {name}! Hope you slept well.",
        "Hey {name}, good to see you this morning!",
    ],
    "afternoon": [
        "Hi {name}!",
        "Hey there, {name}!",
        "Good to see you, {name}!",
    ],
    "evening": [
        "Good evening, {name}!",
        "Hey {name}, how was your day?",
        "Evening, {name}!",
    ],
}

UNKNOWN_PROMPTS = [
    "Hi! I don't think we've met. What's your name?",
    "Hello! I'm Kevin. What should I call you?",
    "Hey there! I don't recognize you. What's your name?",
]

ENROLLMENT_CONFIRM = [
    "Nice to meet you, {name}! I'll remember you.",
    "Got it, {name}! I'll recognize you next time.",
    "Pleasure to meet you, {name}!",
]


def get_time_of_day() -> str:
    """Get current time of day for contextual greetings."""
    hour = time.localtime().tm_hour
    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 18:
        return "afternoon"
    else:
        return "evening"


class ConversationManager:
    """Manages face-based conversations and greetings."""
    
    def __init__(self, 
                 recognizer: FaceRecognizer,
                 speak_fn: Optional[Callable[[str], None]] = None,
                 volume: float = 0.1):
        """Initialize conversation manager.
        
        Args:
            recognizer: FaceRecognizer instance
            speak_fn: Function to speak text (if None, uses stub print)
            volume: TTS volume (0.0-1.0), default 0.1 (10%)
        """
        self.recognizer = recognizer
        self.speak_fn = speak_fn if speak_fn is not None else self._stub_speak
        self.volume = volume
        
        self.last_seen = {}
        self.cooldown_seconds = 300
        
        print(f"ConversationManager: volume={volume:.0%}, cooldown={self.cooldown_seconds}s")
    
    def _stub_speak(self, text: str):
        """Stub speak function for offline demos."""
        print(f"🔊 [TTS @ {self.volume:.0%}]: {text}")
    
    def process_frame(self, image) -> dict:
        """Process a frame and generate appropriate conversation.
        
        Args:
            image: BGR image (numpy array or compatible)
        
        Returns:
            dict with keys: faces, greetings, unknowns, enrollments
        """
        if not _HAS_NUMPY:
            print("Warning: NumPy not available, conversation disabled")
            return {"faces": [], "greetings": [], "unknowns": [], "enrollments": []}
        
        results = self.recognizer.recognize(image, threshold=0.6)
        
        current_time = time.time()
        greetings = []
        unknowns = []
        
        for name, confidence, box in results:
            if name == "unknown":
                if "unknown" not in self.last_seen or \
                   current_time - self.last_seen["unknown"] > self.cooldown_seconds:
                    prompt = random.choice(UNKNOWN_PROMPTS)
                    self.speak_fn(prompt)
                    unknowns.append({"confidence": confidence, "box": box, "prompt": prompt})
                    self.last_seen["unknown"] = current_time
            else:
                if name not in self.last_seen or \
                   current_time - self.last_seen[name] > self.cooldown_seconds:
                    time_of_day = get_time_of_day()
                    greeting = random.choice(GREETINGS[time_of_day]).format(name=name)
                    self.speak_fn(greeting)
                    greetings.append({
                        "name": name,
                        "confidence": confidence,
                        "box": box,
                        "greeting": greeting
                    })
                    self.last_seen[name] = current_time
        
        return {
            "faces": results,
            "greetings": greetings,
            "unknowns": unknowns,
            "enrollments": []
        }
    
    def enroll_from_input(self, image, name: str) -> bool:
        """Enroll a new person from current frame.
        
        Args:
            image: BGR image
            name: Person's name
        
        Returns:
            True if enrollment succeeded
        """
        count = self.recognizer.enroll(name, image)
        if count > 0:
            confirm = random.choice(ENROLLMENT_CONFIRM).format(name=name)
            self.speak_fn(confirm)
            self.last_seen[name] = time.time()
            return True
        return False
    
    def reset_cooldowns(self):
        """Reset all greeting cooldowns."""
        self.last_seen.clear()
    
    def set_volume(self, volume: float):
        """Set TTS volume (0.0-1.0)."""
        self.volume = max(0.0, min(1.0, volume))
        print(f"ConversationManager: volume={self.volume:.0%}")


def create_speak_function(tts_backend: str = "stub", volume: float = 0.1):
    """Create a speak function based on available TTS backend.
    
    Args:
        tts_backend: "stub", "kokoro", "gemini", or "espeak"
        volume: TTS volume (0.0-1.0)
    
    Returns:
        speak function: callable(text: str) -> None
    """
    if tts_backend == "stub":
        def speak(text: str):
            print(f"🔊 [TTS @ {volume:.0%}]: {text}")
        return speak
    
    elif tts_backend == "espeak":
        try:
            import subprocess
            def speak(text: str):
                vol = int(volume * 100)
                subprocess.run(
                    ["espeak", "-a", str(vol), text],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            return speak
        except Exception:
            print("Warning: espeak not available, using stub")
            return create_speak_function("stub", volume)
    
    elif tts_backend == "kokoro":
        print("Warning: Kokoro TTS requires brain_zmq integration")
        return create_speak_function("stub", volume)
    
    elif tts_backend == "gemini":
        print("Warning: Gemini TTS requires ui.py integration")
        return create_speak_function("stub", volume)
    
    else:
        print(f"Warning: Unknown TTS backend {tts_backend}, using stub")
        return create_speak_function("stub", volume)

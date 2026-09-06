"""speech_io.py – Local ears/mouth for Kevin (ReSpeaker + Kokoro).

TTS: Kokoro-ONNX → paplay on ReSpeaker sink.
ASR: faster-whisper now (Parakeet on i777 3090 is the upgrade path).

Volume: apply ONCE via paplay --volume. Do not also scale samples and
pactl-set the sink to the same fraction (that made 30% effectively ~3%).
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import threading

KOKORO_MODEL = os.path.expanduser("~/.kevin/models/kokoro/kokoro-v1.0.onnx")
KOKORO_VOICES = os.path.expanduser("~/.kevin/models/kokoro/voices-v1.0.bin")
RS_SINK = os.environ.get(
    "KEVIN_SPEAK_SINK",
    "alsa_output.usb-SEEED_ReSpeaker_4_Mic_Array__UAC1.0_-00.analog-stereo",
)
RS_SRC = os.environ.get(
    "KEVIN_LISTEN_SRC",
    "alsa_input.usb-SEEED_ReSpeaker_4_Mic_Array__UAC1.0_-00.analog-surround-21",
)
SPEAK_VOLUME = float(os.environ.get("KEVIN_SPEAK_VOL", "0.30"))

_lock = threading.Lock()
_kokoro = None
_whisper = None
_speak_busy = False


def _ensure_kokoro():
    global _kokoro
    if _kokoro is not None:
        return _kokoro
    from kokoro_onnx import Kokoro
    _kokoro = Kokoro(KOKORO_MODEL, KOKORO_VOICES)
    return _kokoro


def _ensure_whisper():
    global _whisper
    if _whisper is not None:
        return _whisper
    from faster_whisper import WhisperModel
    _whisper = WhisperModel("base", device="cpu", compute_type="int8")
    return _whisper


def speak(text: str, voice: str = "am_michael", block: bool = False) -> None:
    text = (text or "").strip()
    if not text:
        return

    def _run():
        global _speak_busy
        with _lock:
            if _speak_busy:
                return
            _speak_busy = True
        try:
            k = _ensure_kokoro()
            samples, sr = k.create(text, voice=voice, speed=1.0)
            import soundfile as sf
            import numpy as np
            vol = max(0.0, min(1.0, float(SPEAK_VOLUME)))
            # Full-scale WAV; loudness only via paplay --volume (single attenuator).
            samples = np.asarray(samples, dtype=np.float32)
            peak = float(np.max(np.abs(samples))) if samples.size else 0.0
            if peak > 1.0:
                samples = samples / peak
            path = os.path.join(tempfile.gettempdir(), "kevin_tts.wav")
            sf.write(path, samples, sr)
            # Keep the physical sink near full; user volume = paplay only.
            try:
                subprocess.run(
                    ["pactl", "set-sink-mute", RS_SINK, "0"],
                    check=False, timeout=2,
                )
                subprocess.run(
                    ["pactl", "set-sink-volume", RS_SINK, "100%"],
                    check=False, timeout=2,
                )
            except Exception:
                pass
            paplay_vol = max(1, int(round(65536 * vol)))
            r = subprocess.run(
                ["paplay", f"--device={RS_SINK}", f"--volume={paplay_vol}", path],
                check=False, timeout=60, capture_output=True, text=True,
            )
            if r.returncode != 0:
                print("speech: paplay rc=%s err=%r" % (r.returncode, (r.stderr or "")[:120]))
            print("speech: said %r (vol=%.0f%% sink=%s)" % (text[:80], vol * 100, RS_SINK.split(".")[-1]))
        except Exception as e:
            print("speech: speak failed: %s" % e)
        finally:
            with _lock:
                _speak_busy = False

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    if block:
        t.join(timeout=60)


def listen_seconds(secs: float = 3.0) -> str:
    path = os.path.join(tempfile.gettempdir(), "kevin_listen.wav")
    try:
        subprocess.run(
            ["parecord", f"--device={RS_SRC}", "--file-format=wav",
             "--channels=1", "--rate=16000", path],
            timeout=secs + 1.5, check=False,
        )
    except subprocess.TimeoutExpired:
        pass
    if not os.path.exists(path) or os.path.getsize(path) < 1000:
        return ""
    try:
        model = _ensure_whisper()
        segments, _info = model.transcribe(path, language="en", beam_size=1)
        text = " ".join(s.text.strip() for s in segments).strip()
        print("speech: heard %r" % (text[:120],))
        return text
    except Exception as e:
        print("speech: listen failed: %s" % e)
        return ""


def is_speaking() -> bool:
    with _lock:
        return _speak_busy

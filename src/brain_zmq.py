"""brain_zmq.py — ZMQ + direct Qwen2.5-VL inference for AnglerDroid.

Replaces HTTP + cloud API with ZMQ REP/REQ and local GPU inference.
Keeps all brain logic: conversation history, twist tracking, dedup, STT, TTS.

Usage (3090):
  python brain_zmq.py --port 5555 --model Qwen/Qwen2.5-VL-3B-Instruct

Jetson side:
  python main.py --brain-url tcp://192.168.50.198:5555 ...
"""

import argparse
import base64
import io
import json
import os
import re
import threading
import time
import wave

import numpy as np
import torch
import zmq
from PIL import Image

# ── Regex for parsing LLM output ──────────────────────────────────
_TWIST_RE   = re.compile(r'twist_for\s*\(')
_TWIST_VALS = re.compile(r'twist_for\s*\(\s*([-.0-9]+)\s*,\s*([-.0-9]+)')
_SPEAK_RE   = re.compile(r'speak\s*\(\s*(["\'])(.*?)\1\s*\)')
_STATE_RE   = re.compile(r'state\s*\(\s*(["\'])(.*?)\1\s*\)')

MAX_CONTEXT = 200
TWIST_HISTORY_LEN = 20

# ── TTS (kokoro) ──────────────────────────────────────────────────
_kokoro = None
_HAS_KOKORO = False

def _init_kokoro():
    global _kokoro, _HAS_KOKORO
    try:
        from kokoro_onnx import Kokoro
        _kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
        _HAS_KOKORO = True
        print("brain: Kokoro TTS loaded")
    except Exception as e:
        print("brain: Kokoro not available (%s)" % e)


def _load_prompt():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prompt.txt')
    try:
        with open(p, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        return "You are a helpful robot. Respond with function calls or '.'"


# ── Model loader ──────────────────────────────────────────────────

def load_model(model_name, max_pixels=501760):
    """Load Qwen2.5-VL model and processor, return (model, processor)."""
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    print("brain: loading %s ..." % model_name)
    t0 = time.time()

    processor = AutoProcessor.from_pretrained(
        model_name,
        min_pixels=28 * 28 * 4,
        max_pixels=max_pixels,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    print("brain: model loaded in %.1fs" % (time.time() - t0))
    return model, processor


# ── Brain ─────────────────────────────────────────────────────────

class Brain:
    def __init__(self, model, processor, system_prompt):
        self._model = model
        self._processor = processor
        self._prompt = system_prompt
        self._conversation = []    # list of {"role":..., "content":...}
        self._twist_history = []
        self._agent_state = ""
        self._last_speak = ""
        self._last_stt = ""
        self._lock = threading.Lock()
        self._turn = 0
        self._api_ms_total = 0.0
        self._stt_model = None
        self._pending_tts = None
        self._tts_thread = None

    def init_stt(self):
        try:
            from faster_whisper import WhisperModel
            self._stt_model = WhisperModel("base", device="cuda",
                                           compute_type="float16")
            print("brain: faster-whisper STT loaded (base, cuda)")
        except Exception as e:
            print("brain: STT not available (%s)" % e)

    def transcribe_audio(self, audio_b64_chunks):
        if not self._stt_model or not audio_b64_chunks:
            return None
        try:
            pcm = b""
            for chunk in audio_b64_chunks:
                pcm += base64.b64decode(chunk)
            if len(pcm) < 3200:
                return None
            buf = io.BytesIO()
            with wave.open(buf, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(pcm)
            buf.seek(0)
            segments, _ = self._stt_model.transcribe(buf, beam_size=1,
                                                      language="en",
                                                      without_timestamps=True)
            text = " ".join(s.text.strip() for s in segments).strip()
            if text:
                print("brain: STT → '%s'" % text[:100])
                return text
        except Exception as e:
            print("brain: STT error: %s" % e)
        return None

    # ── Main inference ─────────────────────────────────────────────

    def infer(self, jpeg_bytes, metadata):
        """Run one turn. Returns (text, api_ms, tts_audio, stt_text)."""
        t_total = time.time()
        tts_ready = self._pending_tts
        self._pending_tts = None

        frame_id = metadata.get("frame_id", 0)
        velocity = metadata.get("velocity", 0.0)
        angular_vel = metadata.get("angular_velocity", 0.0)
        speech = metadata.get("speech", "")
        audio_chunks = metadata.get("audio_chunks")

        with self._lock:
            t_stt = time.time()
            stt_text = self.transcribe_audio(audio_chunks)
            self._last_stt = stt_text or ""
            stt_ms = (time.time() - t_stt) * 1000

            combined = ""
            if speech:
                combined = speech
            if stt_text:
                combined = (combined + " " + stt_text).strip() if combined else stt_text

            lines = ["frame: %d" % frame_id,
                     "wz: %.3f %.3f" % (velocity, angular_vel)]
            if self._twist_history:
                recent = " | ".join("%.2f,%.2f" % (f, a)
                                    for f, a in self._twist_history[-TWIST_HISTORY_LEN:])
                lines.append("recent: " + recent)
            if self._agent_state:
                lines.append("STATE: " + self._agent_state)
            lines.append("SPEECH: " + combined if combined else "SPEECH:")
            text_content = "\n".join(lines)

            # Build user message with image placeholder
            user_msg = {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": text_content},
            ]}
            self._conversation.append(user_msg)
            self._trim()

            # Build messages for processor — strip images from older turns
            messages = [{"role": "system", "content": self._prompt}]
            images_for_processor = []

            for i, msg in enumerate(self._conversation):
                c = msg.get("content")
                if isinstance(c, list):
                    is_current = (i == len(self._conversation) - 1)
                    if is_current:
                        messages.append(msg)
                        for part in c:
                            if part.get("type") == "image":
                                images_for_processor.append(
                                    Image.open(io.BytesIO(jpeg_bytes)).convert("RGB"))
                    else:
                        text_only = [p for p in c if p.get("type") != "image"]
                        if not text_only:
                            text_only = [{"type": "text", "text": "(frame)"}]
                        messages.append({"role": msg["role"], "content": text_only})
                else:
                    messages.append(msg)

            # Tokenize
            t_api = time.time()
            prompt_text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = self._processor(
                text=[prompt_text],
                images=images_for_processor if images_for_processor else None,
                padding=True,
                return_tensors="pt",
            ).to(self._model.device)

            # Generate
            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.5,
                    do_sample=True,
                )

            new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
            result = self._processor.decode(new_tokens, skip_special_tokens=True).strip()
            api_ms = (time.time() - t_api) * 1000
            self._api_ms_total += api_ms
            self._turn += 1

            if not result:
                result = "."

            if result != ".":
                if not _TWIST_RE.search(result):
                    result = "twist_for(0, 0)\n" + result
                result = self._dedup_response(result)
                m = _STATE_RE.search(result)
                if m:
                    self._agent_state = m.group(2)

            tv = _TWIST_VALS.search(result)
            if tv:
                self._twist_history.append(
                    (float(tv.group(1)), float(tv.group(2))))
                if len(self._twist_history) > TWIST_HISTORY_LEN:
                    self._twist_history = self._twist_history[-TWIST_HISTORY_LEN:]

            self._conversation.append({"role": "assistant", "content": result})

            if _HAS_KOKORO and result != ".":
                speak_match = _SPEAK_RE.search(result)
                if speak_match:
                    speak_text = speak_match.group(2)
                    self._tts_thread = threading.Thread(
                        target=self._synthesize_async, args=(speak_text,),
                        daemon=True)
                    self._tts_thread.start()

            total_ms = (time.time() - t_total) * 1000
            n_ctx = len(self._conversation)
            print("#%d  infer=%3.0fms  stt=%.0fms  total=%3.0fms  ctx=%d  %s" % (
                self._turn, api_ms, stt_ms, total_ms, n_ctx, result[:80]))

            if self._turn % 10 == 0:
                avg = self._api_ms_total / self._turn
                print("brain: turn=%d  avg_infer=%.0fms  ctx=%d  state='%s'" % (
                    self._turn, avg, n_ctx, self._agent_state[:50]))

            return result, api_ms, tts_ready, self._last_stt

    def _synthesize_async(self, text):
        try:
            t0 = time.time()
            samples, sr = _kokoro.create(text, voice="am_michael", speed=1.0)
            pcm = (samples * 32767).astype(np.int16).tobytes()
            buf = io.BytesIO()
            with wave.open(buf, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(pcm)
            self._pending_tts = base64.b64encode(buf.getvalue()).decode('ascii')
            print("tts: %.0fms  \"%s\"" % ((time.time() - t0) * 1000, text[:60]))
        except Exception as e:
            print("brain: TTS error: %s" % e)
            self._pending_tts = None

    def _dedup_response(self, result):
        speak_match = _SPEAK_RE.search(result)
        if speak_match:
            speak_text = speak_match.group(2)
            if speak_text == self._last_speak:
                result = _SPEAK_RE.sub("", result).strip()
                if not result:
                    result = "twist_for(0, 0)"
            else:
                self._last_speak = speak_text
        stripped = result.strip()
        while len(self._conversation) >= 3:
            n = len(self._conversation)
            if (self._conversation[n - 2].get("role") == "assistant" and
                    self._conversation[n - 2].get("content", "").strip() == stripped):
                del self._conversation[n - 3:n - 1]
            else:
                break
        return result

    def _trim(self):
        if len(self._conversation) > MAX_CONTEXT:
            self._conversation = self._conversation[-MAX_CONTEXT:]
            while self._conversation and self._conversation[0].get("role") != "user":
                self._conversation.pop(0)

    def reset(self):
        with self._lock:
            self._conversation = []
            self._twist_history = []
            self._agent_state = ""
            self._last_speak = ""
            self._pending_tts = None

    @property
    def turn(self):
        return self._turn


# ── Warm-up ───────────────────────────────────────────────────────

def _warmup(brain, processor):
    """One dummy inference to compile CUDA kernels / warm caches."""
    print("brain: warming up (first inference)...")
    dummy = Image.new("RGB", (384, 384), (128, 128, 128))
    buf = io.BytesIO()
    dummy.save(buf, format="JPEG", quality=60)
    brain.infer(buf.getvalue(), {"frame_id": 0})
    brain.reset()
    print("brain: warm-up done")


# ── ZMQ server loop ──────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="AnglerDroid Brain (ZMQ + Qwen2.5-VL)")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--max-pixels", type=int, default=501760,
                    help="Max vision tokens (lower=faster, default 501760)")
    ap.add_argument("--no-stt", action="store_true")
    ap.add_argument("--name", default="Kevin")
    args = ap.parse_args()

    model, processor = load_model(args.model, args.max_pixels)

    prompt = _load_prompt().replace("Kevin", args.name)
    brain = Brain(model, processor, prompt)

    if not args.no_stt:
        brain.init_stt()
    _init_kokoro()

    _warmup(brain, processor)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind("tcp://0.0.0.0:%d" % args.port)

    print("=" * 60)
    print("AnglerDroid Brain (ZMQ)")
    print("  bind:   tcp://0.0.0.0:%d" % args.port)
    print("  model:  %s" % args.model)
    print("  pixels: %d" % args.max_pixels)
    print("  stt:    %s" % (brain._stt_model is not None))
    print("  tts:    %s" % ("kokoro" if _HAS_KOKORO else "none"))
    print("=" * 60)

    while True:
        try:
            parts = sock.recv_multipart()
            jpeg_bytes = parts[0]
            metadata = json.loads(parts[1].decode("utf-8")) if len(parts) > 1 else {}
            # Audio chunks come as parts[2:]
            if len(parts) > 2:
                metadata["audio_chunks"] = [
                    base64.b64encode(p).decode("ascii") for p in parts[2:]]

            text, api_ms, tts_audio, stt_text = brain.infer(jpeg_bytes, metadata)

            reply = {"text": text, "api_ms": round(api_ms, 1),
                     "turn": brain.turn}
            if tts_audio:
                reply["tts_audio"] = tts_audio
            if stt_text:
                reply["stt_text"] = stt_text

            sock.send(json.dumps(reply).encode("utf-8"))

        except KeyboardInterrupt:
            break
        except Exception as e:
            print("brain: error: %s" % e)
            try:
                sock.send(json.dumps({"text": ".", "error": str(e)}).encode("utf-8"))
            except Exception:
                pass

    print("\nShutdown.")


if __name__ == "__main__":
    main()

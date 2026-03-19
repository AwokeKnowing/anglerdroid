"""brain_zmq.py — ZMQ brain bridge with SGLang backend for AnglerDroid.

Receives JPEG + metadata from Jetson over ZMQ, calls SGLang's
OpenAI-compatible API at localhost for vision-language inference.
Keeps all brain logic: conversation, twist tracking, dedup, STT, TTS.

Usage (3090):
  # Option A: use start_brain.sh (starts SGLang + this script)
  ./start_brain.sh

  # Option B: manual
  python -m sglang.launch_server --model-path Qwen/Qwen3.5-9B --port 30000 --enable-multimodal
  python brain_zmq.py --sglang-url http://127.0.0.1:30000 --model Qwen/Qwen3.5-9B

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
import requests
import zmq

# ── Regex for parsing LLM output ──────────────────────────────────
_TWIST_RE   = re.compile(r'twist_for\s*\(')
_TWIST_VALS = re.compile(r'twist_for\s*\(\s*([-.0-9]+)\s*,\s*([-.0-9]+)')
_SPEAK_RE   = re.compile(r'(?:speak|say)\s*\(\s*(["\'])(.*?)\1')
_STATE_RE   = re.compile(r'(?:state|set_goal)\s*\(\s*(["\'])(.*?)\1')

MAX_CONTEXT = 4
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


# ── Brain ─────────────────────────────────────────────────────────

class Brain:
    def __init__(self, sglang_url, model_name, system_prompt):
        self._sglang_url = sglang_url.rstrip("/") + "/v1/chat/completions"
        self._model = model_name
        self._http = requests.Session()
        self._http.headers["Content-Type"] = "application/json"
        self._prompt = system_prompt
        self._conversation = [
            {"role": "user", "content": [
                {"type": "text", "text": "frame: 0\nwz: 0.000 0.000\nSPEECH:"}]},
            {"role": "assistant",
             "content": 'twist_for(0.15, 0)\nset_goal("explore the living room")'},
        ]
        self._twist_history = [(0.15, 0.0)]
        self._agent_state = "explore the living room"
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
                lines.append("GOAL: " + self._agent_state)
            lines.append("SPEECH: " + combined if combined else "SPEECH:")
            text_content = "\n".join(lines)

            img_b64 = base64.b64encode(jpeg_bytes).decode("ascii")
            img_url = "data:image/jpeg;base64," + img_b64

            user_msg = {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": img_url}},
                {"type": "text", "text": text_content},
            ]}
            self._conversation.append(user_msg)
            self._trim()

            messages = [{"role": "system", "content": self._prompt}]
            for i, msg in enumerate(self._conversation):
                is_current = (i == len(self._conversation) - 1)
                c = msg.get("content")
                if isinstance(c, list) and not is_current:
                    text_only = [p for p in c if p.get("type") != "image_url"]
                    if not text_only:
                        text_only = [{"type": "text", "text": "(frame)"}]
                    messages.append({"role": msg["role"], "content": text_only})
                else:
                    messages.append(msg)

            t_api = time.time()
            result = self._call_sglang(messages)
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
                    full_call = result[speak_match.start():]
                    is_to_self = "to_self" in full_call.lower()
                    if not is_to_self:
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

    def _call_sglang(self, messages):
        body = {
            "model": self._model,
            "messages": messages,
            "max_tokens": 30,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        try:
            resp = self._http.post(self._sglang_url, json=body, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            raw = data["choices"][0]["message"]["content"].strip()
            if not raw or raw == ".":
                print("brain: raw response: %s" % json.dumps(data)[:300])
            return raw
        except requests.HTTPError as e:
            body_text = ""
            try:
                body_text = e.response.text[:500] if e.response else ""
            except Exception:
                pass
            print("brain: SGLang HTTP %d: %s" % (
                e.response.status_code if e.response else 0, body_text))
            return None
        except Exception as e:
            print("brain: SGLang error: %s" % e)
            return None

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

def _warmup(brain):
    """One dummy inference to warm SGLang caches."""
    print("brain: warming up (first inference)...")
    from PIL import Image
    dummy = Image.new("RGB", (112, 336), (128, 128, 128))
    buf = io.BytesIO()
    dummy.save(buf, format="JPEG", quality=60)
    brain.infer(buf.getvalue(), {"frame_id": 0})
    brain.reset()
    print("brain: warm-up done")


# ── ZMQ server loop ──────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="AnglerDroid Brain (ZMQ + SGLang)")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--sglang-url", default="http://127.0.0.1:30000",
                    help="SGLang server URL (default http://127.0.0.1:30000)")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B",
                    help="Model name for SGLang API (must match what SGLang loaded)")
    ap.add_argument("--no-stt", action="store_true")
    ap.add_argument("--name", default="Kevin")
    args = ap.parse_args()

    prompt = _load_prompt().replace("Kevin", args.name)
    brain = Brain(args.sglang_url, args.model, prompt)

    if not args.no_stt:
        brain.init_stt()
    _init_kokoro()

    _warmup(brain)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind("tcp://0.0.0.0:%d" % args.port)

    print("=" * 60)
    print("AnglerDroid Brain (ZMQ + SGLang)")
    print("  bind:    tcp://0.0.0.0:%d" % args.port)
    print("  sglang:  %s" % args.sglang_url)
    print("  model:   %s" % args.model)
    print("  stt:     %s" % (brain._stt_model is not None))
    print("  tts:     %s" % ("kokoro" if _HAS_KOKORO else "none"))
    print("=" * 60)

    while True:
        try:
            parts = sock.recv_multipart()
            jpeg_bytes = parts[0]
            metadata = json.loads(parts[1].decode("utf-8")) if len(parts) > 1 else {}
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

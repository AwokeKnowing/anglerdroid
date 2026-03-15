"""
brain_server.py — Local AI inference server for AnglerDroid (runs on 3090 GPU).

Receives current + previous camera frames and robot state from the Jetson
at 1 FPS via HTTP. Optionally receives audio chunks, runs STT (faster-whisper).
Uses vLLM (OpenAI-compatible API) for decision making.

Usage:
  # Terminal 1: start vLLM (adjust model as needed)
  vllm serve Qwen/Qwen2.5-VL-3B-Instruct --port 8000 \
    --max-model-len 4096 --enforce-eager --gpu-memory-utilization 0.5 \
    --limit-mm-per-prompt '{"image":2,"video":0}'

  # Terminal 2: start brain server
  python brain_server.py --port 8090

  # On Jetson:
  python main.py --brain-url http://DESKTOP_IP:8090
"""

import argparse
import base64
import json
import os
import re
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import io
import wave
import numpy as np

_kokoro = None
_HAS_KOKORO = False

def _init_kokoro():
    global _kokoro, _HAS_KOKORO
    _model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.kokoro')
    _model_file = os.path.join(_model_dir, 'kokoro-v1.0.onnx')
    _voices_file = os.path.join(_model_dir, 'voices-v1.0.bin')
    try:
        from kokoro_onnx import Kokoro
    except ImportError:
        print("tts: kokoro-onnx not installed")
        return
    if not os.path.exists(_model_file) or not os.path.exists(_voices_file):
        print("tts: downloading Kokoro model (~300 MB)...")
        os.makedirs(_model_dir, exist_ok=True)
        _GH = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
        from urllib.request import urlretrieve
        urlretrieve(_GH + "kokoro-v1.0.onnx", _model_file)
        urlretrieve(_GH + "voices-v1.0.bin", _voices_file)
        print("tts: download complete")
    _kokoro = Kokoro(_model_file, _voices_file)
    _HAS_KOKORO = True
    print("tts: Kokoro loaded")

VLLM_DEFAULT_URL = "http://localhost:8000/v1/chat/completions"
GEMINI_OPENAI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
GEMINI_DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"
GROQ_OPENAI_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_DEFAULT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
MAX_CONTEXT = 200
_STATE_RE = re.compile(r'state\s*\(\s*(["\'])(.*?)\1', re.DOTALL)
_SPEAK_RE = re.compile(r'speak\s*\(\s*(["\'])(.*?)\1', re.DOTALL)


_TWIST_VALS_RE = re.compile(r'twist_for\s*\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)')
TWIST_HISTORY_LEN = 20


class Brain:
    _TWIST_RE = re.compile(r'twist_for\s*\(')

    def __init__(self, vllm_url, model_name, system_prompt, vision=True,
                 api_key="", llm_url=""):
        self._api_key = api_key
        self._llm_url = llm_url or vllm_url
        self._model = model_name
        self._prompt = system_prompt
        self._vision = vision
        self._conversation = []
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
            print("brain: STT not available (%s) — text input only" % e)

    def transcribe_audio(self, audio_b64_chunks):
        if not self._stt_model or not audio_b64_chunks:
            return None
        try:
            pcm = b""
            for chunk in audio_b64_chunks:
                pcm += base64.b64decode(chunk)
            if len(pcm) < 3200:
                return None
            audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            segments, _ = self._stt_model.transcribe(audio)
            text = " ".join(s.text for s in segments).strip()
            if text:
                print("brain: STT → '%s'" % text[:100])
                return text
        except Exception as e:
            print("brain: STT error: %s" % e)
        return None

    def infer(self, image_b64, frame_id, velocity, angular_vel,
              speech="", audio_chunks=None, prev_image_b64=None):
        t_total = time.time()

        tts_ready = self._pending_tts
        self._pending_tts = None

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

            lines = [
                "frame: %d" % frame_id,
                "wz: %.3f %.3f" % (velocity, angular_vel),
            ]
            if self._twist_history:
                recent = " | ".join("%.2f,%.2f" % (f, a)
                                    for f, a in self._twist_history[-TWIST_HISTORY_LEN:])
                lines.append("recent: " + recent)
            if self._agent_state:
                lines.append("STATE: " + self._agent_state)
            if combined:
                lines.append("SPEECH: " + combined)
            else:
                lines.append("SPEECH:")
            text_content = "\n".join(lines)

            t_build = time.time()
            if self._vision and image_b64:
                content = []
                if prev_image_b64:
                    content.append({"type": "image_url",
                                    "image_url": {"url": "data:image/jpeg;base64," + prev_image_b64}})
                content.append({"type": "image_url",
                                "image_url": {"url": "data:image/jpeg;base64," + image_b64}})
                content.append({"type": "text", "text": text_content})
                user_msg = {"role": "user", "content": content}
            else:
                user_msg = {"role": "user", "content": text_content}

            self._conversation.append(user_msg)
            self._trim()

            messages = [{"role": "system", "content": self._prompt}]
            for i, msg in enumerate(self._conversation):
                if i < len(self._conversation) - 2 and isinstance(msg.get("content"), list):
                    text_only = [p for p in msg["content"]
                                 if p.get("type") != "image_url"]
                    if not text_only:
                        text_only = [{"type": "text", "text": "(frame)"}]
                    messages.append({"role": msg["role"], "content": text_only})
                else:
                    messages.append(msg)
            build_ms = (time.time() - t_build) * 1000

            self._log_turn(messages)

            t_api = time.time()
            result = self._call_llm(messages)
            api_ms = (time.time() - t_api) * 1000
            self._api_ms_total += api_ms
            self._turn += 1

            if not result:
                result = "."

            if result != ".":
                if not self._TWIST_RE.search(result):
                    result = "twist_for(0, 0)\n" + result

                result = self._dedup_response(result)

                m = _STATE_RE.search(result)
                if m:
                    self._agent_state = m.group(2)

            tv = _TWIST_VALS_RE.search(result)
            if tv:
                self._twist_history.append(
                    (float(tv.group(1)), float(tv.group(2))))
                if len(self._twist_history) > TWIST_HISTORY_LEN:
                    self._twist_history = self._twist_history[-TWIST_HISTORY_LEN:]

            self._conversation.append({"role": "assistant", "content": result})
            self._log_response(result)

            if _HAS_KOKORO and result != ".":
                speak_match = _SPEAK_RE.search(result)
                if speak_match:
                    speak_text = speak_match.group(2)
                    self._tts_thread = threading.Thread(
                        target=self._synthesize_async, args=(speak_text,),
                        daemon=True)
                    self._tts_thread.start()

            total_ms = (time.time() - t_total) * 1000
            n_imgs = 2 if prev_image_b64 else 1
            n_ctx = len(self._conversation)
            print("#%d  api=%3.0fms  build=%.0fms  stt=%.0fms  total=%3.0fms  "
                  "ctx=%d  imgs=%d  %s" % (
                      self._turn, api_ms, build_ms, stt_ms, total_ms,
                      n_ctx, n_imgs, result[:80]))

            if self._turn % 10 == 0:
                avg = self._api_ms_total / self._turn
                print("brain: turn=%d  avg_api=%.0fms  ctx=%d  state='%s'" % (
                    self._turn, avg, n_ctx, self._agent_state[:50]))

            return result, api_ms, tts_ready, self._last_stt

    def _synthesize_async(self, text):
        """Run TTS in background, store result for next response."""
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
        """Remove repeated speak text and collapse duplicate conversation turns."""
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

    def _log_turn(self, messages):
        """Append readable request to debug log file."""
        try:
            log_dir = os.path.dirname(os.path.abspath(__file__))
            with open(os.path.join(log_dir, "brain_debug.log"), "a") as f:
                f.write("\n=== TURN %d ===\n" % (self._turn + 1))
                for msg in messages:
                    role = msg.get("role", "?")
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        parts = []
                        for p in content:
                            if p.get("type") == "image_url":
                                parts.append("[IMAGE]")
                            elif p.get("type") == "text":
                                parts.append(p["text"])
                        f.write("[%s] %s\n" % (role, " ".join(parts)))
                    else:
                        text = content if len(content) < 500 else content[:500] + "..."
                        f.write("[%s] %s\n" % (role, text))
        except Exception:
            pass

    def _log_response(self, result):
        try:
            log_dir = os.path.dirname(os.path.abspath(__file__))
            with open(os.path.join(log_dir, "brain_debug.log"), "a") as f:
                f.write("[response] %s\n" % result)
        except Exception:
            pass

    def _trim(self):
        if len(self._conversation) > MAX_CONTEXT:
            self._conversation = self._conversation[-MAX_CONTEXT:]
            while self._conversation and self._conversation[0].get("role") != "user":
                self._conversation.pop(0)

    def _call_llm(self, messages):
        body = {
            "model": self._model,
            "messages": messages,
            "max_tokens": 100,
            "temperature": 0.5,
        }
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "AnglerDroid/1.0",
        }
        if self._api_key:
            headers["Authorization"] = "Bearer " + self._api_key
        req = Request(self._llm_url,
                      data=json.dumps(body).encode("utf-8"),
                      headers=headers)
        try:
            resp = urlopen(req, timeout=15)
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"].strip()
        except HTTPError as e:
            body_text = ""
            try:
                body_text = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            print("brain: LLM HTTP %d: %s" % (e.code, body_text[:500]))
            return None
        except Exception as e:
            print("brain: LLM error: %s" % e)
            return None

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

    @property
    def state(self):
        return self._agent_state


class Handler(BaseHTTPRequestHandler):
    brain = None

    def do_POST(self):
        if self.path != "/infer":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self.send_error(400, "Bad JSON")
            return

        text, api_ms, tts_audio, stt_text = self.brain.infer(
            image_b64=data.get("image", ""),
            frame_id=int(data.get("frame_id", 0)),
            velocity=float(data.get("velocity", 0)),
            angular_vel=float(data.get("angular_velocity", 0)),
            speech=data.get("speech", ""),
            audio_chunks=data.get("audio_chunks"),
            prev_image_b64=data.get("prev_image", ""),
        )

        result = {"text": text, "api_ms": round(api_ms, 1),
                  "turn": self.brain.turn}
        if tts_audio:
            result["tts_audio"] = tts_audio
        if stt_text:
            result["stt_text"] = stt_text
        payload = json.dumps(result).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/health":
            self._json_ok({"status": "ok", "turns": self.brain.turn,
                           "state": self.brain.state[:200]})
            return
        if self.path == "/reset":
            self.brain.reset()
            self._json_ok({"status": "reset"})
            return
        self.send_error(404)

    def _json_ok(self, obj):
        payload = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        pass


def _load_prompt():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prompt.txt')
    try:
        with open(p, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        return "You are a helpful robot. Respond with function calls or '.'"


def main():
    ap = argparse.ArgumentParser(description="AnglerDroid Brain Server (3090)")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--gemini-key", default="",
                    help="Gemini API key (uses Gemini instead of local vLLM)")
    ap.add_argument("--groq-key", default="",
                    help="Groq API key (fastest option, ~300ms TTFT)")
    ap.add_argument("--vllm-url", default=VLLM_DEFAULT_URL,
                    help="vLLM OpenAI-compatible endpoint (ignored if cloud key set)")
    ap.add_argument("--model", default="",
                    help="Model name override")
    ap.add_argument("--no-vision", action="store_true",
                    help="Don't send images (for text-only models)")
    ap.add_argument("--no-stt", action="store_true",
                    help="Disable STT")
    ap.add_argument("--name", default="Kevin",
                    help="Robot name injected into prompt (default: Kevin)")
    args = ap.parse_args()

    if args.groq_key:
        api_key = args.groq_key
        llm_url = GROQ_OPENAI_URL
        default_model = GROQ_DEFAULT_MODEL
        llm_backend = "groq"
    elif args.gemini_key:
        api_key = args.gemini_key
        llm_url = GEMINI_OPENAI_URL
        default_model = GEMINI_DEFAULT_MODEL
        llm_backend = "gemini"
    else:
        api_key = ""
        llm_url = args.vllm_url
        default_model = "Qwen/Qwen2.5-VL-3B-Instruct"
        llm_backend = "vllm"

    model = args.model or default_model
    prompt = _load_prompt().replace("Kevin", args.name)
    brain = Brain(args.vllm_url, model, prompt,
                  vision=not args.no_vision, api_key=api_key, llm_url=llm_url)

    if not args.no_stt:
        brain.init_stt()
    _init_kokoro()

    Handler.brain = brain

    server = HTTPServer(("0.0.0.0", args.port), Handler)
    print("=" * 60)
    print("AnglerDroid Brain Server")
    print("  port:   %d" % args.port)
    print("  llm:    %s (%s)" % (llm_backend, llm_url[:50]))
    print("  model:  %s" % model)
    print("  vision: %s" % (not args.no_vision))
    print("  stt:    %s" % (brain._stt_model is not None))
    print("  tts:    %s" % ("kokoro" if _HAS_KOKORO else "none"))
    print("=" * 60)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutdown.")


if __name__ == "__main__":
    main()

"""
brain_server.py — Local AI inference server for AnglerDroid (runs on 3090 GPU).

Receives camera frames + robot state from the Jetson at 0.5fps via HTTP.
Optionally receives audio chunks, runs STT (Parakeet TDT).
Uses vLLM (OpenAI-compatible API) for decision making.

Usage:
  # Terminal 1: start vLLM (adjust model as needed)
  vllm serve Qwen/Qwen3-4B-Instruct-2507 --port 8000 \
    --max-model-len 4096 --enforce-eager --gpu-memory-utilization 0.5

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

import numpy as np

VLLM_DEFAULT_URL = "http://localhost:8000/v1/chat/completions"
MAX_CONTEXT = 40
_STATE_RE = re.compile(r'state\s*\(\s*["\']([^"\']*)["\']', re.DOTALL)


class Brain:
    def __init__(self, vllm_url, model_name, system_prompt, vision=True):
        self._vllm_url = vllm_url
        self._model = model_name
        self._prompt = system_prompt
        self._vision = vision
        self._conversation = []
        self._agent_state = ""
        self._lock = threading.Lock()
        self._turn = 0
        self._api_ms_total = 0.0
        self._stt_model = None

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
              speech="", audio_chunks=None):
        with self._lock:
            stt_text = self.transcribe_audio(audio_chunks)

            combined = ""
            if speech:
                combined = speech
            if stt_text:
                combined = (combined + " " + stt_text).strip() if combined else stt_text

            lines = [
                "frame: %d" % frame_id,
                "wz: %.3f %.3f" % (velocity, angular_vel),
            ]
            if self._agent_state:
                lines.append("STATE: " + self._agent_state)
            if combined:
                lines.append("SPEECH: " + combined)
            else:
                lines.append("SPEECH:")
            text_content = "\n".join(lines)

            if self._vision and image_b64:
                user_msg = {
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": "data:image/jpeg;base64," + image_b64}},
                        {"type": "text", "text": text_content},
                    ],
                }
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

            t0 = time.time()
            result = self._call_vllm(messages)
            api_ms = (time.time() - t0) * 1000
            self._api_ms_total += api_ms
            self._turn += 1

            if result:
                self._conversation.append({"role": "assistant", "content": result})
                m = _STATE_RE.search(result)
                if m:
                    self._agent_state = m.group(1)
            else:
                self._conversation.append({"role": "assistant", "content": "."})
                result = "."

            if self._turn % 10 == 0:
                avg = self._api_ms_total / self._turn
                print("brain: turn=%d  avg_api=%.0fms  ctx=%d  state='%s'" % (
                    self._turn, avg, len(self._conversation),
                    self._agent_state[:50]))

            return result, api_ms

    def _trim(self):
        if len(self._conversation) > MAX_CONTEXT:
            self._conversation = self._conversation[-MAX_CONTEXT:]
            while self._conversation and self._conversation[0].get("role") != "user":
                self._conversation.pop(0)

    def _call_vllm(self, messages):
        body = {
            "model": self._model,
            "messages": messages,
            "max_tokens": 200,
            "temperature": 0.3,
            "top_p": 0.9,
        }
        req = Request(self._vllm_url,
                      data=json.dumps(body).encode("utf-8"),
                      headers={"Content-Type": "application/json"})
        try:
            resp = urlopen(req, timeout=10)
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"].strip()
        except HTTPError as e:
            body_text = ""
            try:
                body_text = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            print("brain: vLLM HTTP %d: %s" % (e.code, body_text[:500]))
            return None
        except Exception as e:
            print("brain: vLLM error: %s" % e)
            return None

    def reset(self):
        with self._lock:
            self._conversation = []
            self._agent_state = ""

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

        text, api_ms = self.brain.infer(
            image_b64=data.get("image", ""),
            frame_id=int(data.get("frame_id", 0)),
            velocity=float(data.get("velocity", 0)),
            angular_vel=float(data.get("angular_velocity", 0)),
            speech=data.get("speech", ""),
            audio_chunks=data.get("audio_chunks"),
        )

        result = {"text": text, "api_ms": round(api_ms, 1),
                  "turn": self.brain.turn}
        payload = json.dumps(result).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

        if text and text != ".":
            print("#%d  %.0fms  %s" % (self.brain.turn, api_ms, text[:120]))

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
    ap.add_argument("--vllm-url", default=VLLM_DEFAULT_URL,
                    help="vLLM OpenAI-compatible endpoint")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507",
                    help="Model name for vLLM (must match what vllm serve loaded)")
    ap.add_argument("--no-vision", action="store_true",
                    help="Don't send images (for text-only models)")
    ap.add_argument("--no-stt", action="store_true",
                    help="Disable Parakeet STT")
    args = ap.parse_args()

    prompt = _load_prompt()
    brain = Brain(args.vllm_url, args.model, prompt, vision=not args.no_vision)

    if not args.no_stt:
        brain.init_stt()

    Handler.brain = brain

    server = HTTPServer(("0.0.0.0", args.port), Handler)
    print("=" * 60)
    print("AnglerDroid Brain Server")
    print("  port:   %d" % args.port)
    print("  vllm:   %s" % args.vllm_url)
    print("  model:  %s" % args.model)
    print("  vision: %s" % (not args.no_vision))
    print("  stt:    %s" % (brain._stt_model is not None))
    print("=" * 60)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutdown.")


if __name__ == "__main__":
    main()

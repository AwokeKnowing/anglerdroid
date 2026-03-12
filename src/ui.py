"""
ui.py – Web UI server: HTTPS + WSS + Gemini AI (REST API, no SDK).
Auto-generates a self-signed cert so mic/getUserMedia works on LAN.
All communication with the 30fps main loop is via thread-safe queues/locks.
"""

import io
import os
import ssl
import json
import time
import wave
import base64
import queue
import subprocess
import threading
from http import HTTPStatus
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from typing import List, Optional

import cv2
import numpy as np

try:
    import websockets
    import asyncio
    _HAS_WS = True
except ImportError:
    _HAS_WS = False

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
GEMINI_MODEL_DEFAULT = "gemini-2.5-flash"
GEMINI_TTS_MODEL = "gemini-2.5-flash-preview-tts"
GEMINI_TTS_VOICE = "Umbriel"  # Kore, Charon, Fenrir, Aoede, Puck, Umbriel


def _pcm_to_wav_b64(pcm_bytes, sample_rate=24000, channels=1, sample_width=2):
    """Convert raw PCM bytes to a base64-encoded WAV string."""
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return base64.b64encode(buf.getvalue()).decode('ascii')

_CERT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.certs')
_CERT_FILE = os.path.join(_CERT_DIR, 'server.crt')
_KEY_FILE = os.path.join(_CERT_DIR, 'server.key')


def _ensure_self_signed_cert():
    """Generate a self-signed cert if one doesn't exist (needed for HTTPS → getUserMedia)."""
    if os.path.exists(_CERT_FILE) and os.path.exists(_KEY_FILE):
        return
    os.makedirs(_CERT_DIR, exist_ok=True)
    try:
        subprocess.run([
            'openssl', 'req', '-x509', '-newkey', 'rsa:2048',
            '-keyout', _KEY_FILE, '-out', _CERT_FILE,
            '-days', '365', '-nodes', '-subj', '/CN=anglerdroid',
        ], check=True, capture_output=True)
        print("ui: generated self-signed cert in %s" % _CERT_DIR)
    except Exception as e:
        print("ui: WARNING: could not generate cert (%s) — mic may not work over LAN" % e)


def _jwt_email(token):
    """Extract email from a Google ID token JWT (no crypto verification – local net only)."""
    try:
        payload = token.split('.')[1]
        pad = 4 - len(payload) % 4
        if pad < 4:
            payload += '=' * pad
        return json.loads(base64.urlsafe_b64decode(payload)).get("email", "")
    except Exception:
        return ""


class UI:
    def __init__(self, gemini_key="", gemini_model="", auth_email="",
                 google_client_id="", http_port=8080, ws_port=8081):
        self._gemini_key = gemini_key
        self._gemini_model = gemini_model or GEMINI_MODEL_DEFAULT
        self._auth_email = auth_email
        self._google_client_id = google_client_id
        self._http_port = http_port
        self._ws_port = ws_port
        self._running = False

        self._lock = threading.Lock()
        self._pending_tool_calls = []   # type: List[dict]
        self._last_user_text = ""

        self._latest_atlas = None       # type: Optional[np.ndarray]
        self._atlas_lock = threading.Lock()

        self._ws_loop = None            # type: Optional[asyncio.AbstractEventLoop]
        self._ws_clients = {}           # ws -> {"email": str, "role": str}
        self._broadcast_q = queue.Queue(maxsize=256)

        self._speech_q = queue.Queue(maxsize=64)
        self._audio_q = queue.Queue(maxsize=16)
        self._tts_q = queue.Queue(maxsize=16)

        self._gemini_active = False
        self._conversation = []         # type: List[dict]
        self._last_activity = time.time()
        self._last_gemini_send = 0.0

    # ── lifecycle ───────────────────────────────────────────────────

    def start(self):
        if self._running:
            return
        self._running = True
        _ensure_self_signed_cert()
        self._ssl_ctx = None
        if os.path.exists(_CERT_FILE) and os.path.exists(_KEY_FILE):
            self._ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            self._ssl_ctx.load_cert_chain(_CERT_FILE, _KEY_FILE)
        threading.Thread(target=self._run_http, daemon=True).start()
        if _HAS_WS:
            threading.Thread(target=self._run_ws, daemon=True).start()
        else:
            print("ui: websockets not installed – WebSocket disabled")
        proto = "HTTPS" if self._ssl_ctx else "HTTP"
        print("ui: %s :%d  WSS :%d" % (proto, self._http_port, self._ws_port))

    def stop(self):
        self._running = False
        self._gemini_active = False

    # ── main-loop interface (called from 30 fps thread) ─────────────

    def send_atlas(self, atlas_rgb):
        """Store latest atlas for Gemini and browser streaming."""
        with self._atlas_lock:
            if self._latest_atlas is None or self._latest_atlas.shape != atlas_rgb.shape:
                self._latest_atlas = atlas_rgb.copy()
            else:
                np.copyto(self._latest_atlas, atlas_rgb)

    def get_user_text(self):
        with self._lock:
            out = self._last_user_text
            self._last_user_text = ""
            return out

    def get_pending_tool_calls(self):
        with self._lock:
            out = self._pending_tool_calls[:]
            self._pending_tool_calls.clear()
            return out

    def push_tool_calls(self, calls):
        with self._lock:
            self._pending_tool_calls.extend(calls)

    # ── HTTP server ─────────────────────────────────────────────────

    def _run_http(self):
        ui_ref = self
        base_dir = os.path.dirname(os.path.abspath(__file__))

        class H(SimpleHTTPRequestHandler):
            def __init__(self, *a, **kw):
                super().__init__(*a, directory=base_dir, **kw)

            def do_GET(self):
                if self.path in ('/', '/index.html'):
                    try:
                        with open(os.path.join(base_dir, 'index.html'), 'r') as f:
                            html = f.read()
                        html = html.replace('__GOOGLE_CLIENT_ID__', ui_ref._google_client_id or '')
                        html = html.replace('__WS_PORT__', str(ui_ref._ws_port))
                        self.send_response(200)
                        self.send_header('Content-Type', 'text/html; charset=utf-8')
                        self.end_headers()
                        self.wfile.write(html.encode('utf-8'))
                    except FileNotFoundError:
                        self.send_error(404)
                    return
                super().do_GET()

            def log_message(self, fmt, *args):
                pass

        srv = HTTPServer(('0.0.0.0', self._http_port), H)
        if ui_ref._ssl_ctx:
            srv.socket = ui_ref._ssl_ctx.wrap_socket(srv.socket, server_side=True)
        srv.timeout = 0.5
        while self._running:
            try:
                srv.handle_request()
            except ssl.SSLError:
                pass
            except Exception:
                pass

    # ── WebSocket server ────────────────────────────────────────────

    def _run_ws(self):
        self._ws_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._ws_loop)
        self._ws_loop.run_until_complete(self._ws_main())

    async def _ws_accept_cert(self, path, headers):
        """Serve a simple page for non-WebSocket requests (lets browser accept the self-signed cert)."""
        if headers.get('Upgrade', '').lower() == 'websocket':
            return None
        body = (b'<html><body style="background:#111;color:#eee;font-family:sans-serif;padding:40px">'
                b'<h2>WSS certificate accepted.</h2>'
                b'<p>Close this tab and reload the main page.</p></body></html>')
        return (HTTPStatus.OK, [('Content-Type', 'text/html')], body)

    async def _ws_main(self):
        ws_kwargs = {'process_request': self._ws_accept_cert}
        if self._ssl_ctx:
            ws_kwargs['ssl'] = self._ssl_ctx
        async with websockets.serve(self._ws_handler, '0.0.0.0', self._ws_port, **ws_kwargs):
            last_atlas_bc = 0.0
            while self._running:
                # drain broadcast queue
                while not self._broadcast_q.empty():
                    try:
                        msg = self._broadcast_q.get_nowait()
                        await self._send_all(msg)
                    except queue.Empty:
                        break
                # stream atlas to browser every ~2s
                now = time.time()
                if now - last_atlas_bc > 2.0:
                    with self._atlas_lock:
                        atlas = self._latest_atlas
                    if atlas is not None:
                        try:
                            _, buf = cv2.imencode('.jpg', atlas[:, :, ::-1],
                                                  [cv2.IMWRITE_JPEG_QUALITY, 55])
                            b64 = base64.b64encode(buf.tobytes()).decode('ascii')
                            await self._send_all(json.dumps({"type": "atlas", "image": b64}))
                        except Exception:
                            pass
                    last_atlas_bc = now
                await asyncio.sleep(0.1)

    async def _ws_handler(self, ws, path='/'):
        info = {"email": "", "role": "control" if not self._google_client_id else "observer"}
        self._ws_clients[ws] = info
        try:
            await ws.send(json.dumps({
                "type": "auth_result", "role": info["role"],
                "email": "", "need_auth": bool(self._google_client_id),
            }))
            async for raw in ws:
                try:
                    data = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue
                t = data.get("type", "")

                if t == "auth":
                    email = _jwt_email(data.get("credential", ""))
                    info["email"] = email
                    info["role"] = "control" if (not self._auth_email or email == self._auth_email) else "observer"
                    await ws.send(json.dumps({
                        "type": "auth_result", "role": info["role"], "email": email,
                    }))
                    continue

                if info["role"] != "control":
                    continue

                if t == "speech":
                    text = data.get("text", "").strip()
                    if text:
                        try:
                            self._speech_q.put_nowait(text)
                        except queue.Full:
                            pass
                        self._last_activity = time.time()
                        self._broadcast({"type": "chat", "sender": "user", "text": text})

                elif t == "twist_for":
                    self.push_tool_calls([{"name": "twist_for", "args": {
                        "forward_mps": float(data.get("forward_mps", 0)),
                        "angular_rads": float(data.get("angular_rads", 0)),
                        "duration_secs": float(data.get("duration_secs", 2.0)),
                        "ramp_in_secs": float(data.get("ramp_in_secs", 1.0)),
                        "ramp_out_secs": float(data.get("ramp_out_secs", 1.0)),
                    }}])
                    self._broadcast({"type": "chat", "sender": "sys", "text": "twist_for sent"})

                elif t == "audio":
                    audio_b64 = data.get("data", "")
                    mime = data.get("mime", "audio/webm")
                    if audio_b64:
                        try:
                            self._audio_q.put_nowait({"data": audio_b64, "mime": mime})
                        except queue.Full:
                            pass
                        self._last_activity = time.time()

                elif t == "stop":
                    self.push_tool_calls([{"name": "stop", "args": {}}])
                    self._broadcast({"type": "chat", "sender": "sys", "text": "stop sent"})

                elif t == "navigate":
                    heading = data.get("heading_deg")
                    if heading is not None:
                        self.push_tool_calls([{"name": "navigate",
                                               "args": {"heading_deg": float(heading)}}])
                        self._broadcast({"type": "chat", "sender": "user",
                                         "text": "[NAV] heading %s°" % heading})

                elif t == "nav_stop":
                    self.push_tool_calls([{"name": "stop", "args": {}}])
                    self._broadcast({"type": "chat", "sender": "sys",
                                     "text": "Navigation stopped"})

                elif t == "launch_ai":
                    self._start_gemini()

                elif t == "stop_ai":
                    self._stop_gemini()

        except websockets.ConnectionClosed:
            pass
        finally:
            self._ws_clients.pop(ws, None)

    async def _send_all(self, msg):
        for ws in list(self._ws_clients):
            try:
                await ws.send(msg)
            except Exception:
                self._ws_clients.pop(ws, None)

    def _broadcast(self, data_dict):
        """Thread-safe broadcast (any thread → WS loop)."""
        try:
            self._broadcast_q.put_nowait(json.dumps(data_dict))
        except queue.Full:
            pass

    # ── Gemini AI ───────────────────────────────────────────────────

    MIN_API_INTERVAL = 6.0  # seconds between API calls (free tier: 10 RPM)

    def _start_gemini(self):
        if self._gemini_active:
            return
        if not self._gemini_key:
            self._broadcast({"type": "ai_status", "active": False,
                             "error": "No --gemini-key configured"})
            return
        self._gemini_active = True
        self._conversation = []
        self._last_activity = time.time()
        self._last_api_call = 0.0
        threading.Thread(target=self._gemini_loop, daemon=True).start()
        threading.Thread(target=self._tts_loop, daemon=True).start()
        self._broadcast({"type": "ai_status", "active": True})
        print("ui: Gemini AI started (%s, TTS: %s)" % (self._gemini_model, GEMINI_TTS_MODEL))

    def _stop_gemini(self):
        self._gemini_active = False
        self._broadcast({"type": "ai_status", "active": False})
        print("ui: Gemini AI stopped")

    def _load_prompt(self):
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prompt.txt')
        try:
            with open(p, 'r') as f:
                return f.read().strip()
        except FileNotFoundError:
            return "You are a helpful robot."

    def _gemini_loop(self):
        prompt = self._load_prompt()
        speech_buf = []
        audio_buf = []
        buf_start = 0.0
        BATCH_DELAY = 2.0

        while self._running and self._gemini_active:
            try:
                while True:
                    try:
                        speech_buf.append(self._speech_q.get_nowait())
                        if buf_start == 0.0:
                            buf_start = time.time()
                    except queue.Empty:
                        break

                while True:
                    try:
                        audio_buf.append(self._audio_q.get_nowait())
                        if buf_start == 0.0:
                            buf_start = time.time()
                    except queue.Empty:
                        break

                now = time.time()
                has_input = speech_buf or audio_buf

                if has_input and (now - buf_start >= BATCH_DELAY):
                    text = " ".join(speech_buf) if speech_buf else None
                    audio = audio_buf[:] if audio_buf else None
                    speech_buf = []
                    audio_buf = []
                    buf_start = 0.0
                    if self._latest_atlas is not None:
                        self._send_to_gemini(prompt, user_text=text, audio_parts=audio)
                        self._last_activity = now

                if now - self._last_activity > 900.0:
                    self._conversation = []
                    self._last_activity = now
                    self._broadcast({"type": "ai_status", "active": True,
                                     "info": "Session reset (15 min idle)"})

                time.sleep(0.2)
            except Exception as e:
                print("ui: gemini loop error: %s" % e)
                self._conversation = []
                self._last_activity = time.time()
                time.sleep(3.0)

    @staticmethod
    def _strip_images(conversation):
        """Remove inline_data (images) from all but the last user message."""
        last_user_idx = -1
        for i in range(len(conversation) - 1, -1, -1):
            if conversation[i].get("role") == "user":
                has_img = any("inline_data" in p for p in conversation[i].get("parts", []))
                if has_img:
                    last_user_idx = i
                    break
        for i, msg in enumerate(conversation):
            if i == last_user_idx:
                continue
            if msg.get("role") == "user":
                msg["parts"] = [p for p in msg.get("parts", []) if "inline_data" not in p]
                if not msg["parts"]:
                    msg["parts"] = [{"text": "(image removed from history)"}]

    def _send_to_gemini(self, system_prompt, user_text=None, audio_parts=None):
        with self._atlas_lock:
            atlas = self._latest_atlas
        if atlas is None:
            return

        _, buf = cv2.imencode('.jpg', atlas[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 60])
        img_b64 = base64.b64encode(buf.tobytes()).decode('ascii')

        parts = [
            {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}},
        ]
        if audio_parts:
            for ap in audio_parts:
                parts.append({"inline_data": {"mime_type": ap["mime"], "data": ap["data"]}})
            print("ui: sending %d audio chunk(s) to Gemini" % len(audio_parts))

        if user_text:
            parts.append({"text": user_text})
        elif audio_parts:
            parts.append({"text": "[User sent audio. Listen to it and respond to what they said.]"})
        else:
            parts.append({"text": "[Camera update. Respond only if safety concern.]"})
        self._conversation.append({"role": "user", "parts": parts})
        if len(self._conversation) > 20:
            self._conversation = self._conversation[-20:]
        self._strip_images(self._conversation)

        result = self._call_api(system_prompt)
        if not result:
            return

        cands = result.get("candidates", [])
        if not cands:
            return
        resp_parts = cands[0].get("content", {}).get("parts", [])

        model_parts = []
        func_calls = []
        for p in resp_parts:
            if "text" in p:
                model_parts.append({"text": p["text"]})
                self._broadcast({"type": "chat", "sender": "ai", "text": p["text"]})
                self._last_activity = time.time()
                if p["text"] and not p["text"].startswith("["):
                    try:
                        self._tts_q.put_nowait(p["text"])
                    except queue.Full:
                        pass
            if "functionCall" in p:
                fc = p["functionCall"]
                model_parts.append({"functionCall": fc})
                func_calls.append(fc)
                self.push_tool_calls([{"name": fc["name"], "args": fc.get("args", {})}])
                self._broadcast({"type": "chat", "sender": "ai",
                                 "text": "[%s(%s)]" % (fc["name"], json.dumps(fc.get("args", {})))})
                self._last_activity = time.time()

        if model_parts:
            self._conversation.append({"role": "model", "parts": model_parts})

        # Queue function responses for next turn (no immediate follow-up call)
        if func_calls:
            fr = [{"functionResponse": {"name": fc["name"],
                   "response": {"result": "sent to robot"}}} for fc in func_calls]
            self._conversation.append({"role": "user", "parts": fr})

    # ── Gemini TTS ────────────────────────────────────────────────

    def _tts_loop(self):
        print("ui: TTS worker started (voice: %s)" % GEMINI_TTS_VOICE)
        while self._running and self._gemini_active:
            try:
                text = self._tts_q.get(timeout=1.0)
            except queue.Empty:
                continue
            print("ui: TTS generating for: %s" % text[:80])
            try:
                self._call_tts(text)
            except Exception as e:
                print("ui: TTS error: %s" % e)

    def _call_tts(self, text):
        body = {
            "contents": [{"parts": [{"text": text}]}],
            "generationConfig": {
                "response_modalities": ["AUDIO"],
                "speech_config": {
                    "voice_config": {
                        "prebuilt_voice_config": {
                            "voice_name": GEMINI_TTS_VOICE,
                        }
                    }
                },
            },
        }
        url = GEMINI_URL.format(model=GEMINI_TTS_MODEL, key=self._gemini_key)
        req = Request(url, data=json.dumps(body).encode('utf-8'),
                      headers={"Content-Type": "application/json"})
        try:
            resp = urlopen(req, timeout=30)
            raw = resp.read().decode('utf-8')
            result = json.loads(raw)
        except HTTPError as e:
            body_text = ""
            try:
                body_text = e.read().decode('utf-8', errors='replace')
            except Exception:
                pass
            print("ui: TTS HTTP %d: %s" % (e.code, body_text[:500]))
            return
        except Exception as e:
            print("ui: TTS request error: %s" % e)
            return

        cands = result.get("candidates", [])
        if not cands:
            print("ui: TTS no candidates. Keys: %s" % list(result.keys()))
            if "error" in result:
                print("ui: TTS error: %s" % result["error"])
            return
        parts = cands[0].get("content", {}).get("parts", [])
        if not parts:
            print("ui: TTS no parts in response")
            return
        for p in parts:
            inline = p.get("inlineData") or p.get("inline_data") or {}
            if inline.get("data"):
                pcm_b64 = inline["data"]
                pcm_bytes = base64.b64decode(pcm_b64)
                wav_b64 = _pcm_to_wav_b64(pcm_bytes)
                self._broadcast({"type": "tts_audio", "audio": wav_b64})
                print("ui: TTS sent %d KB wav" % (len(wav_b64) * 3 // 4 // 1024))
                return
            else:
                print("ui: TTS part keys: %s" % list(p.keys()))

    def _call_api(self, system_prompt):
        # Rate limit: enforce minimum interval between calls
        now = time.time()
        wait = self.MIN_API_INTERVAL - (now - self._last_api_call)
        if wait > 0:
            time.sleep(wait)

        tools_def = [{"function_declarations": [
            {"name": "navigate",
             "description": "Navigate toward a heading. The robot avoids obstacles automatically. 0=forward, 90=left, -90=right, 180=backward. Use this for all movement. Call stop() to halt.",
             "parameters": {"type": "object", "properties": {
                 "heading_deg": {"type": "number",
                                 "description": "Goal heading in degrees. 0=forward, 90=left, -90=right, 180=backward"},
             }, "required": ["heading_deg"]}},
            {"name": "twist_for",
             "description": "Direct timed move without obstacle avoidance. forward_mps (m/s), angular_rads (rad/s, +left). Only for precise adjustments where navigate() is too coarse.",
             "parameters": {"type": "object", "properties": {
                 "forward_mps": {"type": "number"},
                 "angular_rads": {"type": "number"},
                 "duration_secs": {"type": "number"},
                 "ramp_in_secs": {"type": "number"},
                 "ramp_out_secs": {"type": "number"},
             }, "required": ["forward_mps", "angular_rads"]}},
            {"name": "stop",
             "description": "Immediately stop all motion and navigation",
             "parameters": {"type": "object", "properties": {}}},
        ]}]

        body = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": self._conversation,
            "tools": tools_def,
            "tool_config": {"function_calling_config": {"mode": "AUTO"}},
        }

        url = GEMINI_URL.format(model=self._gemini_model, key=self._gemini_key)
        req = Request(url, data=json.dumps(body).encode('utf-8'),
                      headers={"Content-Type": "application/json"})

        for attempt in range(3):
            self._last_api_call = time.time()
            try:
                resp = urlopen(req, timeout=30)
                return json.loads(resp.read().decode('utf-8'))
            except HTTPError as e:
                body_text = ""
                try:
                    body_text = e.read().decode('utf-8', errors='replace')
                except Exception:
                    pass
                print("ui: Gemini HTTP %d: %s" % (e.code, body_text[:500]))
                if e.code == 429 and attempt < 2:
                    backoff = (attempt + 1) * 10
                    print("ui: rate limited, retrying in %ds..." % backoff)
                    time.sleep(backoff)
                    continue
                return None
            except Exception as e:
                print("ui: Gemini API error: %s" % e)
                return None
        return None

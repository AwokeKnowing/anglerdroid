"""
ui.py – Web UI server: HTTPS + WSS + AI control loop.
Supports two backends:
  - Gemini API (cloud) — default
  - Brain server (local 3090) — via --brain-url
Auto-generates a self-signed cert so mic/getUserMedia works on LAN.
"""

import io
import os
import re
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
    from kokoro_onnx import Kokoro as _KokoroClass
    _kokoro = _KokoroClass()
    _HAS_KOKORO = True
except Exception:
    _kokoro = None
    _HAS_KOKORO = False

try:
    import websockets
    import asyncio
    _HAS_WS = True
except ImportError:
    _HAS_WS = False

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
GEMINI_MODEL_DEFAULT = "gemini-3.1-flash-lite-preview"
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
                 google_client_id="", http_port=8080, ws_port=8081,
                 brain_url=""):
        self._gemini_key = gemini_key
        self._gemini_model = gemini_model or GEMINI_MODEL_DEFAULT
        self._brain_url = brain_url.rstrip("/") if brain_url else ""
        self._auth_email = auth_email
        self._google_client_id = google_client_id
        self._http_port = http_port
        self._ws_port = ws_port
        self._running = False

        self._lock = threading.Lock()
        self._pending_tool_calls = []   # type: List[dict]
        self._last_user_text = ""

        self._latest_atlas = None       # type: Optional[np.ndarray]
        self._atlas_jpeg = None         # type: Optional[bytes]
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
        self._agent_state = ""

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
        """Store latest atlas for Gemini; encode JPEG for browser streaming."""
        _, buf = cv2.imencode('.jpg', atlas_rgb[:, :, ::-1],
                              [cv2.IMWRITE_JPEG_QUALITY, 70])
        jpeg = buf.tobytes()
        with self._atlas_lock:
            if self._latest_atlas is None or self._latest_atlas.shape != atlas_rgb.shape:
                self._latest_atlas = atlas_rgb.copy()
            else:
                np.copyto(self._latest_atlas, atlas_rgb)
            self._atlas_jpeg = jpeg

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
                now = time.time()
                if now - last_atlas_bc > 0.033:
                    with self._atlas_lock:
                        jpeg = self._atlas_jpeg
                    if jpeg is not None:
                        try:
                            await self._send_all(jpeg)
                        except Exception:
                            pass
                    last_atlas_bc = now
                await asyncio.sleep(0.008)

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

    # ── Gemini AI (0.5 FPS reactive control loop) ──────────────────

    GEMINI_INTERVAL = 2.0   # seconds between frames (0.5 FPS)
    GEMINI_MAX_CONTEXT = 40

    _CALL_RE = re.compile(r'(twist_for|speak|think|state|stop|navigate)\s*\(([^)]*)\)')
    _NUM_RE = re.compile(r'-?[\d.]+')

    def _start_gemini(self):
        if self._gemini_active:
            return
        if not self._brain_url and not self._gemini_key:
            self._broadcast({"type": "ai_status", "active": False,
                             "error": "No --gemini-key or --brain-url configured"})
            return
        self._gemini_active = True
        self._conversation = []
        self._agent_state = ""
        self._last_api_call = 0.0
        threading.Thread(target=self._ai_loop, daemon=True).start()
        threading.Thread(target=self._tts_loop, daemon=True).start()
        self._broadcast({"type": "ai_status", "active": True})
        backend = "brain:%s" % self._brain_url if self._brain_url else self._gemini_model
        print("ai: started (%s, interval=%.1fs)" % (backend, self.GEMINI_INTERVAL))

    def _stop_gemini(self):
        self._gemini_active = False
        self._agent_state = ""
        self.push_tool_calls([{"name": "stop", "args": {}}])
        self._broadcast({"type": "ai_status", "active": False})
        print("ai: stopped")

    def _load_prompt(self):
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prompt.txt')
        try:
            with open(p, 'r') as f:
                return f.read().strip()
        except FileNotFoundError:
            return "You are a helpful robot. Respond with function calls or '.'"

    def _ai_loop(self):
        prompt = self._load_prompt()
        turns = 0
        on_time = 0
        streak = 0
        best_streak = 0
        api_ms_total = 0.0

        while self._running and self._gemini_active:
            t_loop = time.time()
            try:
                user_text = None
                while not self._speech_q.empty():
                    try:
                        t = self._speech_q.get_nowait()
                        user_text = (user_text + " " + t) if user_text else t
                    except queue.Empty:
                        break

                audio_parts = []
                while not self._audio_q.empty():
                    try:
                        audio_parts.append(self._audio_q.get_nowait())
                    except queue.Empty:
                        break

                with self._atlas_lock:
                    atlas = self._latest_atlas
                if atlas is None:
                    time.sleep(0.5)
                    continue

                _, buf = cv2.imencode('.jpg', atlas[:, :, ::-1],
                                      [cv2.IMWRITE_JPEG_QUALITY, 60])
                img_b64 = base64.b64encode(buf.tobytes()).decode('ascii')

                t_api = time.time()

                if self._brain_url:
                    text = self._call_brain(
                        img_b64, turns,
                        speech=user_text or "",
                        audio_chunks=[ap["data"] for ap in audio_parts]
                                     if audio_parts else None)
                else:
                    text = self._call_gemini_turn(
                        img_b64, user_text, audio_parts, prompt)

                api_ms = (time.time() - t_api) * 1000
                api_ms_total += api_ms
                turns += 1

                if text:
                    self._parse_and_execute(text)

                elapsed = time.time() - t_loop
                if elapsed <= self.GEMINI_INTERVAL * 1.5:
                    on_time += 1
                    streak += 1
                    best_streak = max(best_streak, streak)
                else:
                    streak = 0

                if turns % 10 == 0:
                    avg = api_ms_total / turns
                    pct = on_time / turns * 100
                    mode = "brain" if self._brain_url else "gemini"
                    print("ai[%s]: turn=%d  avg=%.0fms  on_time=%.0f%%  "
                          "streak=%d/%d" % (
                              mode, turns, avg, pct, streak, best_streak))

                remaining = self.GEMINI_INTERVAL - (time.time() - t_loop)
                while remaining > 0 and self._gemini_active:
                    time.sleep(min(0.2, remaining))
                    remaining = self.GEMINI_INTERVAL - (time.time() - t_loop)

            except Exception as e:
                print("ai: loop error: %s" % e)
                import traceback
                traceback.print_exc()
                self._conversation = []
                time.sleep(3.0)

    # ── Brain server backend ─────────────────────────────────────

    def _call_brain(self, img_b64, frame_id, speech="", audio_chunks=None):
        """POST frame + metadata to brain server, return response text."""
        body = {
            "image": img_b64,
            "frame_id": frame_id,
            "velocity": 0.0,
            "angular_velocity": 0.0,
            "speech": speech,
        }
        if audio_chunks:
            body["audio_chunks"] = audio_chunks

        req = Request(self._brain_url + "/infer",
                      data=json.dumps(body).encode('utf-8'),
                      headers={"Content-Type": "application/json"})
        try:
            resp = urlopen(req, timeout=10)
            data = json.loads(resp.read())
            return data.get("text", ".")
        except Exception as e:
            print("ai: brain error: %s" % e)
            return None

    # ── Gemini API backend ───────────────────────────────────────

    def _call_gemini_turn(self, img_b64, user_text, audio_parts, prompt):
        """Gemini mode: manage conversation, call API, return response text."""
        parts = [{"inline_data": {"mime_type": "image/jpeg", "data": img_b64}}]
        if audio_parts:
            for ap in audio_parts:
                parts.append({"inline_data": {
                    "mime_type": ap["mime"], "data": ap["data"]}})

        text_lines = []
        if user_text:
            text_lines.append("USER: " + user_text)
        elif audio_parts:
            text_lines.append("USER: [audio message]")
        if self._agent_state:
            text_lines.append("STATE: " + self._agent_state)
        text_lines.append("frame")
        parts.append({"text": "\n".join(text_lines)})

        self._conversation.append({"role": "user", "parts": parts})
        self._trim_conversation()
        self._strip_images()

        result = self._call_api(prompt)
        if not result:
            return None

        cands = result.get("candidates", [])
        if not cands:
            return None
        resp_parts = cands[0].get("content", {}).get("parts", [])
        text = ""
        for p in resp_parts:
            if "text" in p and not p.get("thought"):
                text += p["text"]
        text = text.strip()
        if text:
            self._conversation.append(
                {"role": "model", "parts": [{"text": text}]})
        else:
            self._conversation.append(
                {"role": "model", "parts": [{"text": "."}]})
            text = "."
        return text

    def _trim_conversation(self):
        if len(self._conversation) <= self.GEMINI_MAX_CONTEXT:
            return
        self._conversation = self._conversation[-self.GEMINI_MAX_CONTEXT:]
        while self._conversation and self._conversation[0].get("role") != "user":
            self._conversation.pop(0)

    def _strip_images(self):
        """Keep images only in the last 2 user messages."""
        img_indices = []
        for i, msg in enumerate(self._conversation):
            if msg.get("role") == "user" and \
               any("inline_data" in p for p in msg.get("parts", [])):
                img_indices.append(i)
        for idx in img_indices[:-2]:
            msg = self._conversation[idx]
            msg["parts"] = [p for p in msg["parts"] if "inline_data" not in p]
            if not msg["parts"]:
                msg["parts"] = [{"text": "(frame)"}]

    @staticmethod
    def _parse_string_arg(s):
        s = s.strip()
        for q in ('"', "'"):
            if s.startswith(q) and s.endswith(q) and len(s) > 1:
                return s[1:-1]
        return s

    def _parse_and_execute(self, text):
        if text == ".":
            return

        calls = self._CALL_RE.findall(text)
        if not calls:
            if text and text != ".":
                self._broadcast({"type": "chat", "sender": "ai",
                                 "text": text[:200]})
            return

        log_parts = []
        for name, raw_args in calls:
            if name == "twist_for":
                nums = [float(x) for x in self._NUM_RE.findall(raw_args)]
                fwd = nums[0] if len(nums) > 0 else 0.0
                ang = nums[1] if len(nums) > 1 else 0.0
                dur = nums[2] if len(nums) > 2 else self.GEMINI_INTERVAL + 0.5
                self.push_tool_calls([{"name": "twist_for", "args": {
                    "forward_mps": fwd, "angular_rads": ang,
                    "duration_secs": dur,
                    "ramp_in_secs": 0.0, "ramp_out_secs": 0.0,
                }}])
                log_parts.append("twist_for(%.2f, %.2f)" % (fwd, ang))

            elif name == "speak":
                msg = self._parse_string_arg(raw_args)
                if msg:
                    try:
                        self._tts_q.put_nowait(msg)
                    except queue.Full:
                        pass
                    self._broadcast({"type": "chat", "sender": "ai",
                                     "text": msg})
                    log_parts.append("speak")

            elif name == "think":
                thought = self._parse_string_arg(raw_args)
                log_parts.append('think("%s")' % thought[:50])

            elif name == "state":
                self._agent_state = self._parse_string_arg(raw_args)
                log_parts.append('state("%s")' % self._agent_state[:50])

            elif name == "stop":
                self.push_tool_calls([{"name": "stop", "args": {}}])
                log_parts.append("stop()")

            elif name == "navigate":
                nums = [float(x) for x in self._NUM_RE.findall(raw_args)]
                hdg = nums[0] if nums else 0.0
                self.push_tool_calls([{"name": "navigate",
                                       "args": {"heading_deg": hdg}}])
                log_parts.append("navigate(%.0f)" % hdg)

        if log_parts:
            summary = " | ".join(log_parts)
            print("ai: " + summary)

    # ── Gemini TTS ────────────────────────────────────────────────

    def _tts_loop(self):
        if _HAS_KOKORO:
            print("tts: Kokoro ONNX loaded")
        elif self._gemini_key:
            print("tts: Gemini TTS (voice: %s)" % GEMINI_TTS_VOICE)
        else:
            print("tts: no TTS backend available")
            return

        while self._running and self._gemini_active:
            try:
                text = self._tts_q.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                if _HAS_KOKORO:
                    self._call_kokoro_tts(text)
                elif self._gemini_key:
                    self._call_tts(text)
            except Exception as e:
                print("tts: error: %s" % e)

    def _call_kokoro_tts(self, text):
        samples, sr = _kokoro.create(text, voice="af_heart", speed=1.0)
        pcm = (samples * 32767).astype(np.int16).tobytes()
        wav_b64 = _pcm_to_wav_b64(pcm, sample_rate=sr)
        self._broadcast({"type": "tts_audio", "audio": wav_b64})
        print("tts: kokoro sent %d KB" % (len(wav_b64) * 3 // 4 // 1024))

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
        body = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": self._conversation,
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 200,
            },
        }

        url = GEMINI_URL.format(model=self._gemini_model, key=self._gemini_key)
        req = Request(url, data=json.dumps(body).encode('utf-8'),
                      headers={"Content-Type": "application/json"})

        self._last_api_call = time.time()
        try:
            resp = urlopen(req, timeout=15)
            return json.loads(resp.read().decode('utf-8'))
        except HTTPError as e:
            body_text = ""
            try:
                body_text = e.read().decode('utf-8', errors='replace')
            except Exception:
                pass
            print("ai: Gemini HTTP %d: %s" % (e.code, body_text[:500]))
            if e.code == 429:
                time.sleep(10)
            return None
        except Exception as e:
            print("ai: Gemini error: %s" % e)
            return None

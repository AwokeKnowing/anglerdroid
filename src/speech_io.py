"""speech_io.py – Local ears/mouth for Kevin (ReSpeaker + Kokoro).

TTS: Kokoro-ONNX → paplay on ReSpeaker sink.
ASR: faster-whisper now (Parakeet on i777 3090 is the upgrade path).

Volume: apply ONCE via paplay --volume. Do not also scale samples and
pactl-set the sink to the same fraction (that made 30% effectively ~3%).

Outside TTS + camera stills (LAN):
    POST /say  {"text": "hello"}     or  GET /say?text=hello
  GET  /look.jpg                   current RGB webcam JPEG
  GET  /look.jpg?cam=webcam|rs1|rs2|atlas|all
  GET  /health
  GET  /heard                      recent ASR transcripts
  GET  /status                     pose, safety, bag, topdown
  POST /goal  {"type": "relative_bearing", "bearing_deg": 70, "distance_m": 0.3}
  POST /twist {"forward_mps", "angular_rads", "duration_secs"}
  POST /stop
Default port 8091 (KEVIN_SAY_PORT). apply_speak_sink() at launch + a user
systemd unit keep the ReSpeaker as the default sink across reboots.
"""

from __future__ import annotations

import collections
import json
import os
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

KOKORO_MODEL = os.path.expanduser("~/.kevin/models/kokoro/kokoro-v1.0.onnx")
KOKORO_VOICES = os.path.expanduser("~/.kevin/models/kokoro/voices-v1.0.bin")
DEFAULT_SINK = (
    "alsa_output.usb-SEEED_ReSpeaker_4_Mic_Array__UAC1.0_-00.analog-stereo"
)
RS_SINK = os.environ.get("KEVIN_SPEAK_SINK", DEFAULT_SINK)
DEFAULT_SRC = (
    "alsa_input.usb-SEEED_ReSpeaker_4_Mic_Array__UAC1.0_-00.analog-surround-21"
)
RS_SRC = os.environ.get("KEVIN_LISTEN_SRC", DEFAULT_SRC)
SPEAK_VOLUME = float(os.environ.get("KEVIN_SPEAK_VOL", "0.70"))
SAY_PORT = int(os.environ.get("KEVIN_SAY_PORT", "8091"))
SINK_HINTS = ("SEEED_ReSpeaker", "ReSpeaker", "ArrayUAC10")
HEARD_PATH = os.path.expanduser("~/.kevin/heard.jsonl")

_WP_LUA = os.path.expanduser(
    "~/.config/wireplumber/main.lua.d/51-kevin-respeaker.lua"
)
_USER_UNIT = os.path.expanduser("~/.config/systemd/user/kevin-audio.service")

_lock = threading.Lock()
_kokoro = None
_whisper = None
_speak_busy = False
_say_q = collections.deque(maxlen=6)
_worker = None
_http_srv = None
_heard = collections.deque(maxlen=40)
_heard_lock = threading.Lock()
_nav_cb = None


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


def pick_respeaker_sink(pactl_short: str, fallback: str = DEFAULT_SINK) -> str:
    """Choose the ReSpeaker playback sink from `pactl list short sinks`."""
    found = []
    for line in (pactl_short or "").splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        name = parts[1]
        if any(h.lower() in name.lower() for h in SINK_HINTS):
            found.append(name)
    if found:
        return found[0]
    return fallback


def pick_respeaker_src(pactl_short: str, fallback: str = DEFAULT_SRC) -> str:
    """Choose ReSpeaker capture. Never the playback .monitor (that is TTS echo)."""
    found = []
    for line in (pactl_short or "").splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        name = parts[1]
        if ".monitor" in name.lower():
            continue
        if not any(h.lower() in name.lower() for h in SINK_HINTS):
            continue
        found.append(name)
    inputs = [n for n in found if "alsa_input" in n or "input" in n.lower()]
    if inputs:
        return inputs[0]
    if found:
        return found[0]
    return fallback


def discover_speak_sink() -> str:
    """Live Pulse/PipeWire sink name for the ReSpeaker, else the default."""
    env = (os.environ.get("KEVIN_SPEAK_SINK") or "").strip()
    if env:
        return env
    try:
        r = subprocess.run(
            ["pactl", "list", "short", "sinks"],
            check=False, timeout=3, capture_output=True, text=True,
        )
        return pick_respeaker_sink(r.stdout or "")
    except Exception:
        return DEFAULT_SINK


def _pactl(*args) -> None:
    subprocess.run(["pactl", *args], check=False, timeout=2,
                   capture_output=True)


def apply_speak_sink(persist: bool = True, quiet: bool = False) -> str:
    """Unmute ReSpeaker, set it default, optionally persist across reboots."""
    global RS_SINK
    sink = discover_speak_sink()
    RS_SINK = sink
    try:
        _pactl("set-default-sink", sink)
        _pactl("set-sink-mute", sink, "0")
        _pactl("set-sink-volume", sink, "100%")
    except Exception as e:
        print("speech: pactl sink apply skip: %s" % e)
    if not quiet:
        print("speech: sink %s (default + unmuted + 100%%)" % sink)
    if persist:
        try:
            persist_speak_sink(sink)
        except Exception as e:
            print("speech: persist skip: %s" % e)
    return sink


def persist_speak_sink(sink: str) -> None:
    """WirePlumber 0.4 lua + user systemd oneshot so reboot keeps ReSpeaker."""
    lua_dir = os.path.dirname(_WP_LUA)
    os.makedirs(lua_dir, exist_ok=True)
    lua = """-- Kevin: prefer ReSpeaker over muted onboard analog.
alsa_monitor = alsa_monitor or {}
alsa_monitor.rules = alsa_monitor.rules or {}
table.insert(alsa_monitor.rules, {
  matches = {
    { { "node.name", "matches", "*SEEED_ReSpeaker*" } },
  },
  apply_properties = {
    ["priority.session"] = 2500,
    ["priority.driver"] = 2500,
  },
})
table.insert(alsa_monitor.rules, {
  matches = {
    { { "node.name", "matches", "*platform-sound*" } },
  },
  apply_properties = {
    ["priority.session"] = 1,
  },
})
"""
    with open(_WP_LUA, "w", encoding="utf-8") as f:
        f.write(lua)

    unit_dir = os.path.dirname(_USER_UNIT)
    os.makedirs(unit_dir, exist_ok=True)
    here = os.path.abspath(__file__)
    py = os.path.expanduser("~/anglerdroid/.venv/bin/python")
    if not os.path.isfile(py):
        py = os.path.expanduser("~/.venv/bin/python")
    if not os.path.isfile(py):
        py = "python3"
    unit = """[Unit]
Description=Kevin ReSpeaker default sink
After=pipewire.service pipewire-pulse.service wireplumber.service
Wants=pipewire-pulse.service

[Service]
Type=oneshot
ExecStart=%s %s --setup-sink
RemainAfterExit=yes
Environment=XDG_RUNTIME_DIR=/run/user/%%U

[Install]
WantedBy=default.target
""" % (py, here)
    with open(_USER_UNIT, "w", encoding="utf-8") as f:
        f.write(unit)
    subprocess.run(
        ["systemctl", "--user", "daemon-reload"],
        check=False, timeout=5, capture_output=True,
    )
    subprocess.run(
        ["systemctl", "--user", "enable", "kevin-audio.service"],
        check=False, timeout=5, capture_output=True,
    )
    print("speech: persisted sink via %s and %s" % (_WP_LUA, _USER_UNIT))


def current_sink() -> str:
    return RS_SINK


def discover_listen_src() -> str:
    """Live Pulse/PipeWire source name for the ReSpeaker, else the default."""
    env = (os.environ.get("KEVIN_LISTEN_SRC") or "").strip()
    if env:
        return env
    try:
        r = subprocess.run(
            ["pactl", "list", "short", "sources"],
            check=False, timeout=3, capture_output=True, text=True,
        )
        return pick_respeaker_src(r.stdout or "")
    except Exception:
        return DEFAULT_SRC


def apply_listen_source(quiet: bool = False) -> str:
    """Unmute ReSpeaker capture and set it as the default source."""
    global RS_SRC
    src = discover_listen_src()
    RS_SRC = src
    try:
        _pactl("set-default-source", src)
        _pactl("set-source-mute", src, "0")
        _pactl("set-source-volume", src, "100%")
    except Exception as e:
        print("speech: pactl source apply skip: %s" % e)
    if not quiet:
        print("speech: source %s (default + unmuted)" % src)
    return src


def current_src() -> str:
    return RS_SRC


def set_nav_callback(fn) -> None:
    """people_live registers apply-goal here so POST /goal can drive."""
    global _nav_cb
    _nav_cb = fn


def note_heard(text: str, extra=None) -> dict:
    rec = {"t": time.time(), "text": text}
    if extra:
        rec.update(extra)
    with _heard_lock:
        _heard.append(rec)
    try:
        os.makedirs(os.path.dirname(HEARD_PATH), exist_ok=True)
        with open(HEARD_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass
    return rec


def recent_heard(n: int = 20, after: float = 0.0):
    with _heard_lock:
        recs = list(_heard)
    if after:
        recs = [r for r in recs if float(r.get("t") or 0) > after]
    return recs[-int(n):]


def apply_nav_hint(hint: dict):
    """Dispatch a goal hint through the registered people_live callback."""
    if not isinstance(hint, dict) or not hint:
        return False, "empty hint"
    h = dict(hint)
    if not h.get("type") and "x" in h and "y" in h:
        h["type"] = "xy"
    if _nav_cb is None:
        return False, "no nav callback"
    _nav_cb(h)
    return True, h.get("type") or ""


def speak(text: str, voice: str = "am_michael", block: bool = False) -> None:
    """Queue a line for Kokoro → ReSpeaker. Outside sources should use this."""
    text = (text or "").strip()
    if not text:
        return
    with _lock:
        if _say_q and _say_q[-1][0] == text:
            pass
        else:
            _say_q.append((text, voice))
    _kick_worker()
    if block:
        t0 = __import__("time").monotonic()
        while __import__("time").monotonic() - t0 < 60.0:
            with _lock:
                if not _speak_busy and not _say_q:
                    return
            __import__("time").sleep(0.05)


def _kick_worker() -> None:
    global _worker
    with _lock:
        if _worker is not None and _worker.is_alive():
            return
        _worker = threading.Thread(target=_drain_say, daemon=True, name="kevin-tts")
        _worker.start()


def _drain_say() -> None:
    global _speak_busy
    while True:
        with _lock:
            if not _say_q:
                return
            text, voice = _say_q.popleft()
            _speak_busy = True
        try:
            _play(text, voice)
        finally:
            with _lock:
                _speak_busy = False


def _play(text: str, voice: str) -> None:
    sink = current_sink()
    try:
        apply_speak_sink(persist=False, quiet=True)
        sink = current_sink()
        k = _ensure_kokoro()
        samples, sr = k.create(text, voice=voice, speed=1.0)
        import soundfile as sf
        import numpy as np
        vol = max(0.0, min(1.0, float(os.environ.get("KEVIN_SPEAK_VOL", str(SPEAK_VOLUME)))))
        samples = np.asarray(samples, dtype=np.float32)
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        if peak > 1.0:
            samples = samples / peak
        path = os.path.join(tempfile.gettempdir(), "kevin_tts.wav")
        sf.write(path, samples, sr)
        try:
            import robot_params as _RP
            _on = _RP.V.get("audio_enabled") is True
        except Exception:
            _on = False
        _on = _on or os.environ.get("KEVIN_AUDIO_ENABLED") == "1"
        if os.environ.get("KEVIN_MUTE") == "1" or not _on:          # speaker muted until James enables audio.enabled
            print("speech: MUTED (audio.enabled false / KEVIN_MUTE=1): not played, wav at %s" % path)
            return
        paplay_vol = max(1, int(round(65536 * vol)))
        r = subprocess.run(
            ["paplay", f"--device={sink}", f"--volume={paplay_vol}", path],
            check=False, timeout=60, capture_output=True, text=True,
        )
        if r.returncode != 0:
            print("speech: paplay rc=%s err=%r" % (r.returncode, (r.stderr or "")[:120]))
        print("speech: said %r (vol=%.0f%% sink=%s)" % (
            text[:80], vol * 100, sink.split(".")[-1]))
    except Exception as e:
        print("speech: speak failed: %s" % e)


def listen_seconds(secs: float = 3.0) -> str:
    path = os.path.join(tempfile.gettempdir(), "kevin_listen.wav")
    try:
        apply_listen_source(quiet=True)
    except Exception:
        pass
    src = current_src()
    try:
        subprocess.run(
            ["parecord", f"--device={src}", "--file-format=wav",
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
        if text:
            note_heard(text, extra={"src": src.split(".")[-1]})
        return text
    except Exception as e:
        print("speech: listen failed: %s" % e)
        return ""


def is_speaking() -> bool:
    with _lock:
        return _speak_busy or bool(_say_q)


def _rgb_to_jpeg(img, quality: int = 70, max_w: int = 640) -> bytes:
    import cv2
    import numpy as np
    img = np.asarray(img)
    if img.ndim != 3 or img.shape[2] < 3 or img.size == 0:
        return b""
    bgr = img[:, :, :3][:, :, ::-1].copy()
    h, w = bgr.shape[:2]
    if w > max_w:
        nh = int(round(h * (max_w / float(w))))
        bgr = cv2.resize(bgr, (max_w, max(1, nh)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return bytes(buf) if ok else b""


def _vision_rgb(cam: str):
    """Return one RGB ndarray: webcam, rs1, rs2, atlas, or stacked all."""
    import numpy as np
    import tools
    vis = tools.get_vision()
    if vis is None:
        return None
    frames = getattr(vis, "frames", None)
    atlas = getattr(vis, "atlas", None)
    cam = (cam or "webcam").strip().lower()

    def _live(which):
        if which == "webcam":
            w = getattr(vis, "_webcam", None)
            return getattr(w, "color", None) if w is not None else None
        if which == "rs1":
            r = getattr(vis, "_rs1", None)
            c = getattr(r, "color", None) if r is not None else None
            return None if c is None else c[::-1, ::-1]
        if which == "rs2":
            r = getattr(vis, "_rs2", None)
            return getattr(r, "color", None) if r is not None else None
        return None

    if cam in ("webcam", "rgb", "rgb1", "0"):
        live = _live("webcam")
        return live if live is not None else (None if frames is None or len(frames) < 1 else frames[0])
    if cam in ("rs1", "topdown", "1"):
        live = _live("rs1")
        return live if live is not None else (None if frames is None or len(frames) < 2 else frames[1])
    if cam in ("rs2", "forward", "2"):
        live = _live("rs2")
        return live if live is not None else (None if frames is None or len(frames) < 3 else frames[2])
    if cam == "atlas":
        return atlas
    if cam == "all":
        parts = []
        for fr in (frames or [])[:3]:
            if fr is not None and getattr(fr, "size", 0):
                parts.append(np.asarray(fr)[:, :, :3])
        if not parts:
            return None
        h = max(p.shape[0] for p in parts)
        row = []
        for p in parts:
            if p.shape[0] != h:
                import cv2
                p = cv2.resize(p, (int(p.shape[1] * h / p.shape[0]), h))
            row.append(p)
        return np.concatenate(row, axis=1)
    return None


def _look_jpeg(quality: int = 70, max_w: int = 480, cam: str = "webcam") -> bytes:
    """Current camera JPEG, or empty bytes if no frame."""
    try:
        fr = _vision_rgb(cam)
        if fr is None or getattr(fr, "size", 0) == 0:
            return b""
        return _rgb_to_jpeg(fr, quality=quality, max_w=max_w if cam != "all" else 960)
    except Exception as e:
        print("speech: look jpeg skip: %s" % e)
        return b""


def _status_obj():
    import tools
    vis = tools.get_vision()
    wb = tools.get_wheelbase()
    out = {"ok": True, "speaking": is_speaking()}
    if vis is not None:
        pose = getattr(vis, "_pose", None)
        if pose is not None:
            out["pose"] = {
                "x": float(pose.x), "y": float(pose.y),
                "yaw_deg": float(getattr(pose, "theta", 0.0) * 57.2957795),
            }
        out["topdown_ok"] = bool(getattr(vis, "topdown_depth_ok", False))
        out["safety"] = {
            "fwd": float(getattr(vis, "safety_fwd_scale", 1.0)),
            "bwd": float(getattr(vis, "safety_bwd_scale", 1.0)),
            "ang": float(getattr(vis, "safety_ang_scale", 1.0)),
        }
        bag = getattr(vis, "_raw_bag", None)
        if bag is not None:
            out["bag"] = {
                "n": int(getattr(bag, "n", 0)),
                "drops": int(getattr(bag, "drops", 0)),
                "dest": str(getattr(bag, "dest", "")),
            }
    if wb is not None:
        out["twist_for_active"] = bool(wb.is_twist_for_active())
        snap = None
        try:
            snap = wb.get_encoder_snapshot()
        except Exception:
            snap = None
        if isinstance(snap, dict):
            out["wheels"] = {
                "vl": float(snap.get("vl", 0.0)),
                "vr": float(snap.get("vr", 0.0)),
                "ok": float(snap.get("ok", 0.0)),
                "age_s": float(snap.get("age_s", 99.0)),
            }
        elif snap is not None and hasattr(snap, "__len__"):
            out["wheels"] = list(snap)[:8]
    return out


def _apply_twist(obj) -> tuple:
    """Queue a timed twist. Returns (ok, msg)."""
    import tools
    vis = tools.get_vision()
    wb = tools.get_wheelbase()
    if wb is None:
        return False, "no wheelbase"
    force = bool(obj.get("force"))
    if vis is not None and not force and not getattr(vis, "topdown_depth_ok", True):
        return False, "topdown_lost"
    try:
        fwd = float(obj.get("forward_mps", 0.0))
        ang = float(obj.get("angular_rads", 0.0))
        dur = float(obj.get("duration_secs", 5.0))
    except (TypeError, ValueError):
        return False, "bad numbers"
    fwd = max(-0.25, min(0.25, fwd))
    ang = max(-0.8, min(0.8, ang))
    dur = max(0.2, min(12.0, dur))
    ramp = min(1.0, max(0.15, dur * 0.15))
    ui = tools.get_ui()
    if ui is not None:
        ui.push_tool_calls([{
            "name": "twist_for",
            "args": {
                "forward_mps": fwd,
                "angular_rads": ang,
                "duration_secs": dur,
                "ramp_in_secs": float(obj.get("ramp_in_secs", ramp)),
                "ramp_out_secs": float(obj.get("ramp_out_secs", ramp)),
            },
        }])
    else:
        tools.twist_for(fwd, ang, duration_secs=dur, ramp_in_secs=ramp, ramp_out_secs=ramp)
    rec = {
        "t_wall": time.time(),
        "cmd": "twist_for",
        "forward_mps": fwd,
        "angular_rads": ang,
        "duration_secs": dur,
        "ramp_in_secs": float(obj.get("ramp_in_secs", ramp)),
        "ramp_out_secs": float(obj.get("ramp_out_secs", ramp)),
        "pose": None,
        "safety": None,
        "topdown_ok": None,
    }
    if vis is not None:
        pose = getattr(vis, "_pose", None)
        if pose is not None:
            rec["pose"] = {
                "x": float(pose.x), "y": float(pose.y),
                "yaw_deg": float(getattr(pose, "theta", 0.0) * 57.2957795),
            }
        rec["topdown_ok"] = bool(getattr(vis, "topdown_depth_ok", False))
        rec["safety"] = {
            "fwd": float(getattr(vis, "safety_fwd_scale", 1.0)),
            "bwd": float(getattr(vis, "safety_bwd_scale", 1.0)),
            "ang": float(getattr(vis, "safety_ang_scale", 1.0)),
        }
    _append_cmd_log(rec)
    print("speech: /twist fwd=%.3f ang=%.3f dur=%.1f" % (fwd, ang, dur))
    return True, "twist_for(%.3f, %.3f, %.1fs)" % (fwd, ang, dur)


def _append_cmd_log(rec: dict) -> None:
    path = os.environ.get("KEVIN_CMD_LOG", "").strip()
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
    except Exception as e:
        print("speech: cmd log skip: %s" % e)


def _apply_stop() -> tuple:
    import tools
    ui = tools.get_ui()
    if ui is not None:
        ui.push_tool_calls([{"name": "stop", "args": {}}])
    else:
        tools.stop()
    print("speech: /stop")
    return True, "stop"


class _SayHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print("say-http: " + (fmt % args))

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json")

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        u = urlparse(self.path)
        path = u.path.rstrip("/") or "/"
        if path in ("/health", "/"):
            self._send_json(200, {
                "ok": True,
                "speaking": is_speaking(),
                "sink": current_sink(),
                "vol": float(os.environ.get("KEVIN_SPEAK_VOL", str(SPEAK_VOLUME))),
            })
            return
        if path in ("/look.jpg", "/look"):
            qs = parse_qs(u.query or "")
            cam = (qs.get("cam") or ["webcam"])[0] or "webcam"
            jpeg = _look_jpeg(cam=cam)
            if not jpeg:
                self._send_json(503, {"ok": False, "error": "no %s frame" % cam})
                return
            self._send(200, jpeg, "image/jpeg")
            return
        if path.startswith("/look/") and path.endswith(".jpg"):
            cam = path[len("/look/"):-len(".jpg")] or "webcam"
            jpeg = _look_jpeg(cam=cam, max_w=640 if cam != "all" else 960)
            if not jpeg:
                self._send_json(503, {"ok": False, "error": "no %s frame" % cam})
                return
            self._send(200, jpeg, "image/jpeg")
            return
        if path == "/status":
            self._send_json(200, _status_obj())
            return
        if path == "/say":
            qs = parse_qs(u.query or "")
            text = (qs.get("text") or [""])[0]
            self._handle_say(text)
            return
        if path == "/heard":
            qs = parse_qs(u.query or "")
            try:
                after = float((qs.get("after") or ["0"])[0] or 0)
            except ValueError:
                after = 0.0
            try:
                n = int((qs.get("n") or ["20"])[0] or 20)
            except ValueError:
                n = 20
            recs = recent_heard(n=n, after=after)
            self._send_json(200, {"ok": True, "heard": recs, "n": len(recs)})
            return
        self._send_json(404, {"ok": False, "error": "not found"})

    def do_POST(self):
        u = urlparse(self.path)
        path = u.path.rstrip("/") or "/"
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n > 0 else b""
        obj = {}
        ctype = (self.headers.get("Content-Type") or "").lower()
        if "json" in ctype:
            try:
                obj = json.loads(raw.decode("utf-8") or "{}")
            except Exception:
                obj = {}
        if path == "/goal":
            if not obj and raw:
                try:
                    obj = json.loads(raw.decode("utf-8") or "{}")
                except Exception:
                    obj = {}
            ok, msg = apply_nav_hint(obj if isinstance(obj, dict) else {})
            print("speech: /goal %s %s" % ("ok" if ok else "fail", msg))
            self._send_json(200 if ok else 503, {"ok": ok, "applied": msg, "hint": obj})
            return
        if path == "/twist":
            if not obj and raw:
                try:
                    obj = json.loads(raw.decode("utf-8") or "{}")
                except Exception:
                    obj = {}
            ok, msg = _apply_twist(obj if isinstance(obj, dict) else {})
            st = _status_obj()
            st.update({"ok": ok, "applied": msg})
            self._send_json(200 if ok else 503, st)
            return
        if path == "/stop":
            ok, msg = _apply_stop()
            self._send_json(200, {"ok": ok, "applied": msg})
            return
        if path != "/say":
            self._send_json(404, {"ok": False, "error": "not found"})
            return
        text = ""
        if isinstance(obj, dict) and obj:
            text = str(obj.get("text") or obj.get("say") or "")
        else:
            text = raw.decode("utf-8", "replace")
        self._handle_say(text)

    def _handle_say(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            self._send_json(400, {"ok": False, "error": "empty text"})
            return
        if len(text) > 400:
            text = text[:400]
        speak(text)
        print("speech: /say queued %r" % (text[:80],))
        self._send_json(200, {"ok": True, "queued": text[:80]})


def start_say_server(port: int = None) -> None:
    """LAN TTS + look JPEG. Safe to call once from main after tools.init."""
    global _http_srv
    if _http_srv is not None:
        return
    port = int(port or SAY_PORT)
    srv = ThreadingHTTPServer(("0.0.0.0", port), _SayHandler)
    srv.daemon_threads = True
    _http_srv = srv

    def _run():
        print("speech: say server http://0.0.0.0:%d/say  /look.jpg  /heard  /goal" % port)
        srv.serve_forever()

    threading.Thread(target=_run, daemon=True, name="kevin-say-http").start()


def main_cli(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Kevin TTS / ReSpeaker sink")
    ap.add_argument("--setup-sink", action="store_true",
                    help="Select ReSpeaker as default Pulse sink and persist")
    ap.add_argument("--say", default="", help="Speak one line and exit")
    ap.add_argument("--serve", action="store_true", help="Run /say HTTP server")
    args = ap.parse_args(argv)
    if args.setup_sink or args.say or args.serve:
        apply_speak_sink(persist=True)
    if args.say:
        speak(args.say, block=True)
    if args.serve:
        start_say_server()
        threading.Event().wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())

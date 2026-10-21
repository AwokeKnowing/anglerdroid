"""Kevin's voice service, its own process (not the safety core, not the nav loop).

  TTS     POST /say {"text": "...", "lang": "en"|"es", "voice"?, "speed"?, "wait"?: false} -> Kokoro -> paplay
          (ReSpeaker sink if present, else KEVIN_SPEAK_SINK, else the default sink), volume KEVIN_SPEAK_VOL (0.75).
          nav/server.py forwards POST /api/say on :8080 here, so the public endpoint is the nav server.
  Ears    ReSpeaker capture (parec 16 kHz mono) -> energy VAD -> faster-whisper (multilingual, language detected).
          An utterance that addresses "Kevin" -> local chime at once, then POST to the help webhook
          {"kind": "voice", "robot", "transcript", "lang", "lang_prob", "time", "reply": {how to answer}}.
          No speaker id yet, so "speaker" is omitted.  The mic is ignored while Kevin is talking (no self-trigger).
  Test    python -m nav.voice --test "Kevin, ¿qué hora es?" --lang es [--no-post]   (Kokoro -> wav -> same pipeline)
          python -m nav.voice --wav file.wav [--no-post]

Env: KEVIN_VOICE_PORT (8092), KEVIN_SPEAK_VOL (0.75), KEVIN_SPEAK_SINK, KEVIN_LISTEN_SRC, KEVIN_WHISPER ("base"),
     KEVIN_VOICE_THREADS (4), KEVIN_VOICE_EN ("am_michael"), KEVIN_VOICE_ES ("em_alex"), KEVIN_VOICE_POST (1).
"""
from __future__ import annotations
import argparse, collections, json, os, re, subprocess, sys, tempfile, threading, time, unicodedata
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np

E = os.environ
PORT = int(E.get("KEVIN_VOICE_PORT", "8092"))
VOL = max(0.0, min(1.0, float(E.get("KEVIN_SPEAK_VOL", "0.75"))))
THREADS = int(E.get("KEVIN_VOICE_THREADS", "4"))     # nice 10 on cores 0-5; 2 threads: ASR 3.1 s, 4: 2.0 s
VOICES = {"en": E.get("KEVIN_VOICE_EN", "am_michael"), "es": E.get("KEVIN_VOICE_ES", "em_alex")}
KOKORO_LANG = {"en": "en-us", "es": "es"}
MODEL_DIR = os.path.expanduser("~/.kevin/models/kokoro")
HINTS = ("SEEED_ReSpeaker", "ReSpeaker", "ArrayUAC10")
WAKE = re.compile(r"\b(kevin|kevins|keven|kevan|kevim|kelvin|kebin|quevin|que vin)\b")   # + how whisper writes a Spanish "Kevin"
SR = 16000
REPLY_CMD = ("echo '{\"text\": \"<reply>\", \"lang\": \"%s\"}' | ssh james@i777 \"ssh jetbot@jetbot.local "
             "'curl -s -X POST -H Content-Type:application/json --data-binary @- http://127.0.0.1:8080/api/say'\"")
LOG = os.path.expanduser("~/kevin_bench/voice.log")

_lock = threading.Lock(); _q = collections.deque(maxlen=8); _busy_until = [0.0]; _kok = [None]; _wh = [None]
_stat = dict(said=0, heard=0, addressed=0, posts=[], mic=None, sink=None, last=None)


def log(s):
    line = "%s [voice] %s" % (time.strftime("%H:%M:%S"), s); print(line, flush=True)
    try:
        with open(LOG, "a") as f: f.write(line + "\n")
    except Exception: pass


def _pactl(kind):
    try: return subprocess.run(["pactl", "list", "short", kind], capture_output=True, text=True, timeout=3).stdout
    except Exception: return ""


def sink():
    if E.get("KEVIN_SPEAK_SINK"): return E["KEVIN_SPEAK_SINK"]
    for l in _pactl("sinks").splitlines():
        p = l.split()
        if len(p) > 1 and any(h.lower() in p[1].lower() for h in HINTS): return p[1]
    return None                                                     # paplay default sink


def mic():
    if E.get("KEVIN_LISTEN_SRC"): return E["KEVIN_LISTEN_SRC"]
    for l in _pactl("sources").splitlines():
        p = l.split()
        if len(p) > 1 and ".monitor" not in p[1] and any(h.lower() in p[1].lower() for h in HINTS): return p[1]
    return None


def kokoro():
    if _kok[0] is None:
        from kokoro_onnx import Kokoro
        m, v = os.path.join(MODEL_DIR, "kokoro-v1.0.onnx"), os.path.join(MODEL_DIR, "voices-v1.0.bin")
        try:
            import onnxruntime as ort
            so = ort.SessionOptions(); so.intra_op_num_threads = THREADS; so.inter_op_num_threads = 1
            _kok[0] = Kokoro.from_session(ort.InferenceSession(m, so, providers=["CPUExecutionProvider"]), v)
        except AttributeError:
            _kok[0] = Kokoro(m, v)
    return _kok[0]


def whisper():
    if _wh[0] is None:
        from faster_whisper import WhisperModel
        _wh[0] = WhisperModel(E.get("KEVIN_WHISPER", "base"), device="cpu", compute_type="int8", cpu_threads=THREADS)
    return _wh[0]


def synth(text, lang="en", voice=None, speed=1.0):
    lang = lang if lang in KOKORO_LANG else "en"
    s, sr = kokoro().create(text, voice=voice or VOICES[lang], speed=float(speed), lang=KOKORO_LANG[lang])
    s = np.asarray(s, np.float32); pk = float(np.abs(s).max()) if s.size else 0.0
    return (s / pk if pk > 1.0 else s), sr


def _wav(path, s, sr):
    import soundfile as sf
    sf.write(path, s, sr)


def play(path, secs):
    sk = sink(); _stat["sink"] = sk or "default"
    _busy_until[0] = time.time() + secs + 0.6                       # mic deaf while talking (+ room tail)
    cmd = ["paplay", "--volume=%d" % max(1, int(65536 * VOL))] + (["--device=" + sk] if sk else []) + [path]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=max(10, secs + 10))
    _busy_until[0] = time.time() + 0.6
    if r.returncode: log("paplay rc=%s %s" % (r.returncode, r.stderr.strip()[:120]))
    return r.returncode == 0


def chime():
    """Local acknowledgement: two short rising tones (~0.25 s), played in the background."""
    p = os.path.join(tempfile.gettempdir(), "kevin_chime.wav")
    if not os.path.exists(p):
        t = np.arange(int(0.11 * 24000)) / 24000.0; env = np.sin(np.pi * t / t[-1]) ** 2
        tone = lambda f: 0.5 * env * np.sin(2 * np.pi * f * t)
        _wav(p, np.concatenate([tone(880), np.zeros(600), tone(1320)]).astype(np.float32), 24000)
    threading.Thread(target=play, args=(p, 0.25), daemon=True).start()


def say(text, lang="en", voice=None, speed=1.0, wait=False):
    """Queue one line (wait=True: return after it was spoken, with timings)."""
    job = dict(text=text, lang=lang, voice=voice, speed=speed, t_req=time.time(), done=threading.Event(), res={})
    with _lock: _q.append(job)
    threading.Thread(target=_drain, daemon=True).start()
    if wait: job["done"].wait(120)
    return job["res"] if wait else dict(queued=True)


_drain_lock = threading.Lock()


def _drain():
    with _drain_lock:
        while True:
            with _lock:
                if not _q: return
                j = _q.popleft()
            try:
                t0 = time.time(); s, sr = synth(j["text"], j["lang"], j["voice"], j["speed"]); t1 = time.time()
                path = os.path.join(tempfile.gettempdir(), "kevin_say.wav"); _wav(path, s, sr)
                dur = len(s) / float(sr); ok = play(path, dur)
                j["res"].update(ok=ok, lang=j["lang"], voice=j["voice"] or VOICES.get(j["lang"], VOICES["en"]), synth_s=round(t1 - t0, 3),
                                audio_s=round(dur, 2), first_audio_s=round(t1 - j["t_req"], 3), total_s=round(time.time() - j["t_req"], 3), sink=_stat["sink"])
                _stat["said"] += 1; log("said [%s] %r %s" % (j["lang"], j["text"][:80], j["res"]))
            except Exception as e:
                j["res"].update(ok=False, error=repr(e)); log("say failed: %r" % e)
            j["done"].set()


def _norm(t):
    return "".join(c for c in unicodedata.normalize("NFD", t.lower()) if unicodedata.category(c) != "Mn")


def addressed(text):
    return bool(text) and bool(WAKE.search(_norm(text)))


def transcribe(audio):
    t0 = time.time()
    segs, info = whisper().transcribe(audio, language=None, beam_size=1, vad_filter=False, condition_on_previous_text=False, hotwords="Kevin")
    text = " ".join(s.text.strip() for s in segs).strip()
    return text, info.language, float(info.language_probability), time.time() - t0


def voice_payload(text, lang, prob):
    rl = lang if lang in KOKORO_LANG else "en"
    p = dict(kind="voice", robot="kevin001", transcript=text, lang=lang, lang_prob=round(prob, 3), time=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
             reply=dict(how="POST JSON {\"text\": ..., \"lang\": \"en\"|\"es\"} to Kevin's TTS endpoint http://127.0.0.1:8080/api/say on the Orin "
                            "(jetbot.local): ssh to i777, ssh from there to the Orin, curl it. The JSON goes in on stdin, so the text needs no shell quoting.",
                        lang=rl, command=REPLY_CMD % rl))
    if E.get("KEVIN_VOICE_TEST") == "1": p["test"] = True
    return p


def handle_utterance(audio, post=True, t_end=None):
    """audio: float32 16 kHz mono.  Returns a record with latencies measured from the end of speech."""
    t_end = t_end or time.time()
    text, lang, prob, asr_s = transcribe(audio); _stat["heard"] += 1
    rec = dict(transcript=text, lang=lang, lang_prob=round(prob, 3), asr_s=round(asr_s, 3), audio_s=round(len(audio) / SR, 2), addressed=False)
    if addressed(text):
        rec["addressed"] = True; _stat["addressed"] += 1
        chime(); rec["chime_s"] = round(time.time() - t_end, 3)
        if post and E.get("KEVIN_VOICE_POST", "1") == "1":
            from nav import help_webhook as HW
            s, note = HW.post(voice_payload(text, lang, prob)); rec["post_status"] = s; rec["post_s"] = round(time.time() - t_end, 3)
            _stat["posts"].append(dict(t=time.time(), status=s))
    _stat["last"] = rec; log("heard %s" % rec)
    return rec


def listen_loop(stop):
    """ReSpeaker -> energy VAD -> handle_utterance.  Re-checks every 10 s while no mic is present."""
    while not stop.is_set():
        src = mic(); _stat["mic"] = src
        if not src:
            time.sleep(10); continue
        log("listening on %s" % src)
        p = subprocess.Popen(["parec", "--device=" + src, "--rate=16000", "--channels=1", "--format=s16le", "--raw", "--latency-msec=30"],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        fr = int(0.03 * SR); floor = 300.0; cur = []; silent = 0; voiced = 0
        try:
            while not stop.is_set():
                b = p.stdout.read(fr * 2)
                if not b or len(b) < fr * 2: break
                x = np.frombuffer(b, np.int16).astype(np.float32); rms = float(np.sqrt((x * x).mean()))
                if time.time() < _busy_until[0]: cur = []; voiced = 0; continue
                loud = rms > max(500.0, 3.0 * floor)
                if not cur and not loud: floor = 0.95 * floor + 0.05 * rms; continue
                cur.append(x / 32768.0); voiced += loud; silent = 0 if loud else silent + 1
                if silent * 0.03 > 0.7 or len(cur) * 0.03 > 12:
                    if voiced * 0.03 > 0.3:
                        threading.Thread(target=handle_utterance, args=(np.concatenate(cur),), kwargs=dict(t_end=time.time()), daemon=True).start()
                    cur = []; voiced = 0; silent = 0
        finally:
            p.kill()
        time.sleep(1)


def serve(port):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def _send(self, d, code=200):
            b = json.dumps(d, ensure_ascii=False).encode(); self.send_response(code); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
        def do_GET(self):
            if self.path.split("?")[0] in ("/", "/status"): return self._send(dict(_stat, port=port, vol=VOL, voices=VOICES))
            self._send({}, 404)
        def do_POST(self):
            p = self.path.split("?")[0]
            try: d = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            except Exception: return self._send(dict(ok=False, error="bad json"), 400)
            if p == "/chime": chime(); return self._send(dict(ok=True))
            if p != "/say": return self._send({}, 404)
            text = str(d.get("text") or "").strip()[:1000]
            if not text: return self._send(dict(ok=False, error="empty text"), 400)
            res = say(text, str(d.get("lang") or "en").lower()[:2], d.get("voice"), d.get("speed", 1.0), bool(d.get("wait")))
            self._send(dict(res, ok=res.get("ok", True)))
    srv = ThreadingHTTPServer(("0.0.0.0", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True, name="voice-http").start()
    log("TTS on http://0.0.0.0:%d/say (vol %.2f, voices %s, sink %s, mic %s)" % (port, VOL, VOICES, sink() or "default", mic())); return srv


def _to16k(s, sr):
    n = int(len(s) * SR / sr); return np.interp(np.linspace(0, len(s) - 1, n), np.arange(len(s)), s).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--test"); ap.add_argument("--lang", default="en"); ap.add_argument("--wav"); ap.add_argument("--no-post", action="store_true")
    a = ap.parse_args()
    if a.test or a.wav:
        if a.test:
            t0 = time.time(); s, sr = synth(a.test, a.lang); print("synth %.2f s for %.2f s of audio" % (time.time() - t0, len(s) / sr))
            audio = _to16k(s, sr)
        else:
            import soundfile as sf
            s, sr = sf.read(a.wav, dtype="float32"); audio = _to16k(s if s.ndim == 1 else s.mean(1), sr)
        whisper(); E["KEVIN_VOICE_TEST"] = "1"
        print(json.dumps(handle_utterance(audio, post=not a.no_post), ensure_ascii=False)); time.sleep(0.6); return
    kokoro(); whisper()                                              # load the models before advertising the endpoint
    stop = threading.Event(); serve(a.port)
    threading.Thread(target=listen_loop, args=(stop,), daemon=True, name="ears").start()
    try:
        while True: time.sleep(3600)
    except KeyboardInterrupt: stop.set()


if __name__ == "__main__":
    main()

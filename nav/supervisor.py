"""Kevin stack supervisor: OUTSIDE the safety core and outside nav.  It starts the stack as a child process, keeps it
healthy, and asks for help when it cannot.  It never commands motion: while the stack is down nothing drives the
wheels (and with the stack up the safety core immobilises on missing/invalid depth by itself).

  python -m nav.supervisor -- <stack command ...>          (deploy/orin/nomotor_start.sh runs it; KEVIN_SUPERVISE=0 skips it)

Before EVERY start (first one included) it hardware-resets the RealSenses and waits until both serials are enumerated
and stable (src/rs_reset.py) -- the root cause of the 2026-10-21 RS1 first-start flake was a USB re-enumeration during
startup.  Health (every POLL_S, nav /api/state on --port):
  start   'UI on' in the child's output within START_S, child alive
  run     HTTP answers; no core_alarm; safety core alive; core heartbeat advancing; RS1/RS2 frame age < FRAME_STALE_S
Unhealthy for UNHEALTHY_S (or the child exits) -> stop the child (SIGTERM to its process group, SIGKILL after 10 s),
reset + re-enumerate the cameras, restart; backoff BACKOFF (5, 15, 45, 120 s), up to TRIES consecutive failures (a
stack healthy for HEALTHY_RESET_S resets the count).  After TRIES failures: POST {"kind": "help", ...} (nav.help_webhook,
one per component per 10 min) and keep retrying every GIVEUP_RETRY_S.  Long no_depth (sun / lens covered: the core is
already immobilising, nothing to restart) -> one help POST, component "rs1_depth".
Status: ~/kevin_bench/supervisor.log and /tmp/kevin_supervisor.json.
Env: KEVIN_SUP_TRIES (4), KEVIN_SUP_BACKOFF ("5,15,45,120"), KEVIN_SUP_UNHEALTHY_S (8), KEVIN_SUP_START_S (180),
     KEVIN_SUP_NODEPTH_HELP_S (600), KEVIN_SUP_NO_RESET=1 (skip the camera reset, e.g. sim)."""
from __future__ import annotations
import argparse, collections, json, os, signal, subprocess, sys, threading, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src")]
from nav import help_webhook as HW   # noqa: E402

E = os.environ
TRIES = int(E.get("KEVIN_SUP_TRIES", "4")); BACKOFF = [float(x) for x in E.get("KEVIN_SUP_BACKOFF", "5,15,45,120").split(",")]
UNHEALTHY_S = float(E.get("KEVIN_SUP_UNHEALTHY_S", "8")); START_S = float(E.get("KEVIN_SUP_START_S", "180"))
NODEPTH_HELP_S = float(E.get("KEVIN_SUP_NODEPTH_HELP_S", "600")); POLL_S = 1.0; FRAME_STALE_S = 1.0
HEALTHY_RESET_S = 120.0; GIVEUP_RETRY_S = 600.0
LOG = os.path.expanduser("~/kevin_bench/supervisor.log"); STATUS = "/tmp/kevin_supervisor.json"


def log(s):
    line = "%s [supervisor] %s" % (time.strftime("%H:%M:%S"), s); print(line, flush=True)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a") as f: f.write(line + "\n")


class Stack:
    def __init__(self, cmd, logfile):
        self.cmd, self.logfile = cmd, logfile; self.p = None; self.tail = collections.deque(maxlen=60); self.ui = False

    def start(self):
        self.tail.clear(); self.ui = False
        self.f = open(self.logfile, "a")
        self.p = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                  start_new_session=True, cwd=ROOT, text=True, bufsize=1)
        threading.Thread(target=self._pump, daemon=True).start(); log("started stack pid %d: %s" % (self.p.pid, " ".join(self.cmd)))

    def _pump(self):
        for ln in self.p.stdout:
            self.f.write(ln); self.f.flush(); self.tail.append(ln.rstrip())
            if "UI on" in ln: self.ui = True

    def alive(self): return self.p is not None and self.p.poll() is None

    def stop(self):
        if not self.alive(): return
        try: os.killpg(self.p.pid, signal.SIGTERM)
        except ProcessLookupError: return
        try: self.p.wait(10)
        except subprocess.TimeoutExpired:
            log("stack did not exit on SIGTERM: SIGKILL"); os.killpg(self.p.pid, signal.SIGKILL); self.p.wait(5)


def get_state(port):
    with urllib.request.urlopen("http://127.0.0.1:%d/api/state" % port, timeout=2.0) as r: return json.loads(r.read())


def classify(text):
    """component + error from a log tail / health reason."""
    t = text.lower()
    if "rs1" in t or "815412070676" in t: return "rs1"
    if "rs2" in t or "944622074292" in t or "imu" in t: return "rs2"
    if "safety core" in t or "core" in t: return "safety_core"
    return "nav"


def health(st, last_tdec):
    """-> (ok, reason, t_dec, code)"""
    core = (st.get("safety") or {}).get("core") or {}
    if st.get("core_alarm"): return False, "core alarm: %s" % st["core_alarm"], core.get("t_dec"), core.get("code")
    if core and not core.get("alive", True): return False, "safety core process dead", core.get("t_dec"), core.get("code")
    t = core.get("t_dec")
    if core and last_tdec is not None and t == last_tdec: return False, "safety core heartbeat stuck (t_dec not advancing)", t, core.get("code")
    for k, cam in (("age1", "RS1"), ("age2", "RS2")):
        a = core.get(k)
        if a is not None and a > FRAME_STALE_S: return False, "%s frames stopped (age %.1f s)" % (cam, a), t, core.get("code")
    return True, "ok", t, core.get("code")


def reset_cams(serials):
    if E.get("KEVIN_SUP_NO_RESET") == "1" or E.get("KEVIN_HAL") == "sim": return dict(ok=True, skipped=True)
    try:
        import rs_reset
        return rs_reset.reset_and_wait(serials, log=log)
    except Exception as e:
        log("camera reset failed: %r" % e); return dict(ok=False, error=repr(e))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8080); ap.add_argument("--log", default=os.path.expanduser("~/kevin_bench/nav_latest.log"))
    ap.add_argument("cmd", nargs=argparse.REMAINDER); a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd: sys.exit("supervisor: no stack command")
    try:
        import robot_params as RP; serials = [s for s in (RP.V.get("cam_rs1_serial"), RP.V.get("cam_rs2_serial")) if s]
    except Exception: serials = None
    stack = Stack(cmd, a.log); fails = 0; nodepth_t = None; st = {}
    stopping = [False]
    def _term(*_):
        stopping[0] = True
    signal.signal(signal.SIGTERM, _term); signal.signal(signal.SIGINT, _term)
    status = dict(pid=os.getpid(), state="init", fails=0, restarts=0, last_reason=None, helps=[])
    def put(**k):
        status.update(k, t=time.time()); json.dump(status, open(STATUS + ".tmp", "w")); os.replace(STATUS + ".tmp", STATUS)
    while not stopping[0]:
        put(state="resetting cameras"); rr = reset_cams(serials)
        reason = None
        if not rr.get("ok"):
            reason = "camera(s) not enumerated on USB after reset: %s" % rr.get("missing", rr.get("error"))
        else:
            put(state="starting"); stack.start(); t0 = time.monotonic()
            while not stopping[0] and stack.alive() and not stack.ui and time.monotonic() - t0 < START_S: time.sleep(0.2)
            if stopping[0]: break
            if not stack.ui:
                reason = ("stack exited during startup: " if not stack.alive() else "no 'UI on' within %d s: " % START_S) + \
                         next((l for l in reversed(stack.tail) if "Error" in l or "error" in l), stack.tail[-1] if stack.tail else "")
            else:
                put(state="running"); log("stack up in %.1f s" % (time.monotonic() - t0)); t_up = time.monotonic()
                last_tdec = None; bad_since = None; http_fail = 0
                while not stopping[0]:
                    time.sleep(POLL_S)
                    if not stack.alive(): reason = "stack exited (code %s): %s" % (stack.p.returncode, stack.tail[-1] if stack.tail else ""); break
                    try: st = get_state(a.port); http_fail = 0; ok, why, last_tdec, code = health(st, last_tdec)
                    except Exception as e:
                        http_fail += 1; ok, why, code = http_fail < 3, "nav HTTP not answering (%s)" % e.__class__.__name__, None
                    if ok:
                        bad_since = None
                        if fails and time.monotonic() - t_up > HEALTHY_RESET_S: log("healthy %d s: failure count reset" % HEALTHY_RESET_S); fails = 0; put(fails=0)
                    else:
                        bad_since = bad_since or time.monotonic()
                        if time.monotonic() - bad_since >= UNHEALTHY_S: reason = why; break
                    nodepth_t = (nodepth_t or time.monotonic()) if code == "no_depth" else None
                    if nodepth_t and time.monotonic() - nodepth_t > NODEPTH_HELP_S:
                        s, note = HW.help("rs1_depth", "RS1 depth invalid for %d s (sun / lens covered / fogged): the safety core is immobilising; nothing to restart" % (time.monotonic() - nodepth_t),
                                          attempts=0, log_tail="\n".join(list(stack.tail)[-20:]), state=_brief(st, status))
                        if s: status["helps"].append(dict(t=time.time(), component="rs1_depth", status=s)); log("help POST rs1_depth -> %s" % s)
                        nodepth_t = time.monotonic()
        if stopping[0]: break
        fails += 1; status["restarts"] += 1; comp = classify(reason + " " + " ".join(list(stack.tail)[-8:]))
        log("FAILURE %d/%d [%s]: %s" % (fails, TRIES, comp, reason)); put(state="recovering", fails=fails, last_reason=reason, component=comp)
        stack.stop()
        if fails >= TRIES:
            s, note = HW.help(comp, reason, attempts=fails, log_tail="\n".join(stack.tail), state=_brief(st, status))
            log("recovery failed after %d attempts: help POST [%s] -> %s %s" % (fails, comp, s, note))
            status["helps"].append(dict(t=time.time(), component=comp, status=s, note=note)); put(state="gave up, retrying every %d s" % GIVEUP_RETRY_S)
            _sleep(GIVEUP_RETRY_S, stopping); fails = TRIES - 1        # one more try per period, then help again (rate-limited)
        else:
            b = BACKOFF[min(fails - 1, len(BACKOFF) - 1)]; log("backoff %.0f s" % b); _sleep(b, stopping)
    log("stopping (signal)"); stack.stop(); put(state="stopped")


def _sleep(s, stopping):
    t = time.monotonic() + s
    while time.monotonic() < t and not stopping[0]: time.sleep(0.2)


def _brief(st, status):
    core = (st.get("safety") or {}).get("core") or {}
    return dict(mode=st.get("mode"), core_alarm=st.get("core_alarm"), core_code=core.get("code"), age1=core.get("age1"), age2=core.get("age2"),
                core_alive=core.get("alive"), core_restarts=core.get("restarts"), supervisor=dict(fails=status.get("fails"), restarts=status.get("restarts")))


if __name__ == "__main__":
    main()

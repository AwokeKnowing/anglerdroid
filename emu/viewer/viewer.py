"""Kevin live sim viewer (lean, stdlib + cv2).  usage: viewer.py [port=8095] [nav_port=8090]
Live WebGL: / (default) - Three.js sim. Shared state over WebSocket /ws (~15 Hz, all clients identical, chat log shared),
  /state.json polling fallback. Orbit camera is purely client-side (no server camera). Chat -> nav.agent (grammar, else local LLM).
Legacy JPEG: /jpeg - server-side render with camera frustums
3-D frame: the robot lives in its SLAM world, so the 3-D view draws Kevin, the SLAM occupancy map, path, goal, places and
  LightNav waypoints in nav's map frame.  That frame is placed in the scene by a fixed ANCHOR (nav frame -> env frame,
  taken from GT when nav starts; identity on the real robot), so the sim ground truth (house mesh, true Kevin ghost,
  dog) can be overlaid as an optional reference and SLAM drift shows as the two drifting apart.
Map: nav SLAM map + Kevin footprint (safety-state color) + trajectory + goal + planned path + labeled places + dog.
Chat: text -> small parser -> nav POST /api/command (same interface as the nav page)."""
import sys, os, json, math, re, time, threading, urllib.request, struct, hashlib, numpy as np, cv2
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), _ROOT]
import ws as WSL                       # emu/viewer/ws.py: stdlib WebSocket (works through cloudflared)
from nav import agent as AGENT          # chat agent: grammar pass-through + local LLM -> validated commands
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from emu import shm
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8095; NAV = "http://localhost:%d" % (int(sys.argv[2]) if len(sys.argv) > 2 else 8090)
FP = [(0.33, 0.27), (0.33, -0.27), (-0.33, -0.27), (-0.33, 0.27)]    # display footprint only
COL = {"ok": (80, 200, 80), "limited": (0, 200, 255), "blocked": (40, 40, 255), "no_depth": (200, 0, 200)}
def nav(path, body=None):
    rq = urllib.request.Request(NAV + path, data=None if body is None else json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    r = urllib.request.urlopen(rq, timeout=2).read(); return r
def sim():
    try: return json.load(open("/dev/shm/kevin_sim.json"))
    except Exception: return None
NAVPOSE = [None]
def env2nav(S, x, y):
    """env (GT) -> nav map frame via the current GT pose and nav pose (like emu.eval's gt.to_map); falls back to the
    start pose.  Inherits the current SLAM error, which is the right thing for goals given to nav."""
    gx, gy, ga = S.get("kevin") or S.get("start") or (0.0, 0.0, 0.0)
    nx, ny, na = NAVPOSE[0] or (0.0, 0.0, 0.0)
    dx, dy = x - gx, y - gy; c, s = math.cos(-ga), math.sin(-ga); u, v = c * dx - s * dy, s * dx + c * dy   # robot frame
    c, s = math.cos(na), math.sin(na); return nx + c * u - s * v, ny + s * u + c * v
def nav2env(S, x, y):
    """inverse of env2nav: nav map frame -> env (GT) frame for drawing nav/VLN geometry in the 3-D view."""
    gx, gy, ga = S.get("kevin") or S.get("start") or (0.0, 0.0, 0.0)
    nx, ny, na = NAVPOSE[0] or (0.0, 0.0, 0.0)
    dx, dy = x - nx, y - ny; c, s = math.cos(-na), math.sin(-na); u, v = c * dx - s * dy, s * dx + c * dy
    c, s = math.cos(ga), math.sin(ga); return [round(gx + c * u - s * v, 3), round(gy + s * u + c * v, 3)]
MODE_FILE = os.path.expanduser("~/.kevin/emu/viewer_mode.json")
MODE = {"send_all_vln": False}
try: MODE.update({k: v for k, v in json.load(open(MODE_FILE)).items() if k in MODE})   # survives viewer restarts
except Exception: pass
ANCHOR_FILE = "/dev/shm/kevin_viewer_anchor.json"
ANCHOR = {"t0": None, "a": [0.0, 0.0, 0.0], "src": "identity"}
try: ANCHOR.update(json.load(open(ANCHOR_FILE)))
except Exception: pass
def update_anchor(S, n):
    """nav frame -> scene (env) frame, fixed per nav session (nav t0).  Sim: GT pose composed with the inverse nav pose
    at the first state seen for that session (nav's frame is its start pose, so this is the start alignment when the
    viewer is up while nav starts, as live.sh / restarts do).  Real robot: identity."""
    t0 = n.get("t0")
    if t0 is not None and ANCHOR["t0"] == t0: return
    if S and S.get("kevin") and n.get("pose"):
        gx, gy, ga = S["kevin"][:3]; nx, ny, na = n["pose"][:3]; aa = ga - na; c, s = math.cos(aa), math.sin(aa)
        a = [gx - (c * nx - s * ny), gy - (s * nx + c * ny), aa]; src = "sim GT at nav start"
    else: a = [0.0, 0.0, 0.0]; src = "identity"
    ANCHOR.update(t0=t0, a=[round(v, 5) for v in a], src=src)
    try: json.dump(ANCHOR, open(ANCHOR_FILE, "w"))
    except Exception: pass
TRAJ = []; LOG = []; LOGN = [0]; follow = {"on": False}; lock = threading.Lock(); MAPJPG = [b""]
def log(who, t):
    with lock: LOG.append([time.strftime("%H:%M:%S"), who, t]); del LOG[:-60]; LOGN[0] += 1

SEQ = {"q": [], "gen": 0}; HUB = {"st": {}, "seq": 0}; CLIENTS = set(); CL_LOCK = threading.Lock()

def nav_cmd(t):
    try: r = json.loads(nav("/api/command", {"text": t}) or b"{}")
    except Exception as e: r = {"kind": "error", "msg": str(e)}
    return r

MOTION = {"goto_place", "goto_xy", "relative", "turn", "face", "drive", "ask"}
def sequencer():   # runs multi-step agent plans one motion at a time; any new command (last wins) or stop clears it
    while True:
        time.sleep(0.25)
        with lock:
            if not SEQ["q"]: continue
            gen = SEQ["gen"]
        mode = (HUB["st"].get("mode") or "idle")
        if mode not in ("idle",): continue
        with lock:
            if SEQ["gen"] != gen or not SEQ["q"]: continue
            t = SEQ["q"].pop(0)
        nav_cmd(t); log("kevin", "→ " + t)

def preprocess(t):
    t = t.strip().lower().rstrip(".!?"); t = re.sub(r"^(please|kevin|hey kevin)[ ,]+", "", t); t = re.sub(r"\s+please$", "", t)
    if re.fullmatch(r"(come back|come here|go home|return|go back( home)?|home)", t): t = "go to living room"
    t = re.sub(r"^(go|drive|head|navigate)( over)? to (the )?", "go to ", t)
    return t

def handle(text, who="you"):
    """one chat line -> reply text (also logged).  Grammar commands run immediately (stop < 50 ms); otherwise the LLM agent."""
    t0 = time.monotonic(); t = preprocess(text)
    with lock: SEQ["q"] = []; SEQ["gen"] += 1                 # last command wins
    if re.search(r"\bfollow\b.*\bdog\b", t):
        follow["on"] = True; return "following the dog at ~1.1-1.4 m (re-goal every 1.5 s)"
    follow["on"] = False
    if MODE["send_all_vln"]:                                    # 'Send all to LightNav': only an exact stop is not an instruction
        c = AGENT.exact(t)
        if c is not None and c["kind"] == "stop": nav_cmd("stop"); return "stop (%.0f ms)" % ((time.monotonic() - t0) * 1e3)
        raw = " ".join(str(text).split())                      # the line as typed: no place/LLM rewriting
        res = nav_cmd("ask " + raw)
        return "→ LightNav: '%s'%s" % (raw, "" if res.get("kind") == "ask" else " (%s)" % (res.get("msg") or res))
    places = list((HUB["st"].get("places") or {}).keys())
    r = AGENT.interpret(t, places)
    if not r["cmds"]: return r["reply"] or "?"
    first, rest = r["cmds"][0], r["cmds"][1:]
    res = nav_cmd(first[0]); ms = (time.monotonic() - t0) * 1e3
    with lock:
        motion_seen = first[1]["kind"] in MOTION
        for tx, c in rest:
            if c["kind"] in MOTION or motion_seen: SEQ["q"].append(tx); motion_seen = True
            else: nav_cmd(tx)
    plan = " → ".join(tx for tx, _ in r["cmds"])
    if r["via"] == "grammar": return "%s (%.0f ms)" % (plan, ms)
    return "%s  [%s, %.1f s]" % ((r["reply"] + " " if r["reply"] else "") + "→ " + plan, AGENT.MODEL, r["ms"] / 1e3)

def chat(text, who="you"):
    t = str(text)[:300].strip()
    if not t: return ""
    log(who, t); rep = handle(t, who); log("kevin", rep); return rep

def set_mode(on, who="?"):
    if MODE["send_all_vln"] != on:
        MODE["send_all_vln"] = on; log(who, "Send all to LightNav: %s" % ("ON" if on else "OFF"))
        try: os.makedirs(os.path.dirname(MODE_FILE), exist_ok=True); json.dump(MODE, open(MODE_FILE, "w"))
        except Exception: pass

NAVEV = {"t": None}
def relay_nav_events(events):
    """nav runs commands asynchronously (POST /api/command only queues them), so its outcome - 'unreachable ...',
    'arrived ...', 'VLN: model says stop', 'gave up ...' - arrives later as an event.  Put each new one in the shared
    chat log so a command never looks silently ignored.  Events already there when the viewer starts are not replayed."""
    if not events: return
    if NAVEV["t"] is None: NAVEV["t"] = max(e[0] for e in events); return
    new = [e for e in events if e[0] > NAVEV["t"]]
    for t, s in new:
        shape = re.sub(r"-?\d+(\.\d+)?", "#", s)               # same message up to its numbers (LightNav re-goals ~3/s)
        with lock:
            if LOG and LOG[-1][1] == "nav" and LOG[-1][3:4] == [shape]:   # update that line in place, with a count
                n = LOG[-1][4] + 1; LOG[-1][:] = [time.strftime("%H:%M:%S"), "nav", "%s  (x%d)" % (s, n), shape, n]
            else: LOG.append([time.strftime("%H:%M:%S"), "nav", s, shape, 1]); del LOG[:-60]
            LOGN[0] += 1
    if new: NAVEV["t"] = new[-1][0]

def hub():   # authoritative merged state for every client (WS broadcast + /state.json polling), ~15 Hz
    while True:
        t0 = time.time(); S = sim(); st = {"t": t0}
        if S: st.update(kevin=S.get("kevin"), dog=S.get("dog"), collisions=S.get("collisions"), env=S.get("env"))
        try:
            n = json.loads(nav("/api/state"))
            st.update(goal=n.get("goal"), path=n.get("path"), mode=n.get("mode"), msg=n.get("msg"), core_alarm=n.get("core_alarm"), pose=n.get("pose"), places=n.get("places"),
                      safety=((n.get("safety") or {}).get("last") or {}).get("state"), cmd=n.get("cmd"), events=n.get("events"), hb=n.get("hb"),
                      bump=dict(latched=((n.get("bump") or {}).get("latched")), armed=(n.get("bump") or {}).get("armed")))
            relay_nav_events(n.get("events"))
            update_anchor(S, n)
            st["vln"] = n.get("vln")                            # wp_map / goal: nav frame, drawn under the anchor like everything nav
            st["odom"] = n.get("odom")
        except Exception as e: st["nav_error"] = str(e)[:80]; st["core_alarm"] = "nav not answering: " + str(e)[:60]
        st["send_all_vln"] = MODE["send_all_vln"]; st["anchor"] = ANCHOR["a"]; st["anchor_src"] = ANCHOR["src"]; st["slam_map"] = SLAMMAP["meta"]
        with lock: st["log"] = [r[:3] for r in LOG[-30:]]; st["log_n"] = LOGN[0]
        with CL_LOCK: st["clients"] = len(CLIENTS)
        HUB["seq"] += 1; st["seq"] = HUB["seq"]; HUB["st"] = st
        msg = json.dumps(dict(type="state", **st), default=float)
        with CL_LOCK: cl = list(CLIENTS)
        for c in cl:
            try: c.send(msg)
            except Exception:
                with CL_LOCK: CLIENTS.discard(c)
        time.sleep(max(0.01, 1 / 15 - (time.time() - t0)))

def follower():
    last = 0
    while True:
        time.sleep(0.3)
        if not follow["on"] or time.time() - last < 1.5: continue
        S = sim()
        if not S or not S.get("dog"): continue
        last = time.time(); kx, ky = env2nav(S, *S["kevin"][:2]); dx, dy = env2nav(S, *S["dog"])
        d = math.hypot(dx - kx, dy - ky)
        if d < 1.4: continue                      # close enough: hold (nav keeps its own safety stop)
        gx, gy = dx - 1.1 * (dx - kx) / d, dy - 1.1 * (dy - ky) / d   # goal 1.1 m short of the dog
        try: nav("/api/command", {"text": "go to %.2f %.2f" % (gx, gy)})
        except Exception: pass

def labeler():   # label the env's places in the nav map once (map frame = start pose), like emu.eval does
    while True:
        time.sleep(2)
        S = sim()
        if not S or not S.get("places") or not S.get("start"): continue
        try: st = json.loads(nav("/api/state")); have = set(st.get("places") or {}); NAVPOSE[0] = tuple(st["pose"])
        except Exception: continue
        for nm, (x, y) in S["places"].items():
            if nm.replace(" ", "_") in have or nm in have: continue
            u, v = env2nav(S, x, y)
            try: nav("/api/command", {"text": "label %s at %.2f %.2f" % (nm, u, v)})
            except Exception: pass

EGO = [None]
def egoloop():   # processed obstacle input, robot-centric: RS1 floor-leveled ego labels + RS2 cone (same code nav uses)
    try:
        sys.path[:0] = [__import__('os').path.abspath(__import__('os').path.join(__import__('os').path.dirname(__file__), '..', '..')), __import__('os').path.join(__import__('os').path.abspath(__import__('os').path.join(__import__('os').path.dirname(__file__), '..', '..')), 'src')]
        import nav  # noqa
        from nav.hw import Hardware
        from nav.perceive import Ego, Rs2Cloud
        from perception.labels import CLEAR, OBSTACLE, UNKNOWN
        from twin_geometry import FOOTPRINT_SAFE
        hw = Hardware(drive=False); ego = Ego(); rs2c = Rs2Cloud(hw.rs2_intr)
    except Exception as e:
        print("ego panel disabled:", e, flush=True); return
    S_, ppm, ox, oy = 280, 70.0, 140, 170            # 4 m x 4 m, robot at (140,170), forward = up
    while True:
        t0 = time.time()
        try:
            tv, verts = hw.wait_rs1(200)
            if verts is None: continue
            ego.update(verts); _, _, depth, *_ = hw.rs2_snapshot(); R2O, R2F = rs2c.points(depth)
            B = np.full((S_, S_, 3), 30, np.uint8)
            def bp(X, Y): return np.c_[ox - Y * ppm, oy - X * ppm].astype(np.int32)
            near = np.hypot(ego.X, ego.Y) < 0.6
            for msk, col in ((ego.lab == CLEAR, (130, 130, 130)), ((ego.lab == UNKNOWN) & near & ~ego.body, (200, 0, 200)),
                             (ego.lab == OBSTACLE, (40, 40, 235)), (ego.body, (0, 120, 200))):
                Q = bp(ego.X[msk], ego.Y[msk]); ok = (Q[:, 0] >= 0) & (Q[:, 0] < S_) & (Q[:, 1] >= 0) & (Q[:, 1] < S_); B[Q[ok, 1], Q[ok, 0]] = col
            for P_, col in ((R2F[::2], (60, 170, 60)), (R2O, (0, 165, 255))):
                if len(P_):
                    Q = bp(P_[:, 0], P_[:, 1]); ok = (Q[:, 0] >= 0) & (Q[:, 0] < S_) & (Q[:, 1] >= 0) & (Q[:, 1] < S_); B[Q[ok, 1], Q[ok, 0]] = col
            F = np.asarray(FOOTPRINT_SAFE); cv2.polylines(B, [bp(F[:, 0], F[:, 1])], True, (255, 255, 255), 1)
            cl = getattr(ego, "cliff_px", 0) or 0
            cv2.putText(B, "obstacle input (4x4 m)", (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(B, "red obst  magenta blank<0.6m  green/orange RS2  drop-off px %d" % cl, (4, S_ - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (220, 220, 220), 1, cv2.LINE_AA)
            EGO[0] = B
        except Exception as e:
            time.sleep(0.2)
        time.sleep(max(0.0, 0.15 - (time.time() - t0)))

def mapper():
    while True:
        t0 = time.time()
        try:
            st = json.loads(nav("/api/state")); meta = json.loads(nav("/api/map.json")); png = nav("/api/map.png")
            M = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE); M = cv2.cvtColor(M, cv2.COLOR_GRAY2BGR)
            x0m, y0m, x1m = meta["x0"], meta["y0"], meta["x1"]; H0 = M.shape[0]; r = H0 / (x1m - x0m)
            if time.time() - SLAMMAP["t"] > 1.0: slam_rgba(M[:, :, 0], meta)
            x, y, th = st["pose"]; NAVPOSE[0] = (x, y, th)
            with lock:
                if not TRAJ or math.hypot(TRAJ[-1][0] - x, TRAJ[-1][1] - y) > 0.03: TRAJ.append((x, y)); del TRAJ[:-3000]
                tr = list(TRAJ)
            hw = 4.5; n_ = int(2 * hw * r)
            a0 = int(np.clip((x - hw - x0m) * r, 0, H0 - n_)); b0 = int(np.clip(H0 - (y + hw - y0m) * r, 0, H0 - n_))
            M = M[b0:b0 + n_, a0:a0 + n_]; X0 = x0m + a0 / r; Y1 = y0m + (H0 - b0) / r
            S_ = 560; M = cv2.resize(M, (S_, S_), interpolation=cv2.INTER_NEAREST); k = S_ / n_ * r
            px = lambda u, v: (int((u - X0) * k), int((Y1 - v) * k))
            if len(tr) > 1: cv2.polylines(M, [np.array([px(*p) for p in tr], np.int32)], False, (255, 120, 255), 1, cv2.LINE_AA)
            if st.get("path"): cv2.polylines(M, [np.array([px(*p[:2]) for p in st["path"]], np.int32)], False, (255, 160, 0), 2, cv2.LINE_AA)
            for nm, p in (st.get("places") or {}).items():
                cv2.circle(M, px(*p[:2]), 3, (0, 220, 255), -1); cv2.putText(M, nm, (px(*p[:2])[0] + 4, px(*p[:2])[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 255), 1, cv2.LINE_AA)
            g = st.get("goal")
            if g: cv2.drawMarker(M, px(*g[:2]), (0, 0, 255), cv2.MARKER_STAR, 20, 2)
            S = sim()
            if S and S.get("dog"): cv2.circle(M, px(*env2nav(S, *S["dog"])), 6, (60, 120, 200), -1); cv2.putText(M, "dog (GT)", px(*env2nav(S, *S["dog"])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 120, 200), 1)
            c, s = math.cos(th), math.sin(th); saf = (st.get("safety") or {}).get("last") or {}; ss = saf.get("state", "ok")
            cv2.polylines(M, [np.array([px(x + c * u - s * v, y + s * u + c * v) for u, v in FP], np.int32)], True, COL.get(ss, (80, 200, 80)), 2)
            cv2.arrowedLine(M, px(x, y), px(x + 0.4 * c, y + 0.4 * s), (0, 220, 0), 2, tipLength=0.3)
            cv2.putText(M, "%s | safety %s x%.2f | %s" % (st.get("mode"), ss, saf.get("scale", 1.0), (st.get("msg") or "")[:46]), (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            bl = ((st.get("bump") or {}).get("latched")) or None
            if bl and bl.get("x") is not None:                      # latched contact (nav/bump.py): red X + label
                cv2.drawMarker(M, px(bl["x"], bl["y"]), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 18, 3)
                cv2.putText(M, "BUMP " + str(bl.get("dir")), px(bl["x"] + 0.1, bl["y"] + 0.1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
            if S: cv2.putText(M, "contacts %s  dog contacts %s" % (S.get("collisions"), S.get("dog_contacts")), (6, S_ - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            if EGO[0] is not None: E_ = EGO[0]; M[S_ - E_.shape[0] - 22:S_ - 22, S_ - E_.shape[1]:] = E_
            MAPJPG[0] = cv2.imencode(".jpg", M, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
        except Exception as e:
            pass
        time.sleep(max(0.05, 0.25 - (time.time() - t0)))

SLAMMAP = {"png": b"", "meta": None, "t": 0.0, "v": 0}
def slam_rgba(G, meta):
    """nav map.png (20 m window, 2 cm, row 0 = +y edge; 20 occupied / 200 seen free / 60 unknown) -> RGBA PNG for the
    3-D view (unknown transparent).  The client textures a floor plane with it and extrudes the occupied cells."""
    A = np.zeros(G.shape + (4,), np.uint8)
    occ = G < 40; free = (G > 150)
    A[free] = (225, 232, 238, 150); A[occ] = (30, 34, 44, 255)              # cv2 writes BGRA
    ok, png = cv2.imencode(".png", A)
    if ok:
        SLAMMAP["v"] += 1; SLAMMAP["png"] = png.tobytes(); SLAMMAP["t"] = time.time()
        SLAMMAP["meta"] = dict(meta, v=SLAMMAP["v"], w=int(G.shape[1]), h=int(G.shape[0]))

CAMS = {"rs1_depth": ("kevin_rs1", "depth"), "rs1_color": ("kevin_rs1", "color"), "rs2_depth": ("kevin_rs2", "depth"), "rs2_color": ("kevin_rs2", "color"), "mast_rgb": ("kevin_mast", "color")}
_BLK = {}
def cam_jpeg(name):
    """live JPEG of one sim RealSense stream straight from /dev/shm (depth colorized 0.2..4 m). Returns (bytes, err)."""
    if name not in CAMS: return None, "unknown camera %s (use %s)" % (name, ", ".join(CAMS))
    sh, key = CAMS[name]
    try:
        from emu import shm
        info = json.load(open(shm.INFO_PATH)); pid = info.get("pid")
        b = _BLK.get(sh)
        if b is None or b[1] != pid:
            dev = info["mast"] if sh == "kevin_mast" else [d for d in info["devices"].values() if d.get("shm") == sh][0]
            b = _BLK[sh] = (shm.Block(sh, dev["layout"]), pid)
        blk = b[0]
        for _ in range(5):
            s0 = int(blk.hdr[0])
            if s0 & 1: time.sleep(0.002); continue
            a = blk.arr[key].copy()
            if int(blk.hdr[0]) == s0: break
        if key == "depth":
            d = a.astype(np.float32) / 1000.0
            v = np.clip((d - 0.2) / 3.8, 0, 1); im = cv2.applyColorMap((255 * (1 - v)).astype(np.uint8), cv2.COLORMAP_JET); im[a == 0] = 0
        else: im = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
        ok, jp = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 70]); return jp.tobytes(), None
    except Exception as e:
        _BLK.pop(sh, None); return None, "shm %s unavailable: %s" % (sh, e)
class _CamShm:   # shm-name matching (hotpatch, browser-verified): /cam/<name>.jpg
    get_jpeg = staticmethod(cam_jpeg)
CAMSHM = _CamShm()

PAGE = r"""<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>Kevin sim</title>
<style>body{margin:0;background:#111;color:#ddd;font:14px system-ui;display:flex;height:100vh}#v{flex:1;position:relative;overflow:hidden}
#v img{width:100%;height:100%;object-fit:contain;cursor:grab;user-select:none}#r{width:560px;display:flex;flex-direction:column;border-left:1px solid #333}
#m{width:560px;height:560px;background:#000}#log{flex:1;overflow:auto;padding:6px;font:12px monospace;white-space:pre-wrap}
#f{display:flex;padding:6px;gap:6px}#q{flex:1;font:15px system-ui;padding:6px;background:#222;color:#fff;border:1px solid #555}
#hint{position:absolute;left:8px;top:6px;font:12px system-ui;color:#ccc;background:#0008;padding:3px 6px;border-radius:4px}
@media(max-width:900px){body{flex-direction:column;height:auto}#v{height:56vw}#r,#m{width:100%;height:auto}}</style></head><body>
<div id=v><img id=img alt="3-D view"><div id=hint>3-D: drag to orbit, wheel to zoom. Cyan frustum = depth FOV (colorized depth), yellow = RGB FOV (live RGB) <label style="margin-left:12px"><input type=checkbox id=camvis checked> Camera vis</label></div></div>
<div id=r><img id=m alt=map><div id=log></div><form id=f><input id=q placeholder="go to the kitchen | go to the couch | follow the dog | stop | come back | forward 1 | turn left" autocomplete=off><button>send</button></form></div>
<script>
const img=document.getElementById('img'),m=document.getElementById('m'),lg=document.getElementById('log'),q=document.getElementById('q');
function loop(el,url,ms){let busy=false;setInterval(()=>{if(busy)return;busy=true;const n=new Image();n.onload=()=>{el.src=n.src;busy=false};n.onerror=()=>{busy=false};n.src=url+'?t='+Date.now()},ms)}
loop(img,'/view.jpg?v=2',120);loop(m,'/map.jpg',250);
let cam={az:0,elev:-62,dist:3.2,camvis:true},drag=null;
const camvisEl=document.getElementById('camvis');
camvisEl.onchange=()=>{cam.camvis=camvisEl.checked;sendCam()};
function sendCam(){/* M1: server-shared camera (POST /cam) removed; orbit is per-client in the WebGL viewer (/) */}
img.onmousedown=e=>{drag=[e.clientX,e.clientY];e.preventDefault()};
window.onmouseup=()=>{drag=null};
window.onmousemove=e=>{if(!drag)return;cam.az-=(e.clientX-drag[0])*0.4;cam.elev=Math.max(-89,Math.min(-5,cam.elev-(e.clientY-drag[1])*0.3));drag=[e.clientX,e.clientY];sendCam()};
img.onwheel=e=>{cam.dist=Math.max(0.8,Math.min(9,cam.dist*(e.deltaY>0?1.12:1/1.12)));sendCam();e.preventDefault()};
async function refreshLog(){try{const L=await (await fetch('/log')).json();lg.textContent=L.map(r=>r[0]+' '+r[1]+': '+r[2]).join('\n');lg.scrollTop=1e9}catch(e){}}
setInterval(refreshLog,1000);
document.getElementById('f').onsubmit=async e=>{e.preventDefault();const t=q.value.trim();if(!t)return;q.value='';
 await fetch('/chat',{method:'POST',body:JSON.stringify({text:t})});refreshLog()};
</script></body></html>"""

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, b, ct="application/json", code=200):
        self.send_response(code); self.send_header("Content-Type", ct); self.send_header("Content-Length", str(len(b))); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/": return self._send(open(os.path.join(os.path.dirname(__file__), "live.html")).read().encode(), "text/html; charset=utf-8")
        if p == "/jpeg": return self._send(PAGE.encode(), "text/html; charset=utf-8")
        if p == "/assets/kevin_lite.glb":
            try: return self._send(open(os.path.join(os.path.dirname(__file__), "assets", "kevin_lite.glb"), "rb").read(), "model/gltf-binary")
            except Exception: return self._send(b"", code=404)
        if p == "/safety.png":                                       # safety-core 2-D frame overlay (nav /api/safety.png)
            try: return self._send(nav("/api/safety.png"), "image/png")
            except Exception: return self._send(b"", code=404)
        if p == "/view.jpg":
            try: return self._send(open("/dev/shm/kevin_view.jpg", "rb").read(), "image/jpeg")
            except Exception: return self._send(b"", code=404)
        if p.startswith("/cam/"):
            cam_name = p[5:-4] if p.endswith(".jpg") else None
            if cam_name:
                jpg, err = CAMSHM.get_jpeg(cam_name)
                if jpg:
                    return self._send(jpg, "image/jpeg")
                else:
                    return self._send(json.dumps({"error": err}).encode(), "application/json", 503)
            return self._send(json.dumps({"error": "bad request"}).encode(), "application/json", 400)
        if p == "/slam.png": return self._send(SLAMMAP["png"], "image/png") if SLAMMAP["png"] else self._send(b"", code=404)
        if p == "/map.jpg": return self._send(MAPJPG[0], "image/jpeg") if MAPJPG[0] else self._send(b"", code=404)
        if p == "/log":
            with lock: return self._send(json.dumps([r[:3] for r in LOG]).encode())
        if p == "/sim.json": return self._send(json.dumps(sim()).encode())
        if p == "/scene.json":
            # Export current env geometry - try to load from sim state or use a default
            S = sim()
            if S and S.get("env"):
                try:
                    sys.path[:0] = [os.path.join(os.path.dirname(__file__), "..")]
                    from envs import ENVS
                    env = ENVS.get(S["env"], lambda _: None)(0)
                    if env:
                        geoms = []
                        extent = env.extent if hasattr(env, 'extent') else (-5, -5, 5, 5)
                        print(f"[viewer] exporting scene: env={S['env']}, extent={extent}, geoms={len(env.geoms)}", flush=True)
                        
                        for g in env.geoms:
                            color = {"plaster": 0xdcd8cc, "plaster2": 0xc8d0d8, "wood": 0x8c6138, "fabric": 0x4c5c78,
                                   "metal": 0x9ea2a8, "white": 0xededed, "dark": 0x262629}.get(g.mat, 0x888888)
                            gd = {"kind": g.kind, "pos": list(g.pos), "size": list(g.size), "yaw": g.yaw, "color": color}
                            if g.kind == "mesh" and g.mesh:
                                # Export mesh vertices/faces for walls (trimesh)
                                try:
                                    import trimesh
                                    print(f"[viewer] loading mesh: {g.mesh}", flush=True)
                                    m = trimesh.load(g.mesh)
                                    scale = g.size[0] if isinstance(g.size, (list, tuple)) else 1.0
                                    print(f"[viewer] mesh loaded: {len(m.vertices)} verts, {len(m.faces)} faces, scale={scale}", flush=True)
                                    # Transform vertices: scale + translate by pos (pos is offset from origin)
                                    verts = (m.vertices * scale + np.array(g.pos)).tolist()
                                    faces = m.faces.tolist()
                                    gd["vertices"] = verts
                                    gd["faces"] = faces
                                    print(f"[viewer] mesh exported: {len(verts)} verts, {len(faces)} faces", flush=True)
                                except Exception as e:
                                    print(f"[viewer] mesh export failed for {g.mesh}: {e}", flush=True)
                                    import traceback
                                    traceback.print_exc()
                            geoms.append(gd)
                        
                        result = {"geoms": geoms, "extent": list(extent)}
                        print(f"[viewer] scene.json ready: {len(geoms)} geoms", flush=True)
                        return self._send(json.dumps(result).encode())
                except Exception as e:
                    print(f"[viewer] scene.json export failed: {e}", flush=True)
                    import traceback
                    traceback.print_exc()
            return self._send(json.dumps({"geoms": [], "extent": [-5, -5, 5, 5]}).encode())
        if p == "/ws" and "websocket" in (self.headers.get("Upgrade") or "").lower():
            ws = WSL.WS(self)
            try:
                ws.send(json.dumps(dict(type="hello", seq=HUB["seq"]))); ws.send(json.dumps(dict(type="state", **HUB["st"]), default=float))
                with CL_LOCK: CLIENTS.add(ws)
                while True:
                    m = ws.recv()
                    if m is None: break
                    try: d = json.loads(m)
                    except Exception: continue
                    if d.get("type") == "chat":
                        who = re.sub(r"[^\w .-]", "", str(d.get("who") or "you"))[:24] or "you"
                        txt = str(d.get("text", ""))
                        if AGENT.exact(preprocess(txt)) and AGENT.exact(preprocess(txt))["kind"] == "stop": chat(txt, who)   # inline: instant
                        else: threading.Thread(target=chat, args=(txt, who), daemon=True).start()
                    elif d.get("type") == "mode" and "send_all_vln" in d: set_mode(bool(d["send_all_vln"]), str(d.get("who") or "?")[:24])
                    elif d.get("type") == "ping": ws.send(json.dumps(dict(type="pong", t=d.get("t"), seq=HUB["seq"])))
            except Exception: pass
            finally:
                with CL_LOCK: CLIENTS.discard(ws)
            return
        if p == "/state.json" and HUB["st"]: return self._send(json.dumps(HUB["st"], default=float).encode())
        if p == "/state.json":
            # Polled state (tunnel-friendly, prefer this over SSE)
            S = sim()
            st = {"kevin": None, "dog": None, "goal": None, "path": None}
            if S:
                st["kevin"] = S.get("kevin")
                st["dog"] = S.get("dog")
            try:
                navst = json.loads(nav("/api/state"))
                st["goal"] = navst.get("goal")
                st["path"] = navst.get("path")
            except: pass
            return self._send(json.dumps(st).encode())
        if p == "/state":
            # SSE stream for live state (may not work through cloudflared tunnel)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            try:
                while True:
                    S = sim()
                    if S:
                        # Compact state: robot, dog, goal from sim.json + nav state
                        st = {"kevin": S.get("kevin"), "dog": S.get("dog")}
                        try:
                            navst = json.loads(nav("/api/state"))
                            st["goal"] = navst.get("goal")
                            st["path"] = navst.get("path")
                        except: pass
                        self.wfile.write(("data: %s\n\n" % json.dumps(st)).encode())
                        self.wfile.flush()
                    time.sleep(0.05)
            except: pass
            return
        if p == "/favicon.ico": return self._send(b"", "image/x-icon")
        self._send(b"{}", code=404)
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); d = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/mode":
            if "send_all_vln" in d: set_mode(bool(d["send_all_vln"]), str(d.get("who") or "?")[:24])
            return self._send(json.dumps(MODE).encode())
        if self.path == "/chat":
            who = re.sub(r"[^\w .-]", "", str(d.get("who") or "you"))[:24] or "you"
            return self._send(json.dumps({"reply": chat(d.get("text", ""), who)}).encode())
        self._send(b"{}", code=404)

if __name__ == "__main__":
    for f in (hub, sequencer, follower, mapper, labeler, egoloop): threading.Thread(target=f, daemon=True).start()
    print("viewer on :%d (nav %s)" % (PORT, NAV), flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()

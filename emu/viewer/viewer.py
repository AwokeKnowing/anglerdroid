"""Kevin live sim viewer (lean, stdlib + cv2).  usage: viewer.py [port=8095] [nav_port=8090]
Live WebGL: /live (default) - Three.js sim with WebSocket state stream, orbit controls, chat
Legacy JPEG: /jpeg - server-side render with camera frustums
Map: nav SLAM map + Kevin footprint (safety-state color) + trajectory + goal + planned path + labeled places + dog.
Chat: text -> small parser -> nav POST /api/command (same interface as the nav page)."""
import sys, os, json, math, re, time, threading, urllib.request, struct, hashlib, numpy as np, cv2
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
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
TRAJ = []; LOG = []; follow = {"on": False}; lock = threading.Lock(); MAPJPG = [b""]
def log(who, t):
    with lock: LOG.append([time.strftime("%H:%M:%S"), who, t]); del LOG[:-60]

def handle(text):
    t = text.strip().lower().rstrip(".!?"); t = re.sub(r"^(please|kevin|hey kevin)\s+", "", t); t = re.sub(r"\s+please$", "", t)
    if re.search(r"\bfollow\b.*\bdog\b", t):
        follow["on"] = True; return "following the dog at ~1.1-1.4 m (re-goal every 1.5 s)"
    follow["on"] = False
    if re.fullmatch(r"(come back|come here|go home|return|go back( home)?|home)", t): t = "go to living room"
    t = re.sub(r"^(go|drive|head|navigate)( over)? to (the )?", "go to ", t)
    t = re.sub(r"^go to (?:the )?", "go to ", t)
    m = re.fullmatch(r"(?:go |move |drive )?(forward|ahead|back|backward)s?\s*([\d.]+)?\s*(m|meters?|metres?)?", t)
    if m: t = "%s %s" % ("forward" if m[1] in ("forward", "ahead") else "back", m[2] or "1")
    m = re.fullmatch(r"turn (left|right|around)\s*([\d.]+)?\s*(deg|degrees)?", t)
    if m: t = "turn left 180" if m[1] == "around" else "turn %s %s" % (m[1], m[2] or "90")
    try: r = json.loads(nav("/api/command", {"text": t}) or b"{}")
    except Exception as e: r = {"error": str(e)}
    return "%s -> %s" % (t, r.get("msg") or r.get("reply") or json.dumps(r)[:200])

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
            if S: cv2.putText(M, "contacts %s  dog contacts %s" % (S.get("collisions"), S.get("dog_contacts")), (6, S_ - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            if EGO[0] is not None: E_ = EGO[0]; M[S_ - E_.shape[0] - 22:S_ - 22, S_ - E_.shape[1]:] = E_
            MAPJPG[0] = cv2.imencode(".jpg", M, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
        except Exception as e:
            pass
        time.sleep(max(0.05, 0.25 - (time.time() - t0)))

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
function sendCam(){fetch('/cam',{method:'POST',body:JSON.stringify(cam)})}
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
        if p == "/view.jpg":
            try: return self._send(open("/dev/shm/kevin_view.jpg", "rb").read(), "image/jpeg")
            except Exception: return self._send(b"", code=404)
        if p.startswith("/cam/"):
            # Camera panel endpoints: serve sim camera frames from shm if available
            # TODO: wire lightweight JPEG thumbs from sim shm (rs1_depth.jpg, rs1_color.jpg, rs2_depth.jpg, rs2_color.jpg)
            cam_name = p[5:-4] if p.endswith(".jpg") else None
            if cam_name in ("rs1_depth", "rs1_color", "rs2_depth", "rs2_color"):
                try:
                    return self._send(open(f"/dev/shm/kevin_{cam_name}.jpg", "rb").read(), "image/jpeg")
                except Exception:
                    return self._send(b"", code=404)
            return self._send(b"", code=404)
        if p == "/map.jpg": return self._send(MAPJPG[0], "image/jpeg") if MAPJPG[0] else self._send(b"", code=404)
        if p == "/log":
            with lock: return self._send(json.dumps(LOG).encode())
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
        if self.path == "/cam":
            c = {k: float(d[k]) for k in ("az", "elev", "dist") if k in d}
            c["camvis"] = bool(d.get("camvis", True))
            open("/dev/shm/kevin_view_cam.tmp", "w").write(json.dumps(c)); os.replace("/dev/shm/kevin_view_cam.tmp", "/dev/shm/kevin_view_cam.json")
            return self._send(b"{}")
        if self.path == "/chat":
            t = str(d.get("text", ""))[:200]; log("you", t); r = handle(t); log("kevin", r); return self._send(json.dumps({"reply": r}).encode())
        self._send(b"{}", code=404)

if __name__ == "__main__":
    for f in (follower, mapper, labeler, egoloop): threading.Thread(target=f, daemon=True).start()
    print("viewer on :%d (nav %s)" % (PORT, NAV), flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()

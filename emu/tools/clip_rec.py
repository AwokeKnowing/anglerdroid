"""Clip recorder: polls nav (/api/state, /api/map.png) + sim RS2 color + RS1 ego top-down at FPS and writes
composed 1280x720 JPEG frames to OUTDIR (encode to H.264 elsewhere).  usage: clip_rec.py OUTDIR [port] [fps]
Stops when OUTDIR/STOP exists or nav goes away."""
import sys, os, time, json, math, urllib.request, numpy as np, cv2
sys.path[:0] = [__import__('os').path.abspath(__import__('os').path.join(__import__('os').path.dirname(__file__), '..', '..')), __import__('os').path.join(__import__('os').path.abspath(__import__('os').path.join(__import__('os').path.dirname(__file__), '..', '..')), 'src')]
import nav  # noqa
from nav.hw import Hardware
from nav.perceive import Ego, Rs2Cloud
from perception.labels import CLEAR, OBSTACLE
from twin_geometry import FOOTPRINT_SAFE
out = sys.argv[1]; port = int(sys.argv[2]) if len(sys.argv) > 2 else 8090; fps = float(sys.argv[3]) if len(sys.argv) > 3 else 6
os.makedirs(out, exist_ok=True); U = "http://localhost:%d" % port
def get(p, js=True):
    r = urllib.request.urlopen(U + p, timeout=2).read(); return json.loads(r) if js else r
hw = Hardware(drive=False); ego = Ego(); rs2c = Rs2Cloud(hw.rs2_intr); R2O = np.zeros((0, 2)); R2F = np.zeros((0, 2)); i = 0; fails = 0
COL = {"ok": (80, 200, 80), "limited": (0, 200, 255), "blocked": (40, 40, 255), "no_depth": (200, 0, 200)}
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); _dg_ok = True
while not os.path.exists(os.path.join(out, "STOP")):
    t0 = time.time()
    try:
        st = get("/api/state"); meta = get("/api/map.json"); png = get("/api/map.png", False); fails = 0
    except Exception:
        fails += 1
        if fails > 40 and i > 0: break
        time.sleep(0.25); continue
    F = np.zeros((720, 1280, 3), np.uint8)
    try:
        seq, stt, depth, color, *_ = hw.rs2_snapshot()
        F[0:360, 80:560] = cv2.resize(cv2.cvtColor(color, cv2.COLOR_RGB2BGR), (480, 360))
        tv, verts = hw.wait_rs1(200)
        if verts is not None: ego.update(verts)
        R2O, R2F = rs2c.points(depth)
        B = np.full((360, 640, 3), 40, np.uint8); ppm = 110.0; ox, oy = 170, 180     # robot at (170,180), x forward = right
        def bp(P): return np.c_[ox + P[:, 0] * ppm, oy - P[:, 1] * ppm].astype(np.int32)
        E = (ego.lab == CLEAR) | (ego.lab == OBSTACLE)
        for msk, col in ((ego.lab == CLEAR, (150, 150, 150)), (ego.lab == OBSTACLE, (40, 40, 230)), (ego.body, (200, 120, 0))):
            Q = bp(np.c_[ego.X[msk], ego.Y[msk]]); ok = (Q[:, 0] >= 0) & (Q[:, 0] < 640) & (Q[:, 1] >= 0) & (Q[:, 1] < 360); B[Q[ok, 1], Q[ok, 0]] = col
        for P_, col in ((R2F[::2], (60, 170, 60)), (R2O, (0, 165, 255))):
            if len(P_):
                Q = bp(P_); ok = (Q[:, 0] >= 0) & (Q[:, 0] < 640) & (Q[:, 1] >= 0) & (Q[:, 1] < 360); B[Q[ok, 1], Q[ok, 0]] = col
        hf = math.atan(0.5 * depth.shape[1] / hw.rs2_intr.get("fx", 430.0))
        for sgn in (-1, 1): cv2.line(B, (ox, oy), (int(ox + 4 * ppm * math.cos(hf)), int(oy - sgn * 4 * ppm * math.sin(hf))), (0, 200, 255), 1, cv2.LINE_AA)
        cv2.polylines(B, [bp(np.asarray(FOOTPRINT_SAFE))], True, (255, 255, 255), 1)
        F[360:720, 0:640] = B
    except Exception as e:
        pass
    # map panel 640x720 (crop square 640, center)
    M = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE); M = cv2.cvtColor(M, cv2.COLOR_GRAY2BGR)
    x0m, y0m, x1m, y1m = meta["x0"], meta["y0"], meta["x1"], meta["y1"]; H0 = M.shape[0]; r = H0 / (x1m - x0m)
    cx, cy = st["pose"][0], st["pose"][1]; hw_ = 4.0                    # 8 m window centred on the robot
    a0 = int(np.clip((cx - hw_ - x0m) * r, 0, H0 - 2 * hw_ * r)); b0 = int(np.clip(H0 - (cy + hw_ - y0m) * r, 0, H0 - 2 * hw_ * r))
    n_ = int(2 * hw_ * r); M = M[b0:b0 + n_, a0:a0 + n_]
    x0m = x0m + a0 / r; y1m = y0m + (H0 - b0) / r; y0m = y1m - n_ / r; x1m = x0m + n_ / r
    S = 640; M = cv2.resize(M, (S, S), interpolation=cv2.INTER_NEAREST); k = S / (x1m - x0m)
    def px(x, y): return int((x - x0m) * k), int(S - (y - y0m) * k)
    if st.get("path"):
        P = np.array([px(*p[:2]) for p in st["path"]], np.int32); cv2.polylines(M, [P], False, (255, 160, 0), 2)
    g = st.get("goal") or st.get("goal_last")
    if g: cv2.drawMarker(M, px(*g[:2]), (0, 0, 255), cv2.MARKER_STAR, 22, 2)
    x, y, th = st["pose"]; c, s_ = math.cos(th), math.sin(th)
    saf = (st.get("safety") or {}).get("last") or {}; stt_ = saf.get("state", "ok")
    # envelope: footprint + 3.5 cm margin (offset hull), colored by safety state
    Pw = [(x + c * u - s_ * v, y + s_ * u + c * v) for u, v in FOOTPRINT_SAFE]
    cv2.polylines(M, [np.array([px(*p) for p in Pw], np.int32)], True, (230, 230, 230), 2)
    cen = np.mean(FOOTPRINT_SAFE, 0); Q = [cen + (p - cen) * (1 + 0.035 / max(np.hypot(*(p - cen)), 1e-3)) for p in FOOTPRINT_SAFE]
    Qw = [(x + c * u - s_ * v, y + s_ * u + c * v) for u, v in Q]
    cv2.polylines(M, [np.array([px(*p) for p in Qw], np.int32)], True, COL.get(stt_, (80, 200, 80)), 2)
    for sgn in (-1, 1): cv2.line(M, px(x, y), px(x + 3.5 * math.cos(th + sgn * 0.75), y + 3.5 * math.sin(th + sgn * 0.75)), (0, 200, 255), 1, cv2.LINE_AA)
    cv2.arrowedLine(M, px(x, y), px(x + 0.35 * c, y + 0.35 * s_), (0, 220, 0), 2, tipLength=0.3)
    F[40:680, 640:1280] = M
    lab = "%s | %s | slam kfs %s loops %s" % (st.get("mode"), (st.get("label") or ""), (st.get("slam") or {}).get("kfs"), (st.get("slam") or {}).get("loops"))
    cv2.putText(F, lab, (650, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(F, (st.get("msg") or "")[:70], (650, 705), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1, cv2.LINE_AA)
    cv2.putText(F, "RS2 color (sim)", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(F, "top-down: RS1 ego (grey/red) + RS2 forward cone (green floor, orange obstacles)", (8, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(F, "safety: %s  clr %.3f" % (stt_, saf.get("clearance", float("nan"))), (8, 712), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL.get(stt_, (255, 255, 255)), 1, cv2.LINE_AA)
    if i % 100 == 0: _dg_ok = __import__("diskguard").check()
    if _dg_ok: cv2.imwrite(os.path.join(out, "f%06d.jpg" % i), cv2.resize(F, (F.shape[1] * 2 // 3, F.shape[0] * 2 // 3)) if F.shape[1] > 1100 else F, [cv2.IMWRITE_JPEG_QUALITY, 80])
    i += 1
    time.sleep(max(0, 1 / fps - (time.time() - t0)))
print("frames", i)

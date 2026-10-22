import numpy as np, math, json, glob, cv2, sys
sys.path[:0] = ['.', 'src']
from emu.tools.slamlog.slam2_replay import load, replay
from nav.slam import transform
import trimesh
lg = sys.argv[1]; tq = float(sys.argv[2]); out = sys.argv[3]
J = load(lg + "/slamjobs.pkl"); R = [json.loads(str(np.load(f, allow_pickle=True)["meta"])) for f in sorted(glob.glob(lg + "/nav/t*.npz"))]
r0 = R[0]["t"]; Jq = [j for j in J if j[0] - r0 <= tq]
s, Ts, ms = replay(Jq, timing=False)
G = s.grid; lo = G.L
rr = min(R, key=lambda r: abs(r["t"] - r0 - tq)); x, y, th = rr["pose"]; print("nav pose", rr["pose"], "mode", rr["mode"], "T", s.T)
# GT start (first gt row = robot at rest at nav start) -> walls into the nav frame
gt0 = [json.loads(l) for l in open(lg + "/index.jsonl") if '"gt"' in l][0]["gt"]
from emu.envs import _world_params
stl = _world_params().walls_stl(); m = trimesh.load(stl); ext = m.bounds; sc = 0.001 if (ext[1] - ext[0]).max() > 100 else 1.0; off = -ext[0] * sc
sec = m.section(plane_origin=[0, 0, (0.3 - off[2]) / sc], plane_normal=[0, 0, 1])
c, s_ = math.cos(-gt0[2]), math.sin(-gt0[2])
def w2n(P): d = P - np.array(gt0[:2]); return np.stack([c * d[:, 0] - s_ * d[:, 1], s_ * d[:, 0] + c * d[:, 1]], 1)
H = 2.0; res = G.res; px = 1.0 / res
img = np.zeros((int(2 * H * px), int(2 * H * px), 3), np.uint8)
def to_px(P): return np.round((P - np.array([x - H, y - H])) * px).astype(np.int32)
# map: log odds -> grey
i0, j0 = np.floor((np.array([x - H, y - H]) - G.origin) / res).astype(int); n = img.shape[0]
crop = lo[j0:j0 + n, i0:i0 + n]; img[..., :] = np.clip(128 - crop[..., None] * 20, 0, 255).astype(np.uint8)
for e in sec.entities:
    P = w2n(sec.vertices[e.points][:, :2] * sc + off[:2]); cv2.polylines(img, [to_px(P).reshape(-1, 1, 2)], False, (0, 200, 0), 1)
# live scan at tq (robot frame -> nav)
j = min(Jq, key=lambda j: abs(j[0] - r0 - tq)); Pw = transform(np.asarray(j[2], np.float32), (x, y, th))
for p in to_px(Pw): cv2.circle(img, tuple(int(v) for v in p), 1, (0, 0, 255), -1)
# footprint
from twin_geometry import FOOTPRINT_SAFE as FP
F = transform(np.asarray(FP, np.float32), (x, y, th)); cv2.polylines(img, [to_px(F).reshape(-1, 1, 2)], True, (255, 160, 0), 1)
# trail
tr = np.array([r["pose"][:2] for r in R if r["t"] - r0 <= tq and r["t"] - r0 > tq - 40]); cv2.polylines(img, [to_px(tr).reshape(-1, 1, 2)], False, (255, 0, 255), 1)
img = cv2.flip(img, 0); cv2.imwrite(out, cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)); print("wrote", out)

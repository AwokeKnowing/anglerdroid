"""Photoreal cameras for a recorded sim log (bench/slam/record.py --light): render the RTX casita (tools/realsim,
ovrtx real-time path tracing, the same stage + settings as the live bridge tools/realsim/rtx_bridge.py) at the
ground-truth base pose of every camera frame.  Exact sync by construction: frame i is rendered at gt(t_i) (linear
interpolation of the 100 Hz GT; angles unwrapped), full 6-DoF base pose (x, y, z, roll, pitch, yaw: the base tilts up
to ~2.5 deg on the carpet terrain, which the IMU and the MuJoCo depth see, so the colour must see it too).
  OUT/rtx_mast/%06d.jpg   mast 800x600 (f 252.95, k1 -0.0287, dressing.json)
  OUT/rtx_rs2/%06d.jpg    forward RS2 colour 640x480 (f 625.2)
  OUT/rtx.json            intrinsics + T_base_optical for both cameras (OpenCV optical frame)
run in ~/.venv-ovrtx with XDG_CACHE_HOME=/dev/shm/ovcache:
  python bench/slam/render_rtx.py DATA/<seq> [--usda ~/kevin_bench/m1/casita_slam.usda (dressing.json incl. the bedroom/laundry lamps)] [--warm 40]"""
import argparse, json, math, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); RS = os.path.join(os.path.dirname(os.path.dirname(HERE)), "tools", "realsim")
sys.path.insert(0, RS)
from ovrtx_render import RtxSim  # noqa: E402
from usd_scene import cam_matrix  # noqa: E402
MAST = dict(fwd=0.0291, z=0.99115, pitch=0.027961467909011526, roll=-0.05430334296121204)       # = rtx_bridge.py
RS2 = dict(fwd=-0.130066, left=0.015, z=0.482863, pitch=math.radians(27.0), roll=0.0)
MOUNT = dict(mast=MAST, rs2_640=RS2)


def cam_pose(base, m):
    x, y, t = base; l = m.get("left", 0.0)
    return dict(x=x + m["fwd"] * math.cos(t) - l * math.sin(t), y=y + m["fwd"] * math.sin(t) + l * math.cos(t), z=m["z"], yaw=t, pitch=m["pitch"], roll=m["roll"])


BASE_Z = 0.10065   # mount heights above are world z for the base at rest (twin: mast 0.8905 / rs2 0.382213 above base_link)
FLIP = np.diag([1.0, -1.0, -1.0, 1.0])  # OpenGL camera (-Z fwd, +Y up) <-> OpenCV optical (+Z fwd, +Y down)


def T_base_optical(m):
    """USD camera (row-vector local-to-world, looks -Z, +Y up) at base pose 0 -> 4x4 base_link<-OpenCV optical."""
    M = cam_matrix(cam_pose((0.0, 0.0, 0.0), m)).T @ FLIP; M[2, 3] -= BASE_Z
    return M


def T_world_base(x, y, z, roll, pitch, yaw):
    """extrinsic xyz Euler (= scipy Rotation.from_euler("xyz"), as eval.load_gt): R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]); Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]); Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    T = np.eye(4); T[:3, :3] = Rz @ Ry @ Rx; T[:3, 3] = (x, y, z); return T


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("seq"); ap.add_argument("--usda", default=os.path.expanduser("~/kevin_bench/m1/casita_slam.usda"))
    ap.add_argument("--warm", type=int, default=40); ap.add_argument("--settings", default='{"omni:rtx:post:dlss:execMode":["token","performance"]}')
    ap.add_argument("--start", type=int, default=0)
    a = ap.parse_args(); seq = os.path.abspath(a.seq)
    G = np.genfromtxt(os.path.join(seq, "gt.csv"), delimiter=",", names=True); F = np.genfromtxt(os.path.join(seq, "frames.csv"), delimiter=",", names=True)
    yaw = np.unwrap(G["yaw"]); fx = lambda k, t: np.interp(t, G["t"], k); TBO = {c: T_base_optical(MOUNT[c]) for c in ("mast", "rs2_640")}
    cams = ["mast", "rs2_640"]; dirs = dict(mast="rtx_mast", rs2_640="rtx_rs2")
    for d in dirs.values(): os.makedirs(os.path.join(seq, d), exist_ok=True)
    D = {c["name"]: c for c in json.load(open(os.path.join(RS, "dressing.json")))["cameras"]}
    json.dump({c: dict(intr=D[c], T_base_optical=T_base_optical(MOUNT[c]).tolist(), dir=dirs[c]) for c in cams} | dict(note="rendered at gt(t_frame), full 6-DoF base pose; T_base_optical relative to base_link (gt.csv frame); LdrColor, JPEG q90"),
              open(os.path.join(seq, "rtx.json"), "w"), indent=1)
    sim = RtxSim(a.usda, cams, settings={k: tuple(v) for k, v in json.loads(a.settings).items()})
    pool = ThreadPoolExecutor(4); pend = []; dt = 1.0 / 15
    def save(p, img): Image.fromarray(img).save(p, quality=90)
    def place(t):
        Twb = T_world_base(*(float(fx(k, t)) for k in (G["x"], G["y"], G["z"], G["roll"], G["pitch"], yaw)))
        sim.set_cams({"/World/Cams/" + c: (Twb @ TBO[c] @ FLIP).T for c in cams})
    idx = F["i"].astype(int); ts = F["t"]; t0 = time.time()
    place(ts[a.start])
    for _ in range(a.warm): sim.step(read=("LdrColor",), dt=dt)                   # converge the temporal accumulators at the start pose
    for k in range(a.start, len(idx)):
        if k > a.start: dt = float(ts[k] - ts[k - 1])
        place(ts[k]); o = sim.step(read=("LdrColor",), dt=dt)
        for c in cams:
            img = np.ascontiguousarray(o[(c, "LdrColor")][..., :3]); pend.append(pool.submit(save, os.path.join(seq, dirs[c], "%06d.jpg" % idx[k]), img))
        if len(pend) > 32: [f.result() for f in pend[:-8]]; pend = pend[-8:]
        if k % 300 == 0: print("rtx %d/%d %.1f fps" % (k, len(idx), (k - a.start + 1) / (time.time() - t0)), flush=True)
    [f.result() for f in pend]; sim.close()
    print("rtx done %d frames %.1f s" % (len(idx) - a.start, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()

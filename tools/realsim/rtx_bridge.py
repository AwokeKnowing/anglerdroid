"""Minimal MuJoCo-sim -> ovrtx live bridge (M1 optional): follow the sim robot pose and render its cameras.
Reads the robot base pose (x, y, yaw in the casita world frame) that emu/world.py publishes in /dev/shm/kevin_sim.json,
places the mast + forward RS2 colour cameras with the twin mounts (tools/realsim/poses.py), steps ovrtx once per frame
and atomically writes /dev/shm/kevin_rtx_<cam>.jpg (+ kevin_rtx.json with pose, frame id, latency).  Read-only on the sim.
usage: rtx_bridge.py <casita.usda> [--seconds S] [--cams mast,rs2_640] [--settings '{json}']"""
import argparse, json, math, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ovrtx_render import RtxSim  # noqa: E402
from usd_scene import cam_matrix  # noqa: E402

# twin mounts (tools/realsim/poses.py; duplicated here so the ovrtx venv needs no trimesh)
MAST = dict(fwd=0.0291, z=0.99115, pitch=0.027961467909011526, roll=-0.05430334296121204)
RS2 = dict(fwd=-0.130066, left=0.015, z=0.482863, pitch=math.radians(27.0), roll=0.0)

def cam_pose(base, m):
    x, y, t = base; l = m.get("left", 0.0)
    return dict(x=x + m["fwd"] * math.cos(t) - l * math.sin(t), y=y + m["fwd"] * math.sin(t) + l * math.cos(t), z=m["z"], yaw=t, pitch=m["pitch"], roll=m["roll"])

ap = argparse.ArgumentParser(); ap.add_argument("usda"); ap.add_argument("--seconds", type=float, default=30)
ap.add_argument("--cams", default="mast,rs2_640"); ap.add_argument("--settings", default='{"omni:rtx:post:dlss:execMode":["token","performance"]}')
a = ap.parse_args(); cams = a.cams.split(",")
sim = RtxSim(a.usda, cams, settings={k: tuple(v) for k, v in json.loads(a.settings).items()})
def save(c, img):  # JPEG encode off the render thread (PIL releases the GIL); atomic replace for readers
    Image.fromarray(img).save("/dev/shm/kevin_rtx_%s.tmp.jpg" % c, quality=85)
    os.replace("/dev/shm/kevin_rtx_%s.tmp.jpg" % c, "/dev/shm/kevin_rtx_%s.jpg" % c)

pool = ThreadPoolExecutor(len(cams)); pend = []
t0 = time.time(); n = 0; lat = []; tst = []; tsave = []
while time.time() - t0 < a.seconds:
    t1 = time.perf_counter()
    try: S = json.load(open("/dev/shm/kevin_sim.json")); base = S["kevin"]
    except Exception: time.sleep(0.01); continue
    sim.set_cams({"/World/Cams/" + c: cam_matrix(cam_pose(base, MAST if c == "mast" else RS2)) for c in cams})
    t2 = time.perf_counter(); o = sim.step(read=("LdrColor",)); t3 = time.perf_counter()
    imgs = {c: np.ascontiguousarray(o[(c, "LdrColor")][..., :3]) for c in cams}
    for f in pend: f.result()  # keep at most one frame of encodes in flight
    pend = [pool.submit(save, c, imgs[c]) for c in cams]
    n += 1; lat.append(time.perf_counter() - t1); tst.append(t3 - t2); tsave.append(time.perf_counter() - t3)
    json.dump(dict(frame=n, base=base, sim_t=S.get("t"), ms=round(lat[-1] * 1e3, 1)), open("/dev/shm/kevin_rtx.tmp.json", "w")); os.replace("/dev/shm/kevin_rtx.tmp.json", "/dev/shm/kevin_rtx.json")
for f in pend: f.result()
sim.close(); L = np.array(lat[10:]) * 1e3
print(json.dumps(dict(frames=n, cams=cams, ms_mean=round(float(L.mean()), 1), ms_p95=round(float(np.percentile(L, 95)), 1), fps=round(1e3 / float(L.mean()), 1),
                 step_ms=round(float(np.mean(tst[10:])) * 1e3, 1), post_ms=round(float(np.mean(tsave[10:])) * 1e3, 1))))

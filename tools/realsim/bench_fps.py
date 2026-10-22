"""fps benchmark of the casita stage in ovrtx RTX Real-Time Path-Tracing, one configuration per process.
Each frame: move the camera (0.5 deg yaw/frame, like a turning robot), step (dt 1/30), map LdrColor to CPU (what a sim
bridge needs).  Reports wall-clock ms per frame over N frames after W warm-up frames, plus nvidia-smi samples.
usage: bench_fps.py <casita.usda> <poses.json> <product> '<settings json>' <label> <out.jsonl>  env BF_N (300) BF_W (60)"""
import json, math, os, subprocess, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ovrtx_render import RtxSim  # noqa: E402
from usd_scene import cam_matrix  # noqa: E402
usda, pj, prod, sets, label, outp = sys.argv[1:7]; S = json.loads(sets)
N, W = int(os.environ.get("BF_N", 300)), int(os.environ.get("BF_W", 60))
look = json.load(open(pj))["looks"]["20261021-211921_m0"]; p = dict(look["rs2" if prod.startswith("rs2") else "mast"])
t0 = time.time(); sim = RtxSim(usda, [prod], settings={k: tuple(v) for k, v in S.items()}); t_start = time.time() - t0
cam = "/World/Cams/" + prod; ts = []; smi = []
for k in range(W + N):
    q = dict(p, yaw=p["yaw"] + math.radians(0.5 * k)); a = time.perf_counter()
    sim.set_cams({cam: cam_matrix(q)}); o = sim.step(read=("LdrColor",)); ts.append(time.perf_counter() - a)
    if k == W + N // 2:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,power.draw,clocks.sm,memory.used", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
img = o[(prod, "LdrColor")]; sim.close()
t = np.array(ts[W:]) * 1e3
r = dict(label=label, product=prod, res=list(img.shape[1::-1]), settings=S, n=N, startup_s=round(t_start, 1), ms_mean=round(float(t.mean()), 2),
         ms_p50=round(float(np.median(t)), 2), ms_p95=round(float(np.percentile(t, 95)), 2), fps=round(1e3 / float(t.mean()), 1), gpu=smi,
         time=time.strftime("%H:%M:%S"))
open(outp, "a").write(json.dumps(r) + "\n"); print(json.dumps(r), flush=True)

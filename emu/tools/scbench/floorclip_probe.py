"""Static RS1 floor-clip check on the live stack (sim or robot; read-only: reads the safety core's RS1 cloud from the bus).
  python -m emu.tools.scbench.floorclip_probe [frames=150] [clip_m ...]      (default clips: the resolved one + 0.925)
For every clip, on the SAME frames, through the safety core's own Ego (too-close + cliff levelling + label_rs1_ego +
obstacle_xy filters): safety-grade obstacle px outside the body per frame, frames with any, and how close they are.
Floor statistics (levelled RS1 depth of floor px) give the minimum object height each clip can see on THIS floor:
an object top h above a floor px at depth z reads z - h; detected if < clip.  Reports h where 50 / 95 % of floor px
would detect it, and whether a 4.0 cm object is detected on the median floor."""
import functools, json, sys, time
import numpy as np
sys.path[:0] = ["src", "."]
import robot_params as RP
from nav import bus, perceive as PE
from perception.labels import OBSTACLE

N = int(sys.argv[1]) if len(sys.argv) > 1 else 150
clips = [float(c) for c in sys.argv[2:]] or [float(RP.V["td_floor_clip_m"]), 0.925]
B = bus.Block("rs1", bus.layouts((480, 848))["rs1"])
orig = PE.label_rs1_ego
egos = {c: PE.Ego() for c in clips}
res = {c: dict(obs_px=[], near_m=[]) for c in clips}; floor = []
last = -1; n = 0; t0 = time.time()
while n < N and time.time() - t0 < N / 10 + 30:
    s, r = B.read(["verts", "n", "stamp"])
    if s == last or int(r["n"][0]) == 0: time.sleep(0.005); continue
    last = s; v = r["verts"][: int(r["n"][0])].copy(); n += 1
    for c, e in egos.items():
        PE.label_rs1_ego = functools.partial(orig, floor_clip_m=c)
        e.update(v); P = e.obstacle_xy()
        res[c]["obs_px"].append(len(P)); res[c]["near_m"].append(float(np.hypot(P[:, 0], P[:, 1]).min()) if len(P) else None)
    e = egos[clips[0]]; v2 = e._too_close(v)          # levelled cloud (same as the core's)
    z = v2[:, 2]; z = z[z > 0.3]; med = np.median(z); fl = z[np.abs(z - med) < 0.03]
    floor.append(np.percentile(fl, [1, 50, 99]))
PE.label_rs1_ego = orig
F = np.array(floor); zf_med = float(np.median(F[:, 1])); z_p1 = float(np.median(F[:, 0]))
out = dict(wheels=RP.V["wheels"], hash=RP.HASH, rs1_height_m=RP.V["rs1_height_m"], frames=n,
           floor_depth_m=dict(p1=round(z_p1, 4), p50=round(zf_med, 4), p99=round(float(np.median(F[:, 2])), 4)), clips={})
for c in clips:
    o = np.array(res[c]["obs_px"]); near = [x for x in res[c]["near_m"] if x is not None]
    out["clips"]["%.6f" % c] = dict(
        obstacle_px_mean=round(float(o.mean()), 1), obstacle_px_max=int(o.max()), frames_with_obstacles=int((o > 0).sum()),
        nearest_obstacle_m=round(min(near), 3) if near else None,
        min_height_cm_median_floor=round(100 * (zf_med - c), 2),          # 50 % of floor px see an object this tall
        min_height_cm_p99_floor=round(100 * (float(np.median(F[:, 2])) - c), 2),   # 99 %: worst (deepest) floor px
        floor_margin_cm_p1=round(100 * (z_p1 - c), 2),                     # nearest floor px vs clip (false-obstacle margin)
        detects_4cm_on_median_floor=bool(zf_med - 0.04 < c))
print(json.dumps(out))

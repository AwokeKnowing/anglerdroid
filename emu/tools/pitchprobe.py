"""body pitch/roll/height of base_link in the live sim while driving (read-only shm reader of kevin_campose)."""
import json, sys, time, math, urllib.request as u, numpy as np, mujoco as mj
sys.path.insert(0, "."); from emu import shm, depthcam
m = mj.MjModel.from_binary_path("/dev/shm/kevin_emu_model.mjb"); B = shm.Block("kevin_campose", depthcam.pose_layout(m))
def cmd(t): return json.loads(u.urlopen(u.Request('http://localhost:8090/api/command', json.dumps({'text': t}).encode(), {'Content-Type': 'application/json'})).read())
def st(): return json.loads(u.urlopen('http://localhost:8090/api/state').read())
R = []; last = -1
def sample(tag, secs):
    global last
    t0 = time.time()
    while time.time() - t0 < secs:
        s, ns, d = B.read()
        if s != last:
            last = s; q = d["qpos"]; w, x, y, z = q[3:7]
            roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)); pitch = math.asin(max(-1, min(1, 2 * (w * y - z * x))))
            R.append([ns * 1e-9, q[0], q[1], q[2], math.degrees(pitch), math.degrees(roll), tag])
        time.sleep(0.01)
sample(0, 3)
for c, secs in (("go to kitchen", 40), ("turn left 360", 7), ("go to couch", 40)):
    cmd(c); sample(1, secs)
cmd("stop"); sample(0, 2)
A = np.array(R, float); t = A[:, 0]; xy = A[:, 1:3]
v = np.r_[0, np.hypot(*np.diff(xy, axis=0).T) / np.maximum(np.diff(t), 1e-3)]
st0 = A[:, 6] == 0; mv = v > 0.1
out = dict(n=len(A), z_still_mm=float(np.median(A[st0, 3]) * 1e3), pitch_still_deg=float(np.median(A[st0, 4])), roll_still_deg=float(np.median(A[st0, 5])))
for nm, msk in (("moving", mv), ("all", np.ones(len(A), bool))):
    p = A[msk, 4] - out["pitch_still_deg"]; r = A[msk, 5] - out["roll_still_deg"]; z = (A[msk, 3] - np.median(A[st0, 3])) * 1e3
    out[nm] = dict(pitch_p50_abs=float(np.median(abs(p))), pitch_p95_abs=float(np.percentile(abs(p), 95)), pitch_max_abs=float(abs(p).max()),
                   roll_p95_abs=float(np.percentile(abs(r), 95)), roll_max_abs=float(abs(r).max()), z_p95_abs_mm=float(np.percentile(abs(z), 95)))
np.save("/tmp/cc/pitchprobe.npy", A); print(json.dumps(out, indent=1))

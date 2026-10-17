"""Auto-cal convergence across randomised forward-camera mounts.
python -m emu.autocal_eval --env room --pitches 24 25 26 27 28 29 30 [--out ~/.kevin/emu/autocal]"""
from __future__ import annotations
import argparse, json, math, os, subprocess, time
import numpy as np
from emu.eval import REPO, SIMPY, ROBPY
from emu.shm import read_info
from emu.safety_eval import lane


def truth_rp(pitch_deg, roll_deg):
    """correction (roll, pitch) the auto-cal should find: maps the nominal-believed 'up' onto true up."""
    def Ropt(p):
        s, c = math.sin(math.radians(p)), math.cos(math.radians(p))
        x = np.array([0, -1, 0.]); y = -np.array([s, 0, c]); return np.stack([x, y, np.cross(x, y)], 1)
    a = math.radians(roll_deg); Rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    dR = Rx @ Ropt(pitch_deg) @ Ropt(27.0).T
    v = dR.T @ np.array([0, 0, 1.0])          # same convention as nav.autocal.rot_to_up
    r = math.atan2(v[1], v[2]); p = -math.atan2(v[0], math.hypot(v[1], v[2])); return math.degrees(r), math.degrees(p)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--env", default="room"); ap.add_argument("--pitches", type=float, nargs="+", default=[24, 25, 26, 27, 28, 29, 30])
    ap.add_argument("--windows", type=int, default=6); ap.add_argument("--out", default=os.path.expanduser("~/.kevin/emu/autocal")); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); rng = np.random.default_rng(5); rows = []
    for i, p in enumerate(a.pitches):
        roll, dz = float(rng.uniform(-1, 1)), float(rng.uniform(-0.01, 0.01))
        subprocess.call(["pkill", "-f", "emu.world"]); subprocess.call(["pkill", "-f", "nav.run"]); time.sleep(1.5)
        env = dict(os.environ, MUJOCO_GL="egl", KEVIN_EMU_RS2_PITCH=str(p), KEVIN_EMU_RS2_ROLL=str(roll), KEVIN_EMU_RS2_DZ=str(dz))
        w = subprocess.Popen([SIMPY, "-m", "emu.world", "--env", a.env, "--seed", str(i)], cwd=REPO, env=env,
                             stdout=open(os.path.join(a.out, "world_%d.log" % i), "w"), stderr=subprocess.STDOUT)
        for _ in range(100):
            try:
                if read_info(0.2).get("pid") == w.pid: break
            except Exception: pass
            time.sleep(0.2)
        time.sleep(1.0)
        z = np.load("/dev/shm/kevin_gt_map.npz"); pose = lane(z["occ"].astype(bool), z["origin"], float(z["res"]), 3.5)
        f = os.path.join(a.out, "probe_%d.json" % i); cal = os.path.join(a.out, "calib_%d.json" % i)
        if os.path.exists(cal): os.remove(cal)
        subprocess.call(["scripts/kevin-hal", ROBPY, "-m", "emu.autocal_probe", f, json.dumps(pose), str(a.windows)], cwd=REPO,
                        env=dict(os.environ, KEVIN_HAL="sim", KEVIN_CALIB=cal), stdout=open(os.path.join(a.out, "probe_%d.log" % i), "w"), stderr=subprocess.STDOUT, timeout=300)
        w.terminate(); w.wait(10)
        try: R = json.load(open(f))
        except Exception as e: rows.append(dict(pitch=p, err=str(e))); print(rows[-1], flush=True); continue
        tr, tp = truth_rp(p, roll); e = R["est"]
        rows.append(dict(true_pitch=p, true_roll=round(roll, 2), true_dz_cm=round(100 * dz, 2), want_dpitch=round(tp, 2), want_droll=round(tr, 2),
                         est_dpitch=round(e["pitch_deg"], 2), est_droll=round(e["roll_deg"], 2), est_dz_cm=round(100 * e["dz"], 2),
                         err_pitch=round(e["pitch_deg"] - tp, 2), err_roll=round(e["roll_deg"] - tr, 2), err_dz_cm=round(100 * (e["dz"] - dz), 2),
                         n_grav=e.get("n_grav"), n_floor=e.get("n_floor"), hist_pitch=[round(h["pitch_deg"] - tp, 2) for h in R["hist"]],
                         nominal=R["nominal"], autocal=R["autocal"]))
        print(json.dumps(rows[-1]), flush=True)
    json.dump(rows, open(os.path.join(a.out, "autocal_%s.json" % a.env), "w"), indent=1)


if __name__ == "__main__":
    main()

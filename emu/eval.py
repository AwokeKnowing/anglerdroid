"""End-to-end evaluation in the emulator: for each env, start emu.world + the unmodified nav stack (KEVIN_HAL=sim),
label the env's places (as James would), drive a tour by place name, and log ground truth vs estimate.

  python -m emu.eval --envs room apartment ... [--tour 4] [--safety] [--out ~/.kevin/emu/eval]
Metrics per goal: success (GT within 0.35 m of goal), time, path length, collisions (physics contacts),
SLAM error (map-frame estimate vs GT), odom-only error, safety interventions."""
from __future__ import annotations
import argparse, json, math, os, subprocess, sys, time, urllib.request
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from emu import gt  # noqa
from emu.shm import read_info  # noqa
SIMPY = os.path.expanduser("~/.venv-kevinsim/bin/python"); ROBPY = os.path.expanduser("~/.venv-kevinrobot/bin/python")
URL = "http://localhost:%d"


def http(port, path, body=None):
    r = urllib.request.Request(URL % port + path, None if body is None else json.dumps(body).encode(), method="POST" if body else "GET")
    return json.loads(urllib.request.urlopen(r, timeout=3).read())


def cmd(port, text): return http(port, "/api/command", dict(text=text))


def start(env, seed, port, logdir, extra=()):
    os.makedirs(logdir, exist_ok=True)
    subprocess.call(["pkill", "-f", "emu.world"]); subprocess.call(["pkill", "-f", "nav.run"])
    for _ in range(50):                                     # wait until the previous world released its UDP ports
        if subprocess.call(["pgrep", "-f", "emu.world"], stdout=subprocess.DEVNULL) != 0: break
        time.sleep(0.2)
    time.sleep(0.5)
    w = subprocess.Popen([SIMPY, "-m", "emu.world", "--env", env, "--seed", str(seed)], cwd=REPO,
                         env=dict(os.environ, MUJOCO_GL="egl"), stdout=open(os.path.join(logdir, "world.log"), "w"), stderr=subprocess.STDOUT)
    time.sleep(1.0)
    for _ in range(100):
        try:
            if read_info(0.2).get("pid") == w.pid: break
        except Exception: pass
        time.sleep(0.2)
    n = subprocess.Popen(["scripts/kevin-hal", ROBPY, "-m", os.environ.get("KEVIN_NAV_MOD", "nav.run"), "--port", str(port), *extra], cwd=REPO,
                         env=dict(os.environ, KEVIN_HAL="sim"), stdout=open(os.path.join(logdir, "nav.log"), "w"), stderr=subprocess.STDOUT)
    for _ in range(240):
        try: http(port, "/api/state"); break
        except Exception: time.sleep(0.5)
    time.sleep(3.0)                    # gyro-bias ZUPT + first scans
    return w, n


def stop(*ps):
    for p in ps:
        if p and p.poll() is None: p.terminate()
    for p in ps:
        try: p.wait(10)
        except Exception: p.kill()


class Track:
    def __init__(self, port, start): self.port, self.start = port, start; self.rows = []
    def sample(self):
        s = gt.state(); d = http(self.port, "/api/state")
        g = gt.to_map((s["x"], s["y"], s["yaw"]), self.start)
        e, o = d["pose"], d["odom"]
        row = dict(t=time.monotonic(), gt=g, est=e, odom=o, coll=s["collisions"], mode=d["mode"], msg=d["msg"],
                   v=s["v"], cmd=d.get("cmd"), goal_snapped=d.get("goal_last"), safety=d["safety"], timing=d["timing"], slam=d["slam"], plan=d["plan_timing"])
        self.rows.append(row); return row


_EDT = {}
def reachable_target(place_w):
    """Honest target for a named place: nearest world point with >= 0.26 m GT clearance (the place itself may be a couch)."""
    if "e" not in _EDT:
        from scipy.ndimage import distance_transform_edt
        z = np.load("/dev/shm/kevin_gt_map.npz"); _EDT.update(e=distance_transform_edt(~z["occ"].astype(bool)) * float(z["res"]), o=z["origin"], r=float(z["res"]))
    e, o, r = _EDT["e"], _EDT["o"], _EDT["r"]
    ix, iy = int((place_w[0] - o[0]) / r), int((place_w[1] - o[1]) / r); R = int(1.5 / r)
    ys, xs = np.mgrid[max(iy - R, 0):min(iy + R, e.shape[0]), max(ix - R, 0):min(ix + R, e.shape[1])]
    ok = e[ys, xs] >= 0.26
    if not ok.any(): return None
    d = (ys - iy) ** 2 + (xs - ix) ** 2; d[~ok] = 1 << 30; j = np.unravel_index(np.argmin(d), d.shape)
    return (o[0] + (xs[j] + 0.5) * r, o[1] + (ys[j] + 0.5) * r)


def perr(a, b): return math.hypot(a[0] - b[0], a[1] - b[1]), abs((a[2] - b[2] + math.pi) % (2 * math.pi) - math.pi)


def tour(env, seed, port, n_goals, logdir, timeout=150.0, extra=()):
    w, n = start(env, seed, port, logdir, extra); _EDT.clear()
    info = read_info(); st = info["start"]; places = info["places"]
    tr = Track(port, st); res = []
    try:
        names = [k for k in places][:max(n_goals, 1)]
        for k in names:
            mx, my = gt.to_map((places[k][0], places[k][1], 0.0), st)[:2]
            cmd(port, "label %s at %.3f %.3f" % (k.replace(" ", "_"), mx, my))      # the place as James would click it
        time.sleep(0.5)
        for k in names:
            r0 = tr.sample(); t0 = time.monotonic(); c0 = r0["coll"]; i0 = r0["safety"]["interventions"]
            mx, my = gt.to_map((places[k][0], places[k][1], 0.0), st)[:2]
            rt = reachable_target(places[k]); tx, ty = gt.to_map((rt[0], rt[1], 0.0), st)[:2] if rt else (mx, my)
            cmd(port, "go to " + k.replace(" ", "_")); time.sleep(0.6)
            L = 0.0; last = r0["gt"]
            while time.monotonic() - t0 < timeout:
                r = tr.sample(); L += math.hypot(r["gt"][0] - last[0], r["gt"][1] - last[1]); last = r["gt"]
                if r["mode"] == "idle": break
                time.sleep(0.25)
            r = tr.sample(); dgoal = math.hypot(r["gt"][0] - mx, r["gt"][1] - my)
            dsn = math.hypot(r["gt"][0] - tx, r["gt"][1] - ty)
            # success (ground truth only): robot within 0.30 m of the nearest GT-reachable point of the place, and stopped
            ok = dsn < 0.30 and r["mode"] == "idle"
            res.append(dict(env=env, goal=k, ok=ok, d_place_gt=round(dgoal, 3), d_reachable_gt=round(dsn, 3), place_to_reachable=round(math.hypot(tx - mx, ty - my), 2), t=round(time.monotonic() - t0, 1), path_m=round(L, 2),
                            straight_m=round(math.hypot(mx - r0["gt"][0], my - r0["gt"][1]), 2), collisions=r["coll"] - c0,
                            interventions=r["safety"]["interventions"] - i0, slam_err=[round(x, 3) for x in perr(r["est"], r["gt"])],
                            odom_err=[round(x, 3) for x in perr(r["odom"], r["gt"])], msg=r["msg"]))
            print(json.dumps(res[-1]), flush=True)
            if not ok and r["mode"] != "idle": cmd(port, "stop"); time.sleep(0.5)
    finally:
        try: final = http(port, "/api/state")
        except Exception: final = {}
        stop(n, w)
    json.dump(dict(results=res, track=tr.rows, final=final), open(os.path.join(logdir, "run.json"), "w"))
    return res, tr.rows, final


def summarize(rows):
    e = np.array([perr(r["est"], r["gt"]) for r in rows]); o = np.array([perr(r["odom"], r["gt"]) for r in rows])
    dist = np.r_[0, np.cumsum([math.hypot(b["gt"][0] - a["gt"][0], b["gt"][1] - a["gt"][1]) for a, b in zip(rows, rows[1:])])]
    return dict(dist_m=round(float(dist[-1]), 1),
                slam_xy_mean=round(float(e[:, 0].mean()), 3), slam_xy_max=round(float(e[:, 0].max()), 3), slam_xy_final=round(float(e[-1, 0]), 3),
                slam_yaw_max_deg=round(math.degrees(e[:, 1].max()), 1),
                odom_xy_final=round(float(o[-1, 0]), 3), odom_xy_max=round(float(o[:, 0].max()), 3), odom_yaw_max_deg=round(math.degrees(o[:, 1].max()), 1),
                slam_drift_pct=round(100 * float(e[-1, 0]) / max(dist[-1], 1e-6), 2), odom_drift_pct=round(100 * float(o[-1, 0]) / max(dist[-1], 1e-6), 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", nargs="+", default=["room", "apartment", "corridor", "clutter", "office", "maze", "doors", "casita"])
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tour", type=int, default=9); ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--out", default=os.path.expanduser("~/.kevin/emu/eval")); ap.add_argument("--extra", nargs="*", default=[])
    a = ap.parse_args(); allres = []; summ = {}
    for env in a.envs:
        d = os.path.join(a.out, "%s_s%d" % (env, a.seed))
        res, rows, final = tour(env, a.seed, a.port, a.tour, d, extra=a.extra)
        allres += res
        if rows: summ[env] = dict(summarize(rows), goals=len(res), success=sum(r["ok"] for r in res),
                                 collisions=sum(r["collisions"] for r in res), interventions=sum(r["interventions"] for r in res),
                                 timing=final.get("timing"), plan=final.get("plan_timing"), slam=final.get("slam"))
        print(env, json.dumps(summ.get(env)), flush=True)
    json.dump(dict(results=allres, summary=summ), open(os.path.join(a.out, "summary_s%d.json" % a.seed), "w"), indent=1)


if __name__ == "__main__":
    main()

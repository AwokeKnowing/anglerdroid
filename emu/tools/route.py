"""python -m emu.tools.route OUT.json "turn left 360" "go to kitchen" ... : drive a route through nav (:8090) by command
text, each step to completion (mode back to idle, 45 s cap), and log [command, seconds, nav message].
Before the first step it labels the sim env's places in the nav map (the viewer's labeler, emu/viewer/viewer.py, does the
same continuously), so "go to <place>" works without the viewer.  Places come from the world (/dev/shm/kevin_sim.json)
in the GT frame and are put into the nav frame through the current GT and nav poses (inherits the current SLAM error,
which is what a goal given to nav should do)."""
import json, math, sys, time, urllib.request
NAV = "http://127.0.0.1:8090"


def nav(path, body=None, timeout=30):
    rq = urllib.request.Request(NAV + path, data=None if body is None else json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(rq, timeout=timeout))


def sim():
    try: return json.load(open("/dev/shm/kevin_sim.json"))
    except Exception: return None


def env2nav(S, nav_pose, x, y):
    gx, gy, ga = S.get("kevin") or S.get("start") or (0.0, 0.0, 0.0); nx, ny, na = nav_pose
    dx, dy = x - gx, y - gy; c, s = math.cos(-ga), math.sin(-ga); u, v = c * dx - s * dy, s * dx + c * dy
    c, s = math.cos(na), math.sin(na); return nx + c * u - s * v, ny + s * u + c * v


def label_places(wait_s=20.0):
    """label every env place nav does not know yet; returns the number labelled."""
    t_end = time.monotonic() + wait_s; S = sim()
    while not (S and S.get("places")) and time.monotonic() < t_end: time.sleep(0.5); S = sim()
    if not (S and S.get("places")): return 0
    st = nav("/api/state"); have = set(st.get("places") or {}); n = 0
    for nm, (x, y) in S["places"].items():
        if nm in have or nm.replace(" ", "_") in have: continue
        u, v = env2nav(S, st["pose"], x, y); nav("/api/command", {"text": "label %s at %.2f %.2f" % (nm, u, v)}); n += 1
    return n


def drive(route, cap_s=45.0):
    out = []
    for c in route:
        t0 = time.monotonic(); nav("/api/command", {"text": c}); time.sleep(0.6); s = nav("/api/state")
        while s["mode"] != "idle" and time.monotonic() - t0 < cap_s: time.sleep(0.1); s = nav("/api/state")
        out.append([c, round(time.monotonic() - t0, 1), s.get("msg")]); print(out[-1], flush=True)
    return out


if __name__ == "__main__":
    n = label_places(); print("labelled %d places" % n, flush=True)
    res = drive(sys.argv[2:]); json.dump(dict(route=res, labelled=n), open(sys.argv[1], "w"), indent=1)

#!/usr/bin/env python3
"""Manual pilot for Kevin (sim or real), run ON the robot host over SSH (it reads the nav HTTP API + the read-only shm bus).

  python scripts/pilot.py look [--tag T]             snapshot -> run/pilot/<time>_<tag>/  (PNGs + look.json)
  python scripts/pilot.py goal DX DY [DTHETA_DEG]    robot-frame goal: DX forward, DY left (m), DTHETA left-positive (deg)
                                                    hard caps |(DX,DY)| <= 0.5 m, |DTHETA| <= 90 deg (refused, never clamped)
  python scripts/pilot.py stop                       immediate stop (nav 'stop': cancels the goal, zero command)
  common: --url http://127.0.0.1:8080 (default: KEVIN_PILOT_URL, else the first of :8080 (real nav) / :8090 (sim nav) that answers)
          goal: --timeout S (default 45, then stop is sent)  --tag NAME (look folder prefix)   all: --out DIR (default run/pilot)

goal only PROPOSES: it posts the same text command the UI / chat agent use ('go X Y [H]' in the nav frame, or 'turn left|right
N'), so the planner, governor and the safety core (which owns the wheels) decide everything; nothing here bypasses them.
It waits for arrived / stopped / unreachable / timeout (timeout -> it sends 'stop'), then prints the final pose, the error vs
the request, every safety-core stop code seen while a command was non-zero, the nav events, and saves an after-look.
Ctrl-C during a goal sends 'stop'.

look saves (whatever the host has; missing sources are listed in look.json 'missing'):
  rs1_depth.png   RS1 top-down depth (colourised, fixed 0.80-1.05 m scale; red = nearer than the floor clip)
  rs1_color.png   RS1 colour (sim: emu shm; real: not published - the safety core owns the RS1 camera)
  rs2_color.png, rs2_depth.png   RS2 forward colour / depth (bus rs2img, from the camera process)
  mast_rgb.png    mast camera (sim: shm kevin_mast; real: USB webcam if fitted and free)
  local_map.png   the safety core's RS1 ego labels (1 cm/px, robot frame) with footprint, pose arrow, 0.5 m rings
  map.png, safety.png   nav's own /api/map.png (world map + path + pose) and /api/safety.png (safety-core frame)
  look.json       pose, odometry, mode, goal, slam stats, safety-core state, wheel speeds, resolved robot config hash
"""
import argparse, datetime, json, math, os, signal, sys, time, urllib.request
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "src")]
MAX_D, MAX_TH = 0.5, 90.0
STOP_CODES = ("brake", "collide", "band_front", "band_rear", "inop_age", "inop_cadence", "no_depth")
END_PREFIX = ("arrived", "stopped", "unreachable", "gave up", "refused", "SAFETY CORE ALARM", "BUMP", "drive done",
              "no room to turn", "dropped late")


# ---------------------------------------------------------------- HTTP
def _get(url, path, raw=False, timeout=3.0):
    with urllib.request.urlopen(url + path, timeout=timeout) as r:
        b = r.read()
    return b if raw else json.loads(b)


def _post(url, text, frame=None):
    d = dict(text=text); d.update(frame=frame) if frame else None
    req = urllib.request.Request(url + "/api/command", data=json.dumps(d).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3.0) as r:
        return json.loads(r.read())


def find_url(a):
    if a.url: return a.url.rstrip("/")
    if os.environ.get("KEVIN_PILOT_URL"): return os.environ["KEVIN_PILOT_URL"].rstrip("/")
    for p in (8080, 8090):
        u = "http://127.0.0.1:%d" % p
        try: _get(u, "/api/state", timeout=1.5); return u
        except Exception: pass
    sys.exit("pilot: no nav answering on :8080 or :8090 (start the stack, or pass --url)")


def wrap(a): return (a + math.pi) % (2 * math.pi) - math.pi


# ---------------------------------------------------------------- image sources (read-only)
def _sim_info():
    try:
        d = json.load(open("/dev/shm/kevin_emu.json")); os.kill(int(d["pid"]), 0); return d
    except Exception: return None


def _emu_block(name, layout):
    from emu import shm
    return shm.Block(name, [(n, tuple(s), d) for n, s, d in layout])


def _turbo(img_u8):
    import cv2
    return cv2.applyColorMap(img_u8, cv2.COLORMAP_TURBO)


def _depth_png(d_m, lo, hi, clip=None):
    """depth (m, 0 = invalid) -> BGR: turbo over [lo, hi]; invalid black; < clip outlined red (RS1 obstacle side)."""
    import cv2
    u = np.clip((d_m - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8); im = _turbo(u); im[d_m <= 0] = 0
    if clip is not None:
        m = ((d_m > 0) & (d_m < clip)).astype(np.uint8)
        if m.any():
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(im, cs, -1, (0, 0, 255), 1)
    return im


def _label(im, text):
    import cv2
    cv2.putText(im, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(im, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return im


def grab_images(out, url, st):
    """-> (saved file list, missing dict). Never writes to any bus/shm and never commands anything."""
    import cv2
    saved, missing = [], {}
    sim = _sim_info()
    try:
        import robot_params as RP; clip = float(RP.V["td_floor_clip_m"])
    except Exception as e: clip = None; missing["robot_params"] = repr(e)
    from nav import bus
    def save(name, im, text):
        cv2.imwrite(os.path.join(out, name), _label(im, text)); saved.append(name)
    # RS1 top-down depth: sim -> emu raw depth image; real -> the core's RS1 cloud on the bus re-projected with rs1info
    try:
        if sim:
            dev = [v for v in sim["devices"].values() if v["shm"] == "kevin_rs1"][0]
            _, _, r = _emu_block("kevin_rs1", dev["layout"]).read()
            d = r["depth"].astype(np.float32) * 0.001
            save("rs1_depth.png", _depth_png(d, 0.80, 1.05, clip), "RS1 depth (sim) 0.80-1.05 m, red: < clip %.4f" % clip)
            save("rs1_color.png", r["color"][..., ::-1].copy(), "RS1 colour (sim)")
        else:
            info = bus.Block("rs1info", bus.layouts((480, 848))["rs1info"]).read()[1]["f"]
            fx, fy, cx, cy, W, H = info; W, H = int(W), int(H)
            r = bus.Block("rs1", bus.layouts((480, 848))["rs1"]).read(["verts", "n"])[1]
            v = r["verts"][: int(r["n"][0])]; v = v[v[:, 2] > 0.05]
            u = np.round(v[:, 0] / v[:, 2] * fx + cx).astype(int); w = np.round(v[:, 1] / v[:, 2] * fy + cy).astype(int)
            k = (u >= 0) & (u < W) & (w >= 0) & (w < H); d = np.zeros((H, W), np.float32); d[w[k], u[k]] = v[k, 2]
            d = cv2.dilate(d, np.ones((3, 3), np.uint8))           # the cloud is decimated: fill the gaps for viewing
            save("rs1_depth.png", _depth_png(d, 0.80, 1.05, clip), "RS1 depth (core cloud) 0.80-1.05 m, red: < clip %.4f" % clip)
            missing["rs1_color.png"] = "real robot: RS1 colour is not published (the safety core owns the RS1 camera)"
    except Exception as e: missing["rs1_depth.png"] = repr(e)
    # RS2 colour + depth: the camera process' full frame on the bus
    try:
        f = bus.Block("rs2info", bus.layouts((480, 848))["rs2info"]).read()[1]["f"]; H2, W2 = int(f[5]), int(f[4])
        r = bus.Block("rs2img", bus.layouts((H2, W2))["rs2img"]).read(["depth", "color", "n"])[1]
        if int(r["n"][0]) == 0: raise RuntimeError("rs2img never written")
        col = r["color"]                                          # bus.RS2_COLOR_SHAPE = cameras.RSCamera.color (320x240)
        save("rs2_color.png", cv2.resize(col[..., ::-1], (640, 480), interpolation=cv2.INTER_NEAREST), "RS2 colour (bus, %dx%d)" % (col.shape[1], col.shape[0]))
        save("rs2_depth.png", _depth_png(r["depth"].astype(np.float32) * 0.001, 0.2, 4.0), "RS2 depth 0.2-4.0 m")
    except Exception as e:
        try:                                                      # sim nav without the camera process: emu shm
            dev = [v for v in sim["devices"].values() if v["shm"] == "kevin_rs2"][0]
            _, _, r = _emu_block("kevin_rs2", dev["layout"]).read()
            save("rs2_color.png", r["color"][..., ::-1].copy(), "RS2 colour (sim)")
            save("rs2_depth.png", _depth_png(r["depth"].astype(np.float32) * 0.001, 0.2, 4.0), "RS2 depth (sim) 0.2-4.0 m")
        except Exception as e2: missing["rs2"] = "%r / %r" % (e, e2)
    # mast RGB
    try:
        if sim:
            m = sim["mast"]; _, _, r = _emu_block(m["shm"], m["layout"]).read(); save("mast_rgb.png", r["color"][..., ::-1].copy(), "mast RGB (sim)")
        else:
            from cameras import find_rgb_device, mast_upright
            dev = find_rgb_device()
            if dev is None: raise RuntimeError("no USB mast camera found")
            cap = cv2.VideoCapture(dev); cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            ok, fr = False, None
            for _ in range(8): ok, fr = cap.read()                # let auto-exposure settle
            cap.release()
            if not ok: raise RuntimeError("mast camera %s: no frame (busy?)" % dev)
            save("mast_rgb.png", mast_upright(fr), "mast RGB (%s, raw rotated upright)" % dev)
    except Exception as e: missing["mast_rgb.png"] = repr(e)
    # local obstacle map = the safety core's own RS1 ego labels (robot frame, 1 cm/px), pose drawn
    try:
        import robot_config as RC
        r = bus.Block("ego", bus.layouts((480, 848))["ego"]).read(["lab", "obs_mask", "occluded", "frame"])[1]
        lab = r["lab"]; S = 2
        pal = np.array([[40, 40, 40], [140, 90, 40], [205, 205, 205], [30, 30, 220]], np.uint8)   # unknown self clear obstacle (BGR)
        im = pal[np.clip(lab, 0, 3)]; im[r["obs_mask"] > 0] = (0, 0, 255)
        im = cv2.resize(im, (lab.shape[1] * S, lab.shape[0] * S), interpolation=cv2.INTER_NEAREST)
        ox, oy = RC.RCX * S, RC.RCY * S; px = 100 * S                                # px per metre
        for rr in (0.5, 1.0, 1.5, 2.0): cv2.circle(im, (ox, oy), int(rr * px), (0, 160, 0), 1)
        try:
            import twin_geometry as TG
            poly = np.array([[ox + p[0] * px, oy - p[1] * px] for p in TG.FOOTPRINT_SAFE], np.int32); cv2.polylines(im, [poly], True, (255, 255, 0), 1)
        except Exception: pass
        cv2.arrowedLine(im, (ox, oy), (ox + int(0.3 * px), oy), (0, 200, 255), 2, tipLength=0.25)
        p = st["pose"]
        save("local_map.png", im, "core RS1 ego labels frame %d%s | pose %.2f %.2f %.0fdeg | rings 0.5 m" % (
            int(r["frame"][0]), " OCCLUDED" if int(r["occluded"][0]) else "", p[0], p[1], math.degrees(p[2])))
    except Exception as e: missing["local_map.png"] = repr(e)
    for name, path in (("map.png", "/api/map.png"), ("safety.png", "/api/safety.png")):
        try: open(os.path.join(out, name), "wb").write(_get(url, path, raw=True)); saved.append(name)
        except Exception as e: missing[name] = repr(e)
    return saved, missing, bool(sim)


def look(url, out_root, tag, extra=None):
    st = _get(url, "/api/state")
    out = os.path.join(out_root, datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ("_" + tag if tag else ""))
    os.makedirs(out, exist_ok=True)
    saved, missing, sim = grab_images(out, url, st)
    wheel = None
    try:
        from nav import bus
        w = bus.Block("wheel", bus.layouts((480, 848))["wheel"]).read()[1]; wheel = {k: float(v[0]) for k, v in w.items()}
    except Exception: pass
    try:
        import robot_params as RP; cfg = dict(hash=RP.HASH, wheels=RP.V["wheels"], td_floor_clip_m=RP.V["td_floor_clip_m"])
    except Exception as e: cfg = repr(e)
    core = (st.get("safety") or {}).get("core") or {}
    j = dict(t=time.time(), url=url, host="sim" if sim else "real", pose=st["pose"], odom=st.get("odom"),
             pose_deg=round(math.degrees(st["pose"][2]), 2), mode=st.get("mode"), msg=st.get("msg"), goal=st.get("goal"),
             core_alarm=st.get("core_alarm"), core=core, cmd=st.get("cmd"), slam=st.get("slam"), wheel=wheel, robot_config=cfg,
             events=st.get("events"), saved=saved, missing=missing)
    if extra: j.update(extra)
    json.dump(j, open(os.path.join(out, "look.json"), "w"), indent=1, default=str)
    return out, j


# ---------------------------------------------------------------- goal
def goal(url, a):
    dx, dy, dth = a.dx, a.dy, (a.dtheta or 0.0)
    if not all(map(math.isfinite, (dx, dy, dth))): sys.exit("pilot: refused, non-finite goal")
    if math.hypot(dx, dy) > MAX_D + 1e-9: sys.exit("pilot: refused, |(dx,dy)| = %.3f m > %.1f m cap" % (math.hypot(dx, dy), MAX_D))
    if abs(dth) > MAX_TH + 1e-9: sys.exit("pilot: refused, |dtheta| = %.1f deg > %.0f deg cap" % (abs(dth), MAX_TH))
    st0 = _get(url, "/api/state")
    if st0.get("core_alarm"): sys.exit("pilot: refused, safety core alarm: %s" % st0["core_alarm"])
    x0, y0, th0 = st0["pose"]
    c, s = math.cos(th0), math.sin(th0)
    tx, ty = x0 + c * dx - s * dy, y0 + s * dx + c * dy
    th_t = wrap(th0 + math.radians(dth)) if (a.dtheta is not None and dth != 0.0) or math.hypot(dx, dy) < 1e-6 else None
    if math.hypot(dx, dy) < 1e-6:
        if dth == 0.0: sys.exit("pilot: nothing to do (0 0 0)")
        text = "turn %s %.1f" % ("left" if dth > 0 else "right", abs(dth))
    else:
        text = "go %.3f %.3f" % (tx, ty) + ("" if th_t is None else " %.2f" % math.degrees(th_t))
    before, _ = look(url, a.out, (a.tag + "_" if a.tag else "") + "before", dict(request=dict(dx=dx, dy=dy, dtheta_deg=dth)))
    t_send = time.time()
    rep = _post(url, text, frame="nav")
    print("pilot: from %.3f %.3f %.1fdeg  request dx %.2f dy %.2f dth %.0f -> '%s' (nav frame) -> %s" % (
        x0, y0, math.degrees(th0), dx, dy, dth, text, rep.get("kind")))
    if rep.get("kind") in (None, "error"): sys.exit("pilot: nav rejected the command: %s" % rep)
    stopped_by_me = []
    def _sig(*_):
        _post(url, "stop"); stopped_by_me.append("ctrl-c"); print("\npilot: Ctrl-C -> stop sent")
    signal.signal(signal.SIGINT, _sig)
    codes, seen_ev, started, outcome, track = {}, [], False, None, []
    brakes0 = ((st0.get("safety") or {}).get("core") or {}).get("brakes")
    t_end = time.time() + a.timeout
    while True:
        time.sleep(0.1)
        try: st = _get(url, "/api/state")
        except Exception as e: print("pilot: state read failed: %r" % e); continue
        core = (st.get("safety") or {}).get("core") or {}
        mv = abs(core.get("v_cmd") or 0) > 1e-3 or abs(core.get("w_cmd") or 0) > 1e-3
        code = core.get("code")
        if code is not None:
            codes.setdefault(code, [0, 0]); codes[code][0] += 1; codes[code][1] += int(mv)
        p = st["pose"]; track.append([round(time.time() - t_send, 2), round(p[0], 3), round(p[1], 3), round(p[2], 3), code, round(core.get("v_out") or 0, 3)])
        for t, e in st.get("events") or []:
            if t >= t_send - 0.05 and [t, e] not in seen_ev: seen_ev.append([t, e]); print("  nav: %s" % e)
        if st.get("mode") != "idle": started = True
        fin = [e for t, e in seen_ev if e.startswith(END_PREFIX)]
        if st.get("mode") == "idle" and (started or fin) and (fin or time.time() - t_send > 1.0):
            outcome = fin[-1] if fin else "idle"; break
        if stopped_by_me: outcome = "stopped (ctrl-c)"; break
        if time.time() > t_end:
            _post(url, "stop"); outcome = "timeout %.0f s -> stop sent" % a.timeout; break
    time.sleep(0.5); st = _get(url, "/api/state"); x, y, th = st["pose"]
    brakes1 = ((st.get("safety") or {}).get("core") or {}).get("brakes")
    # error vs the request, in the START robot frame (forward / left) and heading
    ex, ey = x - tx, y - ty; ef, el = c * ex + s * ey, -s * ex + c * ey
    moved_f, moved_l = c * (x - x0) + s * (y - y0), -s * (x - x0) + c * (y - y0)
    safety_stops = {k: v[1] for k, v in codes.items() if k in STOP_CODES and v[1] > 0}
    res = dict(request=dict(dx=dx, dy=dy, dtheta_deg=dth), command=text, outcome=outcome,
               start_pose=[x0, y0, th0], target=[tx, ty, th_t], final_pose=[x, y, th],
               moved=dict(forward_m=round(moved_f, 3), left_m=round(moved_l, 3), dtheta_deg=round(math.degrees(wrap(th - th0)), 1)),
               error=dict(dist_m=round(math.hypot(ex, ey), 3), forward_m=round(ef, 3), left_m=round(el, 3),
                          heading_deg=None if th_t is None else round(math.degrees(wrap(th - th_t)), 1)),
               safety_stops=safety_stops, core_codes={k: v[0] for k, v in codes.items()},
               core_brakes=None if brakes0 is None or brakes1 is None else brakes1 - brakes0,
               nav_events=[e for _, e in seen_ev], elapsed_s=round(time.time() - t_send, 1), before_look=before, track=track)
    after, _ = look(url, a.out, (a.tag + "_" if a.tag else "") + "after", dict(goal_result={k: v for k, v in res.items() if k != "track"}))
    res["after_look"] = after
    json.dump(res, open(os.path.join(after, "goal.json"), "w"), indent=1, default=str)
    print("pilot: outcome  %s  (%.1f s)" % (outcome, res["elapsed_s"]))
    print("pilot: final    %.3f %.3f %.1fdeg   moved fwd %.3f left %.3f dth %.1f" % (x, y, math.degrees(th), moved_f, moved_l, res["moved"]["dtheta_deg"]))
    print("pilot: error    %.3f m (fwd %+.3f, left %+.3f)%s" % (res["error"]["dist_m"], ef, el, "" if th_t is None else ", heading %+.1f deg" % res["error"]["heading_deg"]))
    print("pilot: safety   stops while commanding %s | core brakes %s | codes %s" % (safety_stops or "none", res["core_brakes"], res["core_codes"]))
    print("pilot: looks    before %s\n                after  %s  (goal.json)" % (before, after))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url"); ap.add_argument("--out", default=os.path.join(ROOT, "run", "pilot"))
    sp = ap.add_subparsers(dest="cmd", required=True)
    l = sp.add_parser("look"); l.add_argument("--tag", default="look")
    g = sp.add_parser("goal"); g.add_argument("dx", type=float); g.add_argument("dy", type=float)
    g.add_argument("dtheta", type=float, nargs="?"); g.add_argument("--timeout", type=float, default=45.0); g.add_argument("--tag", default="")
    sp.add_parser("stop")
    a = ap.parse_args(); url = find_url(a)
    if a.cmd == "stop":
        print("pilot: stop -> %s" % _post(url, "stop")); st = _get(url, "/api/state")
        print("pilot: mode %s, pose %.3f %.3f %.1fdeg" % (st["mode"], st["pose"][0], st["pose"][1], math.degrees(st["pose"][2])))
    elif a.cmd == "look":
        out, j = look(url, a.out, a.tag)
        print("pilot: look -> %s\n  saved %s\n  missing %s\n  pose %.3f %.3f %.1fdeg  mode %s  core %s" % (
            out, " ".join(j["saved"]), j["missing"] or "none", j["pose"][0], j["pose"][1], j["pose_deg"], j["mode"], (j["core"] or {}).get("code")))
    else:
        goal(url, a)


if __name__ == "__main__":
    main()

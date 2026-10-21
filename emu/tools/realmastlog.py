"""Real-robot mast camera log for maploc / trajectory-error work (read-only: camera only, never touches motors or nav).
python emu/tools/realmastlog.py <dir> [secs=0 (until Ctrl-C)] [hz=10]
Run it NEXT TO nav with KEVIN_RECORD=<dir>/nav (RS1/RS2 depth, IMU, wheels/odometry, command, pose per tick) and nav's
mast grab disabled (KEVIN_RGB_DEV=none, so VLN does not open the same webcam).  Layout follows emu/tools/sensorlog.py
(mast part): <dir>/mast/<stamp_ns>.jpg, index.jsonl rows {"cam": "mast", "stamp_ns", "file"},
info.json {"mast": lens + pose, "real": true}.  stamp_ns = CLOCK_MONOTONIC at the end of the V4L2 read (nav uses the
same clock); exposure/transfer latency is not removed (USB 2 webcam, ~1 frame).  The lens in info.json is the TWIN lens
(emu.realism.mast_truth, 118 deg DFOV), not a calibration of this physical lens."""
import json, os, sys, time
import cv2
sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"), os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")]
MW, MH, MAST_POS = 640, 480, (0.0291, 0.0, 0.8905)



def _tire():
    """active tyre profile (KEVIN_TIRE, else tire_profiles.json default): maploc takes the camera height from it per log"""
    try:   # same resolution as src/tire_profile.resolve, without importing it (no stderr notice, no module-name clash)
        d = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "tire_profiles.json")))
        n = os.environ.get(d["env"], "").strip() or d["default"]; n = d["aliases"].get(n, n)
        return n if n in d["profiles"] else None
    except Exception:
        return None


def lens():
    try:
        from emu import realism
        t = realism.mast_truth(MW, MH); src = "twin truth (emu.realism.mast_truth); not calibrated on this lens"
    except Exception as e:   # same numbers as the sim's info["mast"] on 2026-10-20
        t = dict(fx=240.34424761102412, fy=240.34424761102412, cx=319.5, cy=239.5, k1=-0.05, k2=0.008, hfov_deg=106.18134969836191, dfov_deg=118.0)
        src = "twin truth constants (emu.realism import failed: %r)" % e
    return dict(intr=dict(w=MW, h=MH, fx=t["fx"], fy=t["fy"], ppx=t["cx"], ppy=t["cy"]), dist=[t["k1"], t["k2"], 0.0, 0.0], dist_model="brown",
                hfov_deg=t["hfov_deg"], dfov_deg=t["dfov_deg"], lens_source=src)


def main():
    d = sys.argv[1]; T = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0; hz = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
    from cameras import find_rgb_device
    if os.environ.get("KEVIN_RGB_DEV", "").lower() == "none": del os.environ["KEVIN_RGB_DEV"]   # "none" is for nav; this tool owns the webcam
    dev = find_rgb_device()
    if dev is None: raise SystemExit("realmastlog: no USB mast camera found")
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2); cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, MW); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, MH); cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened(): raise SystemExit("realmastlog: cannot open %s" % dev)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    os.makedirs(os.path.join(d, "mast"), exist_ok=True)
    json.dump(dict(real=True, device=dev, tire=_tire(), mast=dict(pos=list(MAST_POS), pitch_deg=0.0, hz=hz, frame_w=w, frame_h=h, **lens()),
                   t0_mono=time.monotonic(), t0_wall=time.time()), open(os.path.join(d, "info.json"), "w"), indent=1)
    idx = open(os.path.join(d, "index.jsonl"), "a", buffering=1); n = bad = 0; t0 = time.monotonic(); nxt = t0
    print("realmastlog: %s %dx%d -> %s at %.0f Hz" % (dev, w, h, d, hz), flush=True)
    try:
        while not T or time.monotonic() - t0 < T:
            ok, f = cap.read(); st = time.monotonic_ns()
            if not ok: bad += 1; time.sleep(0.01); continue
            if st * 1e-9 < nxt: continue                     # read at the camera rate (fresh frames), save at hz
            nxt = max(nxt + 1.0 / hz, st * 1e-9 - 0.5 / hz)
            if (w, h) != (MW, MH): f = cv2.resize(f, (MW, MH), interpolation=cv2.INTER_AREA)
            fn = "mast/%d.jpg" % st; cv2.imwrite(os.path.join(d, fn), f, [cv2.IMWRITE_JPEG_QUALITY, 92])
            idx.write(json.dumps(dict(cam="mast", stamp_ns=st, file=fn)) + "\n"); n += 1
    except KeyboardInterrupt: pass
    dt = time.monotonic() - t0
    json.dump(dict(saved=n, read_fail=bad, secs=round(dt, 1), hz=round(n / max(dt, 1e-9), 2)), open(os.path.join(d, "stats.json"), "w"))
    print("realmastlog: saved %d frames in %.1f s (%.1f Hz), read failures %d" % (n, dt, n / max(dt, 1e-9), bad), flush=True)


if __name__ == "__main__":
    main()

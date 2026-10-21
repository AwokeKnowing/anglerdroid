#!/usr/bin/env python3
"""RealSense depth fill-rate / frame-rate evaluation per exposure setting (stack must be STOPPED: opens the cameras).
  python scripts/rs_exposure_eval.py [--secs 2.5] [--out run/exposure_<time>.json]
For each camera x emitter {on, off} x exposure mode it streams depth 848x480 + colour 640x480 @30 and reports:
  fill (valid depth px fraction, whole frame), fill_roi (in the AE ROI = the floor region), fps (median from hardware
  timestamps), worst frame interval, actual exposure / gain (frame metadata), IR saturation (px >= 250).
Exposure modes:  legacy   manual 15 ms, gain 64 (cameras.py before 2026-10-21)
                 ae       auto exposure, firmware default limits (no cap)
                 ae_cap   auto exposure, Auto Exposure Limit <cap> us (toggle on) -> AE can never stretch the frame period
                 ae_cap_roi   ae_cap + AE region of interest on the floor
                 dark_worst   manual exposure = cap, max gain: the longest exposure ae_cap can ever pick (frame-rate check)
"""
import argparse, json, os, sys, time, datetime
import numpy as np
import pyrealsense2 as rs

ROIS = {"815412070676": (560, 40, 840, 470),      # RS1 top-down: floor right of the robot body (x0, y0, x1, y1 in depth px)
        "944622074292": (40, 300, 808, 470)}      # RS2 forward: lower band = floor ahead


def meta(f, k):
    try: return int(f.get_frame_metadata(k))
    except Exception: return None


def run(pipe, prof, secs):
    ts, fills, froi, exps, gains, sats = [], [], [], [], [], []
    ser = prof.get_device().get_info(rs.camera_info.serial_number); x0, y0, x1, y1 = ROIS.get(ser, (0, 0, 848, 480))
    for _ in range(45): pipe.wait_for_frames(2000)              # settle (AE converges in ~1 s)
    t_end = time.monotonic() + secs
    while time.monotonic() < t_end:
        f = pipe.wait_for_frames(2000); d = f.get_depth_frame(); ir = f.get_infrared_frame(1)
        a = np.asanyarray(d.get_data()); ts.append(d.get_timestamp())
        fills.append(float((a > 0).mean())); froi.append(float((a[y0:y1, x0:x1] > 0).mean()))
        exps.append(meta(d, rs.frame_metadata_value.actual_exposure)); gains.append(meta(d, rs.frame_metadata_value.gain_level))
        if ir: sats.append(float((np.asanyarray(ir.get_data()) >= 250).mean()))
    dt = np.diff(ts) / 1000.0
    return dict(fill=round(float(np.mean(fills)), 3), fill_roi=round(float(np.mean(froi)), 3),
                fps=round(float(1 / np.median(dt)), 2), worst_dt_ms=round(float(dt.max() * 1000), 1), frames=len(ts),
                exp_us=int(np.median([e for e in exps if e is not None])) if any(e is not None for e in exps) else None,
                gain=int(np.median([g for g in gains if g is not None])) if any(g is not None for g in gains) else None,
                ir_sat=round(float(np.mean(sats)), 4) if sats else None)


def configure(ds, mode, cap, ser):
    o = rs.option
    def s(k, v):
        if ds.supports(k): ds.set_option(k, v)
    if mode == "legacy": s(o.enable_auto_exposure, 0); s(o.exposure, 15000); s(o.gain, 64)
    elif mode == "dark_worst": s(o.enable_auto_exposure, 0); s(o.exposure, cap); s(o.gain, 248)
    else:
        s(o.auto_exposure_limit_toggle, 0 if mode == "ae" else 1)
        if mode != "ae": s(o.auto_exposure_limit, cap)
        s(o.enable_auto_exposure, 1)
        roi = ds.as_roi_sensor(); r = rs.region_of_interest()
        r.min_x, r.min_y, r.max_x, r.max_y = ROIS.get(ser) if mode == "ae_cap_roi" else (0, 0, 847, 479)
        for _ in range(5):                                          # the firmware rejects ROI briefly after AE toggles
            try: roi.set_region_of_interest(r); break
            except Exception: time.sleep(0.2)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--secs", type=float, default=2.5); ap.add_argument("--cap", type=int, default=30000)
    ap.add_argument("--out"); ap.add_argument("--modes", default="legacy,ae,ae_cap,ae_cap_roi,dark_worst"); ap.add_argument("--note", default="")
    a = ap.parse_args(); res = dict(time=datetime.datetime.now().isoformat(timespec="seconds"), note=a.note, cap_us=a.cap, rows=[])
    for dev in rs.context().query_devices():
        ser = dev.get_info(rs.camera_info.serial_number); name = dev.get_info(rs.camera_info.name)
        p = rs.pipeline(); c = rs.config(); c.enable_device(ser)
        c.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30); c.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        c.enable_stream(rs.stream.infrared, 1, 848, 480, rs.format.y8, 30)
        prof = p.start(c); ds = prof.get_device().first_depth_sensor()
        if ds.supports(rs.option.visual_preset): ds.set_option(rs.option.visual_preset, 3)
        try:
            for em in (1, 0):
                ds.set_option(rs.option.emitter_enabled, em); ds.set_option(rs.option.laser_power, 360 if em else 0)
                for mode in a.modes.split(","):
                    configure(ds, mode, a.cap, ser); r = run(p, prof, a.secs)
                    r.update(camera=name, serial=ser, emitter=em, mode=mode); res["rows"].append(r)
                    print("%-22s %s emitter %d %-11s fill %.3f roi %.3f fps %5.2f worst %5.1f ms exp %6s gain %4s irsat %s" % (
                        name, ser, em, mode, r["fill"], r["fill_roi"], r["fps"], r["worst_dt_ms"], r["exp_us"], r["gain"], r["ir_sat"]), flush=True)
        finally: p.stop()
    out = a.out or os.path.join(os.path.dirname(__file__), "..", "run", "exposure_%s.json" % time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(res, open(out, "w"), indent=1); print("->", out)


if __name__ == "__main__":
    main()

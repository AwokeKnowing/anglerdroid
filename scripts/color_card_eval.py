#!/usr/bin/env python3
"""RealSense colour: white balance / exposure evaluation against the WhiBal neutral card on the deck (stack STOPPED).
  python scripts/color_card_eval.py [--out DIR] [--kelvins 2800,3400,...]
  python scripts/color_card_eval.py --calibrate [--kelvins 4800]   -> rgb_card_gains for robots/<robot>/robot.json cameras
The card is rigidly in RS1's view (deck-mounted, RS1 looks straight down), so its pixels are a fixed ROI in the 640x480
colour frame: CARD_ROI.  It is neutral grey, so the card's mean RGB should be R = G = B.
Metrics per setting (colour 640x480 @30, the production stream):
  card_err_deg   angle between the card's mean RGB and the neutral axis (1,1,1)   0 = perfect white balance
  card_L         card mean luminance (0-255; clipped fraction alongside)
  carpet_*       the same carpet seen by RS1 (beside the robot) and RS2 (floor ahead): chroma r/g, b/g per camera and the
                 cross-camera angle between the two carpet colours (same material, same light -> should agree)
  fps            colour frame rate from hardware timestamps (AE priority 0 = constant frame rate)
Settings:  legacy      manual exposure 200 (x100 us), gain 96, auto white balance (cameras.py before 2026-10-21)
           auto        auto exposure (priority 0) + auto white balance (full frame: carpet / sun dominate)
           fixed_K     auto exposure with the AE ROI on the card (RS1) / floor (RS2) + fixed white balance K kelvin
"""
import argparse, json, os, time, datetime
import numpy as np, cv2
import pyrealsense2 as rs

RS1, RS2 = "815412070676", "944622074292"
import sys; sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")]
try:
    import robot_params as _RP; CARD_ROI = tuple(_RP.V["cam_rs1_card_roi"])     # x0 y0 x1 y1, RS1 colour 640x480
except Exception:
    CARD_ROI = (363, 222, 389, 252)            # grey field above the crosshair
CARPET = {RS1: (400, 10, 500, 60), RS2: (20, 330, 200, 470)}
AE_ROI = {RS1: CARD_ROI, RS2: (40, 300, 600, 470)}


def roi_stats(img_rgb, roi):
    x0, y0, x1, y1 = roi; p = img_rgb[y0:y1, x0:x1].reshape(-1, 3).astype(np.float64)
    m = p.mean(0); clip = float((p >= 254).any(1).mean())
    return m, clip


def neutral_err(m):
    return float(np.degrees(np.arccos(np.clip(m.sum() / (np.linalg.norm(m) * np.sqrt(3)), -1, 1))))


def ang(a, b):
    return float(np.degrees(np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))))


def setup(sens, ser, mode):
    o = rs.option
    def s(k, v):
        if sens.supports(k): sens.set_option(k, v)
    s(o.auto_exposure_priority, 0)
    if mode == "legacy":
        s(o.enable_auto_white_balance, 1); s(o.enable_auto_exposure, 0); s(o.exposure, 200); s(o.gain, 96); return
    s(o.enable_auto_exposure, 1)
    r = rs.region_of_interest(); r.min_x, r.min_y, r.max_x, r.max_y = (0, 0, 639, 479) if mode == "auto" else AE_ROI[ser]
    for _ in range(5):
        try: sens.as_roi_sensor().set_region_of_interest(r); break
        except Exception: time.sleep(0.2)
    if mode == "auto": s(o.enable_auto_white_balance, 1)
    else: s(o.enable_auto_white_balance, 0); s(o.white_balance, float(mode.split("_")[1]))


def grab(pipe, n=45, keep=15):
    ts, ims = [], []
    for i in range(n):
        f = pipe.wait_for_frames(2000).get_color_frame(); ts.append(f.get_timestamp())
        if i >= n - keep: ims.append(np.asanyarray(f.get_data()).astype(np.float32))
    dt = np.diff(ts[-keep:]) / 1000
    return np.mean(ims, 0).astype(np.uint8), float(1 / np.median(dt)), float(dt.max() * 1000)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="run/color_%s" % time.strftime("%Y%m%d_%H%M%S"))
    ap.add_argument("--kelvins", default="2800,3200,3600,4000,4400,4800,5200,5600,6000,6500"); ap.add_argument("--note", default="")
    ap.add_argument("--calibrate", action="store_true")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    if a.calibrate: a.kelvins = a.kelvins if "," not in a.kelvins else "4800"
    modes = ["legacy", "auto"] + ["fixed_%s" % k for k in a.kelvins.split(",")]
    pipes, sens = {}, {}
    for ser in (RS1, RS2):
        p = rs.pipeline(); c = rs.config(); c.enable_device(ser); c.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        pr = p.start(c); pipes[ser] = p
        sens[ser] = [x for x in pr.get_device().query_sensors() if "RGB" in x.get_info(rs.camera_info.name)][0]
    rows = []
    try:
        for mode in modes:
            r = dict(mode=mode); imgs = {}
            for ser in (RS1, RS2): setup(sens[ser], ser, mode)
            for ser in (RS1, RS2):
                img, fps, worst = grab(pipes[ser]); imgs[ser] = img
                r["fps_" + ser[-4:]] = round(fps, 2); r["worst_ms_" + ser[-4:]] = round(worst, 1)
                cm, _ = roi_stats(img, CARPET[ser]); r["carpet_" + ser[-4:]] = [round(float(v), 1) for v in cm]
                r["carpet_rg_bg_" + ser[-4:]] = [round(cm[0] / cm[1], 3), round(cm[2] / cm[1], 3)]
            m, clip = roi_stats(imgs[RS1], CARD_ROI)
            r.update(card_rgb=[round(float(v), 1) for v in m], card_err_deg=round(neutral_err(m), 2), card_L=round(float(m.mean()), 1), card_clip=round(clip, 3),
                     carpet_cross_deg=round(ang(np.array(r["carpet_" + RS1[-4:]]), np.array(r["carpet_" + RS2[-4:]])), 2))
            rows.append(r)
            for ser in (RS1, RS2):
                im = cv2.cvtColor(imgs[ser], cv2.COLOR_RGB2BGR)
                if ser == RS1: x0, y0, x1, y1 = CARD_ROI; cv2.rectangle(im, (x0, y0), (x1, y1), (0, 0, 255), 1)
                x0, y0, x1, y1 = CARPET[ser]; cv2.rectangle(im, (x0, y0), (x1, y1), (255, 0, 255), 1)
                cv2.imwrite(os.path.join(a.out, "%s_%s.png" % (mode, "rs1" if ser == RS1 else "rs2")), im)
            print("%-11s card err %5.2f deg  L %5.1f clip %.3f  carpet RS1 rg/bg %s RS2 %s cross %5.2f deg  fps %s/%s worst %s/%s ms" % (
                mode, r["card_err_deg"], r["card_L"], r["card_clip"], r["carpet_rg_bg_" + RS1[-4:]], r["carpet_rg_bg_" + RS2[-4:]],
                r["carpet_cross_deg"], r["fps_" + RS1[-4:]], r["fps_" + RS2[-4:]], r["worst_ms_" + RS1[-4:]], r["worst_ms_" + RS2[-4:]]), flush=True)
    finally:
        for p in pipes.values(): p.stop()
    json.dump(dict(time=datetime.datetime.now().isoformat(timespec="seconds"), note=a.note, card_roi=CARD_ROI, rows=rows),
              open(os.path.join(a.out, "color_eval.json"), "w"), indent=1)
    if a.calibrate:
        r = rows[-1]; m = np.array(r["card_rgb"]); g = m.mean() / m
        print("rgb_card_gains at %s: [%.4f, %.4f, %.4f]   (card error before %.2f deg)" % (r["mode"], g[0], g[1], g[2], r["card_err_deg"]))
    print("->", a.out)


if __name__ == "__main__":
    main()

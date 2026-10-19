"""Sim sensor log for offline SLAM benches: python emu/tools/sensorlog.py <dir> <secs> [hz]
Saves RS1/RS2 colour (jpg) + depth (png16, mm) and mast colour at <hz> (default 15; mast at its own 10 Hz) with the
camera stamp (ns, monotonic = sim capture time), and ground truth (emu ctl 'state') at ~30 Hz with the monotonic time
of the query.  Wheel speeds / gyro / commands come from nav's KEVIN_RECORD (same monotonic clock).  index.jsonl rows:
{"cam", "stamp_ns", "file"} or {"gt": [x, y, yaw, v, w], "t_mono", "t_sim"}.  info.json: shm info (intrinsics, mounts)."""
import json, os, sys, time
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from emu import shm, gt

def main():
    d = sys.argv[1]; T = float(sys.argv[2]); hz = float(sys.argv[3]) if len(sys.argv) > 3 else 15.0
    os.makedirs(d, exist_ok=True); info = shm.read_info(wait_s=2.0); json.dump(info, open(os.path.join(d, "info.json"), "w"))
    blk = {}
    for k, v in info["devices"].items():
        cam = "rs1" if "RS1" in k else "rs2"; blk[cam] = shm.Block(v["shm"], v["layout"]); os.makedirs(os.path.join(d, cam), exist_ok=True)
    blk["mast"] = shm.Block(info["mast"]["shm"], info["mast"]["layout"]); os.makedirs(os.path.join(d, "mast"), exist_ok=True)
    idx = open(os.path.join(d, "index.jsonl"), "w"); last = {k: (0, 0) for k in blk}; t0 = time.monotonic(); n = {k: 0 for k in blk}
    while time.monotonic() - t0 < T:
        tq = time.monotonic(); s = gt.state()
        idx.write(json.dumps(dict(gt=[s["x"], s["y"], s["yaw"], s["v"], s["w"]], t_mono=(tq + time.monotonic()) / 2, t_sim=s["t"])) + "\n")
        for cam, b in blk.items():
            seq = b.seq
            if seq == last[cam][0]: continue
            per = 1.0 / (10.0 if cam == "mast" else hz)
            if time.monotonic() - last[cam][1] < per * 0.9: continue
            _, st, a = b.read(); last[cam] = (seq, time.monotonic()); f = "%s/%d" % (cam, st)
            cv2.imwrite(os.path.join(d, f + ".jpg"), a["color"][..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
            if "depth" in a: cv2.imwrite(os.path.join(d, f + ".png"), a["depth"])
            idx.write(json.dumps(dict(cam=cam, stamp_ns=st, file=f)) + "\n"); n[cam] += 1
        time.sleep(max(0.0, 1 / 30 - (time.monotonic() - tq)))
    idx.close(); print("sensorlog", d, n)

if __name__ == "__main__": main()

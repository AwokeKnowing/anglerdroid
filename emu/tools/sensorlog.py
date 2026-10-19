"""Sim sensor log for offline SLAM benches: python emu/tools/sensorlog.py <dir> <secs> [hz]
Colour comes from the <dev>_side block (stamp = the pose it was rendered at) and is paired with the depth frame of the
SAME pose stamp (+-5 ms camera jitter) from a ring of recent depth frames: the combined kevin_rs* block carries whatever
colour the slow lane last produced (rendered at an older pose -- up to ~80 ms apart, 12 deg at 2.5 rad/s spins).
Saves RS1/RS2 colour (jpg) + depth (png16, mm) and mast colour at <hz> (default 15; mast at its own 10 Hz) with the
camera stamp (ns, monotonic = sim capture time), and ground truth (emu ctl 'state') at ~30 Hz with the monotonic time
of the query.  Wheel speeds / gyro / commands come from nav's KEVIN_RECORD (same monotonic clock).  index.jsonl rows:
{"cam", "stamp_ns", "file"} or {"gt": [x, y, yaw, v, w], "t_mono", "t_sim"}.  info.json: shm info (intrinsics, mounts)."""
import json, os, sys, time, collections
from concurrent.futures import ThreadPoolExecutor
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from emu import shm, gt

def main():
    d = sys.argv[1]; T = float(sys.argv[2]); hz = float(sys.argv[3]) if len(sys.argv) > 3 else 15.0
    os.makedirs(d, exist_ok=True); info = shm.read_info(wait_s=2.0); json.dump(info, open(os.path.join(d, "info.json"), "w"))
    blk, side, ring = {}, {}, {}
    p = json.load(open("/dev/shm/kevin_depthcam.json")); from emu.depthcam import side_layout
    for k, v in info["devices"].items():
        cam = "rs1" if "RS1" in k else "rs2"; blk[cam] = shm.Block(v["shm"], v["layout"]); side[cam] = shm.Block(v["shm"] + "_side", side_layout(p))
        ring[cam] = collections.deque(maxlen=12); os.makedirs(os.path.join(d, cam), exist_ok=True)
    mast = shm.Block(info["mast"]["shm"], info["mast"]["layout"]); os.makedirs(os.path.join(d, "mast"), exist_ok=True)
    idx = open(os.path.join(d, "index.jsonl"), "w"); ex = ThreadPoolExecutor(3); last = collections.defaultdict(lambda: (0, 0.0))
    t0 = time.monotonic(); n = collections.Counter(); per = 1.0 / hz
    while time.monotonic() - t0 < T:
        tq = time.monotonic(); s = gt.state()
        idx.write(json.dumps(dict(gt=[s["x"], s["y"], s["yaw"], s["v"], s["w"]], t_mono=(tq + time.monotonic()) / 2, t_sim=s["t"])) + "\n")
        for cam in blk:
            b = blk[cam]
            if b.seq != last["d" + cam][0]:
                sq = b.seq; _, st, a = b.read(); last["d" + cam] = (sq, 0); ring[cam].append((st, a["depth"]))
            sb = side[cam]
            if sb.seq == last[cam][0] or time.monotonic() - last[cam][1] < per * 0.9: continue
            sq = sb.seq; _, sc, c = sb.read(); last[cam] = (sq, time.monotonic())
            m = [(abs(st - sc), st, dd) for st, dd in ring[cam]]
            if not m or min(m, key=lambda x: x[0])[0] > 5e6: n[cam + "_unpaired"] += 1; continue
            _, st, dd = min(m, key=lambda x: x[0]); f = "%s/%d" % (cam, sc)
            ex.submit(cv2.imwrite, os.path.join(d, f + ".jpg"), c["color"][..., ::-1].copy(), [cv2.IMWRITE_JPEG_QUALITY, 90])
            ex.submit(cv2.imwrite, os.path.join(d, f + ".png"), dd, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            idx.write(json.dumps(dict(cam=cam, stamp_ns=sc, depth_stamp_ns=st, file=f)) + "\n"); n[cam] += 1
        if mast.seq != last["mast"][0]:
            sq = mast.seq; _, st, a = mast.read(); last["mast"] = (sq, 0); f = "mast/%d" % st
            ex.submit(cv2.imwrite, os.path.join(d, f + ".jpg"), a["color"][..., ::-1].copy(), [cv2.IMWRITE_JPEG_QUALITY, 90])
            idx.write(json.dumps(dict(cam="mast", stamp_ns=st, file=f)) + "\n"); n["mast"] += 1
        time.sleep(max(0.0, 1 / 60 - (time.monotonic() - tq)))
    ex.shutdown(); idx.close(); print("sensorlog", d, dict(n))

if __name__ == "__main__": main()

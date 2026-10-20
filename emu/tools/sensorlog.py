"""Sim sensor log for offline SLAM benches: python emu/tools/sensorlog.py <dir> <secs> [hz]
Colour comes from the <dev>_side block (stamp = the pose it was rendered at) and is paired with the depth frame of the
SAME pose stamp (+-5 ms camera jitter) from a ring of recent depth frames: the combined kevin_rs* block carries whatever
colour the slow lane last produced (rendered at an older pose -- up to ~80 ms apart, 12 deg at 2.5 rad/s spins).
Saves RS1/RS2 colour (jpg) + depth (png16, mm) and mast colour at <hz> (default 30 = every camera frame; mast at its own
10 Hz) with the camera stamp (ns, monotonic = sim capture time), and ground truth (emu ctl 'state') at ~30 Hz with the
monotonic time of the query.  Wheel speeds / gyro / commands come from nav's KEVIN_RECORD (same monotonic clock).
index.jsonl rows: {"cam", "stamp_ns", "file"} or {"gt": [x, y, yaw, v, w], "t_mono", "t_sim"}.  info.json: shm info.

v2 (2026-10-19): L3/L4 had 0.13-0.6 s holes during spins (20-56 deg between saved frames).  The old single loop did a
blocking GT round trip, copied whole camera blocks (depth+colour+IR, ~2.5 MB) for both cameras and kept only the LATEST
colour frame, so any iteration slower than one frame lost frames, and it logged at 10-15 Hz.  Now: one thread per block
waits on its seqlock and copies only what is saved; colour waits (<=100 ms) for its depth partner instead of being dropped;
GT has its own thread.  The writer's frame counter (hdr[2]) makes every loss visible: stats.json counts source frames,
reader misses, unpaired and saved per camera, plus the max gap between saved frames."""
import json, os, sys, time, threading, collections
from concurrent.futures import ThreadPoolExecutor
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from emu import shm, gt

def pread(b, keys):
    """seqlock read of only `keys` -> (stamp_ns, writer count, {k: copy}) or None (writer stuck)."""
    for _ in range(5000):
        s0 = int(b.hdr[0])
        if s0 & 1: time.sleep(0.0001); continue
        r = {k: b.arr[k].copy() for k in keys}; st = int(b.hdr[1]); cnt = int(b.hdr[2])
        if int(b.hdr[0]) == s0: return st, cnt, r
    return None

def main():
    d = sys.argv[1]; T = float(sys.argv[2]); hz = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0
    os.makedirs(d, exist_ok=True); info = shm.read_info(wait_s=2.0); json.dump(info, open(os.path.join(d, "info.json"), "w"))
    p = json.load(open("/dev/shm/kevin_depthcam.json")); from emu.depthcam import side_layout
    idx = open(os.path.join(d, "index.jsonl"), "w"); L = threading.Lock(); ex = ThreadPoolExecutor(4)
    S = collections.defaultdict(collections.Counter); saved = collections.defaultdict(list)
    t_end = time.monotonic() + T; min_dt = 0.9e9 / hz
    # DEFER (default): keep the raw frames in RAM and encode after the recording ends -- encoding 60 jpg + 60 png a
    # second took ~1 core and pushed the safety core's frame age past 100 ms (agewatch 17:59 PT).  ~1.7 MB/frame pair.
    DEFER = os.environ.get("SENSORLOG_DEFER", "1") == "1"; Q = []
    def put(path, img, prm, bgr):
        if DEFER: Q.append((path, img, prm, bgr))
        else: ex.submit(cv2.imwrite, path, img[..., ::-1].copy() if bgr else img, prm)
    def row(r):
        with L: idx.write(json.dumps(r) + "\n")
    def wait(b, last):
        while time.monotonic() < t_end:
            if b.wait_new(last, 0.1): return True
        return False
    def depth_loop(cam, b, ring, cv):
        last = b.seq; lc = None
        while wait(b, last):
            last = b.seq; r = pread(b, ["depth"])
            if r is None: S[cam]["depth_stuck"] += 1; continue
            st, cnt, a = r
            if lc is not None and cnt - lc > 1: S[cam]["depth_reader_missed"] += cnt - lc - 1
            lc = cnt; S[cam]["depth_src"] += 1
            with cv: ring.append((st, a["depth"])); cv.notify_all()
    def colour_loop(cam, sb, ring, cv):
        last = sb.seq; lc = None; last_saved = 0
        while wait(sb, last):
            last = sb.seq; r = pread(sb, ["color"])
            if r is None: S[cam]["colour_stuck"] += 1; continue
            sc, cnt, c = r
            if lc is not None and cnt - lc > 1: S[cam]["colour_reader_missed"] += cnt - lc - 1
            lc = cnt; S[cam]["colour_src"] += 1
            if sc - last_saved < min_dt: continue                        # decimation only when hz < camera rate
            dd = None; tw = time.monotonic() + 0.1
            with cv:
                while True:
                    m = [(abs(st - sc), st, x) for st, x in ring]
                    if m and min(m, key=lambda x: x[0])[0] <= 5e6: _, st, dd = min(m, key=lambda x: x[0]); break
                    if time.monotonic() > tw: break
                    cv.wait(0.01)
            if dd is None: S[cam]["unpaired"] += 1; continue
            f = "%s/%d" % (cam, sc); last_saved = sc
            put(os.path.join(d, f + ".jpg"), c["color"], [cv2.IMWRITE_JPEG_QUALITY, 90], True)
            put(os.path.join(d, f + ".png"), dd, [cv2.IMWRITE_PNG_COMPRESSION, 1], False)
            row(dict(cam=cam, stamp_ns=sc, depth_stamp_ns=st, file=f)); S[cam]["saved"] += 1; saved[cam].append(sc)
    def mast_loop(b):
        last = b.seq
        while wait(b, last):
            last = b.seq; r = pread(b, ["color"])
            if r is None: continue
            st, _, a = r; f = "mast/%d" % st
            put(os.path.join(d, f + ".jpg"), a["color"], [cv2.IMWRITE_JPEG_QUALITY, 90], True)
            row(dict(cam="mast", stamp_ns=st, file=f)); S["mast"]["saved"] += 1
    def gt_loop():
        while time.monotonic() < t_end:
            tq = time.monotonic(); s = gt.state()
            row(dict(gt=[s["x"], s["y"], s["yaw"], s["v"], s["w"]], t_mono=(tq + time.monotonic()) / 2, t_sim=s["t"]))
            time.sleep(max(0.0, 1 / 30 - (time.monotonic() - tq)))
    th = [threading.Thread(target=gt_loop, daemon=True)]
    for k, v in info["devices"].items():
        cam = "rs1" if "RS1" in k else "rs2"; os.makedirs(os.path.join(d, cam), exist_ok=True)
        ring = collections.deque(maxlen=30); cv = threading.Condition()
        th += [threading.Thread(target=depth_loop, args=(cam, shm.Block(v["shm"], v["layout"]), ring, cv), daemon=True),
               threading.Thread(target=colour_loop, args=(cam, shm.Block(v["shm"] + "_side", side_layout(p)), ring, cv), daemon=True)]
    os.makedirs(os.path.join(d, "mast"), exist_ok=True)
    th.append(threading.Thread(target=mast_loop, args=(shm.Block(info["mast"]["shm"], info["mast"]["layout"]),), daemon=True))
    for t in th: t.start()
    for t in th: t.join(T + 5)
    t_enc = time.monotonic(); ex.shutdown()
    if Q:      # after the recording: lowest priority, one thread, so the still-running sim keeps its frame age
        try: os.nice(19)
        except OSError: pass
        cv2.setNumThreads(1); ex = ThreadPoolExecutor(int(os.environ.get("SENSORLOG_ENC_THREADS", "1")))
    for path, img, prm, bgr in Q: ex.submit(cv2.imwrite, path, img[..., ::-1].copy() if bgr else img, prm)
    ex.shutdown(); idx.close(); print("sensorlog: encoded %d deferred images in %.1f s" % (len(Q), time.monotonic() - t_enc), flush=True)
    out = {c: dict(S[c]) for c in S}
    for c, v in saved.items():
        g = np.diff(np.sort(np.array(v, float))) * 1e-9
        if len(g): out[c].update(gap_max_s=round(float(g.max()), 3), gap_p99_s=round(float(np.percentile(g, 99)), 3), gaps_over_50ms=int((g > 0.05).sum()))
    json.dump(out, open(os.path.join(d, "stats.json"), "w"), indent=1); print("sensorlog", d, json.dumps(out))

if __name__ == "__main__": main()

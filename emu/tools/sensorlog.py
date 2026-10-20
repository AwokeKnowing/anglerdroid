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
    idx = open(os.path.join(d, "index.jsonl"), "w"); L = threading.Lock()
    S = collections.defaultdict(collections.Counter); saved = collections.defaultdict(list)
    t_end = time.monotonic() + T; min_dt = 0.9e9 / hz
    # Encoding (v3, 2026-10-20).  v2 kept ALL raw frames in RAM until the end (~100 MB/s of recording): the L15 recording
    # was OOM-killed with the box.  Now memory is bounded and encoding streams DURING the recording:
    #  * a byte-capped RAM queue (SENSORLOG_MEM_MB, default 1024) drained by SCHED_IDLE encoder threads (they only get CPU
    #    nobody else wants).  Measured cost (idle robot, 2x60 s ABAB, core decision age p99 / max): none 63-65 / 73-88 ms,
    #    streaming full 74-79 / 97-99, all-fast 74-79 / 79-102: ~+10-15 ms p99 from hyperthread / memory sharing, no
    #    priority inversion.  300 s spin-heavy drive (L17): 7 of 1227 sampled decisions > 100 ms (max 122);
    #  * depth PNG with the SUB filter (lossless, same file format; measured 15.9 vs 26.4 ms/frame for the adaptive filter at
    #    the same size).  While the queue backlog is > 1/4 of the cap, depth is written as an UNCOMPRESSED PNG (2.8 ms, 816 KB)
    #    and recompressed (lossless, in place) after the recording, so the encoder keeps up at ~0.3 core instead of ~1.6;
    #  * a frame that does not fit the queue, or would leave less than SENSORLOG_DISK_MIN_GB (default 6) free on disk, is
    #    DROPPED: counted per camera in stats.json (enc_dropped / disk_dropped) and printed at the first drop.  Never
    #    unbounded, never silent.
    MEM = float(os.environ.get("SENSORLOG_MEM_MB", "1024")) * 2 ** 20; DMIN = float(os.environ.get("SENSORLOG_DISK_MIN_GB", "6")) * 2 ** 30
    NENC = int(os.environ.get("SENSORLOG_ENC_THREADS", "3")); ENC_CPUS = os.environ.get("SENSORLOG_ENC_CPUS")
    LIVE = os.environ.get("SENSORLOG_DEPTH_LIVE", "auto")      # auto (above) | fast (always uncompressed while recording) | full
    PNG_FULL = [cv2.IMWRITE_PNG_COMPRESSION, 1, cv2.IMWRITE_PNG_FILTER, cv2.IMWRITE_PNG_FILTER_SUB]
    PNG_FAST = [cv2.IMWRITE_PNG_COMPRESSION, 0, cv2.IMWRITE_PNG_FILTER, cv2.IMWRITE_PNG_FILTER_NONE]
    QC = threading.Condition(); Q = collections.deque(); qb = [0, 0]; redo = []; done = [False]; warned = set()
    # v4 (2026-10-20): SENSORLOG_ENC = gpu (default when torch+CUDA import) | cpu (v3 above) | null (read + discard, to
    # measure the reader cost).  CPU encoding at SCHED_IDLE still raised core frame age (L17 max 122 ms): the encoder shares
    # hyperthreads / memory bandwidth with the sim even when it only gets idle CPU.  gpu: colour JPEG on the RTX (nvjpeg via
    # torchvision, ~0.35 ms/frame), depth written RAW (.npy, uint16 mm, a memcpy -- no zlib on the CPU).  The raw depth is
    # packed to the usual PNG16 later by emu/tools/sensorlog_pack.py, which only runs while the sim is stopped
    # (emu/tools/offline_gate.sh); readers (slamlog.log.depth_u16) accept either file.  Raw depth is 49 MB/s for both
    # cameras (6 GB per 120 s): the SENSORLOG_DISK_MIN_GB guard drops (counted) once the disk gets that full.
    MODE = os.environ.get("SENSORLOG_ENC", "gpu")
    if MODE == "gpu":
        try:
            import torch; from torchvision.io import encode_jpeg; assert torch.cuda.is_available()
        except Exception as e:
            print("sensorlog: no GPU encoder (%s) -> cpu" % e, file=sys.stderr); MODE = "cpu"
    if MODE == "gpu":
        if "SENSORLOG_ENC_THREADS" not in os.environ: NENC = 1
        encode_jpeg(torch.zeros((3, 8, 8), dtype=torch.uint8, device="cuda"))            # CUDA / nvjpeg init before t0
    def enc(path, img, prm, bgr):
        if MODE == "null": return
        if MODE == "gpu":
            if path.endswith(".npy"): np.save(path, img, allow_pickle=False); return     # raw depth
            t = torch.from_numpy(np.ascontiguousarray(img)).cuda(non_blocking=False).permute(2, 0, 1)
            open(path, "wb").write(encode_jpeg(t if bgr else t.flip(0), quality=prm[1]).cpu().numpy().tobytes()); return
        cv2.imwrite(path, img[..., ::-1].copy() if bgr else img, prm)
    def encoder():
        try: os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))      # this thread only (Linux: 0 = caller tid)
        except (OSError, AttributeError): os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 19)
        if ENC_CPUS: os.sched_setaffinity(0, [int(c) for c in ENC_CPUS.split(",")])
        while True:
            with QC:
                while not Q and not done[0]: QC.wait(0.2)
                if not Q: return
                path, img, prm, bgr = Q.popleft(); fast = MODE == "cpu" and prm is None and not done[0] and (LIVE == "fast" or (LIVE == "auto" and qb[0] > MEM / 4)); qb[0] -= img.nbytes
            enc(path, img, (PNG_FAST if fast else PNG_FULL) if prm is None else prm, bgr)
            with QC:
                S["enc"]["written"] += 1
                if fast: redo.append(path); S["enc"]["depth_fast"] += 1
    cv2.setNumThreads(1); encs = [threading.Thread(target=encoder, daemon=True, name="enc%d" % i) for i in range(NENC)]
    for t in encs: t.start()
    def drop(cam, why):
        with QC: S[cam][why] += 1
        if why not in warned:
            warned.add(why); print("sensorlog: DROPPING frames (%s): %s -- counted in stats.json" % (why, "RAM queue %.0f MB full" % (MEM / 2 ** 20)
                                   if why == "enc_dropped" else "disk free < %.1f GB" % (DMIN / 2 ** 30)), file=sys.stderr, flush=True)
    t_df = [0.0, 1e18]
    def put(path, img, prm, bgr, cam="mast"):
        """prm None = depth PNG (compression chosen by the encoder from the backlog)."""
        if time.monotonic() - t_df[0] > 1.0:
            st = os.statvfs(d); t_df[:] = [time.monotonic(), st.f_bavail * st.f_frsize]
        if t_df[1] < DMIN: drop(cam, "disk_dropped"); return False
        with QC:
            if qb[0] + img.nbytes <= MEM:
                Q.append((path, img, prm, bgr)); qb[0] += img.nbytes; qb[1] = max(qb[1], qb[0]); QC.notify(); return True
        drop(cam, "enc_dropped"); return False
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
            ok1 = put(os.path.join(d, f + ".jpg"), c["color"], [cv2.IMWRITE_JPEG_QUALITY, 90], True, cam)
            ok2 = put(os.path.join(d, f + (".npy" if MODE == "gpu" else ".png")), dd, None, False, cam)
            if not (ok1 and ok2): continue                               # dropped (counted): not indexed, not 'saved'
            row(dict(cam=cam, stamp_ns=sc, depth_stamp_ns=st, file=f)); S[cam]["saved"] += 1; saved[cam].append(sc)
    def mast_loop(b):
        last = b.seq
        while wait(b, last):
            last = b.seq; r = pread(b, ["color"])
            if r is None: continue
            st, _, a = r; f = "mast/%d" % st
            if not put(os.path.join(d, f + ".jpg"), a["color"], [cv2.IMWRITE_JPEG_QUALITY, 90], True): continue
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
    t_end = time.monotonic() + T                                  # after the (GPU) encoder init
    for t in th: t.start()
    for t in th: t.join(T + 5)
    t_enc = time.monotonic(); backlog = len(Q); done[0] = True
    with QC: QC.notify_all()
    for t in encs: t.join()                                       # drain the RAM queue (still SCHED_IDLE, full compression)
    idx.close(); t_rc = time.monotonic()
    if redo:       # recompress the fast (uncompressed) depth PNGs: lossless, same file, nice 19, after the recording
        try: os.nice(19)
        except OSError: pass
        from concurrent.futures import ThreadPoolExecutor as TPE
        def rc(path):
            im = cv2.imread(path, cv2.IMREAD_UNCHANGED); tmp = path + ".tmp.png"; cv2.imwrite(tmp, im, PNG_FULL); os.replace(tmp, path)
        with TPE(NENC) as ex: list(ex.map(rc, redo))
    S["enc"].update(queue_peak_mb=round(qb[1] / 2 ** 20, 1), backlog_at_end=backlog, drain_s=round(t_rc - t_enc, 1), recompress_s=round(time.monotonic() - t_rc, 1))
    print("sensorlog: wrote %d images (%d depth fast, recompressed after in %.1f s), RAM queue peak %.0f MB, backlog at end %d (drained in %.1f s)"
          % (S["enc"]["written"], len(redo), time.monotonic() - t_rc, qb[1] / 2 ** 20, backlog, t_rc - t_enc), flush=True)
    out = {c: dict(S[c]) for c in S}; out.setdefault("enc", {})["mode"] = MODE
    for c, v in saved.items():
        g = np.diff(np.sort(np.array(v, float))) * 1e-9
        if len(g): out[c].update(gap_max_s=round(float(g.max()), 3), gap_p99_s=round(float(np.percentile(g, 99)), 3), gaps_over_50ms=int((g > 0.05).sum()))
    json.dump(out, open(os.path.join(d, "stats.json"), "w"), indent=1); print("sensorlog", d, json.dumps(out))

if __name__ == "__main__": main()

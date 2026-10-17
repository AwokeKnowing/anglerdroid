"""RGB-D/stereo log recorder for the SLAM benchmark (passive: reads the sim RS2 shm + nav/gt state).
usage: rgbd_rec.py OUTROOT START_MARKER STOP_MARKER [hz]   -> OUTROOT/<env>_<n>/ {frames/000123.npz, log.jsonl}
Each frame: ir1, ir2 (y8 848x480, depth-aligned left imager), depth u16 mm (aligned to ir1), color (640x480 rgb, jpg).
log: t (rs2 stamp), seq, odom (nav wheel+gyro, map frame), gt (world x,y,yaw), sim t."""
import sys, os, time, json, urllib.request, numpy as np, cv2
sys.path[:0] = [__import__('os').path.abspath(__import__('os').path.join(__import__('os').path.dirname(__file__), '..', '..')), __import__('os').path.join(__import__('os').path.abspath(__import__('os').path.join(__import__('os').path.dirname(__file__), '..', '..')), 'src')]
import nav  # noqa
from emu import gt
root, start, stop = sys.argv[1:4]; hz = float(sys.argv[4]) if len(sys.argv) > 4 else 5
while not os.path.exists(start): time.sleep(10)
def nav_state():
    try: return json.loads(urllib.request.urlopen("http://localhost:8090/api/state", timeout=1).read())
    except Exception: return None
hw = None; cur = None; n = 0; lastseq = -1
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); _dg = [True]
while not os.path.exists(stop):
    st = nav_state()
    if st is None:
        if cur: cur = None; hw = None
        time.sleep(1); continue
    if cur is None:
        envname = None
        try:
            for l in open(os.path.expanduser("~/.kevin/emu/s2eval_j.log")).read().splitlines()[::-1]:
                if '"env"' in l: break
        except Exception: pass
        n += 1; cur = os.path.join(root, "run%02d" % n); os.makedirs(cur + "/frames", exist_ok=True)
        from nav.hw import Hardware
        hw = Hardware(drive=False); log = open(cur + "/log.jsonl", "a"); json.dump(dict(rs2_intr=hw.rs2_intr, stored_decimation=2), open(cur + "/intr.json", "w"))
    t0 = time.time()
    try:
        seq, stt, depth, color, ir1, ir2 = hw.rs2_snapshot()
        if seq != lastseq and ir1 is not None:
            lastseq = seq; g = gt.state()
            _, cj = cv2.imencode(".jpg", cv2.cvtColor(color, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
            if seq % 200 == 0: _dg[0] = __import__("diskguard").check()
            if not _dg[0]: time.sleep(0.2); continue
            h2 = lambda a: a[::2, ::2]                                 # reduced res (424x240) storage
            _, cj = cv2.imencode(".jpg", cv2.cvtColor(color[::2, ::2], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
            np.savez_compressed(cur + "/frames/%06d.npz" % seq, ir1=h2(ir1), ir2=h2(ir2), depth=h2(depth), color_jpg=cj)
            log.write(json.dumps(dict(t=stt, seq=seq, odom=st.get("odom"), est=st.get("pose"), gt=[g.get("x"), g.get("y"), g.get("yaw")], simt=g.get("t"), label=st.get("label"))) + "\n"); log.flush()
    except Exception as e:
        print("rec err", repr(e), flush=True); time.sleep(0.5)
    time.sleep(max(0.0, 1.0 / hz - (time.time() - t0)))

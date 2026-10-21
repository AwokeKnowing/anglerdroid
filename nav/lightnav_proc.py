"""LightNav goal producer as its OWN process on a fixed-period loop (default 250 ms).  Only observes and sets goals:
  in : kevin_navstate (odom pose), kevin_task_lightnav (instruction), mast camera (sim shm kevin_mast / USB)
  out: kevin_goal_lightnav (goal in the odometry frame), kevin_stat_lightnav (loop health)
The tick never waits on the model: a worker thread talks to lightnav-serve (WebSocket, ~300-400 ms per prediction);
each tick captures one frame+pose into the worker's queue and publishes the newest finished decision, so goals change
only on the 250 ms grid.  Same decision rule as nav/vln.py (receding horizon: chunk row due at the next prediction).
  python -m nav.lightnav_proc --url ws://localhost:8050 [--period 0.25]"""
from __future__ import annotations
import argparse, base64, collections, json, math, os, threading, time
import numpy as np, cv2
import nav  # noqa
from nav import goalio as GI
from nav.vln import MastCam, to_map, commit_row, LOGDIR


class Worker:
    def __init__(self, url):
        self.url = url; self.q = collections.deque(maxlen=8); self.tseq = 0; self.text = None; self.decision = None; self.dseq = 0
        self.pred_ms = 0.0; self.preds = 0; self.errors = 0; self.lock = threading.Lock(); self.ev = threading.Event(); self.log = None
        threading.Thread(target=self.run, daemon=True, name="lightnav-ws").start()

    def jsonl(self, rec):
        try:
            if self.log is None: os.makedirs(LOGDIR, exist_ok=True); self.log = open(os.path.join(LOGDIR, time.strftime("%Y%m%d-%H%M%S") + "_proc.jsonl"), "a")
            self.log.write(json.dumps(rec) + "\n"); self.log.flush()
        except Exception: pass

    def run(self):
        from websockets.sync.client import connect
        while True:
            self.ev.wait(); self.ev.clear(); tseq = self.tseq
            try:
                with connect(self.url, max_size=None) as ws:
                    rq = lambda a, d: (ws.send(json.dumps(dict(action=a, data=d))), json.loads(ws.recv()))[1]["data"]
                    rq("login", dict(clientId="kevin-proc")); rq("reset", {}); seq = 0; tlast = None; period = 0.0
                    while self.tseq == tseq and self.text:
                        if not self.q: time.sleep(0.005); continue
                        jpg, pose, tcap = self.q.popleft(); seq += 1; instr = self.text
                        if self.q:                  # newer frame waiting: buffer only (history stays evenly spaced)
                            rq("next", dict(seq=seq, image=base64.b64encode(jpg.tobytes()).decode(), instruction="")); continue
                        t0 = time.monotonic(); d = rq("next", dict(seq=seq, image=base64.b64encode(jpg.tobytes()).decode(), instruction=instr))
                        ms = (time.monotonic() - t0) * 1e3
                        if d.get("rc", 0) != 0: raise RuntimeError("lightnav rc=%s %s" % (d.get("rc"), d.get("msg")))
                        raw = (d.get("actions") or {}).get("actions") or []
                        wp = np.asarray(raw, float); wp = wp.reshape(-1, wp.shape[-1]) if wp.size else np.zeros((0, 2))
                        stop = bool(d.get("stop")) or len(wp) == 0
                        if tlast: period = 0.8 * period + 0.2 * min(2.0, time.monotonic() - tlast) if period else min(2.0, time.monotonic() - tlast)
                        tlast = time.monotonic(); k = commit_row(len(wp), period or 0.25) if len(wp) else 0
                        wodom = to_map(pose, wp.tolist())
                        if stop: dec = (GI.K_STOP, (0.0, 0.0, 0.0))
                        else:
                            f, l, yw = (list(wp[k]) + [0.0, 0.0, 0.0])[:3]; hd = pose[2] + yw
                            dec = (GI.K_FACE, (0.0, 0.0, hd)) if math.hypot(f, l) < 0.15 else (GI.K_POSE, (wodom[k][0], wodom[k][1], hd))
                            if dec[0] == GI.K_FACE and abs(yw) <= 0.2: dec = None          # tiny turn-in-place: no goal (as vln.py)
                        with self.lock:
                            self.pred_ms = ms; self.preds += 1
                            if dec is not None and self.tseq == tseq: self.decision = (tseq, dec); self.dseq += 1
                        self.jsonl(dict(t=time.time(), seq=seq, age_ms=round((time.monotonic() - tcap) * 1e3, 1), instruction=instr, ms=round(ms, 1),
                                        wp=wp.round(3).tolist(), wp_odom=wodom, stop=stop, commit_row=k, period_s=round(period, 3), dec=None if dec is None else [dec[0], list(dec[1])]))
                        if stop: self.text = None; break
            except Exception as e:
                self.errors += 1; print("lightnav_proc worker error: %r" % e, flush=True); time.sleep(0.5)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--url", default=os.environ.get("KEVIN_VLN_URL", "ws://localhost:8050"))
    ap.add_argument("--period", type=float, default=float(os.environ.get("KEVIN_LIGHTNAV_PERIOD", "0.25"))); a = ap.parse_args()
    ns = GI.open_block("kevin_navstate", GI.L_NAVSTATE, 60.0); task = GI.open_block("kevin_task_lightnav", GI.L_TASK, 60.0)
    goal = __import__("emu.shm", fromlist=["Block"]).Block("kevin_goal_lightnav", GI.L_GOAL, create=True)
    stat = __import__("emu.shm", fromlist=["Block"]).Block("kevin_stat_lightnav", GI.L_STAT, create=True)
    cam = MastCam(None); W = Worker(a.url); last_tseq = -1; gseq = 0; last_dseq = 0; late = collections.deque(maxlen=400)
    ticks = 0; tprev = None; fps = 0.0
    print("lightnav_proc: period %.0f ms url %s" % (a.period * 1e3, a.url), flush=True)
    t_next = time.monotonic(); ppid = os.getppid()
    while True:
        if os.getppid() != ppid: print('lightnav_proc: parent nav gone, exiting', flush=True); os._exit(0)
        t_next += a.period; dt = t_next - time.monotonic()
        if dt > 0: time.sleep(dt)
        else: t_next = time.monotonic() if dt < -a.period else t_next          # overran a full period: re-phase, never burst
        tn = time.monotonic(); late.append(max(0.0, (tn - t_next) * 1e3)); ticks += 1
        _, _, tk = task.read(); ts = int(tk["tseq"][0])
        if ts != last_tseq:                    # new task or cancel: restart the model session, drop queued frames/decisions
            last_tseq = ts; W.q.clear()
            with W.lock: W.decision = None
            W.text = GI.dec_text(tk["text"]) if tk["active"][0] else None; W.tseq = ts
            if W.text: W.ev.set()
        if W.text:
            color = cam.grab()
            if color is not None:
                _, _, s = ns.read(); pose = tuple(float(v) for v in s["odom"])
                ok, jpg = cv2.imencode(".jpg", color[..., ::-1] if color.ndim == 3 else color, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if tprev: fps = 0.8 * fps + 0.2 / max(tn - tprev, 1e-3) if fps else 1.0 / max(tn - tprev, 1e-3)
                tprev = tn; W.q.append((jpg, pose, tn))
        with W.lock: d, dseq = W.decision, W.dseq
        if d is not None and dseq != last_dseq and d[0] == last_tseq:
            last_dseq = dseq; gseq += 1; k, p = d[1]
            goal.write(time.monotonic_ns(), gseq=gseq, source=GI.SOURCES["lightnav"], kind=k, pose=np.array(p, float), vmax=0.0, ttl=0.0, tseq=last_tseq)
        L = np.array(late)
        stat.write(time.monotonic_ns(), ticks=ticks, late_ms_max=float(L.max()), late_ms_p99=float(np.percentile(L, 99)), period_ms=a.period * 1e3,
                   pred_ms=W.pred_ms, preds=W.preds, errors=W.errors, fps=fps)


if __name__ == "__main__":
    main()

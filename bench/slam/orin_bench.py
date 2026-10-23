"""Offline timing replay of slam2 (nav/slam2.py, the live code path: Slam2._step = scan match + chi2/Kalman update +
planning-map integrate + keyframe/lifelong trigger + loop closure + pose graph + map re-render) on whatever box runs it
(i777 3090 vs Kevin's Jetson Orin).  Input: a KEVIN_SLAM_REC / life24_rec.py job stream (pickle records, optionally gzip).

python bench/slam/orin_bench.py JOBS [--life 1] [--out OUT.json] [--core-load] [--realtime]
python bench/slam/orin_bench.py --compare A.npz B.npz          # trajectory agreement of two replays (same input)

Per job: wall ms of _step (cuda-synchronised), split by EXCLUSIVE time into
  match  search() from _step (GPU correlative scan match)      field  Field.fields() (CPU morph+distance transform)
  grid   GridMap.integrate from _step (planning map)            loop   loop-closure search (_try_loop[_life] incl. its search)
  opt    optimize() (GN pose graph)                             remap  _rebuild[_grid] (map re-render after a closure)
  voxel  _voxel() of the scan / free points               fadd   Field.add/remove of a node scan (outside loop/opt)
  other  rest (chi2/Kalman, covariance, trigger bookkeeping, ...)
--core-load: a separate process burning 20 ms CPU per 33.3 ms period (the measured live 30 Hz core budget), CPU only
             (approximation of the core's load; it does not model its GPU/USB/RealSense work).
--async-lc 0|1 [--lag N]: KEVIN_SLAM2_ASYNC_LC / KEVIN_SLAM2_ASYNC_LAG (loop closure on a worker thread; LAG>0 = apply exactly
             N updates after submit, deterministic).  Only main-thread (per-update) time is split/counted; worker time is
             reported as closure latency (lc: updates and ms from submit to apply).
--realtime : feed jobs at their recorded timestamps and report how many a 1-deep latest-wins queue (Slam2.submit) drops."""
from __future__ import annotations
import argparse, gzip, json, math, os, pickle, platform, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))


def load(p):
    J = []; f = (gzip.open if p.endswith(".gz") else open)(p, "rb")
    while True:
        try: J.append(pickle.load(f))
        except (EOFError, pickle.UnpicklingError, OSError, ValueError): return J   # a truncated recording ends cleanly


def core_load(stop_at):
    """30 Hz: ~20 ms of numpy CPU work, then sleep to the next 33.3 ms tick."""
    os.nice(0); a = np.random.rand(256, 256).astype(np.float32); n = 0; t_next = time.perf_counter()
    while time.perf_counter() < stop_at:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 0.020: a = (a @ a.T) * 1e-3 + 0.5
        t_next += 1 / 30.0; d = t_next - time.perf_counter(); n += 1
        if d > 0: time.sleep(d)
        else: t_next = time.perf_counter()


def compare(a, b):
    A, B = np.load(a), np.load(b); n = min(len(A["pose"]), len(B["pose"])); d = A["pose"][:n] - B["pose"][:n]
    d[:, 2] = (d[:, 2] + math.pi) % (2 * math.pi) - math.pi; e = np.hypot(d[:, 0], d[:, 1])
    r = dict(n=n, max_xy_m=round(float(e.max()), 4), mean_xy_m=round(float(e.mean()), 4), p95_xy_m=round(float(np.percentile(e, 95)), 4),
             max_th_deg=round(float(np.degrees(np.abs(d[:, 2]).max())), 3), final_xy_m=round(float(e[-1]), 4),
             nodes=(int(A["nodes"]), int(B["nodes"])), loops=(int(A["loops"]), int(B["loops"])))
    print(json.dumps(r)); return r


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("jobs", nargs="?"); ap.add_argument("--life", default="1")
    ap.add_argument("--out", default=None); ap.add_argument("--core-load", action="store_true"); ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--compare", nargs=2)
    ap.add_argument("--async-lc", default=None); ap.add_argument("--lag", type=int, default=0)
    a = ap.parse_args()
    if a.compare: return compare(*a.compare)
    os.environ["KEVIN_SLAM2_LIFE"] = a.life
    if a.async_lc is not None: os.environ["KEVIN_SLAM2_ASYNC_LC"] = a.async_lc
    os.environ["KEVIN_SLAM2_ASYNC_LAG"] = str(a.lag)
    if os.environ.get("KEVIN_TORCH_SITE"): sys.path.append(os.environ["KEVIN_TORCH_SITE"])
    sys.path[:0] = [REPO]
    J = load(a.jobs)
    if a.limit: J = J[:a.limit]
    import threading, torch, nav.slam2 as S2, nav.slam as S1
    from nav.slam import GridMap
    sync = (lambda: torch.cuda.current_stream().synchronize()) if S2.DEV == "cuda" else (lambda: None)
    # ---- exclusive-time instrumentation
    acc = {}; stack = []; MAIN = threading.get_ident()
    def wrap_fn(owner, name, cat):
        f = getattr(owner, name)
        def g(*args, **kw):
            if threading.get_ident() != MAIN: return f(*args, **kw)      # loop worker thread: not per-update time
            c = cat(stack) if callable(cat) else cat
            t0 = time.perf_counter(); stack.append([c, 0.0])
            try: return f(*args, **kw)
            finally:
                c_, child = stack.pop(); dt = time.perf_counter() - t0
                acc[c_] = acc.get(c_, 0.0) + dt - child
                if stack: stack[-1][1] += dt
        g.__wrapped__ = f; setattr(owner, name, g)
    in_loop = lambda st: any(s[0] == "loop" for s in st)
    wrap_fn(S2, "search", lambda st: "loop" if in_loop(st) else "match")
    wrap_fn(S2.Field, "fields", lambda st: "loop" if in_loop(st) else "field")
    wrap_fn(S2, "optimize", "opt")
    for n_ in ("_try_loop", "_try_loop_life", "_lc_compute", "_lc_apply"): wrap_fn(S2.Slam2, n_, "loop")
    for n_ in ("_rebuild", "_rebuild_grid", "_lc_build_grid", "_lc_swap_grid"): wrap_fn(S2.Slam2, n_, "remap")
    wrap_fn(GridMap, "integrate", lambda st: "remap" if any(s[0] == "remap" for s in st) else "grid")
    wrap_fn(S2, "_voxel", lambda st: st[-1][0] if st else "voxel")
    wrap_fn(S2.Field, "add", lambda st: st[-1][0] if st else "fadd")
    cats = ("match", "field", "grid", "loop", "opt", "remap", "voxel", "fadd")

    # solver warm-up (scipy.sparse import + first factorisation), like Slam2's CUDA warm-up; not counted
    _o = getattr(S2.optimize, "__wrapped__", S2.optimize); _inf = np.array([1e4, 1e4, 1e4])
    _o([(0, 0, 0), (1, 0, 0), (2, 0, 0)], [(0, 1, (1, 0, 0), _inf), (1, 2, (1, 0, 0), _inf), (0, 2, (2, 0, 0), _inf)], robust=[False, False, True])
    s = S2.Slam2(GridMap(), match=True); s._run = False          # no worker thread: we call _step ourselves (= the worker)
    with s._cv: s._job = None
    proc = None
    if a.core_load:
        import multiprocessing as mp
        proc = mp.get_context("fork").Process(target=core_load, args=(time.perf_counter() + 1e6,), daemon=True); proc.start()
    ms, split, pose, loopflag, nodes = [], [], [], [], []
    drops = 0; t_wall0 = time.perf_counter(); t_rec0 = J[0][0]; busy_until = -1.0
    import resource
    for j in J:
        if a.realtime:
            due = t_wall0 + (j[0] - t_rec0); now = time.perf_counter()
            if now < due: time.sleep(due - now)
            elif now - due > 0.2: drops += 1; continue        # a newer job would have replaced it (latest wins)
        acc.clear(); l0 = s.stats.get("loops", 0); o0 = s.stats.get("opts", 0)
        sync(); t0 = time.perf_counter(); s._step(*j[:6]); sync(); dt = (time.perf_counter() - t0) * 1e3
        ms.append(dt); row = [acc.get(c, 0.0) * 1e3 for c in cats]; row.append(max(dt - sum(row), 0.0)); split.append(row)
        loopflag.append((s.stats.get("loops", 0) > l0) or (s.stats.get("opts", 0) > o0)); nodes.append(len(s.kfs))
        pose.append(S2.compose(s.T, j[1]))
        if a.realtime and a.limit == 0 and len(ms) % 2000 == 0: print("..", len(ms), flush=True)
    if proc is not None: proc.terminate(); proc.join()
    if s._lc is not None: s._lc[0].join()                     # a closure still in flight at the end: not applied
    L = np.array(s.lc_log, float).reshape(-1, 4)              # (submit step, apply step, accepted, submit->apply ms)
    lc = dict(async_lc=s.async_lc, lag_mode=s.lc_lag, tries=s.stats.get("loop_tries", 0), busy_skips=s.stats.get("lc_busy", 0))
    if len(L): lc.update(lag_updates_mean=round(float((L[:, 1] - L[:, 0]).mean()), 2), lag_updates_max=int((L[:, 1] - L[:, 0]).max()),
                         lag_ms_mean=round(float(L[:, 3].mean()), 1), lag_ms_max=round(float(L[:, 3].max()), 1))
    ms = np.array(ms); split = np.array(split); lf = np.array(loopflag)
    pct = lambda x, q: round(float(np.percentile(x, q)), 2) if len(x) else None
    r = dict(host=platform.node(), machine=platform.machine(), torch=torch.__version__, dev=S2.DEV,
             gpu=torch.cuda.get_device_name(0) if S2.DEV == "cuda" else None, life=a.life, core_load=a.core_load, realtime=a.realtime,
             jobs=len(J), steps=len(ms), span_s=round(J[-1][0] - J[0][0], 1), drops_realtime=drops,
             step_ms=dict(mean=round(float(ms.mean()), 2), p50=pct(ms, 50), p95=pct(ms, 95), p99=pct(ms, 99), max=round(float(ms.max()), 1)),
             loop_steps=int(lf.sum()), loop_step_ms=dict(mean=round(float(ms[lf].mean()), 1) if lf.any() else None, max=round(float(ms[lf].max()), 1) if lf.any() else None),
             over_200ms=int((ms > 200).sum()), over_100ms=int((ms > 100).sum()),
             split_mean_ms={c: round(float(split[:, i].mean()), 2) for i, c in enumerate(cats + ("other",))},
             split_p95_ms={c: round(float(np.percentile(split[:, i], 95)), 2) for i, c in enumerate(cats + ("other",))},
             split_max_ms={c: round(float(split[:, i].max()), 1) for i, c in enumerate(cats + ("other",))},
             nodes=len(s.kfs), edges=len(s.edges), loops=s.stats.get("loops", 0), opts=s.stats.get("opts", 0), remaps=s.stats.get("remaps", 0),
             lc=lc, accepted=s.stats.get("accepted", 0), matches=s.stats.get("matches", 0),
             maxrss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
             gpu_mem_peak_mb=round(torch.cuda.max_memory_allocated() / 2**20, 1) if S2.DEV == "cuda" else None)
    print(json.dumps(r))
    if a.out:
        json.dump(r, open(a.out, "w"), indent=1)
        np.savez(a.out.replace(".json", ".npz"), ms=ms, split=split, loop=lf, pose=np.array(pose), lc_log=L, nodes=len(s.kfs), loops=s.stats.get("loops", 0),
                 t=np.array([j[0] for j in J[:len(ms)]]) if not a.realtime else np.zeros(0))


if __name__ == "__main__":
    main()

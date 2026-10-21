"""Recorded-state replay for scbench (offline; run on the box or with the sim stopped).
  python -m emu.tools.scbench.recorded <log dir> [--snap T ...] [--json out]
1) per-decision: every nav tick with a non-zero brain command -> the live 2-D obstacle points of the nearest preceding
   slam job (robot frame, moved into this tick's pose), measured twist from vl/vr -> current core decide() vs prototype;
   tallies codes and every disagreement (sd0 bin, |v|, |w|).
2) --snap T: closed loop from the logged state at nav time T (live scan + replayed Slam2 map cells within 1 m, world
   frame, static), intents: spin +-1.57, reverse 0.12, forward 0.15, arc -> both rules via bench.run."""
from __future__ import annotations
import argparse, collections, glob, json, math, sys
import numpy as np
from nav import safety_core as S
from emu.tools.scbench import bench as B, proto_decel as PD
from emu.tools.slamlog.slam2_replay import load, replay


def to_world(P, p):
    c, s = math.cos(p[2]), math.sin(p[2]); return np.c_[p[0] + c * P[:, 0] - s * P[:, 1], p[1] + s * P[:, 0] + c * P[:, 1]]


def load_log(lg):
    R = [json.loads(str(np.load(f, allow_pickle=True)["meta"])) for f in sorted(glob.glob(lg + "/nav/t*.npz"))]
    return R, load(lg + "/slamjobs.pkl")


def per_decision(R, J, max_age=0.5):
    r0 = R[0]["t"]; jt = np.array([j[0] for j in J]); out = collections.Counter(); dis = []; n = 0
    for r in R:
        v_c, w_c = float(r["v"]), float(r["w"])
        if abs(v_c) < 1e-3 and abs(w_c) < 1e-3: continue
        k = int(np.searchsorted(jt, r["t"], "right")) - 1
        if k < 0 or r["t"] - jt[k] > max_age: continue
        j = J[k]; Pw = to_world(np.asarray(j[2], float), j[1]); P = S.to_body(Pw, *r["pose"])
        v_m = 0.5 * (r["vl"] + r["vr"]); w_m = (r["vr"] - r["vl"]) / S.TRACK_M
        a = S.decide(P, B.StubBands(), v_c, w_c, v_m, w_m); b = PD.decide(P, None, v_c, w_c, v_m, w_m)
        pa = abs(a[0] - v_c) < 1e-6 and abs(a[1] - w_c) < 1e-6; pb = abs(b[0] - v_c) < 1e-6 and abs(b[1] - w_c) < 1e-6
        n += 1; out[("core " + a[2], "proto " + b[2])] += 1
        if pa != pb: dis.append(dict(t=round(r["t"] - r0, 2), v=round(v_c, 3), w=round(w_c, 3), sd0=round(a[6], 3), core=a[2], proto=b[2], mode=r["mode"]))
    return n, out, dis


def snapshot(R, J, lg, tq):
    r0 = R[0]["t"]; Jq = [j for j in J if j[0] - r0 <= tq]; s, _, _ = replay(Jq, timing=False); g = s.grid
    rr = min(R, key=lambda r: abs(r["t"] - r0 - tq)); x, y, th = rr["pose"]; occ = g.occupied()
    iy, ix = np.nonzero(occ); W = np.c_[g.origin[0] + (ix + 0.5) * g.res, g.origin[1] + (iy + 0.5) * g.res]
    W = W[np.hypot(W[:, 0] - x, W[:, 1] - y) < 1.0]
    j = Jq[-1]; live = to_world(np.asarray(j[2], float), j[1]); live = live[np.hypot(live[:, 0] - x, live[:, 1] - y) < 1.0]
    return np.array([x, y, th]), W, live


INTENTS = {"spin+1.57": (0.0, 1.57), "spin-1.57": (0.0, -1.57), "spin+0.5": (0.0, 0.5), "reverse0.12": (-0.12, 0.0), "fwd0.15": (0.15, 0.0), "arc0.1,0.8": (0.1, 0.8)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("log"); ap.add_argument("--snap", type=float, nargs="*", default=[]); ap.add_argument("--json")
    ap.add_argument("--no-perdecision", action="store_true"); a = ap.parse_args()
    R, J = load_log(a.log); res = {}
    if not a.no_perdecision:
        n, tally, dis = per_decision(R, J)
        print("%s: %d decisions with a non-zero brain command (live scan points, brain side)" % (a.log, n))
        for (ca, cb), c in tally.most_common(): print("  %-16s %-16s %5d" % (ca, cb, c))
        print("  disagreements (pass vs not): %d" % len(dis))
        for d in dis[:40]: print("   ", d)
        res["per_decision"] = dict(n=n, tally={"%s|%s" % k: v for k, v in tally.items()}, dis=dis)
    for tq in a.snap:
        p, W, live = snapshot(R, J, a.log, tq); sc = {}
        for src, Pw in (("live", live), ("live+map", np.vstack([live, W]))):
            for nm, (v, w) in INTENTS.items():
                sc["%s|%s" % (src, nm)] = dict(P=Pw, policy=B.const(v, w), T=1.5, start=p, expect="", goal=lambda pp, L: L["deepen"] <= 0.002 and L["sd_min"] >= min(L["sd0"], S.MARGIN) - 0.002)
        P0 = S.to_body(np.vstack([live, W]), *p); print("\nsnapshot %s t=%.1f pose %s  sd0 live %.3f  live+map %.3f  (%d live, %d map pts)" % (a.log, tq, np.round(p, 3), S.sd(S.to_body(live, *p)).min() if len(live) else np.inf, S.sd(P0).min(), len(live), len(W)))
        print("  %-22s | %-38s | %-38s" % ("points|intent (1.5 s)", "CURRENT core", "PROTOTYPE"))
        for k in sc:
            row = []
            for rule in ("core", "proto"):
                sc[k]["goal"] = (lambda pp, L: L["deepen"] <= 0.002)
                L = B.run(k, rule, False, scenes=sc)
                moved = float(np.hypot(L["pose"][0] - p[0], L["pose"][1] - p[1])); rot = math.degrees(L["pose"][2] - p[2])
                row.append("%s %-6s mv%4.0fcm rot%+5.0f sd%+5.1f dp%4.1f" % ("SAFE" if L["ok"] else "DEEPER", max(L["codes"], key=L["codes"].get)[:6], 100 * moved, rot, 100 * L["sd_min"], 100 * L["deepen"]))
                res.setdefault("snap_%.1f" % tq, {})[k + "|" + rule] = dict(ok=L["ok"], moved=moved, rot=rot, sd_min=L["sd_min"], deepen=L["deepen"], codes=dict(L["codes"]))
            print("  %-22s | %-38s | %-38s" % (k, row[0], row[1]))
    if a.json: json.dump(res, open(a.json, "w"), indent=1, default=float)

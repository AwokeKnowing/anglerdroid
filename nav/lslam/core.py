"""LifelongSlam: dataflow per frame
    odom increment -> predict (x, P)            [measured odom model]
    sensor gate (floor plane vs twin)           [measured floor sd, chi2(1) at ALPHA]
    CSM against the reference submap, window = 3 sigma of P (clamped to the matcher's range)
    innovation test nu^T (P+R)^-1 nu < chi2(3)  [R measured]  -> fuse (Kalman) or reject
    state machine; journal + submap write only in TRACKING; NIS-window test -> ROLLBACK to the last trusted checkpoint.
All thresholds are chi-square quantiles at model.ALPHA; the window sizes below are structural (not error weights)."""
import math
from collections import deque
import numpy as np
from nav.slam import Matcher
from . import model as M
from .journal import Journal
from .submap import Submap

TRACKING, DEAD_RECKONING, RELOCALIZING, ROLLBACK = "TRACKING", "DEAD_RECKONING", "RELOCALIZING", "ROLLBACK"
SUBMAP_DIST, SUBMAP_ROT = 3.0, math.radians(400)         # submap extent: distance only (a spin adds view, not drift; 90 deg split
                                                         # left new submaps without overlap with the scan -> no reference, 2026-10-19)
WRITE_DIST, WRITE_ROT = 0.04, math.radians(2.5)          # same as kslam: don't re-integrate while static
MIN_REF_SCANS = 10                                       # a submap is a match reference once it has this many scans
W_NIS = 20                                               # NIS window (accepted matches)
CONFIRM = 3                                              # consecutive consistent fixes to leave DR/RELOC
LIN_MAX, ANG_MAX = 0.3, math.radians(8)                  # local CSM search limit (steps 3 cm / 1 deg, as kslam); beyond -> RELOCALIZING
RELOC_LIN, RELOC_ANG = 1.0, math.radians(30)


class LifelongSlam:
    def __init__(self, odom_model, R, x0=(0.0, 0.0, 0.0), P0=None):
        self.om = odom_model; self.R = np.asarray(R, float); self.x = np.array(x0, float)
        self.P = np.diag([1e-6, 1e-6, 1e-6]) if P0 is None else np.asarray(P0, float)
        self.state = TRACKING; self.j = Journal(); self.subs = [Submap(0, self.x)]
        self.last_odom = None; self.last_t = None; self.last_write = None; self.ok_run = 0
        self.win = deque(maxlen=W_NIS); self.cps = deque(maxlen=W_NIS + 1); self.trusted = None
        self.g3, self.gw = M.gate(3), M.gate(3 * W_NIS); self.events = []; self.nis_rej = []; self.flat_dbg = []; self.nis_acc = []; self.stats = dict(acc=0, rej=0, sensor_rej=0, writes=0, rollbacks=0)

    # ---- helpers
    def _event(self, t, kind, **kw): self.events.append(dict(t=t, kind=kind, **kw))
    def _set_state(self, t, s, why=""):
        if s != self.state: self._event(t, "state", frm=self.state, to=s, why=why); self.state = s

    def _ref(self):
        a = self.subs[-1]
        if a.n >= MIN_REF_SCANS: return a
        return self.subs[-2] if len(self.subs) > 1 else (a if a.n else None)

    def _match(self, sub, pts, lin, ang, step, astep):
        g = sub.lf(); guess = M.between(sub.anchor, self.x)
        P = pts[np.random.default_rng(0).choice(len(pts), 1000, replace=False)] if len(pts) > 1000 else pts
        m = Matcher(g, lin=lin, lin_step=step, ang=ang, ang_step=astep, prior=0.0)
        best, bs, med, s0 = m.match(P, tuple(guess))
        return (M.compose(sub.anchor, np.array(best)) if bs > med else None), bs, med

    # ---- per frame
    def step(self, t, odom, pts, sensor_ok=True):
        odom = np.asarray(odom, float)
        if self.last_odom is not None:
            self.x, self.P = self.om.predict(self.x, self.P, M.between(self.last_odom, odom), t - self.last_t)
        self.last_odom, self.last_t = odom, t
        sd = np.sqrt(np.diag(self.P)); sxy = max(sd[0], sd[1])
        if self.state in (TRACKING, DEAD_RECKONING) and (3 * sxy > LIN_MAX or 3 * sd[2] > ANG_MAX):
            self._set_state(t, RELOCALIZING, "3 sigma beyond the local search")
        if not sensor_ok:
            self.stats["sensor_rej"] += 1; self._reject(t, "sensor"); return self.x.copy(), self.state
        if len(pts) < 50: return self.x.copy(), self.state
        if len(self.subs) == 1 and self.subs[0].n < MIN_REF_SCANS:     # bootstrap: first submap, odom only
            self._write(t, pts); return self.x.copy(), self.state
        ref = self._ref()
        if ref is None: return self.x.copy(), self.state
        if self.state == RELOCALIZING: z, bs, med = self._match(ref, pts, RELOC_LIN, RELOC_ANG, 0.05, math.radians(2.0))
        else:
            lin = min(LIN_MAX, max(0.12, 3 * sxy)); ang = min(ANG_MAX, max(math.radians(4), 3 * sd[2]))
            z, bs, med = self._match(ref, pts, lin, ang, 0.03, math.radians(1.0))
        if z is None:                                                   # no score peak: unmapped view, no evidence for/against
            self.stats["no_overlap"] = self.stats.get("no_overlap", 0) + 1
            if self.state == TRACKING and 3 * sxy < 0.12 and 3 * sd[2] < math.radians(4): self._write(t, pts)   # explore: pose within the local match window
            return self.x.copy(), self.state
        v = M.innovation(z, self.x); S = self.P + self.R; q = M.nis(v, S)
        if q > self.g3: self._reject(t, "gate", nis=q); return self.x.copy(), self.state
        self.x, self.P, _ = M.fuse(self.x, self.P, z, self.R); self.stats["acc"] += 1; self.win.append(q); self.nis_acc.append(q); self.ok_run += 1
        if self.state != TRACKING and self.ok_run >= CONFIRM: self._set_state(t, TRACKING, "%d consistent fixes" % CONFIRM)
        if self.state == TRACKING:
            if len(self.win) == W_NIS and sum(self.win) > self.gw: self._rollback(t, odom); return self.x.copy(), self.state
            self._write(t, pts)
        return self.x.copy(), self.state

    def _reject(self, t, why, **kw):
        self.stats["rej"] += 1; self.ok_run = 0; self.stats["rej_" + why] = self.stats.get("rej_" + why, 0) + 1
        if "nis" in kw: self.nis_rej.append(kw["nis"])
        if why == "flat" and len(self.flat_dbg) < 40: self.flat_dbg.append(kw)
        if self.state == TRACKING: self._set_state(t, DEAD_RECKONING, why)
        self._event(t, "reject", why=why, **kw)

    def _write(self, t, pts):
        a = self.subs[-1]
        if self.last_write is not None:
            d = M.between(self.last_write, self.x)
            if math.hypot(d[0], d[1]) < WRITE_DIST and abs(d[2]) < WRITE_ROT: return
        da = M.between(a.anchor, self.x)
        if math.hypot(da[0], da[1]) > SUBMAP_DIST or abs(da[2]) > SUBMAP_ROT:
            a = Submap(len(self.subs), self.x); self.subs.append(a); self._event(t, "submap", id=a.id)
        # checkpoint BEFORE this write; becomes trusted once W_NIS later matches passed the window test
        self.cps.append(dict(t=t, jlen=len(self.j), nsub=len(self.subs), x=self.x.copy(), P=self.P.copy(), odom=self.last_odom.copy()))
        if len(self.cps) == self.cps.maxlen and len(self.win) == W_NIS and sum(self.win) <= self.gw: self.trusted = self.cps[0]
        ps = M.between(a.anchor, self.x); self.j.append(t, a.id, ps, pts); a.integrate(ps, pts); self.last_write = self.x.copy(); self.stats["writes"] += 1

    def _rollback(self, t, odom):
        cp = self.trusted; self.stats["rollbacks"] += 1; self._set_state(t, ROLLBACK, "NIS window %.0f > %.0f" % (sum(self.win), self.gw))
        if cp is None: self.win.clear(); self._set_state(t, DEAD_RECKONING, "no trusted checkpoint"); return
        dropped = self.j.truncate(cp["jlen"]); del self.subs[cp["nsub"]:]
        a = self.subs[-1]; a.rebuild(self.j.of(a.id))
        self.x, self.P = self.om.predict(cp["x"], cp["P"], M.between(cp["odom"], odom), t - cp["t"])
        self.win.clear(); self.cps.clear(); self.trusted = None; self.last_write = None; self.ok_run = 0
        self._event(t, "rollback", to_t=cp["t"], dropped=len(dropped)); self._set_state(t, DEAD_RECKONING, "after rollback")

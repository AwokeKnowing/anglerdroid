import numpy as np, math, json, glob, sys, os
sys.path[:0] = ['.', 'src']
from emu.tools.slamlog.slam2_replay import load, replay
lg, tq = sys.argv[1], float(sys.argv[2]); goal = [float(v) for v in sys.argv[3].split(",")]; NS = int(sys.argv[4]) if len(sys.argv) > 4 else 90
J = load(lg + "/slamjobs.pkl"); R = [json.loads(str(np.load(f, allow_pickle=True)["meta"])) for f in sorted(glob.glob(lg + "/nav/t*.npz"))]
r0 = R[0]["t"]; Jq = [j for j in J if j[0] - r0 <= tq]; s, _, _ = replay(Jq, timing=False)
rr = min(R, key=lambda r: abs(r["t"] - r0 - tq)); x, y, th = rr["pose"]
from nav.plan import Planner, LIMITS as L
from core import integrate_v
P = Planner(s.grid); P.refresh((x, y)); print("pose", np.round([x, y, th], 3), "clearance", round(P.clearance((x, y)), 3))
ok, m = P.set_goal((x, y, th, 0.0, 0.0), np.array(goal), None); print("set_goal", ok, m, "path len", None if P.path is None else len(P.path))
v = w = 0.0; dt = 1 / 30
for k in range(NS):
    r = P.step((x, y, th, v, w))
    a, al = r; v = integrate_v(v, a, dt, L.v_min, L.v_max); w = float(np.clip(w + al * dt, -L.w_max, L.w_max))
    th += w * dt; x += v * math.cos(th) * dt; y += v * math.sin(th) * dt
    if k % 6 == 0: print("%3d a %.2f al %.2f -> v %.3f w %.3f  pose %.3f %.3f %4.0f  mincost %s clr %.3f" % (k, a, al, v, w, x, y, math.degrees(th), P.timing.get("mincost"), P.clearance((x, y))))
if os.environ.get("DIAG"):
    import torch, mppi_triton as _mt, math as _m
    s_ = P.G if hasattr(P.G, "U") else P.G.M if hasattr(P.G, "M") else None
    M = P.M; print("M type", type(M).__name__, "G type", type(P.G).__name__)
    st = torch.as_tensor(np.array([[*P.to_win((x, y)), th, 0.0, 0.0]], np.float32), device=M.dev)
    B, K, H, dt_, L_ = M.B, M.K, M.H, M.dt, M.L
    eps = torch.randn(B, K, H, 2, device=M.dev) * M.sig; U = (M.U.unsqueeze(1) + eps)
    U[..., 0] = U[..., 0].clamp(-L_.a_max, L_.a_max); U[..., 1] = U[..., 1].clamp(-L_.alpha_max, L_.alpha_max)
    cost, deadf, TX = _mt.rollout(U, st, M.goal, M.fp, M.edt, L_, dt_, 6, erev=M.w.edt_rev)
    dead = (deadf > 0)[0].cpu().numpy(); TXn = TX[0].cpu().numpy(); vmin = TXn[:, :, 3].min(1); fin = TXn[:, -1, :3]
    print("K", K, "H", H, "sig", M.sig.tolist() if hasattr(M.sig, "tolist") else M.sig, "dead frac %.3f" % dead.mean(), "dead among never-reversing %.3f (n %d)" % (dead[vmin >= 0].mean() if (vmin >= 0).any() else -1, (vmin >= 0).sum()))
    ok = ~dead; c = cost[0].cpu().numpy()
    if ok.any(): i = np.argmin(np.where(ok, c, 1e9)); print("best feasible cost %.2f final %s dth %.0f" % (c[i], np.round(fin[i], 3), _m.degrees(fin[i, 2] - th)))
    # pure rotations / pure forward probes
    for a_, al_ in ((0.0, 2.0), (0.0, -2.0), (0.5, 0.0), (0.3, 1.0), (0.3, -1.0)):
        Up = torch.zeros(1, 1, H, 2, device=M.dev); Up[..., 0] = a_; Up[..., 1] = al_
        Up[:, :, 15:, :] = 0
        c2, d2, T2 = _mt.rollout(Up.contiguous(), st, M.goal, M.fp, M.edt, L_, dt_, 6, erev=M.w.edt_rev); t2 = T2[0, 0].cpu().numpy()
        print("probe a %.1f al %.1f dead %s cost %.2f final %s" % (a_, al_, bool(d2[0, 0] > 0), float(c2[0, 0]), np.round([t2[-1, 0], t2[-1, 1], _m.degrees(t2[-1, 2] - th)], 3)))
if os.environ.get("DIAG2"):
    import torch, mppi_triton as _mt, math as _m
    M = P.M; st = torch.as_tensor(np.array([[*P.to_win((x, y)), th, 0.0, 0.0]], np.float32), device=M.dev)
    U = torch.zeros(1, M.K, M.H, 2, device=M.dev); U = M._rot_prims(U, st)[:, -M.nrot:].contiguous()
    c, d, T = _mt.rollout(U, st, M.goal, M.fp, M.edt, M.L, M.dt, 6, erev=M.w.edt_rev)
    ctg = M.ahead_pt * M.path_ctg(T[:, :, -1, :2].contiguous(), st)
    U0 = torch.zeros(1, 1, M.H, 2, device=M.dev); c0, d0, T0 = _mt.rollout(U0, st, M.goal, M.fp, M.edt, M.L, M.dt, 6, erev=M.w.edt_rev)
    print("standstill: kernel %.2f ctg %.2f" % (float(c0[0, 0]), float(M.ahead_pt * M.path_ctg(T0[:, :, -1, :2].contiguous(), st)[0, 0])))
    Tn = T[0].cpu().numpy()
    for i in range(M.nrot):
        print("prim %2d dead %d kernel %.2f ctg %.2f final %s vmax %.2f" % (i, int(d[0, i] > 0), float(c[0, i]), float(ctg[0, i]), np.round([Tn[i, -1, 0], Tn[i, -1, 1], _m.degrees(Tn[i, -1, 2] - th)], 2), Tn[i, :, 3].max()))
if os.environ.get("DIAG3"):
    pp = np.asarray(P.path); d = np.r_[0, np.cumsum(np.hypot(*np.diff(pp, axis=0).T))]
    for k in range(0, len(pp), max(1, len(pp) // 12)):
        b = math.degrees(math.atan2(pp[k, 1] - rr["pose"][1], pp[k, 0] - rr["pose"][0]) - rr["pose"][2]); print("path s %.2f  %s  bearing from robot %+.0f deg" % (d[k], np.round(pp[k], 2), (b + 180) % 360 - 180))

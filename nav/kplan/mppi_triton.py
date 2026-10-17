"""Fused Triton MPPI rollout for kplan.planner.MPPI (one GPU thread per sample, all H steps in registers).
Same dynamics/costs as MPPI.plan_ref; replaces ~2000 small torch kernels per plan with 2 kernel launches."""
import math, torch, triton, triton.language as tl
import sys as _s, os as _o; _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), '..', '..', 'src') in _s.path or _s.path.append(_o.path.join(_o.path.dirname(_o.path.abspath(__file__)), '..', '..', 'src'))   # twin_geometry (single source of the footprint radii)
import twin_geometry as _TG
R_ROTATE = tl.constexpr(_TG.R_ROTATE); R_PASS = tl.constexpr(_TG.R_PASS)

@triton.jit
def _atan2(y, x):
    ax = tl.abs(x); ay = tl.abs(y); mx = tl.maximum(ax, ay); mn = tl.minimum(ax, ay)
    a = mn / tl.maximum(mx, 1e-9); s = a*a
    r = ((-0.0464964749*s + 0.15931422)*s - 0.327622764)*s*a + a
    r = tl.where(ay > ax, 1.57079637 - r, r); r = tl.where(x < 0, 3.14159274 - r, r)
    return tl.where(y < 0, -r, r)

@triton.jit
def _clear(x, y, th, fp, P: tl.constexpr, edt, eoff, Hm, Wm, mask):
    c = tl.cos(th); sn = tl.sin(th)
    m = tl.full(x.shape, 1e9, tl.float32)
    for p in range(P):
        fx = tl.load(fp + 2*p); fy = tl.load(fp + 2*p + 1)
        px = x + c*fx - sn*fy; py = y + sn*fx + c*fy
        ix = tl.minimum(tl.maximum((px*100.0).to(tl.int32), 0), Wm-1)
        iy = tl.minimum(tl.maximum((py*100.0).to(tl.int32), 0), Hm-1)
        e = tl.load(edt + eoff + iy.to(tl.int64)*Wm + ix, mask=mask, other=1.0).to(tl.float32)
        m = tl.minimum(m, e)
    return m

@triton.jit
def _vcap(d, v_max, a_brake, t_react, d_safe, d_full, v_near, d_near):
    xx = tl.maximum(d - d_safe, 0.0)
    hard = a_brake*(-t_react + tl.sqrt(t_react*t_react + 2*xx/a_brake))
    ramp = v_near + (v_max - v_near)*tl.minimum(tl.maximum((d - d_near)/(d_full - d_near), 0.0), 1.0)
    return tl.minimum(tl.minimum(hard, ramp), v_max)

@triton.jit
def _rollout(U, state, goal, fp, edt, erev, estride, Hm, Wm,
             cost_o, dead_o, tx_o, K, dt,
             v_max, v_min, a_brake, t_react, d_safe, d_full, v_near, d_near,
             a_lat, w_max, v_wheel, track,
             H: tl.constexpr, P: tl.constexpr, TXC: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(0); k = tl.program_id(1)*BLOCK + tl.arange(0, BLOCK); mask = k < K
    zero = tl.zeros([BLOCK], tl.float32)
    x = zero + tl.load(state + b*5); y = zero + tl.load(state + b*5 + 1); th = zero + tl.load(state + b*5 + 2)
    v = zero + tl.load(state + b*5 + 3); w = zero + tl.load(state + b*5 + 4)
    gx = tl.load(goal + b*3); gy = tl.load(goal + b*3 + 1); gth = tl.load(goal + b*3 + 2)
    eoff = b.to(tl.int64)*estride
    cost = zero; dead = zero; usq = zero
    clr = _clear(x, y, th, fp, P, edt, eoff, Hm, Wm, mask)
    ub = U + (b*K + k).to(tl.int64)*(H*2)
    txb = tx_o + (b*K + k).to(tl.int64)*(H*TXC)
    for t in range(H):
        a = tl.load(ub + 2*t, mask=mask, other=0.0); al = tl.load(ub + 2*t + 1, mask=mask, other=0.0)
        usq += a*a + al*al
        vc = tl.minimum(_vcap(clr, v_max, a_brake, t_react, d_safe, d_full, v_near, d_near),   # + approach scaling (see mppi_cuda)
                        tl.maximum(0.08, v_max*tl.sqrt((x-gx)*(x-gx) + (y-gy)*(y-gy))/0.6))
        v0 = v
        v = tl.minimum(tl.maximum(v + a*dt, v_min), v_max)
        lim = tl.maximum(vc, tl.abs(v0) - a_brake*dt)
        v = tl.maximum(tl.minimum(v, lim), -tl.minimum(lim, -v_min))
        av = tl.abs(v)
        wl = tl.minimum(w_max, a_lat/tl.maximum(av, 1e-3))
        wl = tl.minimum(wl, tl.maximum((v_wheel - av)/(track/2), 0.0))
        w = tl.maximum(tl.minimum(w + al*dt, wl), -wl)
        th = th + w*dt; x = x + v*tl.cos(th)*dt; y = y + v*tl.sin(th)*dt
        tl.store(txb + t*TXC, x, mask=mask); tl.store(txb + t*TXC + 1, y, mask=mask); tl.store(txb + t*TXC + 2, th, mask=mask)
        if TXC == 6:
            tl.store(txb + t*TXC + 3, v, mask=mask); tl.store(txb + t*TXC + 4, w, mask=mask); tl.store(txb + t*TXC + 5, vc, mask=mask)
        clr = tl.where(v < 0, _clear(x, y, th, fp, P, erev, eoff, Hm, Wm, mask), _clear(x, y, th, fp, P, edt, eoff, Hm, Wm, mask))
        dead = tl.where(clr < 0.012, 1.0, dead)
        ix = tl.minimum(tl.maximum((x*100.0).to(tl.int32), 0), Wm-1)
        iy = tl.minimum(tl.maximum((y*100.0).to(tl.int32), 0), Hm-1)
        ax = tl.load(edt + eoff + iy.to(tl.int64)*Wm + ix, mask=mask, other=1.0).to(tl.float32)
        cost += dead*50 + 20*tl.maximum(0.06 - clr, 0.0)
        cost += dt*(tl.exp(-clr/0.15) + 2.0*tl.minimum(tl.maximum(R_ROTATE - ax, 0.0)/(R_ROTATE - R_PASS), 1.0))
        dx = x - gx; dy = y - gy
        near = tl.exp(-tl.sqrt(dx*dx + dy*dy)/0.25)
        cr = tl.minimum(tl.maximum((clr - 0.05)/0.20, 0.0), 1.0); dg = tl.sqrt(dx*dx + dy*dy)
        cost += 0.1*dt*20*near*(1 - tl.cos(th - gth))*2 + (0.5 + 4.0*near + 4.0*cr*(1 - near))*dt*tl.where(v < 0, 1.0, 0.0)*av/0.2
        hd = th - _atan2(-dy, -dx)
        cost += tl.where(dg > 0.35, 1.5*dt*cr*(1 - near)*(1 - tl.cos(hd)), 0.0)
    dx = x - gx; dy = y - gy
    nT = tl.exp(-tl.sqrt(dx*dx + dy*dy)/0.15)
    cost += 3*nT*(1 - tl.cos(th - gth)) + 2*nT*(tl.abs(v) + tl.abs(w)*0.3) + 0.002*usq
    tl.store(cost_o + b*K + k, cost, mask=mask); tl.store(dead_o + b*K + k, dead, mask=mask)

def rollout(U, state, goal, fp, edt, L, dt, txc=3, cost=None, dead=None, tx=None, erev=None):
    """U (B,K,H,2) f32 contiguous; state (B,5); goal (B,3); edt (H,W) or (B,H,W) f16.
    Returns cost (B,K) [excl. 10*ctg], dead (B,K) f32 0/1, tx (B,K,H,txc)."""
    B, K, H, _ = U.shape
    if cost is None: cost = torch.empty(B, K, device=U.device)
    if dead is None: dead = torch.empty(B, K, device=U.device)
    if tx is None: tx = torch.empty(B, K, H, txc, device=U.device)
    if erev is None: erev = edt                      # reverse clearance (see mppi_cuda.rollout)
    Hm, Wm = edt.shape[-2:]; est = 0 if edt.dim() == 2 else Hm*Wm
    BLOCK = 64 if K >= 64 else 16
    _rollout[(B, triton.cdiv(K, BLOCK))](U, state.contiguous(), goal.contiguous(), fp, edt, erev.contiguous(), est, Hm, Wm, cost, dead, tx, K, float(dt),
        float(L.v_max), float(L.v_min), float(L.a_brake), float(L.t_react), float(L.d_safe), float(L.d_full), float(L.v_near), float(L.d_near),
        float(L.a_lat), float(L.w_max), float(L.v_wheel), float(L.track), H=H, P=fp.shape[0], TXC=txc, BLOCK=BLOCK, num_warps=2)
    return cost, dead, tx

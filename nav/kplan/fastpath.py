"""Fast new-goal planner (no per-goal cost-to-go field). Switch: KPLAN_GLOBAL=field|fast (default field).
Per map update (once, shared by all goals): fine EDT + 10 cm cost/blocked grid (World) -> CPU copies.
Per goal (~1-4 ms CPU):  1) weighted 8-conn A* on the 10 cm grid (same edge cost as core.cost_to_go, so
same clearance preference), C via ctypes;  2) cost-aware string pulling (any-angle shortcut only if free
and not costlier than the grid path -> keeps preferred clearance);  3) elastic-band smoothing at 5 cm with
EDT push (never below R240 circle clearance);  4) GPU upload of a dense path (arclength s).
MPPI terminal cost = remaining path length from the projected point + lateral offset penalty,
projected only inside a progress window (can't 'see' through walls to a later path segment)."""
import ctypes, os, math, time, subprocess, numpy as np, torch, torch.nn.functional as F
from planner import MPPI, World, _mt, bilinear
_here=os.path.dirname(os.path.abspath(__file__)); _so=os.path.join(_here,'libfastpath.so'); _c=os.path.join(_here,'fastpath_astar.c')
if not os.path.exists(_so) or os.path.getmtime(_so)<os.path.getmtime(_c):
    import platform; subprocess.check_call(['gcc','-O3','-mcpu=native' if platform.machine()=='aarch64' else '-march=native','-shared','-fPIC','-o',_so,_c])
_L=ctypes.CDLL(_so)
_so2=_so.replace('.so','_bg.so')       # 2nd instance (own static buffers) for the background refine thread
if not os.path.exists(_so2) or os.path.getmtime(_so2)<os.path.getmtime(_so): import shutil; shutil.copy(_so,_so2)
_L2=ctypes.CDLL(_so2)
_P=ctypes.c_void_p
_L.astar.argtypes=[_P,_P,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_float,_P,ctypes.c_int,_P]
for _l in (_L,_L2):
    _l.astar.argtypes=[_P,_P,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_float,_P,ctypes.c_int,_P]
    _l.stringpull.argtypes=[_P,_P,ctypes.c_int,_P,ctypes.c_int,ctypes.c_float,ctypes.c_int,ctypes.c_int,_P]
    _l.astar_m.argtypes=_l.astar.argtypes+[_P,ctypes.c_int,ctypes.c_int]
    _l.astar_e.argtypes=[_P,_P,_P,_P]+_l.astar.argtypes[2:]
    _l.band.argtypes=[_P,ctypes.c_int,_P,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_float,ctypes.c_float,ctypes.c_float,ctypes.c_float]
R_PLAN=0.24   # tight circle footprint for planning (axle-centred)

class MapCPU:
    """Per map update. Cheap: reuses World's EDT / coarse grids (already needed by MPPI)."""
    def __init__(s, world, rplan=None, share=None):
        t=time.perf_counter()
        s.w=world; s.cres=world.cres
        s.rplan=float(os.environ.get('KPLAN_RPLAN',R_PLAN)) if rplan is None else rplan; s._relaxed=None
        s.cost=share.cost if share else np.ascontiguousarray(world.ccost.float().cpu().numpy())
        # blocked = no point in the 10 cm cell has axle clearance >= R_PLAN (240 mm: half-width 210 + d_safe 30, where the
        # proximity speed rule reaches 0). World.cblocked uses 200 mm, which lets the planner route through 0.45 m doors
        # the robot can never drive (the cross-map failure). KPLAN_RPLAN overrides.
        s.blk=np.ascontiguousarray((world.cmax<s.rplan).cpu().numpy().astype(np.uint8))
        s.edt=share.edt if share else world.edt.cpu().numpy()     # float16 metres, 1 cm
        s.ecm=share.ecm if share else np.ascontiguousarray((world.edt.float()*100).round().clamp(0,255).to(torch.uint8).cpu().numpy())
        s.R,s.C=s.cost.shape; s.buf2=np.empty(2*s.R*s.C//4,np.int32); s.keep2=np.empty(len(s.buf2)//2,np.int32); s.buf=np.empty(2*s.R*s.C//4,np.int32); s.keep=np.empty(len(s.buf)//2,np.int32)
        # hierarchical level (per map): 4x coarser (40 cm) grid, optimistic: free if any child free; cost = mean of free children
        f=s.f=4; Rc,Cc=s.R//f,s.C//f; s.Rc,s.Cc=Rc,Cc
        cb=s.blk[:Rc*f,:Cc*f].reshape(Rc,f,Cc,f); cc=s.cost[:Rc*f,:Cc*f].reshape(Rc,f,Cc,f)
        free=(cb==0); nf=free.sum((1,3))
        s.cblk=np.ascontiguousarray((nf==0).astype(np.uint8))
        s.ccost=np.ascontiguousarray(np.where(nf>0,(cc*free).sum((1,3))/np.maximum(nf,1),1.0).astype(np.float32))
        # coarse edges exist only where two free fine cells touch across the shared block boundary (no tunnelling through thin walls)
        fb=(s.blk[:Rc*f,:Cc*f]==0)
        s.cex=np.ascontiguousarray((fb[:,f-1::f][:, :Cc-1]&fb[:,f::f][:, :Cc-1]).reshape(Rc,f,Cc-1).any(1).astype(np.uint8))
        s.cex=np.ascontiguousarray(np.c_[s.cex,np.zeros((Rc,1),np.uint8)])
        s.cey=np.ascontiguousarray((fb[f-1::f][:Rc-1]&fb[f::f][:Rc-1]).reshape(Rc-1,Cc,f).any(2).astype(np.uint8))
        s.cey=np.ascontiguousarray(np.r_[s.cey,np.zeros((1,Cc),np.uint8)])
        s.cbuf=np.empty(2*Rc*Cc,np.int32); s.allow=np.zeros((Rc,Cc),np.uint8); s.cbuf2=s.cbuf.copy(); s.allow2=s.allow.copy()
        if share is None and os.environ.get('KPLAN_ESC_PREBUILD','1')=='1':   # escape-retry grids built with the map, not on the goal path
            s._lz={r:MapCPU(world,rplan=r,share=s) for r in ESC_LADDER if r<s.rplan-1e-4}
        s.t_ms=(time.perf_counter()-t)*1e3
    def clr(s,P):
        ix=np.clip((P[:,0]*100).astype(int),0,s.edt.shape[1]-1); iy=np.clip((P[:,1]*100).astype(int),0,s.edt.shape[0]-1)
        return s.edt[iy,ix].astype(np.float32)

def _resample(P,ds):
    d=np.r_[0,np.cumsum(np.hypot(*np.diff(P,axis=0).T))]; n=max(2,int(math.ceil(d[-1]/ds))+1)
    q=np.linspace(0,d[-1],n); return np.c_[np.interp(q,d,P[:,0]),np.interp(q,d,P[:,1])]

def smooth(m,P,iters=40,ds=0.05,pref=0.40,lib=None):
    """C elastic band (fast path)."""
    P=np.ascontiguousarray(_resample(P,ds),dtype=np.float32)
    (lib or _L).band(P.ctypes.data,len(P),m.ecm.ctypes.data,m.ecm.shape[0],m.ecm.shape[1],iters,pref,R_PLAN,0.35,0.25)
    return P.astype(np.float64)
def smooth_np(m,P,iters=40,ds=0.05,pref=0.40):
    """(endpoints fixed) elastic band: Laplacian (curvature) + EDT-gradient push toward pref clearance; reject moves below R_PLAN."""
    P=_resample(P,ds).astype(np.float32); E=m.edt; H,W=E.shape
    for _ in range(iters):
        Q=P.copy(); Q[1:-1]+=0.35*(P[:-2]+P[2:]-2*P[1:-1])
        ix=np.clip((Q[:,0]*100).astype(int),2,W-3); iy=np.clip((Q[:,1]*100).astype(int),2,H-3)
        e=E[iy,ix].astype(np.float32)
        gx=(E[iy,ix+2].astype(np.float32)-E[iy,ix-2])/0.04; gy=(E[iy+2,ix].astype(np.float32)-E[iy-2,ix])/0.04
        push=np.clip(pref-e,0,None)*0.25
        Q[1:-1,0]+=(push*gx)[1:-1]; Q[1:-1,1]+=(push*gy)[1:-1]
        e0=m.clr(P); e1=m.clr(Q); ok=(e1>=np.minimum(e0,R_PLAN))
        P[ok]=Q[ok]
    return P

def _plan_path(m, start_xy, goal_xy, eps=2.0, sp_tol=0.0, win=120, smooth_iters=20, smooth_m=5.0, front=None, lib=None, hier=True, fine_m=7.0, gallop=1):
    lib=lib or _L; two=lib is not _L; buf,keep=(m.buf2,m.keep2) if two else (m.buf,m.keep)
    cbuf,Aw=(m.cbuf2,m.allow2) if two else (m.cbuf,m.allow)
    """Returns dict(xy (N,2) 5 cm, s, timings) or None. front: optional callable returning a coarse polyline
    (e.g. James's pathfinder.find_path) used instead of A*."""
    t0=time.perf_counter(); cr=m.cres
    sr,sc=int(start_xy[1]/cr),int(start_xy[0]/cr); gr,gc=int(goal_xy[1]/cr),int(goal_xy[0]/cr)
    ne=ctypes.c_int(0); allow=None; tail=None
    if hier:   # coarse A* (40 cm) -> corridor (+-2 coarse cells) -> fine A* restricted to it
        fc=m.f; nc=lib.astar_e(m.ccost.ctypes.data,m.cblk.ctypes.data,m.cex.ctypes.data,m.cey.ctypes.data,m.Rc,m.Cc,min(sr//fc,m.Rc-1),min(sc//fc,m.Cc-1),min(gr//fc,m.Rc-1),min(gc//fc,m.Cc-1),eps,cbuf.ctypes.data,len(cbuf)//2,ctypes.byref(ne))
        tc=time.perf_counter(); m.dbg=dict(coarse_ms=(tc-t0)*1e3,coarse_exp=ne.value,nc=nc)
        tail=None; fgr,fgc=gr,gc
        if nc>0:
            cp=cbuf[:2*nc].reshape(-1,2); A=Aw
            kf=int(fine_m/(m.f*cr))
            if fine_m and nc>kf+3:   # only the first fine_m metres at 10 cm now; coarse tail (refined in background)
                br,bc=cp[kf]; blkc=m.blk[br*fc:br*fc+fc,bc*fc:bc*fc+fc]; cst=np.where(blkc>0,np.inf,m.cost[br*fc:br*fc+fc,bc*fc:bc*fc+fc])
                if np.isfinite(cst).any():
                    a_=int(np.argmin(cst)); fgr,fgc=br*fc+a_//fc,bc*fc+a_%fc
                    tail=np.c_[(cp[kf+1:,1]+.5)*fc*cr,(cp[kf+1:,0]+.5)*fc*cr]; cp=cp[:kf+1]
            for w in (2,5):   # corridor half-width in coarse cells; widen once if the optimistic abstraction lied
                A[:]=0; o=np.arange(-w,w+1)
                rr=np.clip(cp[:,0,None,None]+o[None,:,None],0,m.Rc-1); cc_=np.clip(cp[:,1,None,None]+o[None,None,:],0,m.Cc-1)
                A[np.broadcast_to(rr,(len(cp),len(o),len(o))),np.broadcast_to(cc_,(len(cp),len(o),len(o)))]=1
                n=lib.astar_m(m.cost.ctypes.data,m.blk.ctypes.data,m.R,m.C,sr,sc,fgr,fgc,eps,buf.ctypes.data,len(buf)//2,ctypes.byref(ne),A.ctypes.data,m.f,m.Cc)
                m.dbg['w%d'%w]=(n,ne.value,(time.perf_counter()-tc)*1e3)
                if n>0: allow=A; break
    if allow is None:
      tail=None
      n=lib.astar(m.cost.ctypes.data,m.blk.ctypes.data,m.R,m.C,sr,sc,gr,gc,eps,buf.ctypes.data,len(buf)//2,ctypes.byref(ne))
    t1=time.perf_counter()
    if n<0: return None
    p=buf[:2*n].copy()
    k=lib.stringpull(m.cost.ctypes.data,m.blk.ctypes.data,m.C,p.ctypes.data,n,sp_tol,win,gallop,keep.ctypes.data)
    idx=keep[:k].copy(); rc=p.reshape(-1,2)[idx]
    V=np.c_[(rc[:,1]+.5)*cr,(rc[:,0]+.5)*cr]; V[0]=start_xy
    if tail is not None: V=np.r_[V,tail,np.asarray(goal_xy,float)[None]]
    V[-1]=goal_xy
    t2=time.perf_counter()
    P=_resample(V,0.05)
    if smooth_iters:   # smooth only the near part now (MPPI looks <=4 m ahead); far part is re-smoothed on later replans
        n=min(len(P),int(smooth_m/0.05)+1); Q=smooth(m,P[:n],smooth_iters,lib=lib)
        if n<len(P): Q[-1]=P[n-1]
        P=np.r_[Q,P[n:]] if n<len(P) else Q
    P[0]=start_xy; P[-1]=goal_xy
    t3=time.perf_counter()
    s=np.r_[0,np.cumsum(np.hypot(*np.diff(P,axis=0).T))]
    return dict(xy=P,s=s,corners=V,partial=tail is not None,expanded=ne.value,ms=dict(astar=(t1-t0)*1e3,pull=(t2-t1)*1e3,smooth=(t3-t2)*1e3,total=(t3-t0)*1e3))

ESC_LADDER=(0.200,0.15,0.10)   # footprint retries for escape (start/goal in a spot tighter than R_PLAN)
def _loose(m,r):
    if not hasattr(m,'_lz'): m._lz={}
    if r not in m._lz: m._lz[r]=MapCPU(m.w,rplan=r,share=m)
    return m._lz[r]
def _cat(*parts):
    P=np.concatenate([p for p in parts if len(p)]); keep=np.r_[True,np.hypot(*np.diff(P,axis=0).T)>1e-6]; P=P[keep]
    return P
def plan_path(m, start_xy, goal_xy, **kw):
    """Plan at R_PLAN (240 mm). If the start or goal sits where that footprint is blocked (trap / nook / dead end),
    escape-aware retry: plan on progressively looser grids (200, 150, 100 mm), keep the loose path only from the
    start to where it first becomes R_PLAN-free (and from where it last leaves R_PLAN-free to the goal), and
    replan the middle on the strict grid. Never lets the loose footprint shortcut through tight gaps mid-route."""
    p=_plan_path(m,start_xy,goal_xy,**kw)
    if p is not None: return p
    t0=time.perf_counter()
    for r in ESC_LADDER:
        if r>=m.rplan-1e-4: continue
        lz=_loose(m,r); q=_plan_path(lz,start_xy,goal_xy,**kw)
        if q is None: continue
        P=q['xy']; cr=m.cres
        fr=m.blk[np.clip((P[:,1]/cr).astype(int),0,m.R-1),np.clip((P[:,0]/cr).astype(int),0,m.C-1)]==0
        if fr.any():
            i=int(np.argmax(fr)); j=len(P)-1-int(np.argmax(fr[::-1]))
            if j>i+2:
                mid=_plan_path(m,P[i],P[j],**kw)
                if mid is not None: P=_cat(P[:i],mid['xy'],P[j+1:])
        s=np.r_[0,np.cumsum(np.hypot(*np.diff(P,axis=0).T))]
        q.update(xy=P,s=s,relaxed=r,escape_ms=(time.perf_counter()-t0)*1e3)
        return q
    return None

def smooth_near(m, p, smooth_iters=20, smooth_m=5.0, lib=None):
    """Smooth the first smooth_m metres of an (unsmoothed) plan_path result; returns a new dict."""
    P=p['xy'].copy(); n=min(len(P),int(smooth_m/0.05)+1); Q=smooth(m,P[:n],smooth_iters,lib=lib)
    if n<len(P): Q[-1]=P[n-1]
    P=np.r_[Q,P[n:]] if n<len(P) else Q; P[0]=p['xy'][0]; P[-1]=p['xy'][-1]
    q=dict(p); q['xy']=P; q['s']=np.r_[0,np.cumsum(np.hypot(*np.diff(P,axis=0).T))]; return q

import threading
class AnytimePlanner:
    """New goal -> fast eps-weighted path immediately (<~2 ms), then a background thread computes the eps=1
    (optimal on the 10 cm cost grid) path + full smoothing and hands it over (ctypes releases the GIL)."""
    def __init__(s,m,eps_fast=2.0): s.m=m; s.eps=eps_fast; s.refined=None; s.th=None; s.t_refine_ms=None
    def new_goal(s,start,goal):
        p=plan_path(s.m,start,goal,eps=s.eps); s.refined=None
        def bg():
            t=time.perf_counter(); r=plan_path(s.m,start,goal,eps=1.0,smooth_m=30.0,lib=_L2,fine_m=0); s.t_refine_ms=(time.perf_counter()-t)*1e3; s.refined=r
        s.th=threading.Thread(target=bg,daemon=True); s.th.start(); return p
    def poll(s):
        r=s.refined; s.refined=None; return r

class PathMPPI(MPPI):
    """MPPI whose terminal cost is path-progress cost-to-go (no grid field). Same rollout kernel."""
    NP=4096; WN=96
    def __init__(s,*a,**k):
        super().__init__(*a,**k); B=s.B
        s.pxy=torch.zeros(B,s.NP,2,device=s.dev); s.ps=torch.zeros(B,s.NP,device=s.dev); s.plen=torch.zeros(B,device=s.dev)
        # progress cost is COST-weighted arclength (same per-cell cost World.ccost the field integrates: ~1 in the open,
        # up to 6.5 in tight spots), so leaving a trap/dead end is rewarded as strongly as with the field. KPLAN_CTGW=len: plain metres.
        s.pc=torch.zeros(B,s.NP,device=s.dev); s.pclen=torch.zeros(B,device=s.dev)
        # cells World blocks (cmax<200 mm) cost KPLAN_ESCW (100) per metre: an escape segment out of a dead end / nook gets the
        # same steep pull the field's 1e3 blocked-cell value gives (without it MPPI sits still: creeping 10 cm out at
        # v_cap ~0.08 m/s never pays for the wall-proximity cost of the samples that move).
        if os.environ.get('KPLAN_CTGW','cost')=='cost':
            w_=a[0]; s.ccost_np=torch.where(w_.cblocked,torch.full_like(w_.ccost.float(),float(os.environ.get('KPLAN_ESCW',100))),w_.ccost.float()).cpu().numpy()
        else: s.ccost_np=None
        s.sprog=torch.zeros(B,device=s.dev)
        s.wlat=float(os.environ.get('KPLAN_WLAT',1.0)); s.ahead_pt=float(os.environ.get('KPLAN_WPROG',10.0))
        sa=os.environ.get('KPLAN_SIG','0.9,1.5')   # (accel, yaw-accel) noise; tuned: 0.9,1.5 + wlat 1 = least wiggle at 5/5 success
        if sa: s.sig=torch.tensor([float(x) for x in sa.split(',')],device=s.dev)
        if os.environ.get('KPLAN_LAM'): s.lam=float(os.environ['KPLAN_LAM'])
    def _cumc(s, xy, ss, b=0):
        if s.ccost_np is None or len(xy)<2: return np.asarray(ss,float)
        C=s.ccost_np if s.ccost_np.ndim==2 else s.ccost_np[min(b,len(s.ccost_np)-1)]; cr=s.w.cres
        mid=0.5*(xy[1:]+xy[:-1]); c=C[np.clip((mid[:,1]/cr).astype(int),0,C.shape[0]-1),np.clip((mid[:,0]/cr).astype(int),0,C.shape[1]-1)]
        return np.r_[0,np.cumsum(np.diff(ss)*c)]
    def set_paths(s, goals, paths, keep_progress=False):
        g=torch.as_tensor(goals,dtype=torch.float32,device=s.dev)
        if hasattr(s,'goal'): s.goal.copy_(g)
        else: s.goal=g.clone()
        if not hasattr(s,'_hbuf'):
            s._hbuf=[(torch.empty((s.B,s.NP,2),pin_memory=True),torch.empty((s.B,s.NP),pin_memory=True),torch.empty(s.B,pin_memory=True),torch.cuda.Event(),torch.empty((s.B,s.NP),pin_memory=True),torch.empty(s.B,pin_memory=True)) for _ in range(2)]; s._hi=0
        s._hi^=1; s._hX,s._hS,s._hL,ev,s._hC,s._hLc=s._hbuf[s._hi]; ev.synchronize()   # this host buffer's previous copy is done
        X,S,Lh,Cc,Lc=s._hX.numpy(),s._hS.numpy(),s._hL.numpy(),s._hC.numpy(),s._hLc.numpy()
        for b,p in enumerate(paths):
            st=max(1,int(math.ceil(len(p['xy'])/s.NP))); xy=p['xy'][::st]; ss=p['s'][::st]; n=len(xy)
            X[b,:n]=xy; X[b,n:]=xy[-1]; S[b,:n]=ss; S[b,n:]=ss[-1]; Lh[b]=p['s'][-1]
            cc=s._cumc(xy,ss,b); Cc[b,:n]=cc; Cc[b,n:]=cc[-1]; Lc[b]=cc[-1]
        s.pxy.copy_(s._hX,non_blocking=True); s.ps.copy_(s._hS,non_blocking=True); s.plen.copy_(s._hL,non_blocking=True); s.pc.copy_(s._hC,non_blocking=True); s.pclen.copy_(s._hLc,non_blocking=True); ev.record()
        s.host_paths={b:p for b,p in enumerate(paths)}
        if not keep_progress: s.sprog.zero_(); s.U.zero_()
    def path_ctg(s, xy, state, back=0.5, ahead=4.0, wlat=None):
        wlat=s.wlat if wlat is None else wlat
        """xy (B,K,2) -> remaining arclength + lateral penalty, nearest point restricted to window around progress."""
        d0=torch.cdist(state[:,None,:2],s.pxy)[:,0]                           # (B,NP) robot -> path
        win0=(s.ps>=s.sprog[:,None]-back)&(s.ps<=s.sprog[:,None]+ahead)
        j=torch.where(win0,d0,torch.full_like(d0,1e9)).argmin(1)
        s.sprog.copy_(torch.maximum(s.sprog,s.ps.gather(1,j[:,None])[:,0]))  # monotone progress (in place: graph safe)
        i0=torch.searchsorted(s.ps,(s.sprog-back)[:,None].contiguous())[:,0].clamp(max=s.NP-s.WN)  # fixed-size window (4.5 m @5 cm)
        idx=i0[:,None]+torch.arange(s.WN,device=s.dev)[None]
        wx=s.pxy.gather(1,idx[...,None].expand(-1,-1,2)); ws=s.pc.gather(1,idx)
        D=torch.cdist(xy,wx)                                                   # (B,K,WN)
        dmin,jj=D.min(2)
        rem=s.pclen[:,None]-ws.gather(1,jj)
        return rem+wlat*dmin
    @torch.no_grad()
    def plan_fast(s, state, nprop=5):
        B,K,H,dt,L=s.B,s.K,s.H,s.dt,s.L
        eps=torch.randn(B,K,H,2,device=s.dev)*s.sig
        eps[:,0]=0; eps[:,1]=-s.U.unsqueeze(1)[:,0]
        U=(s.U.unsqueeze(1)+eps)
        U[...,0]=U[...,0].clamp(-L.a_max,L.a_max); U[...,1]=U[...,1].clamp(-L.alpha_max,L.alpha_max)
        state=state.float().contiguous()
        cost,deadf,TX=_mt.rollout(U,state,s.goal,s.fp,s.edt,L,dt,3)
        dead=deadf>0
        cost=cost+s.ahead_pt*s.path_ctg(TX[:,:,-1,:2].contiguous(),state)
        beta=cost.min(1,keepdim=True).values
        wts=torch.softmax(-(cost-beta)/s.lam,1)
        Unew=(wts[...,None,None]*U).sum(1)
        best=cost.argmin(1); bi=torch.arange(B,device=s.dev)
        _,d1,chx=_mt.rollout(Unew.unsqueeze(1).contiguous(),state,s.goal,s.fp,s.edt,L,dt,6)
        fb=(d1[:,0]>0)&~dead[bi,best]
        Uc=torch.where(fb[:,None,None],U[bi,best],Unew)
        u0=Uc[:,0].clone(); s.U=torch.cat([Uc[:,1:],Uc[:,-1:]*0.0],1)
        dth=(TX[:,:,-1,2]-state[:,2:3]+math.pi)%(2*math.pi)-math.pi
        bins=((dth+math.pi)/(2*math.pi)*nprop).long().clamp(0,nprop-1)
        cpen=cost+dead.float()*1e6
        cb=torch.where(bins[:,None,:]==torch.arange(nprop,device=s.dev)[None,:,None],cpen[:,None,:],torch.full_like(cpen[:,None,:],math.inf))
        props=cb.argmin(2); pc=torch.gather(cpen,1,props); ptraj=TX[bi[:,None],props]
        return u0, dict(chosen=chx[:,0], props=ptraj, pcost=pc, mincost=beta[:,0])

KPLAN_GLOBAL=os.environ.get('KPLAN_GLOBAL','field')   # 'field' (existing cost-to-go field MPPI, default) | 'fast'
def make_planner(world, B, **kw):
    """Switch point: returns (mppi, set_goal_fn). set_goal_fn(goals (B,3), starts (B,2)) does the per-goal work."""
    if KPLAN_GLOBAL=='fast':
        m=MapCPU(world); M=PathMPPI(world,B,**kw); at=[AnytimePlanner(m) for _ in range(B)]
        def set_goal(goals,starts):
            M.set_paths(goals,[a.new_goal(np.asarray(s,float),np.asarray(g[:2],float)) for a,s,g in zip(at,starts,goals)])
        def poll(goals):
            r=[a.poll() for a in at]
            if any(x is not None for x in r): M.set_paths(goals,[x if x is not None else dict(xy=M.pxy[b].cpu().numpy().astype(float),s=M.ps[b].cpu().numpy().astype(float)) for b,x in enumerate(r)])
        M.poll=poll
        return M,set_goal
    M=MPPI(world,B,**kw)
    def set_goal(goals,starts):
        D,_=world.fields(np.asarray(goals)[:,:2]); M.set_goals(goals,D)
    M.poll=lambda goals:None
    return M,set_goal

class FastNewGoal:
    """B=1 (or B) low-latency new-goal pipeline: C global path -> pinned upload -> CUDA-graph MPPI plan."""
    def __init__(s, world, B=1, m=None, eps=2.0, **kw):
        s.m=m or MapCPU(world); s.M=PathMPPI(world,B,**kw); s.B=B; s.eps=eps
        g=np.array([[1.0,1.0,0.0]]*B); p=dict(xy=np.array([[1.0,1.0],[1.1,1.0]]),s=np.array([0,0.1]))
        s.M.set_paths(g,[p]*B)
        from planner import GraphPlanner
        s.G=GraphPlanner(s.M); s.state=s.G.state
    def new_goal(s, state_np, goal):
        """state_np (5,) host; goal (3,). Returns (u0 GPU tensor (B,2), path dict).
        Overlap: the first plan runs on the string-pulled path while the CPU smooths the near part;
        the smoothed path is uploaded behind it in the same stream (used from the next replan)."""
        p=plan_path(s.m,np.asarray(state_np[:2],float),np.asarray(goal[:2],float),eps=s.eps,smooth_iters=0)
        if p is None: return None,None
        G=np.asarray(goal,float)[None].repeat(s.B,0)
        s.M.set_paths(G,[p]*s.B)
        u,_=s.G.plan(torch.as_tensor(np.asarray(state_np,np.float32)[None],device=s.M.dev).expand(s.B,5))
        t=time.perf_counter(); q=smooth_near(s.m,p); q['ms']=dict(p['ms']); q['ms']['smooth']=(time.perf_counter()-t)*1e3; q['ms']['total']+=q['ms']['smooth']
        s.M.set_paths(G,[q]*s.B,keep_progress=True)
        return u,q

# ---------------- sim.py integration helpers (KPLAN_GLOBAL=fast) ----------------
def _pm_ctg_points(s, xy, back=0.5, wlat=None):
    """Pure (no progress update) path cost-to-go at arbitrary points xy (B,N,2) -- replaces bilinear(M.field, xy)."""
    i0=torch.searchsorted(s.ps,(s.sprog-back)[:,None].contiguous())[:,0].clamp(max=s.NP-s.WN)
    idx=i0[:,None]+torch.arange(s.WN,device=s.dev)[None]
    wx=s.pxy.gather(1,idx[...,None].expand(-1,-1,2)); ws=s.pc.gather(1,idx)
    dmin,jj=torch.cdist(xy,wx).min(2)
    return s.pclen[:,None]-ws.gather(1,jj)+(s.wlat if wlat is None else wlat)*dmin
def _pm_set_paths_idx(s, idx, goals, paths):
    """Replace the paths of envs idx (list) only; resets their progress and nominal controls."""
    for b,g,p in zip(idx,goals,paths):
        st=max(1,int(math.ceil(len(p['xy'])/s.NP))); xy=p['xy'][::st]; ss=p['s'][::st]; n=len(xy)
        X=np.empty((s.NP,2),np.float32); S=np.empty(s.NP,np.float32); X[:n]=xy; X[n:]=xy[-1]; S[:n]=ss; S[n:]=ss[-1]
        s.pxy[b].copy_(torch.from_numpy(X)); s.ps[b].copy_(torch.from_numpy(S)); s.plen[b]=float(p['s'][-1])
        cc=s._cumc(xy,ss,b).astype(np.float32); Cc=np.empty(s.NP,np.float32); Cc[:n]=cc; Cc[n:]=cc[-1]; s.pc[b].copy_(torch.from_numpy(Cc)); s.pclen[b]=float(cc[-1])
        s.goal[b]=torch.as_tensor(np.asarray(g,np.float32),device=s.dev); s.sprog[b]=0; s.U[b]=0
    s.host_paths=getattr(s,'host_paths',{}); s.host_paths.update({b:p for b,p in zip(idx,paths)})
PathMPPI.ctg_points=_pm_ctg_points; PathMPPI.set_paths_idx=_pm_set_paths_idx

class SimPathOut:
    """Stand-in for gpath.PathOut in fast mode (viewer/output only): fine 1 cm part (<=10 s) from the robot's
    projection on its fastpath path + coarse remainder every 20 cm. Keys match gpath.pack()."""
    def __init__(s, M, L=None, horizon_s=10.0):
        from core import Limits; s.M=M; s.L=L or Limits; s.hz=horizon_s
    def plan(s, st, v0=None):
        M=s.M; B=M.B; L=s.L; out=[]
        sp=M.sprog.cpu().numpy()
        for b in range(B):
            p=M.host_paths[b]; P,S=p['xy'],p['s']; s0=float(sp[b]); s1=min(S[-1],s0+L.v_max*s.hz)
            q=np.arange(s0,s1+1e-9,0.01); q=q if len(q)>1 else np.array([s0,S[-1]])
            xy=np.c_[np.interp(q,S,P[:,0]),np.interp(q,S,P[:,1])]; xy[0]=st[b,:2].cpu().numpy()
            rem=S[-1]-q; v=np.minimum(L.v_max,np.sqrt(2*L.a_max*0.5*np.maximum(rem,0))); v=np.minimum(v,np.sqrt(2*L.a_max*0.5*(q-s0)+float(v0[b] if v0 is not None else 0)**2))
            t=np.r_[0,np.cumsum(0.01/np.maximum(0.5*(v[1:]+v[:-1]),1e-2))]
            d=np.gradient(xy,axis=0); th=np.arctan2(d[:,1],d[:,0])
            qc=np.arange(s1,S[-1]+1e-9,0.2); cxy=np.c_[np.interp(qc,S,P[:,0]),np.interp(qc,S,P[:,1])] if len(qc) else xy[-1:]
            out.append((xy,v,t,th,cxy))
        Mx=max(len(o[0]) for o in out); Mc=max(len(o[4]) for o in out)
        R=dict(xy=np.zeros((B,Mx,2)),v=np.zeros((B,Mx)),t=np.zeros((B,Mx)),theta=np.zeros((B,Mx)),vcap=np.zeros((B,Mx)),fine_mask=np.zeros((B,Mx),bool),
               spin=np.zeros(B),coarse_xy=np.zeros((B,Mc,2)),coarse_valid=np.zeros((B,Mc),bool))
        for b,(xy,v,t,th,cxy) in enumerate(out):
            n=len(xy); R['xy'][b,:n]=xy; R['xy'][b,n:]=xy[-1]; R['v'][b,:n]=v; R['t'][b,:n]=t; R['theta'][b,:n]=th; R['vcap'][b,:n]=L.v_max; R['fine_mask'][b,:n]=True
            R['coarse_xy'][b,:len(cxy)]=cxy; R['coarse_valid'][b,:len(cxy)]=True
        return {k:torch.from_numpy(np.asarray(v)) for k,v in R.items()}
    def refresh(s,*a): pass

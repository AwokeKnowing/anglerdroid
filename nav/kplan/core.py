"""Kevin GPU planner core (torch). Units: metres, seconds, radians. Map: 1 px = 1 cm, occ[y, x] (row = y)."""
import math, os, time, numpy as np, torch

# 2026-10-13: rear diagonal corners = twin_geometry.FOOTPRINT_SAFE (sim chassis_top_deck corners stick out of SIL12)
SIL12 = [[-0.27371,-0.01022],[-0.1995,-0.15],[-0.08009,-0.20998],[0.08319,-0.20998],[0.12372,-0.15418],[0.12629,-0.13818],
         [0.12629,0.13818],[0.12372,0.15418],[0.08319,0.20998],[-0.08009,0.20998],[-0.1995,0.15],[-0.27371,0.01022]]

class Limits:
    v_max=min(0.6, float(os.environ.get("KEVIN_VMAX", "0.6"))); v_min=-0.2;   # KEVIN_VMAX: bring-up speed cap   # reverse capped at 0.2 m/s (James 2026-10-13: too much reversing)
    rev_w=4.0; head_w=1.5    # MPPI: reverse cost scaled by clearance; face-the-carrot cost (mppi_cuda)
    a_max=1.96; a_lat=1.96; w_max=2.5; alpha_max=6.0; track=0.362; v_wheel=1.0
    # proximity speed rule (hard): v*t_react + v^2/(2 a_brake) <= d - d_safe ; plus a comfort ramp
    a_brake=1.5; t_react=0.10; d_safe=0.03; d_full=0.50; v_near=0.15; d_near=0.05   # a_brake = nav.safety (drive decel ~1.6 m/s^2: ODrive vel_ramp 3 turns/s^2)

def v_cap_clear(d, L=Limits):
    """d = outline clearance (m, min EDT over the 12-gon boundary). Returns the speed cap (m/s)."""
    x=(d-L.d_safe).clamp(min=0)
    hard=L.a_brake*(-L.t_react+torch.sqrt(L.t_react**2+2*x/L.a_brake))
    ramp=L.v_near+(L.v_max-L.v_near)*((d-L.d_near)/(L.d_full-L.d_near)).clamp(0,1)
    return torch.minimum(torch.minimum(hard,ramp),torch.full_like(d,L.v_max))

def footprint_samples(spacing=0.03, dev='cuda'):
    """Boundary points of the 12-gon (body frame) every <= spacing m, plus interior ring."""
    V=np.array(SIL12); pts=[]
    for i in range(len(V)):
        a,b=V[i],V[(i+1)%len(V)]; n=max(1,int(math.ceil(np.linalg.norm(b-a)/spacing)))
        for t in range(n): pts.append(a+(b-a)*t/n)
    return torch.tensor(np.array(pts),dtype=torch.float32,device=dev)

# ---------------- EDT (exact Euclidean, capped) ----------------
@torch.no_grad()
def edt_capped(occ, cap_px=80, chunk=1000):
    """occ: (H,W) bool cuda. Returns float16 metres clearance (dist to nearest occupied px centre), capped."""
    H,W=occ.shape; dev=occ.device
    big=cap_px+1
    g=torch.empty((H,W),dtype=torch.float16,device=dev)
    ar=torch.arange(W,device=dev,dtype=torch.int32)
    for r0 in range(0,H,chunk):
        o=occ[r0:r0+chunk]
        idx=torch.where(o,ar,torch.full_like(ar,-10**6)).cummax(1).values
        f=(ar-idx)
        idx2=torch.where(o.flip(1),ar,torch.full_like(ar,-10**6)).cummax(1).values
        b=(ar-idx2).flip(1)
        g[r0:r0+chunk]=torch.minimum(torch.minimum(f,b),torch.tensor(big,device=dev)).half()
    out=torch.empty((H,W),dtype=torch.float16,device=dev)
    for c0 in range(0,W,chunk*4):
        gc=g[:,c0:c0+chunk*4].float()**2
        best=gc.clone()
        pad=torch.full((cap_px,gc.shape[1]),float(big*big),device=dev)
        gp=torch.cat([pad,gc,pad],0)
        for dy in range(1,cap_px+1):
            dd=dy*dy
            best=torch.minimum(best,gp[cap_px+dy:cap_px+dy+H]+dd)
            best=torch.minimum(best,gp[cap_px-dy:cap_px-dy+H]+dd)
        out[:,c0:c0+chunk*4]=(best.sqrt().clamp(max=big)*0.01).half()
        del gc,best,gp
    return out

# ---------------- cost-to-go (GPU exact 8-connected DP via scan passes) ----------------
class DiagIndex:
    def __init__(s,R,C,dev):
        ii,jj=torch.meshgrid(torch.arange(R,device=dev),torch.arange(C,device=dev),indexing='ij')
        s.maps=[]
        for anti in (False,True):
            k=(jj-ii+R-1) if not anti else (ii+jj)
            nd=R+C-1
            pos=ii  # position along diagonal = row
            L=R
            flat=torch.full((nd,L),-1,dtype=torch.long,device=dev)
            flat[k.reshape(-1),pos.reshape(-1)]=(ii*C+jj).reshape(-1)
            s.maps.append(flat)

def _scan(D,w,blk,rev):
    """D,w,blk: (...,L). w[i] = cost of step (i-1 -> i) along the line. rev=False propagates increasing i."""
    if rev: D,w,blk=D.flip(-1),w.flip(-1),blk.flip(-1); w=torch.roll(w,1,-1)  # step cost reversed alignment
    pref=torch.cumsum(w,-1); seg=torch.cumsum(blk.double(),-1)
    BIG=1e7
    v=D-pref-seg*BIG
    m=torch.cummin(v,-1).values
    new=m+pref+seg*BIG
    new=torch.where(new>BIG*0.5,torch.full_like(new,math.inf),new)
    out=torch.minimum(D,new); out=torch.where(blk,torch.full_like(out,math.inf),out)
    return out.flip(-1) if rev else out

_DIAG={}
@torch.no_grad()
def cost_to_go(cost, blocked, goals_rc, h, iters=400, diag=True, tol=1e-6, D0=None):
    """cost: (R,C) or (B,R,C) traversal multiplier (>=1); blocked same shape (bool); goals_rc (B,2) long.
    D0: optional warm start (B,R,C) (values must be upper bounds or inf). Returns ((B,R,C) float32, iters).
    Exact min-cost 8-connected path cost (edge cost = h*len*(c_a+c_b)/2) by alternating directional scans (GPU)."""
    B=goals_rc.shape[0]; R,C=cost.shape[-2:]; dev=cost.device
    c=cost.double()
    if c.dim()==2: c=c.unsqueeze(0).expand(B,R,C)
    blk=blocked if blocked.dim()==3 else blocked.unsqueeze(0).expand(B,R,C)
    D=torch.full((B,R,C),math.inf,dtype=torch.float64,device=dev) if D0 is None else D0.double().clone()
    D[torch.arange(B),goals_rc[:,0],goals_rc[:,1]]=0
    D=torch.where(blk,torch.full_like(D,math.inf),D)
    wx=torch.zeros(B,R,C,dtype=torch.float64,device=dev); wx[:,:,1:]=h*(c[:,:,1:]+c[:,:,:-1])/2
    wy=torch.zeros(B,R,C,dtype=torch.float64,device=dev); wy[:,1:,:]=h*(c[:,1:,:]+c[:,:-1,:])/2
    wyt=wy.transpose(1,2).contiguous(); blkt=blk.transpose(1,2).contiguous()
    if diag:
        key=(R,C,str(dev))
        if key not in _DIAG: _DIAG[key]=DiagIndex(R,C,dev)
        di=_DIAG[key]; dmaps=[]
        cpad=torch.cat([c.reshape(B,-1),torch.full((B,1),1e9,device=dev,dtype=torch.float64)],1)
        bpad=torch.cat([blk.reshape(B,-1),torch.ones((B,1),dtype=torch.bool,device=dev)],1)
        for M in di.maps:
            Mi=torch.where(M<0,torch.full_like(M,R*C),M)
            cc=cpad[:,Mi]; w=torch.zeros_like(cc); w[:,:,1:]=h*math.sqrt(2)*(cc[:,:,1:]+cc[:,:,:-1])/2
            w=torch.where(w<1e8,w,torch.zeros_like(w))
            dmaps.append((M,Mi,w,bpad[:,Mi]))
    prev=None; it=0
    for it in range(iters):
        D=_scan(D,wx,blk,False); D=_scan(D,wx,blk,True)
        Dt=D.transpose(1,2)
        Dt=_scan(Dt,wyt,blkt,False); Dt=_scan(Dt,wyt,blkt,True)
        D=Dt.transpose(1,2).contiguous()
        if diag:
            for (M,Mi,w,bm) in dmaps:
                Df=torch.cat([D.reshape(B,-1),torch.full((B,1),math.inf,dtype=D.dtype,device=dev)],1)
                X=Df[:,Mi]
                X=_scan(X,w,bm,False); X=_scan(X,w,bm,True)
                valid=M>=0
                Df[:,M[valid]]=X[:,valid]
                D=Df[:,:-1].reshape(B,R,C)
        if it%2==1:
            fin=torch.isfinite(D)
            s=torch.where(fin,D,torch.zeros_like(D)).sum().item()+fin.sum().item()*1e3
            if prev is not None and abs(s-prev)<tol*max(1,abs(s)): break
            prev=s
    return D.float(), it+1

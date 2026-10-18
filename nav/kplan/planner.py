import math, time, numpy as np, torch, torch.nn.functional as F
import os
# fused rollout backend: KPLAN_BACKEND=cuda|triton|auto (auto=cuda; triton only when explicitly requested); KPLAN_FAST=0 -> plan_ref
_mt=None; _BK=os.environ.get('KPLAN_BACKEND','auto')
import sys as _sys; os.environ['PATH']=os.path.dirname(_sys.executable)+os.pathsep+os.environ.get('PATH','')  # venv's ninja for the CUDA-C ext build
if os.environ.get('KPLAN_FAST','1')!='0':
    for _n in (['cuda'] if _BK=='auto' else [_BK]):   # CUDA-C default; Triton only with KPLAN_BACKEND=triton
        try:
            _mt=__import__('mppi_'+_n); KPLAN_BACKEND=_n; break
        except Exception as _e: print('kplan: backend',_n,'unavailable:',repr(_e)[:200])
import sys as _s, os as _o; _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), '..', '..', 'src') in _s.path or _s.path.append(_o.path.join(_o.path.dirname(_o.path.abspath(__file__)), '..', '..', 'src'))   # twin_geometry (single source of the footprint radii)
from twin_geometry import R_ROTATE, R_PASS
from core import Limits, footprint_samples, edt_capped, cost_to_go, SIL12, v_cap_clear

HARD_CLR=0.200   # axle clearance below which a coarse cell is blocked (robot half-width 0.2036 / 12-gon 0.21)
PREF_CLR=0.40
INFEASIBLE=1e4   # MPPI: a rollout with dead=1 (footprint hit, or reversed into space not observed free) gets no weight unless all do

class World:
    def __init__(s, occ_np, dev='cuda', coarse=10, rev_block=None):
        """rev_block: optional bool mask of cells that may not be entered while REVERSING (not currently observed free);
        edt_rev = clearance to obstacles or those cells.  Without it reversing uses the normal clearance."""
        t=time.time(); s.dev=dev
        s.occ=torch.from_numpy(occ_np).to(dev); s.H,s.W=s.occ.shape
        torch.cuda.synchronize(); t1=time.time()
        s.edt=edt_capped(s.occ,cap_px=80)
        s.edt_rev=edt_capped(s.occ|torch.from_numpy(rev_block).to(dev),cap_px=80) if rev_block is not None else s.edt.clone()
        torch.cuda.synchronize(); t2=time.time()
        s.res=0.01; s.cres=coarse*0.01; s.k=coarse
        e=s.edt.float().unsqueeze(0).unsqueeze(0)
        s.cmax=F.max_pool2d(e,coarse)[0,0]; s.cmean=F.avg_pool2d(e,coarse)[0,0]
        s.cblocked=s.cmax<HARD_CLR
        x=((PREF_CLR-s.cmean)/(PREF_CLR-HARD_CLR)).clamp(0,1)
        turn=((R_ROTATE-s.cmean)/(R_ROTATE-R_PASS)).clamp(0,1)   # can't spin in place here (twin_geometry R_ROTATE..R_PASS)
        s.ccost=1+4*x*x+1.5*turn
        torch.cuda.synchronize()
        s.timing=dict(upload=t1-t, edt=t2-t1, coarse=time.time()-t2)
    def fields(s, goals_xy, chunk=16):
        """goals_xy (B,2) metres -> (B,Rc,Cc) cost-to-go metres (float32, inf blocked)."""
        g=torch.as_tensor(goals_xy,dtype=torch.float32,device=s.dev)
        rc=torch.stack([(g[:,1]/s.cres).long(),(g[:,0]/s.cres).long()],1)
        out=[];its=[]
        for i in range(0,len(rc),chunk):
            D,it=cost_to_go(s.ccost,s.cblocked,rc[i:i+chunk],s.cres); out.append(D); its.append(it)
        return torch.cat(out), its

def edt_at(edt, iy, ix):
    """edt (H,W) shared or (B,H,W) per-env (partial observability); iy/ix (B,...) long."""
    if edt.dim()==2: return edt[iy,ix]
    b=torch.arange(edt.shape[0],device=edt.device).view(-1,*([1]*(iy.dim()-1)))
    return edt[b,iy,ix]

def bilinear(field, xy, cres):
    """field (B,R,C) finite; xy (B,N,2) metres -> (B,N)"""
    B,R,C=field.shape
    gx=(xy[...,0]/cres-0.5)/(C-1)*2-1; gy=(xy[...,1]/cres-0.5)/(R-1)*2-1
    return F.grid_sample(field.unsqueeze(1),torch.stack([gx,gy],-1).unsqueeze(1),mode='bilinear',padding_mode='border',align_corners=True)[:,0,0]

class MPPI:
    def __init__(s, world, B, K=384, H=40, dt=0.05, lim=Limits, seed=0):
        s.w=world; s.B,s.K,s.H,s.dt,s.L=B,K,H,dt,lim; d=world.dev; s.dev=d
        s.U=torch.zeros(B,H,2,device=d)
        s.sig=torch.tensor([0.9,3.0],device=d)
        s.fp=footprint_samples(0.03,d)          # (P,2)
        s.gen=torch.Generator(device=d); s.gen.manual_seed(seed)
        s.edt=world.edt; s.lam=0.05
        s.fast=_mt is not None and str(d).startswith('cuda') and os.environ.get('KPLAN_FAST','1')!='0'
    def set_goals(s, goals, fields):
        g=torch.as_tensor(goals,dtype=torch.float32,device=s.dev)  # (B,3)
        f=torch.where(torch.isfinite(fields),fields,torch.full_like(fields,1e3))
        if hasattr(s,'goal'): s.goal.copy_(g); s.field.copy_(f)      # in place (CUDA-graph safe)
        else: s.goal=g.clone(); s.field=f.contiguous()
    @staticmethod
    def step_dyn(x,y,th,v,w,a,al,dt,L,vcap=None):
        v0=v
        v=(v+a*dt).clamp(L.v_min,L.v_max)
        if vcap is not None:   # proximity rule; never brake harder than a_brake per step
            lim=torch.maximum(vcap,v0.abs()-L.a_brake*dt)
            v=torch.maximum(torch.minimum(v,lim),-torch.minimum(lim,torch.full_like(lim,-L.v_min)))
        wl=torch.minimum(torch.full_like(v,L.w_max), L.a_lat/v.abs().clamp(min=1e-3))
        wl=torch.minimum(wl,((L.v_wheel-v.abs())/(L.track/2)).clamp(min=0))
        w=torch.maximum(torch.minimum(w+al*dt,wl),-wl)
        th=th+w*dt; x=x+v*torch.cos(th)*dt; y=y+v*torch.sin(th)*dt
        return x,y,th,v,w
    def clearance(s, x, y, th, edt=None):
        """min EDT over footprint boundary samples; x,y,th (...); edt: field (default s.edt; s.w.edt_rev when reversing)"""
        c,sn=torch.cos(th)[...,None],torch.sin(th)[...,None]
        px=x[...,None]+c*s.fp[:,0]-sn*s.fp[:,1]; py=y[...,None]+sn*s.fp[:,0]+c*s.fp[:,1]
        ix=(px*100).long().clamp(0,s.w.W-1); iy=(py*100).long().clamp(0,s.w.H-1)
        return edt_at(s.edt if edt is None else edt,iy,ix).float().amin(-1)
    @torch.no_grad()
    def plan(s, state, nprop=5):
        if s.fast: return s.plan_fast(state, nprop)
        return s.plan_ref(state, nprop)
    @torch.no_grad()
    def plan_fast(s, state, nprop=5):
        """Same maths as plan_ref, rollouts fused into one Triton kernel (mppi_triton.py)."""
        B,K,H,dt,L=s.B,s.K,s.H,s.dt,s.L
        eps=torch.randn(B,K,H,2,device=s.dev)*s.sig
        eps[:,0]=0
        eps[:,1]=-s.U.unsqueeze(1)[:,0]
        U=(s.U.unsqueeze(1)+eps)
        U[...,0]=U[...,0].clamp(-L.a_max,L.a_max); U[...,1]=U[...,1].clamp(-L.alpha_max,L.alpha_max)
        state=state.float().contiguous()
        cost,deadf,TX=_mt.rollout(U,state,s.goal,s.fp,s.edt,L,dt,3,erev=s.w.edt_rev)
        dead=deadf>0
        cost=cost+10*bilinear(s.field,TX[:,:,-1,:2],s.w.cres)
        csel=cost+dead.float()*INFEASIBLE                                      # colliding / reversing-blind rollouts are infeasible, not just costly
        beta=csel.min(1,keepdim=True).values
        wts=torch.softmax(-(csel-beta)/s.lam,1)
        Unew=(wts[...,None,None]*U).sum(1)
        best=csel.argmin(1); bi=torch.arange(B,device=s.dev)
        _,d1,chx=_mt.rollout(Unew.unsqueeze(1).contiguous(),state,s.goal,s.fp,s.edt,L,dt,6,erev=s.w.edt_rev)
        ch=chx[:,0]; dead1=d1[:,0]>0
        fb=dead1&~dead[bi,best]
        Uc=torch.where(fb[:,None,None],U[bi,best],Unew)
        # command = the first step the rollout actually EXECUTED (after its braking/limits), not the raw sampled
        # acceleration: a capped rollout ignores a, so the raw a could drive the robot faster than what was checked
        # (KEVIN_PLAN_OWNS_CLEARANCE only; default: the raw first step, PR #62)
        if getattr(L, "poc", False):
            _,_,cx=_mt.rollout(Uc.unsqueeze(1).contiguous(),state,s.goal,s.fp,s.edt,L,dt,6,erev=s.w.edt_rev)
            u0=torch.stack([(cx[:,0,0,3]-state[:,3])/dt,(cx[:,0,0,4]-state[:,4])/dt],1)
            ch=torch.where(fb[:,None,None],cx[:,0],ch)
        else: u0=Uc[:,0].clone()
        s.U=torch.cat([Uc[:,1:],Uc[:,-1:]*0.0],1)
        dth=(TX[:,:,-1,2]-state[:,2:3]+math.pi)%(2*math.pi)-math.pi
        bins=((dth+math.pi)/(2*math.pi)*nprop).long().clamp(0,nprop-1)
        cpen=cost+dead.float()*1e6
        cb=torch.where(bins[:,None,:]==torch.arange(nprop,device=s.dev)[None,:,None],cpen[:,None,:],torch.full_like(cpen[:,None,:],math.inf))
        props=cb.argmin(2)
        pc=torch.gather(cpen,1,props)
        ptraj=TX[bi[:,None],props]
        return u0, dict(chosen=ch, props=ptraj, pcost=pc, mincost=beta[:,0])
    @torch.no_grad()
    def plan_ref(s, state, nprop=5):
        """state (B,5) x,y,th,v,w. Returns control (B,2), info dict with chosen traj & proposals."""
        B,K,H,dt,L=s.B,s.K,s.H,s.dt,s.L
        eps=torch.randn(B,K,H,2,device=s.dev)*s.sig
        eps[:,0]=0                                          # keep nominal
        eps[:,1]=-s.U.unsqueeze(1)[:,0]                     # zero-control sample (brake/coast)
        U=(s.U.unsqueeze(1)+eps)
        U[...,0]=U[...,0].clamp(-L.a_max,L.a_max); U[...,1]=U[...,1].clamp(-L.alpha_max,L.alpha_max)
        x,y,th,v,w=[state[:,i:i+1].expand(B,K).clone() for i in range(5)]
        gx,gy,gth=[s.goal[:,i:i+1] for i in range(3)]
        cost=torch.zeros(B,K,device=s.dev); dead=torch.zeros(B,K,dtype=torch.bool,device=s.dev)
        TX=torch.empty(B,K,H,3,device=s.dev)
        clr=s.clearance(x,y,th)
        for t in range(H):
            x,y,th,v,w=s.step_dyn(x,y,th,v,w,U[:,:,t,0],U[:,:,t,1],dt,L,v_cap_clear(clr))
            TX[:,:,t,0]=x;TX[:,:,t,1]=y;TX[:,:,t,2]=th
            clr=torch.where(v<0,s.clearance(x,y,th,s.w.edt_rev),s.clearance(x,y,th))
            dead|=clr<0.012
            ax=edt_at(s.edt,(y*100).long().clamp(0,s.w.H-1),(x*100).long().clamp(0,s.w.W-1)).float()
            cost+=dead.float()*50+20*torch.relu(0.06-clr)
            cost+=dt*(1.0*torch.exp(-clr/0.15)+2.0*(torch.relu(R_ROTATE-ax)/(R_ROTATE-R_PASS)).clamp(max=1))   # proximity + not-turnable
            dg=torch.hypot(x-gx,y-gy)
            near=torch.exp(-dg/0.25)
            cost+=0.1*dt*20*near*(1-torch.cos(th-gth))*2 + (0.5+4.0*near)*dt*(v<0).float()*v.abs()/0.3   # v7: reversing near a goal is expensive (arrive forward)
        ctg=bilinear(s.field,TX[:,:,-1,:2].reshape(B,K,2),s.w.cres)
        dg=torch.hypot(x-gx,y-gy)
        nearT=torch.exp(-dg/0.15)
        cost+=10*ctg + 3*nearT*(1-torch.cos(th-gth)) + 2*nearT*(v.abs()+w.abs()*0.3)
        cost+=0.002*(U**2).sum((2,3))
        csel=cost+dead.float()*INFEASIBLE                                      # colliding / reversing-blind rollouts are infeasible, not just costly
        beta=csel.min(1,keepdim=True).values
        wts=torch.softmax(-(csel-beta)/s.lam,1)
        Unew=(wts[...,None,None]*U).sum(1)
        best=csel.argmin(1)
        s.U=Unew
        bi=torch.arange(B,device=s.dev)
        # chosen trajectory: rollout of the averaged plan; fall back to the best sample if it collides
        x,y,th,v,w=[state[:,i].clone() for i in range(5)]
        ch=torch.empty(B,H,6,device=s.dev); clr=s.clearance(x[:,None],y[:,None],th[:,None])[:,0]
        for t in range(H):
            vc=v_cap_clear(clr)
            x,y,th,v,w=s.step_dyn(x,y,th,v,w,s.U[:,t,0],s.U[:,t,1],dt,L,vc); clr=s.clearance(x[:,None],y[:,None],th[:,None])[:,0]
            ch[:,t]=torch.stack([x,y,th,v,w,vc],1)
        dead1=(s.clearance(ch[...,0],ch[...,1],ch[...,2])<0.012).any(1)
        fb=dead1&~dead[bi,best]
        s.U=torch.where(fb[:,None,None],U[bi,best],s.U)
        u0=s.U[:,0].clone()
        s.U=torch.cat([s.U[:,1:],s.U[:,-1:]*0.0],1)
        # diverse proposals: best collision-free sample per bin of final heading change
        dth=(TX[:,:,-1,2]-state[:,2:3]+math.pi)%(2*math.pi)-math.pi
        bins=((dth+math.pi)/(2*math.pi)*nprop).long().clamp(0,nprop-1)
        cpen=cost+dead.float()*1e6
        props=[]
        for b_ in range(nprop):
            cb=torch.where(bins==b_,cpen,torch.full_like(cpen,math.inf)); props.append(cb.argmin(1))
        props=torch.stack(props,1)  # (B,nprop)
        pc=torch.gather(cpen,1,props)
        ptraj=TX[bi[:,None],props]   # (B,nprop,H,3)
        return u0, dict(chosen=ch, props=ptraj, pcost=pc, mincost=beta[:,0])

def reached(state, goal, pos_tol=0.03, ang_tol=math.radians(5)):
    dp=torch.hypot(state[:,0]-goal[:,0],state[:,1]-goal[:,1])
    da=((state[:,2]-goal[:,2]+math.pi)%(2*math.pi)-math.pi).abs()
    return (dp<pos_tol)&(da<ang_tol)&(state[:,3].abs()<0.05), dp, da

# ---------- global path extraction (start -> goal), 1 cm waypoints + time-parameterised velocity ----------
def extract_path(world, field_b, start, goal, lim=Limits):
    """field_b (R,C) cpu numpy float. Steepest descent on the coarse field, smoothing with clearance push,
    1 cm resample, heading = tangent, velocity profile under a_max, lateral accel, yaw rate and yaw accel."""
    R,C=field_b.shape; cr=world.cres
    r,c=int(start[1]/cr),int(start[0]/cr); pts=[(start[0],start[1])]
    nb=[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    for _ in range(R*C):
        if not np.isfinite(field_b[r,c]) : return None
        if field_b[r,c]<=1e-9: break
        best=None
        for dr,dc in nb:
            rr,cc=r+dr,c+dc
            if 0<=rr<R and 0<=cc<C and field_b[rr,cc]<field_b[r,c] and (best is None or field_b[rr,cc]<field_b[best]): best=(rr,cc)
        if best is None: break
        r,c=best; pts.append(((c+.5)*cr,(r+.5)*cr))
    pts.append((goal[0],goal[1])); P=np.array(pts,float)
    # resample to 2 cm, smooth (Laplacian + push away from obstacles via EDT gradient)
    def resample(P,ds):
        d=np.r_[0,np.cumsum(np.hypot(*np.diff(P,axis=0).T))]; n=max(2,int(d[-1]/ds)+1)
        s=np.linspace(0,d[-1],n); return np.c_[np.interp(s,d,P[:,0]),np.interp(s,d,P[:,1])]
    P=resample(P,0.02)
    E=world.edt
    for it in range(60):
        Q=P.copy(); Q[1:-1]=P[1:-1]+0.4*(P[:-2]+P[2:]-2*P[1:-1])
        ix=np.clip((Q[:,0]*100).astype(int),1,world.W-2); iy=np.clip((Q[:,1]*100).astype(int),1,world.H-2)
        it_=torch.from_numpy(np.stack([iy,ix])).to(world.dev)
        e=E[it_[0],it_[1]].float(); gx=(E[it_[0],it_[1]+1].float()-E[it_[0],it_[1]-1].float())/0.02
        gy=(E[it_[0]+1,it_[1]].float()-E[it_[0]-1,it_[1]].float())/0.02
        e,gx,gy=[a.cpu().numpy() for a in (e,gx,gy)]
        push=np.clip(PREF_CLR-e,0,None)*0.5
        Q[1:-1,0]+=push[1:-1]*gx[1:-1]*0.02/0.02*0.02; Q[1:-1,1]+=push[1:-1]*gy[1:-1]*0.02
        # never move into lower clearance than the hard limit
        e2=E[torch.from_numpy(np.clip((Q[:,1]*100).astype(int),0,world.H-1)).to(world.dev),torch.from_numpy(np.clip((Q[:,0]*100).astype(int),0,world.W-1)).to(world.dev)].float().cpu().numpy()
        ok=(e2>=np.minimum(e,HARD_CLR+0.005))
        P[ok]=Q[ok]
    P=resample(P,0.01)
    tang=np.gradient(P,axis=0); th=np.unwrap(np.arctan2(tang[:,1],tang[:,0]))
    th[0]=th[min(5,len(th)-1)]; ds=0.01
    kap=np.abs(np.gradient(th))/ds; kap=np.convolve(kap,np.ones(5)/5,'same')
    vcap=np.minimum(lim.v_max,np.minimum(np.sqrt(lim.a_lat/np.maximum(kap,1e-6)),lim.w_max/np.maximum(kap,1e-6)))
    vcap=np.minimum(vcap,(lim.v_wheel)/(1+kap*lim.track/2))
    v=vcap.copy(); v[0]=0; v[-1]=0
    for i in range(1,len(v)): v[i]=min(v[i],math.sqrt(v[i-1]**2+2*lim.a_max*ds))
    for i in range(len(v)-2,-1,-1): v[i]=min(v[i],math.sqrt(v[i+1]**2+2*lim.a_max*ds))
    # yaw-accel: limit change of w=v*kap between samples (dw/dt = dw/ds * v)
    for _ in range(3):
        w=v*kap
        for i in range(1,len(v)):
            dtt=ds/max(1e-3,(v[i]+v[i-1])/2)
            if abs(w[i]-w[i-1])>lim.alpha_max*dtt and kap[i]>1e-6: v[i]=min(v[i],(abs(w[i-1])+lim.alpha_max*dtt)/kap[i]); w[i]=v[i]*kap[i]
        for i in range(1,len(v)): v[i]=min(v[i],math.sqrt(v[i-1]**2+2*lim.a_max*ds))
        for i in range(len(v)-2,-1,-1): v[i]=min(v[i],math.sqrt(v[i+1]**2+2*lim.a_max*ds))
    t=np.r_[0,np.cumsum(ds/np.maximum((v[1:]+v[:-1])/2,1e-3))]
    # oriented-footprint check along the path
    fp=np.array(SIL12)
    return dict(xy=P,theta=th,v=v,t=t,kappa=kap,length=len(P)*ds)

class GraphPlanner:
    """CUDA-graph captured MPPI.plan (removes ~2000 kernel launches of Python overhead per replan)."""
    def __init__(s, mppi):
        s.m=mppi; s.state=torch.zeros(mppi.B,5,device=mppi.dev)
        st=torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        U0=mppi.U.clone()
        with torch.cuda.stream(st):
            for _ in range(3): mppi.plan(s.state)
        torch.cuda.current_stream().wait_stream(st)
        s.g=torch.cuda.CUDAGraph()
        mppi.U.copy_(U0); s.Ubuf=mppi.U
        with torch.cuda.graph(s.g):
            s.out=mppi.plan(s.state)
            s.Ubuf.copy_(mppi.U)       # keep nominal in the static buffer
        mppi.U=s.Ubuf
    def plan(s, state):
        s.state.copy_(state); s.g.replay(); return s.out
    def reset(s, mask):
        s.Ubuf[mask]=0

def wrap(a): return (a+math.pi)%(2*math.pi)-math.pi
@torch.no_grad()
def dock_control(st, goal, mode, dt, L=Limits, r_on=0.30, ctg=None):
    """Final-approach pose controller for diff drive (MPPI is poor at the last few cm of a nonholonomic
    parking manoeuvre). mode: (B,) long 0=mppi 1=approach 2=final spin. Returns (u (B,2), mode)."""
    x,y,th,v,w=[st[:,i] for i in range(5)]
    dx,dy=goal[:,0]-x,goal[:,1]-y; d=torch.hypot(dx,dy); eb=wrap(torch.atan2(dy,dx)-th)
    # v7: only back in for a tiny overshoot (<4 cm); otherwise turn in place to face the goal and drive in forward
    rev=(eb.abs()>math.pi/2)&(d<0.04); ebr=torch.where(rev,wrap(eb+math.pi),eb)
    ok=(d<r_on) if ctg is None else (d<r_on)&(ctg<r_on*1.6)
    mode=torch.where((mode==0)&ok,torch.ones_like(mode),mode)
    mode=torch.where((mode==1)&(d<0.015),torch.full_like(mode,2),mode)
    mode=torch.where((mode==2)&(d>0.04),torch.ones_like(mode),mode)
    mode=torch.where((mode>=1)&(d>r_on+0.15),torch.zeros_like(mode),mode)
    vmag=torch.minimum(torch.minimum(torch.full_like(d,0.3),torch.sqrt(2*L.a_max*0.5*d)),1.5*d)*torch.cos(ebr).clamp(min=0)**4
    vd=torch.where(rev,-torch.minimum(vmag,torch.full_like(vmag,-L.v_min)),vmag)
    wd=(3.0*ebr).clamp(-L.w_max,L.w_max)
    eh=wrap(goal[:,2]-th); spin=mode==2
    vd=torch.where(spin,torch.zeros_like(vd),vd); wd=torch.where(spin,(3.0*eh).clamp(-1.5,1.5)*torch.clamp(eh.abs()/0.05,max=1)+0*eh,wd)
    a=((vd-v)/dt).clamp(-L.a_max,L.a_max); al=((wd-w)/dt).clamp(-L.alpha_max,L.alpha_max)
    return torch.stack([a,al],1), mode

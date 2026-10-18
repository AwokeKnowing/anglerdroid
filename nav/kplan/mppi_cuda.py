"""Fused CUDA-C MPPI rollout (portable twin of mppi_triton.py; works where Triton doesn't, e.g. Jetson Orin).
Compiled once via torch.utils.cpp_extension.load_inline (needs nvcc + ninja). Same signature/semantics as mppi_triton.rollout."""
import os, torch
from torch.utils.cpp_extension import load_inline

_SRC = r'''
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>
struct LP { float v_max,v_min,a_brake,t_react,d_safe,d_full,v_near,d_near,a_lat,w_max,v_wheel,track,rev_w,head_w,r_circ; };
__device__ __forceinline__ float clearf(float x,float y,float th,const float* fp,int P,const __half* e,int Hm,int Wm){
  float c=cosf(th), sn=sinf(th), m=1e9f;
  for(int p=0;p<P;p++){ float fx=fp[2*p], fy=fp[2*p+1];
    float px=x+c*fx-sn*fy, py=y+sn*fx+c*fy;
    int ix=min(max((int)(px*100.0f),0),Wm-1), iy=min(max((int)(py*100.0f),0),Hm-1);
    m=fminf(m,__half2float(e[(long long)iy*Wm+ix])); }
  return m; }
__device__ __forceinline__ float vcap(float d,const LP&L){
  float xx=fmaxf(d-L.d_safe,0.f);
  float hard=L.a_brake*(-L.t_react+sqrtf(L.t_react*L.t_react+2*xx/L.a_brake));
  float r=fminf(fmaxf((d-L.d_near)/(L.d_full-L.d_near),0.f),1.f);
  float ramp=L.v_near+(L.v_max-L.v_near)*r;
  return fminf(fminf(hard,ramp),L.v_max); }
// Stopping envelope (core.stop_reach): footprint at 1/3, 2/3, 3/3 of the emergency-stop arc from (x,y,th,v,w).
// Returns the min clearance there; *g = distance the farthest footprint point travels during the stop.
__device__ __forceinline__ float stopclear(float x,float y,float th,float v,float w,const float* fp,int P,const __half* e,int Hm,int Wm,const LP&L,float* g){
  float Tb=(fabsf(v)+fabsf(w)*L.track*0.5f)/L.a_brake, k=L.t_react+0.5f*Tb;
  float s=v*k, ts=w*k; *g=fabsf(s)+fabsf(ts)*L.r_circ;
  float m=1e9f;
  for(int i=1;i<=3;i++){ float f=i/3.f, dth=ts*f, px,py;
    if(fabsf(ts)>1e-4f){ float R=s/ts; px=x+R*(sinf(th+dth)-sinf(th)); py=y-R*(cosf(th+dth)-cosf(th)); }
    else { px=x+s*f*cosf(th); py=y+s*f*sinf(th); }
    m=fminf(m,clearf(px,py,th+dth,fp,P,e,Hm,Wm)); }
  return m; }
#define APPROACH_D 0.6f
#define R_ROTATE @R_ROTATE@f
#define R_PASS @R_PASS@f
#define APPROACH_VMIN 0.08f
__global__ void k_roll(const float* U,const float* st,const float* goal,const float* fp,int P,const __half* edt,const __half* erv,long long es,int Hm,int Wm,
  float* co,float* deo,float* tx,int K,int H,int TXC,float dt,LP L){
  int b=blockIdx.y, k=blockIdx.x*blockDim.x+threadIdx.x; if(k>=K) return;
  const float* s=st+b*5; float x=s[0],y=s[1],th=s[2],v=s[3],w=s[4];
  float gx=goal[b*3],gy=goal[b*3+1],gth=goal[b*3+2];
  const __half* e=edt+(long long)b*es;
  const __half* er=erv+(long long)b*es;   // clearance to obstacles OR not-currently-observed-free space: used while reversing
  float cost=0,dead=0,usq=0;
  float clr=clearf(x,y,th,fp,P,e,Hm,Wm), clr0=clr;
  const float* ub=U+((long long)b*K+k)*(H*2); float* tb=tx+((long long)b*K+k)*(H*TXC);
  for(int t=0;t<H;t++){
    float a=ub[2*t], al=ub[2*t+1]; usq+=a*a+al*al;
    // approach velocity scaling (Nav2 RPP use_approach_linear_velocity_scaling): slow down near the goal, so a goal
    // just behind/beside is reached by turning, not by a fast loop or a reverse
    // approach scaling near the goal only; obstacle clearance is the stopping-envelope rule below (no isotropic ramp)
    float vc=fmaxf(APPROACH_VMIN,L.v_max*sqrtf((x-gx)*(x-gx)+(y-gy)*(y-gy))/APPROACH_D), v0=v, w0=w;
    v=fminf(fmaxf(v+a*dt,L.v_min),L.v_max);
    float lim=fmaxf(vc,fabsf(v0)-L.a_brake*dt);
    v=fmaxf(fminf(v,lim),-fminf(lim,-L.v_min));
    float av=fabsf(v);
    float wl=fminf(L.w_max,L.a_lat/fmaxf(av,1e-3f));
    wl=fminf(wl,fmaxf((L.v_wheel-av)/(L.track/2),0.f));
    w=fmaxf(fminf(w+al*dt,wl),-wl);
    { // planner-owned clearance: the next state must be able to stop with gap; else brake (wheels at a_brake)
      float nth=th+w*dt, nx=x+v*cosf(nth)*dt, ny=y+v*sinf(nth)*dt, g;
      float cs=stopclear(nx,ny,nth,v,w,fp,P,e,Hm,Wm,L,&g);   // obstacles only (not-observed space is the reverse check below)
      if(cs<fminf(L.d_safe+g,clr)){            // already closer than d_safe: allowed only if it does not get closer
        float ab=L.a_brake*dt, wb=2.f*L.a_brake/L.track*dt;
        v=copysignf(fmaxf(fabsf(v0)-ab,0.f),v0); w=copysignf(fmaxf(fabsf(w0)-wb,0.f),w0); av=fabsf(v); } }
    th=th+w*dt; x=x+v*cosf(th)*dt; y=y+v*sinf(th)*dt;
    tb[t*TXC]=x; tb[t*TXC+1]=y; tb[t*TXC+2]=th;
    if(TXC==6){ tb[t*TXC+3]=v; tb[t*TXC+4]=w; tb[t*TXC+5]=vc; }
    clr=clearf(x,y,th,fp,P,e,Hm,Wm);                       // clearance to obstacles
    float crv=v<0.f?clearf(x,y,th,fp,P,er,Hm,Wm):clr;     // reversing: also stay inside space observed free
    // infeasible inside the safety margin (+1 cell): the planner never plans where nav.safety would have to act.
    // Starting inside it (escape), only motions that do not get closer than the start (-5 mm, 1/2 cell) are allowed.
    if(crv<0.012f || clr<fminf(L.d_safe,clr0-0.005f)) dead=1.f;
    int ix=min(max((int)(x*100.0f),0),Wm-1), iy=min(max((int)(y*100.0f),0),Hm-1);
    float ax=__half2float(e[(long long)iy*Wm+ix]);
    cost+=dead*50+20*fmaxf(0.06f-crv,0.f);   // cost terms unchanged (clearance in the field the step uses)
    cost+=dt*(expf(-crv/0.15f)+2.0f*fminf(fmaxf(R_ROTATE-ax,0.f)/(R_ROTATE-R_PASS),1.f));   // not-turnable (twin_geometry)
    float dx=x-gx, dy=y-gy, nr=expf(-sqrtf(dx*dx+dy*dy)/0.25f);
    // reverse is for escapes: cost grows with clearance (free space -> turn and drive forward)
    float cr=fminf(fmaxf((crv-0.05f)/0.20f,0.f),1.f), dg=sqrtf(dx*dx+dy*dy);
    cost+=0.1f*dt*20*nr*(1-cosf(th-gth))*2+(0.5f+4.0f*nr+L.rev_w*cr*(1-nr))*dt*(v<0?1.f:0.f)*av/0.2f;
    // heading alignment: face the carrot (path tangent direction) when away from the goal and there is room
    if(dg>0.35f) cost+=L.head_w*dt*cr*(1-nr)*(1-cosf(th-atan2f(-dy,-dx))); }
  float dx=x-gx, dy=y-gy, nT=expf(-sqrtf(dx*dx+dy*dy)/0.15f);
  cost+=3*nT*(1-cosf(th-gth))+2*nT*(fabsf(v)+fabsf(w)*0.3f)+0.002f*usq;
  co[(long long)b*K+k]=cost; deo[(long long)b*K+k]=dead; }
void roll(torch::Tensor U,torch::Tensor st,torch::Tensor goal,torch::Tensor fp,torch::Tensor edt,torch::Tensor erv,int64_t es,int64_t Hm,int64_t Wm,
  torch::Tensor co,torch::Tensor de,torch::Tensor tx,int64_t TXC,double dt,std::vector<double> l){
  int B=U.size(0),K=U.size(1),H=U.size(2);
  LP L{(float)l[0],(float)l[1],(float)l[2],(float)l[3],(float)l[4],(float)l[5],(float)l[6],(float)l[7],(float)l[8],(float)l[9],(float)l[10],(float)l[11],(float)l[12],(float)l[13],(float)l[14]};
  int T=K>=64?64:32; dim3 g((K+T-1)/T,B);
  k_roll<<<g,T,0,c10::cuda::getCurrentCUDAStream()>>>(U.data_ptr<float>(),st.data_ptr<float>(),goal.data_ptr<float>(),fp.data_ptr<float>(),(int)fp.size(0),
    (const __half*)edt.data_ptr<at::Half>(),(const __half*)erv.data_ptr<at::Half>(),es,(int)Hm,(int)Wm,co.data_ptr<float>(),de.data_ptr<float>(),tx.data_ptr<float>(),K,H,(int)TXC,(float)dt,L); }
'''
import sys as _s, os as _o; _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), '..', '..', 'src') in _s.path or _s.path.append(_o.path.join(_o.path.dirname(_o.path.abspath(__file__)), '..', '..', 'src'))   # twin_geometry (single source of the footprint radii)
import twin_geometry as _TG
_SRC = _SRC.replace('@R_ROTATE@', '%.4f' % _TG.R_ROTATE).replace('@R_PASS@', '%.4f' % _TG.R_PASS)
_ext = load_inline('kplan_mppi_cuda_v9', cpp_sources='void roll(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,double,std::vector<double>);',
                   cuda_sources=_SRC, functions=['roll'], extra_cuda_cflags=['-O3'], verbose=False)

def rollout(U, state, goal, fp, edt, L, dt, txc=3, cost=None, dead=None, tx=None, erev=None):
    """erev: clearance field used for steps with v < 0 (obstacles + space not currently observed free); None -> edt."""
    B, K, H, _ = U.shape
    if erev is None: erev = edt
    if cost is None: cost = torch.empty(B, K, device=U.device)
    if dead is None: dead = torch.empty(B, K, device=U.device)
    if tx is None: tx = torch.empty(B, K, H, txc, device=U.device)
    assert edt.dtype == torch.float16 and erev.dtype == torch.float16 and erev.shape == edt.shape and U.dtype == torch.float32
    Hm, Wm = edt.shape[-2:]; est = 0 if edt.dim() == 2 else Hm*Wm
    _ext.roll(U.contiguous(), state.float().contiguous(), goal.float().contiguous(), fp.float().contiguous(), edt.contiguous(), erev.contiguous(), est, Hm, Wm, cost, dead, tx, txc, float(dt),
              [float(L.v_max), float(L.v_min), float(L.a_brake), float(L.t_react), float(L.d_safe), float(L.d_full), float(L.v_near), float(L.d_near),
               float(L.a_lat), float(L.w_max), float(L.v_wheel), float(L.track), float(getattr(L, "rev_w", 4.0)), float(getattr(L, "head_w", 1.5)), float(getattr(L, "r_circ", 0.274))])
    return cost, dead, tx

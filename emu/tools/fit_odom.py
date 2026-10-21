# Fit Thrun motion-model alphas (probabilistic robotics 5.4) from recorded sim odometry vs ground truth.
# per window: var(rot err) = a1 rot^2 + a2 trans^2 ; var(trans err) = a3 trans^2 + a4 rot^2
import json,glob,math,numpy as np,sys
W=lambda a:(a+math.pi)%(2*math.pi)-math.pi
def rel(p,q):
  c,s=math.cos(p[2]),math.sin(p[2]); dx,dy=q[0]-p[0],q[1]-p[1]
  return np.array([c*dx+s*dy,-s*dx+c*dy,W(q[2]-p[2])])
R=[];T=[]
for f in [g for p in sys.argv[1:] for g in glob.glob(p)]:
  tr=json.load(open(f))["track"]
  for a,b in zip(tr[::3],tr[3::3]):
    g=rel(a["gt"],b["gt"]); o=rel(a["odom"],b["odom"])
    t=math.hypot(o[0],o[1]); r=abs(o[2])
    if t<1e-3 and r<1e-3: continue
    e=g-o; R.append((r*r,t*t,e[2]**2)); T.append((t*t,r*r,e[0]**2+e[1]**2))
R=np.array(R);T=np.array(T)
a12=np.linalg.lstsq(R[:,:2],R[:,2],rcond=None)[0]; a34=np.linalg.lstsq(T[:,:2],T[:,2]/2,rcond=None)[0]
a=np.maximum(np.r_[a12,a34],1e-8); print(json.dumps({"alpha":a.tolist(),"n":len(R),"src":sys.argv[1:]}))

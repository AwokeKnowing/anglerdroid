import sys,pickle,json,math,time,numpy as np
sys.path.insert(0,__import__('os').path.abspath(__import__('os').path.join(__import__('os').path.dirname(__file__), '..', '..')))
import nav
from nav.slam import GridMap,Matcher,compose,inverse,wrap
rec=pickle.load(open(sys.argv[1],'rb')); tr=json.load(open(sys.argv[2]))['track']
tg=np.array([r['t'] for r in tr]); G=np.array([r['gt'] for r in tr]); G[:,2]=np.unwrap(G[:,2])
print('rec',len(rec),'t range',rec[0][0],rec[-1][0],'gt t',tg[0],tg[-1])
def gti(t): return np.array([np.interp(t,tg,G[:,i]) for i in range(3)])
def run(match=True,half=0.5,res=0.05,thr=0.01,sigma=0.06,lin=0.12,minsc=0.35,npts=500,maxr=9.0,warm=0):
    g=GridMap(res=res); M=Matcher(g,lin=lin); T=(0,0,0); last=None; E=[];ms=[]
    for k,(t,po,obs,free,mp) in enumerate(rec):
        pm=compose(T,po)
        moved=last is None or math.hypot(po[0]-last[0],po[1]-last[1])>0.04 or abs(wrap(po[2]-last[2]))>math.radians(2.5)
        if not moved: continue
        if match and g.version>warm and len(mp)>40:
            mp=mp[np.hypot(mp[:,0],mp[:,1])<maxr]
            P=mp[np.random.default_rng(k).choice(len(mp),npts,replace=False)] if len(mp)>npts else mp
            t0=time.perf_counter(); est,sc,med,s0=M.match(P,pm); ms.append(time.perf_counter()-t0)
            if sc>minsc and sc>med+0.04 and sc>(s0 or 0)+thr:
                d=(est[0]-pm[0],est[1]-pm[1],wrap(est[2]-pm[2])); pm=(pm[0]+half*d[0],pm[1]+half*d[1],wrap(pm[2]+half*d[2])); T=compose(pm,inverse(po))
        g.integrate(pm,obs,free); last=po
        if k%2==0: g.refresh_likelihood(sigma)
        if tg[0]<=t<=tg[-1]:
            q=gti(t); E.append((math.hypot(pm[0]-q[0],pm[1]-q[1]),abs(wrap(pm[2]-q[2]))))
    E=np.array(E); return dict(xy_mean=E[:,0].mean().round(3),xy_max=E[:,0].max().round(3),xy_final=E[-1,0].round(3),yaw_max=round(math.degrees(E[:,1].max()),1),match_ms=round(1e3*np.mean(ms),1) if ms else 0)
B=dict(res=0.02,sigma=0.03,half=1.0)
for name,kw in [('r2_full',{}),('np1000',dict(npts=1000)),('maxr3',dict(maxr=3.0)),('warm40',dict(warm=40)),('all',dict(npts=1000,maxr=3.0,warm=40)),('all_half',dict(npts=1000,maxr=3.0,warm=40,half=0.5))]:
    print(name,run(**dict(B,**kw)),flush=True)

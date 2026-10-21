// Weighted 8-connected A* (octile heuristic, admissible: cost>=1) + LOS string pulling on a coarse grid.
// Same edge cost as core.cost_to_go: h*len*(c_a+c_b)/2. Build: gcc -O3 -shared -fPIC -o libfastpath.so fastpath_astar.c
#include <stdlib.h>
#include <math.h>
#include <string.h>
typedef struct {float f; int i;} HN;
static HN* hp; static int hn, hcap;
static void push(float f,int i){ if(hn==hcap){hcap=hcap?hcap*2:65536; hp=realloc(hp,sizeof(HN)*hcap);} int k=hn++; while(k){int p=(k-1)/2; if(hp[p].f<=f)break; hp[k]=hp[p]; k=p;} hp[k].f=f; hp[k].i=i;}
static HN pop(void){ HN r=hp[0]; HN x=hp[--hn]; int k=0; for(;;){int c=2*k+1; if(c>=hn)break; if(c+1<hn&&hp[c+1].f<hp[c].f)c++; if(hp[c].f>=x.f)break; hp[k]=hp[c]; k=c;} hp[k]=x; return r;}
static float *G; static int *PAR; static unsigned *STAMP,*CLOSED; static unsigned cur=0; static int NN=0;
// returns path length in cells written to out (rc pairs, start..goal), -1 if unreachable, -2 if out too small
int astar_m(const float* cost,const unsigned char* blk,int R,int C,int sr,int sc,int gr,int gc,float eps,int* out,int maxout,int* nexp,const unsigned char* allow,int f,int Cc);
int astar(const float* cost,const unsigned char* blk,int R,int C,int sr,int sc,int gr,int gc,float eps,int* out,int maxout,int* nexp){return astar_m(cost,blk,R,C,sr,sc,gr,gc,eps,out,maxout,nexp,0,1,0);}
// allow: optional coarse corridor mask (factor f, Cc coarse cols); cells outside it are treated as blocked
int astar_m(const float* cost,const unsigned char* blk,int R,int C,int sr,int sc,int gr,int gc,float eps,int* out,int maxout,int* nexp,const unsigned char* allow,int f,int Cc){
  int N=R*C; if(N>NN){free(G);free(PAR);free(STAMP);free(CLOSED);G=malloc(4*N);PAR=malloc(4*N);STAMP=calloc(N,4);CLOSED=calloc(N,4);NN=N;cur=0;}
  cur++; hn=0; int s=sr*C+sc, g=gr*C+gc; if(blk[s]||blk[g]) return -1;
  static const int dr[8]={-1,-1,-1,0,0,1,1,1}, dc[8]={-1,0,1,-1,1,-1,0,1};
  const float SQ=1.41421356f;
  #define H(r,c) ({int a=abs((r)-gr),b=abs((c)-gc); float mn=a<b?a:b, mx=a<b?b:a; eps*((mx-mn)+SQ*mn);})
  STAMP[s]=cur; G[s]=0; PAR[s]=-1; push(H(sr,sc),s); int ne=0;
  while(hn){ HN t=pop(); int u=t.i; int ur=u/C, uc=u%C; float gu=G[u];
    if(CLOSED[u]==cur) continue; CLOSED[u]=cur; ne++;   // no reopening (weighted A*)
    if(u==g) break;
    for(int k=0;k<8;k++){int r=ur+dr[k],c=uc+dc[k]; if(r<0||c<0||r>=R||c>=C)continue; int v=r*C+c; if(blk[v]||CLOSED[v]==cur)continue; if(allow&&!allow[(r/f)*Cc+c/f])continue;
      if(dr[k]&&dc[k]&&(blk[ur*C+c]||blk[r*C+uc]))continue; // no corner cutting
      float w=(dr[k]&&dc[k]?SQ:1.f)*0.5f*(cost[u]+cost[v]); float gv=gu+w;
      if(STAMP[v]!=cur||gv<G[v]){STAMP[v]=cur;G[v]=gv;PAR[v]=u;push(gv+H(r,c),v);} } }
  *nexp=ne; if(STAMP[g]!=cur) return -1;
  int n=0; for(int v=g;v!=-1;v=PAR[v]) n++; if(n>maxout) return -2;
  int i=n-1; for(int v=g;v!=-1;v=PAR[v]){out[2*i]=v/C; out[2*i+1]=v%C; i--;} return n;
}
// line cost between cells (supercover-ish sampling at 0.5 cell); returns -1 if any blocked
static float linecost(const float* cost,const unsigned char* blk,int C,float r0,float c0,float r1,float c1){
  float dr=r1-r0, dc=c1-c0, L=sqrtf(dr*dr+dc*dc); int n=(int)ceilf(L*2)+1; float acc=0, prev=-1;
  for(int i=0;i<=n;i++){ float t=(float)i/n; int r=(int)floorf(r0+dr*t+0.5f), c=(int)floorf(c0+dc*t+0.5f); int v=r*C+c; if(blk[v]) return -1.f; float cv=cost[v]; if(i) acc+=0.5f*(cv+prev)*(L/n); prev=cv; }
  return acc;
}
// greedy string pulling: from anchor i, jump to furthest j whose straight segment is free and costs <= (1+tol)*grid path cost.
int stringpull(const float* cost,const unsigned char* blk,int C,const int* p,int n,float tol,int win,int gallop,int* keep){
  float* cum=malloc(4*n); cum[0]=0; for(int i=1;i<n;i++){int a=p[2*i-2]*C+p[2*i-1], b=p[2*i]*C+p[2*i+1]; float d=(p[2*i]!=p[2*i-2]&&p[2*i+1]!=p[2*i-1])?1.41421356f:1.f; cum[i]=cum[i-1]+d*0.5f*(cost[a]+cost[b]);}
  int m=0,i=0; keep[m++]=0;
  while(i<n-1){ int best=i+1; int lo=i+1;
    // exponential then linear scan from the far end of a window
    int jmax=i+win<n-1?i+win:n-1;
    #define VIS(j) ({float lc_=linecost(cost,blk,C,p[2*i],p[2*i+1],p[2*(j)],p[2*(j)+1]); lc_>=0 && lc_<=(1+tol)*(cum[j]-cum[i])+1e-3f;})
    if(gallop){ int good=i+1, bad=-1, d=2;               // galloping + bisection (assumes visibility ~monotone along the path)
      while(1){ int j=i+d; if(j>jmax){ j=jmax; if(j>good && VIS(j)) good=j; else if(j>good) bad=j; break;} if(VIS(j)){good=j; d*=2;} else {bad=j; break;} }
      if(bad>0) while(bad-good>1){ int mid=(good+bad)/2; if(VIS(mid)) good=mid; else bad=mid; }
      best=good; }
    else for(int j=jmax;j>i+1;j--){ if(VIS(j)){best=j;break;} }
    (void)lo; keep[m++]=best; i=best; }
  free(cum); return m;
}
// elastic band on a polyline (x,y metres, n pts, endpoints fixed). E = clearance in cm (uint8, capped), 1 cm px.
static inline float ev(const unsigned char* E,int H,int W,float x,float y){int ix=(int)(x*100),iy=(int)(y*100); if(ix<2)ix=2; if(iy<2)iy=2; if(ix>W-3)ix=W-3; if(iy>H-3)iy=H-3; return E[iy*W+ix]*0.01f;}
void band(float* P,int n,const unsigned char* E,int H,int W,int iters,float pref,float rmin,float kl,float kp){
  float* Q=malloc(8*n);
  for(int it=0;it<iters;it++){
    for(int i=1;i<n-1;i++){ float x=P[2*i],y=P[2*i+1];
      float qx=x+kl*(P[2*i-2]+P[2*i+2]-2*x), qy=y+kl*(P[2*i-1]+P[2*i+3]-2*y);
      int ix=(int)(qx*100),iy=(int)(qy*100); if(ix<2)ix=2; if(iy<2)iy=2; if(ix>W-3)ix=W-3; if(iy>H-3)iy=H-3;
      float e=E[iy*W+ix]*0.01f, gx=(E[iy*W+ix+2]-(float)E[iy*W+ix-2])*0.01f/0.04f, gy=(E[(iy+2)*W+ix]-(float)E[(iy-2)*W+ix])*0.01f/0.04f;
      float pu=pref-e; if(pu<0)pu=0; pu*=kp; qx+=pu*gx; qy+=pu*gy;
      float e0=ev(E,H,W,x,y), e1=ev(E,H,W,qx,qy); float lim=e0<rmin?e0:rmin;
      if(e1>=lim){Q[2*i]=qx;Q[2*i+1]=qy;} else {Q[2*i]=x;Q[2*i+1]=y;} }
    for(int i=1;i<n-1;i++){P[2*i]=Q[2*i];P[2*i+1]=Q[2*i+1];} }
  free(Q);
}
// coarse A* with fine-connectivity edge masks: ex[r*C+c]=1 if (r,c)<->(r,c+1) connected, ey[r*C+c] for (r,c)<->(r+1,c)
static int okE(const unsigned char* ex,const unsigned char* ey,int C,int r0,int c0,int r1,int c1){
  if(r0==r1){int c=c0<c1?c0:c1; return ex[r0*C+c];} int r=r0<r1?r0:r1; return ey[r*C+c0];}
int astar_e(const float* cost,const unsigned char* blk,const unsigned char* ex,const unsigned char* ey,int R,int C,int sr,int sc,int gr,int gc,float eps,int* out,int maxout,int* nexp){
  int N=R*C; if(N>NN){free(G);free(PAR);free(STAMP);free(CLOSED);G=malloc(4*N);PAR=malloc(4*N);STAMP=calloc(N,4);CLOSED=calloc(N,4);NN=N;cur=0;}
  cur++; hn=0; int s=sr*C+sc, g=gr*C+gc; if(blk[s]||blk[g]) return -1;
  static const int dr[8]={-1,-1,-1,0,0,1,1,1}, dc[8]={-1,0,1,-1,1,-1,0,1};
  const float SQ=1.41421356f;
  #define H2(r,c) ({int a=abs((r)-gr),b=abs((c)-gc); float mn=a<b?a:b, mx=a<b?b:a; eps*((mx-mn)+SQ*mn);})
  STAMP[s]=cur; G[s]=0; PAR[s]=-1; push(H2(sr,sc),s); int ne=0;
  while(hn){ HN t=pop(); int u=t.i; int ur=u/C, uc=u%C; float gu=G[u];
    if(CLOSED[u]==cur) continue; CLOSED[u]=cur; ne++; if(u==g) break;
    for(int k=0;k<8;k++){int r=ur+dr[k],c=uc+dc[k]; if(r<0||c<0||r>=R||c>=C)continue; int v=r*C+c; if(blk[v]||CLOSED[v]==cur)continue;
      if(dr[k]&&dc[k]){ int a1=!blk[ur*C+c]&&okE(ex,ey,C,ur,uc,ur,c)&&okE(ex,ey,C,ur,c,r,c), a2=!blk[r*C+uc]&&okE(ex,ey,C,ur,uc,r,uc)&&okE(ex,ey,C,r,uc,r,c); if(!(a1&&a2))continue; }
      else if(!okE(ex,ey,C,ur,uc,r,c)) continue;
      float w=(dr[k]&&dc[k]?SQ:1.f)*0.5f*(cost[u]+cost[v]); float gv=gu+w;
      if(STAMP[v]!=cur||gv<G[v]){STAMP[v]=cur;G[v]=gv;PAR[v]=u;push(gv+H2(r,c),v);} } }
  *nexp=ne; if(STAMP[g]!=cur) return -1;
  int n=0; for(int v=g;v!=-1;v=PAR[v]) n++; if(n>maxout) return -2;
  int i=n-1; for(int v=g;v!=-1;v=PAR[v]){out[2*i]=v/C; out[2*i+1]=v%C; i--;} return n;
}

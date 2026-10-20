"""Final DS02 de-anonymisation fit (balanced objective, km units)."""
import json, os
import numpy as np, pandas as pd
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# ---- reference anchors (OpenCelliD mcc=427 net=2 = Vodafone Qatar) ----------
QX = [(0.01,51.2217),(0.05,51.3714),(0.25,51.4420),(0.50,51.4891),(0.75,51.5246),(0.95,51.5843),(0.99,51.6088)]
QY = [(0.01,24.9809),(0.05,25.1374),(0.25,25.2267),(0.50,25.2739),(0.75,25.3269),(0.95,25.5865),(0.99,25.8831)]
KX = 111.0*np.cos(np.radians(25.4)); KY = 111.0
REF_SX_KM, REF_SY_KM, REF_RHO = 0.07715*KX, 0.1447*KY, -0.1636
REF_LON = (50.7575, 51.6545); REF_LAT = (24.5606, 26.1351)
LAC_REF = {
 110:(51.4484,25.7240),111:(51.4454,25.1789),120:(51.5072,25.3478),121:(51.5284,25.2949),
 130:(51.2836,25.3499),131:(51.4322,25.2077),141:(51.5572,25.2512),142:(51.4821,25.2525),
 143:(51.5159,25.2447),150:(51.4545,25.3203),151:(51.3831,25.2053),160:(51.4516,25.4211),
 161:(51.4907,25.3090),171:(51.5445,25.1284),181:(51.4357,25.2714),1110:(51.4494,25.7364),
 1130:(51.3631,25.3593),1151:(51.3694,25.2099),1160:(51.4443,25.4368),1171:(51.5536,25.1281),
}

def load_rings(path):
    gj = json.load(open(path, encoding="utf-8")); out=[]
    for f in gj["features"]:
        g=f["geometry"]; polys=g["coordinates"] if g["type"]=="MultiPolygon" else [g["coordinates"]]
        for p in polys: out.append(np.array(p[0],float))
    return out
RING = max(load_rings(os.path.join(HERE,"qatar_land.geojson")), key=len)

res=0.0005; LON0,LON1,LAT0,LAT1 = 50.55,51.85,24.35,26.35
W=int(round((LON1-LON0)/res))+1; H=int(round((LAT1-LAT0)/res))+1
img=Image.new("L",(W,H),0)
ImageDraw.Draw(img).polygon([((x-LON0)/res,(LAT1-y)/res) for x,y in RING], fill=255)
land=np.array(img)>0
def off_land_km(lon,lat):
    ix=np.clip(((lon-LON0)/res).round().astype(int),0,W-1)
    iy=np.clip(((LAT1-lat)/res).round().astype(int),0,H-1)
    bad=~land[iy,ix]
    if not bad.any(): return np.zeros(0)
    P=np.c_[lon[bad],lat[bad]]
    d=np.min(np.hypot(P[:,0][:,None]-RING[None,:,0],P[:,1][:,None]-RING[None,:,1]),axis=1)*111.0
    return d

# ---- data ------------------------------------------------------------------
df=pd.read_csv(os.path.join(ROOT,"ANOM_DS02_SiteLocations.csv"))
df["LAT_T"]=np.concatenate([df.LAT.values[1:],[np.nan]])
df["LON_T"]=np.concatenate([df.LON.values[1:],[np.nan]])
d=df.dropna(subset=["LAT_T"])
sites=d.groupby("ANONYMIZED_SITECODE").agg(LON=("LON_T","median"),LAT=("LAT_T","median"),LAC=("LAC_TAC","first"))
A=sites[["LON","LAT"]].values; lac_i=np.array([int(x) for x in sites.LAC.values])
N=len(A)
terms=[(lac_i==l, np.array(v), (lac_i==l).sum()) for l,v in LAC_REF.items() if (lac_i==l).sum()>0]

def parts(p):
    B=p[:4].reshape(2,2); c=p[4:]
    m=A@B.T+c; lon,lat=m[:,0],m[:,1]
    lac=[]; tot_lac=0.0
    for sel,ref,n in terms:
        dm=(m[sel].mean(axis=0)-ref)*np.array([KX,KY]); e=float(np.hypot(*dm))
        lac.append((int(lac_i[sel][0]),e)); tot_lac += (n/N)*min(e,25.0)**2
    qx=np.quantile(lon,[q for q,_ in QX]); qy=np.quantile(lat,[q for q,_ in QY])
    tq=float(np.mean(np.r_[((qx-[v for _,v in QX])*KX)**2, ((qy-[v for _,v in QY])*KY)**2]))
    ts=float((lon.std()*KX-REF_SX_KM)**2 + (lat.std()*KY-REF_SY_KM)**2)
    rho=float(np.corrcoef(lon,lat)[0,1]); tr=50.0*(rho-REF_RHO)**2
    d_off=off_land_km(lon,lat)
    tl=float(len(d_off)/N*np.mean(np.minimum(d_off,20.0)**2)) if len(d_off) else 0.0
    viol=np.maximum(lon-REF_LON[1]-0.005,0)+np.maximum(REF_LON[0]-0.005-lon,0) \
         +np.maximum(lat-REF_LAT[1]-0.005,0)+np.maximum(REF_LAT[0]-0.005-lat,0)
    tbox=float(np.mean(np.minimum(viol,0.15)**2))*2e4   # robust: capped per-point violation
    return dict(lac=tot_lac, q=tq, s=ts, rho=tr, land=tl, box=tbox, m=m, lacres=lac, rho_v=rho, off=d_off)

WQ, WS, WL = 4.0, 2.0, 60.0
def loss(p):
    d=parts(p)
    return WQ*d["q"] + WS*d["s"] + d["rho"] + WL*d["land"] + d["box"] + d["lac"]

def nm(f,x0,step,iters=6000):
    n=len(x0); pts=[np.array(x0,float)]
    for i in range(n):
        q=np.array(x0,float); q[i]+=step[i]; pts.append(q)
    pts=np.array(pts); fs=np.array([f(q) for q in pts])
    for _ in range(iters):
        o=np.argsort(fs); pts,fs=pts[o],fs[o]
        if fs[-1]-fs[0] < 1e-13*(abs(fs[0])+1e-13): break
        cen=pts[:-1].mean(axis=0)
        xr=cen+(cen-pts[-1]); fr=f(xr)
        if fr<fs[0]:
            xe=cen+2*(xr-cen); fe=f(xe)
            pts[-1],fs[-1]=(xe,fe) if fe<fr else (xr,fr)
        elif fr<fs[-2]: pts[-1],fs[-1]=xr,fr
        else:
            xc=cen+0.5*(pts[-1]-cen); fc=f(xc)
            if fc<fs[-1]: pts[-1],fs[-1]=xc,fc
            else:
                pts[1:]=pts[0]+0.5*(pts[1:]-pts[0]); fs[1:]=np.array([f(q) for q in pts[1:]])
    i=int(np.argmin(fs)); return pts[i],fs[i]

# LS init on radio-median centroids
Ar=[];br=[]
for sel,ref,n in terms:
    w=np.sqrt(n); ma=A[sel].mean(axis=0)
    Ar.append(w*np.array([ma[0],ma[1],1,0,0,0])); br.append(w*ref[0])
    Ar.append(w*np.array([0,0,0,ma[0],ma[1],1])); br.append(w*ref[1])
xls,*_=np.linalg.lstsq(np.array(Ar),np.array(br),rcond=None)
starts=[xls]
for f in ("transform_lacfit.json","transform_final.json"):
    p=os.path.join(HERE,f)
    if os.path.exists(p):
        T=json.load(open(p)); starts.append(np.r_[np.array(T["B"]).ravel(), np.array(T["c"])])
rng=np.random.default_rng(1)
base=starts[0]
for s in list(starts):
    for _ in range(120):
        q=s.copy(); q[:4]+=rng.normal(0,0.03,4); q[4:]+=rng.normal(0,0.1,2); starts.append(q)
scored=sorted(((loss(p),p) for p in starts), key=lambda z:z[0])
print("start losses:", ["%.4f"%s for s,_ in scored[:6]])
best=scored[0]
for s0,p0 in scored[:15]:
    p,f=p0,s0
    for st in ([0.01]*4+[0.03]*2,[0.003]*4+[0.01]*2,[0.001]*4+[0.003]*2):
        p2,f2=nm(loss,p,np.array(st))
        if f2<f: p,f=p2,f2
    if f<best[0]: best=(f,p)
print("optimised loss %.5f" % best[0])

p=best[1]; B=p[:4].reshape(2,2); c=p[4:]
d=parts(p)
print("B=\n%s\nc=%s" % (B,c))
print("singular values", np.linalg.svd(B,compute_uv=False))
print("terms: lac %.2f q %.2f s %.2f rho %.2f land %.2f box %.2f" %
      (d["lac"],WQ*d["q"],WS*d["s"],d["rho"],WL*d["land"],d["box"]))
print("\nLAC centroid residuals (km): " + ", ".join("%d:%.1f"%(l,e) for l,e in sorted(d["lacres"],key=lambda z:-z[1])))
m=d["m"]
print("std lon %.2f km (ref %.2f)  std lat %.2f km (ref %.2f)  rho %.3f (ref %.3f)" %
      (m[:,0].std()*KX, REF_SX_KM, m[:,1].std()*KY, REF_SY_KM, d["rho_v"], REF_RHO))
print("lon %.4f..%.4f (ref %.4f..%.4f)  lat %.4f..%.4f (ref %.4f..%.4f)" %
      (m[:,0].min(),m[:,0].max(),*REF_LON,m[:,1].min(),m[:,1].max(),*REF_LAT))
print("off-land points: %d of %d, max %.1f km" % (len(d["off"]), N, d["off"].max() if len(d["off"]) else 0))
json.dump({"B":B.tolist(),"c":c.tolist(),"loss":float(best[0]),
           "KX":KX,"KY":KY,"reference":"OpenCelliD mcc=427 net=2 (Vodafone Qatar)"},
          open(os.path.join(HERE,"transform_final.json"),"w"), indent=1)
sites.assign(LON_REAL=m[:,0],LAT_REAL=m[:,1]).to_csv(os.path.join(HERE,"sites_final.csv"))
np.save(os.path.join(HERE,"mapped_final.npy"), m)
print("saved transform_final.json / sites_final.csv")

"""Compare affine vs projective (homography) inverse on the same DS02 anchors."""
import json, os
import numpy as np, pandas as pd
from PIL import Image, ImageDraw

HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
KX=111.0*np.cos(np.radians(25.4)); KY=111.0
QLEV=[0.10,0.25,0.50,0.75,0.90]
REFQ = {
 110:([51.3055,51.4130,51.4745,51.5040,51.5300],[25.5725,25.6615,25.6945,25.7780,25.9210]),
 111:([51.4210,51.4290,51.4430,51.4620,51.4710],[25.1610,25.1720,25.1800,25.1910,25.1960]),
 120:([51.4590,51.4810,51.5160,51.5280,51.5400],[25.3260,25.3310,25.3480,25.3640,25.3780]),
 121:([51.5150,51.5200,51.5280,51.5390,51.5460],[25.2770,25.2820,25.2880,25.3090,25.3180]),
 130:([51.0635,51.2410,51.3185,51.3795,51.4025],[25.3090,25.3175,25.3430,25.3785,25.4015]),
 131:([51.4100,51.4220,51.4330,51.4430,51.4520],[25.1920,25.1990,25.2090,25.2160,25.2200]),
 141:([51.5380,51.5430,51.5510,51.5650,51.5910],[25.2230,25.2380,25.2560,25.2660,25.2750]),
 142:([51.4520,51.4680,51.4860,51.4990,51.5070],[25.2230,25.2360,25.2520,25.2700,25.2800]),
 143:([51.4940,51.5070,51.5200,51.5260,51.5320],[25.2070,25.2250,25.2540,25.2650,25.2720]),
 150:([51.4350,51.4470,51.4535,51.4655,51.4735],[25.2980,25.3090,25.3195,25.3345,25.3465]),
 151:([51.3145,51.3650,51.3865,51.4310,51.4570],[25.1205,25.1935,25.2185,25.2440,25.2540]),
 160:([51.4100,51.4265,51.4485,51.4785,51.4975],[25.3595,25.3760,25.3985,25.4425,25.5260]),
 161:([51.4695,51.4795,51.4940,51.5040,51.5130],[25.2860,25.2910,25.3075,25.3235,25.3335]),
 171:([51.4650,51.5160,51.5545,51.5915,51.6025],[24.9955,25.1065,25.1595,25.1920,25.2205]),
 181:([51.4130,51.4210,51.4350,51.4490,51.4610],[25.2450,25.2580,25.2690,25.2880,25.3020]),
 1110:([51.4390,51.4700,51.5000,51.5360,51.5900],[25.5700,25.6650,25.7270,25.8630,25.8840]),
 1130:([51.3270,51.3340,51.3620,51.4030,51.4200],[25.2760,25.3160,25.3360,25.3850,25.4890]),
 1151:([51.3080,51.3470,51.3700,51.3950,51.4140],[25.1590,25.1760,25.1920,25.2260,25.2540]),
 1160:([51.3890,51.4160,51.4410,51.4850,51.4980],[25.3670,25.3960,25.4290,25.4680,25.5210]),
 1171:([51.4340,51.5190,51.5880,51.5990,51.6050],[25.0230,25.0730,25.1510,25.1720,25.2030]),
}
QX=[(0.01,51.2217),(0.05,51.3714),(0.25,51.4420),(0.50,51.4891),(0.75,51.5246),(0.95,51.5843),(0.99,51.6088)]
QY=[(0.01,24.9809),(0.05,25.1374),(0.25,25.2267),(0.50,25.2739),(0.75,25.3269),(0.95,25.5865),(0.99,25.8831)]
REF_SX_KM,REF_SY_KM,REF_RHO=0.07715*KX,0.1447*KY,-0.1636
REF_LON=(50.7575,51.6545); REF_LAT=(24.5606,26.1351)

def load_rings(p):
    gj=json.load(open(p,encoding="utf-8")); out=[]
    for f in gj["features"]:
        g=f["geometry"]; polys=g["coordinates"] if g["type"]=="MultiPolygon" else [g["coordinates"]]
        for q in polys: out.append(np.array(q[0],float))
    return out
RING=max(load_rings(os.path.join(HERE,"qatar_land.geojson")),key=len)
res=0.0005; LON0,LON1,LAT0,LAT1=50.55,51.85,24.35,26.35
W=int(round((LON1-LON0)/res))+1; H=int(round((LAT1-LAT0)/res))+1
im=Image.new("L",(W,H),0); ImageDraw.Draw(im).polygon([((x-LON0)/res,(LAT1-y)/res) for x,y in RING],fill=255)
land=np.array(im)>0
def off_km(lon,lat):
    ix=np.clip(((lon-LON0)/res).round().astype(int),0,W-1); iy=np.clip(((LAT1-lat)/res).round().astype(int),0,H-1)
    bad=~land[iy,ix]
    if not bad.any(): return np.zeros(0)
    P=np.c_[lon[bad],lat[bad]]
    return np.min(np.hypot(P[:,0][:,None]-RING[None,:,0],P[:,1][:,None]-RING[None,:,1]),axis=1)*111.0

df=pd.read_csv(os.path.join(ROOT,"ANOM_DS02_SiteLocations.csv"))
df["LAT_T"]=np.concatenate([df.LAT.values[1:],[np.nan]]); df["LON_T"]=np.concatenate([df.LON.values[1:],[np.nan]])
d=df.dropna(subset=["LAT_T"])
sites=d.groupby("ANONYMIZED_SITECODE").agg(LON=("LON_T","median"),LAT=("LAT_T","median"),LAC=("LAC_TAC","first"))
A=sites[["LON","LAT"]].values; LAC=np.array([int(x) for x in sites.LAC.values]); N=len(A)
terms=[(LAC==l,np.array(REFQ[l][0]),np.array(REFQ[l][1]),(LAC==l).sum()) for l in REFQ if (LAC==l).sum()>0]
A1=np.c_[A,np.ones(N)]

def map_pts(p,proj):
    if not proj:
        return np.c_[A, np.ones(N)] @ p[:6].reshape(2,3).T
    Hm=np.r_[p[:6].reshape(2,3),[[p[6],p[7],1.0]]]
    v=A1@Hm.T
    return v[:,:2]/v[:,2:3]

def parts(p,proj):
    m=map_pts(p,proj); lon,lat=m[:,0],m[:,1]
    lq=0.0; res=[]
    for sel,rx,ry,n in terms:
        px=np.quantile(lon[sel],QLEV); py=np.quantile(lat[sel],QLEV)
        e=float(np.mean(np.r_[((px-rx)*KX)**2,((py-ry)*KY)**2])); lq+=(n/N)*e
        res.append((int(LAC[sel][0]),float(np.sqrt(e))))
    gx=np.quantile(lon,[q for q,_ in QX]); gy=np.quantile(lat,[q for q,_ in QY])
    gq=float(np.mean(np.r_[((gx-[v for _,v in QX])*KX)**2,((gy-[v for _,v in QY])*KY)**2]))
    s=float((lon.std()*KX-REF_SX_KM)**2+(lat.std()*KY-REF_SY_KM)**2)
    rho=float(np.corrcoef(lon,lat)[0,1])
    do=off_km(lon,lat); lnd=float(len(do)/N*np.mean(np.minimum(do,25.0)**2)) if len(do) else 0.0
    viol=np.maximum(lon-REF_LON[1]-0.005,0)+np.maximum(REF_LON[0]-0.005-lon,0) \
        +np.maximum(lat-REF_LAT[1]-0.005,0)+np.maximum(REF_LAT[0]-0.005-lat,0)
    box=float(np.mean(np.minimum(viol,0.12)**2))*2e4
    return dict(lq=lq,gq=gq,s=s,rho=rho,lnd=lnd,box=box,m=m,res=res,off=do)

W_LQ,W_GQ,W_S,W_LND=3.0,1.0,2.0,40.0
def make_loss(proj):
    def loss(p):
        if proj:
            m=map_pts(p,True)
            if not np.all(np.isfinite(m)) or (np.abs(m[:,0])>20).any() or (np.abs(m[:,1])>20).any():
                return 1e9
        d=parts(p,proj)
        return W_LQ*d["lq"]+W_GQ*d["gq"]+W_S*d["s"]+50*(d["rho"]-REF_RHO)**2+W_LND*d["lnd"]+d["box"]
    return loss

def nm(f,x0,step,iters=20000):
    n=len(x0); pts=[np.array(x0,float)]
    for i in range(n):
        q=np.array(x0,float); q[i]+=step[i]; pts.append(q)
    pts=np.array(pts); fs=np.array([f(q) for q in pts])
    for _ in range(iters):
        o=np.argsort(fs); pts,fs=pts[o],fs[o]
        if fs[-1]-fs[0]<1e-13*(abs(fs[0])+1e-13): break
        cen=pts[:-1].mean(axis=0); xr=cen+(cen-pts[-1]); fr=f(xr)
        if fr<fs[0]:
            xe=cen+2*(xr-cen); fe=f(xe); pts[-1],fs[-1]=(xe,fe) if fe<fr else (xr,fr)
        elif fr<fs[-2]: pts[-1],fs[-1]=xr,fr
        else:
            xc=cen+0.5*(pts[-1]-cen); fc=f(xc)
            if fc<fs[-1]: pts[-1],fs[-1]=xc,fc
            else:
                pts[1:]=pts[0]+0.5*(pts[1:]-pts[0]); fs[1:]=np.array([f(q) for q in pts[1:]])
    i=int(np.argmin(fs)); return pts[i],fs[i]

T=json.load(open(os.path.join(HERE,"transform_final.json")))
B=np.array(T["B"]); c=np.array(T["c"])
p_aff=np.r_[B.ravel(),c]
_dbg=parts(p_aff,False)
print("DEBUG parts:", {k:(type(v).__name__, np.shape(v)) for k,v in _dbg.items() if k not in ("m","res","off")})
print("DEBUG loss:", f_aff(p_aff) if False else None)
f_aff=make_loss(False)
# affine polish
p_aff,fv_aff=nm(f_aff,p_aff,np.array([0.002]*4+[0.006]*2))
fv_aff=float(np.ravel(fv_aff)[0]); p_aff=np.ravel(p_aff)
p_proj=np.r_[p_aff,0.0,0.0]
f_proj=make_loss(True)
best=None
rng=np.random.default_rng(3)
for p0 in [p_proj]+[np.r_[p_aff,rng.normal(0,0.02,2)] for _ in range(60)]:
    p,f=nm(f_proj,p0,np.array([0.002]*4+[0.006]*2+[0.01]*2),iters=30000)
    if best is None:
        print("DBG first homography result:", type(f), np.shape(f), type(p), np.shape(p))
        print("DBG direct loss:", f_proj(p0), type(f_proj(p0)))
        print("DBG A1 defined:", 'A1' in dir())
    f=float(np.ravel(f)[0]); p=np.ravel(p)
    if best is None or f<best[0]: best=(f,p)
p_proj,fv_proj=best
print("TYPES", type(fv_aff), np.shape(fv_aff), type(fv_proj), np.shape(fv_proj))
print("affine loss %.4f   homography loss %.4f" % (fv_aff,fv_proj))
for nm_,p,proj in (("affine",p_aff,False),("homography",p_proj,True)):
    d=parts(p,proj); m=d["m"]
    print("\n%s: lacQ %.2f globQ %.2f std %.2f land %.2f off %d max %.1f km" %
          (nm_,W_LQ*d["lq"],W_GQ*d["gq"],W_S*d["s"],W_LND*d["lnd"],len(d["off"]),d["off"].max() if len(d["off"]) else 0))
    print("  per-LAC residual km: "+", ".join("%d:%.1f"%(l,e) for l,e in sorted(d["res"],key=lambda z:-z[1])))
    print("  std lon %.2f (ref %.2f) std lat %.2f (ref %.2f) rho %.3f" %
          (m[:,0].std()*KX,REF_SX_KM,m[:,1].std()*KY,REF_SY_KM,d["rho"]))
    print("  lon %.4f..%.4f lat %.4f..%.4f" % (m[:,0].min(),m[:,0].max(),m[:,1].min(),m[:,1].max()))
    print("  params:", np.round(p,5).tolist())
json.dump({"affine":{"p":p_aff.tolist(),"loss":float(fv_aff)},
           "homography":{"p":p_proj.tolist(),"loss":float(fv_proj)}},
          open(os.path.join(HERE,"fit_compare.json"),"w"),indent=1)

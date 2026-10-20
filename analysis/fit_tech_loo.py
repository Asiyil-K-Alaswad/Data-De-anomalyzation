"""Per-technology anchor refit + leave-one-LAC-out validation for DS02.

Anchors (OpenCelliD mcc=427 net=2, Vodafone Qatar), resolved by technology:
  * 4-digit DS02 codes (1110,1130,1151,1160,1171) are GSM areas  -> GSM envelopes
  * 3-digit DS02 codes (110..181) are UMTS/LTE areas             -> UMTS+LTE envelopes
  * 5-digit codes (10101,14101,15101) have no reference          -> unanchored
Model: real = B @ anon + c   (affine; homography compared at the end)
"""
import json, os, sys
import numpy as np, pandas as pd
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
KX = 111.0*np.cos(np.radians(25.4)); KY = 111.0
QLEV = [0.10, 0.25, 0.50, 0.75, 0.90]

# (n, lon q10..q90, lat q10..q90)
GSM = {
 111:(212,[51.421,51.429,51.439,51.453,51.469],[25.161,25.172,25.176,25.185,25.192]),
 120:(240,[51.461,51.500,51.522,51.532,51.534],[25.326,25.331,25.342,25.362,25.378]),
 121:(170,[51.515,51.520,51.527,51.537,51.546],[25.280,25.282,25.290,25.311,25.318]),
 131:(69,[51.406,51.422,51.433,51.439,51.451],[25.192,25.199,25.212,25.216,25.220]),
 141:(179,[51.538,51.543,51.553,51.565,51.585],[25.223,25.240,25.256,25.265,25.274]),
 142:(331,[51.453,51.468,51.485,51.499,51.507],[25.223,25.236,25.253,25.274,25.280]),
 143:(152,[51.502,51.511,51.521,51.526,51.532],[25.220,25.254,25.259,25.266,25.273]),
 150:(60,[51.439,51.448,51.454,51.467,51.473],[25.299,25.309,25.320,25.335,25.353]),
 161:(223,[51.471,51.482,51.495,51.506,51.514],[25.286,25.290,25.307,25.323,25.333]),
 181:(220,[51.414,51.421,51.437,51.450,51.461],[25.245,25.258,25.268,25.288,25.302]),
 1110:(168,[51.439,51.470,51.500,51.536,51.590],[25.570,25.665,25.727,25.863,25.884]),
 1130:(37,[51.327,51.334,51.362,51.403,51.420],[25.276,25.316,25.336,25.385,25.489]),
 1151:(85,[51.308,51.347,51.370,51.395,51.414],[25.159,25.176,25.192,25.226,25.254]),
 1160:(156,[51.389,51.416,51.441,51.485,51.498],[25.367,25.396,25.429,25.468,25.521]),
 1171:(210,[51.434,51.519,51.588,51.599,51.605],[25.023,25.073,25.151,25.172,25.203]),
}
UTL = {
 110:(384,[51.290,51.371,51.480,51.505,51.531],[25.558,25.665,25.694,25.779,26.036]),
 111:(677,[51.418,51.430,51.444,51.462,51.477],[25.162,25.171,25.181,25.194,25.206]),
 120:(1179,[51.458,51.478,51.513,51.527,51.543],[25.324,25.333,25.349,25.367,25.378]),
 121:(779,[51.512,51.519,51.528,51.539,51.547],[25.277,25.282,25.288,25.306,25.319]),
 130:(281,[50.888,51.222,51.342,51.384,51.404],[25.309,25.317,25.337,25.397,25.440]),
 131:(466,[51.411,51.423,51.433,51.445,51.460],[25.194,25.200,25.209,25.217,25.236]),
 141:(1179,[51.495,51.533,51.547,51.566,51.593],[25.222,25.237,25.254,25.268,25.276]),
 142:(1543,[51.446,51.466,51.487,51.502,51.516],[25.221,25.235,25.252,25.269,25.279]),
 143:(934,[51.486,51.502,51.517,51.526,51.533],[25.201,25.222,25.254,25.265,25.273]),
 150:(423,[51.429,51.442,51.452,51.463,51.473],[25.295,25.306,25.320,25.335,25.343]),
 151:(558,[51.312,51.365,51.384,51.429,51.493],[25.134,25.188,25.220,25.242,25.256]),
 160:(740,[51.410,51.427,51.448,51.479,51.498],[25.363,25.376,25.397,25.444,25.529]),
 161:(1344,[51.455,51.474,51.491,51.504,51.516],[25.285,25.292,25.314,25.332,25.369]),
 171:(850,[51.461,51.516,51.554,51.591,51.603],[24.996,25.100,25.159,25.184,25.239]),
 181:(1234,[51.413,51.420,51.433,51.448,51.465],[25.243,25.254,25.271,25.290,25.305]),
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
LAND=np.array(im)>0
def off_km(lon,lat):
    ix=np.clip(((lon-LON0)/res).round().astype(int),0,W-1); iy=np.clip(((LAT1-lat)/res).round().astype(int),0,H-1)
    bad=~LAND[iy,ix]
    if not bad.any(): return np.zeros(0)
    P=np.c_[lon[bad],lat[bad]]
    return np.min(np.hypot(P[:,0][:,None]-RING[None,:,0],P[:,1][:,None]-RING[None,:,1]),axis=1)*111.0

# ---------- data -------------------------------------------------------------
df=pd.read_csv(os.path.join(ROOT,"ANOM_DS02_SiteLocations.csv"))
df["LAT_T"]=np.concatenate([df.LAT.values[1:],[np.nan]]); df["LON_T"]=np.concatenate([df.LON.values[1:],[np.nan]])
d=df.dropna(subset=["LAT_T"])
sites=d.groupby("ANONYMIZED_SITECODE").agg(LON=("LON_T","median"),LAT=("LAT_T","median"))
A=sites[["LON","LAT"]].values; N=len(A)
codes=d.dropna(subset=["LAC_TAC"]).groupby("ANONYMIZED_SITECODE").LAC_TAC.apply(lambda s: frozenset(int(v) for v in s))
codes=codes.reindex(sites.index)
print("sites %d ; codes/site: %s" % (N, codes.apply(len).value_counts().to_dict()))

def tech_of(code):
    if code >= 1000 and code < 10000: return "GSM"
    if code < 1000: return "UTL"
    return None
groups=[]
for code in sorted({c for cs in codes for c in cs}):
    tech=tech_of(code)
    ref = (GSM if tech=="GSM" else UTL).get(code) if tech else None
    if ref is None: continue
    mask=np.array([code in cs for cs in codes])
    if mask.sum()<5: continue
    groups.append(dict(code=code, tech=tech, n=int(mask.sum()), nref=ref[0], mask=mask,
                       rx=np.array(ref[1]), ry=np.array(ref[2])))
print("anchor groups: %d (%s)" % (len(groups), ", ".join("%d%s" % (g["code"], g["tech"][0]) for g in groups)))
tot=sum(g["n"] for g in groups)
print("sites covered by anchors: %d (%.0f%%), total site-group memberships %d" % (tot, 100*tot/N, tot))

W_LQ,W_GQ,W_S,W_LND=3.0,1.0,2.0,60.0
def make_parts(gs, site_excl=None):
    keep = np.ones(N,bool) if site_excl is None else ~site_excl
    def parts(p):
        B=p[:4].reshape(2,2); c=p[4:]
        m=A@B.T+c; lon,lat=m[:,0],m[:,1]
        lq=0.0; res=[]
        for g in gs:
            sel=g["mask"]
            px=np.quantile(lon[sel],QLEV); py=np.quantile(lat[sel],QLEV)
            e=float(np.mean(np.r_[((px-g["rx"])*KX)**2,((py-g["ry"])*KY)**2]))
            lq += (g["n"]/N)*e
            res.append((g["code"],g["tech"],g["n"],float(np.sqrt(e)),
                        float(np.hypot((px[2]-g["rx"][2])*KX,(py[2]-g["ry"][2])*KY))))
        lo=lon[keep]; la=lat[keep]
        gx=np.quantile(lo,[q for q,_ in QX]); gy=np.quantile(la,[q for q,_ in QY])
        gq=float(np.mean(np.r_[((gx-[v for _,v in QX])*KX)**2,((gy-[v for _,v in QY])*KY)**2]))
        s=float((lo.std()*KX-REF_SX_KM)**2+(la.std()*KY-REF_SY_KM)**2)
        rho=float(np.corrcoef(lo,la)[0,1])
        do=off_km(lon,lat)
        lnd=float(len(do)/N*np.mean(np.minimum(do,25.0)**2)) if len(do) else 0.0
        viol=np.maximum(lon-REF_LON[1]-0.005,0)+np.maximum(REF_LON[0]-0.005-lon,0) \
            +np.maximum(lat-REF_LAT[1]-0.005,0)+np.maximum(REF_LAT[0]-0.005-lat,0)
        box=float(np.mean(np.minimum(viol,0.12)**2))*2e4
        return dict(lq=lq,gq=gq,s=s,rho=rho,lnd=lnd,box=box,m=m,res=res,off=do)
    return parts

def loss_of(parts):
    def loss(p):
        d=parts(p)
        return float(W_LQ*d["lq"]+W_GQ*d["gq"]+W_S*d["s"]+50.0*(d["rho"]-REF_RHO)**2
                     +W_LND*d["lnd"]+d["box"])
    return loss

def nm(f,x0,step,iters=20000):
    n=len(x0); pts=[np.array(x0,float)]
    for i in range(n):
        q=np.array(x0,float); q[i]+=step[i]; pts.append(q)
    pts=np.array(pts); fs=np.array([f(q) for q in pts],float)
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
                pts[1:]=pts[0]+0.5*(pts[1:]-pts[0])
                fs[1:]=np.array([f(q) for q in pts[1:]],float)
    i=int(np.argmin(fs)); return pts[i],float(fs[i])

def ls_init(gs):
    Ar=[];br=[]
    for g in gs:
        ma=A[g["mask"]].mean(axis=0); w=np.sqrt(g["n"])
        Ar.append(w*np.array([ma[0],ma[1],1,0,0,0])); br.append(w*g["rx"][2])
        Ar.append(w*np.array([0,0,0,ma[0],ma[1],1])); br.append(w*g["ry"][2])
    x,*_=np.linalg.lstsq(np.array(Ar),np.array(br),rcond=None)
    return x

# ---------- full fit ---------------------------------------------------------
Tprev=[]
for fn in ("transform_final.json","transform_lacfit.json"):
    p=os.path.join(HERE,fn)
    if os.path.exists(p):
        T=json.load(open(p)); Tprev.append(np.r_[np.array(T["B"]).ravel(),np.array(T["c"])])
parts=make_parts(groups); loss=loss_of(parts)
starts=[ls_init(groups)]+Tprev
rng=np.random.default_rng(11)
for s in list(starts):
    for _ in range(60):
        q=s.copy(); q[:4]+=rng.normal(0,0.04,4); q[4:]+=rng.normal(0,0.15,2); starts.append(q)
sc=sorted(((loss(p),p) for p in starts),key=lambda z:z[0])
print("\nstart losses:", ["%.2f"%s for s,_ in sc[:5]])
best=sc[0]
for s0,p0 in sc[:8]:
    p,f=p0,s0
    for st in ([0.01]*4+[0.03]*2,[0.003]*4+[0.01]*2,[0.001]*4+[0.003]*2):
        p2,f2=nm(loss,p,np.array(st),iters=4000)
        if f2<f: p,f=p2,f2
    if f<best[0]: best=(f,p)
p_full=best[1]; B=p_full[:4].reshape(2,2); c=p_full[4:]
d=parts(p_full); m=d["m"]
print("FULL FIT loss %.4f | lacQ %.2f globQ %.2f std %.2f land %.2f box %.2f" %
      (best[0],W_LQ*d["lq"],W_GQ*d["gq"],W_S*d["s"],W_LND*d["lnd"],d["box"]))
print("B=\n%s\nc=%s\nsingular values %s" % (B,c,np.linalg.svd(B,compute_uv=False)))
print("std lon %.2f km (ref %.2f) std lat %.2f (ref %.2f) rho %.3f (ref %.3f)" %
      (m[:,0].std()*KX,REF_SX_KM,m[:,1].std()*KY,REF_SY_KM,d["rho"],REF_RHO))
print("lon %.4f..%.4f lat %.4f..%.4f  off-land %d (max %.1f km)" %
      (m[:,0].min(),m[:,0].max(),m[:,1].min(),m[:,1].max(),len(d["off"]),d["off"].max() if len(d["off"]) else 0))
print("\nper-group in-sample residual (km) / median-offset (km):")
for code,tech,n,e,off in sorted(d["res"],key=lambda z:-z[3]):
    print("   %5d %-5s n=%4d  quantile-resid %5.2f   median-offset %5.2f" % (code,tech,n,e,off))

# ---------- leave-one-group-out --------------------------------------------
print("\nLEAVE-ONE-GROUP-OUT (refit without the group's anchor, and its sites dropped from the global terms):")
rows=[]
for i,g in enumerate(groups):
    gs=[h for h in groups if h is not g]
    pr=make_parts(gs, site_excl=g["mask"]); lf=loss_of(pr)
    p0=ls_init(gs)
    p,f=nm(lf,p0,np.array([0.01]*4+[0.03]*2),iters=4000)
    for st in ([0.003]*4+[0.01]*2,[0.001]*4+[0.003]*2):
        p2,f2=nm(lf,p,np.array(st),iters=4000)
        if f2<f: p,f=p2,f2
    dd=pr(p); mm=dd["m"]
    sel=g["mask"]
    px=np.quantile(mm[sel,0],QLEV); py=np.quantile(mm[sel,1],QLEV)
    e=float(np.sqrt(np.mean(np.r_[((px-g["rx"])*KX)**2,((py-g["ry"])*KY)**2])))
    offc=float(np.hypot((px[2]-g["rx"][2])*KX,(py[2]-g["ry"][2])*KY))
    Bk=p[:4].reshape(2,2)
    rows.append(dict(code=g["code"],tech=g["tech"],n=g["n"],nref=g["nref"],
                     heldout_resid_km=round(e,2),heldout_median_offset_km=round(offc,2),
                     cond=round(float(np.linalg.svd(Bk,compute_uv=False)[0]/np.linalg.svd(Bk,compute_uv=False)[1]),1)))
    print("   %5d %-5s n=%4d/%4d  held-out resid %5.2f km  median offset %5.2f km  (cond %.1f)" %
          (g["code"],g["tech"],g["n"],g["nref"],e,offc,rows[-1]["cond"]))
loo=pd.DataFrame(rows).sort_values("heldout_resid_km",ascending=False)
loo.to_csv(os.path.join(HERE,"loo_results.csv"),index=False)
ins=pd.DataFrame([dict(code=a,tech=b,n=nn,insample_resid_km=round(e,2),median_offset_km=round(o,2))
                  for a,b,nn,e,o in d["res"]])
ins.to_csv(os.path.join(HERE,"insample_residuals.csv"),index=False)
print("\nheld-out residual: median %.2f km | mean %.2f | p90 %.2f | max %.2f" %
      (loo.heldout_resid_km.median(), loo.heldout_resid_km.mean(),
       np.percentile(loo.heldout_resid_km,90), loo.heldout_resid_km.max()))
print("held-out median-offset: median %.2f km | max %.2f" %
      (loo.heldout_median_offset_km.median(), loo.heldout_median_offset_km.max()))
for tech in ("GSM","UTL"):
    sub=loo[loo.tech==tech]
    if len(sub): print("  %s only: median %.2f km (n=%d)" % (tech,sub.heldout_resid_km.median(),len(sub)))
print("  Doha-core groups (%-s): median %.2f km" %
      ("121,142,143,161,150,141,120,181",
       loo[loo.code.isin([121,142,143,161,150,141,120,181])].heldout_resid_km.median()))

json.dump({"B":B.tolist(),"c":c.tolist(),"loss":float(best[0]),
           "reference":"OpenCelliD mcc=427 net=2, per-technology envelopes (GSM for 4-digit codes, UMTS+LTE for 3-digit)",
           "loo_summary":{"median":float(loo.heldout_resid_km.median()),
                          "mean":float(loo.heldout_resid_km.mean()),
                          "p90":float(np.percentile(loo.heldout_resid_km,90)),
                          "max":float(loo.heldout_resid_km.max())}},
          open(os.path.join(HERE,"transform_tech.json"),"w"),indent=1)
sites.assign(LON_REAL=m[:,0],LAT_REAL=m[:,1]).to_csv(os.path.join(HERE,"sites_tech.csv"))
print("\nsaved transform_tech.json / sites_tech.csv / loo_results.csv / insample_residuals.csv")

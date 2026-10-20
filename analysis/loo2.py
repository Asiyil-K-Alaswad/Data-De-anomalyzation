"""LOO v2: per-fold, compare a cold start (LS) and a warm start (full fit),
both re-optimised on the retained anchors; report held-out error and the
retained-anchor loss gap versus the full fit."""
import json, os, numpy as np, pandas as pd

HERE=os.path.dirname(os.path.abspath(__file__))
src=open(os.path.join(HERE,"fit_tech_loo.py")).read()
prefix=src.split("# ---------- full fit")[0]
ns={"__file__":os.path.join(HERE,"fit_tech_loo.py"),"__name__":"prefix"}
exec(compile(prefix,"prefix","exec"),ns)
GSM,UTL,QX,QY=ns["GSM"],ns["UTL"],ns["QX"],ns["QY"]
A,codes,groups,N=ns["A"],ns["codes"],ns["groups"],ns["N"]
make_parts,loss_of,nm,ls_init=ns["make_parts"],ns["loss_of"],ns["nm"],ns["ls_init"]
KX,KY=ns["KX"],ns["KY"]
QLEV=ns["QLEV"]

T=json.load(open(os.path.join(HERE,"transform_tech.json")))
p_full=np.r_[np.array(T["B"]).ravel(),np.array(T["c"])]

def evaluate(p, gs, excl):
    pr=make_parts(gs, site_excl=excl)
    return pr(p)

rows=[]
for g in groups:
    gs=[h for h in groups if h is not g]
    lf=loss_of(make_parts(gs, site_excl=g["mask"]))
    cands=[ls_init(gs), p_full]
    best=None
    for p0 in cands:
        p,f=nm(lf,p0,np.array([0.005]*4+[0.015]*2),iters=3000)
        for st in ([0.0015]*4+[0.005]*2,):
            p2,f2=nm(lf,p,np.array(st),iters=3000)
            if f2<f: p,f=p2,f2
        if best is None or f<best[0]: best=(f,p)
    f_loo,p_loo=best
    f_full_retained=lf(p_full)
    d=evaluate(p_loo,gs,g["mask"]); m=d["m"]; sel=g["mask"]
    px=np.quantile(m[sel,0],QLEV); py=np.quantile(m[sel,1],QLEV)
    e=float(np.sqrt(np.mean(np.r_[((px-g["rx"])*KX)**2,((py-g["ry"])*KY)**2])))
    offc=float(np.hypot((px[2]-g["rx"][2])*KX,(py[2]-g["ry"][2])*KY))
    rows.append(dict(code=g["code"],tech=g["tech"],n=g["n"],nref=g["nref"],
                     loo_resid_km=round(e,2), loo_median_offset_km=round(offc,2),
                     retained_loss_loo=round(f_loo,2), retained_loss_fullfit=round(f_full_retained,2),
                     gap_pct=round(100*(f_loo-f_full_retained)/max(abs(f_full_retained),1e-9),1),
                     warmstart=bool(abs(f_loo-f_full_retained)<1e-12)))
    print("%5d %-4s n=%4d/%4d  held-out %6.2f km  median offset %6.2f km  retained loss %.1f vs full-fit %.1f (gap %+.1f%%)" %
          (g["code"],g["tech"],g["n"],g["nref"],e,offc,f_loo,f_full_retained,rows[-1]["gap_pct"]))

df=pd.DataFrame(rows).sort_values("loo_resid_km",ascending=False)
df.to_csv(os.path.join(HERE,"loo_v2.csv"),index=False)
print("\nHELD-OUT residual: median %.2f | mean %.2f | p90 %.2f | max %.2f km" %
      (df.loo_resid_km.median(),df.loo_resid_km.mean(),np.percentile(df.loo_resid_km,90),df.loo_resid_km.max()))
print("HELD-OUT median offset: median %.2f | max %.2f km" %
      (df.loo_median_offset_km.median(),df.loo_median_offset_km.max()))
print("retained-anchor loss gap (LOO fit vs full fit): median %+.1f%% | max %+.1f%%" %
      (df.gap_pct.median(),df.gap_pct.max()))
dk=df[df.code.isin([121,142,143,161,150,141,120,181])]
print("Doha-core folds (n=%d): held-out median %.2f km" % (len(dk),dk.loo_resid_km.median()))
json.dump({"loo":df.to_dict(orient="list"),
           "summary":{"median_resid":float(df.loo_resid_km.median()),
                      "p90_resid":float(np.percentile(df.loo_resid_km,90)),
                      "max_resid":float(df.loo_resid_km.max()),
                      "median_offset":float(df.loo_median_offset_km.median()),
                      "gap_median_pct":float(df.gap_pct.median())}},
          open(os.path.join(HERE,"loo_v2.json"),"w"),indent=1)

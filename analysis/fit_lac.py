"""Deanonymise DS02 by matching its per-LAC site clouds to OpenCelliD (MCC 427,
MNC 02 = Vodafone Qatar) per-area reference moments."""
import json, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# ---- reference moments from OpenCelliD (ClickHouse play, mcc=427 net=2) -----
# area: n, mean_lon, mean_lat, var_lon, var_lat, cov(lon,lat), minlon, maxlon, minlat, maxlat
REF = {
 110:  (384, 51.435305, 25.730304, 0.010438956, 0.028523519, -0.010822337, 51.0466, 51.5829, 25.2459, 26.1351),
 111:  (889, 51.444716, 25.182382, 0.000692782, 0.00052572, 0.000367969, 51.3357, 51.5183, 25.1190, 25.2871),
 120:  (1419, 51.506022, 25.349623, 0.000969962, 0.000627555, 0.000023948, 51.3958, 51.6419, 25.1223, 25.5686),
 121:  (949, 51.528645, 25.294528, 0.000223885, 0.000319453, -0.000149934, 51.4291, 51.5966, 25.1923, 25.3928),
 130:  (281, 51.257473, 25.356953, 0.033921436, 0.003075094, -0.008181542, 50.7575, 51.6002, 25.1595, 25.5054),
 131:  (535, 51.434126, 25.210993, 0.000587268, 0.000332405, 0.000248308, 51.3316, 51.5536, 25.1676, 25.2747),
 141:  (1358, 51.54686, 25.250693, 0.001540262, 0.00052081, 0.000008566, 51.3810, 51.6231, 25.1292, 25.3201),
 142:  (1874, 51.484451, 25.251669, 0.000917417, 0.000645923, 0.00019388, 51.3796, 51.6186, 25.1827, 25.8323),
 143:  (1086, 51.513376, 25.245611, 0.000410159, 0.000696419, 0.000236488, 51.4332, 51.5870, 25.1637, 25.3942),
 150:  (483, 51.452438, 25.320045, 0.000307137, 0.000360236, -0.000182006, 51.4009, 51.5485, 25.2192, 25.3613),
 151:  (559, 51.385012, 25.202695, 0.009429431, 0.0058228, 0.003036295, 50.8574, 51.5582, 24.5977, 25.6211),
 160:  (740, 51.451399, 25.422256, 0.001383804, 0.006214085, 0.000143138, 51.1722, 51.5650, 25.2885, 25.7279),
 161:  (1567, 51.482783, 25.338252, 0.002286294, 0.016409284, -0.005314214, 51.0459, 51.5636, 25.2489, 26.1289),
 171:  (850, 51.542857, 25.130508, 0.003754151, 0.010924669, 0.001966163, 50.9884, 51.6199, 24.5606, 25.2789),
 181:  (1454, 51.437467, 25.272199, 0.000572641, 0.000560638, -0.000045036, 51.3384, 51.5595, 25.1717, 25.3269),
 1110: (173, 51.502141, 25.738232, 0.004242687, 0.016429455, 0.002582978, 51.1887, 51.6545, 25.0179, 25.9367),
 1130: (37, 51.363134, 25.359265, 0.001975531, 0.005009131, 0.000429873, 51.2217, 51.4301, 25.2459, 25.5412),
 1151: (88, 51.371044, 25.21102, 0.001916706, 0.00494841, 0.00035605, 51.2519, 51.5650, 25.0715, 25.5453),
 1160: (158, 51.444581, 25.436154, 0.001594581, 0.00341762, -0.000762301, 51.3295, 51.5396, 25.3283, 25.6126),
 1171: (210, 51.553576, 25.128107, 0.004105943, 0.006369382, 0.000340654, 51.3824, 51.6103, 24.9696, 25.8186),
}

# ---- DS02 sites -------------------------------------------------------------
df = pd.read_csv(os.path.join(ROOT, "ANOM_DS02_SiteLocations.csv"))
df["LAT_T"] = np.concatenate([df.LAT.values[1:], [np.nan]])
df["LON_T"] = np.concatenate([df.LON.values[1:], [np.nan]])
d = df.dropna(subset=["LAT_T"])
sites = d.groupby("ANONYMIZED_SITECODE").agg(LON=("LON_T", "median"), LAT=("LAT_T", "median"),
                                             LAC=("LAC_TAC", "first"), NCELL=("LON_T", "size"))
print("sites:", len(sites))
lac_counts = sites.LAC.value_counts()
print("lac counts:", lac_counts.head(30).to_dict())

# per-LAC anon moments
rows = []
for lac, g in sites.groupby("LAC"):
    lac = int(lac)
    if lac not in REF:
        continue
    P = g[["LON", "LAT"]].values
    m = P.mean(axis=0)
    C = np.cov(P.T, bias=True) if len(P) > 1 else np.zeros((2, 2))
    rows.append(dict(lac=lac, n=len(P), ma=m, Ca=C, nr=REF[lac][0],
                     mr=np.array(REF[lac][1:3]), Cr=np.array([[REF[lac][3], REF[lac][5]],
                                                              [REF[lac][5], REF[lac][4]]])))
print("\nmatched LACs: %d, sites: %d of %d" % (len(rows), sum(r["n"] for r in rows), len(sites)))
for r in sorted(rows, key=lambda r: -r["n"]):
    print("  LAC %5d  n=%4d refn=%5d  anon mean (%9.5f,%9.5f)  ref mean (%.5f,%.5f)" %
          (r["lac"], r["n"], r["nr"], r["ma"][0], r["ma"][1], r["mr"][0], r["mr"][1]))

# ---- stage 1: least squares on LAC centroids --------------------------------
Arows, brows, wts = [], [], []
for r in rows:
    w = np.sqrt(r["n"])
    Arows.append(w * np.array([r["ma"][0], r["ma"][1], 1.0, 0.0, 0.0, 0.0]))
    brows.append(w * r["mr"][0])
    Arows.append(w * np.array([0.0, 0.0, 0.0, r["ma"][0], r["ma"][1], 1.0]))
    brows.append(w * r["mr"][1])
A = np.array(Arows); b = np.array(brows)
x, res, rank, sv = np.linalg.lstsq(A, b, rcond=None)
B0 = x[[0, 1, 3, 4]].reshape(2, 2); c0 = x[[2, 5]]
print("\nLS centroid fit:\nB=\n%s\nc=%s  singular values %s" % (B0, c0, sv))
pred = A @ x
print("centroid RMS residual: %.5f deg (~%.1f km)" % (np.sqrt(np.mean((pred - b) ** 2)) * 111,
                                                      np.sqrt(np.mean((pred - b) ** 2)) * 111))

def loss(params, ret=False):
    B = params[:4].reshape(2, 2); c = params[4:]
    tot, per = 0.0, []
    for r in rows:
        mp = B @ r["ma"] + c
        e1 = np.sum((mp - r["mr"]) ** 2)
        Cm = B @ r["Ca"] @ B.T
        e2 = np.sum((Cm - r["Cr"]) ** 2)
        w = r["n"]
        # scale covariance error so its typical magnitude matches the centroid error
        per.append((r["lac"], r["n"], np.sqrt(e1) * 111, np.sqrt(e2)))
        tot += w * (e1 * 1.0 + e2 * 0.05)
    if ret:
        return tot / sum(r["n"] for r in rows), per
    return tot / sum(r["n"] for r in rows)

def nm(f, x0, step, iters=20000, tol=1e-14):
    n = len(x0)
    pts = [np.array(x0, float)]
    for i in range(n):
        p = np.array(x0, float); p[i] += step[i]; pts.append(p)
    pts = np.array(pts); fs = np.array([f(p) for p in pts])
    for _ in range(iters):
        o = np.argsort(fs); pts, fs = pts[o], fs[o]
        if abs(fs[-1] - fs[0]) < tol * (abs(fs[0]) + tol):
            break
        cen = pts[:-1].mean(axis=0)
        xr = cen + (cen - pts[-1]); fr = f(xr)
        if fr < fs[0]:
            xe = cen + 2 * (xr - cen); fe = f(xe)
            pts[-1], fs[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < fs[-2]:
            pts[-1], fs[-1] = xr, fr
        else:
            xc = cen + 0.5 * (pts[-1] - cen); fc = f(xc)
            if fc < fs[-1]:
                pts[-1], fs[-1] = xc, fc
            else:
                pts[1:] = pts[0] + 0.5 * (pts[1:] - pts[0])
                fs[1:] = np.array([f(p) for p in pts[1:]])
    i = int(np.argmin(fs))
    return pts[i], fs[i]

x0 = np.concatenate([B0.ravel(), c0])
best = (loss(x0), x0)
print("\nstart loss %.6e" % best[0])
for step in ([0.05, 0.05, 0.05, 0.05, 0.01, 0.01], [0.01]*4 + [0.002]*2, [0.002]*4 + [0.0004]*2):
    xb, fb = nm(loss, best[1], np.array(step))
    if fb < best[0]:
        best = (fb, xb)
    print("  after step %s -> loss %.6e" % (step[0], fb))
B = best[1][:4].reshape(2, 2); c = best[1][4:]
print("\nFINAL B=\n%s\nc=%s\nloss=%.6e" % (B, c, best[0]))
print("singular values of B:", np.linalg.svd(B, compute_uv=False))

tot, per = loss(best[1], ret=True)
print("\nper-LAC centroid error (km) / cov error:")
for lac, n, e1, e2 in sorted(per, key=lambda z: -z[2]):
    print("  LAC %5d n=%4d  centroid err %7.2f km   cov err %.2e" % (lac, n, e1, e2))

mapped = sites[["LON", "LAT"]].values @ B.T + c
sites_out = sites.assign(LON_REAL=mapped[:, 0], LAT_REAL=mapped[:, 1])
sites_out.to_csv(os.path.join(HERE, "sites_lacfit.csv"))
print("\nmapped lon %.4f..%.4f lat %.4f..%.4f" % (mapped[:,0].min(), mapped[:,0].max(),
                                                  mapped[:,1].min(), mapped[:,1].max()))
json.dump({"B": B.tolist(), "c": c.tolist(), "loss": float(best[0])},
          open(os.path.join(HERE, "transform_lacfit.json"), "w"), indent=1)

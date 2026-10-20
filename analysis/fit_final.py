"""Final deanonymisation fit for DS02.

Anchors
  * global marginal quantiles + std/corr of Vodafone Qatar (OpenCelliD mcc=427 net=2,
    site-level dedup, n=14850)
  * per-LAC centroid of the same operator's cells (median over GSM/LTE/UMTS areas,
    robust to LAC/TAC collisions)
  * Qatar land polygon (sites must be on land)
Model:  real = B @ anon + c
"""
import json, os
import numpy as np, pandas as pd
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# ---- reference: site-level marginals of Vodafone Qatar (mcc=427, net=2) -----
QX = [(0.01, 51.2217), (0.05, 51.3714), (0.25, 51.4420), (0.50, 51.4891),
      (0.75, 51.5246), (0.95, 51.5843), (0.99, 51.6088)]
QY = [(0.01, 24.9809), (0.05, 25.1374), (0.25, 25.2267), (0.50, 25.2739),
      (0.75, 25.3269), (0.95, 25.5865), (0.99, 25.8831)]
REF_SX, REF_SY, REF_RHO = 0.07715, 0.1447, -0.1636
REF_LON_MIN, REF_LON_MAX = 50.7575, 51.6545
REF_LAT_MIN, REF_LAT_MAX = 24.5606, 26.1351

# per-LAC reference centroids (median over radio technologies), from ClickHouse
LAC_REF = {
 110: (51.4484, 25.7240), 111: (51.4454, 25.1789), 120: (51.5072, 25.3478),
 121: (51.5284, 25.2949), 130: (51.2836, 25.3499), 131: (51.4322, 25.2077),
 141: (51.5572, 25.2512), 142: (51.4821, 25.2525), 143: (51.5159, 25.2447),
 150: (51.4545, 25.3203), 151: (51.3831, 25.2053), 160: (51.4516, 25.4211),
 161: (51.4907, 25.3090), 171: (51.5445, 25.1284), 181: (51.4357, 25.2714),
 1110: (51.4494, 25.7364), 1130: (51.3631, 25.3593), 1151: (51.3694, 25.2099),
 1160: (51.4443, 25.4368), 1171: (51.5536, 25.1281),
}

# ---- land polygon ----------------------------------------------------------
def load_rings(path):
    gj = json.load(open(path, encoding="utf-8")); out = []
    for f in gj["features"]:
        g = f["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for p in polys: out.append(np.array(p[0], float))
    return out
RING = max(load_rings(os.path.join(HERE, "qatar_land.geojson")), key=len)

res = 0.0005
LON0, LON1, LAT0, LAT1 = 50.55, 51.85, 24.35, 26.35
W = int(round((LON1-LON0)/res))+1; H = int(round((LAT1-LAT0)/res))+1
img = Image.new("L", (W, H), 0)
ImageDraw.Draw(img).polygon([((x-LON0)/res, (LAT1-y)/res) for x, y in RING], fill=255)
land = np.array(img) > 0

def outside_km(lon, lat):
    """0 if on land, else approx distance to nearest land pixel (km)"""
    ix = np.clip(((lon-LON0)/res).round().astype(int), 0, W-1)
    iy = np.clip(((LAT1-lat)/res).round().astype(int), 0, H-1)
    return np.where(land[iy, ix], 0.0, np.nan)

def land_penalty(lon, lat):
    """soft penalty: count of points off land weighted by distance (else 0)"""
    on = outside_km(lon, lat)
    off = np.isnan(on)
    if not off.any():
        return 0.0
    # distance to nearest land pixel via BFS-free approximation: sample ring
    P = np.c_[lon[off], lat[off]]
    d = np.min(np.hypot(P[:,0][:,None]-RING[None,:,0], P[:,1][:,None]-RING[None,:,1]), axis=1)
    return float(np.sum(np.minimum(d*111.0, 20.0)**2) / len(lon))

# ---- data ------------------------------------------------------------------
df = pd.read_csv(os.path.join(ROOT, "ANOM_DS02_SiteLocations.csv"))
df["LAT_T"] = np.concatenate([df.LAT.values[1:], [np.nan]])
df["LON_T"] = np.concatenate([df.LON.values[1:], [np.nan]])
d = df.dropna(subset=["LAT_T"])
sites = d.groupby("ANONYMIZED_SITECODE").agg(LON=("LON_T","median"), LAT=("LAT_T","median"),
                                             LAC=("LAC_TAC","first"))
A = sites[["LON","LAT"]].values
lac = sites.LAC.values
lac_i = np.array([int(x) for x in lac])
print("sites %d" % len(A))

lac_terms = []
for l, (mx, my) in LAC_REF.items():
    sel = lac_i == l
    if sel.sum() == 0: continue
    lac_terms.append((sel, np.array([mx, my]), sel.sum()))
print("lac anchors:", len(lac_terms))

W_CENT, W_Q, W_SC, W_LAND = 1.0, 0.6, 4.0, 0.02

def loss(p, ret=False):
    B = p[:4].reshape(2,2); c = p[4:]
    m = A @ B.T + c
    lon, lat = m[:,0], m[:,1]
    # per-LAC centroid term (Huber on km error)
    tot = 0.0; cent = []
    for sel, ref, n in lac_terms:
        dm = (m[sel].mean(axis=0) - ref) * np.array([111*np.cos(np.radians(25.4)), 111.0])
        e = np.hypot(*dm)
        cent.append((int(lac_i[sel][0]), float(e)))
        tot += W_CENT * (n/len(A)) * min(e, 25.0)**2
    # global marginal term
    qx = np.quantile(lon, [q for q, _ in QX]); qy = np.quantile(lat, [q for q, _ in QY])
    tot += W_Q * np.sum((qx - [v for _, v in QX])**2) * 100
    tot += W_Q * np.sum((qy - [v for _, v in QY])**2) * 100
    tot += W_SC * ((lon.std() - REF_SX)**2 + (lat.std() - REF_SY)**2) * 100
    rho = np.corrcoef(lon, lat)[0,1]
    tot += W_SC * 0.02 * (rho - REF_RHO)**2
    # bounds penalty
    tot += 0.5 * np.sum(np.maximum(lon - REF_LON_MAX - 0.01, 0)**2 + np.maximum(REF_LON_MIN - 0.01 - lon, 0)**2) * 1e4
    tot += 0.5 * np.sum(np.maximum(lat - REF_LAT_MAX - 0.01, 0)**2 + np.maximum(REF_LAT_MIN - 0.01 - lat, 0)**2) * 1e4
    tot += W_LAND * land_penalty(lon, lat)
    if ret:
        return tot, m, cent, qx, qy, rho
    return tot

def nm(f, x0, step, iters=8000):
    n = len(x0); pts = [np.array(x0, float)]
    for i in range(n):
        q = np.array(x0, float); q[i] += step[i]; pts.append(q)
    pts = np.array(pts); fs = np.array([f(q) for q in pts])
    for _ in range(iters):
        o = np.argsort(fs); pts, fs = pts[o], fs[o]
        if fs[-1] - fs[0] < 1e-12 * (abs(fs[0]) + 1e-12): break
        cen = pts[:-1].mean(axis=0)
        xr = cen + (cen - pts[-1]); fr = f(xr)
        if fr < fs[0]:
            xe = cen + 2*(xr-cen); fe = f(xe)
            pts[-1], fs[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < fs[-2]:
            pts[-1], fs[-1] = xr, fr
        else:
            xc = cen + 0.5*(pts[-1]-cen); fc = f(xc)
            if fc < fs[-1]: pts[-1], fs[-1] = xc, fc
            else:
                pts[1:] = pts[0] + 0.5*(pts[1:]-pts[0])
                fs[1:] = np.array([f(q) for q in pts[1:]])
    i = int(np.argmin(fs)); return pts[i], fs[i]

# ---- initial guesses -------------------------------------------------------
starts = []
T0 = json.load(open(os.path.join(HERE, "transform_lacfit.json")))
starts.append(np.concatenate([np.array(T0["B"]).ravel(), np.array(T0["c"])]))
rng = np.random.default_rng(0)
base = starts[0]
for _ in range(400):
    p = base.copy()
    p[:4] += rng.normal(0, 0.06, 4)
    p[4:] += rng.normal(0, 0.25, 2)
    starts.append(p)
scored = sorted(((loss(p), p) for p in starts), key=lambda z: z[0])
print("best starts:", ["%.4f" % s for s, _ in scored[:5]])

best = scored[0]
for s0, p0 in scored[:12]:
    p = p0; f = s0
    for st in ([0.02]*4 + [0.05]*2, [0.005]*4 + [0.01]*2, [0.001]*4 + [0.002]*2):
        p2, f2 = nm(loss, p, np.array(st))
        if f2 < f: p, f = p2, f2
    if f < best[0]: best = (f, p)
print("final loss %.5f" % best[0])

B = best[1][:4].reshape(2,2); c = best[1][4:]
tot, m, cent, qx, qy, rho = loss(best[1], ret=True)
print("B=\n%s\nc=%s" % (B, c))
print("singular values", np.linalg.svd(B, compute_uv=False))
print("\nLAC centroid residuals (km):")
for l, e in sorted(cent, key=lambda z: -z[1]):
    print("   LAC %5d  %6.2f" % (l, e))
print("\nquantiles lon mapped vs ref:")
for (q, v), mv in zip(QX, qx): print("   %4.2f  %.4f  %.4f  (%+.2f km)" % (q, mv, v, (mv-v)*100))
print("quantiles lat mapped vs ref:")
for (q, v), mv in zip(QY, qy): print("   %4.2f  %.4f  %.4f  (%+.2f km)" % (q, mv, v, (mv-v)*111))
print("std lon %.5f (ref %.5f)  std lat %.5f (ref %.5f)  rho %.3f (ref %.3f)" %
      (m[:,0].std(), REF_SX, m[:,1].std(), REF_SY, rho, REF_RHO))
print("mapped lon %.4f..%.4f (ref %.4f..%.4f)" % (m[:,0].min(), m[:,0].max(), REF_LON_MIN, REF_LON_MAX))
print("mapped lat %.4f..%.4f (ref %.4f..%.4f)" % (m[:,1].min(), m[:,1].max(), REF_LAT_MIN, REF_LAT_MAX))
off = outside_km(m[:,0], m[:,1])
print("on land: %.4f" % (1-np.isnan(off).mean()))

json.dump({"B": B.tolist(), "c": c.tolist(), "loss": float(best[0])},
          open(os.path.join(HERE, "transform_final.json"), "w"), indent=1)
sites.assign(LON_REAL=m[:,0], LAT_REAL=m[:,1]).to_csv(os.path.join(HERE, "sites_final.csv"))
np.save(os.path.join(HERE, "mapped_final.npy"), m)
print("\nwrote transform_final.json / sites_final.csv")

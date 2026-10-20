"""Fit an affine inverse of the DS02 coordinate anonymisation.

Model:  real = M @ ((anon - anon_mean)/anon_std) + t     (whitened anon space)
Objective: maximise log|det M| (spread) subject to mapped sites lying on Qatar's
land polygon (penalty on distance outside, capped at 5 km).
"""
import json, os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GEO = os.path.join(HERE, "qatar_land.geojson")

def load_largest_ring(path):
    gj = json.load(open(path, encoding="utf-8"))
    best = None
    for feat in gj["features"]:
        geom = feat["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for poly in polys:
            ring = np.array(poly[0], dtype=float)
            if best is None or len(ring) > len(best):
                best = ring
    return best

RING = load_largest_ring(GEO)
print("land ring points:", len(RING))

# ---- distance-to-land raster (coarse, generous window) ----------------------
res = 0.001
LON0, LON1 = 50.20, 52.30
LAT0, LAT1 = 23.90, 26.90
W = int((LON1 - LON0) / res) + 1
H = int((LAT1 - LAT0) / res) + 1
img = Image.new("L", (W, H), 0)
ImageDraw.Draw(img).polygon([((x - LON0) / res, (LAT1 - y) / res) for x, y in RING], fill=255)
land = np.array(img) > 0
print("grid %dx%d land frac %.3f" % (W, H, land.mean()))

BIG = 1e5
dist = np.where(land, 0.0, BIG).astype(np.float32)
idx = np.arange(H, dtype=np.float32)[:, None]
dist = np.minimum.accumulate(dist - idx, axis=0) + idx
dist = np.minimum.accumulate(dist[::-1] - idx[::-1], axis=0)[::-1] + idx
idxc = np.arange(W, dtype=np.float32)[None, :]
dist = np.minimum.accumulate(dist - idxc, axis=1) + idxc
dist = np.minimum.accumulate(dist[:, ::-1] - idxc[:, ::-1], axis=1)[:, ::-1] + idxc
OUT = dist * res

def outside_at(lon, lat):
    dx = np.maximum(np.maximum(LON0 - lon, lon - LON1), 0.0)
    dy = np.maximum(np.maximum(LAT0 - lat, lat - LAT1), 0.0)
    win = np.hypot(dx, dy)
    ix = np.clip(((lon - LON0) / res).round().astype(int), 0, W - 1)
    iy = np.clip(((LAT1 - lat) / res).round().astype(int), 0, H - 1)
    return OUT[iy, ix] + win

# ---- data -------------------------------------------------------------------
df = pd.read_csv(os.path.join(ROOT, "ANOM_DS02_SiteLocations.csv"))
df["LAT_T"] = np.concatenate([df.LAT.values[1:], [np.nan]])
df["LON_T"] = np.concatenate([df.LON.values[1:], [np.nan]])
sites = df.dropna(subset=["LAT_T"]).groupby("ANONYMIZED_SITECODE").agg(
    LAT=("LAT_T", "median"), LON=("LON_T", "median"), LAC=("LAC_TAC", "first"))
A = sites[["LON", "LAT"]].values
AM, AS = A.mean(axis=0), A.std(axis=0)
Aw = (A - AM) / AS
print("sites %d  anon std %s" % (len(A), AS))

RMEAN = RING.mean(axis=0)
LAM = 3000.0
CAP = 0.045

def score(M, t, ret=False):
    m = Aw @ M.T + t
    od = outside_at(m[:, 0], m[:, 1])
    pen = float(np.mean(np.minimum(od, CAP) ** 2))
    det = abs(np.linalg.det(M))
    if det <= 1e-9:
        return -1e9
    sc = np.log(det) - LAM * pen
    if ret:
        return sc, m, od, pen, det
    return sc

def nm(f, x0, step, iters=4000, tol=1e-11):
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
        if fr > fs[0]:
            xe = cen + 2 * (xr - cen); fe = f(xe)
            pts[-1], fs[-1] = (xe, fe) if fe > fr else (xr, fr)
        elif fr > fs[-2]:
            pts[-1], fs[-1] = xr, fr
        else:
            xc = cen + 0.5 * (pts[-1] - cen); fc = f(xc)
            if fc > fs[-1]:
                pts[-1], fs[-1] = xc, fc
            else:
                pts[1:] = pts[0] + 0.5 * (pts[1:] - pts[0])
                fs[1:] = np.array([f(p) for p in pts[1:]])
    i = int(np.argmax(fs))
    return pts[i], fs[i]

def rot(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

starts = []
for th in np.arange(0, 360, 10):
    for ph in np.arange(0, 180, 10):
        for s1 in (0.25, 0.4, 0.55):
            for asp in (0.5, 1.0, 1.8):
                M = rot(np.radians(th)) @ np.diag([s1 * asp, s1]) @ rot(np.radians(ph))
                t = RMEAN - (Aw.mean(axis=0) @ M.T)
                starts.append((M, t))
print("starts", len(starts))
scored = sorted(((score(M, t), M, t) for M, t in starts), key=lambda z: -z[0])
print("top starts", ["%.3f" % s for s, _, _ in scored[:5]])

best = (scored[0][0], scored[0][1], scored[0][2])
for sc0, M, t in scored[:25]:
    x0 = np.concatenate([M.ravel(), t])
    step = np.array([0.03] * 4 + [0.08, 0.08])
    x, fv = nm(lambda p: score(p[:4].reshape(2, 2), p[4:]), x0, step)
    if fv > best[0]:
        best = (fv, x[:4].reshape(2, 2), x[4:])
        print("best %.4f M=%s t=%s" % (fv, np.round(best[1], 4).tolist(), np.round(best[2], 4).tolist()))

fv, M, t = best
sc, m, od, pen, det = score(M, t, ret=True)
print("\nBEST score %.4f det %.4f pen %.3e" % (fv, det, pen))
print("M =\n", M, "\nt =", t)
print("mapped lon %.4f..%.4f lat %.4f..%.4f" % (m[:,0].min(), m[:,0].max(), m[:,1].min(), m[:,1].max()))
print("inside land %.3f  <1km %.3f  <5km %.3f" % ((od==0).mean(), (od<0.009).mean(), (od<0.045).mean()))

B = M / AS[None, :]
c = t - AM @ B.T
print("B =\n", B, "\nc =", c)
json.dump({"B": B.tolist(), "c": c.tolist(), "score": fv,
           "whitened": {"M": M.tolist(), "t": t.tolist(), "anon_mean": AM.tolist(), "anon_std": AS.tolist()}},
          open(os.path.join(HERE, "transform.json"), "w"), indent=1)

# ---- overlay ---------------------------------------------------------------
S = 1500
LO0, LO1, LA0, LA1 = 50.55, 51.80, 24.40, 26.30
def to_px(lon, lat):
    return ((lon - LO0) / (LO1 - LO0) * S, S - (lat - LA0) / (LA1 - LA0) * S)
im = Image.new("RGB", (S, S), "white"); dr = ImageDraw.Draw(im)
for g in np.arange(LO0, LO1, 0.1):
    x, _ = to_px(g, LA0); dr.line([(x, 0), (x, S)], fill=(240, 240, 240))
for g in np.arange(LA0, LA1, 0.1):
    _, y = to_px(LO0, g); dr.line([(0, y), (S, y)], fill=(240, 240, 240))
dr.polygon([to_px(x, y) for x, y in RING], fill=(214, 232, 246), outline=(50, 80, 160))
CITY = {"Doha": (51.533, 25.286), "Al Wakrah": (51.60, 25.17), "Al Khor": (51.51, 25.68),
        "Dukhan": (50.78, 25.42), "Mesaieed": (51.55, 24.99), "Al Ruwais": (51.21, 26.13),
        "Umm Bab": (50.80, 25.20), "Abu Samra": (50.83, 24.75), "Sealine": (51.55, 24.85),
        "Al Shahaniya": (51.20, 25.37), "Lusail": (51.49, 25.43), "Ras Laffan": (51.53, 25.90),
        "Al Zubarah": (51.03, 25.98), "Fuwayrit": (51.37, 26.02), "Al Karaana": (51.05, 25.05)}
for name, (lo, la) in CITY.items():
    x, y = to_px(lo, la)
    dr.line([(x-5, y), (x+5, y)], fill=(0, 120, 0)); dr.line([(x, y-5), (x, y+5)], fill=(0, 120, 0))
    dr.text((x+6, y-6), name, fill=(0, 100, 0))
for lon, lat in m:
    x, y = to_px(lon, lat)
    if 0 <= x < S and 0 <= y < S:
        dr.ellipse([x-2, y-2, x+2, y+2], fill=(205, 0, 0))
im.save(os.path.join(HERE, "fit_overlay.png"))
print("wrote overlay")
sites.assign(LON_REAL=m[:, 0], LAT_REAL=m[:, 1], OUT_KM=od*111).to_csv(os.path.join(HERE, "site_coords_fit.csv"))
print("done")

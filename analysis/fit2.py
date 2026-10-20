"""Fit an affine inverse of the DS02 coordinate anonymisation.

Model:  real = B @ anon + c ;  fitted as  real = M @ ((anon-mean)/std) + t
Objective: keep every mapped site on Qatar's land polygon while maximising the
spread of the mapped cloud (log|det M|) -> "a national network covers the country".
"""
import json, os, sys
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

def load_rings(path):
    gj = json.load(open(path, encoding="utf-8"))
    rings = []
    for feat in gj["features"]:
        geom = feat["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for poly in polys:
            rings.append(np.array(poly[0], dtype=float))
    return rings

RINGS = load_rings(os.path.join(HERE, "qatar_land.geojson"))
RING = max(RINGS, key=len)
print("rings:", [len(r) for r in RINGS])

# ---------------- rasterised outside-distance (L1 chamfer, correct passes) ----
res = 0.001
LON0, LON1, LAT0, LAT1 = 50.20, 52.30, 23.90, 26.90
W = int(round((LON1 - LON0) / res)) + 1
H = int(round((LAT1 - LAT0) / res)) + 1
img = Image.new("L", (W, H), 0)
ImageDraw.Draw(img).polygon([((x - LON0) / res, (LAT1 - y) / res) for x, y in RING], fill=255)
land = np.array(img) > 0
print("grid %dx%d land frac %.3f" % (W, H, land.mean()))

FAR = 1e6
d = np.where(land, 0.0, FAR)
ri = np.arange(H, dtype=np.float64)[:, None]
ci = np.arange(W, dtype=np.float64)[None, :]
d = np.minimum.accumulate(d - ri, axis=0) + ri
d = (np.minimum.accumulate(d[::-1] - ri, axis=0) + ri)[::-1]
d = np.minimum.accumulate(d - ci, axis=1) + ci
d = (np.minimum.accumulate(d[:, ::-1] - ci, axis=1) + ci)[:, ::-1]
OUT = d * res
print("sanity: Doha outside=%.4f  centre-of-sea(51.0,25.0)=%.4f  far(45,20)=%.3f" %
      (OUT[int((LAT1 - 25.286) / res), int((51.533 - LON0) / res)],
       OUT[int((LAT1 - 25.0) / res), int((51.0 - LON0) / res)],
       OUT[0, 0]))

def outside_at(lon, lat):
    dx = np.maximum(np.maximum(LON0 - lon, lon - LON1), 0.0)
    dy = np.maximum(np.maximum(LAT0 - lat, lat - LAT1), 0.0)
    win = np.hypot(dx, dy)
    ix = np.clip(((lon - LON0) / res).round().astype(int), 0, W - 1)
    iy = np.clip(((LAT1 - lat) / res).round().astype(int), 0, H - 1)
    return OUT[iy, ix] + win

# ---------------- data -------------------------------------------------------
df = pd.read_csv(os.path.join(ROOT, "ANOM_DS02_SiteLocations.csv"))
df["LAT_T"] = np.concatenate([df.LAT.values[1:], [np.nan]])
df["LON_T"] = np.concatenate([df.LON.values[1:], [np.nan]])
sites = df.dropna(subset=["LAT_T"]).groupby("ANONYMIZED_SITECODE").agg(
    LAT=("LAT_T", "median"), LON=("LON_T", "median"), LAC=("LAC_TAC", "first"))
A = sites[["LON", "LAT"]].values
AM, AS = A.mean(axis=0), A.std(axis=0)
Aw = (A - AM) / AS
print("sites %d anon std %s whitened extent lon %.2f lat %.2f" %
      (len(A), np.round(AS, 4), np.ptp(Aw[:,0]), np.ptp(Aw[:,1])))

LAM, CAP = 2.0e4, 0.30

def score(M, t, ret=False):
    m = Aw @ M.T + t
    od = outside_at(m[:, 0], m[:, 1])
    pen = float(np.mean(np.minimum(od, CAP) ** 2))
    det = abs(np.linalg.det(M))
    if det <= 1e-12:
        return -1e12
    sc = np.log(det) - LAM * pen
    if ret:
        return sc, m, od, pen, det
    return sc

def nm(f, x0, step, iters=6000, tol=1e-12):
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

def rot(a): return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

starts = []
for th in np.arange(0, 360, 8):
    for ph in np.arange(0, 180, 8):
        for s1 in (0.06, 0.10, 0.16, 0.24):
            for asp in (0.4, 0.7, 1.0, 1.5):
                M = rot(np.radians(th)) @ np.diag([s1 * asp, s1]) @ rot(np.radians(ph))
                t = RING.mean(axis=0) - (Aw.mean(axis=0) @ M.T)
                starts.append((M, t))
print("starts", len(starts))
scored = sorted(((score(M, t), M, t) for M, t in starts), key=lambda z: -z[0])
print("top start scores", ["%.3f" % z[0] for z in scored[:5]])

sols = []
for sc0, M, t in scored[:40]:
    x0 = np.concatenate([M.ravel(), t])
    x, fv = nm(lambda p: score(p[:4].reshape(2, 2), p[4:]), x0, np.array([0.02]*4 + [0.05, 0.05]))
    sols.append((fv, x[:4].reshape(2, 2), x[4:]))
sols.sort(key=lambda z: -z[0])

# de-duplicate near-identical solutions
uniq = []
for fv, M, t in sols:
    if all(np.abs(M - M2).max() > 1e-3 or np.abs(t - t2).max() > 1e-2 for _, M2, t2 in uniq):
        uniq.append((fv, M, t))
print("distinct optima:", len(uniq))

S = 1500
LO0, LO1, LA0, LA1 = 50.55, 51.85, 24.40, 26.30
def to_px(lon, lat):
    return ((lon - LO0) / (LO1 - LO0) * S, S - (lat - LA0) / (LA1 - LA0) * S)
CITY = {"Doha": (51.533, 25.286), "AlWakrah": (51.60, 25.17), "AlKhor": (51.51, 25.68),
        "Dukhan": (50.78, 25.42), "Mesaieed": (51.55, 24.99), "AlRuwais": (51.21, 26.13),
        "UmmBab": (50.80, 25.20), "AbuSamra": (50.83, 24.75), "Sealine": (51.55, 24.85),
        "Shahaniya": (51.20, 25.37), "Lusail": (51.49, 25.43), "RasLaffan": (51.53, 25.90),
        "Zubarah": (51.03, 25.98), "Fuwayrit": (51.37, 26.02), "Karaana": (51.05, 25.05)}

report = []
for k, (fv, M, t) in enumerate(uniq[:6]):
    sc, m, od, pen, det = score(M, t, ret=True)
    inside = (od == 0).mean(); near = (od < 0.009).mean()
    print("\n--- optimum %d: score %.4f det %.4f pen %.2e inside %.3f <1km %.3f" % (k, fv, det, pen, inside, near))
    print("    mapped lon %.3f..%.3f lat %.3f..%.3f" % (m[:,0].min(), m[:,0].max(), m[:,1].min(), m[:,1].max()))
    report.append((fv, M, t, inside, near))
    im = Image.new("RGB", (S, S), "white"); dr = ImageDraw.Draw(im)
    for lon, lat in m:
        x, y = to_px(lon, lat)
        if 0 <= x < S and 0 <= y < S:
            dr.ellipse([x-2, y-2, x+2, y+2], fill=(205, 0, 0))
    dr.polygon([to_px(x, y) for x, y in RING], outline=(50, 80, 160))
    for name, (lo, la) in CITY.items():
        x, y = to_px(lo, la)
        dr.line([(x-5, y), (x+5, y)], fill=(0, 120, 0)); dr.line([(x, y-5), (x, y+5)], fill=(0, 120, 0))
        dr.text((x+6, y-6), name, fill=(0, 100, 0))
    im.save(os.path.join(HERE, "fit_opt%d.png" % k))

fv, M, t, inside, near = report[0]
B = M / AS[None, :]
c = t - AM @ B.T
print("\nBEST: inside %.3f  <1km %.3f" % (inside, near))
print("B =\n", B, "\nc =", c)
json.dump({"B": B.tolist(), "c": c.tolist(), "score": float(fv), "inside_frac": float(inside),
           "whitened": {"M": M.tolist(), "t": t.tolist(), "anon_mean": AM.tolist(), "anon_std": AS.tolist()}},
          open(os.path.join(HERE, "transform.json"), "w"), indent=1)

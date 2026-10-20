import json, os
import numpy as np, pandas as pd
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
T = json.load(open(os.path.join(HERE, "transform_lacfit.json")))
B = np.array(T["B"]); c = np.array(T["c"])

def load_rings(path):
    gj = json.load(open(path, encoding="utf-8")); out = []
    for f in gj["features"]:
        g = f["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for p in polys:
            out.append(np.array(p[0], float))
    return out
RING = max(load_rings(os.path.join(HERE, "qatar_land.geojson")), key=len)

# Voronoi-free inside test via PIL mask at 0.0005 deg
res = 0.0005
LON0, LON1, LAT0, LAT1 = 50.60, 51.85, 24.35, 26.30
W = int(round((LON1-LON0)/res))+1; H = int(round((LAT1-LAT0)/res))+1
img = Image.new("L", (W, H), 0)
ImageDraw.Draw(img).polygon([((x-LON0)/res, (LAT1-y)/res) for x, y in RING], fill=255)
land = np.array(img) > 0
def inside(lon, lat):
    ix = np.clip(((lon-LON0)/res).round().astype(int), 0, W-1)
    iy = np.clip(((LAT1-lat)/res).round().astype(int), 0, H-1)
    return land[iy, ix], (ix, iy)

sites = pd.read_csv(os.path.join(HERE, "sites_lacfit.csv"))
m = sites[["LON_REAL","LAT_REAL"]].values
ins, _ = inside(m[:,0], m[:,1])
print("sites inside land polygon: %.3f of %d" % (ins.mean(), len(sites)))
print("outside sites:")
print(sites.loc[~ins, ["ANONYMIZED_SITECODE","LAC","LON_REAL","LAT_REAL"]].to_string())

# distance to nearest city for key clusters
CITY = {"Doha": (51.533, 25.286), "Al Wakrah": (51.60, 25.17), "Al Khor": (51.51, 25.68),
        "Dukhan": (50.78, 25.42), "Mesaieed": (51.55, 24.99), "Al Ruwais": (51.21, 26.13),
        "Umm Bab": (50.80, 25.20), "Abu Samra": (50.83, 24.75), "Sealine": (51.55, 24.85),
        "Al Shahaniya": (51.20, 25.37), "Lusail": (51.49, 25.43), "Ras Laffan": (51.53, 25.90),
        "Al Zubarah": (51.03, 25.98), "Fuwayrit": (51.37, 26.02), "Al Karaana": (51.05, 25.05),
        "Halul": (52.42, 25.67), "Salwa": (50.82, 24.68)}

# LAC medoid -> nearest city, to sanity check regional assignment
print("\nper-LAC mapped medoid and nearest city:")
for lac, g in sites.groupby("LAC"):
    p = g[["LON_REAL","LAT_REAL"]].values
    med = np.median(p, axis=0)
    d = {k: np.hypot((med[0]-v[0])*np.cos(np.radians(25.4)), med[1]-v[1])*111 for k, v in CITY.items()}
    best = min(d, key=d.get)
    print("  LAC %5d n=%4d medoid (%.4f,%.4f) -> %s (%.1f km)  extent %.3fx%.3f deg" %
          (lac, len(p), med[0], med[1], best, d[best], np.ptp(p[:,0]), np.ptp(p[:,1])))

# ---------------- overlay ----------------
S = 1600
LO0, LO1, LA0, LA1 = 50.55, 51.80, 24.40, 26.30
def px(lon, lat): return ((lon-LO0)/(LO1-LO0)*S, S-(lat-LA0)/(LA1-LA0)*S)
im = Image.new("RGB", (S, S), "white"); dr = ImageDraw.Draw(im)
for g in np.arange(LO0, LO1, 0.1):
    x,_ = px(g, LA0); dr.line([(x,0),(x,S)], fill=(238,238,238))
for g in np.arange(LA0, LA1, 0.1):
    _,y = px(LO0, g); dr.line([(0,y),(S,y)], fill=(238,238,238))
dr.polygon([px(x,y) for x,y in RING], fill=(216,234,247), outline=(40,80,170))
cols = [(220,0,0),(0,140,0),(0,0,230),(230,140,0),(150,0,200),(0,170,170),(120,80,0),(255,0,150),
        (90,90,90),(0,90,0),(160,0,60),(0,0,120),(200,120,80),(70,140,220),(140,0,0),(0,60,60)]
lacs = sites.LAC.value_counts().index.tolist()
for lon, lat in m:
    x, y = px(lon, lat)
    if 0 <= x < S and 0 <= y < S:
        dr.ellipse([x-2.6, y-2.6, x+2.6, y+2.6], fill=(200,0,0))
for name, (lo, la) in CITY.items():
    x, y = px(lo, la)
    dr.line([(x-6,y),(x+6,y)], fill=(0,110,0)); dr.line([(x,y-6),(x,y+6)], fill=(0,110,0))
    dr.text((x+7,y-7), name, fill=(0,90,0))
im.save(os.path.join(HERE, "lacfit_overlay.png"))

# per-LAC colour plot
im2 = Image.new("RGB", (S, S), "white"); dr2 = ImageDraw.Draw(im2)
dr2.polygon([px(x,y) for x,y in RING], fill=(244,248,252), outline=(150,170,200))
for lac, g in sites.groupby("LAC"):
    col = cols[lacs.index(lac)] if lac in lacs[:16] else (180,180,180)
    for lon, lat in g[["LON_REAL","LAT_REAL"]].values:
        x, y = px(lon, lat)
        if 0 <= x < S and 0 <= y < S:
            dr2.ellipse([x-3, y-3, x+3, y+3], fill=col)
for i, lac in enumerate(lacs[:16]):
    dr2.text((8, 8+14*i), "LAC %g" % lac, fill=cols[i])
im2.save(os.path.join(HERE, "lacfit_byLAC.png"))
print("\nwrote overlays")

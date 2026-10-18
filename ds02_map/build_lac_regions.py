"""Build LAC/TAC regions for Qatar from OpenCelliD and encode them for the LAC map page.

Input: opencellid_427_qatar.csv.gz, every MCC 427 (Qatar) row of the OpenCelliD
dump that ClickHouse mirrors at
https://datasets-documentation.s3.eu-west-3.amazonaws.com/cell_towers/cell_towers.csv.xz
(snapshot up to Feb 2021; OpenCelliD data is CC BY-SA 4.0).

DS02's LAC_TAC values (110, 120, ..., 181 and 1110, 1130, ...) are the area codes of
network 427-02, so regions are built from that network only, one layer per radio.

Method, per layer: lay a 0.5 km grid over Qatar (same km frame as qatar_outline.json),
give each grid square the area code that wins a distance-weighted vote of its nearest
cells (k=9, within 6 km), and leave squares with no cell that close unassigned.
Squares are then dissolved into one (multi)polygon per code for drawing; the grid
itself is shipped too so the page classifies points with exactly the same rule.
"""
import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, box, mapping
from shapely.ops import unary_union

HERE = Path(__file__).resolve().parent
R_KM = 6371.0088
NET = 2
STEP, K, MAX_KM = 0.5, 9, 6.0
LAYERS = [("LTE", "LTE TAC"), ("UMTS", "UMTS LAC"), ("GSM", "GSM LAC")]

outline = json.loads((HERE / "qatar_outline.json").read_text())
lat0, lon0 = outline["lat0"], outline["lon0"]
qatar = Polygon(next(s for s in outline["shapes"] if s["name"] == "Qatar")["rings"][0])
land = qatar.buffer(1.0)   # keep coastal sites that the 1:10m coastline cuts off

cells = pd.read_csv(HERE / "opencellid_427_qatar.csv.gz")
cells = cells[cells.net == NET].copy()
cells["x"] = R_KM * np.radians(cells.lon - lon0) * np.cos(np.radians(lat0))
cells["y"] = R_KM * np.radians(cells.lat - lat0)

minx, miny, maxx, maxy = land.bounds
nx, ny = int(np.ceil((maxx - minx) / STEP)), int(np.ceil((maxy - miny) / STEP))
gx = minx + (np.arange(nx) + 0.5) * STEP
gy = miny + (np.arange(ny) + 0.5) * STEP
GX, GY = np.meshgrid(gx, gy)                       # row = y index, col = x index
centres = np.column_stack([GX.ravel(), GY.ravel()])
from shapely import contains_xy
on_land = contains_xy(land, centres[:, 0], centres[:, 1])

out = {"step": STEP, "x0": minx, "y0": miny, "nx": nx, "ny": ny, "net": f"427-0{NET}",
       "source": "OpenCelliD (CC BY-SA 4.0), ClickHouse mirror, snapshot to Feb 2021",
       "layers": []}
for radio, label in LAYERS:
    c = cells[(cells.radio == radio) & contains_xy(land.buffer(3), cells.x, cells.y)]
    tree = cKDTree(c[["x", "y"]].to_numpy())
    codes_arr = c.area.to_numpy()
    d, idx = tree.query(centres[on_land], k=K, distance_upper_bound=MAX_KM)
    land_codes = np.zeros(len(d), dtype=np.int64)
    for i in range(len(d)):
        ok = np.isfinite(d[i])
        if not ok.any():
            continue
        votes = {}
        for dist, j in zip(d[i][ok], idx[i][ok]):
            votes[codes_arr[j]] = votes.get(codes_arr[j], 0) + 1 / (dist + 0.25)
        land_codes[i] = max(votes, key=votes.get)
    grid_codes = np.zeros(len(centres), dtype=np.int64)
    grid_codes[on_land] = land_codes

    codes = sorted(int(v) for v in np.unique(grid_codes) if v)
    lut = {code: i + 1 for i, code in enumerate(codes)}      # 0 = unassigned
    grid = np.array([lut.get(int(v), 0) for v in grid_codes], dtype=np.uint8)

    regions = []
    for code in codes:
        sq = [box(x - STEP / 2, y - STEP / 2, x + STEP / 2, y + STEP / 2)
              for (x, y), v in zip(centres, grid_codes) if v == code]
        geom = unary_union(sq).simplify(0.15)
        polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
        rings = []
        for p in polys:
            rings.append([[round(a, 2), round(b, 2)] for a, b in p.exterior.coords])
            rings += [[[round(a, 2), round(b, 2)] for a, b in r.coords] for r in p.interiors]
        lp = max(polys, key=lambda p: p.area).representative_point()
        regions.append({"code": code, "rings": rings, "label": [round(lp.x, 2), round(lp.y, 2)],
                        "area_km2": round(geom.area, 1),
                        "n_cells": int((c.area == code).sum())})
    out["layers"].append({"radio": radio, "label": label, "codes": codes, "n_cells": len(c),
                          "grid": base64.b64encode(grid.tobytes()).decode(), "regions": regions})
    print(f"{label}: {len(c)} cells, {len(codes)} codes, "
          f"{(grid > 0).sum() * STEP ** 2:.0f} km2 assigned")

(HERE / "lac_regions.json").write_text(json.dumps(out, separators=(",", ":")))
print("wrote lac_regions.json", (HERE / "lac_regions.json").stat().st_size // 1024, "KB")

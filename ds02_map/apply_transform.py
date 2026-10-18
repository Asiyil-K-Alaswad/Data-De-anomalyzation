"""Apply a transform copied from the map page to every site and export estimated positions.

Usage: python apply_transform.py '<transform JSON from the page>' [out.csv]
Writes each site's position in the Qatar km frame and the implied lat/lon.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R_KM = 6371.0088

T = {"rot": 0, "sx": 1, "sy": 1, "tx": 0, "ty": 0, "flip": False, "latc": True}
T.update(json.loads(sys.argv[1]))
out = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "ds02_sites_fitted.csv"

sites = pd.read_csv(HERE / "ds02_sites_relative.csv")
outline = json.loads((HERE / "qatar_outline.json").read_text())
lat0, lon0 = outline["lat0"], outline["lon0"]

# Same steps, in the same order, as xf() in the page.
latk = np.cos(np.radians(lat0)) / np.cos(np.radians(sites.LAT.mean()))
# Scale in the cloud's own frame first, then rotate (rotation is rigid).
x = sites.X_KM * (latk if T["latc"] else 1) * (-1 if T["flip"] else 1) * T["sx"]
y = sites.Y_KM * T["sy"]
c, s = np.cos(np.radians(T["rot"])), np.sin(np.radians(T["rot"]))
fx = x * c - y * s + T["tx"]
fy = x * s + y * c + T["ty"]

sites["FIT_X_KM"] = fx.round(3)
sites["FIT_Y_KM"] = fy.round(3)
sites["EST_LAT"] = (lat0 + np.degrees(fy / R_KM)).round(6)
sites["EST_LON"] = (lon0 + np.degrees(fx / (R_KM * np.cos(np.radians(lat0))))).round(6)
sites.to_csv(out, index=False)
print(f"wrote {len(sites)} sites to {out}")

"""Build a relative-distance map of the DS02 site locations.

The coordinates in DS02 are anonymized (shifted somewhere on the globe), so the
map is drawn in kilometres relative to the network centroid: only the distances
between sites are meaningful, not their position on Earth.

Data fix applied: the LAT/LON columns are offset by one row from the rest of the
table (the file has one trailing row that carries only coordinates). Shifting
them up by one row makes 2,482 of 2,643 sites have a single consistent
location; the remaining sites differ by at most a few km between cells and are
reduced to their most common coordinate.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
EARTH_R_KM = 6371.0088

raw = pd.read_csv(ROOT / "ANOM_DS02_SiteLocations.csv")

# Realign the coordinate columns with their rows, then drop the coordinate-only tail row.
cells = raw.copy()
cells[["LAT", "LON"]] = raw[["LAT", "LON"]].shift(-1)
cells = cells.dropna(subset=["ANONYMIZED_SITECODE", "LAT", "LON"])

cells["SITETYPE"] = cells["SITETYPE"].str.strip().str.lower().map(
    {"outdoor": "Outdoor", "indoor": "Indoor"}).fillna("Other")


def mode(s):
    return s.value_counts().idxmax()


sites = cells.groupby("ANONYMIZED_SITECODE").agg(
    LAT=("LAT", mode), LON=("LON", mode),
    SITETYPE=("SITETYPE", mode), LAC_TAC=("LAC_TAC", mode),
    CELLS2G=("CELLS2G", "max"), CELLS4G=("CELLS4G", "max"), CELLS5G=("CELLS5G", "max"),
    N_CELLROWS=("ANOM_CELLCODE", "size"),
    N_LOCATIONS=("LAT", "nunique"),
).reset_index()

# Local equirectangular projection around the centroid: accurate for relative
# distances at this scale (network spans ~2 deg).
lat0, lon0 = np.radians(sites.LAT.mean()), np.radians(sites.LON.mean())
sites["X_KM"] = (EARTH_R_KM * (np.radians(sites.LON) - lon0) * np.cos(lat0)).round(3)
sites["Y_KM"] = (EARTH_R_KM * (np.radians(sites.LAT) - lat0)).round(3)
sites["TOTAL_CELLS"] = sites[["CELLS2G", "CELLS4G", "CELLS5G"]].sum(axis=1).astype(int)

# Nearest-neighbour distance per site (brute force is fine for ~2.6k points).
xy = sites[["X_KM", "Y_KM"]].to_numpy()
d = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
np.fill_diagonal(d, np.inf)
sites["NN_KM"] = d.min(axis=1).round(3)

sites.to_csv(OUT / "ds02_sites_relative.csv", index=False)

# ---- static PNG ----------------------------------------------------------
COLORS = {"Outdoor": "#2a78d6", "Indoor": "#eb6834", "Other": "#1baf7a"}
fig, ax = plt.subplots(figsize=(11, 11), dpi=150)
fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
for t in ["Outdoor", "Indoor", "Other"]:
    s = sites[sites.SITETYPE == t]
    ax.scatter(s.X_KM, s.Y_KM, s=8 + s.TOTAL_CELLS * 0.6, c=COLORS[t], alpha=0.75,
               edgecolors="#fcfcfb", linewidths=0.4, label=f"{t} ({len(s):,})")
ax.set_aspect("equal")
ax.grid(color="#e4e3df", linewidth=0.6); ax.set_axisbelow(True)
for sp in ax.spines.values():
    sp.set_color("#c9c8c2")
ax.tick_params(colors="#52514e")
ax.set_xlabel("km east of centroid", color="#52514e")
ax.set_ylabel("km north of centroid", color="#52514e")
ax.set_title(f"DS02 site locations — relative distances ({len(sites):,} sites)",
             loc="left", color="#0b0b0b", fontsize=14)
ax.legend(title="Site type (marker size = total cells)", frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "ds02_site_map.png")

# ---- data for the interactive HTML --------------------------------------
cols = ["ANONYMIZED_SITECODE", "X_KM", "Y_KM", "SITETYPE", "LAC_TAC", "CELLS2G",
        "CELLS4G", "CELLS5G", "TOTAL_CELLS", "NN_KM"]
rows = sites[cols].astype(object).values.tolist()
html = (OUT / "ds02_site_map.template.html").read_text()
(OUT / "ds02_site_map.html").write_text(html.replace("__DATA__", json.dumps(rows)))

print(sites[["X_KM", "Y_KM", "NN_KM"]].describe().round(2))
print(sites.SITETYPE.value_counts(), (sites.N_LOCATIONS > 1).sum(), "sites with >1 location")

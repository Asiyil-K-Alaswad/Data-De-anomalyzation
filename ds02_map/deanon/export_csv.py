"""Export DS02 with its de-anonymised coordinates, per cell and per site.

Run after fit.py:  python3 export_csv.py
Writes next to this file:
- DS02_deanonymised_per_cell.csv: every DS02 cell row as in ANOM_DS02_SiteLocations.csv (the
  coordinate-only trailing row is dropped, SITETYPE is normalised, see below), plus DEANON_LAT,
  DEANON_LON (the cell's site's de-anonymised position) and LAC_TAC_MATCH.
- DS02_deanonymised_per_site.csv: one row per site with the same DS02 columns except
  AZIMUTHSFORPLOT (a per-cell value), plus the same three columns.

SITETYPE is either Outdoor or Indoor, ignoring capitalisation: "outdoor" -> Outdoor; "indoor",
"Indoor-SO" and "Pico" -> Indoor (all Indoor-SO and Pico cells have azimuth 0, i.e. no
direction, as do all but 4 of the 11,864 indoor cells). Per site it is the type of the majority of the site's cells
(6 sites mix both; a tie would go to Outdoor).

LAC_TAC_MATCH: does the de-anonymised position fall in the OpenCelliD region (network 427-02,
built exactly as in build_lac_regions.py) that carries the cell's own code? 3-digit codes are
looked up in the LTE TAC region map (the map pages' default layer) and 4-digit codes in the GSM
LAC map. The UMTS map is not used: its OpenCelliD codes include stale ones DS02 never uses
(e.g. 28160, 10), which split up areas and would mark otherwise matching cells False. A position
in no region (at sea, or with no OpenCelliD cell within 6 km) is False. It is left blank when the code never appears in
that technology's OpenCelliD cells (codes 1, 10101, 14101, 15101), because there is no region
to compare with. Per site it is True only if every comparable code of the site matches.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import build_lac_regions as regions   # noqa: E402

R_KM = 6371.0088

raw = pd.read_csv(HERE.parent.parent / "ANOM_DS02_SiteLocations.csv", dtype=str, keep_default_na=False)
cells = raw[raw.ANONYMIZED_SITECODE != ""].copy()          # drop the coordinate-only last row
fit = pd.read_csv(HERE / "ds02_sites_deanonymised.csv").set_index("ANONYMIZED_SITECODE")
anon = pd.read_csv(HERE.parent / "ds02_sites_relative.csv").set_index("ANONYMIZED_SITECODE")

# One region map per technology; remember which codes each technology's cells carry.
grids = {}
for tech, radios in (("3d", ["LTE"]), ("4d", ["GSM"])):
    grid_codes, tech_cells = regions.vote_grid(radios)
    grids[tech] = (grid_codes, set(int(a) for a in tech_cells.area.unique()))

# Region code under every site's de-anonymised position, per technology.
x = R_KM * np.radians(fit.LON - regions.lon0) * np.cos(np.radians(regions.lat0))
y = R_KM * np.radians(fit.LAT - regions.lat0)
region_at = {tech: dict(zip(fit.index, regions.code_at(g, x, y))) for tech, (g, _) in grids.items()}


def match(site, code):
    """True/False, or None when OpenCelliD has no region for this code to compare with."""
    tech = "4d" if code >= 1000 else "3d"
    if code not in grids[tech][1]:
        return None
    return bool(region_at[tech][site] == code)


fmt = lambda v: f"{v:.7f}"
bool_txt = lambda b: "" if b is None else str(b)


def site_type(raw):
    t = raw.strip().lower()
    if t == "outdoor":
        return "Outdoor"
    if t in ("indoor", "indoor-so", "pico"):
        return "Indoor"
    raise ValueError(f"unknown SITETYPE {raw!r}")


cells["SITETYPE"] = cells.SITETYPE.map(site_type)


# ---- per cell ----------------------------------------------------------------
cell_match = [match(s, int(c)) for s, c in zip(cells.ANONYMIZED_SITECODE, cells.LAC_TAC)]
cells["DEANON_LAT"] = cells.ANONYMIZED_SITECODE.map(fit.LAT).map(fmt)
cells["DEANON_LON"] = cells.ANONYMIZED_SITECODE.map(fit.LON).map(fmt)
cells["LAC_TAC_MATCH"] = [bool_txt(m) for m in cell_match]
cells.to_csv(HERE / "DS02_deanonymised_per_cell.csv", index=False)


# ---- per site ----------------------------------------------------------------
def distinct(values):
    """The value if a site's cells agree, else the distinct values in order, ';'-joined."""
    return ";".join(dict.fromkeys(values))


rows = []
for site, g in cells.groupby("ANONYMIZED_SITECODE", sort=False):
    codes = sorted({int(c) for c in g.LAC_TAC})
    ms = [m for m in (match(site, c) for c in codes) if m is not None]
    rows.append({
        "ANONYMIZED_SITECODE": site,
        "ANOM_CELLCODE": ";".join(g.ANOM_CELLCODE),             # every cell, in DS02 order
        "LAC_TAC": ";".join(str(c) for c in codes),
        "SITETYPE": "Indoor" if (g.SITETYPE == "Indoor").sum() > (g.SITETYPE == "Outdoor").sum() else "Outdoor",
        "CELLS2G": distinct(g.CELLS2G),
        "CELLS4G": distinct(g.CELLS4G),
        "CELLS5G": distinct(g.CELLS5G),
        # anonymised site coordinate after the one-row LAT/LON offset fix (the fit's input)
        "LAT": f"{anon.LAT[site]:.8f}",
        "LON": f"{anon.LON[site]:.8f}",
        "DEANON_LAT": fmt(fit.LAT[site]),
        "DEANON_LON": fmt(fit.LON[site]),
        "LAC_TAC_MATCH": bool_txt(all(ms) if ms else None),
    })
sites = pd.DataFrame(rows)
sites.to_csv(HERE / "DS02_deanonymised_per_site.csv", index=False)

for name, col in (("per cell", cells.LAC_TAC_MATCH), ("per site", sites.LAC_TAC_MATCH)):
    vc = col.replace("", "blank").value_counts()
    print(f"{name}: {len(col)} rows, LAC_TAC_MATCH {vc.to_dict()}")

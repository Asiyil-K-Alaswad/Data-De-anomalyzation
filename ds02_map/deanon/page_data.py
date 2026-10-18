"""Turn the de-anonymisation fit (deanon_fit.json, written by fit.py) into data for
the de-anonymised map page, which reuses ds02_lac_map.template.html."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R_KM = 6371.0088
LAT0, LON0 = 25.3, 51.2                 # km frame shared with qatar_outline.json
QS = [0.10, 0.25, 0.50, 0.75, 0.90]


def to_km(lon, lat):
    return (R_KM * np.radians(np.asarray(lon) - LON0) * np.cos(np.radians(LAT0)),
            R_KM * np.radians(np.asarray(lat) - LAT0))


def anchor_envelopes(codes):
    """OpenCelliD 427-02 quantile envelope per code (GSM for 4-digit, UMTS+LTE otherwise)."""
    o = pd.read_csv(HERE.parent / "opencellid_427_qatar.csv.gz")
    o = o[o.net == 2]
    env = {}
    for code in codes:
        radios = ["GSM"] if code >= 1000 else ["UMTS", "LTE"]
        a = o[(o.area == code) & o.radio.isin(radios)][["lon", "lat"]].drop_duplicates()
        x, y = to_km(a.lon, a.lat)
        env[code] = ([round(float(v), 3) for v in np.quantile(x, QS)],
                     [round(float(v), 3) for v in np.quantile(y, QS)], len(a))
    return env


def deanon_page(sites, lac_rows, meta):
    fit = json.loads((HERE / "deanon_fit.json").read_text())
    B, c = np.array(fit["B_deg"]), np.array(fit["c_deg"])
    lonlat = B @ np.vstack([sites.LON.to_numpy(), sites.LAT.to_numpy()]) + c[:, None]
    x, y = to_km(lonlat[0], lonlat[1])
    cx, cy = float(np.mean(x)), float(np.mean(y))

    # Same row layout as the LAC page; points are stored about the fitted centroid
    # and the page's default transform moves them back onto their fitted positions.
    rows = [[r[0], round(float(xi - cx), 4), round(float(yi - cy), 4), *r[3:]]
            for r, xi, yi in zip(lac_rows, x, y)]

    groups = sorted(fit["groups"], key=lambda g: int(g["code"]))
    env = anchor_envelopes([int(g["code"]) for g in groups])
    s = fit["loo_summary"]
    on_land = fit["on_land_pct"]
    sv = fit["singular_values_km"]
    page_meta = {
        **meta,
        "storageKey": "ds02-deanon-fit",
        "defaultT": {"tx": round(cx, 4), "ty": round(cy, 4), "latc": False},
        "fit": {
            "offLand": fit["off_land_ids"],
            "groups": [{"code": int(g["code"]),
                        "tech": "GSM" if int(g["code"]) >= 1000 else "UMTS + LTE",
                        "n_sites": int(g["n_sites"]), "n_anchor": env[int(g["code"])][2],
                        "res": g["residual_km"], "loo": g["loo_residual_km"],
                        "ex": env[int(g["code"])][0], "ey": env[int(g["code"])][1]}
                       for g in groups],
            "tiles": [
                [f"{s['median']:.1f} km", "median held-out error: refit without a code's anchor, "
                                          "then measure how far that code's sites land from it"],
                [f"{s['p90']:.1f} / {s['max']:.1f} km", "90th percentile / worst held-out error"],
                [f"{s['doha_core_median']:.1f} km", f"held-out error in the Doha core "
                                                    f"({', '.join(str(g) for g in s['doha_core_groups'])})"],
                [f"{on_land:.1f}%", f"of sites land on Qatar's land; {len(fit['off_land_ids'])} "
                                    "do not and are ringed on the map"],
            ],
            "text": fit["page_text"],
            "note": fit["page_note"],
            "B_km": fit["B_km"], "singular_values_km": sv,
        },
    }
    text = ("DS02 De-anonymised Map",
            "DS02 sites at their de-anonymised positions",
            "Each site is moved from its anonymised coordinate to an estimated real position in Qatar by "
            "one affine map (rotation, stretch, shear and shift) fitted so that each LAC/TAC code's sites "
            "spread like that code's cells in OpenCelliD. Colours show whether a site now sits in an "
            "OpenCelliD region carrying its own code. You can still adjust the points; Reset transform "
            "returns them to the fit.")
    return rows, page_meta, text

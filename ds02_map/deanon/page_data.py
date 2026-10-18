"""Turn the de-anonymisation fit (deanon_fit.json from fit.py) and its hold-out checks
(holdout.json from holdout.py) into data for the de-anonymised map page, which reuses
ds02_lac_map.template.html."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R_KM = 6371.0088
LAT0, LON0 = 25.3, 51.2                 # km frame shared with qatar_outline.json
QS = [0.10, 0.25, 0.50, 0.75, 0.90]
PAIRS = {110: 1110, 130: 1130, 151: 1151, 160: 1160, 171: 1171}


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
    ho = json.loads((HERE / "holdout.json").read_text())["summary"]
    code_out, area_out = ho["A_spec_leakfree"], ho["C_leave_area_out"]

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
    off = fit["off_land_detail"]
    n_off, n_off_1km = len(off), fit["n_off_land_1km"]
    far = [o for o in off if o["dist_km"] > 5]
    sv, mo = fit["singular_values_km"], fit["moments"]
    ins = fit["insample_summary"]
    f1 = lambda v: f"{v:.1f}"

    text = (
        "Every site is moved by one affine map, <i>real = B · anonymised + c</i> (six numbers: "
        "rotation, two stretches, shear and a shift), and nothing else. It is fitted so that, for each "
        "of the 20 LAC/TAC codes that also appear in OpenCelliD, the spread of that code's DS02 sites "
        "(10th, 25th, 50th, 75th and 90th percentiles east–west and north–south) matches the spread of "
        "that code's Vodafone Qatar cells: GSM cells for the 4-digit codes, UMTS and LTE cells for the "
        "3-digit ones. Smaller terms keep the whole cloud's spread close to Vodafone's, and push sites "
        "onto land and inside Vodafone's coverage. "
        f"The best map mirrors the anonymised cloud (the best unmirrored map scores "
        f"{fit['orientation']['ratio_objective']:.1f}× worse). It stretches the cloud's thin axis about "
        f"{sv[0] / sv[1]:.0f} times more than its long one (singular values {sv[0]:.2f} and {sv[1]:.3f} km "
        "per anonymised km), which is why the anonymised points looked like a thin line. "
        "Three implementations written separately, each with over 60 random starts, reached the same "
        "map to within centimetres, and an independent search with 240 more starts found nothing better.")
    note = (
        f"<b>Held-out error</b>: refit without a code's OpenCelliD anchor, then measure how far that "
        f"code's DS02 median lands from the anchor median. The five 2G/4G pairs "
        f"({', '.join(f'{a}/{b}' for a, b in PAIRS.items())}) share 93–100% of their sites, so "
        "holding out one code of a pair while its partner stays in the fit is nearly in-sample. The "
        "\"whole area\" column holds out both codes of a pair together, and the headline numbers use it. "
        f"Holding out one code at a time gives median {f1(code_out['median'])}, 90th percentile "
        f"{f1(code_out['p90'])} and worst {f1(code_out['max'])} km. "
        f"<b>On land</b> uses the Natural Earth 1:10m coastline, which leaves out reclaimed land such as "
        f"The Pearl and West Bay; only {n_off_1km} of the {n_off} off-land sites are more than 1 km "
        f"out, and {len(far)} of those, all coded {', '.join(sorted({o['lac_tacs'] for o in far}))}, sit "
        f"{f1(min(o['dist_km'] for o in far))}–{f1(max(o['dist_km'] for o in far))} km offshore, where "
        "Vodafone has no cells, so the fit is clearly wrong for them. "
        f"<b>Spread</b>: the fitted cloud measures {f1(mo['fit']['sigma_x'])} km east–west and "
        f"{f1(mo['fit']['sigma_y'])} km north–south (standard deviation), against "
        f"{f1(mo['ref']['sigma_x'])} and {f1(mo['ref']['sigma_y'])} km for Vodafone's cells. "
        "OpenCelliD positions are crowd-sourced estimates from up to Feb 2021, and the method assumes "
        "the anonymisation was a single affine warp.")

    page_meta = {
        **meta,
        "storageKey": "ds02-deanon-fit",
        "defaultT": {"tx": round(cx, 4), "ty": round(cy, 4), "latc": False},
        "fit": {
            "offLand": [[o["id"], round(o["dist_km"], 2)] for o in off],
            "groups": [{"code": int(g["code"]),
                        "tech": "GSM" if int(g["code"]) >= 1000 else "UMTS + LTE",
                        "n_sites": int(g["n_sites"]), "n_anchor": env[int(g["code"])][2],
                        "res": g["residual_km"],
                        "loo": code_out["per_group"][str(g["code"])],
                        "looArea": area_out["per_group"][str(g["code"])],
                        "ex": env[int(g["code"])][0], "ey": env[int(g["code"])][1]}
                       for g in groups],
            "tiles": [
                [f"{f1(area_out['median'])} km", "median held-out error across the 20 codes "
                                                 "(whole-area hold-out)"],
                [f"{f1(area_out['p90'])} / {f1(area_out['max'])} km",
                 "90th percentile / worst held-out error"],
                [f"{f1(area_out['doha_core_median'])} km",
                 f"held-out error in the Doha core ({', '.join(str(g) for g in ho_core(fit))})"],
                [f"{f1(ins['min'])}–{f1(ins['max'])} km", "in-sample error range (fit includes the anchor)"],
                [f"{fit['on_land_pct']:.1f}%", f"of sites on Qatar's land; the {n_off} others are "
                                               "ringed on the map"],
            ],
            "text": text,
            "note": note,
        },
    }
    page_text = ("DS02 De-anonymised Map",
                 "DS02 sites at their de-anonymised positions",
                 "Each site is moved from its anonymised coordinate to an estimated real position in Qatar "
                 "by one affine map, fitted so that each LAC/TAC code's sites spread like that code's cells "
                 "in OpenCelliD. Colours show whether a site now sits in an OpenCelliD region carrying its "
                 "own code. You can still adjust the points; Reset transform returns them to the fit.")
    return rows, page_meta, page_text


def ho_core(fit):
    return fit["loo_summary"]["doha_core_groups"]

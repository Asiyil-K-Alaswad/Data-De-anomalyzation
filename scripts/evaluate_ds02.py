"""Reproducible quality/utility evaluation of ANOM_DS02_SiteLocations.csv.

Usage:
    python3 scripts/evaluate_ds02.py [--data-dir .] [--out corrected_sites.csv]

Prints every figure quoted in DS02_EVALUATION.md. With --out, also writes a
corrected one-row-per-site table (LAT/LON realigned, SITETYPE normalised).
Requires only pandas and numpy.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KM_PER_DEG = 111.32


def to_km(lat, lon, lat0, lon0):
    return np.c_[(lat - lat0) * KM_PER_DEG,
                 (lon - lon0) * KM_PER_DEG * np.cos(np.radians(lat0))]


def pairwise(X):
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    return D


def knn_label_purity(X, labels, k=5):
    nb = np.argsort(pairwise(X), 1)[:, :k]
    return (labels[nb] == labels[:, None]).mean()


def section(title):
    print(f"\n=== {title} ===")


def load_ds01_site_metrics(data_dir):
    parts = sorted(data_dir.glob("ANOM_DS01_Rectifier_Data Part *.csv"))
    header = pd.read_csv(parts[0], nrows=0).columns.tolist()
    want = ["SITECODE", "D.G. Supply (kWh)", "Mains Outage Duration (h)",
            "Solar Supply (kWh)", "Avg DC Load Power (kW)", "TRAFFIC_GB"]
    idx = [header.index(c) for c in want]
    frames = []
    for p in parts:
        # Parts 2 and 3 ship without a header row.
        has_header = pd.read_csv(p, nrows=0).columns[0] == "SITECODE"
        t = pd.read_csv(p, header=0 if has_header else None, usecols=idx)
        t.columns = [header[i] for i in sorted(idx)]
        frames.append(t)
    r = pd.concat(frames)
    for c in want[1:]:
        r[c] = pd.to_numeric(r[c], errors="coerce")
    return r.groupby("SITECODE").agg(
        dg_days=("D.G. Supply (kWh)", lambda s: (s > 0).mean()),
        mains_outage_h=("Mains Outage Duration (h)", "mean"),
        solar_days=("Solar Supply (kWh)", lambda s: (s > 0).mean()),
        dc_load_kw=("Avg DC Load Power (kW)", "mean"),
        traffic_gb=("TRAFFIC_GB", "mean"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    df = pd.read_csv(args.data_dir / "ANOM_DS02_SiteLocations.csv")

    section("1. Structure")
    blank = df.ANONYMIZED_SITECODE.isna()
    print(f"rows={len(df)}  cols={df.shape[1]}  rows with no site/cell id={blank.sum()} "
          f"(index {list(df.index[blank])})")
    sc = df.ANONYMIZED_SITECODE.dropna()
    print(f"sites={sc.nunique()}  cells={df.ANOM_CELLCODE.nunique()}  "
          f"site row-runs={(sc != sc.shift()).cumsum().nunique()} (rows are interleaved, not grouped)")

    section("2. LAT/LON row misalignment")
    def coords_per_site(lat, lon):
        t = df.assign(ll=list(zip(lat, lon))).dropna(subset=["ANONYMIZED_SITECODE"])
        return t.groupby("ANONYMIZED_SITECODE").ll.nunique()
    raw_n = coords_per_site(df.LAT, df.LON)
    up_n = coords_per_site(df.LAT.shift(-1), df.LON.shift(-1))
    down_n = coords_per_site(df.LAT.shift(1), df.LON.shift(1))
    for name, n in [("as shipped", raw_n), ("LAT/LON shifted up 1", up_n), ("shifted down 1", down_n)]:
        print(f"{name:22s}: sites with exactly 1 coordinate = {(n == 1).sum():4d} / {len(n)}  "
              f"(median distinct coords/site = {n.median():.0f})")
    df["LAT"], df["LON"] = df.LAT.shift(-1), df.LON.shift(-1)
    d = df.dropna(subset=["ANONYMIZED_SITECODE"]).copy()
    d["SITETYPE"] = d.SITETYPE.str.strip().str.lower()

    raw = pd.read_csv(args.data_dir / "ANOM_DS02_SiteLocations.csv").dropna(subset=["ANONYMIZED_SITECODE"])
    print(f"rows whose shipped coordinate differs from the realigned one: "
          f"{((raw.LAT.values != d.LAT.values) | (raw.LON.values != d.LON.values)).sum()} / {len(d)}")

    lat0, lon0 = d.LAT.mean(), d.LON.mean()
    g = d.groupby("ANONYMIZED_SITECODE")
    site = g.agg(lat=("LAT", lambda s: s.mode()[0]), lon=("LON", lambda s: s.mode()[0]),
                 sitetype=("SITETYPE", lambda s: s.mode()[0]),
                 lac=("LAC_TAC", lambda s: s.mode()[0]), n_rows=("ANOM_CELLCODE", "size"),
                 cells2g=("CELLS2G", "first"), cells4g=("CELLS4G", "first"), cells5g=("CELLS5G", "first"))
    first = raw.drop_duplicates("ANONYMIZED_SITECODE").set_index("ANONYMIZED_SITECODE").loc[site.index]
    err = np.hypot(*to_km(first.LAT.values, first.LON.values, lat0, lon0).T
                   - to_km(site.lat.values, site.lon.values, lat0, lon0).T)
    print(f"naive 'first row per site' lookup on shipped file: {(err > 0.001).sum()} / {len(site)} sites wrong, "
          f"median error {np.median(err):.1f} km, p90 {np.percentile(err, 90):.1f} km, max {err.max():.0f} km")

    dev = np.hypot(*(to_km(d.LAT.values, d.LON.values, lat0, lon0)
                     - to_km(g.LAT.transform(lambda s: s.mode()[0]).values,
                             g.LON.transform(lambda s: s.mode()[0]).values, lat0, lon0)).T)
    print(f"after realignment: {(up_n > 1).sum()} sites still have >1 coordinate; "
          f"{(dev > 0).sum()} rows deviate from their site's modal point "
          f"(median {np.median(dev[dev > 0]) * 1000:.0f} m, max {dev.max():.1f} km); "
          f"sites with a cell >1 km from the modal point: {d.ANONYMIZED_SITECODE[dev > 1].nunique()}")

    section("3. Attribute quality")
    raw_types = raw.SITETYPE.value_counts().to_dict()
    print(f"SITETYPE raw labels: {raw_types}")
    print(f"sites with mixed SITETYPE before/after case-folding: "
          f"{(raw.groupby('ANONYMIZED_SITECODE').SITETYPE.nunique() > 1).sum()} / {(g.SITETYPE.nunique() > 1).sum()}")
    pairs = g.LAC_TAC.agg(lambda s: tuple(sorted(s.unique().astype(int))))
    print(f"LAC_TAC values per site: {pairs.map(len).value_counts().sort_index().to_dict()}")
    print(f"most common 2-value pairs: {pairs[pairs.map(len) > 1].value_counts().head(6).to_dict()}")
    print(f"rare LAC_TAC codes (<=30 rows): {d.LAC_TAC.value_counts()[lambda s: s <= 30].to_dict()}")
    az = d.AZIMUTHSFORPLOT
    print(f"azimuth range {az.min():.0f}-{az.max():.0f}; share == 0 by type: "
          f"{d.groupby('SITETYPE').AZIMUTHSFORPLOT.apply(lambda s: round((s == 0).mean(), 3)).to_dict()}")
    dup = d[d.ANOM_CELLCODE.duplicated(keep=False)]
    print(f"duplicated ANOM_CELLCODE: {dup.ANOM_CELLCODE.nunique()} codes on {dup.ANONYMIZED_SITECODE.nunique()} different sites")
    tot = site[["cells2g", "cells4g", "cells5g"]].sum(axis=1)
    print(f"rows/site == CELLS2G+4G+5G for {(site.n_rows == tot).sum()} / {len(site)} sites "
          f"(column semantics undocumented)")

    section("4. Geometry of the anonymised coordinates")
    X = to_km(site.lat.values, site.lon.values, lat0, lon0)
    X = X - X.mean(0)
    _, sv, vt = np.linalg.svd(X, full_matrices=False)
    P = X @ vt.T
    print(f"bounding box lat [{site.lat.min():.2f}, {site.lat.max():.2f}] lon [{site.lon.min():.2f}, {site.lon.max():.2f}] "
          f"(open South Atlantic - not a real-world placement)")
    print(f"corr(lat, lon) = {np.corrcoef(site.lat, site.lon)[0, 1]:.4f}; "
          f"PC1 explains {sv[0] ** 2 / (sv ** 2).sum():.2%}; PC1 std {P[:, 0].std():.1f} km vs PC2 std {P[:, 1].std():.1f} km")
    lac = site.lac.values
    for stretch in [1, 2, 4, 8, 16, 32]:
        print(f"  5-NN same-LAC with PC2 stretched x{stretch:<2}: {knn_label_purity(np.c_[P[:, 0], P[:, 1] * stretch], lac):.3f}")
    print(f"  random-label baseline: {np.mean([(rng.permutation(lac) == lac).mean() for _ in range(50)]):.3f}")
    first_xy = to_km(first.LAT.values, first.LON.values, lat0, lon0)
    print(f"  5-NN same-LAC using naive first-row coords: {knn_label_purity(first_xy, lac):.3f}")
    D = pairwise(X)
    nn = D.min(1)
    print(f"NN distance km p5/p25/p50/p75/p95: {np.round(np.percentile(nn, [5, 25, 50, 75, 95]), 3).tolist()}")
    shared = site.groupby(["lat", "lon"]).size()
    print(f"points shared by >=2 sites: {(shared > 1).sum()} ({shared[shared > 1].sum()} sites, max {shared.max()}); "
          f"types at shared points: {site.set_index(['lat', 'lon']).loc[shared[shared > 1].index].sitetype.value_counts().to_dict()}")
    print(f"isolated sites (NN > 5 km): {site.index[nn > 5].tolist()} NN={np.round(nn[nn > 5], 1).tolist()} km")

    section("5. Linkage and predictive signal")
    ds1 = load_ds01_site_metrics(args.data_dir)
    ds3 = set(pd.read_csv(args.data_dir / "ANOM_DS03_SiteInventory_in_.csv", usecols=["CODE"], dtype=str).CODE.dropna())
    s2 = set(site.index)
    print(f"DS01 sites with a DS02 location: {len(s2 & set(ds1.index))} / {len(ds1)}; "
          f"DS03 sites with a DS02 location: {len(s2 & ds3)} / {len(ds3)}")
    j = site.join(ds1, how="inner")
    Xj = to_km(j.lat.values, j.lon.values, lat0, lon0)
    nb = np.argsort(pairwise(Xj), 1)[:, :5]
    for f in ["traffic_gb", "dg_days", "mains_outage_h", "solar_days", "dc_load_kw"]:
        v = j[f].values

        def nbr_corr(vals):
            m = np.array([np.nanmean(vals[row]) if np.isfinite(vals[row]).any() else np.nan for row in nb])
            ok = np.isfinite(vals) & np.isfinite(m)
            return np.corrcoef(vals[ok], m[ok])[0, 1], ok.sum()
        c, n = nbr_corr(v)
        null = [nbr_corr(v[rng.permutation(len(v))])[0] for _ in range(30)]
        print(f"  {f:15s} corr(site, mean of 5 nearest) = {c:+.3f}   permutation null {np.mean(null):+.3f} +/- {np.std(null):.3f}  (n={n})")

    if args.out:
        site.drop(columns=["n_rows"]).to_csv(args.out)
        print(f"\nwrote corrected site table -> {args.out}")


if __name__ == "__main__":
    main()

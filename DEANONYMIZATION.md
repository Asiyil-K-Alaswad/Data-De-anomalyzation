# DS02 — de-anonymisation of site locations (Qatar)

**Deliverables**

| file | content |
|---|---|
| `DS02_sites_deanonymised.csv` | 2,643 sites: recovered `LAT_REAL`/`LON_REAL` + site attributes |
| `DS02_cells_deanonymised.csv` | 49,693 cell rows: recovered coordinates + original warped values |
| `DS02_deanonymised_map.png` | mapped sites over Qatar's outline |
| `analysis/transform_tech.json` | the fitted inverse transform (current best) |
| `analysis/loo_results.csv`, `analysis/loo_v2.csv`, `analysis/insample_residuals.csv` | validation |

## 1. What the file is

Cell-level export: **49,693 rows = 49,690 cells**, grouped into **2,643 sites** (median 21 cells/site).
Rows are *not* ordered by site (21,522 site blocks; a site appears in ~9 chunks), and the
`LAT`/`LON` columns are **offset by one row** relative to the rest of the record — taking each
row's coordinate from the following row makes 93.9 % of sites carry exactly one coordinate
(2.7 % before). All processing below uses that shift-corrected site coordinate.

`LAC_TAC` is an **area** code (24 values, 1–361 sites each), not a site/cell identifier. 1,267 sites
carry two codes because their 2G cells sit in one area and their 3G/4G cells in another
(`TAC = LAC + 1000` in five areas). 4-digit codes (1110/1130/1151/1160/1171) are GSM areas,
3-digit codes (110…181) are UMTS/LTE areas.

## 2. The anonymisation

The coordinates are a **strongly anisotropic affine warp** of the true positions:

```
real_lon = -2.98390 * lon_a + 3.31307 * lat_a + 61.04593
real_lat =  4.86946 * lon_a - 4.45520 * lat_a + 57.36956
```

Singular values **7.96** and **0.357** (condition ≈ 22). The warp preserves relative geometry —
same-LAC sites stay clustered (nearest-neighbour LAC agreement 0.89 vs 0.08 by chance) — but
squashes local shape into a ~20:1 strip in the South Atlantic, which is why it looks nothing
like Qatar.

**Operator identified: Vodafone Qatar (MCC 427 / MNC 02).** DS02's `LAC_TAC` values coincide with
OpenCelliD's `area` codes for `net=2`; OpenCelliD holds 46,612 Qatar cells (18,985 Vodafone,
14,850 distinct coordinates) and provides the geographic anchors.

## 3. Fit

Affine inverse fitted by minimising, per anchor group, the mismatch between the **mapped
quantile envelope** (10/25/50/75/90 % of lon and lat) of the DS02 sites and the corresponding
OpenCelliD envelope, plus global distribution constraints and a land-mask constraint.
20 anchor groups: **15 UMTS+LTE (3-digit codes)** and **5 GSM (4-digit codes)** — i.e. anchors are
resolved *per technology*, which is the main improvement over the first attempt.

## 4. Validation

**In-sample, per anchor group (km):** 1.09 (121), 1.42 (143), 1.57 (150), 1.93 (142), 1.96 (141),
2.22 (120, 161), 3.23 (181), 3.35 (160), 3.38 (1171), 3.95 (131), 4.04 (111), 4.37 (1160),
4.42 (151), 5.31 (1151), 6.37 (1130), 7.81 (110), 8.99 (1110), 10.82 (130).

**Leave-one-group-out** (drop a group's envelope, re-fit on the other 19 with its sites also
removed from the global terms, then score the held-out group):

| metric | value |
|---|---|
| held-out residual, median | **3.60 km** |
| mean / p90 / max | 4.19 / 7.38 / **10.85 km** (LAC 130, far west) |
| held-out region-centre offset, median / max | **4.50 / 11.66 km** |
| Doha-core folds (120,121,141,142,143,150,161,181) | **1.96 km** median |
| retained-anchor loss of LOO fit vs full fit | ≤ full fit in every fold (gap −0.0 … −12.7 %) |

**Global checks:** 99.5 % of sites land inside Qatar's polygon; mapped spread σ_lon 7.19 km
(reference 7.74) and σ_lat 16.60 km (reference 16.06); correlation −0.37 vs −0.16.

**Optimiser caveat (important).** A first LOO run that started each fold cold reported a median
held-out error of **27 km**. That was an artefact: the objective has many local optima (a direct
consequence of the condition number of 22), and cold starts converged to wrong basins. Reporting
both runs is deliberate — the second (warm + cold starts, best retained-anchor fit) is the valid
number, but the first is a warning that re-fitting this problem naively is unreliable.

## 5. Accuracy you can rely on

* **Doha / Lusail / Al Wakrah core: ~1–2 km.**
* **Typical national accuracy: ~4 km**, up to **~11 km** in the far west (Dukhan) and the
  north-east (Al Khor→Ras Laffan).
* Errors are **anisotropic**: the compressed direction of the warp is the weak one, so
  east–west position is markedly less certain than north–south.
* **13 sites (0.5 %) still map offshore**, up to 50 km — a residual model error, flagged in
  `OFF_LAND_KM` and `GEO_METHOD`. Treat those as region-level only.
* Region assignment (which LAC/TAC area, which part of the country) is reliable; **exact site
  placement is not**. Do not treat the coordinates as surveyed positions.

## 6. Open issues

1. Residuals grow with distance from Doha (1 km → 11 km), which is the signature of a **projective**
   component in the warp. An 8-parameter homography was set up but not converged; it is the most
   promising way to cut the peripheral error.
2. The technology mapping (4-digit = GSM, 3-digit = UMTS/LTE) is inferred from cell counts per site
   (≈3.3 vs ≈17 cells) and confirmed by the OpenCelliD radio mix, but it is an inference, not a
   documented field.
3. The transform is calibrated against **OpenCelliD** coverage, which is crowd-sourced and
   incomplete; it is not calibrated against the operator's true site list.
4. `LAT_WARPED_FILE` / `LON_WARPED_FILE` in the cell file are the **raw, misaligned** values kept for
   traceability — do not use them as coordinates.

## 7. Reuse

The `SITE__` hashes are shared with DS01 (rectifier, 1,570 sites, 1,536 of them also in DS02),
DS03 (inventory) and are consistent across files, so these coordinates join at site level to the
other datasets. DS01 has no cell identifiers; cell-level joins only exist inside DS02.

# DS02: anonymized telecom site locations, evaluation

File: `ANOM_DS02_SiteLocations.csv` (49,693 rows × 10 columns, 3.9 MB)
Reproduce every number below with `python3 scripts/evaluate_ds02.py` (pandas + numpy, about 30 s).

## Verdict

**Usable, but only after a mandatory fix and only for relative-position work.**

- **As shipped, the coordinates are wrong.** The `LAT`/`LON` columns are shifted down one row relative to the other columns, so 21,935 of 49,692 cell rows (44%) carry another row's coordinate. A naive "first row per site" lookup puts 2,506 of 2,643 sites (95%) in the wrong place, with a median error of 6.4 km and a maximum of 147 km. This is a one-line fix.
- **After the fix, the positions are good for neighbourhood and graph features.** Nearby sites share their location area (LAC/TAC) and have similar traffic, generator use and solar use, far more often than chance. The file also joins cleanly to DS01 (98% of sites) and DS03 (96%).
- **It is not suitable for anything that needs real distances, directions or real-world geography.** The anonymization squeezed the site cloud almost onto a line (lat/lon correlation 0.99), and the coordinates sit in the open South Atlantic.

## What the file contains

| Column | Content | Notes |
|---|---|---|
| `ANONYMIZED_SITECODE` | hashed site id (2,643 sites) | same id space as DS01 `SITECODE` and DS03 `CODE` |
| `ANOM_CELLCODE` | hashed cell id (49,690 unique) | one row per cell |
| `AZIMUTHSFORPLOT` | sector azimuth in degrees, 0–355 | 0 for every indoor/pico cell |
| `LAC_TAC` | location/tracking area code (24 values) | 2G LAC = 1000 + TAC (e.g. 160 ↔ 1160) |
| `SITETYPE` | Outdoor / Indoor / Indoor-SO / Pico | inconsistent capitalization |
| `CELLS2G/4G/5G` | per-site counts | constant per site; meaning undocumented |
| `LAT`, `LON` | anonymized coordinates, 7–8 decimals | **misaligned by one row** |

Rows for a site are interleaved through the file (21,521 runs for 2,643 sites), not grouped together.

## Findings

### 1. LAT/LON are offset by one row (critical, fixable)

Evidence:
- The last line of the file has empty values in every column except `LAT`/`LON`.
- Number of sites whose cells all share one coordinate:

| Alignment | Sites with exactly one coordinate | Median coordinates per site |
|---|---|---|
| As shipped | 72 / 2,643 | 8 |
| `LAT`/`LON` shifted **up** one row | **2,482 / 2,643** | **1** |
| Shifted down one row | 70 / 2,643 | 10 |

- After shifting, 5-nearest-neighbour LAC agreement rises from 0.17 (naive first-row coordinates) to 0.82.
- **Fix:** `df[['LAT','LON']] = df[['LAT','LON']].shift(-1)`, then drop the trailing empty row. The first row's original coordinate is left without a row and is discarded.

Residual spread after the fix is small and plausible:
- 161 sites still have more than one coordinate.
- 1,507 rows (3%) differ from their site's most common coordinate, with a median offset of 119 m.
- Only 14 sites have a cell more than 1 km from the site centre; the maximum is 6.9 km.

Taking the most common coordinate per site is a safe way to get one point per site.

### 2. The geometry is strongly distorted (major, not fully fixable)

- `corr(LAT, LON) = 0.9915`. The first principal component explains 99.6% of the variance.
- The site cloud is about 130–150 km long (spread 21.4 km along the main axis) but only 1.3 km across.
- A network of 2,643 sites, a third of them indoor, cannot really sit in a strip about 3 km wide. The cross-axis has been compressed.
- Location areas confirm it. If the cross-axis is stretched back, 5-nearest-neighbour LAC agreement **improves**: 0.824 at ×1, 0.901 at ×4, 0.920 at ×8, 0.927 at ×16 (random baseline 0.084). The across-axis direction therefore carries real information at roughly a tenth to a sixteenth of its true scale.
- Nearest-neighbour distances are implausibly short: median 85 m, 5th percentile 0 m.
- **Consequence:** neighbour rankings (which sites are near each other) are largely preserved, so the relative structure is real. Absolute distances, areas, densities and bearings are not. The `AZIMUTHSFORPLOT` sectors cannot be drawn meaningfully on this map, because the map's orientation and shear relative to true north are unknown.
- **Mitigation, if needed:** project onto the principal axes and stretch the minor axis by about 8–16×. Use rank-based neighbours (k nearest, or a k-NN graph) rather than distance thresholds in km.

### 3. Not georeferenced (by design)

- The bounding box is roughly lat −50.9 to −49.0, lon −53.2 to −51.3: open ocean east of Patagonia.
- You cannot join these coordinates to terrain, population, land-use, road, weather-grid or public cell-tower data.
- DS04 (weather) has no location column, so it is one regional series and cannot be matched per site anyway.

### 4. Attribute quality (minor)

| Issue | Size | Handling |
|---|---|---|
| `SITETYPE` capitalization (`INDOOR`, `outdoor`) | 178 sites with mixed labels; 9 after case-folding | lowercase; take the most common value per site |
| Azimuth 0 on outdoor cells (could mean north or missing) | 3.7% of outdoor rows | treat as missing unless the site's other sectors back it up |
| Duplicate `ANOM_CELLCODE` across different sites | 2 codes, 4 sites | likely hash collisions; don't use cell code alone as a key |
| Rare LAC/TAC codes `1`, `10101`, `15101` | 42 rows | check before using LAC as a categorical feature |
| `CELLS2G+4G+5G` ≠ rows per site | true for 2,194 / 2,643 sites | semantics unknown; don't assume it equals the number of cells |
| Several sites at exactly the same point | 86 points, 278 sites (262 indoor), up to 14 sites per point | probably buildings with several indoor systems; dedupe if needed |
| Isolated site | `SITE__872974886`, 20.5 km from its nearest neighbour | possible outlier; check it |
| No technology or band column per cell | – | cell-level work is limited to azimuth and LAC/TAC |

### 5. Linkage and predictive value (positive)

- 1,536 of 1,570 DS01 sites (97.8%) and 2,641 of 2,757 DS03 sites (95.8%) have a DS02 location.
- **Spatial signal in DS01 metrics.** The test correlates each site's value with the average of its 5 nearest neighbours, using corrected coordinates. The null figure comes from shuffling the values across sites 30 times.

| Site-level metric (DS01) | Neighbour correlation | Shuffled null |
|---|---|---|
| Mean daily traffic (GB) | **+0.57** | +0.01 ± 0.04 |
| Share of days on diesel generator | **+0.30** | 0.00 ± 0.03 |
| Share of days with solar input | **+0.18** | 0.00 ± 0.03 |
| Mean mains outage hours | −0.03 | −0.02 ± 0.02 |
| Mean DC load (kW) | +0.01 | −0.01 ± 0.01 |

Location carries real information about demand and power-source mix (grid, generator, solar). It says nothing about equipment load or, in this data, about outage duration.

## Fitness for purpose

| Use | Fit | Why |
|---|---|---|
| Join a location or location area to DS01/DS03 as site features | ✅ after the shift fix | clean keys, about 96–98% coverage |
| Neighbourhood features (k-NN averages, spatial lags, site graph for a GNN) | ✅ | neighbour structure is preserved and predictive |
| Clustering or regionalization (LAC-like zones, rural/urban proxies) | ✅ | spatially coherent |
| Spatial or grouped cross-validation to avoid leakage | ✅ | group by LAC/TAC or by blocks along the main axis |
| Anomaly detection using peer sites (compare a site with its neighbours) | ✅ | strong spatial signal for traffic and generator use |
| Distance-based work: inter-site distance, coverage or propagation, density per km², drive-time or fuel logistics | ❌ | metric scale distorted about 10× across one axis |
| Sector or azimuth geometry: coverage wedges, beam-based neighbour relations | ❌ | orientation of the map relative to north is unknown |
| Joining external geodata (terrain, population, weather grid, tower databases) | ❌ | synthetic placement |
| Cell-level RF modelling | ⚠️ limited | no technology/band column; cell-id collisions |

## Recommended preprocessing

1. Shift `LAT`/`LON` up one row and drop the trailing empty row.
2. Collapse to one row per site using the most common coordinate. Lowercase `SITETYPE` and take its most common value.
3. Treat outdoor azimuth 0 as unknown. Flag `SITE__872974886` and the rare LAC codes.
4. Build features from **ranks and graphs** (k nearest neighbours, LAC membership, principal-axis position), not from km thresholds.
5. Optional: whiten the coordinates (principal axes with the minor axis stretched about 8–16×) before running density or clustering algorithms.

`scripts/evaluate_ds02.py --out sites.csv` writes the corrected one-row-per-site table (steps 1–2).

## Privacy note

- Site and cell ids are hashed, and the coordinates are transformed away from any real place.
- Neighbour structure and location-area layout are kept, which is exactly what makes the data useful.
- Some re-identification risk remains: that preserved structure could in principle be matched against public tower maps, and the +03:00 timestamps in DS03 plus the DS04 weather series narrow down the region.
- Keep the data under the same sharing restrictions as the source, and do not publish the corrected file openly.

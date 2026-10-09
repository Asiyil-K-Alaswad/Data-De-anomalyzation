# Phase 0 – Inventory & Phase 1 – Schema Mapping

> Status: **observation only — no raw file has been modified.**
> Reproduce: `python3 scripts/phase0_inventory.py` → `reports/phase0/` (per-column profiles + head samples).

## Phase 0 — Inventory

| Dataset | File(s) | Rows | Cols | Size |
|---|---|---:|---:|---:|
| DS01 Rectifier Data | `ANOM_DS01_Rectifier_Data Part 1/2/3.csv` | 867,257 (417,024 + 223,642 + 226,591) | 63 | 196 MB |
| DS02 Site Locations | `ANOM_DS02_SiteLocations.csv` | 49,693 | 10 | 4 MB |
| DS03 Site Inventory | `ANOM_DS03_SiteInventory_in_.csv` | 175,379 | 26 | 39 MB |
| DS04 Weather Data | `ANOM_DS04_WeatherData.csv` | 14,088 | 34 | 3 MB |

All files: comma-delimited, UTF-8, LF line endings, consistent field count on every record (verified with a strict CSV reader — no ragged or truncated rows).

**Method.** Every column was read as raw text (no type inference, no NA conversion) so the profile shows the data exactly as stored. Numeric min/max are what the column *would* parse to; nothing was converted. Full per-column stats: `reports/phase0/<dataset>_profile.csv`.

### Structural observations
1. **DS01 is one table split into three files.** Only Part 1 has a header row; Parts 2 and 3 are headerless continuations with the same 63 columns, and dates continue across the file boundaries (Part 1 ends 7/29/2025, Part 2 starts 7/29/2025; Part 2 ends 01/06/2026, Part 3 starts 01/06/2026).
2. **Null encodings differ by dataset:** DS01/DS02/DS04 use empty strings; DS03 uses the MySQL-style token `\N`, plus `#N/A` (an Excel lookup failure) in 1,141 rows of `CODE`.
3. **DS01 percent columns are mixed text/number.** In rows dated 2025-01-01 to 2025-06-07 (Part 1 only), `…Ratio (%)` and `…Humidity (%)` values carry a literal `%` suffix (`"82.58%"`); later rows are bare numbers (`82.58`). Same unit, different text.
4. **Six DS01 columns are 100% empty** (`Total Traffic (kbit)`, all four `…Outdoor Temperature` columns, `Indoor High/Low Temperature Duration`). Three are constants (`Subnet`=`/`, `Site Grade`=`Undefined`, `Load Work Mode`=`Undefined`).
5. **Date formats (see the ambiguity discussion below).**

## Phase 1 — Data dictionary

Related columns are grouped (e.g. Avg/Max/Min). *Confidence* says how the meaning was established (✔ = verified against the data, ~ = inferred from the name/domain, ? = ambiguous).

### DS01 Rectifier Data — one row per site per day (power-system KPIs from the site's rectifier/energy controller)

| Column(s) | Type | Unit | Meaning | Conf. |
|---|---|---|---|---|
| `SITECODE` | key (str) | — | Anonymised site ID (`SITE__<n>`), 1,570 sites | ✔ |
| `Date` | date | day | Reporting day, **MM/DD/YYYY** with inconsistent zero-padding | ✔ (see A1) |
| `Subnet`, `Site Grade`, `Load Work Mode` | category | — | Constant (`/`, `Undefined`, `Undefined`) — no information | ✔ |
| `Site Type` | category | — | Outdoor / Indoor / Undefined; 61 sites change type over time | ✔ |
| `Avg/Max/Min DC Load Current (A)` | float | A | DC current drawn by telecom load | ✔ (V = P/I ≈ 54 V, standard −48 V plant) |
| `Avg/Max/Min DC Load Power (kW)` | float | kW | DC load power | ✔ |
| `DC Load Energy Consumption (kWh)` | float | kWh/day | DC energy consumed by load; = Avg Power × 24 (100% of rows) | ✔ |
| `Total Energy Input (kWh)` | float | kWh/day | Energy into the power system; = Mains + DG + Solar + Unknown (100%) | ✔ |
| `Mains / D.G. / Solar / Unknown Input Supply (kWh)` | float | kWh/day | Energy by source (D.G. = diesel generator) | ✔ |
| `… Supply Duration (h)` | float | h/day | Hours each source supplied the site | ~ |
| `… kWh Ratio (%)` | float | % | Source share of total input | ~ |
| `Multi-energy Concurrent Supply Duration (h)` | float | h | Hours ≥2 sources supplied simultaneously | ~ |
| `BLVD / LLVD Duration (h)` | float | h | Battery / Load Low-Voltage-Disconnect active time (i.e. load shed) | ~ |
| `Battery Discharge Duration (h)` | float | h | Hours on battery | ~ |
| `Mains Outage Duration (h)`, `Daily Avg Outage Duration (h)` | float | h | Grid outage hours; **the two columns are identical** (100%) | ✔ |
| `Mains Supply Duration (h)` | float | h | = 24 − Mains Outage Duration (100%) | ✔ |
| `Longest Outage Duration (h)` | float | h | Longest single outage that day | ~ |
| `Mains Outage Times(h)` | int | **count** (label says h) | Number of outages; values 0–259 | ? (A4) |
| `Site Fuel Consumption(L)`, `Site Daily Avg Fuel Consumption(L/Day)`, `D.G. Fuel Consumption(L)` | float | L | DG fuel used; **all three identical** (100%) | ✔ |
| `Site Refueling Volume(L)`, `Site Abnormal Fuel Consumption(L)` | float | L | Refuel events / theft-leak flag volume; 99.9% empty | ~ |
| `Avg Fuel Consumption/hour(L/h)`, `/kWh(L/kWh)` | float | L/h, L/kWh | DG efficiency ratios | ~ |
| `D.G. Abnormal Runtime (h)`, `D.G. Start Times` | float/int | h / count | DG anomalies, number of starts | ~ |
| `Daily Avg Supply Duration (h)` | float | h | = D.G. Supply Duration in 98.3% of rows | ? (A5) |
| `Energy Consumption (kWh)`, `Running Duration (h)` | float | kWh, h | Median 5.8 kWh — not mains/DC; possibly a sub-meter (A/C?) | ? (A6) |
| `Avg/Highest Battery Temperature (℃)` | float | °C | Battery temperature | ✔ |
| `Avg/Highest Indoor Temperature (℃)` | float | °C | Shelter/cabinet temperature | ~ |
| `Avg/Highest/Lowest Indoor Humidity (%)` | float | %RH | Indoor humidity | ~ |
| `ENERGYBOARD`, `ENERGYLTE`, `ENERGYNR` | float | kWh/day (assumed) | RAN-equipment energy: whole board, LTE (4G) and NR (5G) share. Never empty — likely joined in from a different (OSS) source | ? (A7) |
| `TRAFFIC_GB` | float | GB/day | Data traffic carried by the site — the **telecom usage** measure | ~ |
| 6 all-empty columns (listed above) | — | — | No data | ✔ |

### DS02 Site Locations — one row per cell (sector antenna)

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `ANONYMIZED_SITECODE` | key (str) | — | Site ID, joins to DS01 `SITECODE` |
| `ANOM_CELLCODE` | key (str) | — | Cell ID; near-unique (2 codes appear at 2 different sites) |
| `AZIMUTHSFORPLOT` | int | degrees | Antenna bearing (0–355); 27% are `0`, possibly a placeholder |
| `LAC_TAC` | int | — | Location / Tracking Area Code |
| `SITETYPE` | category | — | Outdoor/Indoor with case variants (`INDOOR`, `outdoor`), plus `Indoor-SO`, `Pico` |
| `CELLS2G/4G/5G` | int | count | Number of cells per technology **at the site** (constant per site — site-level attribute repeated per cell) |
| `LAT`, `LON` | float | degrees | Coordinates — **anonymised/shifted** (see A3) |

### DS03 Site Inventory — one row per hardware unit (board/module) at a site

| Column | Type | Meaning |
|---|---|---|
| `CODE` | key (str) | Site ID, joins to DS01/DS02; 1,141 rows are `#N/A` |
| `INVENTORYUNITID` | int | Unit index within the site's inventory (0–40) — not globally unique |
| `BOARDNAME`, `BOARDTYPE`, `MODULETYPE`, `MODEL` | category | Hardware identity (e.g. `UBBP` baseband, `MRRU` radio, `FModule` optical SFP) |
| `VENDORUNITFAMILYTYPE` | category | FModule / Board / FAN / PSU / HX / PMU / BATTERY / PDB |
| `MANUFACTURERDATA`, `EXTINFO` | text | Free-text description; `EXTINFO` embeds a serial number (`…SN:xxxx`) |
| `BIOSVER`, `BIOSVEREX`, `SOFTVER`, `LOGICVER`, `ISSUENUMBER` | **text** (version strings — must not be numeric-cast) | Firmware/hardware revision |
| `CLEICODE`, `ITEM` | text | Equipment catalogue codes |
| `CREDITABLE` | bool-ish | YES/NO/`\N` |
| `CREDITABLECHANGEDTIME` | timestamp | ISO-8601 with `+03:00` offset |
| `DATEOFMANUFACTURE`, `FIRSTPOWERONTIME`, `LASTKICKOFFTIME` | date | M/D/YYYY (unambiguous: first part never > 12) |
| `PORTNO` | int | Port/slot; `-1` = not applicable |
| `INVENTORYUNITTYPE`, `MODULENO`, `SOURCEFLAG`, `VENDORUNITTYPENUMBER` | constant | No information |

### DS04 Weather Data — one row per hour, **no site key** (single location)

| Column(s) | Type | Unit | Meaning |
|---|---|---|---|
| `TIME_EPOCH` | int | s (Unix, UTC) | Timestamp |
| `TIME` | datetime | local | = epoch + 3 h for every row (UTC+03:00) |
| `TEMP_C/F`, `FEELSLIKE_*`, `WINDCHILL_*`, `HEATINDEX_*`, `DEWPOINT_*` | float | °C / °F | Paired units — the F column is redundant |
| `WIND_MPH/KPH`, `GUST_MPH/KPH`, `WIND_DEGREE`, `WIND_DIR` | float/cat | mph/kph, °, compass | Wind |
| `PRESSURE_MB/IN`, `PRECIP_MM/IN`, `VIS_KM/MILES` | float | paired units | |
| `HUMIDITY`, `CLOUD`, `CHANCE_OF_RAIN` | int | % | |
| `IS_DAY`, `WILL_IT_RAIN` | bool (0/1) | — | |
| `SNOW_CM`, `WILL_IT_SNOW`, `CHANCE_OF_SNOW` | constant 0 | — | No information |
| `CONDITION` | **embedded dict** (`{'text':…, 'icon':…, 'code':…}`) | — | Weather condition; 16 distinct texts |
| `UV` | float | index | UV index |

Coverage: 2024-10-31 00:00 → 2026-06-10 23:00 hourly, one 24-hour gap.

## Entity-relationship map

```
                      DS02 SiteLocations  (cell grain)
                      2,643 sites · 49,692 cells
                               │ ANONYMIZED_SITECODE
                               │ 1 site : N cells
                               │
DS03 SiteInventory ──CODE──  [ SITE ]  ──SITECODE──  DS01 Rectifier Data
(hardware-unit grain)     (conformed key,            (site × day grain)
2,757 sites · 175k units   ~2,770 distinct)          1,570 sites × 515 days
1 site : N units                                     1 site : N days
                                                          │ Date (day)
                                                          │ N days : 24 hours
                                                     DS04 WeatherData
                                                     (hour grain, NO site key —
                                                      one location for all sites)
```

| Join | Key | Cardinality | Coverage |
|---|---|---|---|
| DS01 → DS02 | `SITECODE` = `ANONYMIZED_SITECODE` | many-days : many-cells (**many-to-many** if joined at cell grain → aggregate DS02 to site first) | 1,536 / 1,570 DS01 sites found |
| DS01 → DS03 | `SITECODE` = `CODE` | many-days : many-units (**many-to-many** → aggregate DS03 to site first) | 1,551 / 1,570 |
| DS02 ↔ DS03 | site code | many-to-many | 2,641 shared |
| DS01 → DS04 | `Date` = date(`TIME`) | 1 day : 24 hours (aggregate DS04 to daily first) | every DS01 date covered |
| All three site tables | — | — | 1,536 sites in all three; **19 DS01 sites in neither DS02 nor DS03** |

Principle: never join two "many" sides directly — row counts multiply and sums double-count. Each satellite table gets aggregated to the grain of the table it's joined to (site, or site × day) before joining.

## Ambiguities flagged for your decision

| # | Issue | Evidence | My reading |
|---|---|---|---|
| **A1** | DS01 `Date` is zero-padded (`01/02/2025`) only when the 2nd part ≤ 12, unpadded (`1/13/2025`) when ≥ 13. That's a typical day/month-swap fingerprint. | File order is strictly `01/01 → 01/12 → 1/13 → 1/31 → 02/01`, row counts per day rise smoothly | **All dates are MM/DD/YYYY**; only the padding differs. No swap. |
| **A2** | 9 calendar days are missing in DS01 (2025-05-05, 05-09, 05-14, 05-16, 05-19, 05-26, 06-02, 06-04, 12-24) | — | Genuine gaps; leave them out, never impute whole days |
| **A3** | DS02 coordinates are at lat ≈ −50, lon ≈ −52 (South Atlantic Ocean), while weather (12–45 °C) and the UTC+3 offset point to a hot UTC+3 region. Coordinates also differ between cells of the same site for 97% of sites. | — | Coordinates are **anonymised/displaced**; don't use them as real geography. Relative positions may still hold. |
| **A4** | `Mains Outage Times(h)` — label says hours, values look like counts (integers 0–259) | — | Treat as **count**; 259/day is an outlier to flag |
| **A5** | `Daily Avg Supply Duration (h)` equals DG supply duration in 98.3% of rows, not total supply | — | Meaning unclear; keep it but don't rely on it |
| **A6** | `Energy Consumption (kWh)` + `Running Duration (h)` — median 5.8 kWh, doesn't match any other energy column | — | Unknown sub-meter; needs domain input |
| **A7** | `ENERGYBOARD` ≠ `ENERGYLTE + ENERGYNR` (matches only 18%); units not stated | ENERGYBOARD median 89 vs DC load 117 kWh | Assume kWh/day; BOARD may include 2G/common equipment |
| **A8** | DS04 has one location and no site key, so every site gets the same weather | — | OK only if the sites are in one region; site-level temperature (battery/indoor) comes from DS01 |
| **A9** | Duplicates are large: 152,345 exact full-row duplicates in DS01; 305,528 rows share `SITECODE+Date` with another row; 37,512 exact duplicates in DS03 | — | → Phase 2 |
| **A10** | Implausible maxima: `Avg DC Load Power` 14,189 kW (median ~5), `DC Load Energy` 340,537 kWh, fuel 96,500 L/day, `Avg Indoor Temperature` 80 °C | — | → Phase 3 outlier review |

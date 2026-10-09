"""PHASE 0 - Inventory. Observation only: reads raw files, writes reports, modifies nothing.

All columns are read as raw strings (no NA conversion) so we see the data exactly as stored.
For each column we then report what it *could* be parsed as, without converting it.
"""
import os, sys, glob
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "reports", "phase0")
NULL_TOKENS = {"\\N", "NULL", "null", "NA", "N/A", "nan", "NaN", "None", "-", "?"}

DS01_PARTS = [f"ANOM_DS01_Rectifier_Data Part {i}.csv" for i in (1, 2, 3)]


def read_raw(path, header="infer", names=None):
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False,
                       header=header, names=names, encoding="utf-8")


def load_ds01():
    p1 = read_raw(os.path.join(ROOT, DS01_PARTS[0]))
    parts = [p1.assign(_source_file=DS01_PARTS[0], _source_line=range(2, len(p1) + 2))]
    for p in DS01_PARTS[1:]:  # parts 2/3 carry no header row
        d = read_raw(os.path.join(ROOT, p), header=None, names=list(p1.columns))
        parts.append(d.assign(_source_file=p, _source_line=range(1, len(d) + 1)))
    return pd.concat(parts, ignore_index=True)


def profile(df):
    rows = []
    for c in [c for c in df.columns if not c.startswith("_source")]:
        s = df[c]
        empty = s.str.strip().eq("")
        tok = s.isin(NULL_TOKENS)
        present = s[~empty & ~tok]
        num = pd.to_numeric(present, errors="coerce")
        n_num = int(num.notna().sum())
        r = {
            "column": c,
            "raw_dtype": "str (raw)",
            "pct_empty": round(100 * empty.mean(), 2),
            "pct_null_token": round(100 * tok.mean(), 2),
            "null_tokens_seen": ",".join(sorted(set(s[tok]))),
            "n_unique": int(s.nunique()),
            "pct_numeric_parsable": round(100 * n_num / max(len(present), 1), 2),
            "min": num.min() if n_num else None,
            "max": num.max() if n_num else None,
            "top_values": "; ".join(f"{k!r}:{v}" for k, v in present.value_counts().head(5).items()),
        }
        if n_num and n_num < len(present):  # mixed column: show non-numeric examples
            r["non_numeric_examples"] = "; ".join(map(repr, present[num.isna()].unique()[:5]))
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    datasets = {"DS01_Rectifier_Data": (load_ds01(), DS01_PARTS)}
    for name, fn in [("DS02_SiteLocations", "ANOM_DS02_SiteLocations.csv"),
                     ("DS03_SiteInventory", "ANOM_DS03_SiteInventory_in_.csv"),
                     ("DS04_WeatherData", "ANOM_DS04_WeatherData.csv")]:
        datasets[name] = (read_raw(os.path.join(ROOT, fn)), [fn])
    inv = []
    for name, (df, files) in datasets.items():
        size = sum(os.path.getsize(os.path.join(ROOT, f)) for f in files)
        ncols = len([c for c in df.columns if not c.startswith("_source")])
        inv.append({"dataset": name, "files": " | ".join(files), "format": "CSV (comma, UTF-8, LF)",
                    "rows": len(df), "columns": ncols, "size_MB": round(size / 1e6, 1)})
        prof = profile(df)
        prof.to_csv(os.path.join(OUT, f"{name}_profile.csv"), index=False)
        df.head(5).to_csv(os.path.join(OUT, f"{name}_head.csv"), index=False)
    pd.DataFrame(inv).to_csv(os.path.join(OUT, "inventory.csv"), index=False)
    print(pd.DataFrame(inv).to_string(index=False))


if __name__ == "__main__":
    main()

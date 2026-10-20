import pandas as pd, numpy as np, os
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df=pd.read_csv(os.path.join(ROOT,"ANOM_DS02_SiteLocations.csv"))
df["ndig"]=df.LAC_TAC.astype(str).str.replace(r"\.0$","",regex=True).str.len()

# rows per (site, digit-length) vs the site's declared cell counts
piv=df.pivot_table(index="ANONYMIZED_SITECODE", columns="ndig", values="ANOM_CELLCODE",
                   aggfunc="size", fill_value=0)
tech=df.groupby("ANONYMIZED_SITECODE").agg(c2=("CELLS2G","max"),c4=("CELLS4G","max"),
                                           c5=("CELLS5G","max")).fillna(0)
t=piv.join(tech).fillna(0)
t.columns=["rows_3digit" if c==3 else ("rows_4digit" if c==4 else
          ("rows_5digit" if c==5 else "rows_1digit")) if isinstance(c,float) else c
          for c in t.columns]

two=t[(t.get("rows_3digit",0)>0)&(t.get("rows_4digit",0)>0)]
print("sites with a 3-digit and a 4-digit code: %d" % len(two))
print("mean rows on 3-digit code: %.2f | mean CELLS2G: %.2f" %
      (two.rows_3digit.mean(), two.c2.mean()))
print("mean rows on 4-digit code: %.2f | mean CELLS4G+5G: %.2f" %
      (two.rows_4digit.mean(), (two.c4+two.c5).mean()))
print("corr(rows_3digit, CELLS2G) = %.3f" % np.corrcoef(two.rows_3digit, two.c2)[0,1])
print("corr(rows_4digit, CELLS4G+5G) = %.3f" % np.corrcoef(two.rows_4digit, two.c4+two.c5)[0,1])
print("\nshare of sites where rows_3digit == CELLS2G exactly: %.3f" %
      (two.rows_3digit==two.c2).mean())
print("share where rows_4digit == CELLS4G+CELLS5G exactly: %.3f" %
      (two.rows_4digit==(two.c4+two.c5)).mean())
print("\nexamples:")
print(two[["rows_3digit","rows_4digit","c2","c4","c5"]].head(8).to_string())

print("\n--- single-code 2G+4G sites (n=%d): which code length? ---" %
      ((t.get("rows_3digit",0)>0)&(t.get("rows_4digit",0)==0)).sum())
one=t[(t.get("rows_3digit",0)>0)&(t.get("rows_4digit",0)==0)]
print("  rows on the single 3-digit code vs CELLS2G: %.2f vs %.2f ; vs all cells: %.2f" %
      (one.rows_3digit.mean(), one.c2.mean(), (one.c2+one.c4+one.c5).mean()))
print("  corr(rows, total cells) = %.3f" % np.corrcoef(one.rows_3digit, one.c2+one.c4+one.c5)[0,1])

print("\n--- pairs observed (3-digit <-> 4-digit) ---")
pairs=df[df.ndig.isin([3,4,5])].groupby(["ANONYMIZED_SITECODE","ndig"]).LAC_TAC.first().unstack()
pp=pairs.dropna().astype(int)
print(pp.groupby([3,4]).size().head(0) if False else "")
combo=(pairs[3].astype("Int64").astype(str)+" -> "+pairs[4].astype("Int64").astype(str))
print(combo.value_counts().head(12).to_string())

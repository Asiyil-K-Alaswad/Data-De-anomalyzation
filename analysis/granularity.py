import pandas as pd, numpy as np, os
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df=pd.read_csv(os.path.join(ROOT,"ANOM_DS02_SiteLocations.csv"))

print("rows %d | unique sites %d | unique cells %d" %
      (len(df), df.ANONYMIZED_SITECODE.nunique(), df.ANOM_CELLCODE.nunique()))
dup=df.ANOM_CELLCODE.value_counts()
print("cell codes appearing >1 time:", int((dup>1).sum()), "->", dup[dup>1].to_dict())

per=df.groupby("ANONYMIZED_SITECODE").size()
print("\nrows per site: min %d  median %d  mean %.1f  max %d" %
      (per.min(), int(per.median()), per.mean(), per.max()))
print("rows-per-site histogram (top):", per.value_counts().head(8).to_dict())

# is each site one contiguous block?
run=(df.ANONYMIZED_SITECODE!=df.ANONYMIZED_SITECODE.shift()).cumsum()
blocks=df.assign(run=run).groupby("ANONYMIZED_SITECODE").run.nunique()
print("\ncontiguous blocks per site: median %d mean %.1f max %d ; sites with >1 block: %d" %
      (int(blocks.median()), blocks.mean(), blocks.max(), int((blocks>1).sum())))
print("total blocks:", int(blocks.sum()))

# are site-level attributes constant within a site?
g=df.groupby("ANONYMIZED_SITECODE")
for col in ["LAC_TAC","SITETYPE","CELLS2G","CELLS4G","CELLS5G"]:
    n=g[col].nunique(dropna=False)
    print("%-10s constant in %d/%d sites (median distinct %d)" %
          (col, int((n<=1).sum()), len(n), int(n.median())))
print("\nrows vs declared cells (site totals): mean rows %.2f, mean 2G+4G+5G %.2f" %
      (per.mean(), (g.CELLS2G.max().fillna(0)+g.CELLS4G.max().fillna(0)+g.CELLS5G.max().fillna(0)).mean()))

# azimuth behaviour: cell-level?
print("\nazimuth distinct per site (median %d); cells per (site,azimuth):" % int(g.AZIMUTHSFORPLOT.nunique().median()))
print(df.groupby(["ANONYMIZED_SITECODE","AZIMUTHSFORPLOT"]).size().value_counts().head(5).to_dict())
print("\nexample site rows:")
ex=per.sort_values(ascending=False).index[3]
print(df[df.ANONYMIZED_SITECODE==ex][["ANONYMIZED_SITECODE","ANOM_CELLCODE","AZIMUTHSFORPLOT",
      "LAC_TAC","SITETYPE","CELLS2G","CELLS4G","CELLS5G"]].head(24).to_string())

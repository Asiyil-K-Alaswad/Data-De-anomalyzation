import pandas as pd, numpy as np, os
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df=pd.read_csv(os.path.join(ROOT,"ANOM_DS02_SiteLocations.csv"))
n=len(df)
site=df.ANONYMIZED_SITECODE.values; lac=df.LAC_TAC.values
print("rows with missing CELLCODE:", int(df.ANOM_CELLCODE.isna().sum()),
      "| missing LAC_TAC:", int(df.LAC_TAC.isna().sum()),
      "| missing azimuth:", int(df.AZIMUTHSFORPLOT.isna().sum()))
code=pd.to_numeric(df.ANOM_CELLCODE.astype(str).str.replace("CELL__",""), errors="coerce").fillna(-1).astype("int64").values
az=df.AZIMUTHSFORPLOT.values

def runs(a):
    return int((a[1:]!=a[:-1]).sum()+1)
print("runs in the SITE column      : %d" % runs(site))
print("runs in the LAC_TAC column   : %d" % runs(lac))
print("runs in the AZIMUTH column   : %d" % runs(az))
r=pd.Series(code).rank().values
print("rank corr(row index, cell code): %.3f" % np.corrcoef(np.arange(n), r)[0,1])
print("frac consecutive rows same site: %.3f  same LAC: %.3f" %
      ((site[1:]==site[:-1]).mean(), (lac[1:]==lac[:-1]).mean()))

# within each LAC, are sites contiguous?
df2=df.reset_index(drop=True)
df2["li"]=pd.factorize(df2.LAC_TAC)[0]
g=df2.groupby("li").apply(lambda d: runs(d.ANONYMIZED_SITECODE.values), include_groups=False)
print("\nper-LAC-block: number of site runs (sum %d) -> sites are NOT grouped inside LAC either" % g.sum())
print(g.head(8).to_dict())

# how many blocks per (site, LAC) pair
pairs=df2.groupby(["ANONYMIZED_SITECODE","LAC_TAC"]).apply(
    lambda d: runs(d.index.values) , include_groups=False)
print("\ncontiguous blocks per (site, LAC) pair: median %d mean %.1f max %d" %
      (int(pairs.median()), pairs.mean(), pairs.max()))

print("\nfirst 30 rows (site index, LAC, azimuth):")
lab={s:i for i,s in enumerate(pd.unique(site))}
for i in range(30):
    print("  %3d  site%-5d LAC%-7s az%-5s" % (i, lab[site[i]], lac[i], az[i]))

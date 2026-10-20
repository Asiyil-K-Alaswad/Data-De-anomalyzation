import pandas as pd, numpy as np, os
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df=pd.read_csv(os.path.join(ROOT,"ANOM_DS02_SiteLocations.csv"))

# per code: distinct sites that have ANY cell on that code, and cells
per=df.groupby("LAC_TAC").agg(cells=("ANOM_CELLCODE","size"),
                              sites=("ANONYMIZED_SITECODE","nunique"))
per["sites_per_code"]=per.sites
print("24 codes; sites per code: min %d median %d max %d" %
      (per.sites.min(), int(per.sites.median()), per.sites.max()))
print("\ncode   cells  sites  cells/site")
for code, r in per.sort_values("sites").iterrows():
    print("%7g %6d %6d %10.1f" % (code, r.cells, r.sites, r.cells/r.sites))

# how many codes per site, and does it track technology?
sc=df.groupby("ANONYMIZED_SITECODE").LAC_TAC.nunique()
print("\ncodes per site:", sc.value_counts().to_dict())

tech=df.groupby("ANONYMIZED_SITECODE").agg(g2=("CELLS2G","max"),g4=("CELLS4G","max"),g5=("CELLS5G","max"))
tech=tech.fillna(0)
h2=tech.g2>0; h4=tech.g4>0; h5=tech.g5>0
lab=np.where(h2&h4,"2G+4G",np.where(h4&h5,"4G+5G",np.where(h4,"4G only",np.where(h2,"2G only","other"))))
tab=pd.crosstab(pd.Series(lab,index=tech.index,name="tech"), sc, dropna=False)
print("\ncodes-per-site by technology mix (sites):")
print(tab.to_string())

# 3-digit vs 4/5-digit splits
df["ndig"]=df.LAC_TAC.astype(str).str.replace(r"\.0$","",regex=True).str.len()
print("\nLAC_TAC by digit length (cells):", df.ndig.value_counts().to_dict())
site_dig=df.groupby("ANONYMIZED_SITECODE").LAC_TAC.apply(lambda s: tuple(sorted(set(s.astype(int)))))
pat=site_dig.value_counts().head(12)
print("\nmost common (code) sets per site:")
for k,v in pat.items(): print("   %-22s %d sites" % (k, v))

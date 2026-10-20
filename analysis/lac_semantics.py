import pandas as pd, numpy as np, os
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df=pd.read_csv(os.path.join(ROOT,"ANOM_DS02_SiteLocations.csv"))

print("rows", len(df), "sites", df.ANONYMIZED_SITECODE.nunique(),
      "cells", df.ANOM_CELLCODE.nunique())
print("\nLAC_TAC nulls:", df.LAC_TAC.isna().sum())

# is LAC_TAC constant within a site?
g=df.groupby("ANONYMIZED_SITECODE").LAC_TAC.nunique()
print("\ndistinct LAC_TAC per site:")
print(g.value_counts().to_string())
print("sites with >1 LAC_TAC:", (g>1).sum())

# is it constant within a cell? (a cell appears once, but check duplicates)
c=df.groupby("ANOM_CELLCODE").LAC_TAC.nunique()
print("\nrows per cell: mean %.2f max %d" % (df.groupby("ANOM_CELLCODE").size().mean(),
                                             df.groupby("ANOM_CELLCODE").size().max()))
print("cells with >1 LAC_TAC:", (c>1).sum())

# distribution: LAC -> sites / cells
sites=(df.drop_duplicates("ANONYMIZED_SITECODE")
         .groupby("LAC_TAC").ANONYMIZED_SITECODE.size().rename("sites"))
cells=df.groupby("LAC_TAC").ANOM_CELLCODE.size().rename("cells")
t=pd.concat([sites,cells],axis=1).sort_values("sites",ascending=False)
t["cells_per_site"]=(t.cells/t.sites).round(1)
print("\nLAC_TAC -> #sites / #cells (all %d codes):" % len(t))
print(t.to_string())
print("\nunique codes:", df.LAC_TAC.nunique(), " -> sites/code: min %d max %d median %d" %
      (t.sites.min(), t.sites.max(), int(t.sites.median())))

# do sites mix technologies under one code? site-level tech signature
tech=df.groupby("ANONYMIZED_SITECODE").agg(g2=("CELLS2G","first"),g4=("CELLS4G","first"),g5=("CELLS5G","first"))
has2=tech.g2.fillna(0)>0; has4=tech.g4.fillna(0)>0; has5=tech.g5.fillna(0)>0
mm=df.groupby("ANONYMIZED_SITECODE").LAC_TAC.first().to_frame().join(tech)
mm["tech"]=np.select([has4&has5,has4,~has2],"4G+5G" ,default="?")
print("\nsite technology mix across the whole file (sites):",
      int((has4&has5).sum()), "4G+5G", int(has4&~has5).sum(), "4G only")

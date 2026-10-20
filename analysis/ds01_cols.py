import pandas as pd, os, re
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p=os.path.join(ROOT,"ANOM_DS01_Rectifier_Data Part 1.csv")
hdr=pd.read_csv(p, nrows=3)
print("DS01 columns (%d):" % hdr.shape[1])
for c in hdr.columns: print("   ", c)
print("\nany column mentioning cell/CID:", [c for c in hdr.columns if re.search(r"cell|cid|ci$", c, re.I)])

d=pd.read_csv(p, usecols=["SITECODE","Subnet","Date"], low_memory=False)
print("\nrows(part 1) %d | unique SITECODE %d" % (len(d), d.SITECODE.nunique()))
print("SITECODE examples:", d.SITECODE.unique()[:5].tolist())
print("Subnet values:", d.Subnet.value_counts(dropna=False).head(10).to_dict())

d2=pd.read_csv(os.path.join(ROOT,"ANOM_DS02_SiteLocations.csv"), usecols=["ANONYMIZED_SITECODE"])
s1=set(d.SITECODE.unique()); s2=set(d2.ANONYMIZED_SITECODE.unique())
print("\nDS01 sites: %d | DS02 sites: %d | overlap: %d (%.1f%% of DS01)" %
      (len(s1), len(s2), len(s1&s2), 100*len(s1&s2)/len(s1)))
print("DS02 sites not in DS01:", len(s2-s1))

# DS03
d3=pd.read_csv(os.path.join(ROOT,"ANOM_DS03_SiteInventory_in_.csv"), usecols=["CODE","BOARDNAME"], low_memory=False, nrows=200000)
print("\nDS03 CODE examples:", d3.CODE.dropna().unique()[:4].tolist(),
      "| unique CODE in first 200k rows:", d3.CODE.nunique())

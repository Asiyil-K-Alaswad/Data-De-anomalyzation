import pandas as pd, os
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parts=["ANOM_DS01_Rectifier_Data Part 1.csv","ANOM_DS01_Rectifier_Data Part 2.csv","ANOM_DS01_Rectifier_Data Part 3.csv"]
u=set(); rows=0
for i,f in enumerate(parts):
    kw=dict(usecols=["SITECODE","Date"]) if i==0 else dict(usecols=[0,2],header=None)
    d=pd.read_csv(os.path.join(ROOT,f), low_memory=False, **kw)
    d.columns=["SITECODE","Date"]
    rows+=len(d); u|=set(d.SITECODE.dropna().unique())
    print("%-42s header=%-5s rows %7d sites %5d" % (f, i==0, len(d), d.SITECODE.nunique()))
print("\nDS01 total rows %d | unique sites %d" % (rows, len(u)))
d2=pd.read_csv(os.path.join(ROOT,"ANOM_DS02_SiteLocations.csv"), usecols=["ANONYMIZED_SITECODE"])
s2=set(d2.ANONYMIZED_SITECODE.dropna().unique())
print("DS02 sites %d | overlap %d (%.1f%% of DS02) | DS02-only %d | DS01-only %d" %
      (len(s2), len(u&s2), 100*len(u&s2)/len(s2), len(s2-u), len(u-s2)))

import pandas as pd, os
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parts=["ANOM_DS01_Rectifier_Data Part 1.csv","ANOM_DS01_Rectifier_Data Part 2.csv","ANOM_DS01_Rectifier_Data Part 3.csv"]
u=set(); rows=0
for f in parts:
    d=pd.read_csv(os.path.join(ROOT,f), usecols=["SITECODE","Date"], low_memory=False)
    rows+=len(d); u|=set(d.SITECODE.dropna().unique())
    print("%-42s rows %7d sites %5d dates %s..%s" % (f, len(d), d.SITECODE.nunique(), d.Date.iloc[0], d.Date.iloc[-1]))
print("\nDS01 total rows %d | unique sites %d" % (rows, len(u)))
d2=pd.read_csv(os.path.join(ROOT,"ANOM_DS02_SiteLocations.csv"), usecols=["ANONYMIZED_SITECODE","ANOM_CELLCODE"])
s2=set(d2.ANONYMIZED_SITECODE.dropna().unique())
print("DS02 sites %d | overlap with DS01 %d (%.1f%% of DS02) | DS02-only %d | DS01-only %d" %
      (len(s2), len(u&s2), 100*len(u&s2)/len(s2), len(s2-u), len(u-s2)))
print("cell codes in DS01?  columns:", "no CELL__ column" if "ANOM_CELLCODE" not in d2.columns else "")

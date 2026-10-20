import pandas as pd, numpy as np

f = r"C:\Users\aselk\OneDrive\Desktop\De-anomalized data\Data-De-anomalyzation\ANOM_DS02_SiteLocations.csv"
df = pd.read_csv(f)
print("rows", len(df))
print("LAT min/max", df.LAT.min(), df.LAT.max(), "span", df.LAT.max()-df.LAT.min())
print("LON min/max", df.LON.min(), df.LON.max(), "span", df.LON.max()-df.LON.min())
print("n unique sites", df.ANONYMIZED_SITECODE.nunique())
print("n unique coords", df[['LAT','LON']].drop_duplicates().shape[0])

# sites
sites = df.groupby('ANONYMIZED_SITECODE').agg(LAT=('LAT','first'), LON=('LON','first'),
                                              LAC=('LAC_TAC','first'), T=('SITETYPE','first'),
                                              n=('ANOM_CELLCODE','count'))
print("\nunique site coords", sites[['LAT','LON']].drop_duplicates().shape[0])

# decimal digit structure
def dec(x):
    s = repr(float(x))
    return len(s.split('.')[1]) if '.' in s else 0
print("\nlat decimals hist", pd.Series([dec(v) for v in sites.LAT]).value_counts().to_dict())

# look at the isolated southern outliers
print("\nsouthernmost 15 sites\n", sites.sort_values('LAT').head(15).to_string())

# nearest neighbour distance distribution (deg) among unique coords
c = sites[['LAT','LON']].drop_duplicates().values
from numpy.linalg import norm
d = []
for i in range(len(c)):
    dd = np.hypot((c[:,0]-c[i,0]), (c[:,1]-c[i,1]))
    dd[i] = 1e9
    d.append(dd.min())
d = np.array(d)
print("\nNN dist deg: min %.7f p25 %.6f med %.6f p75 %.6f max %.6f" % (d.min(), np.percentile(d,25), np.median(d), np.percentile(d,75), d.max()))

# LAC distribution per approximate region
print("\nLAC_TAC counts", df.LAC_TAC.value_counts().to_dict())
print("\nSITETYPE", df.SITETYPE.value_counts().to_dict())

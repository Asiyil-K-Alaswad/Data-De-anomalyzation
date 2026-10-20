import pandas as pd, numpy as np
from collections import Counter

f = r"C:\Users\aselk\OneDrive\Desktop\De-anomalized data\Data-De-anomalyzation\ANOM_DS02_SiteLocations.csv"
df = pd.read_csv(f)
d = df.copy()
d['LAT2'] = np.concatenate([df.LAT.values[1:], [np.nan]])
d['LON2'] = np.concatenate([df.LON.values[1:], [np.nan]])
d = d.dropna(subset=['LAT2'])
g = d.groupby('ANONYMIZED_SITECODE').agg(LAT=('LAT2','median'), LON=('LON2','median'))
c = g[['LON','LAT']].drop_duplicates().values  # (x=lon, y=lat)
print("unique site coords", len(c))

D = c[:,None,:] - c[None,:,:]
dist = np.hypot(D[...,0], D[...,1])
iu = np.triu_indices(len(c), 1)
dv = D[iu]
dd = dist[iu]
print("min pair distance", dd.min())

for thr in (1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5):
    m = dd < thr
    print("pairs with |d|<%g : %d" % (thr, m.sum()))

m = dd < 2e-5
v = dv[m]
ang = np.degrees(np.arctan2(v[:,1], v[:,0])) % 180
print("\nangle histogram (deg, 0-180) for %d near pairs:" % m.sum())
h, e = np.histogram(ang, bins=36, range=(0,180))
for i in range(36):
    if h[i]:
        print("%6.1f-%6.1f %4d %s" % (e[i], e[i+1], h[i], '#'*h[i]))

# cluster angles
print("\ntop angle bins:")
idx = np.argsort(-h)[:8]
for i in idx:
    print("  %6.1f deg  n=%d" % ((e[i]+e[i+1])/2, h[i]))

# for the dominant directions, look at the vectors
print("\nsample small-difference vectors (scaled by 1e6):")
for k in np.argsort(dd)[:40]:
    print("  d=(%+0.3f, %+0.3f)e-6  |d|=%.3fe-6  angle=%.1f" % (dv[k,0]*1e6, dv[k,1]*1e6, dd[k]*1e6, np.degrees(np.arctan2(dv[k,1], dv[k,0]))%180))

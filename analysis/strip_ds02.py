import pandas as pd, numpy as np

f = r"C:\Users\aselk\OneDrive\Desktop\De-anomalized data\Data-De-anomalyzation\ANOM_DS02_SiteLocations.csv"
df = pd.read_csv(f)
c = df[['LAT','LON']].drop_duplicates().values
mlat = np.radians(c[:,0].mean())
x = c[:,1]*np.cos(mlat); y = c[:,0]
P = np.vstack([x - x.mean(), y - y.mean()])
w, v = np.linalg.eigh(P @ P.T / P.shape[1])
u = v[:,1]  # principal
perp = np.array([-u[1], u[0]])
t = u @ P; s = perp @ P
print("along-stream range %.4f..%.4f  perp std %.5f" % (t.min(), t.max(), s.std()))
# outliers
idx = np.argsort(-np.abs(s))[:40]
print("\ntop perpendicular outliers (lat, lon, perp, along):")
for i in sorted(idx, key=lambda i: -abs(s[i])):
    print("  %.6f %.6f  perp %+0.4f along %+0.4f" % (c[i,0], c[i,1], s[i], t[i]))

# how many points beyond 5 sigma
for k in (3,4,5,8,12):
    print("points with |perp|>%d*std: %d" % (k, (np.abs(s) > k*s.std()).sum()))

# quantiles of perp
print("\nperp quantiles", np.percentile(s, [0,1,5,25,50,75,95,99,100]).round(4))

# along-stream histogram: where is density concentrated?
import collections
h, edges = np.histogram(t, bins=40)
for i in range(40):
    print("%8.3f..%8.3f %6d %s" % (edges[i], edges[i+1], h[i], '#'*int(h[i]/25)))

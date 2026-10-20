import pandas as pd, numpy as np

f = r"C:\Users\aselk\OneDrive\Desktop\De-anomalized data\Data-De-anomalyzation\ANOM_DS02_SiteLocations.csv"
df = pd.read_csv(f)
n = len(df)
# true coord of row i = displayed[i+1]
d = df.copy()
d['LAT2'] = np.concatenate([df.LAT.values[1:], [np.nan]])
d['LON2'] = np.concatenate([df.LON.values[1:], [np.nan]])
d = d.dropna(subset=['LAT2'])

# site level: mode coordinate & LAC
g = d.groupby('ANONYMIZED_SITECODE').agg(LAT=('LAT2','median'), LON=('LON2','median'),
                                        LAC=('LAC_TAC','first'), T=('SITETYPE','first'),
                                        rows=('LAT2','size'),
                                        ncoord=('LAT2','nunique'))
print("sites", len(g), "with 1 coord", (g.ncoord==1).sum(), "rows/site mean %.1f" % g.rows.mean())
print("sites with >1 coord:", (g.ncoord>1).sum())
print(g.ncoord.value_counts().head(10).to_dict())

c = g[['LAT','LON']].values
lac = g.LAC.values
mlat = np.radians(c[:,0].mean())
X = np.c_[c[:,1]*np.cos(mlat), c[:,0]]

# nearest neighbour LAC agreement
from numpy.linalg import norm
D = np.hypot(X[:,0][:,None]-X[:,0][None,:], X[:,1][:,None]-X[:,1][None,:])
np.fill_diagonal(D, 1e9)
nn = D.argmin(axis=1)
agree = (lac == lac[nn]).mean()
p = pd.Series(lac).value_counts(normalize=True).values
print("\nnearest-neighbour LAC agreement: %.3f  (chance level %.3f)" % (agree, (p**2).sum()))

# k nearest (k=5)
k = 10
idx = np.argsort(D, axis=1)[:, :k]
agree_k = (lac[:,None] == lac[idx]).mean()
print("within %d nearest: LAC agreement %.3f" % (k, agree_k))

# distance-weighted: how far to nearest same-LAC vs any
# spatial extent of each LAC (in anon units)
sz = pd.DataFrame({'lac': lac, 'x': X[:,0], 'y': X[:,1]}).groupby('lac').agg(
    n=('x','size'), xs=('x', lambda s: s.max()-s.min()), ys=('y', lambda s: s.max()-s.min()))
print("\ntop LACs by size:\n", sz.sort_values('n', ascending=False).head(12).to_string())

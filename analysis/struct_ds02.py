import pandas as pd, numpy as np

f = r"C:\Users\aselk\OneDrive\Desktop\De-anomalized data\Data-De-anomalyzation\ANOM_DS02_SiteLocations.csv"
df = pd.read_csv(f)
lat, lon = df.LAT.values, df.LON.values
r1 = pd.Series(lat).rank().values; r2 = pd.Series(lon).rank().values
print("Spearman LAT~LON:", np.corrcoef(r1, r2)[0,1])
print("Pearson  LAT~LON:", np.corrcoef(lat, lon)[0,1])

# metric space
mlat = np.radians((lat.mean()))
x = lon * np.cos(mlat); y = lat
A = np.vstack([x - x.mean(), y - y.mean()])
cov = A @ A.T / A.shape[1]
w, v = np.linalg.eigh(cov)
print("metric PCA eigenvalues", w, "aspect(std ratio)", np.sqrt(w[1]/w[0]))
print("principal dir (dx,dy)", v[:,1])

# row-order monotonicity
d = np.diff(lat)
print("\nrow-order: frac increasing", (d>0).mean(), "frac equal", (d==0).mean(), "frac decreasing", (d<0).mean())

# runs of identical coordinate pairs
key = df.LAT.astype(str) + "|" + df.LON.astype(str)
runs = (key != key.shift()).cumsum()
rl = runs.value_counts()
print("\nrun count", len(rl), "run len describe\n", rl.describe())
print("top run lengths", rl.head(8).to_dict())

# per-site coordinate spread (in the anon space)
g = df.groupby('ANONYMIZED_SITECODE').agg(n=('LAT','size'), latmin=('LAT','min'), latmax=('LAT','max'),
                                         lonmin=('LON','min'), lonmax=('LON','max'), nuniq=('LAT','nunique'))
g['latspread'] = g.latmax-g.latmin; g['lonspread'] = g.lonmax-g.lonmin
print("\nsites with >1 cell:", (g.n>1).sum())
print("per-site lat spread (deg) describe\n", g.latspread.describe())
print("\nsites with spread>0.01:", (g.latspread>0.01).sum(), " >0.1:", (g.latspread>0.1).sum())
print(g.sort_values('latspread', ascending=False).head(10).to_string())

# check whether site order in file is contiguous blocks
print("\nsite block contiguity: number of site runs", (df.ANONYMIZED_SITECODE != df.ANONYMIZED_SITECODE.shift()).sum())
print("n sites", df.ANONYMIZED_SITECODE.nunique())

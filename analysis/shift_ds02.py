import pandas as pd, numpy as np

f = r"C:\Users\aselk\OneDrive\Desktop\De-anomalized data\Data-De-anomalyzation\ANOM_DS02_SiteLocations.csv"
df = pd.read_csv(f)
n = len(df)
lat = df.LAT.values; lon = df.LON.values
site = df.ANONYMIZED_SITECODE.values

def uniq_per_site(la, lo, s):
    key = la.astype(str) + "|" + lo.astype(str)
    t = pd.DataFrame({'s': s, 'k': key})
    g = t.groupby('s').k.nunique()
    return g

print("shift  mean_uniq_coords_per_site  frac_sites_with_1  n_sites")
for k in range(-6, 7):
    if k >= 0:
        la, lo, s = lat, lon, site
        # real[i] = displayed[i+k] -> shift array up by k
        la2 = np.concatenate([lat[k:], [np.nan]*k])
        lo2 = np.concatenate([lon[k:], [np.nan]*k])
        s2 = site
        m = ~np.isnan(la2)
    else:
        kk = -k
        la2 = np.concatenate([[np.nan]*kk, lat[:n-kk]])
        lo2 = np.concatenate([[np.nan]*kk, lon[:n-kk]])
        s2 = site
        m = ~np.isnan(la2)
    g = uniq_per_site(la2[m], lo2[m], s2[m])
    print("%+3d    %.3f                     %.3f              %d" % (k, g.mean(), (g == 1).mean(), len(g)))

# Azimuth consistency within site blocks (are rows properly grouped?)
az = df.AZIMUTHSFORPLOT.values
blocks = (site != np.roll(site, 1)); blocks[0] = True
bid = np.cumsum(blocks)
d = pd.DataFrame({'b': bid, 's': site, 'az': az, 'lat': lat, 'lon': lon})
gg = d.groupby('b').agg(n=('az','size'), s=('s','nunique'), azn=('az','nunique'), coord=('lat','nunique'))
print("\nblocks:", len(gg), "blocks with >1 site:", (gg.s > 1).sum(), "blocks where azn==n:", (gg.azn == gg.n).sum(), "of", len(gg))
print("block size distribution:\n", gg.n.value_counts().head(12))
print("\nmean coords per block:", gg.coord.mean(), "frac blocks with 1 coord:", (gg.coord == 1).mean())

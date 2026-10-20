import pandas as pd, numpy as np
from PIL import Image, ImageDraw

f = r"C:\Users\aselk\OneDrive\Desktop\De-anomalized data\Data-De-anomalyzation\ANOM_DS02_SiteLocations.csv"
df = pd.read_csv(f)
d = df.copy()
d['LAT2'] = np.concatenate([df.LAT.values[1:], [np.nan]])
d['LON2'] = np.concatenate([df.LON.values[1:], [np.nan]])
d = d.dropna(subset=['LAT2'])
g = d.groupby('ANONYMIZED_SITECODE').agg(LAT=('LAT2','median'), LON=('LON2','median'),
                                        LAC=('LAC_TAC','first'), rows=('LAT2','size'))
c = g[['LAT','LON']].values
mlat = np.radians(c[:,0].mean())
X = np.c_[c[:,1]*np.cos(mlat), c[:,0]]
mu = X.mean(axis=0)
P = (X-mu).T
w, v = np.linalg.eigh(P @ P.T / P.shape[1])
u = v[:,1]; pp = np.array([-u[1], u[0]])
t = u @ P; s = pp @ P
print("along %.3f..%.3f perp %.4f..%.4f" % (t.min(), t.max(), s.min(), s.max()))

W, H = 1600, 700
def px(a, b):
    return 40 + (a - t.min())/(t.max()-t.min())*(W-80), H/2 - b*20*((H-80)/(s.max()-s.min())*0)
img = Image.new('RGB', (W, H), 'white')
dr = ImageDraw.Draw(img)
span = max(abs(s.min()), abs(s.max()))
for a, b in zip(t, s):
    x = 40 + (a - t.min())/(t.max()-t.min())*(W-80)
    y = H/2 - b/span*(H/2-40)
    dr.ellipse([x-2, y-2, x+2, y+2], fill=(150,150,150))
dr.text((10, 5), "perp stretched: full y-range = +-%.4f deg" % span, fill=(0,0,0))
img.save(r"C:\Users\aselk\OneDrive\Desktop\De-anomalized data\Data-De-anomalyzation\analysis\strip_perp.png")

# colour by top LACs
cols = [(220,0,0),(0,130,0),(0,0,230),(230,140,0),(150,0,200),(0,170,170),(120,80,0),(255,0,150)]
lacs = g.LAC.value_counts().head(8).index.tolist()
img2 = Image.new('RGB', (W, H), 'white')
dr2 = ImageDraw.Draw(img2)
for a, b, l in zip(t, s, g.LAC.values):
    x = 40 + (a - t.min())/(t.max()-t.min())*(W-80)
    y = H/2 - b/span*(H/2-40)
    col = cols[lacs.index(l)] if l in lacs else (210,210,210)
    dr2.ellipse([x-3, y-3, x+3, y+3], fill=col)
for i, l in enumerate(lacs):
    dr2.text((10, 10+14*i), "LAC %g" % l, fill=cols[i])
img2.save(r"C:\Users\aselk\OneDrive\Desktop\De-anomalized data\Data-De-anomalyzation\analysis\strip_lac.png")
print("saved")
print(g.LAC.value_counts().head(8))

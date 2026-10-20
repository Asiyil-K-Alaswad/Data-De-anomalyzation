import pandas as pd, numpy as np
from PIL import Image, ImageDraw

f = r"C:\Users\aselk\OneDrive\Desktop\De-anomalized data\Data-De-anomalyzation\ANOM_DS02_SiteLocations.csv"
out = r"C:\Users\aselk\OneDrive\Desktop\De-anomalized data\Data-De-anomalyzation\analysis\anon_scatter.png"
df = pd.read_csv(f)
c = df[['LON','LAT']].drop_duplicates().values
print("unique coords", len(c))

W = H = 1200
lon0, lon1 = c[:,0].min(), c[:,0].max()
lat0, lat1 = c[:,1].min(), c[:,1].max()
print("lon", lon0, lon1, "lat", lat0, lat1)
# preserve aspect in km: x scale = cos(mean lat)
mx = np.cos(np.radians((lat0+lat1)/2))
spanx = (lon1-lon0)*mx
spany = (lat1-lat0)
pad = 30
sc = min((W-2*pad)/spanx, (H-2*pad)/spany)
def px(lon, lat):
    x = pad + (lon-lon0)*mx*sc
    y = H - pad - (lat-lat0)*sc
    return x, y
img = Image.new('RGB', (W, H), 'white')
dr = ImageDraw.Draw(img)
# grid every 0.1 deg
for g in np.arange(np.floor(lon0*10)/10, lon1+0.1, 0.1):
    x,_ = px(g, lat0); dr.line([(x,0),(x,H)], fill=(235,235,235))
for g in np.arange(np.floor(lat0*10)/10, lat1+0.1, 0.1):
    _,y = px(lon0, g); dr.line([(0,y),(W,y)], fill=(235,235,235))
for lon, lat in c:
    x, y = px(lon, lat)
    dr.ellipse([x-2, y-2, x+2, y+2], fill=(200,0,0))
# axes labels for a few grid lines
for g in np.arange(np.floor(lon0*10)/10, lon1+0.1, 0.1):
    x,_ = px(g, lat0); dr.text((x+2, H-18), "%.1f" % g, fill=(0,0,0))
for g in np.arange(np.floor(lat0*10)/10, lat1+0.1, 0.1):
    _,y = px(lon0, g); dr.text((2, y-12), "%.1f" % g, fill=(0,0,0))
img.save(out)
print("wrote", out)

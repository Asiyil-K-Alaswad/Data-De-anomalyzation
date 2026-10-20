import json, os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
T = json.load(open(os.path.join(HERE, "transform_lacfit.json")))
B = np.array(T["B"]); c = np.array(T["c"])
REF = json.load(open(os.path.join(HERE, "ref_bbox.json"))) if os.path.exists(os.path.join(HERE, "ref_bbox.json")) else None

sites = pd.read_csv(os.path.join(HERE, "sites_lacfit.csv"))
print("LAC   n   mapped bbox (lon,lat)                 ref bbox (lon,lat)              dW dE dS dN (km)")
# reference bboxes copied from the ClickHouse per-area moments query (net=2)
BB = {
 110: (51.0466, 51.5829, 25.2459, 26.1351), 111: (51.3357, 51.5183, 25.1190, 25.2871),
 120: (51.3958, 51.6419, 25.1223, 25.5686), 121: (51.4291, 51.5966, 25.1923, 25.3928),
 130: (50.7575, 51.6002, 25.1595, 25.5054), 131: (51.3316, 51.5536, 25.1676, 25.2747),
 141: (51.3810, 51.6231, 25.1292, 25.3201), 142: (51.3796, 51.6186, 25.1827, 25.8323),
 143: (51.4332, 51.5870, 25.1637, 25.3942), 150: (51.4009, 51.5485, 25.2192, 25.3613),
 151: (50.8574, 51.5582, 24.5977, 25.6211), 160: (51.1722, 51.5650, 25.2885, 25.7279),
 161: (51.0459, 51.5636, 25.2489, 26.1289), 171: (50.9884, 51.6199, 24.5606, 25.2789),
 181: (51.3384, 51.5595, 25.1717, 25.3269), 1110: (51.1887, 51.6545, 25.0179, 25.9367),
 1130: (51.2217, 51.4301, 25.2459, 25.5412), 1151: (51.2519, 51.5650, 25.0715, 25.5453),
 1160: (51.3295, 51.5396, 25.3283, 25.6126), 1171: (51.3824, 51.6103, 24.9696, 25.8186),
}
km = 111.0
for lac, g in sites.groupby("LAC"):
    lac = int(lac)
    if lac not in BB: continue
    p = g[["LON_REAL","LAT_REAL"]].values
    w, e, s, n = p[:,0].min(), p[:,0].max(), p[:,1].min(), p[:,1].max()
    rw, re, rs, rn = BB[lac]
    print("%5d %4d  (%.4f-%.4f, %.4f-%.4f)  (%.4f-%.4f, %.4f-%.4f)  %+5.1f %+5.1f %+5.1f %+5.1f" %
          (lac, len(p), w, e, s, n, rw, re, rs, rn,
           (w-rw)*km*np.cos(np.radians(25.4)), (e-re)*km*np.cos(np.radians(25.4)), (s-rs)*km, (n-rn)*km))

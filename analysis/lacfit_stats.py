import numpy as np, pandas as pd, os
HERE = os.path.dirname(os.path.abspath(__file__))
s = pd.read_csv(os.path.join(HERE, "sites_lacfit.csv"))
m = s[["LON_REAL","LAT_REAL"]].values
qx = np.quantile(m[:,0], [0.01,0.05,0.25,0.5,0.75,0.95,0.99])
qy = np.quantile(m[:,1], [0.01,0.05,0.25,0.5,0.75,0.95,0.99])
RX = [51.2217,51.3714,51.4420,51.4891,51.5246,51.5843,51.6088]
RY = [24.9809,25.1374,25.2267,25.2739,25.3269,25.5865,25.8831]
print("lon pct     mapped     ref      diff km")
for q, a, b in zip([1,5,25,50,75,95,99], qx, RX):
    print("  %3d   %.4f  %.4f  %+6.1f" % (q, a, b, (a-b)*100.3))
print("lat pct     mapped     ref      diff km")
for q, a, b in zip([1,5,25,50,75,95,99], qy, RY):
    print("  %3d   %.4f  %.4f  %+6.1f" % (q, a, b, (a-b)*111.0))
print("std lon %.5f deg = %.2f km (ref %.2f km)" % (m[:,0].std(), m[:,0].std()*100.3, 0.07715*100.3))
print("std lat %.5f deg = %.2f km (ref %.2f km)" % (m[:,1].std(), m[:,1].std()*111, 0.1447*111))
print("rho %.3f (ref -0.164)" % np.corrcoef(m[:,0], m[:,1])[0,1])
print("lon range %.4f..%.4f (ref 50.7575..51.6545)" % (m[:,0].min(), m[:,0].max()))
print("lat range %.4f..%.4f (ref 24.5606..26.1351)" % (m[:,1].min(), m[:,1].max()))

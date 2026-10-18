"""Independent re-implementation of the de-anonymisation objective (written separately
from fit.py, which it cross-checks), plus the start helpers the hold-out tests use.

Everything is in the real km frame (R=6371.0088, cos(25.3)) unless stated.
Parameters p = (M11, M12, M21, M22, t1, t2): real_km = M @ u + t, u = whitened anon km.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from scipy.optimize import minimize

DATA = Path(__file__).resolve().parent.parent
R = 6371.0088
LAT0, LON0 = 25.3, 51.2
KX = R * np.radians(1.0) * np.cos(np.radians(LAT0))
KY = R * np.radians(1.0)
Q = np.array([0.10, 0.25, 0.50, 0.75, 0.90])
DOHA = (25.2854, 51.5310)
PAIRS = {110: 1110, 130: 1130, 151: 1151, 160: 1160, 171: 1171}

# ------------------------------------------------------------------ sites
S = pd.read_csv(f'{DATA}/ds02_sites_relative.csv', dtype={'LAC_TACS': str})
N = len(S)
LONa, LATa = S.LON.to_numpy(float), S.LAT.to_numpy(float)
CODES_SITE = [set(int(float(c)) for c in str(v).split(';') if c.strip()) for v in S.LAC_TACS]
LAT_A0, LON_A0 = LATa.mean(), LONa.mean()
KXA = R * np.radians(1.0) * np.cos(np.radians(LAT_A0))
A_KM = np.c_[KXA * (LONa - LON_A0), KY * (LATa - LAT_A0)]
MU = A_KM.mean(0)
ev, V = np.linalg.eigh(np.cov((A_KM - MU).T, bias=True))
o = np.argsort(ev)[::-1]
ev, V = ev[o], V[:, o]
W = np.diag(1 / np.sqrt(ev)) @ V.T
U = (A_KM - MU) @ W.T

# ------------------------------------------------------------------ OpenCelliD
OC = pd.read_csv(f'{DATA}/opencellid_427_qatar.csv.gz')
VO = OC[OC.net == 2]
ALL_CODES = sorted(set().union(*CODES_SITE))
GROUPS = []
ANCH = {}
for g in ALL_CODES:
    radios = ['UMTS', 'LTE'] if 100 <= g <= 999 else (['GSM'] if 1000 <= g <= 9999 else None)
    if radios is None:
        assert (VO.area == g).sum() == 0
        continue
    a = VO[(VO.area == g) & VO.radio.isin(radios)][['lon', 'lat']].drop_duplicates()
    if len(a) == 0:
        continue
    GROUPS.append(g)
    ANCH[g] = np.c_[KX * (a.lon.to_numpy() - LON0), KY * (a.lat.to_numpy() - LAT0)]
G = len(GROUPS)
assert G == 20, GROUPS
MEMB = [np.array([i for i in range(N) if g in CODES_SITE[i]]) for g in GROUPS]
NG = np.array([len(m) for m in MEMB])
WG = np.sqrt(NG) / np.sqrt(NG).sum()
AQX = np.array([np.quantile(ANCH[g][:, 0], Q) for g in GROUPS])
AQY = np.array([np.quantile(ANCH[g][:, 1], Q) for g in GROUPS])
AMED = np.array([np.median(ANCH[g], axis=0) for g in GROUPS])


def make_ref(exclude_areas=()):
    """Global reference from distinct net==2 coords, optionally excluding some area codes."""
    r = VO[~VO.area.isin(list(exclude_areas))][['lon', 'lat']].drop_duplicates()
    rx, ry = KX * (r.lon.to_numpy() - LON0), KY * (r.lat.to_numpy() - LAT0)
    return dict(n=len(rx), sx=rx.std(), sy=ry.std(), rho=np.corrcoef(rx, ry)[0, 1],
                qx=np.quantile(rx, Q), qy=np.quantile(ry, Q),
                bounds=(rx.min(), rx.max(), ry.min(), ry.max()))


REF = make_ref()

# ------------------------------------------------------------------ land
OL = json.load(open(f'{DATA}/qatar_outline.json'))
LAND = shapely.Polygon([s for s in OL['shapes'] if s['name'] == 'Qatar'][0]['rings'][0])
LAND1 = LAND.buffer(1.0)
shapely.prepare(LAND)
H = 0.25
MARG = 200.0
b = LAND.bounds
GX = np.arange(np.floor(b[0] - MARG), np.ceil(b[2] + MARG) + H / 2, H)
GY = np.arange(np.floor(b[1] - MARG), np.ceil(b[3] + MARG) + H / 2, H)
_XX, _YY = np.meshgrid(GX, GY)
RAST = shapely.distance(shapely.points(_XX.ravel(), _YY.ravel()), LAND).reshape(_XX.shape)
del _XX, _YY


def land_dist_raster(x, y):
    xc = np.clip(x, GX[0], GX[-1] - 1e-9)
    yc = np.clip(y, GY[0], GY[-1] - 1e-9)
    fx, fy = (xc - GX[0]) / H, (yc - GY[0]) / H
    i, j = fx.astype(int), fy.astype(int)
    ax, ay = fx - i, fy - j
    d = (RAST[j, i] * (1 - ax) * (1 - ay) + RAST[j, i + 1] * ax * (1 - ay)
         + RAST[j + 1, i] * (1 - ax) * ay + RAST[j + 1, i + 1] * ax * ay)
    return d + np.hypot(x - xc, y - yc)


def land_dist_exact(x, y):
    return shapely.distance(shapely.points(x, y), LAND)


# ------------------------------------------------------------------ objective
NMAX = NG.max()
IDX = np.full((G, NMAX), N)
for k, m in enumerate(MEMB):
    IDX[k, :len(m)] = m
_pos = (NG[:, None] - 1) * Q
LO = np.floor(_pos).astype(int)
HI = np.minimum(LO + 1, NG[:, None] - 1)
FR = _pos - LO
RW = np.arange(G)[:, None]
_pN = (N - 1) * Q
LON_ = np.floor(_pN).astype(int)
HIN = np.minimum(LON_ + 1, N - 1)
FRN = _pN - LON_


def xy(p):
    return p[0] * U[:, 0] + p[1] * U[:, 1] + p[4], p[2] * U[:, 0] + p[3] * U[:, 1] + p[5]


def gquant(v):
    s = np.sort(np.append(v, np.inf)[IDX], axis=1)
    return s[RW, LO] + FR * (s[RW, HI] - s[RW, LO])


def terms(p, ref=REF, exact_land=False):
    x, y = xy(p)
    qx, qy = gquant(x), gquant(y)
    E = (((qx - AQX) ** 2).sum(1) + ((qy - AQY) ** 2).sum(1)) / 10.0
    xs, ys = np.sort(x), np.sort(y)
    gx = xs[LON_] + FRN * (xs[HIN] - xs[LON_])
    gy = ys[LON_] + FRN * (ys[HIN] - ys[LON_])
    sx, sy = x.std(), y.std()
    rho = np.mean((x - x.mean()) * (y - y.mean())) / (sx * sy)
    Gt = (((gx - ref['qx']) ** 2).sum() + ((gy - ref['qy']) ** 2).sum()) / 10.0 \
        + (sx - ref['sx']) ** 2 + (sy - ref['sy']) ** 2 + (10 * (rho - ref['rho'])) ** 2
    d = land_dist_exact(x, y) if exact_land else land_dist_raster(x, y)
    Land = np.mean(d * d)
    x0, x1, y0, y1 = ref['bounds']
    ex = np.maximum(x0 - x, 0) + np.maximum(x - x1, 0)
    ey = np.maximum(y0 - y, 0) + np.maximum(y - y1, 0)
    Bnd = np.mean(ex * ex + ey * ey)
    return E, Gt, Land, Bnd


def objective(p, w=WG, ref=REF, exact_land=False):
    E, Gt, Land, Bnd = terms(p, ref, exact_land)
    v = float(w @ E + 0.2 * Gt + Land + Bnd)
    return v if np.isfinite(v) else 1e30


def group_medians(p):
    x, y = xy(p)
    return np.array([[np.median(x[m]), np.median(y[m])] for m in MEMB])


def residuals(p):
    return np.hypot(*(group_medians(p) - AMED).T)


def env_rms(p, ref=REF):
    return np.sqrt(terms(p, ref)[0])


# ------------------------------------------------------------------ conversions
KR = np.diag([KX, KY])
KA = np.diag([KXA, KY])
A0 = np.array([LON_A0, LAT_A0])
R0 = np.array([LON0, LAT0])


def p_from_Bdeg(Bdeg, cdeg):
    """Invert real_deg = B anon_deg + c into whitened params."""
    Bdeg, cdeg = np.asarray(Bdeg), np.asarray(cdeg)
    B_km = KR @ Bdeg @ np.linalg.inv(KA)
    # real_km = KR (B a + c - R0) ; a = A0 + KA^-1 (a_km); a_km = W^-1 u + MU
    M = B_km @ np.linalg.inv(W)
    t = KR @ (Bdeg @ A0 + cdeg - R0) + B_km @ MU
    return np.r_[M.ravel(), t]


def Bdeg_from_p(p):
    M, t = p[:4].reshape(2, 2), p[4:]
    B_km = M @ W
    c_km = t - B_km @ MU
    Bdeg = np.linalg.inv(KR) @ B_km @ KA
    cdeg = R0 + np.linalg.inv(KR) @ c_km - Bdeg @ A0
    return Bdeg, cdeg, B_km, c_km


# ------------------------------------------------------------------ optimiser & starts
def nm_chain(p0, w=WG, ref=REF, max_rounds=40, step=None):
    p = np.asarray(p0, float)
    f = objective(p, w, ref)
    for _ in range(max_rounds):
        st = np.maximum(0.1 * np.abs(p), 0.5) if step is None else step
        sim = np.vstack([p] + [p + st[i] * np.eye(6)[i] for i in range(6)])
        r = minimize(objective, p, args=(w, ref), method='Nelder-Mead',
                     options=dict(initial_simplex=sim, xatol=1e-8, fatol=1e-11, maxiter=40000, maxfev=40000))
        if r.fun < f - 1e-11 * max(1, abs(f)):
            p, f = r.x, r.fun
        else:
            if r.fun < f:
                p, f = r.x, r.fun
            break
    return p, float(f)


def umed(keep):
    """Median of anon LON/LAT per group (degrees), mapped to whitened u."""
    med = np.array([[np.median(LONa[MEMB[k]]), np.median(LATa[MEMB[k]])] for k in keep])
    akm = np.c_[KXA * (med[:, 0] - LON_A0), KY * (med[:, 1] - LAT_A0)]
    return (akm - MU) @ W.T


def closed_form(keep):
    """Weighted LS (w = WG) of group medians -> anchor medians using ONLY groups in keep;
    returns starts for both orientations + Umeyama similarity fits for both reflections."""
    keep = list(keep)
    Um, Am, w = umed(keep), AMED[keep], WG[keep]
    Z = np.c_[Um, np.ones(len(keep))]
    P = np.linalg.solve(Z.T @ (w[:, None] * Z), Z.T @ (w[:, None] * Am))
    M = P[:2].T
    out = [np.r_[M.ravel(), P[2]]]
    Uu, s, Vt = np.linalg.svd(M)
    M2 = Uu @ np.diag([s[0], -s[1]]) @ Vt
    wn = w / w.sum()
    out.append(np.r_[M2.ravel(), wn @ Am - M2 @ (wn @ Um)])
    uc, ac = wn @ Um, wn @ Am
    for refl in (1, -1):
        F = np.diag([1.0, refl])
        Ur = (Um - uc) @ F.T
        Hm = (w[:, None] * Ur).T @ (Am - ac)
        a, sh, bt = np.linalg.svd(Hm)
        d = np.sign(np.linalg.det(bt.T @ a.T))
        Rm = bt.T @ np.diag([1, d]) @ a.T
        sc = (sh @ np.array([1, d])) / (w * (Ur ** 2).sum(1)).sum()
        Ms = sc * Rm @ F
        out.append(np.r_[Ms.ravel(), ac - Ms @ uc])
    return out


def rot(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def random_starts(n, rng, keep, smin=0.3, smax=80.0):
    keep = list(keep)
    wn = WG[keep] / WG[keep].sum()
    t0 = wn @ AMED[keep]
    out = []
    for i in range(n):
        s = np.exp(rng.uniform(np.log(smin), np.log(smax), 2))
        refl = np.diag([1.0, -1.0 if i % 2 else 1.0])
        M = rot(rng.uniform(0, 2 * np.pi)) @ np.diag(s) @ rot(rng.uniform(0, 2 * np.pi)) @ refl
        out.append(np.r_[M.ravel(), t0])
    return out


def det_sign_B(p):
    return int(np.sign(np.linalg.det(p[:4].reshape(2, 2) @ W)))

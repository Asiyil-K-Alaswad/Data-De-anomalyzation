#!/usr/bin/env python3
"""Affine de-anonymisation of DS02 site coordinates (independent re-implementation).

Model:  [lon, lat]_real = B @ [LON, LAT]_anon + c   (degrees, 6 parameters)
Fitted in a whitened parameterisation  real_km = M @ u + t  where u are the
whitened anonymised coordinates.  Anchors are per-area quantile envelopes of
OpenCelliD cells of MCC 427 / MNC 2 (Vodafone Qatar).

Objective  L = sum_g w_g E_g + 0.2 G + 1.0 Land + 1.0 Bnd   (see README in result.json notes)

Run:  python3 fit.py   (about 6 min on 4 cores; writes deanon_fit.json,
      ds02_sites_deanonymised.csv and fit_diagnostic.png next to this file)
Then run holdout.py for the leak-free leave-one-code-out / leave-one-area-out checks.
"""
import json
import os
import sys
import tempfile
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Polygon
from scipy.optimize import minimize

T_START = time.time()
OUT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(OUT)

# ----------------------------------------------------------------------------- constants
R = 6371.0088
LAT0, LON0 = 25.3, 51.2
KX = R * np.pi / 180 * np.cos(np.radians(LAT0))   # km per degree lon (real frame)
KY = R * np.pi / 180                               # km per degree lat
Q = np.array([0.10, 0.25, 0.50, 0.75, 0.90])
CODES = [110, 111, 120, 121, 130, 131, 141, 142, 143, 150, 151, 160, 161, 171, 181,
         1110, 1130, 1151, 1160, 1171]
EXCLUDED = [1, 10101, 14101, 15101]
W_GLOBAL, W_LAND, W_BND = 0.2, 1.0, 1.0
RASTER_RES, RASTER_MARGIN = 0.25, 60.0
DOHA_LAT, DOHA_LON = 25.2854, 51.5310
DOHA_RADIUS_KM = 8.0
SEED = 20261018
N_RANDOM = 64          # random starts for the full fit (half per orientation)
N_RANDOM_LOO = 10      # random starts per LOO refit (half per orientation)
N_POLISH = 4           # best distinct NM solutions per orientation that get polished
SCALE = 10.0           # theta units: M = SCALE*theta[:4], t = T0 + SCALE*theta[4:]
NPROC = max(1, min(4, os.cpu_count() or 1))


NOTES = (
    'Independent re-implementation. Objective L = sum_g w_g E_g + 0.2 G + 1.0 Land + 1.0 Bnd with '
    'w_g = sqrt(n_g)/sum sqrt(n_h); quantiles use numpy linear interpolation; sigma is population SD '
    '(computed analytically from M because u is exactly whitened); Bnd = mean over sites of '
    '(excess_x^2 + excess_y^2). Land distance from a 0.25 km bilinear raster (60 km margin; beyond it '
    'clamp + Euclidean offset) of the Natural Earth 1:10m Qatar ring (GADM unavailable); that coastline '
    'lacks reclaimed land (The Pearl / West Bay), so 39 of the off-land sites are within 1.5 km of the '
    'coast (n_off_land_1km counts those beyond a 1 km buffer). Opposite-orientation median start = nearest '
    'matrix with flipped det (smaller singular value negated) with t refit by LS. Unconstrained NM/Powell '
    'cross det(B)=0 freely, so each orientation was also searched separately in an orientation-preserving '
    'QR parameterisation M = Rot(phi)[[e^a, b],[0, s e^c]]; in det>0 the infimum is a rank-1 collapse '
    '(orientation.det_pos_degenerate_infimum) and the best non-degenerate det>0 local minimum is '
    'orientation.det_pos. L-BFGS-B uses finite-difference gradients on a piecewise-smooth objective and '
    'rarely moved; Powell + a short NM chain gave the final polish. LOO drops w_g E_g without '
    'renormalising the other weights; loo_warm_vs_leakfree_km = distance between held-out group fitted '
    'medians of the best warm-start and best leak-free solutions.')


def to_km(lon, lat):
    return KX * (np.asarray(lon) - LON0), KY * (np.asarray(lat) - LAT0)


def to_deg(x, y):
    return LON0 + np.asarray(x) / KX, LAT0 + np.asarray(y) / KY


# ----------------------------------------------------------------------------- DS02 sites
sites = pd.read_csv(f'{SRC}/ds02_sites_relative.csv', dtype={'LAC_TACS': str})
N = len(sites)
LAT_A0, LON_A0 = float(sites.LAT.mean()), float(sites.LON.mean())
KXA = R * np.pi / 180 * np.cos(np.radians(LAT_A0))
KYA = R * np.pi / 180
A_KM = np.c_[KXA * (sites.LON.values - LON_A0), KYA * (sites.LAT.values - LAT_A0)]
MU = A_KM.mean(0)
cov = np.cov((A_KM - MU).T, bias=True)
ev, V = np.linalg.eigh(cov)
order = np.argsort(ev)[::-1]
ev, V = ev[order], V[:, order]
SD = np.sqrt(ev)
WHITEN = (V / SD).T                      # u = WHITEN @ (a_km - MU)
U = (A_KM - MU) @ WHITEN.T               # (N,2), unit variance, uncorrelated
DET_WHITEN_SIGN = np.sign(np.linalg.det(WHITEN))

site_codes = [set(int(c) for c in str(v).split(';') if c.strip() != '') for v in sites.LAC_TACS]
all_codes = sorted(set().union(*site_codes))

# ----------------------------------------------------------------------------- OpenCelliD
ocid = pd.read_csv(f'{SRC}/opencellid_427_qatar.csv.gz')
voda = ocid[ocid.net == 2]
anchor = {}
for g in CODES:
    r = voda[voda.area == g]
    r = r[r.radio.isin(['UMTS', 'LTE'])] if g < 1000 else r[r.radio == 'GSM']
    d = r[['lon', 'lat']].drop_duplicates()
    ax, ay = to_km(d.lon.values, d.lat.values)
    anchor[g] = dict(x=ax, y=ay, n_cells=int(len(r)), n=int(len(d)))
for g in EXCLUDED:
    assert (voda.area == g).sum() == 0, g

ref = voda[['lon', 'lat']].drop_duplicates()
RX, RY = to_km(ref.lon.values, ref.lat.values)
REF = dict(sigma_x=float(RX.std()), sigma_y=float(RY.std()),
           rho=float(np.corrcoef(RX, RY)[0, 1]),
           qx=np.quantile(RX, Q), qy=np.quantile(RY, Q), n=int(len(RX)))
BX0, BX1, BY0, BY1 = RX.min(), RX.max(), RY.min(), RY.max()

# ----------------------------------------------------------------------------- groups
G = len(CODES)
members = [np.array([i for i in range(N) if g in site_codes[i]]) for g in CODES]
NG = np.array([len(m) for m in members])
WG = np.sqrt(NG) / np.sqrt(NG).sum()
NMAX = NG.max()
UP = np.zeros((G, NMAX, 2))
PAD = np.ones((G, NMAX), bool)
for k, m in enumerate(members):
    UP[k, :len(m)] = U[m]
    PAD[k, :len(m)] = False


def qpos(n):
    p = Q * (n - 1)
    lo = np.floor(p).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    return lo, hi, p - lo


_lohif = [qpos(n) for n in NG]
GLO = np.array([a[0] for a in _lohif])         # (G,5)
GHI = np.array([a[1] for a in _lohif])
GF = np.array([a[2] for a in _lohif])[..., None]
GI = np.repeat(np.arange(G)[:, None], len(Q), 1)
NLO, NHI, NF = qpos(N)
AQ = np.array([[np.quantile(anchor[g]['x'], Q), np.quantile(anchor[g]['y'], Q)] for g in CODES])
AQ = AQ.transpose(0, 2, 1)                     # (G,5,2) to match fitted layout
AMED = np.array([[np.median(anchor[g]['x']), np.median(anchor[g]['y'])] for g in CODES])
T0 = (WG[:, None] * AMED).sum(0)               # anchor centroid (weighted group medians)
REFQ = np.c_[REF['qx'], REF['qy']]             # (5,2)

# ----------------------------------------------------------------------------- land raster
with open(f'{SRC}/qatar_outline.json') as fh:
    outline = json.load(fh)
qshape = [s for s in outline['shapes'] if s['name'] == 'Qatar'][0]
assert len(qshape['rings']) == 1
LAND = Polygon(qshape['rings'][0])
assert LAND.is_valid
LAND1 = LAND.buffer(1.0)
lx0, ly0, lx1, ly1 = LAND.bounds
GX0 = np.floor((lx0 - RASTER_MARGIN) / RASTER_RES) * RASTER_RES
GY0 = np.floor((ly0 - RASTER_MARGIN) / RASTER_RES) * RASTER_RES
NXR = int(np.ceil((lx1 + RASTER_MARGIN - GX0) / RASTER_RES)) + 1
NYR = int(np.ceil((ly1 + RASTER_MARGIN - GY0) / RASTER_RES)) + 1
_cache = f'{tempfile.gettempdir()}/ds02_land_raster_{RASTER_RES}_{RASTER_MARGIN}.npy'
if os.path.exists(_cache):
    DR = np.load(_cache)
else:
    gx = GX0 + RASTER_RES * np.arange(NXR)
    gy = GY0 + RASTER_RES * np.arange(NYR)
    XX, YY = np.meshgrid(gx, gy)
    DR = shapely.distance(LAND, shapely.points(XX.ravel(), YY.ravel())).reshape(NYR, NXR)
    np.save(_cache, DR)


DRF = DR.ravel()


def land_dist_xy(x, y):
    fx = (x - GX0) / RASTER_RES
    fy = (y - GY0) / RASTER_RES
    fxc = np.clip(fx, 0, NXR - 1 - 1e-9)
    fyc = np.clip(fy, 0, NYR - 1 - 1e-9)
    i = fxc.astype(np.int64)
    j = fyc.astype(np.int64)
    ax = fxc - i
    ay = fyc - j
    k = j * NXR + i
    d00 = DRF[k]
    d01 = DRF[k + 1]
    d10 = DRF[k + NXR]
    d11 = DRF[k + NXR + 1]
    d0 = d00 + ax * (d01 - d00)
    d1 = d10 + ax * (d11 - d10)
    d = d0 + ay * (d1 - d0)
    return d + RASTER_RES * np.hypot(fx - fxc, fy - fyc)   # beyond raster: clamp + offset


def land_dist(X):
    return land_dist_xy(X[:, 0], X[:, 1])


# ----------------------------------------------------------------------------- model / objective
def unpack(theta):
    th = np.asarray(theta, float)
    return SCALE * th[:4].reshape(2, 2), T0 + SCALE * th[4:]


def pack(M, t):
    return np.r_[np.asarray(M).ravel() / SCALE, (np.asarray(t) - T0) / SCALE]


U0 = np.ascontiguousarray(U[:, 0])
U1 = np.ascontiguousarray(U[:, 1])
IDX = np.full((G, NMAX), N, dtype=np.int64)           # padded group index -> +inf sentinel
for _k, _m in enumerate(members):
    IDX[_k, :len(_m)] = _m
GLO_F = (np.arange(G)[:, None] * NMAX + GLO)          # flat positions into sorted (G,NMAX)
GHI_F = (np.arange(G)[:, None] * NMAX + GHI)
GF2 = GF[..., 0]
AQX = AQ[:, :, 0]
AQY = AQ[:, :, 1]
_INF1 = np.array([np.inf])


def components(theta):
    """Per-group envelope errors E_g (G,), global term G, Land, Bnd.

    Because u is exactly whitened (population covariance = I), the fitted cloud's
    covariance is M M^T, so sigma_x, sigma_y and rho are computed analytically."""
    M, t = unpack(theta)
    x = U0 * M[0, 0] + U1 * M[0, 1] + t[0]
    y = U0 * M[1, 0] + U1 * M[1, 1] + t[1]
    sx = np.sort(np.concatenate((x, _INF1))[IDX], axis=1).ravel()
    sy = np.sort(np.concatenate((y, _INF1))[IDX], axis=1).ravel()
    fqx = sx[GLO_F] * (1 - GF2) + sx[GHI_F] * GF2
    fqy = sy[GLO_F] * (1 - GF2) + sy[GHI_F] * GF2
    E = (((fqx - AQX) ** 2).sum(1) + ((fqy - AQY) ** 2).sum(1)) / (2 * len(Q))
    xs = np.sort(x)
    ys = np.sort(y)
    gqx = xs[NLO] * (1 - NF) + xs[NHI] * NF
    gqy = ys[NLO] * (1 - NF) + ys[NHI] * NF
    sdx = np.hypot(M[0, 0], M[0, 1])
    sdy = np.hypot(M[1, 0], M[1, 1])
    rho = (M[0, 0] * M[1, 0] + M[0, 1] * M[1, 1]) / (sdx * sdy + 1e-300)
    Gt = ((((gqx - REFQ[:, 0]) ** 2).sum() + ((gqy - REFQ[:, 1]) ** 2).sum()) / (2 * len(Q))
          + (sdx - REF['sigma_x']) ** 2 + (sdy - REF['sigma_y']) ** 2 + (10 * (rho - REF['rho'])) ** 2)
    dl = land_dist_xy(x, y)
    land = np.dot(dl, dl) / N
    ex = np.maximum(BX0 - x, 0) + np.maximum(x - BX1, 0)
    ey = np.maximum(BY0 - y, 0) + np.maximum(y - BY1, 0)
    bnd = (np.dot(ex, ex) + np.dot(ey, ey)) / N
    return E, Gt, land, bnd


def components_slow(theta):
    """Straightforward reference implementation (np.quantile / np.std) for self-checks."""
    M, t = unpack(theta)
    X = U @ M.T + t
    E = np.array([np.mean([(np.quantile(X[m, a], Q) - AQ[k, :, a]) ** 2 for a in (0, 1)])
                  for k, m in enumerate(members)])
    gq = np.c_[np.quantile(X[:, 0], Q), np.quantile(X[:, 1], Q)]
    sd = X.std(0)
    rho = np.corrcoef(X[:, 0], X[:, 1])[0, 1]
    Gt = (((gq - REFQ) ** 2).mean() + (sd[0] - REF['sigma_x']) ** 2 + (sd[1] - REF['sigma_y']) ** 2
          + (10 * (rho - REF['rho'])) ** 2)
    on = shapely.contains_xy(LAND, X[:, 0], X[:, 1])
    dl = np.where(on, 0.0, shapely.distance(LAND, shapely.points(X[:, 0], X[:, 1])))
    land = (dl ** 2).mean()
    ex = np.clip(BX0 - X[:, 0], 0, None) + np.clip(X[:, 0] - BX1, 0, None)
    ey = np.clip(BY0 - X[:, 1], 0, None) + np.clip(X[:, 1] - BY1, 0, None)
    return E, Gt, land, (ex ** 2 + ey ** 2).mean()


def objective(theta, w=WG):
    E, Gt, land, bnd = components(theta)
    val = float(w @ E + W_GLOBAL * Gt + W_LAND * land + W_BND * bnd)
    return val if np.isfinite(val) else 1e30


def to_B(theta):
    """Return B_km, c_km (anon km frame -> real km frame) and B_deg, c_deg."""
    M, t = unpack(theta)
    B_km = M @ WHITEN
    c_km = t - B_km @ MU
    Da = np.diag([KXA, KYA])
    Dr_inv = np.diag([1 / KX, 1 / KY])
    B_deg = Dr_inv @ B_km @ Da
    c_deg = np.array([LON0, LAT0]) + Dr_inv @ (c_km - B_km @ Da @ np.array([LON_A0, LAT_A0]))
    return B_km, c_km, B_deg, c_deg


def det_sign(theta):
    return int(np.sign(np.linalg.det(to_B(theta)[0])))


def fitted(theta):
    M, t = unpack(theta)
    return U @ M.T + t


# ----------------------------------------------------------------------------- starts
def median_fit(exclude=None):
    """Closed-form weighted LS: anon group median (whitened) -> anchor median.
    Returns starts for both orientations (unconstrained + nearest opposite-det)."""
    keep = [k for k in range(G) if k != exclude]
    medd = np.array([[np.median(sites.LON.values[members[k]]), np.median(sites.LAT.values[members[k]])]
                     for k in keep])
    ma = np.c_[KXA * (medd[:, 0] - LON_A0), KYA * (medd[:, 1] - LAT_A0)]
    mu_ = (ma - MU) @ WHITEN.T
    a_ = AMED[keep]
    w = WG[keep]
    Z = np.c_[mu_, np.ones(len(keep))]
    P = np.linalg.solve(Z.T @ (w[:, None] * Z), Z.T @ (w[:, None] * a_))
    M1, t1 = P[:2].T, P[2]
    # opposite orientation: flip sign of the smaller singular value (nearest matrix
    # with opposite det in Frobenius norm), then closed-form LS for t given M.
    Uu, s, Vt = np.linalg.svd(M1)
    M2 = Uu @ np.diag([s[0], -s[1]]) @ Vt
    wm = w / w.sum()
    t2 = (wm[:, None] * a_).sum(0) - M2 @ (wm[:, None] * mu_).sum(0)
    out = {}
    for M_, t_ in [(M1, t1), (M2, t2)]:
        th = pack(M_, t_)
        out[det_sign(th)] = th
    return out


def random_starts(n, rng):
    res = []
    for k in range(n):
        ang = rng.uniform(0, 2 * np.pi)
        Rm = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        sc = np.exp(rng.uniform(np.log(2.0), np.log(40.0), 2))
        want = 1 if k % 2 == 0 else -1                 # alternate B orientation
        f = want * DET_WHITEN_SIGN
        M = Rm @ np.diag(sc) @ np.diag([1.0, f])
        res.append(pack(M, T0))
    return res


# ----------------------------------------------------------------------------- optimiser
# Orientation-preserving parameterisation (QR form):  M = Rot(phi) @ SCALE*[[e^la, b], [0, sg*e^lc]],
# t = T0 + SCALE*[t1, t2];  q = (phi, la, b, lc, t1, t2).  sg = sign(det M) is fixed, so an
# optimiser working in q can never cross det = 0 (used to get the best fit in EACH orientation).
def to_q(theta):
    M, t = unpack(theta)
    phi = np.arctan2(M[1, 0], M[0, 0])
    c, s_ = np.cos(phi), np.sin(phi)
    Rm = np.array([[c, s_], [-s_, c]]) @ M
    q = np.r_[phi, np.log(max(Rm[0, 0], 1e-9) / SCALE), Rm[0, 1] / SCALE,
              np.log(max(abs(Rm[1, 1]), 1e-9) / SCALE), (t - T0) / SCALE]
    return q, int(np.sign(Rm[1, 1])) or 1


def from_q(q, sg):
    phi, la, b, lc = q[:4]
    c, s_ = np.cos(phi), np.sin(phi)
    Rm = SCALE * np.array([[np.exp(la), b], [0.0, sg * np.exp(lc)]])
    return pack(np.array([[c, -s_], [s_, c]]) @ Rm, T0 + SCALE * np.asarray(q[4:]))


def make_f(w, sg):
    if sg is None:
        return lambda th: objective(th, w)
    return lambda q: objective(from_q(q, sg), w) if np.all(np.abs(q[[1, 3]]) < 50) else 1e30


def nm_generic(f, x0, step=0.2, restep=0.05, max_rounds=40, tol=1e-10):
    """Nelder-Mead restarted from its own result (fresh simplex) until it stops improving."""
    x = np.asarray(x0, float)
    fx = f(x)
    nfev, rounds = 1, 0
    st = step
    while rounds < max_rounds:
        rounds += 1
        simplex = np.vstack([x] + [x + st * e for e in np.eye(len(x))])
        r = minimize(f, x, method='Nelder-Mead',
                     options=dict(initial_simplex=simplex, xatol=1e-8, fatol=1e-12,
                                  maxiter=30000, maxfev=30000))
        nfev += r.nfev
        if r.fun < fx - tol * max(1.0, abs(fx)):
            x, fx = r.x, r.fun
            st = restep
        else:
            break
    return dict(x=x, f=float(fx), nfev=int(nfev), rounds=rounds)


def nm_restart(x0, w=WG, step=0.2, restep=0.05, max_rounds=40, tol=1e-10):
    return nm_generic(make_f(w, None), x0, step, restep, max_rounds, tol)


def polish(x0, w=WG, sg=None):
    """Polish with Powell and L-BFGS-B (finite-difference gradient) on the same objective,
    then a short NM chain from the best of them; keep the best.  sg=None works in theta
    (unconstrained); sg=+-1 works in the orientation-preserving q parameterisation, x0 in q."""
    f = make_f(w, sg)
    x0 = np.asarray(x0, float)
    out = {'nm': (x0, f(x0))}
    try:
        r = minimize(f, x0, method='Powell', options=dict(xtol=1e-9, ftol=1e-13, maxiter=200000, maxfev=200000))
        out['powell'] = (r.x, float(r.fun))
    except Exception as e:  # pragma: no cover
        print('powell failed', e)
    try:
        r = minimize(f, x0, method='L-BFGS-B', options=dict(maxiter=20000, maxfun=200000, ftol=1e-15, gtol=1e-10, eps=1e-8))
        out['lbfgsb'] = (r.x, float(r.fun))
    except Exception as e:  # pragma: no cover
        print('lbfgsb failed', e)
    bk = min(out, key=lambda k: out[k][1])
    r = nm_generic(f, out[bk][0], step=0.02, restep=0.02)
    out['chain_nm'] = (r['x'], r['f'])
    best = min(out, key=lambda k: out[k][1])
    xb = out[best][0]
    return dict(x=xb if sg is None else from_q(xb, sg), q=None if sg is None else xb, f=out[best][1],
                method=best, f_by_method={k: float(v[1]) for k, v in out.items()})


def _job(args):
    tag, x0, w, step = args
    r = nm_restart(x0, w=w, step=step)
    r['tag'] = tag
    r['x0'] = np.asarray(x0)
    return r


def _ojob(args):
    """Orientation-constrained NM multi-start job: x0 is a theta, forced into orientation sg."""
    tag, x0, w, sg, step = args
    q0, _ = to_q(x0)
    r = nm_generic(make_f(w, sg), q0, step=step)
    r['q'] = r['x']
    r['x'] = from_q(r['q'], sg)
    r['tag'] = tag
    return r


def _polish_job(args):
    tag, x0, w = args[:3]
    sg = args[3] if len(args) > 3 else None
    r = polish(to_q(x0)[0] if sg is not None else x0, w, sg)
    r['tag'] = tag
    return r


def distinct(results, tol=1e-3):
    keep = []
    for r in sorted(results, key=lambda r: r['f']):
        if all(np.max(np.abs(r['x'] - k['x'])) > tol for k in keep):
            keep.append(r)
    return keep


# ----------------------------------------------------------------------------- metrics
def group_metrics(theta):
    X = fitted(theta)
    E = components(theta)[0]
    res = []
    for k, g in enumerate(CODES):
        xm = np.median(X[members[k]], axis=0)
        res.append(dict(residual_km=float(np.hypot(*(xm - AMED[k]))), env_rms_km=float(np.sqrt(E[k])),
                        fit_median_km=xm))
    return res


def moments(X):
    return dict(sigma_x=float(X[:, 0].std()), sigma_y=float(X[:, 1].std()),
                rho=float(np.corrcoef(X[:, 0], X[:, 1])[0, 1]))


def svals(Bm):
    s = np.linalg.svd(Bm, compute_uv=False)
    return [float(v) for v in s], float(s[0] / s[-1])


def describe(theta):
    B_km, c_km, B_deg, c_deg = to_B(theta)
    skm, ckm = svals(B_km)
    sdg, cdg = svals(B_deg)
    X = fitted(theta)
    on = shapely.contains_xy(LAND, X[:, 0], X[:, 1])
    E, Gt, land, bnd = components(theta)
    gm = group_metrics(theta)
    r = np.array([m['residual_km'] for m in gm])
    return dict(objective=objective(theta), det_sign=int(np.sign(np.linalg.det(B_km))),
                B_deg=B_deg.tolist(), c_deg=c_deg.tolist(), B_km=B_km.tolist(), c_km=c_km.tolist(),
                singular_values_km=skm, condition_km=ckm, singular_values_deg=sdg, condition_deg=cdg,
                terms=dict(anchor=float(WG @ E), global_=float(Gt), land=float(land), bnd=float(bnd)),
                on_land_pct=float(100 * on.mean()), n_off_land=int((~on).sum()),
                residual_min_km=float(r.min()), residual_median_km=float(np.median(r)),
                residual_max_km=float(r.max()), moments=moments(X))


# ============================================================================= main
def main():
    rng = np.random.default_rng(SEED)
    log = {}
    print(f'sites={N}  groups={G}  n_g={NG.tolist()}')
    print(f'whitening sd (anon km) = {SD}, ratio {SD[0] / SD[1]:.2f}')
    print(f'ref: {REF["n"]} distinct coords  sigma_x={REF["sigma_x"]:.3f} sigma_y={REF["sigma_y"]:.3f} rho={REF["rho"]:.4f}')
    print(f'raster {NXR}x{NYR} @ {RASTER_RES} km')

    # --- raster validation against shapely
    vr = np.random.default_rng(1)
    bx = vr.uniform(lx0 - 30, lx1 + 30, 4000)
    by = vr.uniform(ly0 - 30, ly1 + 30, 4000)
    exact = shapely.distance(LAND, shapely.points(bx, by))
    approx = land_dist(np.c_[bx, by])
    far = np.c_[vr.uniform(lx0 - 300, lx1 + 300, 1000), vr.uniform(ly0 - 300, ly1 + 300, 1000)]
    exf = shapely.distance(LAND, shapely.points(far[:, 0], far[:, 1]))
    apf = land_dist(far)
    log['raster_validation'] = dict(
        n=4000, max_abs_err_km=float(np.abs(exact - approx).max()),
        mean_abs_err_km=float(np.abs(exact - approx).mean()),
        far_n=1000, far_max_rel_overestimate=float(np.max((apf - exf) / np.maximum(exf, 1))),
        far_min_rel=float(np.min((apf - exf) / np.maximum(exf, 1))))
    print('raster validation', log['raster_validation'])

    # --- fast objective vs straightforward reference implementation
    chk = []
    for k in range(6):
        th = random_starts(2, vr)[k % 2] + vr.normal(0, 0.05, 6)
        a = components(th)
        b = components_slow(th)
        chk.append(max(np.max(np.abs(a[0] - b[0])), abs(a[1] - b[1]), abs(a[2] - b[2]) * 0, abs(a[3] - b[3])))
        chk.append(abs(a[2] - b[2]) / max(b[2], 1e-9) if b[2] > 1 else abs(a[2] - b[2]))
    log['fast_vs_slow_max_diff'] = float(np.max(chk))
    print('fast vs slow objective components max diff', log['fast_vs_slow_max_diff'])

    t0 = time.time()
    for _ in range(200):
        objective(pack(np.eye(2) * 10, T0))
    log['objective_eval_us'] = (time.time() - t0) / 200 * 1e6
    print(f'objective eval {log["objective_eval_us"]:.0f} us')

    # --- full fit: starts
    mf = median_fit()
    starts = [(f'median_start{s:+d}', x, WG, 0.2) for s, x in sorted(mf.items())]
    rand_full = random_starts(N_RANDOM, rng)
    for k, x in enumerate(rand_full):
        starts.append((f'random{k:02d}_start{det_sign(x):+d}', x, WG, 0.5))
    print(f'full fit: {len(starts)} starts; median-fit starts objective',
          {s: round(objective(x), 3) for s, x in mf.items()})
    with Pool(NPROC) as pool:
        nm = pool.map(_job, starts)
    for r in nm:
        r['det'] = det_sign(r['x'])
    print(f'NM done {time.time() - T_START:.0f}s')
    nm_sorted = sorted(nm, key=lambda r: r['f'])
    for r in nm_sorted[:10]:
        print(f"  {r['tag']:22s} f={r['f']:.5f} det={r['det']:+d} nfev={r['nfev']} rounds={r['rounds']}")

    # --- unconstrained polish (theta space) of the best few distinct NM solutions per final orientation
    pjobs = []
    for s in (+1, -1):
        cand = distinct([r for r in nm if r['det'] == s])[:N_POLISH]
        pjobs += [(r['tag'], r['x'], WG) for r in cand]
    with Pool(NPROC) as pool:
        pol = pool.map(_polish_job, pjobs)
    nm_by_tag = {r['tag']: r for r in nm}
    for p in pol:
        p['det'] = det_sign(p['x'])
        p['f_nm'] = nm_by_tag[p['tag']]['f']
        p['f_nm_det'] = nm_by_tag[p['tag']]['det']
        print(f"  polish {p['tag']:22s} nm={p['f_nm']:.6f} (det {p['f_nm_det']:+d}) -> {p['f']:.6f} via {p['method']} det={p['det']:+d}  {p['f_by_method']}")

    # --- orientation-constrained multi-start + polish: best solution in EACH orientation.
    # Unconstrained NM/Powell can walk through det(B)=0, so the det>0 basin is searched
    # separately in the QR parameterisation, which keeps sign(det) fixed.
    ojobs = []
    for s in (+1, -1):
        sg = s * int(DET_WHITEN_SIGN)
        seeds = [(f'o{s:+d}_median', mf[s])] + [(f'o{s:+d}_random{k:02d}', x) for k, x in enumerate(rand_full)]
        seeds += [(f'o{s:+d}_fromNM_{r["tag"]}', r['x']) for r in distinct([r for r in nm if r['det'] == s])[:N_POLISH]]
        ojobs += [(tag, x, WG, sg, 0.2) for tag, x in seeds]
    with Pool(NPROC) as pool:
        onm = pool.map(_ojob, ojobs)
    # Orientation of oriented results = the constraint sign (numerical det of a collapsed,
    # rank-1 map is meaningless); 'degenerate' flags a collapse onto a line.
    def sv_ratio(th):
        sv = np.linalg.svd(to_B(th)[0], compute_uv=False)
        return float(sv[-1] / sv[0])
    DEGEN = 1e-3
    for r in onm:
        r['det'] = +1 if r['tag'].startswith('o+1') else -1
    opjobs = []
    for s in (+1, -1):
        sg = s * int(DET_WHITEN_SIGN)
        cand = distinct([r for r in onm if r['det'] == s])[:N_POLISH]
        cand += [r for r in distinct([r for r in onm if r['det'] == s and sv_ratio(r['x']) > DEGEN])[:N_POLISH]
                 if all(r['tag'] != c['tag'] for c in cand)]
        opjobs += [(r['tag'], r['x'], WG, sg) for r in cand]
    with Pool(NPROC) as pool:
        opol = pool.map(_polish_job, opjobs)
    onm_by_tag = {r['tag']: r for r in onm}
    for p in opol:
        p['det'] = onm_by_tag[p['tag']]['det']
        p['f_nm'] = onm_by_tag[p['tag']]['f']
        print(f"  o-polish {p['tag']:30s} nm={p['f_nm']:.6f} -> {p['f']:.6f} via {p['method']} orient={p['det']:+d} "
              f"sv_ratio={sv_ratio(p['x']):.2e}")
    orient_stats = {}
    for s in (+1, -1):
        rs = [r for r in onm if r['det'] == s]
        fs = np.array([r['f'] for r in rs])
        nd = np.array([sv_ratio(r['x']) > DEGEN for r in rs])
        orient_stats['det_pos' if s > 0 else 'det_neg'] = dict(
            n_starts=int(len(fs)), best_nm=float(fs.min()), n_within_0p001_of_best=int((fs < fs.min() + 1e-3).sum()),
            n_degenerate=int((~nd).sum()), best_nm_nondegenerate=float(fs[nd].min()) if nd.any() else None)
    print('orientation-constrained NM stats', orient_stats)
    for r in nm + pol:
        r['det'] = det_sign(r['x'])
    allsol = [dict(x=r['x'], f=r['f'], det=r['det'], tag=r['tag'], method='nm') for r in nm] + \
             [dict(x=p['x'], f=p['f'], det=p['det'], tag=p['tag'], method=p['method']) for p in pol] + \
             [dict(x=r['x'], f=r['f'], det=r['det'], tag=r['tag'], method='nm_oriented') for r in onm] + \
             [dict(x=p['x'], f=p['f'], det=p['det'], tag=p['tag'], method=p['method'] + '_oriented') for p in opol]
    for a_ in allsol:
        a_['sv_ratio'] = sv_ratio(a_['x'])
        a_['degenerate'] = bool(a_['sv_ratio'] <= DEGEN)
    best_by_det, best_nd_by_det = {}, {}
    for s in (+1, -1):
        c = [a_ for a_ in allsol if a_['det'] == s]
        if c:
            best_by_det[s] = min(c, key=lambda a_: a_['f'])
        c = [a_ for a_ in c if not a_['degenerate']]
        if c:
            best_nd_by_det[s] = min(c, key=lambda a_: a_['f'])
    best = min(best_nd_by_det.values(), key=lambda a_: a_['f'])
    theta = best['x']
    Lstar = objective(theta)
    print(f'BEST f={Lstar:.6f} det={best["det"]:+d} from {best["tag"]} ({best["method"]})')
    orient = {}
    for s in (+1, -1):
        key = 'det_pos' if s > 0 else 'det_neg'
        if s in best_nd_by_det:
            b_ = best_nd_by_det[s]
            d = describe(b_['x'])
            d.update(tag=b_['tag'], method=b_['method'], sv_ratio=b_['sv_ratio'], degenerate=False)
            orient[key] = d
        if s in best_by_det and best_by_det[s]['degenerate']:
            b_ = best_by_det[s]
            d = describe(b_['x'])
            d.update(tag=b_['tag'], method=b_['method'], sv_ratio=b_['sv_ratio'], degenerate=True,
                     note='infimum in this orientation is reached by collapsing the map to rank 1 '
                          '(det -> 0); numerical det sign of this point is not meaningful')
            orient[key + '_degenerate_infimum'] = d
    if len(best_nd_by_det) == 2:
        lose_s = +1 if best['det'] < 0 else -1
        lose = best_nd_by_det[lose_s]
        orient['winning_orientation'] = 'det_pos' if best['det'] > 0 else 'det_neg'
        orient['losing_orientation'] = 'det_pos' if lose_s > 0 else 'det_neg'
        orient['delta_objective'] = float(lose['f'] - best['f'])
        orient['ratio_objective'] = float(lose['f'] / best['f'])
        if lose_s in best_by_det:
            orient['delta_objective_vs_losing_infimum'] = float(best_by_det[lose_s]['f'] - best['f'])
            orient['ratio_objective_vs_losing_infimum'] = float(best_by_det[lose_s]['f'] / best['f'])
    orient['constrained_multistart'] = orient_stats
    for k in ('det_pos', 'det_pos_degenerate_infimum', 'det_neg', 'det_neg_degenerate_infimum'):
        if k in orient:
            o = orient[k]
            print(f"  {k}: f={o['objective']:.5f} sv_km={np.round(o['singular_values_km'], 4)} on_land={o['on_land_pct']:.2f}% "
                  f"resid {o['residual_min_km']:.2f}-{o['residual_max_km']:.2f} km")
    print('  orientation delta', {k: orient.get(k) for k in ('losing_orientation', 'delta_objective', 'ratio_objective',
                                                            'delta_objective_vs_losing_infimum', 'ratio_objective_vs_losing_infimum')})

    # --- full-fit metrics
    full = describe(theta)
    gm = group_metrics(theta)
    X = fitted(theta)
    on = shapely.contains_xy(LAND, X[:, 0], X[:, 1])
    on1 = shapely.contains_xy(LAND1, X[:, 0], X[:, 1])
    lon, lat = to_deg(X[:, 0], X[:, 1])

    # --- leave-one-group-out
    loo_jobs = []
    for k in range(G):
        w = WG.copy()
        w[k] = 0.0
        loo_jobs.append((f'loo{k:02d}_warm', theta, w, 0.05))
        for s, x in sorted(median_fit(exclude=k).items()):
            loo_jobs.append((f'loo{k:02d}_median_start{s:+d}', x, w, 0.2))
        for j, x in enumerate(random_starts(N_RANDOM_LOO, rng)):
            loo_jobs.append((f'loo{k:02d}_random{j:02d}', x, w, 0.5))
    print(f'LOO: {len(loo_jobs)} NM jobs')
    with Pool(NPROC) as pool:
        loo_nm = pool.map(_job, loo_jobs)
    print(f'LOO NM done {time.time() - T_START:.0f}s')
    # polish best warm + best leak-free per group
    lp_jobs = []
    for k in range(G):
        w = WG.copy()
        w[k] = 0.0
        rk = [r for r in loo_nm if r['tag'].startswith(f'loo{k:02d}_')]
        warm = [r for r in rk if r['tag'].endswith('_warm')][0]
        lf = min([r for r in rk if not r['tag'].endswith('_warm')], key=lambda r: r['f'])
        lp_jobs += [(warm['tag'], warm['x'], w), (lf['tag'], lf['x'], w)]
    with Pool(NPROC) as pool:
        loo_pol = pool.map(_polish_job, lp_jobs)
    print(f'LOO polish done {time.time() - T_START:.0f}s')

    groups = []
    loo_res = []
    for k, g in enumerate(CODES):
        w = WG.copy()
        w[k] = 0.0
        L_ret_full = objective(theta, w)
        pk = [p for p in loo_pol if p['tag'].startswith(f'loo{k:02d}_')]
        pw = [p for p in pk if p['tag'].endswith('_warm')][0]
        plf = [p for p in pk if not p['tag'].endswith('_warm')][0]
        bestk = pw if pw['f'] <= plf['f'] else plf
        th_k = bestk['x']
        Xk = fitted(th_k)
        med_k = np.median(Xk[members[k]], axis=0)
        Ek = components(th_k)[0][k]
        med_w = np.median(fitted(pw['x'])[members[k]], axis=0)
        med_lf = np.median(fitted(plf['x'])[members[k]], axis=0)
        gm_lat_lon = to_deg(AMED[k, 0], AMED[k, 1])
        rec = dict(code=g, n_sites=int(NG[k]), n_anchor=int(anchor[g]['n']),
                   n_anchor_cells_raw=int(anchor[g]['n_cells']), weight=float(WG[k]),
                   residual_km=gm[k]['residual_km'], env_rms_km=gm[k]['env_rms_km'],
                   loo_residual_km=float(np.hypot(*(med_k - AMED[k]))),
                   loo_env_rms_km=float(np.sqrt(Ek)),
                   loo_converged=bool(bestk['f'] <= L_ret_full + 1e-9),
                   loo_retained_objective=float(bestk['f']),
                   loo_retained_objective_at_fullfit=float(L_ret_full),
                   loo_best_start=bestk['tag'], loo_best_method=bestk['method'],
                   loo_det_sign=det_sign(th_k),
                   loo_leakfree_converged=bool(plf['f'] <= L_ret_full + 1e-9),
                   loo_leakfree_objective=float(plf['f']), loo_warm_objective=float(pw['f']),
                   loo_leakfree_det_sign=det_sign(plf['x']),
                   loo_warm_vs_leakfree_km=float(np.hypot(*(med_w - med_lf))),
                   loo_warm_vs_leakfree_max_site_km=float(np.max(np.hypot(*(fitted(pw['x']) - fitted(plf['x'])).T))),
                   loo_warm_residual_km=float(np.hypot(*(med_w - AMED[k]))),
                   loo_leakfree_residual_km=float(np.hypot(*(med_lf - AMED[k]))),
                   anchor_median_lat=float(gm_lat_lon[1]), anchor_median_lon=float(gm_lat_lon[0]),
                   anchor_median_x_km=float(AMED[k, 0]), anchor_median_y_km=float(AMED[k, 1]),
                   fit_median_lat=float(to_deg(*gm[k]['fit_median_km'])[1]),
                   fit_median_lon=float(to_deg(*gm[k]['fit_median_km'])[0]))
        groups.append(rec)
        loo_res.append(rec['loo_residual_km'])
        print(f"  {g:5d} n={NG[k]:3d} res={rec['residual_km']:.2f} env={rec['env_rms_km']:.2f} | "
              f"LOO res={rec['loo_residual_km']:.2f} env={rec['loo_env_rms_km']:.2f} conv={rec['loo_converged']} "
              f"lf_conv={rec['loo_leakfree_converged']} warm-lf={rec['loo_warm_vs_leakfree_km']:.3f} km "
              f"Lret={bestk['f']:.5f}/{L_ret_full:.5f}")
    loo_res = np.array(loo_res)
    dx, dy = to_km(DOHA_LON, DOHA_LAT)
    doha = [r['code'] for r in groups if np.hypot(r['anchor_median_x_km'] - dx, r['anchor_median_y_km'] - dy) <= DOHA_RADIUS_KM]
    doha_res = [r['loo_residual_km'] for r in groups if r['code'] in doha]
    loo_summary = dict(median=float(np.median(loo_res)), p90=float(np.percentile(loo_res, 90)),
                       max=float(loo_res.max()), mean=float(loo_res.mean()),
                       doha_core_groups=doha, doha_core_median=float(np.median(doha_res)) if doha_res else None,
                       doha_core_insample_median=float(np.median([r['residual_km'] for r in groups if r['code'] in doha])) if doha else None,
                       all_converged=bool(all(r['loo_converged'] for r in groups)),
                       all_leakfree_converged=bool(all(r['loo_leakfree_converged'] for r in groups)),
                       max_warm_vs_leakfree_km=float(max(r['loo_warm_vs_leakfree_km'] for r in groups)))
    print('LOO summary', loo_summary)

    # --- outputs
    out_sites = pd.DataFrame(dict(ANONYMIZED_SITECODE=sites.ANONYMIZED_SITECODE, ANON_LAT=sites.LAT, ANON_LON=sites.LON,
                                  LAT=np.round(lat, 7), LON=np.round(lon, 7), X_KM=np.round(X[:, 0], 4),
                                  Y_KM=np.round(X[:, 1], 4), ON_LAND=on, LAC_TACS=sites.LAC_TACS))
    out_sites.to_csv(f'{OUT}/ds02_sites_deanonymised.csv', index=False)
    off_ids = sites.ANONYMIZED_SITECODE[~on].tolist()
    off_detail = [dict(id=sites.ANONYMIZED_SITECODE[i], lat=float(lat[i]), lon=float(lon[i]),
                       dist_km=float(shapely.distance(LAND, shapely.points(X[i, 0], X[i, 1]))),
                       lac_tacs=str(sites.LAC_TACS[i])) for i in np.where(~on)[0]]
    result = dict(
        B_deg=full['B_deg'], c_deg=full['c_deg'], B_km=full['B_km'], c_km=full['c_km'],
        det_sign=int(best['det']), det_sign_numeric=full['det_sign'], objective=Lstar, objective_terms=full['terms'],
        singular_values_km=full['singular_values_km'], condition_km=full['condition_km'],
        singular_values_deg=full['singular_values_deg'], condition_deg=full['condition_deg'],
        groups=groups, on_land_pct=full['on_land_pct'], n_off_land=int((~on).sum()), off_land_ids=off_ids,
        off_land_detail=off_detail, n_off_land_1km=int((~on1).sum()),
        moments=dict(fit=moments(X), ref=dict(sigma_x=REF['sigma_x'], sigma_y=REF['sigma_y'], rho=REF['rho'])),
        insample_summary=dict(min=full['residual_min_km'], median=full['residual_median_km'], max=full['residual_max_km']),
        loo_summary=loo_summary,
        orientation=orient,
        n_starts=len(starts) + len(ojobs), n_starts_detail=dict(unconstrained_nm=len(starts), median_closed_form=len(mf), random=N_RANDOM,
                                                   polished=len(pjobs), oriented_nm=len(ojobs), oriented_polished=len(opjobs),
                                                   loo_jobs=len(loo_jobs)),
        best_start=dict(tag=best['tag'], method=best['method']),
        polish=[dict(tag=p['tag'], det=p['det'], f_nm=p['f_nm'], f_nm_det=p['f_nm_det'], f=p['f'], method=p['method'],
                     f_by_method=p['f_by_method']) for p in pol],
        polish_oriented=[dict(tag=p['tag'], det=p['det'], f_nm=p['f_nm'], f=p['f'], method=p['method'],
                              f_by_method=p['f_by_method']) for p in opol],
        frames=dict(real=dict(R=R, lat0=LAT0, lon0=LON0), anon=dict(lat_a0=LAT_A0, lon_a0=LON_A0)),
        whitening=dict(mean_km=MU.tolist(), sd_km=SD.tolist(), axes=V.tolist()),
        reference=dict(n_coords=REF['n'], qx=REF['qx'].tolist(), qy=REF['qy'].tolist(),
                       bounds_km=[float(BX0), float(BX1), float(BY0), float(BY1)]),
        land_raster=dict(res_km=RASTER_RES, margin_km=RASTER_MARGIN, shape=[NYR, NXR], **log['raster_validation']),
        objective_eval_us=log['objective_eval_us'],
        runtime_s=time.time() - T_START,
        notes=NOTES)
    with open(f'{OUT}/deanon_fit.json', 'w') as fh:
        json.dump(result, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    print(f'wrote outputs, runtime {time.time() - T_START:.0f}s')

    # --- diagnostic map
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 10))
        lx, ly = LAND.exterior.xy
        for a in ax:
            a.fill(*to_deg(np.array(lx), np.array(ly)), color='#eee', ec='#888', lw=0.8)
            a.set_aspect(1 / np.cos(np.radians(LAT0)))
        ax[0].scatter(ref.lon, ref.lat, s=1, c='#bbb', label='OpenCelliD MNC 2 (distinct)')
        ax[0].scatter(lon[on], lat[on], s=3, c='C0', label='DS02 fitted (on land)')
        ax[0].scatter(lon[~on], lat[~on], s=12, c='C3', label='DS02 fitted (off land)')
        ax[0].legend(loc='upper left', fontsize=8)
        ax[0].set_title(f'DS02 fitted sites, L={Lstar:.3f}, det={full["det_sign"]:+d}')
        for k, g in enumerate(CODES):
            fm = to_deg(*gm[k]['fit_median_km'])
            am = to_deg(*AMED[k])
            ax[1].plot([am[0], fm[0]], [am[1], fm[1]], 'k-', lw=0.7)
            ax[1].plot(*am, 'o', c='C2', ms=4)
            ax[1].plot(*fm, 'x', c='C3', ms=5)
            ax[1].annotate(str(g), am, fontsize=7)
        ax[1].set_title('group medians: anchor (o) vs fit (x)')
        ax[1].set_xlim(51.25, 51.7)
        ax[1].set_ylim(25.05, 25.8)
        fig.tight_layout()
        fig.savefig(f'{OUT}/fit_diagnostic.png', dpi=110)
    except Exception as e:  # pragma: no cover
        print('plot failed', e)


if __name__ == '__main__':
    main()

"""Hold-out validation of the de-anonymisation fit (run after fit.py).

Variants (all refits use ONLY leak-free starts for the scored solution: closed-form median LS
+ Umeyama starts computed without the held-out group(s), and random starts whose translation is
the anchor centroid computed without the held-out group(s); no warm start from the full fit).
A warm start is run separately as a diagnostic only (to check the leak-free search converged).

  A  spec      : drop w_g E_g; global reference/bounds unchanged (exactly the spec's LOO)
  B  strictref : A + held-out area's OpenCelliD cells (any radio) removed from global reference
                 quantiles/sigma/rho and coverage bounds
  C  area      : leave-one-AREA-out for the five paired codes (g and g+1000 dropped together)
  D  areastrict: C + both areas' cells removed from the global reference/bounds

C matters most: the five TAC = LAC + 1000 pairs share 93-100% of their sites, so holding
out one code of a pair while its partner's envelope stays in the fit is close to in-sample.
Writes holdout.json next to this file (about 30 min on 4 cores).
"""
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import holdout_core as C

OUT = Path(__file__).resolve().parent
N_RAND = 24
FIT = json.load(open(OUT / "deanon_fit.json"))
P_FULL = C.p_from_Bdeg(FIT["B_deg"], FIT["c_deg"])
INSAMPLE = {g["code"]: g["residual_km"] for g in FIT["groups"]}
REFS = {}


def get_ref(excl):
    key = tuple(sorted(excl))
    if key not in REFS:
        REFS[key] = C.make_ref(key) if key else C.REF
    return REFS[key]


def job(a):
    vid, tag, p0, drop, excl = a
    w = C.WG.copy()
    w[list(drop)] = 0.0
    ref = get_ref(excl)
    p, f = C.nm_chain(p0, w, ref)
    return vid, tag, p, f


def variants():
    V = []
    for k, g in enumerate(C.GROUPS):
        V.append((f'A:{g}', (k,), ()))
        V.append((f'B:{g}', (k,), (g,)))
    for a, b in C.PAIRS.items():
        ka, kb = C.GROUPS.index(a), C.GROUPS.index(b)
        V.append((f'C:{a}+{b}', (ka, kb), ()))
        V.append((f'D:{a}+{b}', (ka, kb), (a, b)))
    return V


if __name__ == '__main__':
    t0 = time.time()
    jobs = []
    for i, (vid, drop, excl) in enumerate(variants()):
        keep = [k for k in range(C.G) if k not in drop]
        rng = np.random.default_rng(1000 + i)
        for j, s in enumerate(C.closed_form(keep)):
            jobs.append((vid, f'cf{j}', s, drop, excl))
        for j, s in enumerate(C.random_starts(N_RAND, rng, keep)):
            jobs.append((vid, f'rnd{j}', s, drop, excl))
        jobs.append((vid, 'WARM_DIAGNOSTIC', P_FULL.copy(), drop, excl))
    print(f'{len(jobs)} jobs', flush=True)
    with Pool(4) as pool:
        res = pool.map(job, jobs, chunksize=1)
    print(f'done in {time.time()-t0:.0f}s', flush=True)

    out = {}
    for vid, drop, excl in variants():
        rs = [r for r in res if r[0] == vid]
        lf = sorted([r for r in rs if r[1] != 'WARM_DIAGNOSTIC'], key=lambda r: r[3])
        warm = [r for r in rs if r[1] == 'WARM_DIAGNOSTIC'][0]
        best = lf[0]
        w = C.WG.copy()
        w[list(drop)] = 0.0
        ref = get_ref(excl)
        f_at_full = C.objective(P_FULL, w, ref)
        res_k = C.residuals(best[2])
        env_k = C.env_rms(best[2], ref)
        res_w = C.residuals(warm[2])
        xb, yb = C.xy(best[2])
        xw, yw = C.xy(warm[2])
        n_reach = sum(1 for r in lf if r[3] <= best[3] + 1e-7)
        rec = dict(drop=[C.GROUPS[k] for k in drop], excl_ref=list(excl),
                   leakfree_L=best[3], leakfree_tag=best[1], warm_L=warm[3], L_at_full=f_at_full,
                   leakfree_converged=bool(best[3] <= f_at_full + 1e-9),
                   leakfree_minus_warm=best[3] - warm[3],
                   warm_vs_leakfree_max_site_km=float(np.max(np.hypot(xb - xw, yb - yw))),
                   n_leakfree_reaching_best=n_reach, n_leakfree=len(lf), det=C.det_sign_B(best[2]),
                   heldout={})
        for k in drop:
            g = C.GROUPS[k]
            rec['heldout'][g] = dict(resid=float(res_k[k]), env_rms=float(env_k[k]), resid_warm=float(res_w[k]),
                                     insample=INSAMPLE[g])
        out[vid] = rec
        ho = '; '.join(f"{g}: resid {h['resid']:.3f} (warm {h['resid_warm']:.3f}, in-sample {h['insample']:.3f}) env {h['env_rms']:.3f}"
                       for g, h in rec['heldout'].items())
        print(f"{vid:>12s} Llf={best[3]:.7f} Lwarm={warm[3]:.7f} Lfull={f_at_full:.7f} conv={rec['leakfree_converged']} "
              f"lf-warm={rec['leakfree_minus_warm']:+.2e} disp={rec['warm_vs_leakfree_max_site_km']:.3f}km reach={n_reach}/{len(lf)} det={rec['det']:+d} | {ho}", flush=True)

    # summaries
    def summ(vals):
        v = np.array(vals)
        return dict(median=float(np.median(v)), p90=float(np.percentile(v, 90)), max=float(v.max()))
    dx, dy = C.KX * (C.DOHA[1] - C.LON0), C.KY * (C.DOHA[0] - C.LAT0)
    core = [C.GROUPS[k] for k in range(C.G) if np.hypot(C.AMED[k, 0] - dx, C.AMED[k, 1] - dy) <= 8]
    paired = set(C.PAIRS) | set(C.PAIRS.values())
    A = {g: out[f'A:{g}']['heldout'][g]['resid'] for g in C.GROUPS}
    B = {g: out[f'B:{g}']['heldout'][g]['resid'] for g in C.GROUPS}
    Carea, Darea = dict(A), dict(B)
    for a, b in C.PAIRS.items():
        for g in (a, b):
            Carea[g] = out[f'C:{a}+{b}']['heldout'][g]['resid']
            Darea[g] = out[f'D:{a}+{b}']['heldout'][g]['resid']
    S = {}
    for name, dct in (('A_spec_leakfree', A), ('B_strict_ref', B), ('C_leave_area_out', Carea), ('D_leave_area_out_strict_ref', Darea)):
        S[name] = summ(list(dct.values()))
        S[name]['doha_core_median'] = float(np.median([dct[g] for g in core]))
        S[name]['paired_median'] = float(np.median([dct[g] for g in paired]))
        S[name]['unpaired_median'] = float(np.median([dct[g] for g in C.GROUPS if g not in paired]))
        S[name]['per_group'] = dct
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in S[name].items() if k != 'per_group'})
    print('spec LOO in deanon_fit.json vs leak-free A, max |diff| km:',
          max(abs(A[g] - next(x['loo_residual_km'] for x in FIT['groups'] if x['code'] == g)) for g in C.GROUPS))
    for ex in sorted(REFS):
        r = REFS[ex]
        print('ref excl', ex, 'n', r['n'], 'sx %.3f sy %.3f rho %.4f' % (r['sx'], r['sy'], r['rho']),
              'bounds', np.round(r['bounds'], 2).tolist())
    json.dump(dict(variants=out, summary=S, doha_core=core), open(OUT / 'holdout.json', 'w'), indent=1, default=float)

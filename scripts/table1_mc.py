"""E7: the consolidated run behind Table 1 of the Letter (frame: data-driven exponent at fixed
dimension). Every arm on the same samples; one Newton step from the bisquare M-estimate, MAD scale.

Usage:
  table1_mc.py run <out_dir> <R list> <N list> [law index list] [max N whose data are saved for asm]
  table1_mc.py collect <out_dir>

Arms (slope MSE / OLS slope MSE):
  M        bisquare M-estimate
  a=...    fractional basis sign(tau)|tau|^{p_i(alpha)}, i = 1..3, plus tau^2, at each fixed alpha
           of the grid; b by symmetric difference, delta = N^(-1/5)
  adapt    the same basis with alpha chosen in-sample by the estimated asymptotic variance
  NwT6     Newey (1988) transformed powers (u/(1+|u|))^j, J = 6
  NwW6     Newey (1988) weighted powers exp(-u^2/2) u^j, J = 6
  asm      R package asm (symmetric = FALSE), run by baselines_asm.R on the saved samples
The four-function fractional basis is compared with six-function Newey bases on purpose: these
were the strongest Newey arms in E4 and E4b. Seed: [20261120, law index, n].
"""
import glob
import math
import os
import sys

import numpy as np

from baselines_mc import (GRID, P, SPECS_PATP, bimodal, contaminate, gamma4, gauss, irls, mad,
                          tau_columns, w_bi, w_hub, weights)
from newey_weighted_mc import transformed, weighted


def gen_gauss(beta):
    scale = math.sqrt(math.gamma(3.0 / beta) / math.gamma(1.0 / beta))

    def f(n, rng):
        return rng.gamma(1.0 / beta, 1.0, n) ** (1.0 / beta) * rng.choice([-1.0, 1.0], n) / scale
    return f


LAWS = [
    ("Гаус (контроль)", gauss),
    ("Лаплас", lambda n, rng: rng.laplace(0, 1 / math.sqrt(2), n)),
    ("GG(0.7)", gen_gauss(0.7)),
    ("t3", lambda n, rng: rng.standard_t(3, n) / math.sqrt(3.0)),
    ("бімодальний", bimodal),
    ("Gamma(4)", gamma4),
    ("Гаус + 5% ±10", contaminate(gauss)),
    ("Gamma(4) + 5% ±10", contaminate(gamma4)),
    ("бімодальний + 5% ±10", contaminate(bimodal)),
]
COLS = ("OLS", "M") + tuple(f"a={a}" for a in GRID) + ("adapt", "alpha", "NwT6", "NwW6")
K0 = 2  # index of the first fixed-alpha column


def one_rep(X, y):
    n = len(y)
    XtX_inv = np.linalg.inv(X.T @ X)
    ols = XtX_inv @ (X.T @ y)
    base = irls(X, y, irls(X, y, ols, w_hub), w_bi)
    r = y - X @ base
    s = mad(r)
    u = r / s
    delta = n ** -0.2

    def step(psi, slope):
        return base + s * XtX_inv @ (X.T @ psi) / slope if slope > 1e-10 else base

    def sse(beta):
        return float(np.sum(beta[1:] ** 2))

    out = [sse(ols), sse(base)]
    crit = []
    for sp in SPECS_PATP:
        psi, slope = weights(*tau_columns(u, sp, delta))
        out.append(sse(step(psi, slope)))
        crit.append(1.0 / slope if slope > 1e-10 else np.inf)
    k = int(np.argmin(crit))
    out += [out[K0 + k], GRID[k]]
    for basis in (transformed(u, 6), weighted(u, 6)):
        out.append(sse(step(*weights(*basis))))
    return out


def run(out_dir, Rs, ns, law_ids, save_max_n):
    for li in law_ids:
        name, law = LAWS[li]
        for n, R in zip(ns, Rs):
            rng = np.random.default_rng([20261120, li, n])
            data = np.empty((R, n, P), dtype=np.float32) if n <= save_max_n else None
            res = np.empty((R, len(COLS)))
            for k in range(R):
                x = rng.standard_normal((n, P - 1)).astype(np.float32).astype(float)
                e = law(n, rng).astype(np.float32).astype(float)
                if data is not None:
                    data[k, :, : P - 1] = x
                    data[k, :, P - 1] = e
                res[k] = one_rep(np.column_stack([np.ones(n), x]), e)
            if data is not None:
                data.tofile(os.path.join(out_dir, f"data_{li}_{n}_{R}.bin"))
            np.save(os.path.join(out_dir, f"t1_{li}_{n}_{R}.npy"), res)
            print(name, n, "done", flush=True)


def ratio_se(a, b):
    ratio = a.mean() / b.mean()
    return ratio, np.std((a - ratio * b) / b.mean(), ddof=1) / math.sqrt(len(a))


def collect(out_dir):
    cells = {}
    for path in glob.glob(os.path.join(out_dir, "t1_*.npy")):
        li, n, R = (int(v) for v in os.path.basename(path)[3:-4].split("_"))
        res = np.load(path)
        apath = os.path.join(out_dir, f"asm_{li}_{n}_{R}.txt")
        asm = np.genfromtxt(apath, skip_header=1)[:, 0] if os.path.exists(apath) else None
        cells[(n, li)] = (R, res, asm)
    fixed = slice(K0, K0 + len(GRID))
    print("E7: slope MSE / OLS slope MSE, same samples for every arm; tau = bisquare; one Newton step")
    print("best = smallest fixed-alpha value (alpha in brackets); a=1 = integer exponents; "
          "adapt = data-driven alpha (mean selected alpha in brackets)")
    for n in sorted({k[0] for k in cells}):
        print(f"\n=== N = {n} ===")
        print(f"{'закон':22s}{'R':>6s}{'M':>8s}{'best':>14s}{'a=1':>8s}{'adapt':>14s}"
              f"{'NwT6':>8s}{'NwW6':>8s}{'asm':>8s}")
        for (nn, li), (R, res, asm) in sorted(cells.items()):
            if nn != n:
                continue
            m = res.mean(0)
            rel = m / m[0]
            kb = int(np.argmin(m[fixed]))
            line = (f"{LAWS[li][0]:22s}{R:6d}{rel[1]:8.3f}{rel[K0 + kb]:8.3f} [{GRID[kb]:5.3f}]"
                    f"{rel[K0 + len(GRID) - 1]:8.3f}{rel[COLS.index('adapt')]:8.3f} "
                    f"[{m[COLS.index('alpha')]:4.2f}]{rel[COLS.index('NwT6')]:8.3f}"
                    f"{rel[COLS.index('NwW6')]:8.3f}")
            if asm is not None:
                ok = ~np.isnan(asm)
                line += f"{asm[ok].mean() / res[ok, 0].mean():8.3f}"
            else:
                line += f"{'-':>8s}"
            print(line)
    print("\nAll fixed alpha (slope MSE / OLS):")
    for n in sorted({k[0] for k in cells}):
        print(f"\n=== N = {n} ===")
        print(f"{'закон':22s}" + "".join(f"{a:>8.3f}" for a in GRID))
        for (nn, li), (R, res, asm) in sorted(cells.items()):
            if nn == n:
                m = res.mean(0)
                print(f"{LAWS[li][0]:22s}" + "".join(f"{v:8.3f}" for v in m[fixed] / m[0]))
    print("\nPaired ratios of slope MSE, adaptive / other arm, with delta-method standard errors:")
    for n in sorted({k[0] for k in cells}):
        print(f"\n=== N = {n} ===")
        print(f"{'закон':22s}{'/ best fixed':>16s}{'/ a=1':>16s}{'/ NwT6':>16s}{'/ NwW6':>16s}{'/ asm':>16s}")
        for (nn, li), (R, res, asm) in sorted(cells.items()):
            if nn != n:
                continue
            ad = res[:, COLS.index("adapt")]
            kb = int(np.argmin(res.mean(0)[fixed]))
            others = [res[:, K0 + kb], res[:, K0 + len(GRID) - 1], res[:, COLS.index("NwT6")],
                      res[:, COLS.index("NwW6")]]
            line = f"{LAWS[li][0]:22s}" + "".join("{:9.3f} ({:.3f})".format(*ratio_se(ad, o)) for o in others)
            if asm is not None:
                ok = ~np.isnan(asm)
                line += "{:9.3f} ({:.3f})".format(*ratio_se(ad[ok], asm[ok]))
            print(line)


if __name__ == "__main__":
    if sys.argv[1] == "run":
        ns = [int(v) for v in sys.argv[4].split(",")]
        ids = [int(v) for v in sys.argv[5].split(",")] if len(sys.argv) > 5 and sys.argv[5] else range(len(LAWS))
        run(sys.argv[2], [int(v) for v in sys.argv[3].split(",")], ns, ids,
            int(sys.argv[6]) if len(sys.argv) > 6 else 2000)
    else:
        collect(sys.argv[2])

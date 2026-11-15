"""Finite-sample check of PMM on powers of a robust generating element tau.

y = X beta + e, X = [1, N(0,1)^4], beta = 0. For tau in {Huber, bisquare, Welsch} at the
95%-efficiency constant: S = 1 is the classical M-estimator (IRLS, MAD scale); S = 2, 3
take three Newton steps from it with psi = h'(tau, .., tau^S), h = F^{-1} b re-estimated
from the current residuals. Metric: slope MSE / OLS slope MSE, 95% percentile bootstrap
over replicates; paired ratio S=3 / S=1 per generating element.
"""
import math
import sys
import time

import numpy as np

R = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
NS = (500, 2000)
P = 5
SD_BI = math.sqrt(1 + 1.5 ** 2)


def gamma4(n, rng):
    return (rng.gamma(4.0, 1.0, n) - 4.0) / 2.0


def bimodal(n, rng):
    return (rng.choice([-1.5, 1.5], n) + rng.standard_normal(n)) / SD_BI


def gauss(n, rng):
    return rng.standard_normal(n)


def contaminate(base, frac=0.05, loc=10.0, two_sided=True):
    def f(n, rng):
        e = base(n, rng)
        bad = rng.random(n) < frac
        k = int(bad.sum())
        centre = rng.choice([-loc, loc], k) if two_sided else np.full(k, loc)
        e[bad] = centre + rng.standard_normal(k)
        return e
    return f


LAWS = {
    "Гаус (контроль)": gauss,
    "Лаплас": lambda n, rng: rng.laplace(0, 1 / math.sqrt(2), n),
    "Gamma(4)": gamma4,
    "Гаус + 5% ±10": contaminate(gauss),
    "Гаус + 5% +10": contaminate(gauss, two_sided=False),
    "Gamma(4) + 5% ±10": contaminate(gamma4),
    "бімодальний + 5% ±10": contaminate(bimodal),
}

# generating elements: tau(u), tau'(u), IRLS weight tau(u)/u
C_H, C_B, C_W = 1.345, 4.685, 2.985
TAU = {
    "Huber": (lambda u: np.clip(u, -C_H, C_H),
              lambda u: (np.abs(u) < C_H).astype(float),
              lambda u: np.minimum(1.0, C_H / np.maximum(np.abs(u), 1e-12))),
    "bisquare": (lambda u: np.where(np.abs(u) < C_B, u * (1 - (u / C_B) ** 2) ** 2, 0.0),
                 lambda u: np.where(np.abs(u) < C_B, (1 - (u / C_B) ** 2) * (1 - 5 * (u / C_B) ** 2), 0.0),
                 lambda u: np.where(np.abs(u) < C_B, (1 - (u / C_B) ** 2) ** 2, 0.0)),
    "Welsch": (lambda u: u * np.exp(-(u / C_W) ** 2),
               lambda u: (1 - 2 * (u / C_W) ** 2) * np.exp(-(u / C_W) ** 2),
               lambda u: np.exp(-(u / C_W) ** 2)),
}


def mad(r):
    return 1.4826 * np.median(np.abs(r - np.median(r)))


def irls(X, y, beta, weight_fn, iters=50):
    s = mad(y - X @ beta)
    for _ in range(iters):
        w = weight_fn((y - X @ beta) / s)
        Xw = X * w[:, None]
        new = np.linalg.solve(X.T @ Xw, Xw.T @ y)
        if np.max(np.abs(new - beta)) < 1e-8:
            return new
        beta = new
    return beta


def tau_pmm(X, y, beta, tau, dtau, S, XtX_inv, steps=3):
    for _ in range(steps):
        r = y - X @ beta
        s = mad(r)
        u = r / s
        t, d = tau(u), dtau(u)
        Phi = np.column_stack([t ** i for i in range(1, S + 1)])
        dPhi = np.column_stack([i * t ** (i - 1) * d for i in range(1, S + 1)])
        Phi_c = Phi - Phi.mean(0)
        F = Phi_c.T @ Phi_c / len(u)
        b = dPhi.mean(0)
        h = np.linalg.solve(F + 1e-10 * np.trace(F) * np.eye(S), b)
        slope = float(h @ b)
        if slope <= 1e-10:
            break
        beta = beta + s * XtX_inv @ (X.T @ (Phi_c @ h)) / slope
    return beta


EST = ["OLS"] + [f"{k}-{S}" for k in TAU for S in (1, 2, 3)]


def one_rep(law, n, rng):
    X = np.column_stack([np.ones(n), rng.standard_normal((n, P - 1))])
    y = law(n, rng)
    XtX_inv = np.linalg.inv(X.T @ X)
    ols = XtX_inv @ (X.T @ y)
    out = {"OLS": ols}
    hub = irls(X, y, ols, TAU["Huber"][2])
    for k, (tau, dtau, w) in TAU.items():
        base = hub if k == "Huber" else irls(X, y, hub, w)
        out[f"{k}-1"] = base
        for S in (2, 3):
            out[f"{k}-{S}"] = tau_pmm(X, y, base, tau, dtau, S, XtX_inv)
    return {k: float(np.sum(v[1:] ** 2)) for k, v in out.items()}


def boot(a, b, rng, B=2000):
    idx = rng.integers(0, len(a), (B, len(a)))
    return np.percentile(a[idx].mean(1) / b[idx].mean(1), [2.5, 97.5])


rng = np.random.default_rng(20261113)
print(f"R = {R}; slope MSE / OLS slope MSE (half-width of the 95% bootstrap interval)")
for n in NS:
    print(f"\n=== N = {n} ===")
    print(f"{'закон':22s}" + "".join(f"{e:>14s}" for e in EST[1:]))
    paired = []
    for name, law in LAWS.items():
        res = {e: np.empty(R) for e in EST}
        for r in range(R):
            d = one_rep(law, n, rng)
            for e in EST:
                res[e][r] = d[e]
        cells = []
        for e in EST[1:]:
            lo, hi = boot(res[e], res["OLS"], rng)
            cells.append(f"{res[e].mean() / res['OLS'].mean():.3f}±{(hi - lo) / 2:.3f}")
        print(f"{name:22s}" + "".join(f"{c:>14s}" for c in cells))
        row = []
        for k in TAU:
            lo, hi = boot(res[f"{k}-3"], res[f"{k}-1"], rng)
            row.append(f"{k}: {res[f'{k}-3'].mean() / res[f'{k}-1'].mean():.3f} [{lo:.3f},{hi:.3f}]")
        paired.append(f"{name:22s}" + "   ".join(row))
        sys.stdout.flush()
    print("\n  paired ratio S=3 / S=1 (same generating element):")
    for p in paired:
        print("  " + p)

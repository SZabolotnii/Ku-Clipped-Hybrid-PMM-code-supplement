"""Finite-sample check of PATP / mixed columns on a robust generating element tau.

Same design as generating_elements_mc.py. Columns are (p, signed): sign(tau)|tau|^p if
signed else |tau|^p. For tau in {Huber, bisquare}:
  base   the M-estimator (S = 1)
  PMM3   {tau, tau^2, tau^3}
  PATP3  {tau, sgn|tau|^(1/2), sgn|tau|^(1/3)}          (PATP, alpha = 0)
  mix4   {tau, tau^2, tau^3, sgn|tau|^(1/2)}
Three Newton steps from the base estimate, h = F^{-1} b re-estimated each step.
"""
import math
import sys

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


def contaminate(base, frac=0.05, loc=10.0):
    def f(n, rng):
        e = base(n, rng)
        bad = rng.random(n) < frac
        k = int(bad.sum())
        e[bad] = rng.choice([-loc, loc], k) + rng.standard_normal(k)
        return e
    return f


LAWS = {
    "Гаус (контроль)": gauss,
    "Лаплас": lambda n, rng: rng.laplace(0, 1 / math.sqrt(2), n),
    "бімодальний": bimodal,
    "Gamma(4)": gamma4,
    "Гаус + 5% ±10": contaminate(gauss),
    "Gamma(4) + 5% ±10": contaminate(gamma4),
    "бімодальний + 5% ±10": contaminate(bimodal),
}

C_H, C_B = 1.345, 4.685
TAU = {
    "Huber": (lambda u: np.clip(u, -C_H, C_H),
              lambda u: (np.abs(u) < C_H).astype(float),
              lambda u: np.minimum(1.0, C_H / np.maximum(np.abs(u), 1e-12))),
    "bisquare": (lambda u: np.where(np.abs(u) < C_B, u * (1 - (u / C_B) ** 2) ** 2, 0.0),
                 lambda u: np.where(np.abs(u) < C_B, (1 - (u / C_B) ** 2) * (1 - 5 * (u / C_B) ** 2), 0.0),
                 lambda u: np.where(np.abs(u) < C_B, (1 - (u / C_B) ** 2) ** 2, 0.0)),
}
def p_i(i, alpha):
    return 1 / i + (4 - i - 3 / i) * alpha + (2 * i - 4 + 2 / i) * alpha ** 2


# argv[2] = "v2": PATP at alpha = 0.75 (exponents 1, 1.4375, 1.8333: no singular derivative)
# alone and with the even column tau^2. Default: the first run (alpha = 0 and mix4).
if len(sys.argv) > 2 and sys.argv[2] == "v2":
    _a = [(p_i(i, 0.75), True) for i in (1, 2, 3)]
    BASES = {"PMM3": [(1, True), (2, False), (3, True)], "PATP3a.75": _a, "PATP.75+ev": _a + [(2, False)]}
else:
    BASES = {
        "PMM3": [(1, True), (2, False), (3, True)],
        "PATP3": [(1, True), (0.5, True), (1 / 3, True)],
        "mix4": [(1, True), (2, False), (3, True), (0.5, True)],
    }
N_STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 3


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


def columns(t, d, specs):
    at = np.abs(t)
    sg = np.sign(t)
    ok = at > 1e-12
    safe = np.where(ok, at, 1.0)
    Phi, dPhi = [], []
    for p, signed in specs:
        Phi.append(np.where(ok, (sg if signed else 1.0) * safe ** p, 0.0))
        dPhi.append(np.where(ok, p * safe ** (p - 1) * (1.0 if signed else sg) * d, 0.0))
    return np.column_stack(Phi), np.column_stack(dPhi)


def steps(X, y, beta, tau, dtau, specs, XtX_inv, n_steps=3):
    k = len(specs)
    for _ in range(n_steps):
        r = y - X @ beta
        s = mad(r)
        u = r / s
        Phi, dPhi = columns(tau(u), dtau(u), specs)
        Phi_c = Phi - Phi.mean(0)
        F = Phi_c.T @ Phi_c / len(u)
        b = dPhi.mean(0)
        h = np.linalg.solve(F + 1e-10 * np.trace(F) * np.eye(k), b)
        slope = float(h @ b)
        if slope <= 1e-10:
            break
        beta = beta + s * XtX_inv @ (X.T @ (Phi_c @ h)) / slope
    return beta


EST = ["OLS"] + [f"{k}-{b}" for k in TAU for b in ("base", *BASES)]


def one_rep(law, n, rng):
    X = np.column_stack([np.ones(n), rng.standard_normal((n, P - 1))])
    y = law(n, rng)
    XtX_inv = np.linalg.inv(X.T @ X)
    ols = XtX_inv @ (X.T @ y)
    out = {"OLS": ols}
    hub = irls(X, y, ols, TAU["Huber"][2])
    for k, (tau, dtau, w) in TAU.items():
        base = hub if k == "Huber" else irls(X, y, hub, w)
        out[f"{k}-base"] = base
        for name, specs in BASES.items():
            out[f"{k}-{name}"] = steps(X, y, base, tau, dtau, specs, XtX_inv, N_STEPS)
    return {k: float(np.sum(v[1:] ** 2)) for k, v in out.items()}


def boot(a, b, rng, B=2000):
    idx = rng.integers(0, len(a), (B, len(a)))
    return np.percentile(a[idx].mean(1) / b[idx].mean(1), [2.5, 97.5])


rng = np.random.default_rng(20261114)
print(f"R = {R}; Newton steps = {N_STEPS}; slope MSE / OLS slope MSE (half-width of the 95% bootstrap interval)")
for n in NS:
    print(f"\n=== N = {n} ===")
    print(f"{'закон':22s}" + "".join(f"{e:>15s}" for e in EST[1:]))
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
        print(f"{name:22s}" + "".join(f"{c:>15s}" for c in cells))
        sys.stdout.flush()

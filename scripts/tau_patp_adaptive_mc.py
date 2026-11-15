"""Data-driven alpha for PATP on a robust generating element (tau: argv[4], default bisquare).

argv: R, DELTA_C, law filter (comma list of prefixes, "" = all), tau, N list, R list.

Basis B(alpha) = {sign(tau)|tau|^{p_i(alpha)}, i = 1..3} + {tau^2}; alpha on a grid without
the degenerate midpoint 1/2. Residuals come from the bisquare M-estimate; one Newton step.
Selection rules for alpha (criterion = estimated asymptotic variance of the slopes):
  in     argmin of  1 / (b'F^{-1}b)  on the full sample (in-sample, optimistic);
  cf     2-fold cross-fit: h from one half, variance h'Fh / (h'b)^2 on the other half;
  *+     the same rules restricted to alpha >= 0.625 (no exponent below 1/2).
Reported: slope MSE / OLS slope MSE for every fixed alpha (so the oracle fixed alpha is
visible), for PMM-3, and for the four adaptive rules; mean selected alpha.
"""
import math
import sys

import numpy as np

R = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
DELTA_C = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0     # b by symmetric difference, delta = DELTA_C n^(-1/5)
ONLY = sys.argv[3].split(",") if len(sys.argv) > 3 else None   # optional law filter (prefix match)
TAU_NAME = sys.argv[4] if len(sys.argv) > 4 else "bisquare"
NS = tuple(int(v) for v in sys.argv[5].split(",")) if len(sys.argv) > 5 else (500, 2000)
RS = tuple(int(v) for v in sys.argv[6].split(",")) if len(sys.argv) > 6 else (R,) * len(NS)
P = 5
SD_BI = math.sqrt(1 + 1.5 ** 2)
GRID = (0.0, 0.125, 0.25, 0.375, 0.625, 0.75, 0.875, 1.0)
HI = [k for k, a in enumerate(GRID) if a >= 0.6]


def p_i(i, alpha):
    return 1 / i + (4 - i - 3 / i) * alpha + (2 * i - 4 + 2 / i) * alpha ** 2


def gamma4(n, rng):
    return (rng.gamma(4.0, 1.0, n) - 4.0) / 2.0


def bimodal(n, rng):
    return (rng.choice([-1.5, 1.5], n) + rng.standard_normal(n)) / SD_BI


def gen_gauss(n, rng, beta):
    """Generalised Gaussian, density ~ exp(-|x|^beta), standardised; beta < 1 is sharper than Laplace."""
    x = rng.gamma(1.0 / beta, 1.0, n) ** (1.0 / beta) * rng.choice([-1.0, 1.0], n)
    return x / math.sqrt(math.gamma(3.0 / beta) / math.gamma(1.0 / beta))


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
    "t3": lambda n, rng: rng.standard_t(3, n) / math.sqrt(3.0),
    "GG(0.7)": lambda n, rng: gen_gauss(n, rng, 0.7),
    "бімодальний": bimodal,
    "Gamma(4)": gamma4,
    "Гаус + 5% ±10": contaminate(gauss),
    "Gamma(4) + 5% ±10": contaminate(gamma4),
    "бімодальний + 5% ±10": contaminate(bimodal),
}

C_H, C_B, C_W = 1.345, 4.685, 2.985
w_hub = lambda u: np.minimum(1.0, C_H / np.maximum(np.abs(u), 1e-12))
TAUS = {   # tau, tau', IRLS weight
    "Huber": (lambda u: np.clip(u, -C_H, C_H), lambda u: (np.abs(u) < C_H).astype(float), w_hub),
    "bisquare": (lambda u: np.where(np.abs(u) < C_B, u * (1 - (u / C_B) ** 2) ** 2, 0.0),
                 lambda u: np.where(np.abs(u) < C_B, (1 - (u / C_B) ** 2) * (1 - 5 * (u / C_B) ** 2), 0.0),
                 lambda u: np.where(np.abs(u) < C_B, (1 - (u / C_B) ** 2) ** 2, 0.0)),
    "Welsch": (lambda u: u * np.exp(-(u / C_W) ** 2),
               lambda u: (1 - 2 * (u / C_W) ** 2) * np.exp(-(u / C_W) ** 2),
               lambda u: np.exp(-(u / C_W) ** 2)),
}
tau, dtau, w_tau = TAUS[TAU_NAME]


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


def phi_of(u, specs):
    t = tau(u)
    at = np.abs(t)
    sg = np.sign(t)
    return np.column_stack([(sg if s else 1.0) * at ** p for p, s in specs])


def columns(u, specs, delta):
    """Phi and the per-observation rows whose mean is b.

    delta = 0: derivative form phi'(u). For an exponent p < 1 its variance is infinite
    (E|u|^{2p-2} diverges at zero) and the estimate inflates the apparent gain.
    delta > 0: symmetric difference (phi(u + delta) - phi(u - delta)) / (2 delta), i.e. the
    derivative of the location-smoothed mean; finite variance for every p > 0.
    """
    Phi = phi_of(u, specs)
    if delta > 0:
        return Phi, (phi_of(u + delta, specs) - phi_of(u - delta, specs)) / (2 * delta)
    t, d = tau(u), dtau(u)
    at = np.abs(t)
    sg = np.sign(t)
    ok = at > 1e-12
    safe = np.where(ok, at, 1.0)
    dPhi = [np.where(ok, p * safe ** (p - 1) * (1.0 if s else sg) * d, 0.0) for p, s in specs]
    return Phi, np.column_stack(dPhi)


def fb(Phi, dPhi):
    Pc = Phi - Phi.mean(0)
    return Pc.T @ Pc / len(Phi), dPhi.mean(0), Pc


def solve(F, b):
    return np.linalg.solve(F + 1e-10 * np.trace(F) * np.eye(len(b)), b)


def specs_patp(alpha):
    return [(p_i(i, alpha), True) for i in (1, 2, 3)] + [(2, False)]


SPECS = [specs_patp(a) for a in GRID]
PMM3 = [(1, True), (2, False), (3, True)]


def one_rep(law, n, rng):
    X = np.column_stack([np.ones(n), rng.standard_normal((n, P - 1))])
    y = law(n, rng)
    XtX_inv = np.linalg.inv(X.T @ X)
    ols = XtX_inv @ (X.T @ y)
    base = irls(X, y, ols, w_hub)
    if TAU_NAME != "Huber":
        base = irls(X, y, base, w_tau)
    r = y - X @ base
    s = mad(r)
    u = r / s
    delta = DELTA_C * n ** -0.2
    perm = rng.permutation(n)
    A, B = perm[: n // 2], perm[n // 2:]

    def step(specs):
        Phi, dPhi = columns(u, specs, delta)
        F, b, Pc = fb(Phi, dPhi)
        h = solve(F, b)
        slope = float(h @ b)
        v_in = 1.0 / slope if slope > 1e-10 else np.inf          # = 1 / (b'F^{-1}b)
        v_cf = 0.0
        for I, J in ((A, B), (B, A)):
            FI, bI, _ = fb(Phi[I], dPhi[I])
            hI = solve(FI, bI)
            FJ, bJ, _ = fb(Phi[J], dPhi[J])
            den = float(hI @ bJ)
            v_cf += float(hI @ FJ @ hI) / den ** 2 if den > 1e-10 else np.inf
        beta = base + s * XtX_inv @ (X.T @ (Pc @ h)) / slope if slope > 1e-10 else base
        return float(np.sum(beta[1:] ** 2)), v_in, v_cf / 2

    fixed = [step(sp) for sp in SPECS]
    mse = np.array([f[0] for f in fixed])
    vin = np.array([f[1] for f in fixed])
    vcf = np.array([f[2] for f in fixed])
    pick = {"in": int(np.argmin(vin)), "cf": int(np.argmin(vcf)),
            "in+": HI[int(np.argmin(vin[HI]))], "cf+": HI[int(np.argmin(vcf[HI]))]}
    return (float(np.sum(ols[1:] ** 2)), float(np.sum(base[1:] ** 2)), step(PMM3)[0], mse,
            {k: (mse[v], GRID[v]) for k, v in pick.items()})


rng = np.random.default_rng(20261115)
RULES = ("in", "cf", "in+", "cf+")
if ONLY:
    LAWS = {k: v for k, v in LAWS.items() if any(k.startswith(o) for o in ONLY)}
print(f"R per N = {dict(zip(NS, RS))}; tau = {TAU_NAME}; one Newton step; b: " +
      (f"symmetric difference, delta = {DELTA_C} n^(-1/5)" if DELTA_C > 0 else "derivative form") +
      "; slope MSE / OLS slope MSE")
for n, R in zip(NS, RS):
    print(f"\n=== N = {n} ===")
    print(f"{'закон':22s}{'база':>7s}{'PMM3':>7s} |" + "".join(f"{'a=' + str(a):>8s}" for a in GRID) + " |" +
          "".join(f"{k:>7s}" for k in RULES) + " |  mean alpha: " + " ".join(RULES))
    for name, law in LAWS.items():
        o, b, p3 = np.empty(R), np.empty(R), np.empty(R)
        fx = np.empty((R, len(GRID)))
        ad = {k: np.empty(R) for k in RULES}
        al = {k: np.empty(R) for k in RULES}
        for r in range(R):
            o[r], b[r], p3[r], fx[r], pk = one_rep(law, n, rng)
            for k in RULES:
                ad[k][r], al[k][r] = pk[k]
        m = o.mean()
        print(f"{name:22s}{b.mean() / m:7.3f}{p3.mean() / m:7.3f} |" +
              "".join(f"{v:8.3f}" for v in fx.mean(0) / m) + " |" +
              "".join(f"{ad[k].mean() / m:7.3f}" for k in RULES) + " |  " +
              " ".join(f"{al[k].mean():.2f}" for k in RULES))
        sys.stdout.flush()

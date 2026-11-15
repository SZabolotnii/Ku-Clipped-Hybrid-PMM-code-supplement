"""E4b: tau-PMM-3 against both moment families of Newey (1988), read from the original.

Usage:
  newey_weighted_mc.py run <out_dir> <R> <N list> [law index list]
  newey_weighted_mc.py collect <out_dir>

Newey's sampling experiments use two families (his Section 4):
  transformed   m_j(e) = (e / (1 + |e|))^j
  weighted      m_j(e) = exp(-e^2 / 2) e^j        (bounded and redescending for every j)
with the OLS estimate as pilot and the mean and standard deviation of the OLS residuals as
location and scale. E4 (baselines_mc.py) had only the transformed family, on a robust pilot.

Arms (slope MSE / OLS slope MSE):
  M            bisquare M-estimate
  tPMM3        powers of the bisquare psi, S = 3, bisquare pilot, MAD scale
  NwT3, NwT6   transformed family, J = 3 and 6, bisquare pilot, MAD scale   (= Newey3/6 of E4)
  NwW3, NwW6   weighted family,    J = 3 and 6, bisquare pilot, MAD scale
  NwT3o, NwW3o transformed / weighted, J = 3, as in Newey's experiments: OLS pilot, SD scale
All arms take one Newton step with weights F^{-1} b, which is Newey's linearised GMM estimator
(newey_identity_check.py). Laws and contamination as in baselines_mc.py. Seed: [20261118, law, n].
"""
import glob
import math
import os
import sys

import numpy as np

from baselines_mc import C_B, LAWS, P, irls, mad, tau, w_bi, w_hub

ARMS = ("OLS", "M", "tPMM3", "NwT3", "NwT6", "NwW3", "NwW6", "NwT3o", "NwW3o")
PAIRS = (("tPMM3", "NwT3"), ("tPMM3", "NwT6"), ("tPMM3", "NwW3"), ("tPMM3", "NwW6"),
         ("tPMM3", "NwT3o"), ("tPMM3", "NwW3o"))


def tau_powers(u, S=3):
    t = tau(u)
    d = np.where(np.abs(u) < C_B, (1 - (u / C_B) ** 2) * (1 - 5 * (u / C_B) ** 2), 0.0)
    i = np.arange(1, S + 1)
    return t[:, None] ** i, i * t[:, None] ** (i - 1) * d[:, None]


def transformed(u, J):
    t, d = u / (1 + np.abs(u)), 1 / (1 + np.abs(u)) ** 2
    j = np.arange(1, J + 1)
    return t[:, None] ** j, j * t[:, None] ** (j - 1) * d[:, None]


def weighted(u, J):
    w = np.exp(-0.5 * u * u)[:, None]
    j = np.arange(1, J + 1)
    c = u[:, None]
    return w * c ** j, w * (j * c ** (j - 1) - c ** (j + 1))


def one_rep(X, y):
    XtX_inv = np.linalg.inv(X.T @ X)
    ols = XtX_inv @ (X.T @ y)

    def step(pilot, scale, basis):
        Phi, dPhi = basis
        Pc = Phi - Phi.mean(0)
        F = Pc.T @ Pc / len(y)
        b = dPhi.mean(0)
        h = np.linalg.solve(F + 1e-10 * np.trace(F) * np.eye(len(b)), b)
        slope = float(h @ b)
        if not slope > 1e-10:
            return pilot
        return pilot + scale * XtX_inv @ (X.T @ (Pc @ h)) / slope

    def sse(beta):
        return float(np.sum(beta[1:] ** 2))

    base = irls(X, y, irls(X, y, ols, w_hub), w_bi)
    r = y - X @ base
    s = mad(r)
    u = r / s
    r0 = y - X @ ols
    s0 = r0.std(ddof=1)
    u0 = r0 / s0
    return [sse(ols), sse(base),
            sse(step(base, s, tau_powers(u))),
            sse(step(base, s, transformed(u, 3))), sse(step(base, s, transformed(u, 6))),
            sse(step(base, s, weighted(u, 3))), sse(step(base, s, weighted(u, 6))),
            sse(step(ols, s0, transformed(u0, 3))), sse(step(ols, s0, weighted(u0, 3)))]


def run(out_dir, R, ns, law_ids):
    for li in law_ids:
        name, law = LAWS[li]
        for n in ns:
            rng = np.random.default_rng([20261118, li, n])
            res = np.empty((R, len(ARMS)))
            for k in range(R):
                x = rng.standard_normal((n, P - 1))
                res[k] = one_rep(np.column_stack([np.ones(n), x]), law(n, rng))
            np.save(os.path.join(out_dir, f"nw_{li}_{n}_{R}.npy"), res)
            print(name, n, "done", flush=True)


def collect(out_dir):
    cells = {}
    for path in glob.glob(os.path.join(out_dir, "nw_*.npy")):
        li, n, R = (int(v) for v in os.path.basename(path)[3:-4].split("_"))
        cells[(n, li)] = (R, np.load(path))
    print("E4b: slope MSE / OLS slope MSE, same samples for every arm; one Newton step")
    print("T = transformed (e/(1+|e|))^j, W = weighted exp(-e^2/2) e^j; suffix o = OLS pilot, SD scale")
    for n in sorted({k[0] for k in cells}):
        print(f"\n=== N = {n} ===")
        print(f"{'закон':22s}{'R':>6s}" + "".join(f"{a:>8s}" for a in ARMS[1:]))
        for (nn, li), (R, res) in sorted(cells.items()):
            if nn == n:
                print(f"{LAWS[li][0]:22s}{R:6d}" +
                      "".join(f"{v:8.3f}" for v in res[:, 1:].mean(0) / res[:, 0].mean()))
    print("\nPaired ratios of slope MSE with delta-method standard errors (same samples):")
    for n in sorted({k[0] for k in cells}):
        print(f"\n=== N = {n} ===")
        print(f"{'закон':22s}" + "".join(f"{a + '/' + b:>18s}" for a, b in PAIRS))
        for (nn, li), (R, res) in sorted(cells.items()):
            if nn != n:
                continue
            col = dict(zip(ARMS, res.T))
            line = f"{LAWS[li][0]:22s}"
            for a, b in PAIRS:
                ratio = col[a].mean() / col[b].mean()
                se = np.std((col[a] - ratio * col[b]) / col[b].mean(), ddof=1) / math.sqrt(R)
                line += f"{ratio:10.3f} ({se:.3f})"
            print(line)


if __name__ == "__main__":
    if sys.argv[1] == "run":
        ids = [int(v) for v in sys.argv[5].split(",")] if len(sys.argv) > 5 else range(len(LAWS))
        run(sys.argv[2], int(sys.argv[3]), [int(v) for v in sys.argv[4].split(",")], ids)
    else:
        collect(sys.argv[2])

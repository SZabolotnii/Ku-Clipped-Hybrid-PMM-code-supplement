"""E4: tau-PMM / tau-PATP against prior-art adaptive estimators on the same data.

Usage:
  baselines_mc.py run <data_dir> <R> <N list> [law index list]   simulate, save data and results
  baselines_mc.py collect <data_dir>                             print the table (reads asm_*.txt too)

Every arm starts from the same bisquare M-estimate (MAD scale) and takes one Newton step.
Arms (slope MSE / OLS slope MSE is reported):
  M          bisquare M-estimate, the order-1 member
  tPMM3      basis tau^i, i = 1..3, tau = bisquare psi
  tPATP      basis sign(tau)|tau|^{p_i(alpha)}, i = 1..3, plus tau^2; alpha chosen in-sample
  PMM3       unclipped powers u, u^2, u^3 (classical PMM, needs moments up to 6)
  Newey3/6   powers of the bounded transform u / (1 + |u|), K = 3 and 6 (Newey 1988 type sieve)
  Kernel     one-step with the kernel score estimate -f'/f, 2-fold cross-fitted (Bickel 1982 type);
             Gaussian kernel, bandwidth nrd0, score set to 0 where f < FLOOR * max f
  asm        R package asm (Feng, Kao, Xu, Samworth), run by baselines_asm.R on the saved data;
             asm_asym = symmetric=FALSE, asm_sym = package default symmetric=TRUE; LAD pilot

The data of every replication are saved as float32 (and used after that rounding) so that the
R run sees exactly the same samples. Seed: [20261116, law index, n].
"""
import glob
import math
import os
import sys

import numpy as np

P = 5
SD_BI = math.sqrt(1 + 1.5 ** 2)
GRID = (0.0, 0.125, 0.25, 0.375, 0.625, 0.75, 0.875, 1.0)
C_H, C_B = 1.345, 4.685
FLOOR = 1e-3
ARMS = ("OLS", "M", "tPMM3", "tPATP", "PMM3", "Newey3", "Newey6", "Kernel")


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


LAWS = [
    ("Гаус (контроль)", gauss),
    ("Лаплас", lambda n, rng: rng.laplace(0, 1 / math.sqrt(2), n)),
    ("t3", lambda n, rng: rng.standard_t(3, n) / math.sqrt(3.0)),
    ("бімодальний", bimodal),
    ("Gamma(4)", gamma4),
    ("Гаус + 5% ±10", contaminate(gauss)),
    ("Gamma(4) + 5% ±10", contaminate(gamma4)),
    ("бімодальний + 5% ±10", contaminate(bimodal)),
]


def p_i(i, alpha):
    return 1 / i + (4 - i - 3 / i) * alpha + (2 * i - 4 + 2 / i) * alpha ** 2


def tau(u):
    return np.where(np.abs(u) < C_B, u * (1 - (u / C_B) ** 2) ** 2, 0.0)


w_hub = lambda u: np.minimum(1.0, C_H / np.maximum(np.abs(u), 1e-12))
w_bi = lambda u: np.where(np.abs(u) < C_B, (1 - (u / C_B) ** 2) ** 2, 0.0)


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


def phi_tau(u, specs):
    t = tau(u)
    at, sg = np.abs(t), np.sign(t)
    return np.column_stack([(sg if s else 1.0) * at ** p for p, s in specs])


def tau_columns(u, specs, delta):
    """Phi and the rows whose mean is b, by symmetric difference (finite variance for p < 1)."""
    return phi_tau(u, specs), (phi_tau(u + delta, specs) - phi_tau(u - delta, specs)) / (2 * delta)


def weights(Phi, dPhi):
    Pc = Phi - Phi.mean(0)
    F = Pc.T @ Pc / len(Phi)
    b = dPhi.mean(0)
    h = np.linalg.solve(F + 1e-10 * np.trace(F) * np.eye(len(b)), b)
    return Pc @ h, float(h @ b)


def kernel_score(train, at):
    m = len(train)
    q75, q25 = np.percentile(train, [75, 25])
    bw = 0.9 * min(train.std(ddof=1), (q75 - q25) / 1.34) * m ** -0.2
    z = (at[:, None] - train[None, :]) / bw
    k = np.exp(-0.5 * z * z)
    f = k.mean(1)
    return np.where(f > FLOOR * f.max(), (z * k).mean(1) / (bw * np.maximum(f, 1e-300)), 0.0)


SPECS_PATP = [[(p_i(i, a), True) for i in (1, 2, 3)] + [(2, False)] for a in GRID]
SPECS_PMM3 = [(1, True), (2, False), (3, True)]


def one_rep(X, y, rng):
    n = len(y)
    XtX_inv = np.linalg.inv(X.T @ X)
    ols = XtX_inv @ (X.T @ y)
    base = irls(X, y, irls(X, y, ols, w_hub), w_bi)
    r = y - X @ base
    s = mad(r)
    u = r / s
    delta = n ** -0.2

    def step(psi, slope, scale=s):
        if not slope > 1e-10:
            return base
        return base + scale * XtX_inv @ (X.T @ psi) / slope

    def sse(beta):
        return float(np.sum(beta[1:] ** 2))

    out = [sse(ols), sse(base)]
    out.append(sse(step(*weights(*tau_columns(u, SPECS_PMM3, delta)))))
    best = (np.inf, base)
    for sp in SPECS_PATP:
        psi, slope = weights(*tau_columns(u, sp, delta))
        v = 1.0 / slope if slope > 1e-10 else np.inf
        if v < best[0]:
            best = (v, step(psi, slope))
    out.append(sse(best[1]))
    out.append(sse(step(*weights(np.column_stack([u, u ** 2, u ** 3]),
                                 np.column_stack([np.ones(n), 2 * u, 3 * u ** 2])))))
    t, d = u / (1 + np.abs(u)), 1 / (1 + np.abs(u)) ** 2
    for K in (3, 6):
        ks = np.arange(1, K + 1)
        out.append(sse(step(*weights(t[:, None] ** ks, ks * t[:, None] ** (ks - 1) * d[:, None]))))
    perm = rng.permutation(n)
    A, B = perm[: n // 2], perm[n // 2:]
    psi = np.empty(n)
    psi[B] = kernel_score(r[A], r[B])
    psi[A] = kernel_score(r[B], r[A])
    psi -= psi.mean()
    out.append(sse(step(psi, float(np.mean(psi ** 2)), scale=1.0)))
    return out


def run(data_dir, R, ns, law_ids):
    for li in law_ids:
        name, law = LAWS[li]
        for n in ns:
            rng = np.random.default_rng([20261116, li, n])
            data = np.empty((R, n, P), dtype=np.float32)
            res = np.empty((R, len(ARMS)))
            for k in range(R):
                data[k, :, : P - 1] = rng.standard_normal((n, P - 1))
                data[k, :, P - 1] = law(n, rng)
                d = data[k].astype(float)
                res[k] = one_rep(np.column_stack([np.ones(n), d[:, : P - 1]]), d[:, P - 1], rng)
            data.tofile(os.path.join(data_dir, f"data_{li}_{n}_{R}.bin"))
            np.save(os.path.join(data_dir, f"py_{li}_{n}_{R}.npy"), res)
            print(name, n, "done", flush=True)


def collect(data_dir):
    cells = {}
    for path in glob.glob(os.path.join(data_dir, "py_*.npy")):
        li, n, R = (int(v) for v in os.path.basename(path)[3:-4].split("_"))
        cells[(n, li)] = (R, path)
    print("E4: slope MSE / OLS slope MSE, same samples for every arm; tau = bisquare; one Newton step")
    print("asm columns: R package asm; 'fail' = replications where asm.fit returned an error or NA")
    for n in sorted({k[0] for k in cells}):
        print(f"\n=== N = {n} ===")
        print(f"{'закон':22s}{'R':>6s}" + "".join(f"{a:>8s}" for a in ARMS[1:]) +
              f"{'asm_asym':>10s}{'asm_sym':>9s}{'fail':>6s}")
        for (nn, li), (R, path) in sorted(cells.items()):
            if nn != n:
                continue
            res = np.load(path)
            m = res[:, 0].mean()
            line = f"{LAWS[li][0]:22s}{R:6d}" + "".join(f"{v:8.3f}" for v in res[:, 1:].mean(0) / m)
            apath = os.path.join(data_dir, f"asm_{li}_{n}_{R}.txt")
            if os.path.exists(apath):
                a = np.genfromtxt(apath, skip_header=1)
                ok = ~np.isnan(a).any(1)
                mo = res[ok, 0].mean()
                line += f"{a[ok, 0].mean() / mo:10.3f}{a[ok, 1].mean() / mo:9.3f}{int((~ok).sum()):6d}"
            else:
                line += f"{'-':>10s}{'-':>9s}{'-':>6s}"
            print(line)
    print("\nPaired ratios of slope MSE with delta-method standard errors (same samples):")
    pairs = (("tPMM3", "Newey3"), ("tPMM3", "Newey6"), ("tPMM3", "Kernel"), ("tPMM3", "asm_asym"),
             ("tPATP", "Newey6"), ("tPATP", "asm_asym"))
    for n in sorted({k[0] for k in cells}):
        print(f"\n=== N = {n} ===")
        print(f"{'закон':22s}" + "".join(f"{a + '/' + b:>18s}" for a, b in pairs))
        for (nn, li), (R, path) in sorted(cells.items()):
            apath = os.path.join(data_dir, f"asm_{li}_{n}_{R}.txt")
            if nn != n or not os.path.exists(apath):
                continue
            col = dict(zip(ARMS, np.load(path).T))
            col["asm_asym"] = np.genfromtxt(apath, skip_header=1)[:, 0]
            line = f"{LAWS[li][0]:22s}"
            for a, b in pairs:
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

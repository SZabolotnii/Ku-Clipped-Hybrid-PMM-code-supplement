"""Numerical check of T9 (theory/t9_bv_one_step.tex): the one-step estimator on a basis with
fractional exponents, with b estimated by the symmetric difference of width delta = N^(-1/5),
has slope covariance s^2 / (b'F^{-1}b) * Sigma^{-1} / N in the limit.

Basis: PATP at alpha = 0.25 on the bisquare tau, exponents (1, 0.6875, 0.5), plus tau^2.
Population quantities by quadrature from the known density of e; alpha* (bisquare location
functional) and s (normalised MAD) from one sample of size 4e6. b = int f_u d(phi) is computed
as int phi(v) (f_u(v - d0) - f_u(v + d0)) / (2 d0) dv with d0 = 1e-4.

Reported per law and N:
  feasible / theory     N * MSE of the slopes over (p - 1) s^2 / (b'F^{-1}b)
  fixed-delta / theory  the first-order variance at the finite width delta_N (Proposition 2 of
                        theory/t8_selector.tex: weights F^{-1}b_delta, slope b_delta'F^{-1}b_delta,
                        pilot share rho) over the same limit
T9 says the first ratio tends to 1; the second shows how much of the gap at a given N is the
smoothing width. argv: R list for N = 2000, 8000, 32000 (default 1000,600,300).
"""
import math
import sys

import numpy as np
from scipy import integrate

from baselines_mc import C_B, P, contaminate, gamma4, gauss, irls, mad, p_i, tau, w_bi, w_hub

RS = [int(v) for v in sys.argv[1].split(",")] if len(sys.argv) > 1 else [1000, 600, 300]
NS = (2000, 8000, 32000)
ALPHA = 0.25
EXPS = [p_i(i, ALPHA) for i in (1, 2, 3)]
B_LAP = 1 / math.sqrt(2)
phi_n = lambda z: math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)


def f_gamma4(e):
    x = 2 * e + 4
    return 2 * x ** 3 * math.exp(-x) / 6 if x > 0 else 0.0


LAWS = {
    "Гаус": (gauss, phi_n),
    "Лаплас": (lambda n, rng: rng.laplace(0, B_LAP, n), lambda e: math.exp(-abs(e) / B_LAP) / (2 * B_LAP)),
    "Gamma(4) + 5% ±10": (contaminate(gamma4),
                          lambda e: 0.95 * f_gamma4(e) + 0.025 * (phi_n(e - 10) + phi_n(e + 10))),
}


def tau1(v):
    return float(tau(np.array(v)))


def columns_scalar():
    cols = [lambda v, p=p: math.copysign(abs(tau1(v)) ** p, v) for p in EXPS]
    cols.append(lambda v: tau1(v) ** 2)
    return cols


def phi_matrix(u):
    t = tau(u)
    at, sg = np.abs(t), np.sign(t)
    return np.column_stack([sg * at ** p for p in EXPS] + [t ** 2])


def quad(g, pts=()):
    edges = sorted({-C_B, 0.0, C_B, *[p for p in pts if -C_B < p < C_B]})
    return sum(integrate.quad(g, lo, hi, limit=400, epsabs=1e-13, epsrel=1e-11)[0]
               for lo, hi in zip(edges, edges[1:]))


def slope_vector(cols, fu, d):
    """E[(phi(u + d) - phi(u - d)) / (2 d)] = int phi(v) (f_u(v - d) - f_u(v + d)) / (2 d) dv."""
    return np.array([quad(lambda v: g(v) * (fu(v - d) - fu(v + d)) / (2 * d), (-d, d)) for g in cols])


def population(law, dens, rng):
    e = law(4_000_000, rng)
    one = np.ones((len(e), 1))
    a = irls(one, e, irls(one, e, np.array([np.median(e)]), w_hub), w_bi)[0]
    s = mad(e)
    fu = lambda v: s * dens(a + s * v)
    cols = columns_scalar()
    m = np.array([quad(lambda v: g(v) * fu(v)) for g in cols])
    F = np.array([[quad(lambda v: gi(v) * gj(v) * fu(v)) for gj in cols] for gi in cols]) - np.outer(m, m)
    b = slope_vector(cols, fu, 1e-4)
    m_tau = quad(lambda v: tau1(v) * fu(v))
    var_tau = quad(lambda v: tau1(v) ** 2 * fu(v)) - m_tau ** 2
    b_M = slope_vector([tau1], fu, 1e-4)[0]
    cov_tau = np.array([quad(lambda v: g(v) * tau1(v) * fu(v)) for g in cols]) - m * m_tau
    J = float(b @ np.linalg.solve(F, b))

    def fixed_delta(d):
        bd = slope_vector(cols, fu, d)
        hd = np.linalg.solve(F, bd)
        Jd = float(hd @ bd)
        rho = 1 - float(hd @ b) / Jd
        return (rho ** 2 * var_tau / b_M ** 2 + 2 * rho * float(hd @ cov_tau) / (b_M * Jd) + 1 / Jd) * J

    return s, J, fixed_delta


rng = np.random.default_rng(20261119)
print(f"alpha = {ALPHA}, exponents {[round(p, 4) for p in EXPS]} + tau^2; tau = bisquare; one Newton step; "
      f"b: symmetric difference, delta = N^(-1/5); R = {dict(zip(NS, RS))}")
print(f"{'закон':20s}{'theory/OLS':>11s} |" + "".join(f"{'N=' + str(n):>22s}" for n in NS))
print(f"{'':20s}{'':>11s} |" + f"{'feasible  fixed-delta':>22s}" * len(NS))
for name, (law, dens) in LAWS.items():
    s, J, fixed_delta = population(law, dens, rng)
    theory = (P - 1) * s * s / J
    var_e = float(np.var(law(4_000_000, rng)))
    line = f"{name:20s}{s * s / J / var_e:11.3f} |"
    for n, R in zip(NS, RS):
        delta = n ** -0.2
        acc = 0.0
        for _ in range(R):
            X = np.column_stack([np.ones(n), rng.standard_normal((n, P - 1))])
            y = law(n, rng)
            XtX_inv = np.linalg.inv(X.T @ X)
            base = irls(X, y, irls(X, y, XtX_inv @ (X.T @ y), w_hub), w_bi)
            r = y - X @ base
            sh = mad(r)
            u = r / sh
            Phi = phi_matrix(u)
            Pc = Phi - Phi.mean(0)
            bh = ((phi_matrix(u + delta) - phi_matrix(u - delta)) / (2 * delta)).mean(0)
            hh = np.linalg.solve(Pc.T @ Pc / n, bh)
            beta = base + sh * XtX_inv @ (X.T @ (Pc @ hh)) / float(hh @ bh)
            acc += float(np.sum(beta[1:] ** 2))
        line += f"{n * acc / R / theory:11.3f}{fixed_delta(delta):11.3f}"
    print(line, flush=True)

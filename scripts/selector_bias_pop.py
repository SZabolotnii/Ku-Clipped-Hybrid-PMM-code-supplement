"""Why the alpha-selector lags the best fixed alpha at large N: a population calculation.

Law: generalised Gaussian GG(beta) (density ~ exp(-|x|^beta), unit variance), symmetric, so the
pilot location is 0. u = e / s with s the normalised MAD. Basis B(alpha) as in
tau_patp_adaptive_mc.py: sign(tau)|tau|^{p_i(alpha)}, i = 1..3, plus tau^2; tau = bisquare.

For a smoothing width delta (the script uses delta = N^(-1/5) in u units) define
  b        = E[phi * score]                       the exact vector (Stein form, score known)
  b_delta  = E[(phi(u+delta) - phi(u-delta)) / (2 delta)]   what the symmetric difference estimates
  F        = Cov(phi),  h_delta = F^{-1} b_delta
  J        = b'F^{-1}b                            information of the basis
  J_delta  = b_delta'F^{-1}b_delta                limit of the selection criterion
The one-step estimator with weights h_delta and slope J_delta has, to first order,
  beta_hat - beta = rho (beta_pilot - beta) + s (X'X)^{-1} X' psi_delta / J_delta,
  rho = 1 - h_delta'b / J_delta,   psi_delta = h_delta'(phi - E phi),
so its slope variance, relative to OLS, is
  V_real = s^2 Var(rho tau / b_M + psi_delta / J_delta),   b_M = E[tau * score].
Printed per N: ideal s^2/J, the criterion s^2/J_delta, V_real, for every alpha of the grid; the
alpha the criterion selects in the limit, the alpha with the smallest V_real, and the lag
V_real(selected) / min V_real. Last block: the local exponent of the bias of b_delta for the
column with the smallest exponent at alpha = 0 (p = 1/3); the expected rate is delta^(p + beta)
for beta < 2, against delta^2 for a smooth density.
argv: beta (default 0.7), delta constant c in delta = c N^(-1/5) (default 1.0).
"""
import math
import sys

import numpy as np
from scipy import integrate, special

BETA = float(sys.argv[1]) if len(sys.argv) > 1 else 0.7
DELTA_C = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
C_B = 4.685
GRID = (0.0, 0.125, 0.25, 0.375, 0.625, 0.75, 0.875, 1.0)

c = math.sqrt(math.gamma(3 / BETA) / math.gamma(1 / BETA))          # x = g / c, g ~ exp(-|g|^beta)
med_abs_x = special.gammaincinv(1 / BETA, 0.5) ** (1 / BETA) / c     # median |x|
s = 1.4826 * med_abs_x
k = c * s                                                            # g = k u
norm = BETA * k / (2 * math.gamma(1 / BETA))


def f(u):
    return norm * math.exp(-abs(k * u) ** BETA)


def score(u):                                                        # -f'/f, u > 0
    return BETA * k ** BETA * u ** (BETA - 1)


def tau(u):
    return u * (1 - (u / C_B) ** 2) ** 2 if abs(u) < C_B else 0.0


def p_i(i, a):
    return 1 / i + (4 - i - 3 / i) * a + (2 * i - 4 + 2 / i) * a ** 2


def quad(g, a, b, pts=()):
    pts = sorted(p for p in pts if a < p < b)
    edges = [a] + pts + [b]
    return sum(integrate.quad(g, lo, hi, limit=400, epsabs=1e-13, epsrel=1e-11)[0]
               for lo, hi in zip(edges, edges[1:]))


def basis(alpha):
    """Odd columns sign(tau)|tau|^p (as functions of u on the whole line) and the even column tau^2."""
    ps = [p_i(i, alpha) for i in (1, 2, 3)]
    odd = [lambda u, p=p: math.copysign(abs(tau(u)) ** p, u) for p in ps]
    even = lambda u: tau(u) ** 2
    return odd, even


def population(alpha, delta):
    odd, even = basis(alpha)
    # By symmetry odd and even columns are uncorrelated and the even column has b = 0, so it gets
    # zero weight in the population; only the 3x3 odd block matters.
    F = np.array([[2 * quad(lambda u: gi(u) * gj(u) * f(u), 0, C_B) for gj in odd] for gi in odd])
    b = np.array([2 * quad(lambda u: g(u) * score(u) * f(u), 0, C_B) for g in odd])
    brk = (delta, C_B - delta, C_B)
    bd = np.array([2 * quad(lambda u: (g(u + delta) - g(u - delta)) / (2 * delta) * f(u),
                            0, C_B + delta, brk) for g in odd])
    cov_tau = np.array([2 * quad(lambda u: g(u) * tau(u) * f(u), 0, C_B) for g in odd])
    return F, b, bd, cov_tau


E_tau2 = 2 * quad(lambda u: tau(u) ** 2 * f(u), 0, C_B)
b_M = 2 * quad(lambda u: tau(u) * score(u) * f(u), 0, C_B)
print(f"GG({BETA}): s = {s:.4f}; bisquare M-estimate / OLS = {s * s * E_tau2 / b_M ** 2:.3f}; "
      f"Cramer-Rao / OLS = {1 / (c * c * BETA ** 2 * math.gamma(2 - 1 / BETA) / math.gamma(1 / BETA)):.3f}"
      if BETA > 0.5 else f"GG({BETA}): s = {s:.4f}")
print(f"delta = {DELTA_C} N^(-1/5) in u units")
print("ideal = s^2/J;  crit = s^2/J_delta (limit of the selection criterion);  real = V_real / OLS")
for n in (2000, 8000, 32000, 10 ** 6, 10 ** 8):
    delta = DELTA_C * n ** -0.2
    rows = []
    for a in GRID:
        F, b, bd, ct = population(a, delta)
        J = float(b @ np.linalg.solve(F, b))
        hd = np.linalg.solve(F, bd)
        Jd = float(hd @ bd)
        rho = 1 - float(hd @ b) / Jd
        real = rho ** 2 * E_tau2 / b_M ** 2 + 2 * rho * float(hd @ ct) / (b_M * Jd) + 1 / Jd
        rows.append((s * s / J, s * s / Jd, s * s * real, rho))
    rows = np.array(rows)
    sel, best = int(np.argmin(rows[:, 1])), int(np.argmin(rows[:, 2]))
    print(f"\n=== N = {n}, delta = {delta:.4f} ===")
    print(f"{'alpha':>7s}" + "".join(f"{a:>8.3f}" for a in GRID))
    for name, col in (("ideal", 0), ("crit", 1), ("real", 2), ("rho", 3)):
        print(f"{name:>7s}" + "".join(f"{v:8.3f}" for v in rows[:, col]))
    print(f"criterion selects alpha = {GRID[sel]}; smallest real variance at alpha = {GRID[best]}; "
          f"lag = {rows[sel, 2] / rows[best, 2]:.3f}; "
          f"ideal selection would lose {rows[int(np.argmin(rows[:, 0])), 2] / rows[best, 2]:.3f}")

print(f"\nBias of b_delta, column p = 1/3 (alpha = 0); expected exponent p + beta = {1 / 3 + BETA:.3f}")
g = basis(0.0)[0][2]
b_exact = 2 * quad(lambda u: g(u) * score(u) * f(u), 0, C_B)
prev = None
for delta in (0.08, 0.04, 0.02, 0.01, 0.005):
    bd = 2 * quad(lambda u: (g(u + delta) - g(u - delta)) / (2 * delta) * f(u),
                  0, C_B + delta, (delta, C_B - delta, C_B))
    rel = (bd - b_exact) / b_exact
    rate = "" if prev is None else f"   local exponent {math.log(prev / rel) / math.log(2):.3f}"
    print(f"delta = {delta:6.3f}   relative bias = {rel:+.5f}{rate}")
    prev = rel

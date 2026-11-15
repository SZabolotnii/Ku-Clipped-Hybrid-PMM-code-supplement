"""PATP on a robust generating element: exact population variance ratios.

PATP (Form B, arXiv:2605.14610): phi_i(xi; alpha) = sign(xi)|xi|^{p_i(alpha)},
p_i(alpha) = 1/i + (4 - i - 3/i) alpha + (2i - 4 + 2/i) alpha^2  (1/i, 1, i at alpha = 0, 1/2, 1).
Here xi = tau(u): the psi-function of a robust M-estimator (or tau(u) = u, the plain PATP).
All PATP columns are odd, so asymmetry needs an even column; 'mix' adds tau^2.
Columns: S=1 (the M-estimator), PMM-3 {tau, tau^2, tau^3}, PATP-3 at four alpha,
mix3 {tau, tau^2, sgn|tau|^(1/2)}, mix4 {tau, tau^2, tau^3, sgn|tau|^(1/2)}, bound.
"""
import math
import sys

import numpy as np

from generating_elements_pop import bisquare, huber, welsch
from poplaws import build


def p_i(i, alpha):
    return 1 / i + (4 - i - 3 / i) * alpha + (2 * i - 4 + 2 / i) * alpha ** 2


def identity():
    return (lambda u: u, lambda u: 1.0, [])


ELEMENTS = {"u (без обрізання)": identity, "Huber": huber, "bisquare": bisquare, "Welsch": welsch}


def col(tau, dtau, p, signed):
    """sign(tau)^signed |tau|^p and its derivative in u."""
    def f(u):
        t = tau(u)
        if t == 0.0:
            return 0.0
        return (math.copysign(1.0, t) if signed else 1.0) * abs(t) ** p

    def d(u):
        t = tau(u)
        if t == 0.0:
            return 0.0
        return p * abs(t) ** (p - 1) * (1.0 if signed else math.copysign(1.0, t)) * dtau(u)
    return f, d


def ratio(law, cols, a, s, kinks):
    kk = sorted({a + s * k for k in kinks} | {a})
    u = lambda x: (x - a) / s
    k = len(cols)
    m = [law.E(lambda x, f=f: f(u(x)), kk) for f, _ in cols]
    b = np.array([law.E(lambda x, d=d: d(u(x)) / s, kk) for _, d in cols])
    F = np.empty((k, k))
    for i in range(k):
        for j in range(i, k):
            F[i, j] = F[j, i] = law.E(lambda x, f=cols[i][0], g=cols[j][0]: f(u(x)) * g(u(x)), kk) - m[i] * m[j]
    return 1.0 / (law.var * float(b @ np.linalg.solve(F, b)))


ALPHAS = (0.0, 0.25, 0.75, 1.0)

if __name__ == "__main__":
    laws = build()
    head = ["S=1", "PMM-3"] + [f"PATP3 a={a}" for a in ALPHAS] + ["mix3", "mix4", "межа"]
    for name, make in ELEMENTS.items():
        tau, dtau, kinks = make()
        unb = name.startswith("u")
        print(f"\n=== xi = tau(u), tau = {name} ===")
        print(f"{'закон':24s}" + "".join(f"{h:>13s}" for h in head))
        for law in laws:
            s = law.madn
            a = law.mean if unb else law.huber_centre(1.345 * s)

            def run(specs):
                top = max(p for p, _ in specs)
                if unb and 2 * top > law.max_moment:
                    return float("nan")
                return ratio(law, [col(tau, dtau, p, sg) for p, sg in specs], a, s, kinks)

            vals = [run([(1, True)]), run([(1, True), (2, False), (3, True)])]
            vals += [run([(p_i(i, al), True) for i in (1, 2, 3)]) for al in ALPHAS]
            vals += [run([(1, True), (2, False), (0.5, True)]),
                     run([(1, True), (2, False), (3, True), (0.5, True)])]
            vals.append(1.0 / (law.var * law.fisher()))
            print(f"{law.name:24s}" + "".join(f"{'н/д':>13s}" if not np.isfinite(v) else f"{v:13.3f}" for v in vals))
            sys.stdout.flush()

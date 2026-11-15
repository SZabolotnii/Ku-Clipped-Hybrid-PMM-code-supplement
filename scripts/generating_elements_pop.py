"""PMM on powers of a robust generating element tau: exact population variance ratios.

Basis phi_i(e) = tau(u)^i, u = (e - a) / s, i = 1..S, with tau the psi-function of a
classical robust M-estimator at its 95%-Gaussian-efficiency constant. S = 1 is that
M-estimator itself; S = 2, 3 add the Godambe-optimal polynomial correction h = F^{-1} b.
Output: slope variance / OLS variance = 1 / (sigma^2 b'F^{-1}b); nested spans, so the
ratio cannot increase with S. Scale s = MADN, centre a = Huber(1.345) location.
"""
import math
import sys

import numpy as np

from poplaws import build


def huber(c=1.345):
    return (lambda u: max(-c, min(c, u)), lambda u: 1.0 if abs(u) < c else 0.0, [-c, c])


def logistic(c=1.205):
    return (lambda u: c * math.tanh(u / c), lambda u: 1 - math.tanh(u / c) ** 2, [])


def hampel(a=1.3524, b=3.1556, r=7.2129):
    def tau(u):
        x = abs(u)
        sg = 1.0 if u > 0 else -1.0
        if x <= a:
            return u
        if x <= b:
            return a * sg
        if x <= r:
            return a * sg * (r - x) / (r - b)
        return 0.0

    def d(u):
        x = abs(u)
        if x < a:
            return 1.0
        if b < x < r:
            return -a / (r - b)
        return 0.0
    return (tau, d, [-r, -b, -a, a, b, r])


def bisquare(c=4.685):
    return (lambda u: u * (1 - (u / c) ** 2) ** 2 if abs(u) < c else 0.0,
            lambda u: (1 - (u / c) ** 2) * (1 - 5 * (u / c) ** 2) if abs(u) < c else 0.0, [-c, c])


def andrews(c=1.339):
    lim = c * math.pi
    return (lambda u: c * math.sin(u / c) if abs(u) < lim else 0.0,
            lambda u: math.cos(u / c) if abs(u) < lim else 0.0, [-lim, lim])


def welsch(c=2.985):
    return (lambda u: u * math.exp(-(u / c) ** 2),
            lambda u: (1 - 2 * (u / c) ** 2) * math.exp(-(u / c) ** 2), [])


def cauchy(c=2.385):
    return (lambda u: u / (1 + (u / c) ** 2),
            lambda u: (1 - (u / c) ** 2) / (1 + (u / c) ** 2) ** 2, [])


ELEMENTS = {"Huber": huber, "logistic": logistic, "Hampel": hampel, "bisquare": bisquare,
            "Andrews": andrews, "Welsch": welsch, "Cauchy": cauchy}


def ratios(law, elem, a, s, S=3):
    tau, dtau, kinks = elem
    kk = sorted({a + s * k for k in kinks} | {a})
    u = lambda x: (x - a) / s
    m = [law.E(lambda x, i=i: tau(u(x)) ** i, kk) for i in range(1, 2 * S + 1)]   # E tau^i
    b = np.array([law.E(lambda x, i=i: i * tau(u(x)) ** (i - 1) * dtau(u(x)) / s, kk) for i in range(1, S + 1)])
    F = np.array([[m[i + j + 1] - m[i] * m[j] for j in range(S)] for i in range(S)])
    out = []
    for k in range(1, S + 1):
        out.append(1.0 / (law.var * float(b[:k] @ np.linalg.solve(F[:k, :k], b[:k]))))
    return out


if __name__ == "__main__":
    laws = build()
    names = sys.argv[1:] or list(ELEMENTS)
    info = {}
    for law in laws:
        info[law.name] = (law.huber_centre(1.345 * law.madn), law.madn, 1.0 / (law.var * law.fisher()))
    for name in names:
        print(f"\n=== tau = {name} (S = 1 is the classical {name} M-estimator) ===")
        print(f"{'закон':24s}{'S=1':>8s}{'S=2':>8s}{'S=3':>8s}{'межа':>8s}")
        for law in laws:
            a, s, bound = info[law.name]
            r = ratios(law, ELEMENTS[name](), a, s)
            print(f"{law.name:24s}" + "".join(f"{v:8.3f}" for v in r) + f"{bound:8.3f}")
            sys.stdout.flush()

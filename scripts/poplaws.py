"""Error laws for exact population calculations: fast scalar pdf/cdf, expectations by
adaptive quadrature split at the law's break points, Fisher information, median, MADN."""
import math

import numpy as np
from scipy import integrate, optimize, special, stats

SQ2PI = math.sqrt(2 * math.pi)


def norm_c(m, s):
    return (lambda x: math.exp(-0.5 * ((x - m) / s) ** 2) / (s * SQ2PI),
            lambda x: 0.5 * math.erfc(-(x - m) / (s * math.sqrt(2))))


def laplace_c(b):
    return (lambda x: math.exp(-abs(x) / b) / (2 * b),
            lambda x: 0.5 * math.exp(x / b) if x < 0 else 1 - 0.5 * math.exp(-x / b))


def t6_c():
    s = 1 / math.sqrt(1.5)
    C = math.gamma(3.5) / (math.sqrt(6 * math.pi) * math.gamma(3.0))
    return (lambda x: C * (1 + (x / s) ** 2 / 6) ** -3.5 / s,
            lambda x: float(stats.t.cdf(x / s, 6)))


def gamma4_c():
    def pdf(e):
        X = 2 * e + 4
        return 2 * X ** 3 * math.exp(-X) / 6 if X > 0 else 0.0
    return (pdf, lambda e: float(special.gammainc(4, max(2 * e + 4, 0.0))))


class Law:
    def __init__(self, name, comps, breaks=(), max_moment=99):
        self.name, self.comps, self.breaks, self.max_moment = name, comps, list(breaks), max_moment
        self.mean = self.E(lambda x: x)
        self.var = self.E(lambda x: (x - self.mean) ** 2)
        self.med = optimize.brentq(lambda x: self.cdf(x) - 0.5, -50, 50)
        self.madn = optimize.brentq(
            lambda t: self.cdf(self.med + t) - self.cdf(self.med - t) - 0.5, 1e-9, 50) / 0.674489750196

    def pdf(self, x):
        return sum(w * p(x) for w, (p, _) in self.comps)

    def cdf(self, x):
        return sum(w * c(x) for w, (_, c) in self.comps)

    def E(self, g, kinks=()):
        pts = sorted(set(list(kinks) + self.breaks))
        edges = [-np.inf] + pts + [np.inf]
        tot = 0.0
        for a, b in zip(edges[:-1], edges[1:]):
            v, _ = integrate.quad(lambda x: g(x) * self.pdf(x), a, b, limit=400)
            tot += v
        return tot

    def fisher(self):
        def integrand(x):
            f = self.pdf(x)
            h = 1e-5
            fm, fp = self.pdf(x - h), self.pdf(x + h)
            if f < 1e-280 or fm <= 0 or fp <= 0:
                return 0.0
            s = (math.log(fp) - math.log(fm)) / (2 * h)
            return s * s * f
        pts = sorted(set(self.breaks))
        edges = [-np.inf] + pts + [np.inf]
        return sum(integrate.quad(integrand, a + (1e-4 if np.isfinite(a) else 0),
                                  b - (1e-4 if np.isfinite(b) else 0), limit=400)[0]
                   for a, b in zip(edges[:-1], edges[1:]))

    def huber_centre(self, c):
        psi = lambda u: max(-c, min(c, u))
        g = lambda a: self.E(lambda x: psi(x - a), [a - c, a, a + c])
        return optimize.brentq(g, self.med - 5 * self.madn - 1, self.med + 5 * self.madn + 1, xtol=1e-9)


def _out(ms):
    return [m + d for m in ms for d in (-6, -3, 0, 3, 6)]


def _scaled(comps, w):
    return [(w * a, c) for a, c in comps]


def build():
    sd_bi = math.sqrt(1 + 1.5 ** 2)
    bi = [(0.5, norm_c(-1.5 / sd_bi, 1 / sd_bi)), (0.5, norm_c(1.5 / sd_bi, 1 / sd_bi))]
    sym = [(0.025, norm_c(10, 1)), (0.025, norm_c(-10, 1))]
    return [
        Law("Гаус", [(1, norm_c(0, 1))], breaks=[-4, 0, 4]),
        Law("Лаплас", [(1, laplace_c(1 / math.sqrt(2)))], breaks=[-5, 0, 5]),
        Law("t6", [(1, t6_c())], breaks=[-5, 0, 5], max_moment=5),
        Law("бімодальний", bi, breaks=[-3, -1, 0, 1, 3]),
        Law("Gamma(4)", [(1, gamma4_c())], breaks=[-2, 0, 3, 8]),
        Law("Гаус + 5% ±10", [(0.95, norm_c(0, 1))] + sym, breaks=[-4, 0, 4] + _out([-10, 10])),
        Law("Гаус + 5% +10", [(0.95, norm_c(0, 1)), (0.05, norm_c(10, 1))], breaks=[-4, 0, 4] + _out([10])),
        Law("Gamma(4) + 5% ±10", [(0.95, gamma4_c())] + sym, breaks=[-2, 0, 3] + _out([-10, 10])),
        Law("бімодальний + 5% ±10", _scaled(bi, 0.95) + sym, breaks=[-3, -1, 0, 1, 3] + _out([-10, 10])),
    ]

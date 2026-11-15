"""Check: the one-step estimator with weights h = F^{-1} b used in this track is, for the slopes,
the linearised GMM estimator of Newey (1988, J. Econometrics 38, 301-339), eq. (2.9), computed
from the same pilot, location and scale.

Newey's estimator is the 3SLS solution
    theta = (Z'(S^{-1} (x) P_X) Z)^{-1} Z'(S^{-1} (x) P_X) Y,     theta = (beta, mu_1..mu_J),
with Z = [M (x) x, I_J (x) e], Y_jt = m_j(e_t) + M_j x_t' beta_pilot, M_j = mean dm_j/de,
S the sample covariance of the moment functions and P_X the projector on [e, x].
Ours is beta_pilot + s (X'X)^{-1} X' psi / (h'b), psi = (Phi - mean Phi) h, h = F^{-1} b, with b the
mean derivative in u = e / s units (no ridge here, so that the comparison is exact).

Printed: max |ours - Newey| over the slopes, per basis. Bases: powers of the bisquare psi
(S = 3) and powers of u / (1 + |u|) (J = 3 and 6). One sample, n = 400, Gamma(4) + 5% outliers.
"""
import numpy as np

from baselines_mc import C_B, contaminate, gamma4, irls, mad, tau, w_bi, w_hub

n, k = 400, 4
rng = np.random.default_rng(20261113)
x = rng.standard_normal((n, k))
y = contaminate(gamma4)(n, rng)
X = np.column_stack([np.ones(n), x])
XtX_inv = np.linalg.inv(X.T @ X)
base = irls(X, y, irls(X, y, XtX_inv @ (X.T @ y), w_hub), w_bi)
r = y - X @ base
s = mad(r)
u = r / s


def tau_powers(S):
    t = tau(u)
    d = np.where(np.abs(u) < C_B, (1 - (u / C_B) ** 2) * (1 - 5 * (u / C_B) ** 2), 0.0)
    i = np.arange(1, S + 1)
    return t[:, None] ** i, i * t[:, None] ** (i - 1) * d[:, None]


def newey_powers(J):
    t, d = u / (1 + np.abs(u)), 1 / (1 + np.abs(u)) ** 2
    j = np.arange(1, J + 1)
    return t[:, None] ** j, j * t[:, None] ** (j - 1) * d[:, None]


def ours(Phi, dPhi):
    Pc = Phi - Phi.mean(0)
    b = dPhi.mean(0)
    h = np.linalg.solve(Pc.T @ Pc / n, b)
    return (base + s * XtX_inv @ (X.T @ (Pc @ h)) / float(h @ b))[1:]


def newey_lgmm(Phi, dPhi):
    J = Phi.shape[1]
    M = dPhi.mean(0) / s
    Pc = Phi - Phi.mean(0)
    Sigma_inv = np.linalg.inv(Pc.T @ Pc / n)
    Z = np.hstack([np.kron(M[:, None], x), np.kron(np.eye(J), np.ones((n, 1)))])
    Y = (Phi + np.outer(x @ base[1:], M)).T.reshape(-1)
    W = np.kron(Sigma_inv, X @ XtX_inv @ X.T)
    return np.linalg.solve(Z.T @ W @ Z, Z.T @ W @ Y)[:k]


def fractional(alpha):
    """Fractional basis of the Letter; the slope vector by symmetric difference, delta = n^(-1/5).
    Newey's formula takes the vector M as given, so the identity does not depend on how b is estimated."""
    from baselines_mc import p_i, tau_columns
    specs = [(p_i(i, alpha), True) for i in (1, 2, 3)] + [(2, False)]
    return tau_columns(u, specs, n ** -0.2)


print(f"n = {n}, k = {k}, Gamma(4) + 5% +-10, bisquare pilot, MAD scale")
for name, (Phi, dPhi) in (("tau^i, S = 3 (bisquare)", tau_powers(3)),
                          ("(u/(1+|u|))^j, J = 3", newey_powers(3)),
                          ("(u/(1+|u|))^j, J = 6", newey_powers(6)),
                          ("fractional, alpha = 0.25", fractional(0.25)),
                          ("fractional, alpha = 0.75", fractional(0.75))):
    a, b_ = ours(Phi, dPhi), newey_lgmm(Phi, dPhi)
    print(f"{name:28s} max|ours - Newey (2.9)| = {np.max(np.abs(a - b_)):.2e}"
          f"   max|step from pilot| = {np.max(np.abs(a - base[1:])):.2e}")

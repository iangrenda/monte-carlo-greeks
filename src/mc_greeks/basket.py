"""Phase 5E: two-asset weighted basket Asian digital with conditional smoothing.

Payoff 1{A >= K}, A = w1 mean_i S1_{t_i} + w2 mean_i S2_{t_i}, under correlated GBM
(correlation rho between the two Brownian motions), t_i = i dt, i = 1..m.

Common Gaussian direction. Let (Z1, Z2) be the two standardised first-step
increments, Corr = rho, and

    Z0 = (Z1 + Z2) / sqrt(2 (1 + rho)) ~ N(0, 1).

By the multivariate-normal regression formula, every Brownian value splits as

    W^j_{t_i} = c sqrt(dt) Z0 + X^j_i,   c = Cov(Z_j, Z0) = sqrt((1 + rho) / 2),

with the residual vector X = (X^1_1..X^1_m, X^2_1..X^2_m) Gaussian and independent
of Z0 (Cov(X^j_i, X^l_k) = dt (rho_jl min(i, k) - c^2), rank 2m - 1). Hence

    A(z) = a1 e^{s1 z} + a2 e^{s2 z},   s_j = sigma_j c sqrt(dt) > 0,
    a_j = w_j mean_i x_j exp((r - sigma_j^2/2) t_i + sigma_j X^j_i) >= 0,

is strictly increasing in z = Z0, so {A >= K} = {Z0 >= z*} with z* the unique
root of A(z*) = K, and

    P(A >= K | X) = Phi(-z*).

Unlike the one-asset case there is no closed form for z* when sigma1 != sigma2;
it is found by Newton's method on h(z) = ln A(z) - ln K, which is increasing and
convex (log-sum-exp), so Newton converges monotonically after the first step.
The Greeks are derivatives of the smooth per-path price D Phi(-z*) and follow
from the implicit function theorem, dz*/dtheta = -(dA/dtheta) / (dA/dz):

    Delta_j  = D phi(z*) a_j e^{s_j z*} / (x_j A'),      A' = sum_l s_l a_l e^{s_l z*},
    Gamma_jj = D [ z* phi(z*) z_x^2 - phi(z*) z_xx ],
    Vega_j   = D phi(z*) (dA/dsigma_j) / A',
    dA/dsigma_j = w_j mean_i S^j_i(z*) (X^j_i + c sqrt(dt) z* - sigma_j t_i).

This is the "conditional QMC" smoothing of Zhang & Wang (2020) / Xiao & Wang
(2018) applied along Z0; nothing here is claimed to be new. Per-path formulas
are validated against finite differences of the per-path smoothed price.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import special

NAMES = ("delta1", "delta2", "gamma11", "gamma22", "vega1", "vega2")
_SQ2PI = math.sqrt(2.0 * math.pi)


@dataclass(frozen=True)
class BasketAsianDigital:
    K: float
    m: int
    w1: float = 0.5
    w2: float = 0.5


@dataclass(frozen=True)
class TwoAssetParams:
    S1: float
    S2: float
    r: float
    sigma1: float
    sigma2: float
    rho: float
    T: float

    @property
    def discount(self) -> float:
        return math.exp(-self.r * self.T)


def common_loading(rho: float) -> float:
    return math.sqrt((1.0 + rho) / 2.0)


def residual_cov(m: int, dt: float, rho: float) -> np.ndarray:
    """Covariance (2m x 2m, rank 2m - 1) of X = (X^1_1..X^1_m, X^2_1..X^2_m)."""
    k = np.arange(1, m + 1)
    M = np.minimum.outer(k, k).astype(float)
    c2 = (1.0 + rho) / 2.0
    return dt * np.block([[M - c2, rho * M - c2], [rho * M - c2, M - c2]])


@lru_cache(maxsize=64)
def _pca_basis(m: int, rho: float):
    w, V = np.linalg.eigh(residual_cov(m, 1.0, rho))
    idx = np.argsort(w)[::-1][: 2 * m - 1]
    V, w = V[:, idx], np.clip(w[idx], 0.0, None)
    V = V * np.where(V.sum(axis=0) < 0, -1.0, 1.0)
    return V, w


def pca_factor(m: int, dt: float, rho: float) -> np.ndarray:
    """L (2m x (2m-1)) with L L^T = residual_cov: the path-PCA construction of the residual."""
    V, w = _pca_basis(m, float(rho))
    return math.sqrt(dt) * V * np.sqrt(w)


def _state(f: BasketAsianDigital, p: TwoAssetParams, X: np.ndarray):
    m = f.m
    dt = p.T / m
    t = dt * np.arange(1, m + 1)
    c = common_loading(p.rho)
    out = []
    for j, (x, sig, w) in enumerate(((p.S1, p.sigma1, f.w1), (p.S2, p.sigma2, f.w2))):
        Xj = X[:, j * m:(j + 1) * m]
        Rj = x * np.exp((p.r - 0.5 * sig * sig) * t + sig * Xj)
        out.append((Xj, Rj, w, sig, sig * c * math.sqrt(dt), x))
    return out, t, c, dt


def solve_root(f: BasketAsianDigital, p: TwoAssetParams, X: np.ndarray, iters: int = 60, tol: float = 1e-13) -> np.ndarray:
    """z* with a1 e^{s1 z*} + a2 e^{s2 z*} = K, per row of X."""
    st, _, _, _ = _state(f, p, X)
    with np.errstate(divide="ignore"):
        la = [np.log(w * R.mean(axis=1)) if w > 0 else np.full(X.shape[0], -np.inf) for (_, R, w, _, _, _) in st]
    s = [x[4] for x in st]
    lk = math.log(f.K)
    sbar = max(s) if min(w for (_, _, w, _, _, _) in st) <= 0 else 0.5 * (s[0] + s[1])
    z = (lk - np.logaddexp(la[0], la[1])) / sbar
    for _ in range(iters):
        l1, l2 = la[0] + s[0] * z, la[1] + s[1] * z
        tot = np.logaddexp(l1, l2)
        hp = s[0] * np.exp(l1 - tot) + s[1] * np.exp(l2 - tot)
        step = (tot - lk) / hp
        z = z - step
        if np.max(np.abs(step)) < tol:
            break
    return z


def smoothed(f: BasketAsianDigital, p: TwoAssetParams, X: np.ndarray, with_price: bool = False) -> np.ndarray:
    """Per-path conditional estimators, columns NAMES (price first if asked). X: (N, 2m) residual paths."""
    st, t, c, dt = _state(f, p, X)
    z = solve_root(f, p, X)
    D = p.discount
    phi = np.exp(-0.5 * z * z) / _SQ2PI
    a = [w * R.mean(axis=1) * np.exp(s * z) for (_, R, w, _, s, _) in st]      # a_j e^{s_j z*}; a[0] + a[1] = K
    s = [x[4] for x in st]
    Ap = s[0] * a[0] + s[1] * a[1]
    out = {}
    for j in (0, 1):
        Xj, Rj, w, sig, sj, x = st[j]
        l = 1 - j
        out[f"delta{j + 1}"] = D * phi * a[j] / (x * Ap)
        zx = -a[j] / (x * Ap)
        daj = a[j] / x + sj * a[j] * zx
        dal = s[l] * a[l] * zx
        dAp = sj * daj + s[l] * dal
        zxx = -(daj / (x * Ap) - a[j] / (x * x * Ap) - a[j] * dAp / (x * Ap * Ap))
        out[f"gamma{j + 1}{j + 1}"] = D * (z * phi * zx * zx - phi * zxx)
        dA = np.exp(sj * z) * w * (Rj * (Xj + (c * math.sqrt(dt)) * z[:, None] - sig * t)).mean(axis=1)
        out[f"vega{j + 1}"] = D * phi * dA / Ap
    cols = [out[n] for n in NAMES]
    if with_price:
        cols = [D * special.ndtr(-z)] + cols
    return np.stack(cols, axis=1)


def raw_payoff(f: BasketAsianDigital, p: TwoAssetParams, X: np.ndarray, z0: np.ndarray) -> np.ndarray:
    """Discounted digital payoff from residual paths X and the common variable Z0 (unsmoothed check)."""
    st, _, _, _ = _state(f, p, X)
    A = sum(w * R.mean(axis=1) * np.exp(s * z0) for (_, R, w, _, s, _) in st)
    return p.discount * (A >= f.K)


def fd_gradients(f: BasketAsianDigital, p: TwoAssetParams, L: np.ndarray, xi: np.ndarray, h: float = 1e-4) -> np.ndarray:
    """Central finite-difference gradients of the smoothed integrands w.r.t. xi (X = xi L^T).

    Returns (N, n_greeks, d). The integrands are smooth, so the O(h^2) error is
    negligible for estimating a gradient covariance; the rotation built from it
    never affects unbiasedness.
    """
    n, d = xi.shape
    out = np.empty((n, len(NAMES), d))
    for k in range(d):
        e = np.zeros(d)
        e[k] = h
        out[:, :, k] = (smoothed(f, p, (xi + e) @ L.T) - smoothed(f, p, (xi - e) @ L.T)) / (2 * h)
    return out

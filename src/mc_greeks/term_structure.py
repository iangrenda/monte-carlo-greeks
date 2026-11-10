"""Phase 7: deterministic, piecewise-constant volatility term structures.

Model. dS_t = r S_t dt + sigma(t) S_t dW_t with sigma(t) = sigma_j on (t_{j-1}, t_j],
t_j = j dt, dt = T/m, j = 1..m. The m pieces coincide with the Asian monitoring dates.
With a_j = sigma_j sqrt(dt) and Z_j i.i.d. N(0, 1),

    ln S_{t_i} = ln S0 + mu_i + sum_{j<=i} a_j Z_j,      mu_i = sum_{j<=i} (r - sigma_j^2/2) dt.

A forecast (Phase 7D) supplies the curve; it is fixed when the simulation starts, so the
model stays Gaussian in log space and every formula below is exact.

European digital 1{S_T > K}: ln S_T is normal with variance V = sum_j sigma_j^2 dt, so the
Black-Scholes digital formulas hold with sigma^2 T replaced by V. The likelihood-ratio Delta
weight is X / (S0 V) with X = sum_j a_j Z_j: the score of the N(ln S0 + mu_m, V) density
with respect to S0 (validated in tests/test_term_structure.py).

Arithmetic Asian digital 1{mean_i S_{t_i} >= K}, CPW-first. Every S_{t_i} contains the factor
exp(a_1 Z_1), so S_{t_i} = exp(a_1 Z_1) R_i with R_i = S0 exp(mu_i + B_i), where

    B_1 = 0,  B_i = sum_{j=2}^{i} a_j Z_j  (the residual log-noise, independent of Z_1).

Conditioning on B (Rao-Blackwell): P(A >= K | B) = Phi(delta), delta = ln(Abar/K)/a_1,
Abar = mean_i R_i. Hence, per path,

    price  D Phi(delta)
    Delta  D phi(delta) / (S0 a_1)
    Gamma  -D phi(delta) (delta + a_1) / (S0 a_1)^2
    bucket vega_k = d/d sigma_k = D phi(delta) [c_k W_k - 1{k=1} delta sqrt(dt)] / a_1,

with w_i = R_i / sum R, tail sums W_k = sum_{i>=k} w_i and
c_k = d ln R_i / d sigma_k (for i >= k) = -sigma_k dt + 1{k>=2} sqrt(dt) Z_k.
(Differentiate delta with Z held fixed: d ln Abar/d sigma_k = sum_i w_i 1{k<=i} c_k = c_k W_k,
and d a_1 / d sigma_k = 1{k=1} sqrt(dt).) The parallel vega (every sigma_j bumped by the
same h) is sum_k vega_k; for a flat curve it equals the Phase-2 CPW-first Vega.

Path constructions for (R)QMC. B is a Brownian motion run on the variance clock
tau_k = sum_{j=2}^{k} a_j^2 (k = 2..m), so Cov(B_k, B_l) = tau_{min(k,l)}. Any L with
L L^T = C gives the same law; the constructions differ only in which Sobol' coordinate
drives which direction:
* "seq":   B = cumsum(a_j xi_j);
* "bb":    Brownian bridge on the variance clock (terminal value first);
* "pca":   eigenvectors of C ("term-structure aware");
* "naive": the constant-volatility PCA of Phases 3-6 applied to the increments Z, then
           scaled by a_j. Exact in law, but its ordering ignores the term structure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import special

from .asian import AsianDigital

_SQ2PI = math.sqrt(2.0 * math.pi)
CONSTRUCTIONS = ("seq", "bb", "pca", "naive")


@dataclass(frozen=True)
class TSParams:
    """Black-Scholes with a piecewise-constant volatility curve on m equal intervals of [0, T]."""
    S0: float
    r: float
    T: float
    sigmas: tuple

    def __post_init__(self) -> None:
        object.__setattr__(self, "sigmas", tuple(float(s) for s in self.sigmas))
        if self.S0 <= 0 or self.T <= 0 or not self.sigmas or min(self.sigmas) <= 0:
            raise ValueError("need S0 > 0, T > 0 and positive volatilities")

    @classmethod
    def flat(cls, S0: float, r: float, sigma: float, T: float, m: int) -> "TSParams":
        return cls(S0, r, T, (sigma,) * m)

    @property
    def m(self) -> int:
        return len(self.sigmas)

    @property
    def dt(self) -> float:
        return self.T / self.m

    @property
    def sig(self) -> np.ndarray:
        return np.asarray(self.sigmas)

    @property
    def a(self) -> np.ndarray:
        return self.sig * math.sqrt(self.dt)

    @property
    def mu(self) -> np.ndarray:
        """mu_i = sum_{j<=i} (r - sigma_j^2/2) dt, i = 1..m."""
        return np.cumsum((self.r - 0.5 * self.sig**2) * self.dt)

    @property
    def total_variance(self) -> float:
        return float((self.sig**2).sum() * self.dt)

    @property
    def discount(self) -> float:
        return math.exp(-self.r * self.T)

    def with_sigmas(self, sigmas) -> "TSParams":
        return TSParams(self.S0, self.r, self.T, tuple(sigmas))

    def with_S0(self, S0: float) -> "TSParams":
        return TSParams(S0, self.r, self.T, self.sigmas)


def simulate_paths(p: TSParams, Z: np.ndarray) -> np.ndarray:
    """Prices at the m dates, shape (N, m)."""
    Z = np.atleast_2d(np.asarray(Z, float))
    return p.S0 * np.exp(p.mu + np.cumsum(p.a * Z, axis=1))


# --- European digital (validation and the likelihood-ratio weight) ------------------------------

def european_digital(p: TSParams, K: float) -> dict:
    """Closed forms: price, Delta, Gamma and bucket vegas (array of length m) of 1{S_T > K}."""
    V = p.total_variance
    sV = math.sqrt(V)
    c = math.log(p.S0 / K) + p.r * p.T
    d2 = (c - 0.5 * V) / sV
    D, ph = p.discount, math.exp(-0.5 * d2 * d2) / _SQ2PI
    dd2_dV = -0.5 / sV - d2 / (2.0 * V)
    return {"price": D * special.ndtr(d2), "delta": D * ph / (p.S0 * sV),
            "gamma": -D * ph * (d2 + sV) / (p.S0**2 * V),
            "bucket_vega": D * ph * dd2_dV * 2.0 * p.sig * p.dt}


def european_lr_delta_samples(p: TSParams, K: float, Z: np.ndarray) -> np.ndarray:
    """Likelihood-ratio Delta of the European digital: D 1{S_T > K} X / (S0 V), X = sum_j a_j Z_j."""
    X = Z @ p.a
    ST = p.S0 * np.exp(p.mu[-1] + X)
    return p.discount * (ST > K) * X / (p.S0 * p.total_variance)


# --- Asian digital, CPW-first --------------------------------------------------------------------

def _residual_R(p: TSParams, Z: np.ndarray) -> np.ndarray:
    """R_i = S0 exp(mu_i + B_i) with B from Z_2..Z_m (Z_1 is not used)."""
    B = np.zeros((Z.shape[0], p.m))
    B[:, 1:] = np.cumsum(p.a[1:] * Z[:, 1:], axis=1)
    return p.S0 * np.exp(p.mu + B)


def cpw_first(f: AsianDigital, p: TSParams, Z: np.ndarray, bucket: bool = False) -> dict:
    """Per-path CPW-first estimators. Returns dict of arrays (N,): price, delta, gamma, vega
    (parallel); with bucket=True also 'bucket_vega' (N, m)."""
    if f.m != p.m:
        raise ValueError("the curve must have one volatility per monitoring interval")
    Z = np.atleast_2d(np.asarray(Z, float))
    a1, dt = p.a[0], p.dt
    R = _residual_R(p, Z)
    SR = R.sum(axis=1)
    dl = np.log(SR / (p.m * f.K)) / a1
    Dphi = p.discount * np.exp(-0.5 * dl * dl) / _SQ2PI
    w = R / SR[:, None]
    W = np.cumsum(w[:, ::-1], axis=1)[:, ::-1]                       # tail sums W_k
    c = -p.sig * dt + np.concatenate([np.zeros((Z.shape[0], 1)), math.sqrt(dt) * Z[:, 1:]], axis=1)
    bv = c * W
    bv[:, 0] -= dl * math.sqrt(dt)
    bv *= (Dphi / a1)[:, None]
    out = {"price": p.discount * special.ndtr(dl), "delta": Dphi / (p.S0 * a1),
           "gamma": -Dphi * (dl + a1) / (p.S0 * a1) ** 2, "vega": bv.sum(axis=1)}
    if bucket:
        out["bucket_vega"] = bv
    return out


def smoothed_price(f: AsianDigital, p: TSParams, Z: np.ndarray) -> np.ndarray:
    """Per-path D Phi(delta) (used for finite-difference checks with Z held fixed)."""
    return cpw_first(f, p, Z)["price"]


def raw_payoff(f: AsianDigital, p: TSParams, Z: np.ndarray) -> np.ndarray:
    """Unsmoothed discounted payoff, for unbiasedness checks."""
    return p.discount * (simulate_paths(p, Z).mean(axis=1) >= f.K)


# --- path constructions --------------------------------------------------------------------------

def variance_clock(p: TSParams) -> np.ndarray:
    """tau_k = sum_{j=2}^{k} a_j^2, k = 2..m (length m-1)."""
    return np.cumsum(p.a[1:] ** 2)


def residual_cov(p: TSParams) -> np.ndarray:
    tau = variance_clock(p)
    return np.minimum.outer(tau, tau)


def bridge_matrix_times(times: np.ndarray) -> np.ndarray:
    """Brownian-bridge construction on arbitrary increasing times: terminal value first, then
    recursive index midpoints (as asian_qmc.bridge_matrix). L L^T = min(t_k, t_l) exactly."""
    times = np.asarray(times, float)
    n = len(times)
    order, queue = [(n - 1, -1, None)], [(-1, n - 1)]
    while queue:
        nxt = []
        for lo, hi in queue:
            if hi - lo > 1:
                c = (lo + hi) // 2
                order.append((c, lo, hi))
                nxt += [(lo, c), (c, hi)]
        queue = nxt
    L = np.zeros((n, n))
    for j in range(n):
        path = np.zeros(n)
        for k, (tgt, lo, hi) in enumerate(order):
            x = 1.0 if k == j else 0.0
            if hi is None:
                path[tgt] = math.sqrt(times[tgt]) * x
                continue
            ta, Ba = (0.0, 0.0) if lo < 0 else (times[lo], path[lo])
            tb, Bb, tc = times[hi], path[hi], times[tgt]
            path[tgt] = ((tb - tc) * Ba + (tc - ta) * Bb) / (tb - ta) + math.sqrt((tc - ta) * (tb - tc) / (tb - ta)) * x
        L[:, j] = path
    return L


def construction_matrix(kind: str, p: TSParams) -> np.ndarray:
    """L ((m-1) x (m-1)) with L L^T = residual_cov(p): B_{2..m} = L xi, xi ~ N(0, I)."""
    n, a = p.m - 1, p.a[1:]
    if kind == "seq":
        return np.tril(np.ones((n, n))) * a
    if kind == "bb":
        return bridge_matrix_times(variance_clock(p))
    if kind == "pca":
        w, V = np.linalg.eigh(residual_cov(p))
        idx = np.argsort(w)[::-1]
        V = V[:, idx] * np.where(V[:, idx].sum(axis=0) < 0, -1.0, 1.0)
        return V * np.sqrt(np.clip(w[idx], 0.0, None))
    if kind == "naive":
        k = np.arange(1, n + 1)
        wf, Vf = np.linalg.eigh(np.minimum.outer(k, k).astype(float))   # unit-step Brownian motion
        idx = np.argsort(wf)[::-1]
        Lf = Vf[:, idx] * np.sqrt(np.clip(wf[idx], 0.0, None))
        Dinc = np.diff(np.vstack([np.zeros((1, n)), Lf]), axis=0)        # increments ~ N(0, I) in law
        return np.tril(np.ones((n, n))) @ (a[:, None] * Dinc)
    raise ValueError(f"unknown construction {kind!r}")


def increments_from_residual(B: np.ndarray, p: TSParams) -> np.ndarray:
    """Residual log-noise B (N, m-1) -> standard normals Z (N, m) with Z_1 = 0 (unused)."""
    N = B.shape[0]
    full = np.concatenate([np.zeros((N, 1)), B], axis=1)
    Z = np.zeros_like(full)
    Z[:, 1:] = np.diff(full, axis=1) / p.a[1:]
    return Z


# --- Greek-aware rotations for a volatility curve (Phases 4-6 generalised) ----------------------
#
# Coordinates: B_{2..m} = L xi with L = construction_matrix("pca", p); a rotation U gives L U.
# The joint objective is the trace-normalised average of the gradient covariances of the
# per-path Delta, Gamma and parallel Vega (as in Phases 4-6).

JOINT = ("delta", "gamma", "vega")


def per_path_vector(f: AsianDigital, p: TSParams, xi: np.ndarray, L: np.ndarray) -> np.ndarray:
    """(N, 3) per-path (Delta, Gamma, Vega) at B = L xi."""
    out = cpw_first(f, p, increments_from_residual(xi @ L.T, p))
    return np.stack([out[g] for g in JOINT], axis=1)


def gradient_rotation(f: AsianDigital, p: TSParams, n_train: int, rng: np.random.Generator,
                      h: float = 1e-4) -> np.ndarray:
    """Joint gradient-PCA rotation (in PCA coordinates) from n_train paths; gradients by central
    differences in xi (the integrands are smooth; the rotation never affects unbiasedness)."""
    L = construction_matrix("pca", p)
    d = L.shape[1]
    xi = rng.standard_normal((n_train, d))
    G = np.zeros((3, d, d))
    grads = np.empty((n_train, 3, d))
    for k in range(d):
        e = np.zeros(d)
        e[k] = h
        grads[:, :, k] = (per_path_vector(f, p, xi + e, L) - per_path_vector(f, p, xi - e, L)) / (2 * h)
    for g in range(3):
        G[g] = grads[:, g, :].T @ grads[:, g, :] / n_train
    Gj = sum(G[g] / np.trace(G[g]) for g in range(3)) / 3
    w, U = np.linalg.eigh(0.5 * (Gj + Gj.T))
    return U[:, np.argsort(w)[::-1]]


def _state(f: AsianDigital, p: TSParams, B: np.ndarray):
    """R_i (i = 1..m), Abar, weights w_i = R_i / sum R, for one residual log-noise vector B (m-1,)."""
    R = p.S0 * np.exp(p.mu + np.concatenate([[0.0], B]))
    return R, R.mean(), R / R.sum()


def design_point(f: AsianDigital, p: TSParams, iters: int = 100, tol: float = 1e-12) -> np.ndarray:
    """B* = argmin 0.5 B^T C^-1 B subject to ln Abar(B) = ln K (HL-RF iteration; grad ln Abar = w_{2..m})."""
    C = residual_cov(p)
    B = np.zeros(p.m - 1)
    for _ in range(iters):
        _, Abar, w = _state(f, p, B)
        g, a = math.log(Abar / f.K), w[1:]
        Ca = C @ a
        Bn = (a @ B - g) / (a @ Ca) * Ca
        done = np.max(np.abs(Bn - B)) < tol
        B = Bn
        if done:
            break
    return B


def weighted_rotation(f: AsianDigital, p: TSParams, at: str = "dominant", nodes: int = 80) -> np.ndarray:
    """Training-free rotation (Phase 6B 'weighted-dominant', on the variance clock), PCA coordinates.

    grad_xi ln Abar ~= a + K (xi - xi0), a = L^T w(B0), K = L^T (diag(w) - w w^T)(B0) L; delta is
    linear along a_hat; E over the other directions in closed form; the Greek weights
    h_g'(delta)^2 by Gauss-Hermite quadrature; Vega's path term frozen at B0."""
    L = construction_matrix("pca", p)
    lam = np.einsum("ij,ij->j", L, L)                      # L = V sqrt(lambda): column norms^2 = lambda
    V = L / np.sqrt(lam)
    B0 = np.zeros(p.m - 1) if at == "origin" else design_point(f, p)
    xi0 = (V.T @ B0) / np.sqrt(lam)
    R, Abar, w = _state(f, p, B0)
    a = L.T @ w[1:]
    Hs = np.diag(w[1:]) - np.outer(w[1:], w[1:])
    K = L.T @ Hs @ L
    c = a - K @ xi0
    na = np.linalg.norm(a)
    ah = a / na
    Ka = K @ ah
    a1, dt = p.a[0], p.dt
    delta0 = math.log(Abar / f.K) / a1
    t0 = float(ah @ xi0)
    # Vega's path term ell = sum_k c_k W_k at B0 (Z_k = increments of B0 / a_k)
    Z0 = np.concatenate([[0.0], np.diff(np.concatenate([[0.0], B0])) / p.a[1:]])
    Wt = np.cumsum(w[::-1])[::-1]
    ell = float(((-p.sig * dt + np.concatenate([[0.0], math.sqrt(dt) * Z0[1:]])) * Wt).sum())
    phi = lambda d: np.exp(-0.5 * d * d) / _SQ2PI
    hp = {"delta": lambda d: -d * phi(d), "gamma": lambda d: phi(d) * (d * (d + a1) - 1.0),
          "vega": lambda d: -d * phi(d) * (ell - d * math.sqrt(dt)) - phi(d) * math.sqrt(dt)}
    x, wq = np.polynomial.hermite_e.hermegauss(nodes)
    wq = wq / wq.sum()
    dl = delta0 + (na / a1) * (x - t0)
    cc, cK, KK, kk = np.outer(c, c), np.outer(c, Ka), K @ K.T, np.outer(Ka, Ka)
    cK = cK + cK.T
    G = np.zeros((len(a), len(a)))
    for h in hp.values():
        ww = h(dl) ** 2
        E0, E1, E2 = wq @ ww, wq @ (ww * x), wq @ (ww * x * x)
        Gg = E0 * (cc + KK - kk) + E1 * cK + E2 * kk
        G += Gg / np.trace(Gg)
    ev, U = np.linalg.eigh(0.5 * (G + G.T))
    U = U[:, np.argsort(ev)[::-1]]
    return U * np.where(U[0] < 0, -1.0, 1.0)

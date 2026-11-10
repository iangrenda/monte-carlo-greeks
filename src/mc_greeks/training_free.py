"""Phase 6B: training-free (deterministic) joint rotations for the CPW-first Greeks.

Phase 5 found the gradient covariance G_g = E[grad f_g grad f_g^T] of every Greek to be
almost exactly rank one, with the same top eigenvector u* for Delta, Gamma and Vega. The
reason (PHASE5_MATHEMATICS.md, section 5): every CPW-first Greek is h_g(delta(B)), so

    grad_B f_g = h_g'(delta) grad_B delta,    (grad_B delta)_i = sigma R_i / (m Abar s),

a random scalar times a vector of positive weights R_i / (m Abar) that varies little from
path to path. If grad_B delta were the same vector v on every path, G_g would be exactly
c_g (L0^T v)(L0^T v)^T for every Greek, and u* = L0^T v / ||L0^T v|| with no training.

This module replaces E[...] by one deterministic evaluation of grad_B delta:

* "ones":     v_i = 1. Independent of the contract: the gradient of the arithmetic mean
              of the residual path (and the limit sigma -> 0, r -> 0 of the others).
* "origin":   v = grad_B delta at B = 0 (the median path): v_i = exp((r - sigma^2/2) t_i).
* "dominant": v = grad_B delta at the most likely path on the exercise boundary,
              B* = argmin_B 0.5 B^T C^-1 B  subject to  Abar(B) = K
              (the "design point" of structural reliability; Hasofer-Lind-Rackwitz-Fiessler
              iteration). The Greeks' integrands are concentrated near delta = 0, i.e. near
              that boundary, so this is where the gradient matters most.

The rotation is completed by the Householder reflection H that maps the first PCA
coordinate e_1 to u (in PCA coordinates): H = I - 2 w w^T, w = (e_1 - u)/||e_1 - u||.
H is orthogonal and symmetric, so L = L_pca H still has L L^T = C (unbiased for any u),
and the remaining coordinates stay as close as possible to the PCA directions.

None of this is new as a method: using the integrand's gradient at a chosen point to
build the first column of an orthogonal transform is the linear-transformation (LT)
method of Imai & Tan (2006); the design point is classical in reliability analysis and
in importance sampling for options (e.g. Glasserman, Heidelberger & Shahabuddin 1999).
What Phase 6B tests is whether such a deterministic rotation can replace the trained
joint gradient-PCA rotation for the vector (Delta, Gamma, Vega).
"""

from __future__ import annotations

import math
import time

import numpy as np

from .asian import AsianDigital
from .asian_qmc import residual_cov
from .joint_greeks import pca_basis, pca_factor
from .models import BlackScholesParams

KINDS = ("ones", "origin", "dominant")          # first-order: one direction + Householder
TAYLOR_KINDS = ("taylor-origin", "taylor-dominant", "weighted-origin", "weighted-dominant")  # all directions
ALL_KINDS = KINDS + TAYLOR_KINDS


def householder_to(u: np.ndarray) -> np.ndarray:
    """Orthogonal symmetric H with H e_1 = u (u a unit vector; its sign is fixed so u[0] >= 0)."""
    u = np.asarray(u, float)
    u = u / np.linalg.norm(u)
    if u[0] < 0:
        u = -u
    n = len(u)
    w = -u.copy()
    w[0] += 1.0
    nw = np.linalg.norm(w)
    if nw < 1e-14:
        return np.eye(n)
    w /= nw
    return np.eye(n) - 2.0 * np.outer(w, w)


def _residual_state(f: AsianDigital, p: BlackScholesParams, B: np.ndarray):
    m, dt = f.m, p.T / f.m
    t = dt * np.arange(1, m + 1)
    Bf = np.concatenate([[0.0], B])
    R = p.S0 * np.exp((p.r - 0.5 * p.sigma**2) * t + p.sigma * Bf)
    return R, R.mean()


def grad_log_abar(f: AsianDigital, p: BlackScholesParams, B: np.ndarray) -> np.ndarray:
    """d ln Abar / d B_i, i = 2..m (proportional to d delta / d B_i)."""
    R, Abar = _residual_state(f, p, B)
    return p.sigma * R[1:] / (f.m * Abar)


def dominant_point(f: AsianDigital, p: BlackScholesParams, iters: int = 100, tol: float = 1e-12) -> np.ndarray:
    """B* = argmin 0.5 B^T C^-1 B subject to ln Abar(B) = ln K (HL-RF iteration).

    In whitened coordinates y (B = L y) the HL-RF step is y <- (a^T y - g) a / |a|^2 with
    a = L^T grad g; mapped back, B <- ((grad g)^T B - g) / ((grad g)^T C grad g) * C grad g.
    """
    m, dt = f.m, p.T / f.m
    C = residual_cov(m, dt)
    B = np.zeros(m - 1)
    for _ in range(iters):
        R, Abar = _residual_state(f, p, B)
        g = math.log(Abar / f.K)
        a = grad_log_abar(f, p, B)
        Ca = C @ a
        Bn = (a @ B - g) / (a @ Ca) * Ca
        if np.max(np.abs(Bn - B)) < tol:
            B = Bn
            break
        B = Bn
    return B


def direction_B(kind: str, f: AsianDigital, p: BlackScholesParams) -> np.ndarray:
    """Deterministic surrogate v for grad_B delta (shape (m-1,), any positive scale)."""
    m = f.m
    if kind == "ones":
        return np.ones(m - 1)
    if kind == "origin":
        return grad_log_abar(f, p, np.zeros(m - 1))
    if kind == "dominant":
        return grad_log_abar(f, p, dominant_point(f, p))
    raise ValueError(f"unknown kind {kind!r}")


def first_direction(kind: str, f: AsianDigital, p: BlackScholesParams) -> np.ndarray:
    """Unit vector u = L_pca^T v / ||L_pca^T v|| in PCA coordinates."""
    u = pca_factor(f.m, p.T / f.m).T @ direction_B(kind, f, p)
    return u / np.linalg.norm(u)


def pca_coordinates(f: AsianDigital, p: BlackScholesParams, B: np.ndarray) -> np.ndarray:
    """xi = L_pca^{-1} B = Lambda^{-1/2} V^T B / sqrt(dt) (exact; V orthogonal)."""
    V, w = pca_basis(f.m)
    return (V.T @ B) / (math.sqrt(p.T / f.m) * np.sqrt(w))


def hessian_log_abar(f: AsianDigital, p: BlackScholesParams, B: np.ndarray) -> np.ndarray:
    """d^2 ln Abar / dB_i dB_j = sigma^2 (diag(w) - w w^T), w_i = R_i / sum_k R_k (i, j = 2..m)."""
    R, _ = _residual_state(f, p, B)
    w = R[1:] / R.sum()
    return p.sigma**2 * (np.diag(w) - np.outer(w, w))


def taylor_gradient_covariance(f: AsianDigital, p: BlackScholesParams, at: str = "origin") -> np.ndarray:
    """Deterministic second-order surrogate for the joint gradient covariance, in PCA coordinates.

    Linearise the gradient of ln Abar around a point B0 = L xi0 (L = L_pca):
        grad_xi ln Abar(L xi) ~= a + K (xi - xi0),  a = L^T g(B0),  K = L^T H(B0) L,
    and take the expectation over xi ~ N(0, I):
        G_taylor = (a - K xi0)(a - K xi0)^T + K K^T.
    Every CPW-first Greek is h_g(delta) with delta = (ln Abar - ln K)/s, so this is the
    common gradient covariance up to the weight h_g'(delta)^2, which is ignored."""
    L = pca_factor(f.m, p.T / f.m)
    B0 = np.zeros(f.m - 1) if at == "origin" else dominant_point(f, p)
    xi0 = pca_coordinates(f, p, B0)
    a = L.T @ grad_log_abar(f, p, B0)
    K = L.T @ hessian_log_abar(f, p, B0) @ L
    c = a - K @ xi0
    return np.outer(c, c) + K @ K.T


def _greek_weight_derivatives(f: AsianDigital, p: BlackScholesParams, B0: np.ndarray):
    """h_g'(delta) for the three CPW-first Greeks as functions of delta, up to constant factors
    (constants cancel after trace normalisation). Vega also depends on the path through
    ell = d ln Abar / d sigma; it is frozen at its value at B0."""
    m, dt = f.m, p.T / f.m
    s = p.sigma * math.sqrt(dt)
    t = dt * np.arange(1, m + 1)
    R, _ = _residual_state(f, p, B0)
    Bf = np.concatenate([[0.0], B0])
    ell = float((R * (Bf - p.sigma * t)).sum() / R.sum())
    phi = lambda d: np.exp(-0.5 * d * d) / math.sqrt(2 * math.pi)
    return {
        "delta": lambda d: -d * phi(d),
        "gamma": lambda d: phi(d) * (d * (d + s) - 1.0),
        "vega": lambda d: -d * phi(d) * (ell / s - d / p.sigma) - phi(d) / p.sigma,
    }


def weighted_gradient_covariance(f: AsianDigital, p: BlackScholesParams, at: str = "origin",
                                 nodes: int = 80) -> np.ndarray:
    """Deterministic surrogate for the trace-normalised joint gradient covariance (PCA coordinates).

    With the linearisation grad_xi delta ~= (c + K xi)/s, c = a - K xi0 (as in
    taylor_gradient_covariance) and delta ~= delta_lin(t) = delta0 + (|a|/s)(t - t0) along
    t = a_hat^T xi ~ N(0, 1), write xi = t a_hat + eta with eta independent of t. Then for
    w_g(t) = h_g'(delta_lin(t))^2:
        E[w_g (c + K xi)(c + K xi)^T] = E[w] c c^T + E[w t](c (K a_hat)^T + (K a_hat) c^T)
                                        + K (E[w t^2] a_hat a_hat^T + E[w] (I - a_hat a_hat^T)) K^T.
    The three one-dimensional Gaussian expectations are computed by Gauss-Hermite quadrature.
    Each Greek's matrix is divided by its trace and the three are averaged (as in Phase 4/5)."""
    L = pca_factor(f.m, p.T / f.m)
    s = p.sigma * math.sqrt(p.T / f.m)
    B0 = np.zeros(f.m - 1) if at == "origin" else dominant_point(f, p)
    xi0 = pca_coordinates(f, p, B0)
    a = L.T @ grad_log_abar(f, p, B0)
    K = L.T @ hessian_log_abar(f, p, B0) @ L
    c = a - K @ xi0
    na = np.linalg.norm(a)
    ah = a / na
    Ka = K @ ah
    _, Abar = _residual_state(f, p, B0)
    delta0 = math.log(Abar / f.K) / s
    t0 = float(ah @ xi0)
    x, wq = np.polynomial.hermite_e.hermegauss(nodes)
    wq = wq / wq.sum()
    dl = delta0 + (na / s) * (x - t0)
    cc, cK, KK, kk = np.outer(c, c), np.outer(c, Ka), K @ K.T, np.outer(Ka, Ka)
    cK = cK + cK.T
    G = np.zeros((len(a), len(a)))
    for h in _greek_weight_derivatives(f, p, B0).values():
        w = h(dl) ** 2
        E0, E1, E2 = wq @ w, wq @ (w * x), wq @ (w * x * x)
        Gg = E0 * (cc + KK - kk) + E1 * cK + E2 * kk       # K (E2 a a^T + E0 (I - a a^T)) K^T expanded
        G += Gg / np.trace(Gg)
    return G / 3.0


def training_free_rotation(kind: str, f: AsianDigital, p: BlackScholesParams) -> tuple[np.ndarray, float]:
    """(U, seconds): the deterministic rotation for ``kind`` and its measured construction time
    (the PCA basis is cached, as for the trained rotations)."""
    pca_factor(f.m, p.T / f.m)
    t0 = time.perf_counter()
    if kind in TAYLOR_KINDS:
        at = kind.split("-", 1)[1]
        G = taylor_gradient_covariance(f, p, at) if kind.startswith("taylor") else weighted_gradient_covariance(f, p, at)
        w, U = np.linalg.eigh(0.5 * (G + G.T))
        U = U[:, np.argsort(w)[::-1]]
        U = U * np.where(U[0] < 0, -1.0, 1.0)
    else:
        U = householder_to(first_direction(kind, f, p))
    return U, time.perf_counter() - t0

"""Phase 4, Part C: Greek-aware (gradient-PCA) orthogonal transforms for RQMC.

Setting (see docs/mathematical_derivations.md, Part IV). CPW-first (level
variable Z_1 integrated out exactly) is a smooth function psi_g(B) of the residual
Brownian path B in R^n (n = m - 1), with B = L0 xi, xi ~ N(0, I_n), L0 L0^T = C.
In the xi coordinates the integrand is f_g(xi) = psi_g(L0 xi), with gradient

    grad_xi f_g = L0^T grad_B psi_g(B).

Gradient covariance ("gradient PCA", Xiao & Wang 2017/2018; active subspaces,
Constantine 2015; Liu & Owen 2023):

    G_g = E[grad_xi f_g grad_xi f_g^T]  (n x n, symmetric PSD),
    G_g = U Lambda U^T  (eigenvalues decreasing).

The transform L_g = L0 U satisfies L_g L_g^T = L0 U U^T L0^T = C exactly, so
xi' ~ N(0, I) mapped through L_g has the same law as before. Only the
assignment of Sobol coordinates to directions changes: the first coordinate
now drives the direction of largest mean-squared sensitivity of f_g.
L_g does not depend on the choice of L0 (up to eigenvector signs/ties): if
L0' = L0 Q with Q orthogonal, then G' = Q^T G Q and L0' U' = L0 U.

Joint transform: G_joint = sum_g w_g G_g / tr(G_g), with w_g = 1/3. Trace
normalisation makes each Greek contribute unit total gradient energy, so the
result does not depend on the Greeks' units. This is a heuristic proxy; it
does not minimise any RQMC error criterion directly.

All gradients are analytic and validated against finite differences in
tests/test_greek_aware_qmc.py.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from scipy import special, stats
from scipy.stats import qmc

from .asian import AsianDigital, cpw_first_samples
from .asian_qmc import TimedEstimate, construction_matrix, increments_from_residual, mc_block, residual_cov
from .models import BlackScholesParams

GREEKS = ("delta", "gamma", "vega")
_SQ2PI = math.sqrt(2.0 * math.pi)


# --- analytic gradient of the CPW-first integrand w.r.t. the residual path ------------------

def cpw_first_and_grad(f: AsianDigital, p: BlackScholesParams, B: np.ndarray, greek: str):
    """CPW-first per-path value psi_g and its gradient w.r.t. B = (B_2..B_m).

    B has shape (N, m-1). Returns (value (N,), grad (N, m-1)). Notation:
    R_i = x exp((r - sigma^2/2) t_i + sigma B_i) with B_1 = 0, Abar = mean_i R_i,
    s = sigma sqrt(dt), delta = ln(Abar/K)/s, d delta/dB_i = sigma R_i / (m Abar s).
    """
    m = f.m
    dt = p.T / m
    s = p.sigma * math.sqrt(dt)
    x, sig, D = p.S0, p.sigma, p.discount
    t = dt * np.arange(1, m + 1)
    N = B.shape[0]
    Bf = np.concatenate([np.zeros((N, 1)), B], axis=1)          # (N, m), B_1 = 0
    R = x * np.exp((p.r - 0.5 * sig**2) * t + sig * Bf)
    SR = R.sum(axis=1)
    Abar = SR / m
    dl = np.log(Abar / f.K) / s
    phi = np.exp(-0.5 * dl * dl) / _SQ2PI
    ddl = (sig * R[:, 1:] / (m * Abar * s)[:, None])           # d delta / dB_i, i = 2..m
    if greek == "price":
        return D * special.ndtr(dl), (D * phi)[:, None] * ddl
    if greek == "delta":
        v = D * phi / (x * s)
        return v, (-D * dl * phi / (x * s))[:, None] * ddl
    if greek == "gamma":
        v = -D * phi * (dl + s) / (x * s) ** 2
        dv = D * phi * (dl * (dl + s) - 1.0) / (x * s) ** 2
        return v, dv[:, None] * ddl
    if greek == "vega":
        u = Bf - sig * t
        ell = (R * u).sum(axis=1) / SR
        bracket = ell / s - dl / sig
        v = D * phi * bracket
        dell = R[:, 1:] * (sig * (u[:, 1:] - ell[:, None]) + 1.0) / SR[:, None]
        grad = D * ((-dl * phi * bracket)[:, None] * ddl + phi[:, None] * (dell / s - ddl / sig))
        return v, grad
    raise ValueError(f"unknown Greek {greek!r}")


# --- gradient covariance and transforms ------------------------------------------------------

def gradient_covariance(f: AsianDigital, p: BlackScholesParams, greek: str, L0: np.ndarray, n_train: int,
                        rng: np.random.Generator, block: int = 4096) -> np.ndarray:
    """MC estimate of G_g = E[grad_xi f grad_xi f^T] in the coordinates of L0 (symmetric PSD)."""
    n = L0.shape[1]
    G = np.zeros((n, n))
    done = 0
    while done < n_train:
        k = min(block, n_train - done)
        xi = rng.standard_normal((k, n))
        _, gB = cpw_first_and_grad(f, p, xi @ L0.T, greek)
        gx = gB @ L0                      # rows: (L0^T grad_B)^T
        G += gx.T @ gx
        done += k
    return G / n_train


def rotation_from(G: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvectors U (columns, decreasing eigenvalues) and eigenvalues of symmetric G."""
    w, U = np.linalg.eigh(0.5 * (G + G.T))
    idx = np.argsort(w)[::-1]
    return U[:, idx], w[idx]


@dataclass
class Transform:
    name: str
    L: np.ndarray
    setup_seconds: float
    eigenvalues: np.ndarray | None = None


def build_transform(kind: str, f: AsianDigital, p: BlackScholesParams, n_train: int = 1024,
                    rng: np.random.Generator | None = None, base: str = "pca") -> Transform:
    """kind: 'seq' | 'bb' | 'pca' | 'gpca-delta' | 'gpca-gamma' | 'gpca-vega' | 'gpca-joint'.

    setup_seconds is the measured wall time of the whole construction
    (covariance factor, gradient sample, eigendecomposition).
    """
    m, dt = f.m, p.T / f.m
    t0 = time.perf_counter()
    if kind in ("seq", "bb", "pca"):
        L = construction_matrix(kind, m, dt)
        return Transform(kind, L, time.perf_counter() - t0)
    L0 = construction_matrix(base, m, dt)
    if kind == "gpca-joint":
        Gs = [gradient_covariance(f, p, g, L0, n_train, rng) for g in GREEKS]
        G = sum(Gi / max(np.trace(Gi), 1e-300) for Gi in Gs) / len(GREEKS)
    elif kind.startswith("gpca-"):
        G = gradient_covariance(f, p, kind.split("-", 1)[1], L0, n_train, rng)
    else:
        raise ValueError(f"unknown transform {kind!r}")
    U, w = rotation_from(G)
    return Transform(kind, L0 @ U, time.perf_counter() - t0, w)


def capture_curve(L: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Fraction of total gradient energy in the first k coordinates, k = 1..n.

    H = E[grad_B psi grad_B psi^T] (estimated on an independent sample). In the
    coordinates of L the gradient covariance is L^T H L; its diagonal gives
    E[(d f / d xi_k)^2] (a derivative-based global sensitivity measure, cf.
    Sobol' & Kucherenko 2009). This is a proxy for effective dimension: it
    bounds variance-based (Sobol') indices only up to Poincare-type constants.
    """
    d = np.einsum("ij,jk,ki->i", L.T, H, L)
    return np.cumsum(d) / d.sum()


def path_gradient_cov(f: AsianDigital, p: BlackScholesParams, greek: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """H = E[grad_B psi grad_B psi^T] with B ~ N(0, C) (coordinate-free)."""
    L = construction_matrix("seq", f.m, p.T / f.m)
    xi = rng.standard_normal((n, f.m - 1))
    _, gB = cpw_first_and_grad(f, p, xi @ L.T, greek)
    return gB.T @ gB / n


# --- RQMC with an explicit transform ---------------------------------------------------------

def rqmc_fixed_n(f: AsianDigital, p: BlackScholesParams, L: np.ndarray, n_points: int, n_scrambles: int,
                 rng: np.random.Generator, greeks=GREEKS) -> dict:
    """Scramble means of every Greek from the SAME scrambled points (fixed sample size)."""
    m, dt = f.m, p.T / f.m
    eps = np.finfo(float).eps
    out = {g: [] for g in greeks}
    for _ in range(n_scrambles):
        sob = qmc.Sobol(d=m - 1, scramble=True, seed=rng)
        xi = special.ndtri(np.clip(sob.random(n_points), eps, 1 - eps))
        Z = increments_from_residual(xi @ L.T, dt)
        for g in greeks:
            out[g].append(float(cpw_first_samples(f, p, Z, g).mean()))
    return {g: np.array(v) for g, v in out.items()}


def timed_with_transform(greek: str, f: AsianDigital, p: BlackScholesParams, budget: float, rng: np.random.Generator,
                         L: np.ndarray, points_per_scramble: int, min_units: int = 4) -> TimedEstimate:
    """Wall-clock-budgeted RQMC with a given construction matrix (cf. asian_qmc.timed_estimate)."""
    m, dt = f.m, p.T / f.m
    eps = np.finfo(float).eps
    t0 = time.perf_counter()
    rows = (1 << 18) // max(m - 1, 1)
    blk = max(64, min(points_per_scramble, 1 << (rows.bit_length() - 1)))
    means = []
    while len(means) < min_units or time.perf_counter() - t0 < budget:
        sob = qmc.Sobol(d=m - 1, scramble=True, seed=rng)
        acc, cnt = 0.0, 0
        while cnt < points_per_scramble:
            k = min(blk, points_per_scramble - cnt)
            xi = special.ndtri(np.clip(sob.random(k), eps, 1 - eps))
            acc += float(cpw_first_samples(f, p, increments_from_residual(xi @ L.T, dt), greek).sum())
            cnt += k
        means.append(acc / cnt)
    v = np.array(means)
    R = len(v)
    se = float(v.std(ddof=1) / math.sqrt(R))
    return TimedEstimate(float(v.mean()), se, R * points_per_scramble, R, time.perf_counter() - t0,
                         float(stats.t.ppf(0.975, R - 1)) * se)

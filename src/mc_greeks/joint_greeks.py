"""Phase 5C/5D: Delta, Gamma and Vega as one risk vector from one simulation.

``cpw_first_joint`` evaluates the three CPW-first Greeks on the same paths. The
expensive part of a path (exponentials, the average Abar, delta = ln(Abar/K)/s
and the normal density) is shared, so the vector costs little more than one Greek.

Rotations. Every transform used here has the form L = L_pca(m, dt) @ U with U
orthogonal, where L_pca = sqrt(dt) V Lambda^{1/2} and (V, Lambda) is the
eigen-decomposition of the matrix min(k, l), k, l = 1..m-1. V and Lambda do
not depend on dt, sigma, K or T, so a rotation U trained on one contract can be
applied to another contract with the same m (transfer, Phase 5D): L L^T is the
correct residual covariance for the new dt whatever U is, hence the estimator
stays unbiased; only its variance depends on how well U fits.

Joint rotation: U = eigenvectors of G_w = sum_g w_g G_g / tr(G_g), where
G_g = E[grad f_g grad f_g^T] in the PCA coordinates (Phase 4 used w = 1/3 each).

Risk-vector loss: L = sum_g w_g MSE_g / s_g^2 with scales s_g fixed in advance.
We use s_g = per-path standard deviation of the plain-MC CPW-first estimator of
Greek g, so MSE_g / s_g^2 = 1 / (number of plain-MC paths that would give the
same MSE). It is unit-free and never divides by a Greek that is near zero.
"""

from __future__ import annotations

import math
import time
from functools import lru_cache

import numpy as np
from scipy import special

from .asian import AsianDigital, monitoring_times
from .greek_aware_qmc import cpw_first_and_grad
from .models import BlackScholesParams

GREEKS = ("delta", "gamma", "vega")
_SQ2PI = math.sqrt(2.0 * math.pi)


def cpw_first_joint(f: AsianDigital, p: BlackScholesParams, Z: np.ndarray, with_price: bool = False) -> np.ndarray:
    """CPW-first (Delta, Gamma, Vega) per path, shape (N, 3); (N, 4) with the price first if asked.

    Same formulas as ``asian.cpw_first_samples`` (which stays the single-Greek
    reference implementation); Z[:, 0] is not used.
    """
    m = f.m
    dt = p.T / m
    s = p.sigma * math.sqrt(dt)
    t = monitoring_times(p, m)
    Wt = math.sqrt(dt) * (np.cumsum(Z, axis=1) - Z[:, :1])
    R = p.S0 * np.exp((p.r - 0.5 * p.sigma**2) * t + p.sigma * Wt)
    SR = R.sum(axis=1)
    dl = np.log(SR / (m * f.K)) / s
    Dpdf = p.discount * np.exp(-0.5 * dl * dl) / _SQ2PI
    xs = p.S0 * s
    out = np.empty((Z.shape[0], 4 if with_price else 3))
    k = 1 if with_price else 0
    if with_price:
        out[:, 0] = p.discount * special.ndtr(dl)
    out[:, k] = Dpdf / xs
    out[:, k + 1] = -Dpdf * (dl + s) / (xs * xs)
    dlogA = (R * (Wt - p.sigma * t)).sum(axis=1) / SR
    out[:, k + 2] = Dpdf * (dlogA / s - dl / p.sigma)
    return out


# --- PCA basis and rotations ----------------------------------------------------------------

@lru_cache(maxsize=64)
def pca_basis(m: int) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvectors V (columns, decreasing eigenvalue, sign fixed so each column sums >= 0)
    and eigenvalues of the matrix min(k, l), k, l = 1..m-1. Independent of dt."""
    k = np.arange(1, m)
    w, V = np.linalg.eigh(np.minimum.outer(k, k).astype(float))
    idx = np.argsort(w)[::-1]
    V, w = V[:, idx], np.clip(w[idx], 0.0, None)
    V = V * np.where(V.sum(axis=0) < 0, -1.0, 1.0)
    V.setflags(write=False)
    w.setflags(write=False)
    return V, w


def pca_factor(m: int, dt: float) -> np.ndarray:
    """L_pca with L L^T = dt * min(k, l): the path-PCA construction, with a dt-independent basis."""
    V, w = pca_basis(m)
    return math.sqrt(dt) * V * np.sqrt(w)


def rotated_factor(U: np.ndarray | None, m: int, dt: float) -> np.ndarray:
    """Construction matrix L_pca(m, dt) @ U (U = None gives path PCA)."""
    L = pca_factor(m, dt)
    return L if U is None else L @ U


def gradient_covariances(f: AsianDigital, p: BlackScholesParams, n_train: int, rng: np.random.Generator,
                         greeks=GREEKS) -> dict[str, np.ndarray]:
    """G_g = E[grad_xi f_g grad_xi f_g^T] in PCA coordinates, all Greeks from the same n_train paths."""
    L0 = pca_factor(f.m, p.T / f.m)
    xi = rng.standard_normal((n_train, f.m - 1))
    B = xi @ L0.T
    out = {}
    for g in greeks:
        _, gB = cpw_first_and_grad(f, p, B, g)
        gx = gB @ L0
        out[g] = gx.T @ gx / n_train
    return out


def joint_rotation(Gs: dict[str, np.ndarray], weights: dict[str, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Orthogonal U and eigenvalues of G_w = sum_g w_g G_g / tr(G_g) (weights default to equal)."""
    keys = list(Gs)
    w = {g: 1.0 / len(keys) for g in keys} if weights is None else weights
    G = sum(w[g] * Gs[g] / max(float(np.trace(Gs[g])), 1e-300) for g in keys if w.get(g, 0.0) != 0.0)
    ev, U = np.linalg.eigh(0.5 * (G + G.T))
    idx = np.argsort(ev)[::-1]
    return U[:, idx], ev[idx]


def train_rotation(f: AsianDigital, p: BlackScholesParams, n_train: int, rng: np.random.Generator,
                   weights: dict[str, float] | None = None, greeks=GREEKS):
    """Train one rotation. Returns (U, seconds, Gs); seconds is the measured wall time of
    sampling, gradients and the eigen-decomposition (the PCA basis is cached and not charged)."""
    pca_basis(f.m)
    t0 = time.perf_counter()
    Gs = gradient_covariances(f, p, n_train, rng, greeks)
    U, _ = joint_rotation(Gs, weights)
    return U, time.perf_counter() - t0, Gs


def top_alignment(U1: np.ndarray, U2: np.ndarray, k: int = 1) -> float:
    """Smallest singular value of U1[:, :k]^T U2[:, :k]: the cosine of the largest principal
    angle between the two leading k-dimensional subspaces (1 = identical, 0 = orthogonal)."""
    return float(np.linalg.svd(U1[:, :k].T @ U2[:, :k], compute_uv=False).min())


def effective_rank(G: np.ndarray) -> dict:
    """Spectrum summaries of a PSD matrix: top-eigenvalue share and participation ratio."""
    ev = np.clip(np.linalg.eigvalsh(0.5 * (G + G.T)), 0.0, None)
    tot = ev.sum()
    return {"top_share": float(ev.max() / tot), "participation_ratio": float(tot**2 / (ev**2).sum())}


# --- risk-vector loss -------------------------------------------------------------------------

def risk_scales(f: AsianDigital, p: BlackScholesParams, rng: np.random.Generator, n: int = 1 << 16) -> np.ndarray:
    """s_g: per-path standard deviation of the plain-MC CPW-first estimator of each Greek."""
    Z = rng.standard_normal((n, f.m))
    return cpw_first_joint(f, p, Z).std(axis=0, ddof=1)


def normalised_loss(mse: np.ndarray, scales: np.ndarray, weights=(1 / 3, 1 / 3, 1 / 3)) -> float:
    """L = sum_g w_g MSE_g / s_g^2."""
    return float(np.sum(np.asarray(weights) * np.asarray(mse) / np.asarray(scales) ** 2))

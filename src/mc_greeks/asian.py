"""Discretely monitored Asian digital option: simulation and Greek estimators.

Contract: payoff exp(-rT) 1{A >= K}, A = (1/m) sum_{i=1}^m S_{t_i}, t_i = i T/m.
Model: Black-Scholes GBM, simulated exactly on the monitoring grid from
an (N, m) matrix Z of independent N(0,1) increments:

    W_{t_i} = sqrt(dt) * (Z_1 + ... + Z_i),
    S_{t_i} = S0 exp((r - sigma^2/2) t_i + sigma W_{t_i}).

Estimator families (derivations in docs/mathematical_derivations.md, Part II):

* "LR"            likelihood ratio = Malliavin weight with a(t) = 1/t_1 (FLLLT Prop. 3.2).
                  Delta/Gamma use the score of the first increment; Vega uses
                  the score of all m increments. Unbiased for bounded payoffs.
* "Malliavin-avg" anticipating Malliavin Delta weight with direction u = 1 on [0,T]
                  (Skorohod integral). Differs from LR when m > 1. Delta only.
* "CPW-last"      conditional pathwise, integrating out the LAST increment
                  analytically (the conditioning used by Giles' 2009 "vibrato" MC),
                  then differentiating pathwise. Unbiased.
* "CPW-first"     conditional pathwise, integrating out the FIRST increment Z_1. Z_1
                  shifts every log S_{t_i} by the same amount, so A = exp(sigma sqrt(dt) Z_1) * Abar
                  and P(A >= K | Z_2..Z_m) = Phi(ln(Abar/K)/(sigma sqrt(dt))) exactly, smooth
                  everywhere. (Conditional pathwise for Asian options: Zhang & Wang 2020.)
* "FD-raw"        central finite differences of the raw indicator payoff, CRN.
* "FD-smooth"     central finite differences of the CPW-first smoothed price, CRN.

Each family can be driven by pseudorandom normals ("MC") or by scrambled
Sobol points mapped through the inverse normal CDF ("RQMC", sequential
path construction, independent scramblings for error bars).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from scipy import special
from scipy.stats import qmc

from .finite_difference import stencil
from .models import BlackScholesParams
from .payoffs import InapplicableEstimatorError

ASIAN_METHODS = ("LR", "Malliavin-avg", "CPW-last", "CPW-first", "FD-raw", "FD-smooth")
GREEKS = ("price", "delta", "gamma", "vega")
_SQRT2PI = math.sqrt(2 * math.pi)


@dataclass(frozen=True)
class AsianDigital:
    """1{(1/m) sum_i S_{t_i} >= K} with m equally spaced dates, t_m = T."""

    K: float
    m: int

    def __post_init__(self) -> None:
        if not self.K > 0:
            raise ValueError("strike K must be positive")
        if not (isinstance(self.m, (int, np.integer)) and self.m >= 1):
            raise ValueError("number of monitoring dates m must be an integer >= 1")

    @property
    def name(self) -> str:
        return f"asian_digital_m{self.m}"


def monitoring_times(p: BlackScholesParams, m: int) -> np.ndarray:
    return p.T * np.arange(1, m + 1) / m


def simulate_paths(p: BlackScholesParams, m: int, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (S, W, t): prices and Brownian values at the m dates, shape (N, m)."""
    Z = np.atleast_2d(np.asarray(Z, dtype=float))
    if Z.shape[1] != m:
        raise ValueError(f"Z must have m={m} columns, got {Z.shape}")
    t = monitoring_times(p, m)
    W = math.sqrt(p.T / m) * np.cumsum(Z, axis=1)
    S = p.S0 * np.exp((p.r - 0.5 * p.sigma**2) * t + p.sigma * W)
    return S, W, t


def _phi(x):
    return np.exp(-0.5 * x * x) / _SQRT2PI


# --- raw payoff and likelihood-ratio / Malliavin weights -------------------------------

def raw_payoff(f: AsianDigital, S: np.ndarray) -> np.ndarray:
    return (S.mean(axis=1) >= f.K).astype(float)


def lr_samples(f: AsianDigital, p: BlackScholesParams, Z: np.ndarray, greek: str) -> np.ndarray:
    """Likelihood-ratio (= FLLLT weight with a = 1/t_1) per-path terms."""
    S, _, _ = simulate_paths(p, f.m, Z)
    y = p.discount * raw_payoff(f, S)
    if greek == "price":
        return y
    dt = p.T / f.m
    sd = p.sigma * math.sqrt(dt)
    z1 = Z[:, 0]
    if greek == "delta":
        return y * z1 / (p.S0 * sd)
    if greek == "gamma":
        return y * (z1 * z1 - 1.0 - sd * z1) / (p.S0**2 * sd * sd)
    if greek == "vega":
        return y * (((Z * Z - 1.0) / p.sigma) - math.sqrt(dt) * Z).sum(axis=1)
    raise ValueError(f"unknown Greek {greek!r}")


def malliavin_avg_samples(f: AsianDigital, p: BlackScholesParams, Z: np.ndarray, greek: str) -> np.ndarray:
    """Anticipating Malliavin Delta with u = 1 on [0, T].

    With D_u S_i = sigma t_i S_i, D_u A = (sigma/m) sum t_i S_i, D_u D_u A =
    (sigma^2/m) sum t_i^2 S_i and G = A / D_u A:
        Delta = (D/x) E[ phi(A) (G W_T - 1 + A D_uD_uA / (D_uA)^2) ].
    Reduces to W_T/(x sigma T) when m = 1.
    """
    S, W, t = simulate_paths(p, f.m, Z)
    y = p.discount * raw_payoff(f, S)
    if greek == "price":
        return y
    if greek != "delta":
        raise InapplicableEstimatorError("Malliavin-avg is implemented for Delta only")
    A = S.mean(axis=1)
    DA = p.sigma * (S * t).mean(axis=1)
    DDA = p.sigma**2 * (S * t * t).mean(axis=1)
    weight = (A / DA) * W[:, -1] - 1.0 + A * DDA / DA**2
    return y * weight / p.S0


# --- conditional pathwise ---------------------------------------------------------------

def cpw_last_samples(f: AsianDigital, p: BlackScholesParams, Z: np.ndarray, greek: str) -> np.ndarray:
    """Conditional pathwise, last increment integrated out (uses Z_1..Z_{m-1}).

    Given S_1..S_{m-1}, let H = mK - sum_{i<m} S_i. Then A >= K iff S_T >= H, and
        P(A >= K | F_{t_{m-1}}) = 1 if H <= 0, else Phi(d),
        d = [ln(S_{m-1}/H) + (r - sigma^2/2) dt] / (sigma sqrt(dt)),
    with S_0 = S0 when m = 1. The map x -> Phi(d) is C^1 (indeed C^inf) in
    (S0, sigma) on H > 0 and joins the constant 1 smoothly at H = 0, which
    justifies pathwise differentiation (docs, Part II §3).
    """
    m = f.m
    dt = p.T / m
    s = p.sigma * math.sqrt(dt)
    c = (p.r - 0.5 * p.sigma**2) * dt
    n = Z.shape[0]
    if m > 1:
        S, W, t = _first_steps(p, m, Z)
        prev = S[:, -1]
        sum_prev = S.sum(axis=1)
    else:
        S = np.zeros((n, 0)); W = np.zeros((n, 0)); t = np.zeros(0)
        prev = np.full(n, p.S0)
        sum_prev = np.zeros(n)
    H = m * f.K - sum_prev
    pos = H > 0
    Hs = np.where(pos, H, 1.0)
    N = np.log(prev / Hs) + c
    d = N / s
    D = p.discount
    if greek == "price":
        return D * np.where(pos, special.ndtr(d), 1.0)
    pdf = np.where(pos, _phi(d), 0.0)
    x = p.S0
    if greek == "delta":
        dd = m * f.K / (s * x * Hs)
        return D * pdf * dd
    if greek == "gamma":
        dd = m * f.K / (s * x * Hs)
        d2 = -m * f.K * (Hs - sum_prev) / (s * (x * Hs) ** 2)
        return D * pdf * (d2 - d * dd * dd)
    if greek == "vega":
        if m > 1:
            dlogprev = W[:, -1] - p.sigma * t[-1]
            dsum = (S * (W - p.sigma * t)).sum(axis=1)
        else:
            dlogprev = np.zeros(n)
            dsum = np.zeros(n)
        dN = dlogprev + dsum / Hs - p.sigma * dt
        dd = dN / s - d / p.sigma
        return D * pdf * dd
    raise ValueError(f"unknown Greek {greek!r}")


def cpw_first_samples(f: AsianDigital, p: BlackScholesParams, Z: np.ndarray, greek: str) -> np.ndarray:
    """Conditional pathwise, first increment integrated out (uses Z_2..Z_m).

    Write S_i = exp(s Z_1) R_i with s = sigma sqrt(dt) and
    R_i = S0 exp((r - sigma^2/2) t_i + sigma (W_i - sqrt(dt) Z_1)); then
    A = exp(s Z_1) Abar with Abar = mean(R_i) > 0, so
        P(A >= K | Z_2..Z_m) = Phi(delta),   delta = ln(Abar/K) / s.
    Abar is proportional to S0, hence d delta/dS0 = 1/(S0 s) and
        Delta = D phi(delta)/(S0 s),
        Gamma = -D phi(delta) (delta + s) / (S0 s)^2,
        Vega  = D phi(delta) [ (d ln Abar/d sigma)/s - delta/sigma ],
    with d ln Abar/d sigma = sum_i R_i (W_i - sqrt(dt) Z_1 - sigma t_i) / sum_i R_i.
    For m = 1, Abar is deterministic and the estimator is exact.
    """
    m = f.m
    dt = p.T / m
    s = p.sigma * math.sqrt(dt)
    t = monitoring_times(p, m)
    Wt = math.sqrt(dt) * (np.cumsum(Z, axis=1) - Z[:, :1])  # W_i - sqrt(dt) Z_1
    R = p.S0 * np.exp((p.r - 0.5 * p.sigma**2) * t + p.sigma * Wt)
    Abar = R.mean(axis=1)
    dl = np.log(Abar / f.K) / s
    D = p.discount
    if greek == "price":
        return D * special.ndtr(dl)
    pdf = _phi(dl)
    if greek == "delta":
        return D * pdf / (p.S0 * s)
    if greek == "gamma":
        return -D * pdf * (dl + s) / (p.S0 * s) ** 2
    if greek == "vega":
        dlogA = (R * (Wt - p.sigma * t)).sum(axis=1) / R.sum(axis=1)
        return D * pdf * (dlogA / s - dl / p.sigma)
    raise ValueError(f"unknown Greek {greek!r}")


def _first_steps(p: BlackScholesParams, m: int, Z: np.ndarray):
    """Prices at t_1..t_{m-1} on the m-date grid (dt = T/m)."""
    dt = p.T / m
    t = dt * np.arange(1, m)
    W = math.sqrt(dt) * np.cumsum(Z[:, : m - 1], axis=1)
    S = p.S0 * np.exp((p.r - 0.5 * p.sigma**2) * t + p.sigma * W)
    return S, W, t


# --- finite differences (CRN) ------------------------------------------------------------

def fd_samples(f: AsianDigital, p: BlackScholesParams, Z: np.ndarray, greek: str, h: float,
               base: str = "raw", scheme: str = "central") -> np.ndarray:
    """CRN finite difference of the raw (base='raw') or CPW-first-smoothed (base='smooth') price."""
    st = stencil(p, greek, h, scheme)
    out = np.zeros(Z.shape[0])
    for q, cf in zip(st.params, st.coeffs):
        if base == "raw":
            S, _, _ = simulate_paths(q, f.m, Z)
            out += cf * q.discount * raw_payoff(f, S)
        elif base == "smooth":
            out += cf * cpw_first_samples(f, q, Z, "price")
        else:
            raise ValueError("base must be 'raw' or 'smooth'")
    return out / st.divisor


def evals_per_path(method: str, greek: str) -> int:
    if method.startswith("FD"):
        return 3 if greek == "gamma" else 2
    return 1


def check_applicable(method: str, greek: str) -> None:
    if method not in ASIAN_METHODS:
        raise ValueError(f"unknown method {method!r}")
    if method == "Malliavin-avg" and greek not in ("delta", "price"):
        raise InapplicableEstimatorError("Malliavin-avg: Delta only")
    if method.startswith("FD") and greek == "price":
        raise InapplicableEstimatorError("finite differences estimate Greeks only")


def per_path(method: str, greek: str, f: AsianDigital, p: BlackScholesParams, h: float | None = None):
    check_applicable(method, greek)
    if method == "LR":
        return lambda Z: lr_samples(f, p, Z, greek)
    if method == "Malliavin-avg":
        return lambda Z: malliavin_avg_samples(f, p, Z, greek)
    if method == "CPW-last":
        return lambda Z: cpw_last_samples(f, p, Z, greek)
    if method == "CPW-first":
        return lambda Z: cpw_first_samples(f, p, Z, greek)
    base = "raw" if method == "FD-raw" else "smooth"
    return lambda Z: fd_samples(f, p, Z, greek, h, base)


# --- sampling and estimation -------------------------------------------------------------

def block_rows(m: int) -> int:
    """Rows per block (a power of two) so a block holds <= ~2^18 normals."""
    b = max(256, (1 << 18) // m)
    return 1 << (b.bit_length() - 1)


def rqmc_layout(n: int, min_scrambles: int = 8) -> tuple[int, int]:
    """(points per scramble, number of scrambles) using ~n points in total.

    Points per scramble is the largest power of two with at least
    ``min_scrambles`` scrambles; extra scrambles absorb the remaining budget,
    so between n - per and n points are used.
    """
    per = max(2, 1 << (max(n // min_scrambles, 2).bit_length() - 1))
    return per, max(min_scrambles, n // per)


def sobol_normal_blocks(n: int, m: int, rng: np.random.Generator):
    """Yield blocks of an Owen-scrambled Sobol sequence in R^m mapped to N(0, I)."""
    sampler = qmc.Sobol(d=m, scramble=True, seed=rng)
    eps = np.finfo(float).eps
    done, b = 0, block_rows(m)
    while done < n:
        k = min(b, n - done)
        u = sampler.random(k)
        yield special.ndtri(np.clip(u, eps, 1 - eps))
        done += k


@dataclass(frozen=True)
class AsianEstimate:
    value: float
    stderr: float
    n_paths: int
    path_evals: int
    runtime: float
    n_nonzero: int = -1  # paths with a nonzero per-path term (pilot diagnostics)


def _mean_var_blocks(fn, blocks, counter=None) -> tuple[float, float, int]:
    n_tot, mean, m2 = 0, 0.0, 0.0
    for Zb in blocks:
        y = fn(Zb)
        if counter is not None:
            counter[0] += int(np.count_nonzero(y))
        nb, mb = y.size, float(y.mean())
        m2b = float(((y - mb) ** 2).sum())
        delta = mb - mean
        nn = n_tot + nb
        mean += delta * nb / nn
        m2 += m2b + delta**2 * n_tot * nb / nn
        n_tot = nn
    return mean, m2 / max(n_tot - 1, 1), n_tot


def asian_estimate(method: str, greek: str, f: AsianDigital, p: BlackScholesParams, n: int,
                   rng: np.random.Generator, sampler: str = "MC", h: float | None = None,
                   n_scrambles: int = 8) -> AsianEstimate:
    """Estimate a Greek with ``n`` paths in total.

    MC: one i.i.d. sample, SE from the per-path variance.
    RQMC: ``n_scrambles`` independent scramblings of n // n_scrambles points
    each (rounded down to a power of two when possible); the estimate is the
    mean of the scramble means and the SE their standard deviation / sqrt(R).
    """
    fn = per_path(method, greek, f, p, h)
    nz = [0]
    t0 = time.perf_counter()
    if sampler == "MC":
        def gen():
            done, b = 0, block_rows(f.m)
            while done < n:
                k = min(b, n - done)
                yield rng.standard_normal((k, f.m))
                done += k
        mean, var, used = _mean_var_blocks(fn, gen(), nz)
        se = math.sqrt(var / used)
    elif sampler == "RQMC":
        per, n_scr = rqmc_layout(n, n_scrambles)
        # CPW-first ignores Z_1: give the first (best) Sobol coordinates to Z_2, Z_3, ...
        roll = 1 if method in ("CPW-first", "FD-smooth") else 0
        blocks = lambda: (np.roll(Zb, roll, axis=1) for Zb in sobol_normal_blocks(per, f.m, rng))
        means = np.array([_mean_var_blocks(fn, blocks(), nz)[0] for _ in range(n_scr)])
        mean, se, used = float(means.mean()), float(means.std(ddof=1) / math.sqrt(n_scr)), per * n_scr
    else:
        raise ValueError("sampler must be 'MC' or 'RQMC'")
    return AsianEstimate(mean, se, used, used * evals_per_path(method, greek), time.perf_counter() - t0, nz[0])

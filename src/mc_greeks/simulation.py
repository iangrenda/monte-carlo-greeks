"""Random-number generation and exact simulation of S_T.

For a European payoff only the terminal value S_T matters, and under
Black-Scholes it is available in closed form

    S_T = S0 * exp((r - sigma^2/2) T + sigma W_T),   W_T = sqrt(T) Z,  Z ~ N(0,1).

Sampling Z and applying this map is *exact*: there is no time-discretisation
bias, unlike an Euler-Maruyama scheme. Every estimator in this package is
therefore a function of a vector of standard normals Z, which makes common
random numbers (CRN) and paired comparisons straightforward: two estimators
evaluated on the same Z are coupled, two estimators evaluated on Z vectors
drawn from independent streams are independent.

Seed handling: every experiment owns a master seed. ``spawn_generators``
uses ``numpy.random.SeedSequence.spawn`` to derive statistically independent
child streams (one per replication), so replications are independent but the
whole experiment is reproducible from one integer.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats
from scipy.stats import qmc

from .models import BlackScholesParams


def make_generator(seed: int | np.random.SeedSequence | None) -> np.random.Generator:
    """Create a PCG64 generator from an int seed or a SeedSequence."""
    return np.random.Generator(np.random.PCG64(seed))


def spawn_generators(seed: int | np.random.SeedSequence, n: int) -> list[np.random.Generator]:
    """Return ``n`` independent generators derived from one master seed."""
    ss = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    return [make_generator(child) for child in ss.spawn(n)]


def pseudo_normals(rng: np.random.Generator, n: int) -> np.ndarray:
    """``n`` i.i.d. standard normal draws (pseudorandom)."""
    if n <= 0:
        raise ValueError("n must be positive")
    return rng.standard_normal(n)


def sobol_normals(n: int, seed: int | np.random.Generator | None = None, scramble: bool = True) -> np.ndarray:
    """First ``n`` points of a (scrambled) 1-D Sobol sequence mapped to N(0,1).

    The uniform points are mapped through the inverse normal CDF. The
    sequence is generated in a block of size 2^m >= n and truncated, so
    prefixes (running averages) are the natural low-discrepancy prefixes.
    With ``scramble=True`` (Owen scrambling as implemented in SciPy) each
    seed gives an independent randomisation, which allows error bars
    (randomised QMC). Points are clipped away from {0, 1} so the inverse CDF
    stays finite (the unscrambled sequence starts at 0).
    """
    if n <= 0:
        raise ValueError("n must be positive")
    m = max(0, math.ceil(math.log2(n)))
    sampler = qmc.Sobol(d=1, scramble=scramble, seed=seed)
    u = sampler.random_base2(m)[:n, 0]
    eps = np.finfo(float).eps
    u = np.clip(u, eps, 1.0 - eps)
    return stats.norm.ppf(u)


def terminal_brownian(params: BlackScholesParams, Z: np.ndarray) -> np.ndarray:
    """W_T = sqrt(T) Z."""
    return params.sqrtT * np.asarray(Z, dtype=float)


def terminal_price(params: BlackScholesParams, Z: np.ndarray) -> np.ndarray:
    """Exact sample of S_T driven by the standard normals ``Z``."""
    Z = np.asarray(Z, dtype=float)
    drift = (params.r - 0.5 * params.sigma**2) * params.T
    return params.S0 * np.exp(drift + params.sigma * params.sqrtT * Z)


def z_of_price(params: BlackScholesParams, s: float) -> float:
    """Inverse of ``terminal_price``: the value of Z at which S_T = s.

    Used to place quadrature breakpoints at payoff discontinuities.
    """
    if s <= 0:
        return -math.inf
    drift = (params.r - 0.5 * params.sigma**2) * params.T
    return (math.log(s / params.S0) - drift) / (params.sigma * params.sqrtT)

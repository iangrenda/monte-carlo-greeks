"""Unified interface to every estimator, plus workload accounting.

Every method is identified by a string:

    "MC"            plain Monte Carlo price (greek="price" only)
    "Malliavin"     global Malliavin weight (paper, Section 4)
    "Malliavin-loc" localised Malliavin, European call only (paper, end of Section 4)
    "Pathwise"      pathwise / IPA (Lipschitz payoffs, first-order Greeks only)
    "FD-CRN-fwd", "FD-CRN-ctr"   finite differences with common random numbers
    "FD-IND-fwd", "FD-IND-ctr"   finite differences with independent random numbers

``estimate`` takes a *base* normal vector Z of length N that is shared by all
methods in the same replication (which gives paired comparisons). The IND
finite differences use Z for their first bumped price and draw the
remaining ones from ``rng``, so they are independent of each other.

Workload: ``payoff_evals`` counts terminal-state + payoff evaluations
(N per bumped price). The Malliavin and pathwise estimators need one per
path; FD needs 2 (Delta, Vega) or 3 (Gamma) per path.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import integrate, stats

from . import analytical
from .finite_difference import fd_crn_samples, fd_expected_value, fd_independent, stencil
from .malliavin import localized_call_samples, malliavin_samples
from .models import BlackScholesParams
from .pathwise import pathwise_samples
from .payoffs import InapplicableEstimatorError, Payoff
from .simulation import terminal_price, z_of_price

METHODS = (
    "MC", "Malliavin", "Malliavin-loc", "Pathwise",
    "FD-CRN-fwd", "FD-CRN-ctr", "FD-IND-fwd", "FD-IND-ctr",
)
FD_METHODS = ("FD-CRN-fwd", "FD-CRN-ctr", "FD-IND-fwd", "FD-IND-ctr")


@dataclass(frozen=True)
class Estimate:
    value: float
    stderr: float
    n_paths: int
    payoff_evals: int
    runtime: float  # seconds spent inside the estimator (excl. base-normal generation)


def _fd_parts(method: str) -> tuple[bool, str]:
    crn = "-CRN-" in method
    scheme = "forward" if method.endswith("fwd") else "central"
    return crn, scheme


def check_applicable(method: str, greek: str, payoff: Payoff) -> None:
    """Raise InapplicableEstimatorError if (method, greek, payoff) is invalid."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")
    if method == "MC" and greek != "price":
        raise InapplicableEstimatorError("MC estimates the price only")
    if method in FD_METHODS and greek == "price":
        raise InapplicableEstimatorError("finite differences estimate Greeks, not the price")
    if method == "Pathwise":
        pathwise_samples(payoff, _DUMMY, np.zeros(1), greek)  # raises if invalid
    if method == "Malliavin-loc":
        localized_call_samples(payoff, _DUMMY, np.zeros(1), greek, width=min(1.0, 0.5 * getattr(payoff, "K", 2.0)))


_DUMMY = BlackScholesParams(100.0, 0.0, 0.2, 1.0)


def per_path_function(
    method: str, greek: str, payoff: Payoff, p: BlackScholesParams,
    h: float | None = None, loc_width: float | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return Z -> per-path terms for single-stream (CRN-type) estimators."""
    check_applicable(method, greek, payoff)
    if method == "MC":
        return lambda Z: p.discount * payoff(terminal_price(p, Z))
    if method == "Malliavin":
        return lambda Z: malliavin_samples(payoff, p, Z, greek)
    if method == "Malliavin-loc":
        return lambda Z: localized_call_samples(payoff, p, Z, greek, loc_width)
    if method == "Pathwise":
        return lambda Z: pathwise_samples(payoff, p, Z, greek)
    crn, scheme = _fd_parts(method)
    if not crn:
        raise ValueError("IND finite differences are not a single-stream estimator")
    return lambda Z: fd_crn_samples(payoff, p, Z, greek, h, scheme)


def payoff_evals_per_path(method: str, greek: str, p: BlackScholesParams, h: float | None = None) -> int:
    if method in FD_METHODS:
        _, scheme = _fd_parts(method)
        return stencil(p, greek, h, scheme).n_evals
    return 1


def estimate(
    method: str, greek: str, payoff: Payoff, p: BlackScholesParams, Z: np.ndarray,
    rng: np.random.Generator | None = None, h: float | None = None, loc_width: float | None = None,
) -> Estimate:
    """Estimate ``greek`` with ``method`` from base normals ``Z``."""
    n = int(np.size(Z))
    t0 = time.perf_counter()
    if method in ("FD-IND-fwd", "FD-IND-ctr"):
        if rng is None:
            raise ValueError("independent finite differences need an rng for the extra draws")
        _, scheme = _fd_parts(method)
        k = stencil(p, greek, h, scheme).n_evals
        Zs = [Z] + [rng.standard_normal(n) for _ in range(k - 1)]
        val, se = fd_independent(payoff, p, Zs, greek, h, scheme)
        evals = k * n
    else:
        m, v = blocked_mean_var(per_path_function(method, greek, payoff, p, h, loc_width), Z)
        val, se = m, math.sqrt(v / n)
        evals = payoff_evals_per_path(method, greek, p, h) * n
    return Estimate(float(val), float(se), n, evals, time.perf_counter() - t0)


#: Paths per block. Processing in cache-sized blocks keeps wall-clock time
#: proportional to work: on the development machine a single 150k-element
#: pass was ~3-4x slower per path than 50k (temporaries fall out of cache),
#: which would unfairly penalise methods that use more paths per estimate.
BLOCK = 16_384


def blocked_mean_var(fn: Callable[[np.ndarray], np.ndarray], Z: np.ndarray, block: int = BLOCK) -> tuple[float, float]:
    """Sample mean and unbiased variance of fn(Z), evaluated block by block.

    Block statistics are merged with Chan et al.'s pairwise update, which is
    numerically stable (no E[y^2] - E[y]^2 cancellation).
    """
    n_tot, mean, m2 = 0, 0.0, 0.0
    for i in range(0, np.size(Z), block):
        y = fn(Z[i:i + block])
        nb = y.size
        mb = float(y.mean())
        m2b = float(((y - mb) ** 2).sum())
        delta = mb - mean
        n_new = n_tot + nb
        mean += delta * nb / n_new
        m2 += m2b + delta**2 * n_tot * nb / n_new
        n_tot = n_new
    return mean, (m2 / (n_tot - 1) if n_tot > 1 else float("nan"))


# --- exact (quadrature) moments of the estimators -------------------------------------

def gaussian_moments(fn: Callable[[np.ndarray], np.ndarray], breakpoints=(), L: float = 12.0) -> tuple[float, float]:
    """(E[f(Z)], E[f(Z)^2]) for Z ~ N(0,1) by adaptive quadrature.

    ``breakpoints`` are the Z-locations where f jumps or kinks; the integral
    is split there so each piece is smooth. Truncation at |z| = L costs
    ~phi(12) ~ 1e-32 times polynomial/exponential growth, which is negligible here.
    """
    pts = sorted({-L, L, *[b for b in breakpoints if -L < b < L]})
    m1 = m2 = 0.0
    for lo, hi in zip(pts[:-1], pts[1:]):
        g1 = lambda z: float(fn(np.array([z]))[0]) * stats.norm.pdf(z)
        g2 = lambda z: float(fn(np.array([z]))[0]) ** 2 * stats.norm.pdf(z)
        m1 += integrate.quad(g1, lo, hi, limit=200, epsabs=1e-15, epsrel=1e-11)[0]
        m2 += integrate.quad(g2, lo, hi, limit=200, epsabs=1e-15, epsrel=1e-11)[0]
    return m1, m2


def _breakpoints(payoff: Payoff, params_list, extra_s=()) -> list[float]:
    pts = []
    for q in params_list:
        for s in (*payoff.singular_points(), *extra_s):
            pts.append(z_of_price(q, s))
    return pts


def exact_estimator_moments(
    method: str, greek: str, payoff: Payoff, p: BlackScholesParams,
    h: float | None = None, loc_width: float | None = None,
) -> dict[str, float]:
    """Exact mean and per-path variance of an estimator (no simulation noise).

    Returns ``mean`` (= E[theta_hat]), ``bias`` (vs the analytical Greek) and
    ``var1`` = N * Var(theta_hat), so that Var(theta_hat) = var1 / N and
    RMSE(N) = sqrt(bias^2 + var1/N). For FD the mean comes from the
    analytical stencil; for single-stream estimators the variance comes from
    1-D Gaussian quadrature split at the payoff's discontinuities.
    """
    check_applicable(method, greek, payoff)
    truth = analytical.greek(payoff, p, greek)
    if method in ("FD-IND-fwd", "FD-IND-ctr"):
        _, scheme = _fd_parts(method)
        st = stencil(p, greek, h, scheme)
        var1 = 0.0
        for q, c in zip(st.params, st.coeffs):
            f = lambda Z, q=q: q.discount * payoff(terminal_price(q, Z))
            m1, m2 = gaussian_moments(f, _breakpoints(payoff, [q]))
            var1 += c**2 * (m2 - m1**2)
        mean = fd_expected_value(lambda q: analytical.greek(payoff, q, "price"), p, greek, h, scheme)
        return {"mean": mean, "bias": mean - truth, "var1": var1 / st.divisor**2, "truth": truth}
    fn = per_path_function(method, greek, payoff, p, h, loc_width)
    if method in FD_METHODS:
        _, scheme = _fd_parts(method)
        plist = stencil(p, greek, h, scheme).params
    else:
        plist = [p]
    extra = ()
    if method == "Malliavin-loc":
        extra = (payoff.K - loc_width, payoff.K + loc_width)
    m1, m2 = gaussian_moments(fn, _breakpoints(payoff, plist, extra))
    if method in FD_METHODS:
        _, scheme = _fd_parts(method)
        m1_exact = fd_expected_value(lambda q: analytical.greek(payoff, q, "price"), p, greek, h, scheme)
    else:
        m1_exact = m1
    return {"mean": m1_exact, "bias": m1_exact - truth, "var1": m2 - m1**2, "truth": truth}

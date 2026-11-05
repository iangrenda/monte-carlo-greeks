"""Closed-form Black-Scholes prices and Greeks (benchmarks).

Notation: d2(K) = [ln(S0/K) + (r - sigma^2/2) T] / (sigma sqrt T),
d1 = d2 + sigma sqrt T, Phi / phi standard normal CDF / PDF,
D = exp(-r T).

Cash-or-nothing digital call, V = D Phi(d2):
    Delta = D phi(d2) / (S0 sigma sqrt T)
    Gamma = -D phi(d2) d1 / (S0^2 sigma^2 T)
    Vega  = -D phi(d2) d1 / sigma                     (per unit sigma)
using d d2/d S0 = 1/(S0 sigma sqrt T) and d d2/d sigma = -d1/sigma.

European call, V = S0 Phi(d1) - K D Phi(d2):
    Delta = Phi(d1), Gamma = phi(d1)/(S0 sigma sqrt T), Vega = S0 phi(d1) sqrt T.

Interval digital 1{a <= S_T <= b} = digital(a) - digital(b) almost surely,
so every Greek is the difference of the two digital Greeks.

Derivations are in docs/mathematical_derivations.md; tests check each
formula against high-precision finite differences of the closed-form price.
"""

from __future__ import annotations

import math

from scipy.stats import norm

from .models import BlackScholesParams
from .payoffs import CashDigitalCall, EuropeanCall, IntervalDigital, Payoff

GREEKS = ("price", "delta", "gamma", "vega")


def d2(p: BlackScholesParams, K: float) -> float:
    return (math.log(p.S0 / K) + (p.r - 0.5 * p.sigma**2) * p.T) / (p.sigma * p.sqrtT)


def d1(p: BlackScholesParams, K: float) -> float:
    return d2(p, K) + p.sigma * p.sqrtT


# --- cash-or-nothing digital call -------------------------------------------------

def digital_price(p: BlackScholesParams, K: float) -> float:
    return p.discount * norm.cdf(d2(p, K))


def digital_delta(p: BlackScholesParams, K: float) -> float:
    return p.discount * norm.pdf(d2(p, K)) / (p.S0 * p.sigma * p.sqrtT)


def digital_gamma(p: BlackScholesParams, K: float) -> float:
    return -p.discount * norm.pdf(d2(p, K)) * d1(p, K) / (p.S0**2 * p.sigma**2 * p.T)


def digital_vega(p: BlackScholesParams, K: float) -> float:
    return -p.discount * norm.pdf(d2(p, K)) * d1(p, K) / p.sigma


# --- European call ----------------------------------------------------------------

def call_price(p: BlackScholesParams, K: float) -> float:
    return p.S0 * norm.cdf(d1(p, K)) - K * p.discount * norm.cdf(d2(p, K))


def call_delta(p: BlackScholesParams, K: float) -> float:
    return norm.cdf(d1(p, K))


def call_gamma(p: BlackScholesParams, K: float) -> float:
    return norm.pdf(d1(p, K)) / (p.S0 * p.sigma * p.sqrtT)


def call_vega(p: BlackScholesParams, K: float) -> float:
    return p.S0 * norm.pdf(d1(p, K)) * p.sqrtT


_DIGITAL = {"price": digital_price, "delta": digital_delta, "gamma": digital_gamma, "vega": digital_vega}
_CALL = {"price": call_price, "delta": call_delta, "gamma": call_gamma, "vega": call_vega}


def greek(payoff: Payoff, p: BlackScholesParams, name: str) -> float:
    """Analytical value of ``name`` in {price, delta, gamma, vega} for ``payoff``."""
    if name not in GREEKS:
        raise ValueError(f"unknown Greek {name!r}; expected one of {GREEKS}")
    if isinstance(payoff, EuropeanCall):
        return float(_CALL[name](p, payoff.K))
    if isinstance(payoff, CashDigitalCall):
        return float(_DIGITAL[name](p, payoff.K))
    if isinstance(payoff, IntervalDigital):
        f = _DIGITAL[name]
        return float(f(p, payoff.a) - f(p, payoff.b))
    raise NotImplementedError(f"no analytical benchmark registered for {type(payoff).__name__}")


def greeks(payoff: Payoff, p: BlackScholesParams) -> dict[str, float]:
    """All analytical benchmarks as a dict."""
    return {g: greek(payoff, p, g) for g in GREEKS}

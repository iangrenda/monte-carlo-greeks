"""Phase 8D: European option pricing on the forward (Black-76), implied volatility, Greeks, forward inference.

For a European option on an index with forward price F for expiry T and discount factor D = P(0, T):

    call = D [F N(d1) - K N(d2)],  put = D [K N(-d2) - F N(-d1)],
    d1 = (ln(F/K) + sigma^2 T / 2) / (sigma sqrt T),  d2 = d1 - sigma sqrt T.

With a continuous dividend yield q and rate r, F = S e^{(r - q) T}, D = e^{-r T}, and this is
Black-Scholes-Merton. Working on the forward avoids having to know q: the forward is inferred
from put-call parity on the same chain (C - P = D (F - K)), which is how index options are
handled in practice. Greeks are reported with respect to the spot S (dF/dS = F/S) and sigma.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import optimize, special

_SQ2PI = math.sqrt(2.0 * math.pi)


def _d12(F, K, T, sigma):
    sv = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sv * sv) / sv
    return d1, d1 - sv


def black_price(F, K, T, D, sigma, cp):
    """European price on the forward. cp = +1 call, -1 put (arrays broadcast)."""
    F, K, T, D, sigma, cp = map(np.asarray, (F, K, T, D, sigma, cp))
    d1, d2 = _d12(F, K, T, sigma)
    return D * cp * (F * special.ndtr(cp * d1) - K * special.ndtr(cp * d2))


def black_greeks(S, F, K, T, D, sigma, cp) -> dict:
    """Delta and Gamma with respect to spot S (forward F = S x carry), Vega per unit sigma."""
    S, F, K, T, D, sigma, cp = map(np.asarray, (S, F, K, T, D, sigma, cp))
    d1, _ = _d12(F, K, T, sigma)
    phi = np.exp(-0.5 * d1 * d1) / _SQ2PI
    dFdS = F / S
    sv = sigma * np.sqrt(T)
    delta = D * cp * special.ndtr(cp * d1) * dFdS
    gamma = D * phi / (F * sv) * dFdS**2
    vega = D * F * phi * np.sqrt(T)
    return {"delta": delta, "gamma": gamma, "vega": vega}


def no_arbitrage_bounds(F, K, D, cp):
    """(lower, upper) European price bounds on the forward: intrinsic D (cp (F - K))^+ and D F (call) / D K (put)."""
    lo = D * max(cp * (F - K), 0.0)
    hi = D * F if cp > 0 else D * K
    return lo, hi


def implied_vol(price, F, K, T, D, cp, lo=1e-6, hi=5.0, tol=1e-12) -> float:
    """Black implied volatility by Brent's method; NaN if the price is outside the no-arbitrage bounds."""
    lb, ub = no_arbitrage_bounds(F, K, D, cp)
    if not (lb < price < ub) or T <= 0:
        return float("nan")
    f = lambda s: float(black_price(F, K, T, D, s, cp)) - price
    if f(lo) > 0 or f(hi) < 0:
        return float("nan")
    return float(optimize.brentq(f, lo, hi, xtol=tol, rtol=1e-14, maxiter=200))


def implied_forward(strikes, call_mid, put_mid, D, n_closest: int = 5) -> float:
    """Forward from put-call parity C - P = D (F - K): median of K + (C - P)/D over the n strikes with
    the smallest |C - P| (closest to the money, where both quotes are most reliable)."""
    K, C, P = map(np.asarray, (strikes, call_mid, put_mid))
    ok = np.isfinite(C) & np.isfinite(P)
    K, C, P = K[ok], C[ok], P[ok]
    idx = np.argsort(np.abs(C - P))[:n_closest]
    return float(np.median(K[idx] + (C[idx] - P[idx]) / D))

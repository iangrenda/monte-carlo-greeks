"""Malliavin / integration-by-parts weights in the Black-Scholes model.

Section 4 of Fournie et al. (1999), with the choice a(t) = 1/T, gives for a
European payoff phi(S_T) and W_T = sqrt(T) Z:

    Delta = E[ D phi(S_T) * W_T / (x sigma T) ]
    Gamma = E[ D phi(S_T) * (1/(x^2 sigma T)) (W_T^2/(sigma T) - W_T - 1/sigma) ]
    Vega  = E[ D phi(S_T) * (W_T^2/(sigma T) - W_T - 1/sigma) ]

with D = exp(-rT). The weight multiplies the payoff; phi is never
differentiated, which is why these estimators remain unbiased (and have
O(1) per-sample variance) for discontinuous payoffs. The weight does not
depend on phi. In this one-dimensional European setting the weights
coincide with likelihood-ratio (score-function) weights of the lognormal
density; the paper's contribution is that the Malliavin route extends to
settings where no density is available. Only the explicit Black-Scholes
formulas are implemented here, not the general framework.

Also implemented: the paper's *localised* estimator for the European call
(end of Section 4), which splits (s - K)^+ = G_delta(s) + F_delta(s) where
G_delta is a C^1 smoothing of the kink on [K - delta, K + delta] that is
differentiated pathwise and F_delta has compact support and is handled by
the Malliavin weight. The Delta and Gamma versions are the paper's formulas.
The Vega version applies the same decomposition and is our derivation.
"""

from __future__ import annotations

import numpy as np

from .models import BlackScholesParams
from .payoffs import EuropeanCall, InapplicableEstimatorError, Payoff
from .simulation import terminal_brownian, terminal_price


def malliavin_weights(p: BlackScholesParams, Z: np.ndarray) -> dict[str, np.ndarray]:
    """Return the price/Delta/Gamma/Vega weights evaluated on normals ``Z``."""
    W = terminal_brownian(p, Z)
    sT = p.sigma * p.T
    w_vega = W**2 / sT - W - 1.0 / p.sigma
    return {
        "price": np.ones_like(W),
        "delta": W / (p.S0 * sT),
        "gamma": w_vega / (p.S0**2 * sT),
        "vega": w_vega,
    }


def malliavin_samples(payoff: Payoff, p: BlackScholesParams, Z: np.ndarray, greek: str) -> np.ndarray:
    """Per-path terms D * phi(S_T) * pi_greek, whose mean estimates the Greek.

    Only the requested weight is formed (cheaper than ``malliavin_weights``)."""
    if greek not in ("price", "delta", "gamma", "vega"):
        raise ValueError(f"unknown Greek {greek!r}")
    y = p.discount * payoff(terminal_price(p, Z))
    if greek == "price":
        return y
    W = terminal_brownian(p, Z)
    sT = p.sigma * p.T
    if greek == "delta":
        return y * (W / (p.S0 * sT))
    w_vega = W * W / sT - W - 1.0 / p.sigma
    return y * (w_vega / (p.S0**2 * sT) if greek == "gamma" else w_vega)


# --- localisation (European call only) -----------------------------------------------

def _G(s: np.ndarray, K: float, d: float) -> np.ndarray:
    """C^1 smoothing of (s-K)^+: 0 below K-d, (s-K+d)^2/(4d) inside, s-K above."""
    return np.where(s <= K - d, 0.0, np.where(s >= K + d, s - K, (s - K + d) ** 2 / (4 * d)))


def _H(s: np.ndarray, K: float, d: float) -> np.ndarray:
    """G' : the paper's H_delta, a linear ramp from 0 to 1 on [K-d, K+d]."""
    return np.clip((s - (K - d)) / (2 * d), 0.0, 1.0)


def _I(s: np.ndarray, K: float, d: float) -> np.ndarray:
    """G'' : the paper's I_delta = 1{|s-K| < d} / (2d)."""
    return (np.abs(s - K) < d) / (2 * d)


def localized_call_samples(
    payoff: Payoff, p: BlackScholesParams, Z: np.ndarray, greek: str, width: float
) -> np.ndarray:
    """Per-path terms of the localised Malliavin estimator for a call.

    Delta = E[D H(S_T) S_T/x]            + E[D F(S_T) pi_delta]
    Gamma = E[D I(S_T) (S_T/x)^2]        + E[D F(S_T) pi_gamma]
    Vega  = E[D H(S_T) S_T (W_T - sigma T)] + E[D F(S_T) pi_vega]   (our extension)

    with F = (s-K)^+ - G, which vanishes outside (K - width, K + width).
    """
    if not isinstance(payoff, EuropeanCall):
        raise InapplicableEstimatorError("localised estimator implemented for the European call only")
    if not 0 < width < payoff.K:
        raise ValueError("localisation width must satisfy 0 < width < K")
    if greek == "price":
        return p.discount * payoff(terminal_price(p, Z))
    K, d = payoff.K, width
    S = terminal_price(p, Z)
    W = terminal_brownian(p, Z)
    F = payoff(S) - _G(S, K, d)
    w = malliavin_weights(p, Z)
    if greek == "delta":
        smooth = _H(S, K, d) * S / p.S0
    elif greek == "gamma":
        smooth = _I(S, K, d) * (S / p.S0) ** 2
    elif greek == "vega":
        smooth = _H(S, K, d) * S * (W - p.sigma * p.T)
    else:
        raise ValueError(f"unknown Greek {greek!r}")
    return p.discount * (smooth + F * w[greek])

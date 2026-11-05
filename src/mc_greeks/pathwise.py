"""Pathwise (infinitesimal perturbation analysis) estimators.

If theta -> D phi(S_T(theta)) is almost surely differentiable and Lipschitz
in theta with an integrable Lipschitz constant (true when phi is Lipschitz,
e.g. the call), dominated convergence allows exchanging d/dtheta and E:

    Delta = D E[phi'(S_T) dS_T/dS0] = D E[phi'(S_T) S_T / S0]
    Vega  = D E[phi'(S_T) dS_T/dsigma] = D E[phi'(S_T) S_T (W_T - sigma T)]

For the call phi'(s) = 1{s > K}.

When the pathwise estimator fails:
* digital payoffs: phi' = 0 almost everywhere, so the naive pathwise
  estimator is identically 0, while the true Delta is not. The interchange
  is invalid because phi is not continuous. We raise
  InapplicableEstimatorError instead of returning 0.
* call Gamma: differentiating 1{S_T > K} S_T/S0 pathwise again gives 0 a.s.
  (S_T/S0 does not depend on S0) and misses the entire true Gamma, which is
  the boundary term D (K/S0)^2 f_{S_T}(K) coming from the kink
  (f_{S_T} = lognormal density). So pathwise Gamma is also
  inapplicable. (Mixed pathwise/likelihood-ratio Gamma estimators exist,
  but they are not pathwise estimators and are not implemented here.)
"""

from __future__ import annotations

import numpy as np

from .models import BlackScholesParams
from .payoffs import InapplicableEstimatorError, Payoff
from .simulation import terminal_brownian, terminal_price


def pathwise_samples(payoff: Payoff, p: BlackScholesParams, Z: np.ndarray, greek: str) -> np.ndarray:
    """Per-path pathwise terms, whose mean estimates the Greek."""
    if greek == "price":
        return p.discount * payoff(terminal_price(p, Z))
    if not payoff.lipschitz:
        raise InapplicableEstimatorError(
            f"pathwise {greek} is invalid for the discontinuous payoff '{payoff.name}'"
        )
    if greek == "gamma":
        raise InapplicableEstimatorError(
            "pathwise gamma is invalid: phi' has a jump, so differentiating it pathwise "
            "drops the boundary (local-time) contribution"
        )
    S = terminal_price(p, Z)
    dphi = payoff.derivative(S)
    if greek == "delta":
        return p.discount * dphi * S / p.S0
    if greek == "vega":
        W = terminal_brownian(p, Z)
        return p.discount * dphi * S * (W - p.sigma * p.T)
    raise ValueError(f"unknown Greek {greek!r}")

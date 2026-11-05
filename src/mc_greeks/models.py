"""Black-Scholes model parameters.

The risk-neutral dynamics are

    dS_t = r S_t dt + sigma S_t dW_t,     S_0 = x,

with constant r and sigma, which is the setting of Section 4 of
Fournie, Lasry, Lebuchoux, Lions and Touzi (1999).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class BlackScholesParams:
    """Parameters of the one-dimensional Black-Scholes model.

    Attributes
    ----------
    S0 : initial asset price (x in the paper), must be > 0.
    r : constant risk-free rate (continuously compounded).
    sigma : volatility, must be > 0. Vega is always reported per *unit*
        change of sigma (i.e. d V / d sigma), not per percentage point.
    T : maturity in years, must be > 0.
    """

    S0: float
    r: float
    sigma: float
    T: float

    def __post_init__(self) -> None:
        for name in ("S0", "r", "sigma", "T"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite real number, got {value!r}")
        if self.S0 <= 0:
            raise ValueError(f"S0 must be positive, got {self.S0}")
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")
        if self.T <= 0:
            raise ValueError(f"T (maturity) must be positive, got {self.T}")

    @property
    def discount(self) -> float:
        """Discount factor exp(-r T)."""
        return math.exp(-self.r * self.T)

    @property
    def sqrtT(self) -> float:
        return math.sqrt(self.T)

    def bumped(self, **changes: float) -> "BlackScholesParams":
        """Return a copy with some parameters replaced (validation re-runs)."""
        return replace(self, **changes)


#: Parameters of the numerical experiments in Section 4 of Fournie et al. (1999).
PAPER_PARAMS = BlackScholesParams(S0=100.0, r=0.10, sigma=0.20, T=1.0)

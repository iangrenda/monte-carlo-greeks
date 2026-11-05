"""European payoff functions phi(S_T).

To add a payoff, subclass :class:`Payoff` and implement ``__call__`` and
``singular_points``. Implement ``derivative`` only if the payoff is
Lipschitz (absolutely continuous), because only then is the pathwise
(IPA) estimator unbiased; leave ``lipschitz = False`` otherwise. Analytical
benchmarks are registered separately in :mod:`mc_greeks.analytical`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class InapplicableEstimatorError(ValueError):
    """Raised when an estimator is mathematically invalid for a payoff/Greek.

    Experiments catch this and record the estimator as "inapplicable"
    instead of reporting a misleading number (e.g. a pathwise digital
    Delta, which is identically zero but the true Delta is not).
    """


class Payoff(ABC):
    """Base class for a European payoff depending on S_T only."""

    name: str = "payoff"
    #: True if phi is continuous everywhere.
    continuous: bool = False
    #: True if phi is Lipschitz, so the pathwise first-order estimator is valid.
    lipschitz: bool = False

    @abstractmethod
    def __call__(self, s: np.ndarray) -> np.ndarray:
        """Evaluate phi elementwise."""

    @abstractmethod
    def singular_points(self) -> tuple[float, ...]:
        """Points where phi or phi' is discontinuous (jumps or kinks)."""

    def derivative(self, s: np.ndarray) -> np.ndarray:
        """Almost-everywhere derivative phi'(s), for pathwise estimators."""
        raise InapplicableEstimatorError(
            f"{self.name}: payoff is not Lipschitz; the pathwise derivative misses "
            "the Dirac mass at the discontinuity and would be biased (it is 0 a.s.)."
        )


@dataclass(frozen=True)
class EuropeanCall(Payoff):
    """phi(s) = (s - K)^+ : continuous, Lipschitz, kink at K."""

    K: float
    name: str = "call"
    continuous: bool = True
    lipschitz: bool = True

    def __post_init__(self) -> None:
        if not self.K > 0:
            raise ValueError("strike K must be positive")

    def __call__(self, s):
        return np.maximum(np.asarray(s, dtype=float) - self.K, 0.0)

    def derivative(self, s):
        return (np.asarray(s, dtype=float) > self.K).astype(float)

    def singular_points(self):
        return (self.K,)


@dataclass(frozen=True)
class CashDigitalCall(Payoff):
    """phi(s) = 1{s > K} : jump of size 1 at K."""

    K: float
    name: str = "digital"
    continuous: bool = False
    lipschitz: bool = False

    def __post_init__(self) -> None:
        if not self.K > 0:
            raise ValueError("strike K must be positive")

    def __call__(self, s):
        return (np.asarray(s, dtype=float) > self.K).astype(float)

    def singular_points(self):
        return (self.K,)


@dataclass(frozen=True)
class IntervalDigital(Payoff):
    """phi(s) = 1{a <= s <= b} : the paper's Section 4 example with a=100, b=110."""

    a: float
    b: float
    name: str = "interval"
    continuous: bool = False
    lipschitz: bool = False

    def __post_init__(self) -> None:
        if not (self.a > 0 and self.b > self.a):
            raise ValueError(f"need 0 < a < b, got a={self.a}, b={self.b}")

    def __call__(self, s):
        s = np.asarray(s, dtype=float)
        return ((s >= self.a) & (s <= self.b)).astype(float)

    def singular_points(self):
        return (self.a, self.b)


#: Paper's payoff: 1_{[100, 110]}(S_T).
PAPER_PAYOFF = IntervalDigital(100.0, 110.0)

"""Monte Carlo Greeks: finite differences, pathwise and Malliavin estimators.

Computational study of Fournie, Lasry, Lebuchoux, Lions & Touzi (1999),
"Applications of Malliavin calculus to Monte Carlo methods in finance",
Finance and Stochastics 3, 391-412, in the Black-Scholes model.
"""

from . import analytical
from .estimators import METHODS, Estimate, estimate, exact_estimator_moments
from .models import PAPER_PARAMS, BlackScholesParams
from .payoffs import (
    PAPER_PAYOFF,
    CashDigitalCall,
    EuropeanCall,
    InapplicableEstimatorError,
    IntervalDigital,
    Payoff,
)
from .simulation import make_generator, pseudo_normals, sobol_normals, spawn_generators, terminal_price

__all__ = [
    "analytical", "METHODS", "Estimate", "estimate", "exact_estimator_moments",
    "PAPER_PARAMS", "BlackScholesParams", "PAPER_PAYOFF", "CashDigitalCall", "EuropeanCall",
    "InapplicableEstimatorError", "IntervalDigital", "Payoff", "make_generator",
    "pseudo_normals", "sobol_normals", "spawn_generators", "terminal_price",
]
__version__ = "0.1.0"

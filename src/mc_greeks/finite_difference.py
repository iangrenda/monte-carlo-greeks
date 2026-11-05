"""Finite-difference (bump-and-reprice) estimators.

A finite-difference estimator is a linear combination of Monte Carlo prices
at bumped parameter values,

    theta_hat = (1/div) * sum_i c_i * V_hat(params_i),

with the stencils below (h is an absolute bump: in price units for Delta and
Gamma, in volatility units for Vega).

    delta  forward : [V(x+h) - V(x)] / h
    delta  central : [V(x+h) - V(x-h)] / (2h)
    gamma  central : [V(x+h) - 2V(x) + V(x-h)] / h^2
    gamma  forward : [V(x+2h) - 2V(x+h) + V(x)] / h^2
    vega   forward : [V(s+h) - V(s)] / h
    vega   central : [V(s+h) - V(s-h)] / (2h)

Two sources of error must be kept apart:
* truncation (finite-difference) bias: E[theta_hat] - theta_true, which is
  deterministic and equals the same stencil applied to the *exact* price,
  O(h) for forward and O(h^2) for central schemes;
* Monte Carlo variance, which depends on how the V_hat(params_i) are coupled.

Common random numbers (CRN): all V_hat(params_i) use the same normals Z, so
theta_hat is the sample mean of i.i.d. per-path differences and its variance
is Var(sum_i c_i D phi(S_T(params_i; Z)))/(div^2 N). For a Lipschitz payoff
the per-path difference is O(h) and the variance stays bounded as h -> 0.
For a jump payoff the per-path difference is O(1)/h on an event of
probability O(h), giving variance O(1/(N h)) for Delta and O(1/(N h^3)) for
Gamma.

Independent random numbers (IND): each V_hat(params_i) uses its own Z, so
the variance is sum_i c_i^2 Var(D phi(S_T(params_i)))/(div^2 N), which is
O(1/(N h^2)) for first differences and O(1/(N h^4)) for second differences,
whatever the payoff.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import BlackScholesParams
from .payoffs import Payoff
from .simulation import terminal_price

_STENCILS: dict[tuple[str, str], tuple[tuple[float, float], ...]] = {
    # (greek, scheme): ((offset in units of h, coefficient), ...); divisor below
    ("delta", "forward"): ((1.0, 1.0), (0.0, -1.0)),
    ("delta", "central"): ((1.0, 1.0), (-1.0, -1.0)),
    ("gamma", "central"): ((1.0, 1.0), (0.0, -2.0), (-1.0, 1.0)),
    ("gamma", "forward"): ((2.0, 1.0), (1.0, -2.0), (0.0, 1.0)),
    ("vega", "forward"): ((1.0, 1.0), (0.0, -1.0)),
    ("vega", "central"): ((1.0, 1.0), (-1.0, -1.0)),
}


@dataclass(frozen=True)
class Stencil:
    params: tuple[BlackScholesParams, ...]
    coeffs: tuple[float, ...]
    divisor: float

    @property
    def n_evals(self) -> int:
        return len(self.params)


def stencil(p: BlackScholesParams, greek: str, h: float, scheme: str = "central") -> Stencil:
    """Bumped parameter sets, coefficients and divisor for an FD scheme."""
    key = (greek, scheme)
    if key not in _STENCILS:
        raise ValueError(f"no finite-difference stencil for greek={greek!r}, scheme={scheme!r}")
    if not h > 0:
        raise ValueError(f"bump size must be positive, got {h}")
    field = "sigma" if greek == "vega" else "S0"
    base = getattr(p, field)
    pts = _STENCILS[key]
    params = []
    for k, _ in pts:
        value = base + k * h
        if value <= 0:
            raise ValueError(f"bump h={h} makes {field} non-positive ({value})")
        params.append(p.bumped(**{field: value}))
    divisor = h**2 if greek == "gamma" else (2 * h if scheme == "central" else h)
    return Stencil(tuple(params), tuple(c for _, c in pts), divisor)


def fd_crn_samples(
    payoff: Payoff, p: BlackScholesParams, Z: np.ndarray, greek: str, h: float, scheme: str = "central"
) -> np.ndarray:
    """Per-path CRN finite differences: every bumped price uses the same ``Z``."""
    st = stencil(p, greek, h, scheme)
    out = np.zeros(np.shape(Z), dtype=float)
    for q, c in zip(st.params, st.coeffs):
        out += c * q.discount * payoff(terminal_price(q, Z))
    return out / st.divisor


def fd_independent(
    payoff: Payoff, p: BlackScholesParams, Z_list: list[np.ndarray], greek: str, h: float, scheme: str = "central"
) -> tuple[float, float]:
    """Independent-random-number finite difference.

    ``Z_list[i]`` drives the i-th bumped price; the caller guarantees the
    arrays come from independent streams. Returns (estimate, standard error),
    with SE^2 = sum_i c_i^2 s_i^2 / N_i / div^2.
    """
    st = stencil(p, greek, h, scheme)
    if len(Z_list) != st.n_evals:
        raise ValueError(f"need {st.n_evals} independent normal vectors, got {len(Z_list)}")
    from .estimators import blocked_mean_var  # local import avoids a cycle

    est, var = 0.0, 0.0
    for q, c, Z in zip(st.params, st.coeffs, Z_list):
        m, v = blocked_mean_var(lambda z, q=q: q.discount * payoff(terminal_price(q, z)), Z)
        est += c * m
        var += c**2 * v / np.size(Z)
    return est / st.divisor, float(np.sqrt(var)) / st.divisor


def fd_expected_value(price_fn, p: BlackScholesParams, greek: str, h: float, scheme: str = "central") -> float:
    """Expectation of the FD estimator = stencil applied to the exact price.

    ``price_fn(params) -> float`` is typically the analytical price. The
    difference from the analytical Greek is the exact truncation bias
    (the same for CRN and IND, since coupling changes only the variance).
    """
    st = stencil(p, greek, h, scheme)
    return sum(c * price_fn(q) for q, c in zip(st.params, st.coeffs)) / st.divisor

"""Finite-difference estimators: stencils, CRN coupling, bias, variance."""
import numpy as np
import pytest

from conftest import assert_within_se
from mc_greeks import CashDigitalCall, EuropeanCall, IntervalDigital, analytical, estimate
from mc_greeks.estimators import exact_estimator_moments
from mc_greeks.finite_difference import fd_crn_samples, fd_expected_value, fd_independent, stencil
from mc_greeks.payoffs import Payoff
from mc_greeks.simulation import terminal_price


class Linear(Payoff):
    """phi(s) = s; exact Delta = 1, Gamma = 0 (forward contract on S)."""
    name = "linear"
    continuous = True
    lipschitz = True

    def __call__(self, s):
        return np.asarray(s, dtype=float)

    def derivative(self, s):
        return np.ones_like(np.asarray(s, dtype=float))

    def singular_points(self):
        return ()


def test_stencil_shapes(params):
    st = stencil(params, "gamma", 2.0, "central")
    assert [q.S0 for q in st.params] == [102.0, 100.0, 98.0]
    assert st.coeffs == (1.0, -2.0, 1.0) and st.divisor == 4.0
    st = stencil(params, "vega", 0.01, "forward")
    assert [q.sigma for q in st.params] == pytest.approx([0.21, 0.20]) and st.divisor == 0.01
    with pytest.raises(ValueError):
        stencil(params, "delta", 0.0)
    with pytest.raises(ValueError):
        stencil(params, "vega", 0.25, "central")  # sigma - h < 0
    with pytest.raises(ValueError):
        stencil(params, "rho", 1.0)


def test_crn_shares_random_numbers(params, rng):
    """CRN: the per-path difference is computed from ONE Z for all bumps.

    For phi(s) = s, S_T is linear in S0, so the CRN central difference equals
    D * S_T(Z)/S0 path by path (to rounding) -- impossible unless both bumped
    prices used the same Z.
    """
    Z = rng.standard_normal(1000)
    y = fd_crn_samples(Linear(), params, Z, "delta", h=1.0, scheme="central")
    np.testing.assert_allclose(y, params.discount * terminal_price(params, Z) / params.S0, rtol=1e-10)
    # and the CRN Gamma of a linear payoff is zero path by path
    g = fd_crn_samples(Linear(), params, Z, "gamma", h=1.0)
    assert np.max(np.abs(g)) < 1e-9


def test_crn_estimator_reuses_base_normals(params, rng):
    Z = rng.standard_normal(5000)
    f = CashDigitalCall(100)
    e = estimate("FD-CRN-ctr", "delta", f, params, Z, h=1.0)
    manual = (params.discount * (f(terminal_price(params.bumped(S0=101), Z)) - f(terminal_price(params.bumped(S0=99), Z)))).mean() / 2
    assert e.value == pytest.approx(manual, rel=1e-12)


def test_independent_uses_fresh_normals(params):
    """IND: extra bumped prices are driven by new draws from the rng, so the
    estimator differs from the CRN one built on the same base Z, and its
    per-bump sample means are uncorrelated across replications."""
    f = Linear()
    rng = np.random.default_rng(1)
    Z = rng.standard_normal(2000)
    crn = estimate("FD-CRN-ctr", "delta", f, params, Z, h=1.0)
    ind = estimate("FD-IND-ctr", "delta", f, params, Z, rng=rng, h=1.0)
    assert ind.value != pytest.approx(crn.value, rel=1e-6)
    # empirical correlation between the up and down sample means over replications ~ 0
    ups, dns = [], []
    for seed in range(400):
        r = np.random.default_rng(seed)
        z_list = [r.standard_normal(200), r.standard_normal(200)]
        ups.append(terminal_price(params.bumped(S0=101), z_list[0]).mean())
        dns.append(terminal_price(params.bumped(S0=99), z_list[1]).mean())
    assert abs(np.corrcoef(ups, dns)[0, 1]) < 4.5 / np.sqrt(400)
    # IND variance for the linear payoff is large (no cancellation), CRN tiny
    assert ind.stderr > 20 * crn.stderr


def test_independent_requires_rng(params):
    with pytest.raises(ValueError):
        estimate("FD-IND-ctr", "delta", CashDigitalCall(100), params, np.zeros(10), h=1.0)
    with pytest.raises(ValueError):
        fd_independent(CashDigitalCall(100), params, [np.zeros(3)], "delta", 1.0)


@pytest.mark.parametrize("payoff", [EuropeanCall(100), CashDigitalCall(100), IntervalDigital(100, 110)])
@pytest.mark.parametrize("greek,h", [("delta", 1.0), ("gamma", 2.0), ("vega", 0.01)])
def test_fd_bias_is_truncation_error(params, payoff, greek, h):
    """E[FD] = stencil on exact price; for central schemes bias = O(h^2), so
    halving h cuts the bias by ~4 (checked as a ratio in [3, 5])."""
    price = lambda q: analytical.greek(payoff, q, "price")
    truth = analytical.greek(payoff, params, greek)
    b1 = fd_expected_value(price, params, greek, h) - truth
    b2 = fd_expected_value(price, params, greek, h / 2) - truth
    assert 3.0 < b1 / b2 < 5.0


def test_forward_bias_first_order(params):
    f = EuropeanCall(100)
    price = lambda q: analytical.greek(f, q, "price")
    truth = analytical.greek(f, params, "delta")
    b1 = fd_expected_value(price, params, "delta", 2.0, "forward") - truth
    b2 = fd_expected_value(price, params, "delta", 1.0, "forward") - truth
    assert 1.7 < b1 / b2 < 2.3


@pytest.mark.parametrize("method", ["FD-CRN-ctr", "FD-IND-ctr", "FD-CRN-fwd", "FD-IND-fwd"])
@pytest.mark.parametrize("greek,h", [("delta", 1.0), ("vega", 0.01), ("gamma", 5.0)])
def test_fd_mc_matches_its_expectation(params, rng, method, greek, h):
    """Simulated FD estimate is consistent with E[FD] (truth + truncation bias)."""
    f = CashDigitalCall(100)
    Z = rng.standard_normal(400_000)
    e = estimate(method, greek, f, params, Z, rng=rng, h=h)
    scheme = "forward" if method.endswith("fwd") else "central"
    target = fd_expected_value(lambda q: analytical.greek(f, q, "price"), params, greek, h, scheme)
    assert_within_se(e.value, target, e.stderr)


def test_exact_variance_matches_empirical(params):
    """Quadrature per-path variance agrees with the replication variance."""
    f = IntervalDigital(100, 110)
    for method in ("FD-CRN-ctr", "FD-IND-ctr"):
        m = exact_estimator_moments(method, "delta", f, params, h=1.0)
        ests = []
        for seed in range(300):
            r = np.random.default_rng(seed)
            ests.append(estimate(method, "delta", f, params, r.standard_normal(2000), rng=r, h=1.0).value)
        v_emp, v_th = np.var(ests, ddof=1), m["var1"] / 2000
        # chi-square-based 99.9% band for a sample variance with 299 df: about [0.75, 1.30]
        assert 0.72 < v_emp / v_th < 1.32, (method, v_emp / v_th)


def test_crn_variance_scaling_for_digital(params):
    """For a jump payoff N*Var(CRN central Delta) ~ C/h: shrinking h by 4 raises it ~4x."""
    f = CashDigitalCall(100)
    v1 = exact_estimator_moments("FD-CRN-ctr", "delta", f, params, h=0.4)["var1"]
    v2 = exact_estimator_moments("FD-CRN-ctr", "delta", f, params, h=0.1)["var1"]
    assert 3.5 < v2 / v1 < 4.5
    # and for the smooth call, CRN variance stays bounded as h -> 0
    c = EuropeanCall(100)
    w1 = exact_estimator_moments("FD-CRN-ctr", "delta", c, params, h=0.4)["var1"]
    w2 = exact_estimator_moments("FD-CRN-ctr", "delta", c, params, h=0.1)["var1"]
    assert w2 / w1 == pytest.approx(1.0, abs=0.02)
    # IND variance ~ 1/h^2 for any payoff
    u1 = exact_estimator_moments("FD-IND-ctr", "delta", c, params, h=0.4)["var1"]
    u2 = exact_estimator_moments("FD-IND-ctr", "delta", c, params, h=0.1)["var1"]
    assert u2 / u1 == pytest.approx(16.0, rel=0.02)

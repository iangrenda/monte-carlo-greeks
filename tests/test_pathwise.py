import numpy as np
import pytest

from conftest import assert_within_se
from mc_greeks import (BlackScholesParams, CashDigitalCall, EuropeanCall, InapplicableEstimatorError, IntervalDigital,
                       analytical, estimate)
from mc_greeks.estimators import exact_estimator_moments


@pytest.mark.parametrize("K", [80.0, 100.0, 120.0])
@pytest.mark.parametrize("g", ["delta", "vega"])
def test_pathwise_call_exact(params, K, g):
    m = exact_estimator_moments("Pathwise", g, EuropeanCall(K), params)
    assert m["mean"] == pytest.approx(m["truth"], rel=1e-8)


@pytest.mark.parametrize("g", ["delta", "vega"])
def test_pathwise_call_mc(params, rng, g):
    c = EuropeanCall(100)
    e = estimate("Pathwise", g, c, params, rng.standard_normal(400_000))
    assert_within_se(e.value, analytical.greek(c, params, g), e.stderr)


@pytest.mark.parametrize("payoff", [CashDigitalCall(100), IntervalDigital(100, 110)])
@pytest.mark.parametrize("g", ["delta", "gamma", "vega"])
def test_pathwise_rejected_for_digitals(params, payoff, g):
    """Naive pathwise digital Greeks would be identically 0; we refuse them."""
    with pytest.raises(InapplicableEstimatorError):
        estimate("Pathwise", g, payoff, params, np.ones(10))


def test_pathwise_call_gamma_rejected(params):
    with pytest.raises(InapplicableEstimatorError):
        estimate("Pathwise", "gamma", EuropeanCall(100), params, np.ones(10))


def test_pathwise_lower_variance_than_malliavin_for_call(params):
    """Known result we re-check: for the smooth call Delta, the pathwise
    estimator has smaller variance than the (global) Malliavin weight."""
    c = EuropeanCall(100)
    pw = exact_estimator_moments("Pathwise", "delta", c, params)["var1"]
    ml = exact_estimator_moments("Malliavin", "delta", c, params)["var1"]
    assert pw < ml


def test_discounting_consistent(params, rng):
    """Every estimator includes exp(-rT): doubling r changes the estimate in
    line with the analytical Greek (catches a missing discount factor)."""
    p2 = BlackScholesParams(100, 0.3, 0.2, 1.0)
    c = EuropeanCall(100)
    Z = rng.standard_normal(400_000)
    for method in ("Pathwise", "Malliavin"):
        e = estimate(method, "delta", c, p2, Z)
        assert_within_se(e.value, analytical.greek(c, p2, "delta"), e.stderr)
    f = CashDigitalCall(100)
    e = estimate("MC", "price", f, p2, Z)
    assert_within_se(e.value, analytical.greek(f, p2, "price"), e.stderr)
    e = estimate("FD-CRN-ctr", "vega", f, p2, Z, h=0.005)
    from mc_greeks.finite_difference import fd_expected_value
    assert_within_se(e.value, fd_expected_value(lambda q: analytical.greek(f, q, "price"), p2, "vega", 0.005), e.stderr)

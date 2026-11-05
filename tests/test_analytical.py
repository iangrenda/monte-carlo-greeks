"""Deterministic checks of the closed-form benchmarks."""
import math

import numpy as np
import pytest
from scipy import integrate, stats

from mc_greeks import BlackScholesParams, CashDigitalCall, EuropeanCall, IntervalDigital, analytical
from mc_greeks.analytical import greek
from mc_greeks.payoffs import Payoff

PAPER_TARGETS = {"price": 0.16823763, "delta": -0.00133486, "gamma": -0.00038876, "vega": -0.77751371}


def test_paper_interval_digital_benchmarks(params):
    g = analytical.greeks(IntervalDigital(100, 110), params)
    for k, v in PAPER_TARGETS.items():
        assert g[k] == pytest.approx(v, abs=5e-9), k


def test_call_values_standard(params):
    # d1 = 0.6 exactly for S0 = K = 100, r = 0.1, sigma = 0.2, T = 1
    c = EuropeanCall(100.0)
    assert analytical.d1(params, 100.0) == pytest.approx(0.6)
    assert greek(c, params, "delta") == pytest.approx(stats.norm.cdf(0.6))
    assert greek(c, params, "gamma") == pytest.approx(stats.norm.pdf(0.6) / 20.0)
    assert greek(c, params, "vega") == pytest.approx(100 * stats.norm.pdf(0.6))
    expected = 100 * stats.norm.cdf(0.6) - 100 * math.exp(-0.1) * stats.norm.cdf(0.4)
    assert greek(c, params, "price") == pytest.approx(expected, rel=1e-14)


def test_digital_price_by_integration(params):
    """V = e^{-rT} P(S_T > K), integrating the lognormal density numerically."""
    K = 105.0
    mu = math.log(params.S0) + (params.r - 0.5 * params.sigma**2) * params.T
    s = params.sigma * params.sqrtT
    prob = integrate.quad(lambda x: stats.lognorm.pdf(x, s, scale=math.exp(mu)), K, np.inf)[0]
    assert greek(CashDigitalCall(K), params, "price") == pytest.approx(params.discount * prob, rel=1e-9)


def test_call_digital_asset_identity(params):
    """call + K * digital = asset-or-nothing = S0 Phi(d1)."""
    K = 95.0
    lhs = greek(EuropeanCall(K), params, "price") + K * greek(CashDigitalCall(K), params, "price")
    assert lhs == pytest.approx(params.S0 * stats.norm.cdf(analytical.d1(params, K)), rel=1e-12)


def test_interval_equals_difference_of_digitals(params):
    for g in ("price", "delta", "gamma", "vega"):
        diff = greek(CashDigitalCall(100), params, g) - greek(CashDigitalCall(110), params, g)
        assert greek(IntervalDigital(100, 110), params, g) == pytest.approx(diff, rel=1e-14)


@pytest.mark.parametrize("payoff", [EuropeanCall(90), EuropeanCall(120), CashDigitalCall(100),
                                    CashDigitalCall(80), IntervalDigital(100, 110)])
@pytest.mark.parametrize("p", [BlackScholesParams(100, 0.1, 0.2, 1.0), BlackScholesParams(100, 0.03, 0.45, 0.25)])
def test_greeks_match_finite_differences_of_closed_form(payoff, p):
    """Analytical Greeks vs Richardson-extrapolated central differences of the
    noise-free closed-form price (truncation error O(h^4))."""
    price = lambda q: greek(payoff, q, "price")

    def d_s(h):
        return (price(p.bumped(S0=p.S0 + h)) - price(p.bumped(S0=p.S0 - h))) / (2 * h)

    def dd_s(h):
        return (price(p.bumped(S0=p.S0 + h)) - 2 * price(p) + price(p.bumped(S0=p.S0 - h))) / h**2

    def d_sig(h):
        return (price(p.bumped(sigma=p.sigma + h)) - price(p.bumped(sigma=p.sigma - h))) / (2 * h)

    rich = lambda f, h: (4 * f(h / 2) - f(h)) / 3
    h = 2e-3 * p.S0
    assert greek(payoff, p, "delta") == pytest.approx(rich(d_s, h), rel=1e-7, abs=1e-13)
    assert greek(payoff, p, "gamma") == pytest.approx(rich(dd_s, h), rel=1e-7, abs=1e-13)
    assert greek(payoff, p, "vega") == pytest.approx(rich(d_sig, 2e-3), rel=1e-6, abs=1e-10)


def test_vega_is_per_unit_vol(params):
    """Per unit sigma: V(0.21) - V(0.20) ~ 0.01 * Vega (not 1 * Vega)."""
    c = EuropeanCall(100)
    dv = greek(c, params.bumped(sigma=0.21), "price") - greek(c, params, "price")
    assert dv == pytest.approx(0.01 * greek(c, params, "vega"), rel=0.02)


def test_bs_vega_gamma_identity(params):
    """For any European payoff in Black-Scholes: Vega = S0^2 sigma T Gamma."""
    for f in (EuropeanCall(105), CashDigitalCall(95), IntervalDigital(100, 110)):
        lhs = greek(f, params, "vega")
        rhs = params.S0**2 * params.sigma * params.T * greek(f, params, "gamma")
        assert lhs == pytest.approx(rhs, rel=1e-12)


def test_unregistered_payoff_and_unknown_greek(params):
    class Other(Payoff):
        def __call__(self, s):
            return np.asarray(s)

        def singular_points(self):
            return ()

    with pytest.raises(NotImplementedError):
        greek(Other(), params, "price")
    with pytest.raises(ValueError):
        greek(EuropeanCall(100), params, "theta")

"""Malliavin weights: exact (quadrature) checks and Monte Carlo convergence."""
import numpy as np
import pytest

from conftest import assert_within_se
from mc_greeks import BlackScholesParams, CashDigitalCall, EuropeanCall, IntervalDigital, analytical, estimate
from mc_greeks.estimators import exact_estimator_moments
from mc_greeks.malliavin import malliavin_weights

PAYOFFS = [IntervalDigital(100, 110), CashDigitalCall(100), CashDigitalCall(120), EuropeanCall(100), EuropeanCall(85)]
PARAMS = [BlackScholesParams(100, 0.10, 0.20, 1.0), BlackScholesParams(100, 0.02, 0.10, 0.25),
          BlackScholesParams(90, 0.05, 0.50, 2.0)]


@pytest.mark.parametrize("payoff", PAYOFFS)
@pytest.mark.parametrize("p", PARAMS)
@pytest.mark.parametrize("g", ["delta", "gamma", "vega"])
def test_weights_exact_in_expectation(payoff, p, g):
    """E[D phi(S_T) pi] by quadrature equals the closed-form Greek (no MC noise).

    This is the sharpest check of the weight formulas: any error in a sign,
    power of sigma/T or the 1/sigma term breaks it at the 1e-8 level.
    """
    m = exact_estimator_moments("Malliavin", g, payoff, p)
    assert m["mean"] == pytest.approx(m["truth"], rel=1e-7, abs=1e-12)


@pytest.mark.filterwarnings("ignore::scipy.integrate.IntegrationWarning")
def test_weights_have_mean_zero(params):
    """E[pi] = 0 for each Greek weight (derivative of a constant payoff is 0)."""
    from mc_greeks.estimators import gaussian_moments
    for g in ("delta", "gamma", "vega"):
        m1, _ = gaussian_moments(lambda Z: malliavin_weights(params, Z)[g])
        assert abs(m1) < 1e-12


def test_gamma_weight_is_vega_weight_scaled(params, rng):
    Z = rng.standard_normal(10)
    w = malliavin_weights(params, Z)
    np.testing.assert_allclose(w["gamma"], w["vega"] / (params.S0**2 * params.sigma * params.T), rtol=1e-14)


@pytest.mark.parametrize("g", ["price", "delta", "gamma", "vega"])
def test_mc_converges_paper_case(params, rng, g):
    f = IntervalDigital(100, 110)
    Z = rng.standard_normal(1_000_000)
    method = "MC" if g == "price" else "Malliavin"
    e = estimate(method, g, f, params, Z)
    assert_within_se(e.value, analytical.greek(f, params, g), e.stderr)


def test_wrong_weight_would_fail(params, rng):
    """Sanity check that the tolerance is tight: dropping the -1/sigma term in the
    Vega weight produces an estimate many SEs away from the truth."""
    f = IntervalDigital(100, 110)
    Z = rng.standard_normal(1_000_000)
    from mc_greeks.simulation import terminal_brownian, terminal_price
    W = terminal_brownian(params, Z)
    y = params.discount * f(terminal_price(params, Z)) * (W**2 / (params.sigma * params.T) - W)
    z = abs(y.mean() - analytical.greek(f, params, "vega")) / (y.std() / np.sqrt(y.size))
    assert z > 20


def test_localized_call_unbiased_and_lower_variance(params):
    c = EuropeanCall(100)
    for g in ("delta", "gamma", "vega"):
        glob = exact_estimator_moments("Malliavin", g, c, params)
        loc = exact_estimator_moments("Malliavin-loc", g, c, params, loc_width=10.0)
        assert loc["mean"] == pytest.approx(loc["truth"], rel=1e-8)
        assert loc["var1"] < glob["var1"]


def test_localized_mc(params, rng):
    c = EuropeanCall(100)
    Z = rng.standard_normal(400_000)
    e = estimate("Malliavin-loc", "gamma", c, params, Z, loc_width=10.0)
    assert_within_se(e.value, analytical.greek(c, params, "gamma"), e.stderr)


def test_localization_rejected_for_digital(params):
    from mc_greeks import InapplicableEstimatorError
    with pytest.raises(InapplicableEstimatorError):
        estimate("Malliavin-loc", "delta", CashDigitalCall(100), params, np.zeros(10), loc_width=5.0)

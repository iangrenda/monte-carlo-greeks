"""Asian digital estimators: exact reductions, per-path derivative checks,
quadrature references, simulation moments, RQMC layout and reproducibility."""
import math

import numpy as np
import pytest
from scipy import integrate, stats

from conftest import assert_within_se
from mc_greeks import BlackScholesParams, CashDigitalCall, InapplicableEstimatorError, analytical
from mc_greeks.asian import (AsianDigital, asian_estimate, cpw_first_samples, cpw_last_samples, fd_samples,
                             lr_samples, malliavin_avg_samples, rqmc_layout, simulate_paths)
from mc_greeks.malliavin import malliavin_samples

P0 = BlackScholesParams(100.0, 0.10, 0.20, 1.0)
P1 = BlackScholesParams(95.0, 0.03, 0.35, 0.5)


@pytest.mark.parametrize("p", [P0, P1])
@pytest.mark.parametrize("g", ["price", "delta", "gamma", "vega"])
def test_m1_conditional_estimators_are_exact(p, g):
    """With one date, both conditional estimators equal the closed-form digital Greek on every path."""
    f, d = AsianDigital(100.0, 1), CashDigitalCall(100.0)
    Z = np.random.default_rng(0).standard_normal((50, 1))
    truth = analytical.greek(d, p, g)
    np.testing.assert_allclose(cpw_first_samples(f, p, Z, g), truth, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(cpw_last_samples(f, p, Z, g), truth, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("g", ["price", "delta", "gamma", "vega"])
def test_m1_weights_reduce_to_european_malliavin(g):
    f, d = AsianDigital(100.0, 1), CashDigitalCall(100.0)
    Z = np.random.default_rng(1).standard_normal((200, 1))
    ref = malliavin_samples(d, P0, Z[:, 0], g)
    np.testing.assert_allclose(lr_samples(f, P0, Z, g), ref, rtol=1e-12, atol=1e-15)
    if g in ("price", "delta"):
        np.testing.assert_allclose(malliavin_avg_samples(f, P0, Z, g), ref, rtol=1e-10, atol=1e-15)


@pytest.mark.parametrize("fn", [cpw_first_samples, cpw_last_samples])
@pytest.mark.parametrize("m", [2, 5, 12])
def test_conditional_derivatives_match_per_path_finite_differences(fn, m):
    """The hand-derived pathwise derivatives of the conditional price equal a
    Richardson finite difference of the same smooth function on fixed Z."""
    f = AsianDigital(102.0, m)
    Z = np.random.default_rng(2).standard_normal((300, m))
    price = lambda q: fn(f, q, Z, "price")
    p = P1

    def rich(g, h):
        if g == "delta":
            d = lambda e: (price(p.bumped(S0=p.S0 + e)) - price(p.bumped(S0=p.S0 - e))) / (2 * e)
        elif g == "gamma":
            d = lambda e: (price(p.bumped(S0=p.S0 + e)) - 2 * price(p) + price(p.bumped(S0=p.S0 - e))) / e**2
        else:
            d = lambda e: (price(p.bumped(sigma=p.sigma + e)) - price(p.bumped(sigma=p.sigma - e))) / (2 * e)
        return (4 * d(h / 2) - d(h)) / 3

    for g, h in (("delta", 0.05), ("gamma", 0.2), ("vega", 1e-3)):
        est, num = fn(f, p, Z, g), rich(g, h)
        scale = np.abs(num).max() + 1e-12
        assert np.max(np.abs(est - num)) / scale < 2e-4, g


def _quad_ref(f, p, g):
    """m = 2: CPW-first depends on Z_2 only, so 1-D quadrature gives a near-exact reference."""
    fn = lambda z: float(cpw_first_samples(f, p, np.array([[0.0, z]]), g)[0]) * stats.norm.pdf(z)
    return integrate.quad(fn, -12, 12, limit=400, epsabs=1e-15, epsrel=1e-12)[0]


@pytest.mark.parametrize("K", [95.0, 110.0])
@pytest.mark.parametrize("g", ["price", "delta", "gamma", "vega"])
def test_all_estimators_unbiased_against_m2_quadrature(K, g):
    f = AsianDigital(K, 2)
    ref = _quad_ref(f, P0, g)
    rng = np.random.default_rng(3)
    for method in ("LR", "Malliavin-avg", "CPW-last", "CPW-first"):
        try:
            e = asian_estimate(method, g, f, P0, 2**18, rng, "MC")
        except InapplicableEstimatorError:
            continue
        assert_within_se(e.value, ref, e.stderr)
    e = asian_estimate("CPW-first", g, f, P0, 2**14, rng, "RQMC")
    assert_within_se(e.value, ref, e.stderr, k=5.0)


def test_fd_smooth_matches_its_expectation_m2():
    """FD-smooth estimates the stencil of the smoothed price, so it is consistent with
    the stencil applied to quadrature prices (truncation bias included)."""
    f = AsianDigital(100.0, 2)
    h = 2.0
    target = (_quad_ref(f, P0.bumped(S0=102.0), "price") - _quad_ref(f, P0.bumped(S0=98.0), "price")) / (2 * h)
    e = asian_estimate("FD-smooth", "delta", f, P0, 2**17, np.random.default_rng(4), "MC", h=h)
    assert_within_se(e.value, target, e.stderr)


def test_fd_uses_common_random_numbers():
    """CRN: the bumped prices are evaluated on the same Z (checked by recomputing by hand)."""
    f = AsianDigital(100.0, 4)
    Z = np.random.default_rng(5).standard_normal((1000, 4))
    y = fd_samples(f, P0, Z, "delta", 1.0, base="raw")
    up, _, _ = simulate_paths(P0.bumped(S0=101.0), 4, Z)
    dn, _, _ = simulate_paths(P0.bumped(S0=99.0), 4, Z)
    manual = P0.discount * ((up.mean(1) >= 100.0).astype(float) - (dn.mean(1) >= 100.0).astype(float)) / 2.0
    np.testing.assert_array_equal(y, manual)


def test_path_simulation_moments():
    m = 6
    Z = np.random.default_rng(6).standard_normal((400_000, m))
    S, W, t = simulate_paths(P0, m, Z)
    mean = P0.S0 * np.exp(P0.r * t)
    sd = mean * np.sqrt(np.exp(P0.sigma**2 * t) - 1)
    assert np.all(np.abs(S.mean(0) - mean) <= 4.5 * sd / math.sqrt(len(Z)))
    cov = np.cov(W[:, 1], W[:, 4])[0, 1]  # Cov(W_s, W_t) = min(s, t)
    assert abs(cov - t[1]) < 4.5 * math.sqrt((t[1] * t[4] + t[1] ** 2) / len(Z))


def test_rqmc_layout_and_reproducibility():
    for n in (1000, 2**15, 100_000):
        per, k = rqmc_layout(n)
        assert per & (per - 1) == 0 and k >= 8 and n - per < per * k <= n
    f = AsianDigital(100.0, 8)
    a = asian_estimate("CPW-first", "delta", f, P0, 4096, np.random.default_rng(9), "RQMC")
    b = asian_estimate("CPW-first", "delta", f, P0, 4096, np.random.default_rng(9), "RQMC")
    assert a.value == b.value and a.stderr == b.stderr


def test_adaptive_selector_guard_and_fallback():
    """The selector returns an applicable candidate; with an impossible nonzero
    requirement every candidate is excluded and the documented fallback is used;
    pilot cost is charged to the budget."""
    import pandas as pd
    from mc_greeks.asian_experiments import DEFAULT_FALLBACK, CostModel, Scenario, adaptive_select, CANDIDATES, GREEKS3
    sc = Scenario(0, m=4, moneyness=1.0, sigma=0.2, T=1.0)
    cost = CostModel(pd.DataFrame([{"m": 4, "method": m, "sampler": s, "greek": g, "sec_per_path": 1e-7}
                                   for (m, s) in CANDIDATES for g in GREEKS3]))
    best, preds, remaining, info = adaptive_select(sc, "delta", cost, 0.002, 0.1, np.random.default_rng(0),
                                                   accounting="nominal")
    assert best in preds and remaining == pytest.approx(0.0018)
    # charged accounting: the final run gets the budget minus the pilot actually charged
    best, preds, remaining, info = adaptive_select(sc, "delta", cost, 0.002, 0.1, np.random.default_rng(0))
    assert remaining == pytest.approx(0.002 - info["pilot_charged"]) and info["pilot_wall"] > 0
    best, preds, _, _ = adaptive_select(sc, "gamma", cost, 0.002, 0.1, np.random.default_rng(0), min_nonzero=10**9)
    assert best == DEFAULT_FALLBACK


def test_nonzero_counter():
    f = AsianDigital(1000.0, 4)  # deep OTM: raw-payoff estimators see no exercise
    e = asian_estimate("LR", "delta", f, P0, 512, np.random.default_rng(0), "MC")
    assert e.n_nonzero == 0 and e.stderr == 0.0
    e = asian_estimate("CPW-first", "delta", AsianDigital(100.0, 4), P0, 512, np.random.default_rng(0), "MC")
    assert e.n_nonzero == 512


def test_invalid_inputs():
    with pytest.raises(ValueError):
        AsianDigital(-1.0, 4)
    with pytest.raises(ValueError):
        AsianDigital(100.0, 0)
    with pytest.raises(InapplicableEstimatorError):
        asian_estimate("Malliavin-avg", "vega", AsianDigital(100.0, 4), P0, 100, np.random.default_rng(0))
    with pytest.raises(InapplicableEstimatorError):
        asian_estimate("FD-raw", "price", AsianDigital(100.0, 4), P0, 100, np.random.default_rng(0), h=1.0)

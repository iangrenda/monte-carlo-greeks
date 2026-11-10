"""Phase 7: piecewise-constant volatility curves (reduction, closed forms, derivatives, constructions)."""
import math

import numpy as np
import pytest
from scipy import special
from scipy.stats import qmc

from mc_greeks import analytical
from mc_greeks import term_structure as ts
from mc_greeks.asian import AsianDigital, cpw_first_samples, simulate_paths
from mc_greeks.asian_qmc import construction_matrix as flat_construction
from mc_greeks.models import BlackScholesParams

CURVES = {
    "flat": (0.25,) * 8,
    "rising": tuple(np.linspace(0.15, 0.40, 8)),
    "falling": tuple(np.linspace(0.40, 0.15, 8)),
    "event": (0.2, 0.2, 0.2, 0.6, 0.2, 0.2, 0.2, 0.2),
}


@pytest.fixture(params=list(CURVES))
def curve(request):
    return ts.TSParams(100.0, 0.03, 1.2, CURVES[request.param])


# --- reduction to the constant-volatility code ----------------------------------------------------

def test_flat_curve_reproduces_the_phase2_code():
    m = 12
    p0 = BlackScholesParams(100.0, 0.03, 0.27, 0.9)
    p = ts.TSParams.flat(100.0, 0.03, 0.27, 0.9, m)
    f = AsianDigital(103.0, m)
    Z = np.random.default_rng(0).standard_normal((300, m))
    np.testing.assert_allclose(ts.simulate_paths(p, Z), simulate_paths(p0, m, Z)[0], rtol=1e-12)
    out = ts.cpw_first(f, p, Z)
    for g in ("price", "delta", "gamma", "vega"):
        np.testing.assert_allclose(out[g], cpw_first_samples(f, p0, Z, g), rtol=1e-10, atol=1e-14)


def test_flat_constructions_match_the_phase3_constructions():
    m, sig, T = 10, 0.3, 1.0
    p = ts.TSParams.flat(100.0, 0.02, sig, T, m)
    for kind in ("seq", "bb", "pca"):
        np.testing.assert_allclose(np.abs(ts.construction_matrix(kind, p)),
                                   sig * np.abs(flat_construction(kind, m, T / m)), rtol=1e-8, atol=1e-12)


# --- European digital: closed form and the likelihood-ratio weight ---------------------------------

def test_european_closed_form_reduces_to_black_scholes():
    p = ts.TSParams.flat(100.0, 0.03, 0.3, 1.5, 6)
    bs = BlackScholesParams(100.0, 0.03, 0.3, 1.5)
    e = ts.european_digital(p, 104.0)
    assert e["price"] == pytest.approx(analytical.digital_price(bs, 104.0), rel=1e-12)
    assert e["delta"] == pytest.approx(analytical.digital_delta(bs, 104.0), rel=1e-12)
    assert e["gamma"] == pytest.approx(analytical.digital_gamma(bs, 104.0), rel=1e-12)
    assert e["bucket_vega"].sum() == pytest.approx(analytical.digital_vega(bs, 104.0), rel=1e-12)


def test_european_closed_form_derivatives(curve):
    K, e = 104.0, ts.european_digital(curve, 104.0)
    h = 1e-4 * curve.S0
    up, dn = ts.european_digital(curve.with_S0(curve.S0 + h), K), ts.european_digital(curve.with_S0(curve.S0 - h), K)
    assert e["delta"] == pytest.approx((up["price"] - dn["price"]) / (2 * h), rel=1e-6)
    assert e["gamma"] == pytest.approx((up["delta"] - dn["delta"]) / (2 * h), rel=1e-6)
    for k in range(curve.m):
        s = np.array(curve.sigmas)
        s[k] += 1e-6
        s2 = np.array(curve.sigmas)
        s2[k] -= 1e-6
        fd = (ts.european_digital(curve.with_sigmas(s), K)["price"] - ts.european_digital(curve.with_sigmas(s2), K)["price"]) / 2e-6
        assert e["bucket_vega"][k] == pytest.approx(fd, rel=1e-5, abs=1e-10)


def test_european_lr_delta_weight_is_unbiased(curve):
    """The likelihood-ratio weight X / (S0 V) with X = sum_j a_j Z_j, for a time-varying curve."""
    K = 104.0
    Z = np.random.default_rng(1).standard_normal((400_000, curve.m))
    y = ts.european_lr_delta_samples(curve, K, Z)
    exact = ts.european_digital(curve, K)["delta"]
    assert abs(y.mean() - exact) < 4.5 * y.std(ddof=1) / math.sqrt(len(y))


# --- Asian digital, CPW-first ------------------------------------------------------------------------

def test_one_date_asian_is_the_european_digital():
    p = ts.TSParams(100.0, 0.03, 0.8, (0.35,))
    f = AsianDigital(97.0, 1)
    out = ts.cpw_first(f, p, np.zeros((1, 1)), bucket=True)
    e = ts.european_digital(p, 97.0)
    for g in ("price", "delta", "gamma"):
        assert out[g][0] == pytest.approx(e[g], rel=1e-12)
    assert out["bucket_vega"][0, 0] == pytest.approx(e["bucket_vega"][0], rel=1e-12)


def test_per_path_greeks_are_derivatives_of_the_smoothed_price(curve):
    f = AsianDigital(102.0, curve.m)
    Z = np.random.default_rng(2).standard_normal((200, curve.m))
    out = ts.cpw_first(f, curve, Z, bucket=True)
    h = 1e-5 * curve.S0                      # central differences: O(h^2) truncation, ~1e-8 relative here
    up, dn = ts.cpw_first(f, curve.with_S0(curve.S0 + h), Z), ts.cpw_first(f, curve.with_S0(curve.S0 - h), Z)
    np.testing.assert_allclose(out["delta"], (up["price"] - dn["price"]) / (2 * h), rtol=1e-6, atol=1e-11)
    np.testing.assert_allclose(out["gamma"], (up["delta"] - dn["delta"]) / (2 * h), rtol=1e-6, atol=1e-11)
    for k in range(curve.m):
        s, s2 = np.array(curve.sigmas), np.array(curve.sigmas)
        s[k] += 1e-6
        s2[k] -= 1e-6
        fd = (ts.smoothed_price(f, curve.with_sigmas(s), Z) - ts.smoothed_price(f, curve.with_sigmas(s2), Z)) / 2e-6
        np.testing.assert_allclose(out["bucket_vega"][:, k], fd, rtol=2e-5, atol=1e-9)
    np.testing.assert_allclose(out["vega"], out["bucket_vega"].sum(axis=1), rtol=1e-12)


def test_smoothed_estimators_are_unbiased_against_the_raw_payoff(curve):
    f = AsianDigital(102.0, curve.m)
    rng = np.random.default_rng(3)
    Z = rng.standard_normal((400_000, curve.m))
    sm, raw = ts.cpw_first(f, curve, Z)["price"], ts.raw_payoff(f, curve, Z)
    z = (sm.mean() - raw.mean()) / math.hypot(sm.std() / math.sqrt(len(Z)), raw.std() / math.sqrt(len(Z)))
    assert abs(z) < 4.5
    # Delta against a common-random-number central difference of the raw price
    h = 0.5
    up, dn = ts.raw_payoff(f, curve.with_S0(curve.S0 + h), Z), ts.raw_payoff(f, curve.with_S0(curve.S0 - h), Z)
    d = ts.cpw_first(f, curve, Z)["delta"]
    fd = (up - dn) / (2 * h)
    assert abs(d.mean() - fd.mean()) < 4.5 * math.hypot(d.std(), fd.std()) / math.sqrt(len(Z)) + 1e-4  # O(h^2) bias


# --- constructions ------------------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ts.CONSTRUCTIONS)
def test_constructions_reproduce_the_residual_covariance(curve, kind):
    L = ts.construction_matrix(kind, curve)
    np.testing.assert_allclose(L @ L.T, ts.residual_cov(curve), rtol=1e-10, atol=1e-13)


def test_bridge_on_arbitrary_times():
    t = np.array([0.1, 0.25, 0.3, 0.7, 0.75, 1.2, 1.9])
    L = ts.bridge_matrix_times(t)
    np.testing.assert_allclose(L @ L.T, np.minimum.outer(t, t), atol=1e-13)
    assert np.count_nonzero(np.abs(L[:, 0]) > 0) == len(t)       # first coordinate: terminal value, drives all dates


@pytest.mark.parametrize("kind", ts.CONSTRUCTIONS)
def test_rqmc_with_each_construction_is_unbiased(curve, kind):
    f = AsianDigital(102.0, curve.m)
    L = ts.construction_matrix(kind, curve)
    rng = np.random.default_rng(4)
    eps = np.finfo(float).eps
    means = []
    for _ in range(16):
        xi = special.ndtri(np.clip(qmc.Sobol(curve.m - 1, scramble=True, seed=rng).random(1024), eps, 1 - eps))
        Z = ts.increments_from_residual(xi @ L.T, curve)
        means.append([v.mean() for v in ts.cpw_first(f, curve, Z).values()])
    means = np.array(means)
    Zr = np.random.default_rng(5).standard_normal((400_000, curve.m))
    ref = ts.cpw_first(f, curve, Zr)
    for k, g in enumerate(("price", "delta", "gamma", "vega")):
        se = math.hypot(means[:, k].std(ddof=1) / 4.0, ref[g].std() / math.sqrt(len(Zr)))
        assert abs(means[:, k].mean() - ref[g].mean()) < 4.5 * se, (kind, g)


# --- Greek-aware rotations -----------------------------------------------------------------------------

def test_design_point_is_on_the_boundary_and_satisfies_kkt(curve):
    f = AsianDigital(108.0, curve.m)
    B = ts.design_point(f, curve)
    R, Abar, w = ts._state(f, curve, B)
    assert abs(math.log(Abar / f.K)) < 1e-10
    y = np.linalg.solve(ts.residual_cov(curve), B)
    a = w[1:]
    assert abs(abs(y @ a) / (np.linalg.norm(y) * np.linalg.norm(a)) - 1) < 1e-8


def test_rotations_are_orthogonal_and_keep_the_law(curve):
    f = AsianDigital(102.0, curve.m)
    L = ts.construction_matrix("pca", curve)
    for U in (ts.weighted_rotation(f, curve), ts.gradient_rotation(f, curve, 512, np.random.default_rng(6))):
        np.testing.assert_allclose(U.T @ U, np.eye(curve.m - 1), atol=1e-10)
        np.testing.assert_allclose((L @ U) @ (L @ U).T, ts.residual_cov(curve), rtol=1e-10, atol=1e-13)


def test_training_free_and_trained_leading_directions_agree(curve):
    f = AsianDigital(102.0, curve.m)
    Ut = ts.gradient_rotation(f, curve, 8192, np.random.default_rng(7))
    Uw = ts.weighted_rotation(f, curve)
    assert abs(Ut[:, 0] @ Uw[:, 0]) > 0.98


def test_weighted_rotation_reduces_to_the_phase6_rotation_for_a_flat_curve():
    from mc_greeks import training_free as tf
    m = 10
    f = AsianDigital(104.0, m)
    p = ts.TSParams.flat(100.0, 0.03, 0.3, 1.0, m)
    U6, _ = tf.training_free_rotation("weighted-dominant", f, BlackScholesParams(100.0, 0.03, 0.3, 1.0))
    U7 = ts.weighted_rotation(f, p)
    # both use the PCA basis with the same sign convention (columns summing to >= 0)
    from mc_greeks.joint_greeks import pca_factor
    np.testing.assert_allclose(ts.construction_matrix("pca", p), 0.3 * pca_factor(m, 0.1), atol=1e-12)
    for k in range(3):
        assert abs(U7[:, k] @ U6[:, k]) > 0.9999, k

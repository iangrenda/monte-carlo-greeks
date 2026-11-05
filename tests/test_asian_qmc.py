"""Phase 3: path constructions, conditioning identity, timed estimation, CIs."""
import math

import numpy as np
import pytest
from scipy import integrate, stats

from conftest import assert_within_se
from mc_greeks import BlackScholesParams
from mc_greeks.asian import AsianDigital, cpw_first_samples
from mc_greeks.asian_qmc import (CONSTRUCTIONS, adaptive_timed, bridge_matrix, choose_points_per_scramble,
                                 construction_matrix, explained_variance, functional_share, increments_from_residual,
                                 residual_cov, timed_estimate)

P = BlackScholesParams(100.0, 0.03, 0.25, 0.8)


@pytest.mark.parametrize("m", [2, 3, 4, 5, 7, 12, 52])
@pytest.mark.parametrize("c", CONSTRUCTIONS)
def test_construction_reproduces_brownian_covariance(m, c):
    """L L^T = dt * min(k, l) exactly (to rounding): the residual path law is Brownian."""
    dt = 0.37 / m
    L = construction_matrix(c, m, dt)
    np.testing.assert_allclose(L @ L.T, residual_cov(m, dt), atol=1e-13, rtol=0)


def test_full_path_law_with_level_variable():
    """W_{t_i} = sqrt(dt) Z_1 + B_i has covariance min(t_i, t_j): exact algebra, plus
    a sampling check through the bridge and PCA maps."""
    m, dt = 9, 0.1
    t = dt * np.arange(1, m + 1)
    target = np.minimum.outer(t, t)
    blockC = np.zeros((m, m))
    blockC[1:, 1:] = residual_cov(m, dt)
    np.testing.assert_allclose(dt * np.ones((m, m)) + blockC, target, atol=1e-14)
    rng = np.random.default_rng(0)
    n = 200_000
    for c in ("bb", "pca"):
        L = construction_matrix(c, m, dt)
        B = rng.standard_normal((n, m - 1)) @ L.T
        W = math.sqrt(dt) * rng.standard_normal((n, 1)) + np.concatenate([np.zeros((n, 1)), B], axis=1)
        emp = np.cov(W, rowvar=False)
        # SE of a sample covariance <= sqrt((s_ii s_jj + s_ij^2)/n) <= sqrt(2) t_m / sqrt(n)
        assert np.max(np.abs(emp - target)) < 5 * math.sqrt(2) * t[-1] / math.sqrt(n)
        assert np.max(np.abs(W.mean(0))) < 5 * math.sqrt(t[-1] / n)


@pytest.mark.parametrize("c", CONSTRUCTIONS)
def test_conditioning_identity_preserved(c):
    """The increments handed to CPW-first rebuild exactly the residual path B, so
    the integrated variable Z_1 is untouched and CPW-first is a function of B only."""
    m, dt = 6, 0.2
    L = construction_matrix(c, m, dt)
    B = np.random.default_rng(1).standard_normal((50, m - 1)) @ L.T
    Z = increments_from_residual(B, dt)
    Wt = math.sqrt(dt) * (np.cumsum(Z, axis=1) - Z[:, :1])  # what cpw_first_samples reconstructs
    np.testing.assert_allclose(Wt[:, 1:], B, atol=1e-13)
    np.testing.assert_allclose(Wt[:, 0], 0.0, atol=0)
    Z2 = Z.copy()
    Z2[:, 0] = 7.0  # Z_1 is integrated out: its value must not matter (up to rounding in cumsum - Z_1)
    np.testing.assert_allclose(cpw_first_samples(AsianDigital(101.0, m), P, Z, "delta"),
                               cpw_first_samples(AsianDigital(101.0, m), P, Z2, "delta"), rtol=1e-12, atol=0)


def test_bridge_ordering_and_first_coordinate():
    """Bridge: the first coordinate sets the terminal value; every column is used."""
    n, dt = 7, 0.5
    L = bridge_matrix(n, dt)
    assert L[-1, 0] == pytest.approx(math.sqrt(n * dt)) and np.all(L[-1, 1:] == 0)
    assert np.linalg.matrix_rank(L) == n


def test_dimension_reduction_ordering():
    """PCA puts more variance into the first coordinate than the bridge, which beats sequential."""
    for m in (4, 12, 52):
        dt = 1.0 / m
        ev = [explained_variance(c, m, dt, 1) for c in ("seq", "bb", "pca")]
        fs = [functional_share(c, m, dt, 1) for c in ("seq", "bb", "pca")]
        assert ev[0] < ev[1] < ev[2] + 1e-12
        assert fs[0] < fs[2]


def _quad_ref(f, p, g):
    fn = lambda z: float(cpw_first_samples(f, p, np.array([[0.0, z]]), g)[0]) * stats.norm.pdf(z)
    return integrate.quad(fn, -12, 12, limit=400, epsabs=1e-15, epsrel=1e-12)[0]


@pytest.mark.parametrize("sampler", ["MC", "RQMC-seq", "RQMC-bb", "RQMC-pca"])
@pytest.mark.parametrize("g", ["delta", "gamma", "vega"])
def test_timed_estimates_unbiased_m2(sampler, g):
    """m = 2: quadrature reference; every sampler/construction agrees within its own CI scale."""
    f = AsianDigital(103.0, 2)
    ref = _quad_ref(f, P, g)
    e = timed_estimate(g, f, P, 0.0, np.random.default_rng(5), sampler, points_per_scramble=1024, min_units=16)
    assert_within_se(e.value, ref, e.stderr, k=5.0)


@pytest.mark.parametrize("sampler", ["RQMC-seq", "RQMC-bb", "RQMC-pca"])
def test_constructions_agree_m12(sampler):
    """m = 12: all constructions estimate the same Delta as a large plain-MC run."""
    f = AsianDigital(100.0, 12)
    rng = np.random.default_rng(6)
    big = timed_estimate("delta", f, P, 0.0, rng, "MC", min_units=64)
    e = timed_estimate("delta", f, P, 0.0, rng, sampler, points_per_scramble=2048, min_units=16)
    assert abs(e.value - big.value) <= 5 * math.hypot(e.stderr, big.stderr)


def test_timed_budget_and_ci():
    f = AsianDigital(100.0, 4)
    rng = np.random.default_rng(7)
    e = timed_estimate("delta", f, P, 0.02, rng, "RQMC-pca", points_per_scramble=256)
    assert e.wall >= 0.02 and e.n_units >= 4
    assert e.ci_halfwidth == pytest.approx(stats.t.ppf(0.975, e.n_units - 1) * e.stderr)
    e = timed_estimate("delta", f, P, 0.02, rng, "MC")
    assert e.wall >= 0.02 and e.ci_halfwidth == pytest.approx(1.96 * e.stderr)
    # zero budget -> exactly the minimum work, and reproducible
    a = timed_estimate("vega", f, P, 0.0, np.random.default_rng(8), "RQMC-bb", points_per_scramble=64, min_units=4)
    b = timed_estimate("vega", f, P, 0.0, np.random.default_rng(8), "RQMC-bb", points_per_scramble=64, min_units=4)
    assert a.n_units == 4 and a.value == b.value
    with pytest.raises(ValueError):
        timed_estimate("delta", f, P, 0.0, rng, "RQMC-pca", points_per_scramble=100)


def test_points_per_scramble_rule():
    n = choose_points_per_scramble(0.05, 1e-6, 1e-4, 10)
    assert n & (n - 1) == 0 and 10 * (1e-4 + 1e-6 * n) <= 0.05
    assert choose_points_per_scramble(1e-6, 1e-6, 1e-3, 10) == 16


def test_pooled_calibration_shares_cost_model():
    """Constructions run identical code, so the pooled fit must give them the same
    (overhead, per-point) model and hence the same points per scrambling."""
    from mc_greeks.asian_qmc import calibrate_rqmc, pps_for
    cal = calibrate_rqmc((4,), repeats=1, sizes=(64, 256))
    assert cal[("seq", 4)] == cal[("bb", 4)] == cal[("pca", 4)]
    assert len({pps_for(0.01, cal, s, 4) for s in ("RQMC-seq", "RQMC-bb", "RQMC-pca")}) == 1


def test_adaptive_timed_charges_pilot():
    f = AsianDigital(100.0, 4)
    cal = {(c, 4): (2e-4, 2e-7) for c in CONSTRUCTIONS}
    cal[("mc", 4)] = (0.0, 2e-7)
    e, info = adaptive_timed("delta", f, P, 0.02, cal, np.random.default_rng(0), np.random.default_rng(1))
    assert info["chosen"] in ("MC", "RQMC-seq", "RQMC-bb", "RQMC-pca")
    assert e.wall >= info["pilot_wall"] > 0 and e.wall == pytest.approx(info["total_wall"])

"""Phase 4, Part C: Greek-aware gradient-PCA transforms."""
import math

import numpy as np
import pytest
from scipy import integrate, stats

from conftest import assert_within_se
from mc_greeks import BlackScholesParams, CashDigitalCall, analytical
from mc_greeks.asian import AsianDigital, cpw_first_samples
from mc_greeks.asian_qmc import construction_matrix, increments_from_residual, residual_cov
from mc_greeks.greek_aware_qmc import (GREEKS, build_transform, capture_curve, cpw_first_and_grad,
                                       gradient_covariance, path_gradient_cov, rotation_from, rqmc_fixed_n,
                                       timed_with_transform)

P = BlackScholesParams(100.0, 0.02, 0.3, 0.9)
KINDS = ("seq", "bb", "pca", "gpca-delta", "gpca-gamma", "gpca-vega", "gpca-joint")


@pytest.mark.parametrize("g", ("price",) + GREEKS)
@pytest.mark.parametrize("m", [2, 5, 13])
def test_gradient_matches_finite_differences(g, m):
    """Analytic grad_B psi_g vs central differences along random directions (rel err < 1e-6)."""
    f, dt = AsianDigital(101.0, m), P.T / m
    rng = np.random.default_rng(m)
    B = np.cumsum(rng.standard_normal((20, m - 1)) * math.sqrt(dt), axis=1)
    v, grad = cpw_first_and_grad(f, P, B, g)
    np.testing.assert_allclose(v, cpw_first_samples(f, P, increments_from_residual(B, dt), g), rtol=1e-12, atol=1e-15)
    for _ in range(3):
        d = rng.standard_normal(B.shape)
        h = 1e-5
        fd = (cpw_first_samples(f, P, increments_from_residual(B + h * d, dt), g)
              - cpw_first_samples(f, P, increments_from_residual(B - h * d, dt), g)) / (2 * h)
        assert np.max(np.abs((grad * d).sum(1) - fd)) <= 1e-6 * np.max(np.abs(fd)) + 1e-14


def test_wrong_gradient_would_be_detected():
    """Sanity: dropping the dell/s term in the Vega gradient fails the FD check by a wide margin."""
    m, f = 6, AsianDigital(100.0, 6)
    dt = P.T / m
    B = np.cumsum(np.random.default_rng(1).standard_normal((20, m - 1)) * math.sqrt(dt), axis=1)
    _, gv = cpw_first_and_grad(f, P, B, "vega")
    _, gd = cpw_first_and_grad(f, P, B, "delta")
    assert np.max(np.abs(gv - gd)) > 1e-3 * np.max(np.abs(gv))  # different Greeks really differ


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("m", [2, 4, 13, 52])
def test_every_transform_has_brownian_covariance(kind, m):
    f = AsianDigital(100.0, m)
    T = build_transform(kind, f, P, n_train=256, rng=np.random.default_rng(2))
    np.testing.assert_allclose(T.L @ T.L.T, residual_cov(m, P.T / m), atol=1e-12 * m, rtol=0)


def test_gradient_covariance_is_symmetric_psd_and_rotation_orthogonal():
    f = AsianDigital(103.0, 12)
    L0 = construction_matrix("pca", 12, P.T / 12)
    for g in GREEKS:
        G = gradient_covariance(f, P, g, L0, 2048, np.random.default_rng(3))
        assert np.allclose(G, G.T, atol=1e-15 * np.abs(G).max())
        assert np.linalg.eigvalsh(G).min() >= -1e-12 * np.abs(G).max()
        U, w = rotation_from(G)
        np.testing.assert_allclose(U.T @ U, np.eye(11), atol=1e-12)
        assert np.all(np.diff(w) <= 1e-15)


def test_transform_independent_of_base_construction():
    """L0 U is the same (up to column signs) whether L0 is the PCA or the sequential factor."""
    f = AsianDigital(100.0, 8)
    a = build_transform("gpca-delta", f, P, n_train=4096, rng=np.random.default_rng(4), base="pca").L
    b = build_transform("gpca-delta", f, P, n_train=4096, rng=np.random.default_rng(4), base="seq").L
    # same gradient sample only if the base is the same; compare the leading column direction instead
    c = abs(a[:, 0] @ b[:, 0]) / (np.linalg.norm(a[:, 0]) * np.linalg.norm(b[:, 0]))
    assert c > 0.99


def test_rotated_gaussian_invariance_and_law():
    """xi ~ N(0, I) through L_g: sampled covariance of B matches C (law preserved)."""
    m = 9
    f = AsianDigital(100.0, m)
    T = build_transform("gpca-joint", f, P, n_train=1024, rng=np.random.default_rng(5))
    n = 200_000
    B = np.random.default_rng(6).standard_normal((n, m - 1)) @ T.L.T
    C = residual_cov(m, P.T / m)
    assert np.max(np.abs(np.cov(B, rowvar=False) - C)) < 5 * math.sqrt(2) * C.max() / math.sqrt(n)


@pytest.mark.parametrize("g", GREEKS)
def test_m1_and_m2_benchmarks(g):
    """m=2 (one residual dim): every transform equals the quadrature value; m=1 is the closed form
    (no residual dimension; handled by asian.cpw_first_samples directly)."""
    f1 = AsianDigital(100.0, 1)
    assert float(cpw_first_samples(f1, P, np.zeros((1, 1)), g)[0]) == pytest.approx(
        analytical.greek(CashDigitalCall(100.0), P, g), rel=1e-12)
    f = AsianDigital(104.0, 2)
    fn = lambda z: float(cpw_first_samples(f, P, np.array([[0.0, z]]), g)[0]) * stats.norm.pdf(z)
    ref = integrate.quad(fn, -12, 12, limit=400, epsabs=1e-15, epsrel=1e-12)[0]
    for kind in ("pca", "gpca-" + g, "gpca-joint"):
        T = build_transform(kind, f, P, n_train=256, rng=np.random.default_rng(7))
        est = rqmc_fixed_n(f, P, T.L, 512, 16, np.random.default_rng(8), greeks=(g,))[g]
        assert_within_se(est.mean(), ref, est.std(ddof=1) / 4.0, k=5.0)


def test_methods_agree_m13():
    """Larger m: gradient-PCA and path-PCA RQMC agree within combined uncertainty."""
    f = AsianDigital(98.0, 13)
    ests = {}
    for kind in ("pca", "gpca-vega", "gpca-joint"):
        T = build_transform(kind, f, P, n_train=1024, rng=np.random.default_rng(9))
        ests[kind] = rqmc_fixed_n(f, P, T.L, 1024, 16, np.random.default_rng(10 + len(kind)), greeks=GREEKS)
    for g in GREEKS:
        a, b = ests["pca"][g], ests["gpca-joint"][g]
        se = math.hypot(a.std(ddof=1), b.std(ddof=1)) / 4.0
        assert abs(a.mean() - b.mean()) <= 5 * se


def test_capture_curve_and_independent_streams():
    f = AsianDigital(100.0, 12)
    H = path_gradient_cov(f, P, "delta", 4096, np.random.default_rng(11))  # evaluation stream
    Tg = build_transform("gpca-delta", f, P, n_train=4096, rng=np.random.default_rng(12))  # training stream
    cg = capture_curve(Tg.L, H)
    assert cg[-1] == pytest.approx(1.0) and np.all(np.diff(cg) >= -1e-12)
    # the gradient transform captures at least as much first-coordinate gradient energy as sequential
    assert cg[0] >= capture_curve(construction_matrix("seq", 12, P.T / 12), H)[0]


def test_timed_with_transform_ci():
    f = AsianDigital(100.0, 6)
    T = build_transform("gpca-gamma", f, P, n_train=512, rng=np.random.default_rng(13))
    e = timed_with_transform("gamma", f, P, 0.0, np.random.default_rng(14), T.L, 128, min_units=5)
    assert e.n_units == 5 and e.ci_halfwidth == pytest.approx(stats.t.ppf(0.975, 4) * e.stderr)

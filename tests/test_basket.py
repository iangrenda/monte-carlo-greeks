"""Phase 5E: two-asset basket Asian digital, conditional smoothing along the common direction."""
import math

import numpy as np
import pytest

from mc_greeks import analytical, basket as bk
from mc_greeks.asian import AsianDigital, cpw_first_samples
from mc_greeks.models import BlackScholesParams


@pytest.fixture
def setup():
    f = bk.BasketAsianDigital(K=104.0, m=6, w1=0.6, w2=0.4)
    p = bk.TwoAssetParams(S1=100.0, S2=110.0, r=0.02, sigma1=0.2, sigma2=0.35, rho=0.4, T=1.0)
    L = bk.pca_factor(f.m, p.T / f.m, p.rho)
    return f, p, L


@pytest.mark.parametrize("rho", [-0.5, 0.0, 0.4, 0.9])
def test_residual_covariance_and_factor(rho):
    m, dt = 5, 0.2
    C = bk.residual_cov(m, dt, rho)
    assert np.linalg.matrix_rank(C, tol=1e-10) == 2 * m - 1
    null = np.zeros(2 * m)
    null[0] = null[m] = 1.0               # X^1_1 + X^2_1 = 0: the common direction has been removed
    np.testing.assert_allclose(C @ null, 0.0, atol=1e-13)
    L = bk.pca_factor(m, dt, rho)
    assert L.shape == (2 * m, 2 * m - 1)
    np.testing.assert_allclose(L @ L.T, C, atol=1e-12)


def test_residual_covariance_matches_simulation():
    m, dt, rho, n = 4, 0.25, 0.3, 400_000
    rng = np.random.default_rng(0)
    Z1 = rng.standard_normal((n, m))
    Z2 = rho * Z1 + math.sqrt(1 - rho**2) * rng.standard_normal((n, m))
    W = math.sqrt(dt) * np.concatenate([np.cumsum(Z1, 1), np.cumsum(Z2, 1)], axis=1)
    z0 = (Z1[:, 0] + Z2[:, 0]) / math.sqrt(2 * (1 + rho))
    X = W - bk.common_loading(rho) * math.sqrt(dt) * z0[:, None]
    np.testing.assert_allclose(np.cov(X, rowvar=False), bk.residual_cov(m, dt, rho), atol=6e-3)
    assert abs(np.corrcoef(X.T, z0)[-1, :-1]).max() < 6e-3      # residual independent of Z0


def test_root_solves_the_barrier_equation(setup):
    f, p, L = setup
    X = np.random.default_rng(1).standard_normal((2000, L.shape[1])) @ L.T
    z = bk.solve_root(f, p, X)
    st, _, _, _ = bk._state(f, p, X)
    A = sum(w * R.mean(axis=1) * np.exp(s * z) for (_, R, w, _, s, _) in st)
    np.testing.assert_allclose(A, f.K, rtol=1e-11)


def test_greeks_are_derivatives_of_the_per_path_smoothed_price(setup):
    f, p, L = setup
    X = np.random.default_rng(2).standard_normal((300, L.shape[1]))
    V = bk.smoothed(f, p, X @ L.T, with_price=True)

    def vals(**kw):
        q = bk.TwoAssetParams(**{**p.__dict__, **kw})
        # sigma enters the residual path only through the exponent; X (unit-volatility Brownian residual) is held fixed
        return bk.smoothed(f, q, X @ L.T, with_price=True)

    for j, (S, sg) in enumerate((("S1", "sigma1"), ("S2", "sigma2"))):
        x0, h = getattr(p, S), 1e-4 * getattr(p, S)
        up, dn = vals(**{S: x0 + h}), vals(**{S: x0 - h})
        np.testing.assert_allclose(V[:, 1 + j], (up[:, 0] - dn[:, 0]) / (2 * h), rtol=5e-6, atol=1e-10)
        # Gamma as the derivative of the (just validated) per-path Delta
        np.testing.assert_allclose(V[:, 3 + j], (up[:, 1 + j] - dn[:, 1 + j]) / (2 * h), rtol=5e-6, atol=1e-10)
        s0, hs = getattr(p, sg), 1e-5
        np.testing.assert_allclose(V[:, 5 + j], (vals(**{sg: s0 + hs})[:, 0] - vals(**{sg: s0 - hs})[:, 0]) / (2 * hs), rtol=5e-6, atol=1e-9)


def test_smoothed_price_agrees_with_the_raw_digital_payoff(setup):
    f, p, L = setup
    rng = np.random.default_rng(3)
    X = rng.standard_normal((400_000, L.shape[1])) @ L.T
    sm = bk.smoothed(f, p, X, with_price=True)[:, 0]
    raw = bk.raw_payoff(f, p, X, rng.standard_normal(len(X)))
    z = (sm.mean() - raw.mean()) / math.hypot(sm.std() / math.sqrt(len(X)), raw.std() / math.sqrt(len(X)))
    assert abs(z) < 4.5
    assert sm.var() < raw.var()                                  # Rao-Blackwell


def test_one_period_single_asset_limit_is_the_black_scholes_digital():
    """m = 1, w2 = 0: A = w1 S1_T, a plain digital with strike K / w1. The residual is one-dimensional,
    so the expectation of the smoothed estimator is an exact Gauss-Hermite integral."""
    f = bk.BasketAsianDigital(K=63.0, m=1, w1=0.6, w2=0.0)
    p = bk.TwoAssetParams(100.0, 80.0, 0.03, 0.25, 0.4, 0.3, 0.75)
    L = bk.pca_factor(1, p.T, p.rho)
    x, w = np.polynomial.hermite_e.hermegauss(120)
    w = w / w.sum()
    V = bk.smoothed(f, p, x[:, None] @ L.T, with_price=True)
    bs = BlackScholesParams(p.S1, p.r, p.sigma1, p.T)
    K = f.K / f.w1
    exact = dict(price=analytical.digital_price(bs, K), delta1=analytical.digital_delta(bs, K),
                 gamma11=analytical.digital_gamma(bs, K), vega1=analytical.digital_vega(bs, K))
    got = dict(zip(("price",) + bk.NAMES, w @ V))
    for k, v in exact.items():
        assert got[k] == pytest.approx(v, rel=1e-8), k
    for k in ("delta2", "gamma22", "vega2"):
        assert abs(got[k]) < 1e-14


def test_single_asset_limit_matches_the_one_asset_estimator_in_expectation():
    f = bk.BasketAsianDigital(K=101.0, m=6, w1=1.0, w2=0.0)
    p = bk.TwoAssetParams(100.0, 90.0, 0.02, 0.3, 0.2, 0.5, 1.0)
    L = bk.pca_factor(f.m, p.T / f.m, p.rho)
    rng = np.random.default_rng(5)
    n = 300_000
    V = bk.smoothed(f, p, rng.standard_normal((n, L.shape[1])) @ L.T)
    f1, p1 = AsianDigital(101.0, 6), BlackScholesParams(100.0, 0.02, 0.3, 1.0)
    Z = rng.standard_normal((n, 6))
    for name, g in (("delta1", "delta"), ("gamma11", "gamma"), ("vega1", "vega")):
        a, b = V[:, bk.NAMES.index(name)], cpw_first_samples(f1, p1, Z, g)
        z = (a.mean() - b.mean()) / math.hypot(a.std() / math.sqrt(n), b.std() / math.sqrt(n))
        assert abs(z) < 4.5, (name, z)


def test_swapping_the_assets_swaps_the_greeks(setup):
    f, p, L = setup
    xi = np.random.default_rng(6).standard_normal((50, L.shape[1]))
    X = xi @ L.T
    V = bk.smoothed(f, p, X)
    f2 = bk.BasketAsianDigital(f.K, f.m, f.w2, f.w1)
    p2 = bk.TwoAssetParams(p.S2, p.S1, p.r, p.sigma2, p.sigma1, p.rho, p.T)
    V2 = bk.smoothed(f2, p2, np.concatenate([X[:, f.m:], X[:, :f.m]], axis=1))
    np.testing.assert_allclose(V2[:, [1, 0, 3, 2, 5, 4]], V, rtol=1e-10, atol=1e-14)


def test_fd_gradients_shape_and_accuracy(setup):
    f, p, L = setup
    xi = np.random.default_rng(7).standard_normal((20, L.shape[1]))
    g = bk.fd_gradients(f, p, L, xi)
    assert g.shape == (20, len(bk.NAMES), L.shape[1])
    g2 = bk.fd_gradients(f, p, L, xi, h=1e-3)
    np.testing.assert_allclose(g, g2, rtol=5e-3, atol=1e-6 * np.abs(g).max())

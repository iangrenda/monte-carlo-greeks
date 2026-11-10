"""Phase 5C/5D: shared-path joint Greeks, dt-independent PCA basis, rotations, loss."""
import numpy as np
import pytest

from mc_greeks import joint_greeks as jg
from mc_greeks.asian import AsianDigital, cpw_first_samples
from mc_greeks.asian_qmc import construction_matrix, residual_cov
from mc_greeks.models import BlackScholesParams


@pytest.fixture
def setup():
    return AsianDigital(103.0, 12), BlackScholesParams(100.0, 0.02, 0.3, 0.8)


def test_joint_matches_single_greek_formulas(setup):
    f, p = setup
    Z = np.random.default_rng(1).standard_normal((500, f.m))
    J = jg.cpw_first_joint(f, p, Z)
    assert J.shape == (500, 3)
    for k, g in enumerate(jg.GREEKS):
        np.testing.assert_allclose(J[:, k], cpw_first_samples(f, p, Z, g), rtol=1e-12, atol=0)
    J4 = jg.cpw_first_joint(f, p, Z, with_price=True)
    np.testing.assert_allclose(J4[:, 0], cpw_first_samples(f, p, Z, "price"), rtol=1e-12)
    np.testing.assert_allclose(J4[:, 1:], J, rtol=0, atol=0)


def test_joint_does_not_depend_on_first_increment(setup):
    f, p = setup
    Z = np.random.default_rng(2).standard_normal((50, f.m))
    Z2 = Z.copy()
    Z2[:, 0] += 3.0
    np.testing.assert_allclose(jg.cpw_first_joint(f, p, Z), jg.cpw_first_joint(f, p, Z2), rtol=1e-12)


@pytest.mark.parametrize("m,dt", [(2, 0.5), (5, 0.1), (24, 1.0 / 24), (52, 0.03)])
def test_pca_factor_reproduces_residual_covariance(m, dt):
    L = jg.pca_factor(m, dt)
    np.testing.assert_allclose(L @ L.T, residual_cov(m, dt), rtol=1e-10, atol=1e-13)
    L0 = construction_matrix("pca", m, dt)
    np.testing.assert_allclose(np.abs(L), np.abs(L0), rtol=1e-7, atol=1e-10)  # same up to column signs
    np.testing.assert_allclose(jg.pca_factor(m, 4 * dt), 2 * L, rtol=1e-12)  # basis independent of dt


def test_rotated_factor_keeps_covariance_for_any_orthogonal_u():
    m, dt = 9, 0.07
    Q, _ = np.linalg.qr(np.random.default_rng(3).standard_normal((m - 1, m - 1)))
    L = jg.rotated_factor(Q, m, dt)
    np.testing.assert_allclose(L @ L.T, residual_cov(m, dt), rtol=1e-10, atol=1e-13)
    np.testing.assert_allclose(jg.rotated_factor(None, m, dt), jg.pca_factor(m, dt))


def test_joint_rotation_is_orthogonal_and_respects_weights(setup):
    f, p = setup
    Gs = jg.gradient_covariances(f, p, 2000, np.random.default_rng(4))
    for G in Gs.values():
        np.testing.assert_allclose(G, G.T, rtol=1e-12)
        assert np.linalg.eigvalsh(G).min() > -1e-10 * np.trace(G)
    U, ev = jg.joint_rotation(Gs)
    np.testing.assert_allclose(U.T @ U, np.eye(f.m - 1), atol=1e-10)
    assert np.all(np.diff(ev) <= 1e-12) and abs(ev.sum() - 1.0) < 1e-10  # trace-normalised, equal weights
    Ud, _ = jg.joint_rotation(Gs, {"delta": 1.0, "gamma": 0.0, "vega": 0.0})
    w, V = np.linalg.eigh(Gs["delta"])
    assert abs(Ud[:, 0] @ V[:, -1]) > 1 - 1e-8


def test_trained_rotation_transfers_to_another_contract_without_bias(setup):
    f, p = setup
    U, sec, Gs = jg.train_rotation(f, p, 256, np.random.default_rng(5))
    assert sec > 0 and set(Gs) == set(jg.GREEKS)
    p2 = BlackScholesParams(100.0, 0.02, 0.5, 2.0)  # different sigma and maturity, same m
    L = jg.rotated_factor(U, f.m, p2.T / f.m)
    np.testing.assert_allclose(L @ L.T, residual_cov(f.m, p2.T / f.m), rtol=1e-10, atol=1e-13)


def test_alignment_rank_and_loss():
    Q, _ = np.linalg.qr(np.random.default_rng(6).standard_normal((6, 6)))
    assert jg.top_alignment(Q, Q, 1) == pytest.approx(1.0)
    assert jg.top_alignment(Q, Q, 3) == pytest.approx(1.0)
    assert jg.top_alignment(Q, Q[:, ::-1], 1) == pytest.approx(0.0, abs=1e-12)
    v = np.arange(1.0, 5.0)
    r = jg.effective_rank(np.outer(v, v))
    assert r["top_share"] == pytest.approx(1.0) and r["participation_ratio"] == pytest.approx(1.0)
    assert jg.effective_rank(np.eye(4))["participation_ratio"] == pytest.approx(4.0)
    assert jg.normalised_loss([4.0, 9.0, 1.0], [2.0, 3.0, 1.0]) == pytest.approx(1.0)
    assert jg.normalised_loss([4.0, 9.0, 1.0], [2.0, 3.0, 1.0], (1.0, 0.0, 0.0)) == pytest.approx(1.0)


def test_risk_scales_match_per_path_standard_deviation(setup):
    f, p = setup
    s = jg.risk_scales(f, p, np.random.default_rng(7), 4096)
    Z = np.random.default_rng(7).standard_normal((4096, f.m))
    for k, g in enumerate(jg.GREEKS):
        assert s[k] == pytest.approx(cpw_first_samples(f, p, Z, g).std(ddof=1), rel=1e-10)

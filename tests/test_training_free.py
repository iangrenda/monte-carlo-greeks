"""Phase 6B: deterministic rotations (Householder completion, design point, alignment)."""
import math

import numpy as np
import pytest

from mc_greeks import training_free as tf
from mc_greeks.asian import AsianDigital
from mc_greeks.asian_qmc import residual_cov
from mc_greeks.joint_greeks import gradient_covariances, joint_rotation, pca_factor, rotated_factor
from mc_greeks.models import BlackScholesParams


@pytest.fixture
def setup():
    return AsianDigital(107.0, 16), BlackScholesParams(100.0, 0.035, 0.35, 1.75)


@pytest.mark.parametrize("seed", range(5))
def test_householder_is_orthogonal_and_maps_e1(seed):
    u = np.random.default_rng(seed).standard_normal(7)
    H = tf.householder_to(u)
    np.testing.assert_allclose(H @ H.T, np.eye(7), atol=1e-12)
    np.testing.assert_allclose(H, H.T, atol=1e-14)
    v = u / np.linalg.norm(u) * np.sign(u[0])
    np.testing.assert_allclose(H[:, 0], v, atol=1e-12)
    np.testing.assert_allclose(tf.householder_to(np.eye(5)[0]), np.eye(5))


@pytest.mark.parametrize("kind", tf.ALL_KINDS)
def test_rotations_keep_the_path_law(setup, kind):
    f, p = setup
    U, sec = tf.training_free_rotation(kind, f, p)
    assert sec >= 0
    L = rotated_factor(U, f.m, p.T / f.m)
    np.testing.assert_allclose(L @ L.T, residual_cov(f.m, p.T / f.m), rtol=1e-10, atol=1e-13)


def test_origin_direction_is_the_gradient_at_zero(setup):
    f, p = setup
    dt = p.T / f.m
    t = dt * np.arange(2, f.m + 1)
    v = tf.direction_B("origin", f, p)
    np.testing.assert_allclose(v / v[0], np.exp((p.r - 0.5 * p.sigma**2) * (t - t[0])), rtol=1e-12)


def test_dominant_point_is_on_the_boundary_and_optimal(setup):
    f, p = setup
    B = tf.dominant_point(f, p)
    R, Abar = tf._residual_state(f, p, B)
    assert abs(math.log(Abar / f.K)) < 1e-10
    # KKT: C^-1 B is parallel to grad ln Abar
    a = tf.grad_log_abar(f, p, B)
    y = np.linalg.solve(residual_cov(f.m, p.T / f.m), B)
    assert abs(abs(y @ a) / (np.linalg.norm(y) * np.linalg.norm(a)) - 1) < 1e-9
    # no boundary point nearby has a smaller Mahalanobis norm
    C = residual_cov(f.m, p.T / f.m)
    q = lambda b: b @ np.linalg.solve(C, b)
    rng = np.random.default_rng(0)
    for _ in range(50):
        d = rng.standard_normal(f.m - 1) * 0.05
        lo, hi = -5.0, 5.0      # move along B* + d + s * (C a) back onto the boundary
        Ca = C @ a
        g = lambda s: math.log(tf._residual_state(f, p, B + d + s * Ca)[1] / f.K)
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if g(mid) < 0 else (lo, mid)
        assert q(B + d + 0.5 * (lo + hi) * Ca) >= q(B) - 1e-9


def test_deterministic_direction_is_close_to_the_trained_one(setup):
    """The premise of 6B: grad delta varies little, so u_det is nearly the top eigenvector of G."""
    f, p = setup
    Gs = gradient_covariances(f, p, 20_000, np.random.default_rng(3))
    U, _ = joint_rotation(Gs)
    for kind in tf.KINDS:
        assert abs(tf.first_direction(kind, f, p) @ U[:, 0]) > 0.98, kind


def test_hessian_matches_finite_differences(setup):
    f, p = setup
    B = np.random.default_rng(4).standard_normal(f.m - 1) * 0.1
    H = tf.hessian_log_abar(f, p, B)
    h = 1e-6
    for i in range(0, f.m - 1, 4):
        e = np.zeros(f.m - 1); e[i] = h
        col = (tf.grad_log_abar(f, p, B + e) - tf.grad_log_abar(f, p, B - e)) / (2 * h)
        np.testing.assert_allclose(H[:, i], col, rtol=1e-6, atol=1e-10)


def test_taylor_covariance_matches_the_linearised_monte_carlo_average(setup):
    f, p = setup
    L = pca_factor(f.m, p.T / f.m)
    a = L.T @ tf.grad_log_abar(f, p, np.zeros(f.m - 1))
    K = L.T @ tf.hessian_log_abar(f, p, np.zeros(f.m - 1)) @ L
    xi = np.random.default_rng(5).standard_normal((200_000, f.m - 1))
    g = a + xi @ K.T
    np.testing.assert_allclose(g.T @ g / len(xi), tf.taylor_gradient_covariance(f, p, "origin"),
                               rtol=0.03, atol=3e-3 * np.abs(a).max() ** 2)


def test_pca_coordinates_invert_the_pca_factor(setup):
    f, p = setup
    B = np.random.default_rng(6).standard_normal(f.m - 1)
    np.testing.assert_allclose(pca_factor(f.m, p.T / f.m) @ tf.pca_coordinates(f, p, B), B, rtol=1e-10, atol=1e-12)


def test_weighted_covariance_matches_monte_carlo_under_its_own_approximations(setup):
    """Check the quadrature algebra: draw xi, apply the stated linearisations, average."""
    f, p = setup
    L = pca_factor(f.m, p.T / f.m)
    s = p.sigma * np.sqrt(p.T / f.m)
    B0 = tf.dominant_point(f, p)
    xi0 = tf.pca_coordinates(f, p, B0)
    a = L.T @ tf.grad_log_abar(f, p, B0)
    K = L.T @ tf.hessian_log_abar(f, p, B0) @ L
    ah = a / np.linalg.norm(a)
    xi = np.random.default_rng(7).standard_normal((400_000, f.m - 1))
    t = xi @ ah
    dl = 0.0 + np.linalg.norm(a) / s * (t - ah @ xi0)
    grad = (a - K @ xi0) + xi @ K.T
    G = np.zeros((f.m - 1, f.m - 1))
    for h in tf._greek_weight_derivatives(f, p, B0).values():
        g = h(dl)[:, None] * grad
        Gg = g.T @ g / len(xi)
        G += Gg / np.trace(Gg)
    G /= 3
    Gq = tf.weighted_gradient_covariance(f, p, "dominant")
    lead = lambda M: np.linalg.eigh(M)[1][:, -1]
    assert abs(lead(G) @ lead(Gq)) > 0.9999
    np.testing.assert_allclose(np.diag(Gq)[:5], np.diag(G)[:5], rtol=0.05, atol=1e-4)

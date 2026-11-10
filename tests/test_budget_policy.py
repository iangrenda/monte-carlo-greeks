"""Phase 5B: digital shifts, executors, cost model, schedule optimisation, feasibility."""
import numpy as np
import pytest

from mc_greeks import budget_policy as bp
from mc_greeks.asian import AsianDigital, cpw_first_samples
from mc_greeks.joint_greeks import pca_factor
from mc_greeks.models import BlackScholesParams

CM = bp.CostModel(m=24, what="delta", rq={"owen": (1e-4, 1e-3, 1e-6), "shift": (1e-4, 2e-5, 1e-6)},
                  mc=(5e-5, 2e-5, 4e-7), fit_error={})


@pytest.fixture
def setup():
    f, p = AsianDigital(101.0, 8), BlackScholesParams(100.0, 0.02, 0.25, 1.0)
    return f, p, pca_factor(f.m, p.T / f.m), p.T / f.m


# --- digital shift ---------------------------------------------------------------------------

def test_shifted_points_are_in_unit_cube_and_keep_the_net_property():
    d, n = 7, 64
    ints = bp.sobol_int_points(d, n, cache={})
    rng = np.random.default_rng(0)
    for _ in range(20):
        u = bp.shifted_uniforms(ints, rng)
        assert u.shape == (n, d) and u.min() >= 0.0 and u.max() < 1.0
        # every one-dimensional projection has exactly one point in each interval [i/n, (i+1)/n)
        cells = np.sort(np.floor(u * n).astype(int), axis=0)
        np.testing.assert_array_equal(cells, np.tile(np.arange(n)[:, None], (1, d)))


def test_shifted_points_are_unbiased():
    d, n = 5, 32
    ints = bp.sobol_int_points(d, n, cache={})
    rng = np.random.default_rng(1)
    est = np.array([np.prod(1.0 + (bp.shifted_uniforms(ints, rng) - 0.5), axis=1).mean() for _ in range(4000)])
    assert abs(est.mean() - 1.0) < 5 * est.std(ddof=1) / np.sqrt(len(est))  # integral of prod(0.5 + u) is 1
    first = np.array([bp.shifted_uniforms(ints, rng)[0, 0] for _ in range(4000)])  # the image of the origin
    assert abs(first.mean() - 0.5) < 5 * np.sqrt(1 / 12 / 4000)


# --- executors -------------------------------------------------------------------------------

@pytest.mark.parametrize("rand", bp.RANDOMISATIONS)
def test_joint_and_single_greek_runs_agree_on_the_same_points(setup, rand):
    f, p, L, dt = setup
    ej = bp.run_rqmc(bp.make_evaluator(f, p, "joint"), L, dt, 64, 0.0, np.random.default_rng(3), rand, min_units=5, max_units=5,
                     cache={})
    assert ej.n_units == 5 and ej.n_paths == 320 and ej.value.shape == (3,) and ej.cov.shape == (3, 3)
    for k, g in enumerate(("delta", "gamma", "vega")):
        e1 = bp.run_rqmc(bp.make_evaluator(f, p, g), L, dt, 64, 0.0, np.random.default_rng(3), rand, min_units=5, max_units=5,
                         cache={})
        np.testing.assert_allclose(e1.value[0], ej.value[k], rtol=1e-11)
        np.testing.assert_allclose(e1.stderr[0], ej.stderr[k], rtol=1e-9)
    np.testing.assert_allclose(np.sqrt(np.diag(ej.cov)), ej.stderr)
    np.testing.assert_allclose(ej.ci_halfwidth, bp.t975(4) * ej.stderr)


@pytest.mark.parametrize("rand", bp.RANDOMISATIONS)
def test_rqmc_estimates_are_consistent_with_a_plain_mc_reference(setup, rand):
    f, p, L, dt = setup
    Z = np.random.default_rng(10).standard_normal((400_000, f.m))
    e = bp.run_rqmc(bp.make_evaluator(f, p, "joint"), L, dt, 1024, 0.0, np.random.default_rng(11), rand, min_units=16, max_units=16)
    for k, g in enumerate(("delta", "gamma", "vega")):
        y = cpw_first_samples(f, p, Z, g)
        z = (e.value[k] - y.mean()) / np.hypot(e.stderr[k], y.std(ddof=1) / np.sqrt(len(y)))
        assert abs(z) < 5, (rand, g, z)


def test_run_mc_matches_direct_sample_statistics(setup):
    f, p, _, _ = setup
    e = bp.run_mc(bp.make_evaluator(f, p, "joint"), f.m, 256, 0.0, np.random.default_rng(5), min_units=3, max_units=3)
    rng = np.random.default_rng(5)
    Z = np.concatenate([rng.standard_normal((256, f.m)) for _ in range(3)])
    from mc_greeks.joint_greeks import cpw_first_joint
    Y = cpw_first_joint(f, p, Z)
    np.testing.assert_allclose(e.value, Y.mean(axis=0), rtol=1e-12)
    np.testing.assert_allclose(e.cov, np.cov(Y, rowvar=False) / len(Y), rtol=1e-8)
    assert e.n_paths == 768 and e.n_units == 3 and e.method == "MC"


def test_lookahead_stops_before_starting_a_unit_that_cannot_fit(setup):
    f, p, L, dt = setup
    ev = bp.make_evaluator(f, p, "delta")
    # the predicted unit cost exceeds the budget: only the two units needed for a standard error run
    assert bp.run_rqmc(ev, L, dt, 16, 1.0, np.random.default_rng(0), lookahead_cost=2.0).n_units == 2
    assert bp.run_mc(ev, f.m, 16, 1.0, np.random.default_rng(0), lookahead_cost=2.0).n_units == 2
    # Phase-3/4 behaviour (no look-ahead) with a zero budget: exactly min_units
    assert bp.run_rqmc(ev, L, dt, 16, 0.0, np.random.default_rng(0), min_units=4).n_units == 4
    with pytest.raises(ValueError):
        bp.run_rqmc(ev, L, dt, 48, 0.0, np.random.default_rng(0))


# --- planning --------------------------------------------------------------------------------

@pytest.mark.parametrize("objective", ["mse", "ci"])
@pytest.mark.parametrize("rand", bp.RANDOMISATIONS)
@pytest.mark.parametrize("budget", [0.004, 0.011, 0.02, 0.05, 0.3])
def test_optimised_schedule_is_feasible_and_optimal(objective, rand, budget):
    alpha, r_min, margin = 2.0, 8, 0.05
    pl = bp.optimise_schedule(budget, CM, rand, alpha, objective, r_min, margin=margin)
    floor_r = r_min if objective == "mse" else 3
    c0, a, b = CM.rq[rand]

    def obj(R, N):
        return N ** (-alpha) / R * (bp.t975(R - 1) ** 2 if objective == "ci" else 1.0)

    brute = [(obj(R, N), R, N) for N in (16 << k for k in range(15)) for R in range(floor_r, 400)
             if c0 + R * (a + b * N) <= (1 - margin) * budget]
    if not brute:
        assert not pl.feasible and pl.method == "infeasible"
        return
    assert pl.feasible and pl.units >= floor_r
    assert pl.predicted_cost <= (1 - margin) * budget + 1e-15
    assert pl.predicted_cost == pytest.approx(CM.rqmc_cost(pl.units, pl.n_points, rand))
    assert obj(pl.units, pl.n_points) == pytest.approx(min(brute)[0])


def test_infeasible_requests_are_reported_not_scheduled(setup):
    f, p, L, dt = setup
    pl = bp.optimise_schedule(0.002, CM, "owen")  # 8 scramblings of 16 points need > 8 ms
    assert not pl.feasible
    e = bp.execute(pl, bp.make_evaluator(f, p, "delta"), 1, L, f.m, dt, 0.002, np.random.default_rng(0))
    assert not e.feasible and e.n_units == 0 and np.isnan(e.value).all() and e.method == "infeasible"
    assert bp.optimise_schedule(0.002, CM, "shift").feasible  # the shift has a much smaller per-replicate cost
    assert not bp.fixed_n_schedule(0.002, CM, 64).feasible and bp.fixed_n_schedule(0.05, CM, 64).units >= 8


def test_mc_schedule_and_fallback_chain():
    pl = bp.mc_schedule(0.01, CM)
    assert pl.feasible and pl.method == "MC" and pl.units >= 2 and pl.predicted_cost <= 0.0095 + 1e-15
    assert pl.n_points & (pl.n_points - 1) == 0
    assert not bp.mc_schedule(1e-5, CM).feasible
    assert bp.feasible_plan(0.2, CM).method == "RQMC"
    assert bp.feasible_plan(0.004, CM).method == "MC"       # below the frozen-form threshold 10 (a + 64 b)
    assert bp.feasible_plan(1e-5, CM).method == "infeasible"
    assert bp.frozen_rule_prefers_rqmc(0.0107, CM) and not bp.frozen_rule_prefers_rqmc(0.0106, CM)


def test_calibration_returns_a_usable_model():
    cm = bp.calibrate_cost_model(4, "delta", seed=0, repeats=1, sizes=(16, 64), units=(2, 4))
    for rand in bp.RANDOMISATIONS:
        assert all(c >= 0 for c in cm.rq[rand]) and cm.rqmc_cost(8, 64, rand) > 0
    assert all(c >= 0 for c in cm.mc) and cm.mc_cost(4, 256) > 0


# --- Phase 6: cached LMS+shift --------------------------------------------------------------------

def test_direction_numbers_regenerate_the_scipy_point_set():
    d, k = 9, 8
    V = bp.sobol_direction_ints(d, k, cache={})
    X = np.zeros((1 << k, d), dtype=np.uint32)
    for b in range(k):
        X[1 << b:2 << b] = X[:1 << b] ^ V[:, b]
    ref = bp.sobol_int_points(d, 1 << k, cache={})
    # same point set (SciPy uses Gray-code order)
    np.testing.assert_array_equal(np.sort(X.view([("", np.uint32)] * d).ravel()),
                                  np.sort(np.ascontiguousarray(ref).view([("", np.uint32)] * d).ravel()))


def test_lms_identity_and_linearity():
    rng = np.random.default_rng(0)
    d = 4
    eye = (np.uint32(1) << (bp.BITS - 1 - np.arange(bp.BITS, dtype=np.uint32)))[None, :].repeat(d, 0)
    x = rng.integers(0, 1 << bp.BITS, size=(d, 50), dtype=np.uint32)
    np.testing.assert_array_equal(bp.lms_apply(eye, x), x)
    M = bp.lms_masks(d, rng)
    y = rng.integers(0, 1 << bp.BITS, size=(d, 50), dtype=np.uint32)
    np.testing.assert_array_equal(bp.lms_apply(M, x ^ y), bp.lms_apply(M, x) ^ bp.lms_apply(M, y))
    # lower triangular with unit diagonal: the leading digit is preserved
    top = np.uint32(1 << (bp.BITS - 1))
    np.testing.assert_array_equal(bp.lms_apply(M, x) & top, x & top)


def test_lms_points_are_uniform_nets_and_unbiased():
    d, n = 6, 64
    V = bp.sobol_direction_ints(d, 6, cache={})
    rng = np.random.default_rng(1)
    for _ in range(20):
        u = bp.lms_uniforms(V, n, rng)
        assert u.shape == (n, d) and u.min() >= 0.0 and u.max() < 1.0
        cells = np.sort(np.floor(u * n).astype(int), axis=0)
        np.testing.assert_array_equal(cells, np.tile(np.arange(n)[:, None], (1, d)))
    est = np.array([np.prod(0.5 + bp.lms_uniforms(V, 32, rng), axis=1).mean() for _ in range(4000)])
    assert abs(est.mean() - 1.0) < 5 * est.std(ddof=1) / np.sqrt(len(est))


def test_own_lms_has_the_same_variance_as_scipy_lms():
    """Same randomisation, two implementations: replicate variances must agree (F-type check)."""
    from scipy.stats import qmc
    d, n, R = 5, 256, 600
    f = lambda u: np.exp(np.sum(u * np.array([1.0, 0.7, 0.5, 0.3, 0.2]), axis=1))
    V = bp.sobol_direction_ints(d, 8, cache={})
    rng = np.random.default_rng(2)
    own = np.array([f(bp.lms_uniforms(V, n, rng)).mean() for _ in range(R)])
    sci = np.array([f(qmc.Sobol(d, scramble=True, seed=rng).random(n)).mean() for _ in range(R)])
    sh = np.array([f(bp.shifted_uniforms(bp.sobol_int_points(d, n, cache={}), rng)).mean() for _ in range(R)])
    ratio = own.var(ddof=1) / sci.var(ddof=1)
    assert 0.8 < ratio < 1.25, ratio            # ~ +-9% sampling sd for R = 600 each
    assert abs(own.mean() - sci.mean()) < 5 * np.sqrt(own.var() / R + sci.var() / R)
    assert sh.var(ddof=1) > 1.5 * own.var(ddof=1)  # digital shift alone is worse for a smooth integrand


def test_run_rqmc_with_lms_agrees_with_reference(setup):
    f, p, L, dt = setup
    e = bp.run_rqmc(bp.make_evaluator(f, p, "joint"), L, dt, 1024, 0.0, np.random.default_rng(11), "lms",
                    min_units=16, max_units=16)
    Z = np.random.default_rng(10).standard_normal((400_000, f.m))
    for k, g in enumerate(("delta", "gamma", "vega")):
        y = cpw_first_samples(f, p, Z, g)
        z = (e.value[k] - y.mean()) / np.hypot(e.stderr[k], y.std(ddof=1) / np.sqrt(len(y)))
        assert abs(z) < 5, (g, z)

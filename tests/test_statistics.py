import numpy as np
import pytest

from mc_greeks.statistics import loglog_slope, paired_mse_comparison, replication_summary, theoretical_rmse


def test_summary_deterministic():
    est = np.array([1.0, 2.0, 3.0, 4.0])
    s = replication_summary(est, stderrs=np.full(4, 0.5), truth=2.0)
    assert s["mean"] == 2.5 and s["bias"] == 0.5
    assert s["variance"] == pytest.approx(np.var(est, ddof=1))
    assert s["mse"] == pytest.approx(np.mean((est - 2.0) ** 2))
    assert s["rmse"] == pytest.approx(np.sqrt(1.5))
    # MSE = bias^2 + (R-1)/R * variance (ddof correction)
    assert s["mse"] == pytest.approx(s["bias"] ** 2 + 3 / 4 * s["variance"])
    # CI = est +- 0.98; covers 2.0 for est in {1? no, 2 yes, 3? no, 4 no}
    assert s["coverage"] == 0.25


def test_coverage_nominal_for_unbiased_gaussian():
    rng = np.random.default_rng(0)
    est = rng.normal(0.0, 1.0, 20000)
    s = replication_summary(est, np.ones_like(est), truth=0.0)
    assert s["coverage"] == pytest.approx(0.95, abs=0.01)
    # a biased estimator with the same SE under-covers
    s2 = replication_summary(est + 2.0, np.ones_like(est), truth=0.0)
    assert s2["coverage"] < 0.6


def test_loglog_slope_recovers_rate():
    n = np.array([1e3, 1e4, 1e5, 1e6])
    fit = loglog_slope(n, 3.0 * n**-0.5)
    assert fit["slope"] == pytest.approx(-0.5, abs=1e-12)
    assert fit["r2"] == pytest.approx(1.0)


def test_paired_comparison_and_theory():
    rng = np.random.default_rng(1)
    common = rng.normal(size=5000)
    a = common + 0.1 * rng.normal(size=5000)
    b = 2 * common
    out = paired_mse_comparison(a, b, truth=0.0)
    assert out["mse_ratio"] < 1 and out["diff_hi"] < 0
    assert theoretical_rmse(0.3, 0.16 * 100, 100) == pytest.approx(0.5)


def test_summary_needs_two_reps():
    with pytest.raises(ValueError):
        replication_summary([1.0], [1.0], 1.0)

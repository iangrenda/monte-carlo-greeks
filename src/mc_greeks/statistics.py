"""Replication statistics for Monte Carlo estimators.

Given R independent replications theta_hat_1..R of an estimator with known
true value theta, we estimate

    bias     = mean(theta_hat) - theta                 (SE: sd / sqrt(R))
    variance = sample variance of theta_hat (ddof=1)
    MSE      = mean((theta_hat - theta)^2)             (SE: sd of squared errors / sqrt(R))
    RMSE     = sqrt(MSE)                               (SE via delta method: SE(MSE) / (2 RMSE))

and the MSE = bias^2 + variance identity (holds exactly up to the ddof
correction). ``coverage`` is the fraction of replications whose nominal 95%
confidence interval theta_hat +- 1.96 * SE_hat contains theta. That interval
accounts for Monte Carlo noise only, so a biased estimator (e.g. finite
differences with a large bump) under-covers even when SE_hat is accurate.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

Z95 = stats.norm.ppf(0.975)


def replication_summary(estimates, stderrs, truth: float, runtimes=None, payoff_evals=None) -> dict[str, float]:
    est = np.asarray(estimates, dtype=float)
    se = np.asarray(stderrs, dtype=float)
    R = est.size
    if R < 2:
        raise ValueError("need at least two replications")
    err = est - truth
    sd = est.std(ddof=1)
    sq = err**2
    mse = sq.mean()
    rmse = math.sqrt(mse)
    mse_se = sq.std(ddof=1) / math.sqrt(R)
    covered = np.abs(err) <= Z95 * se
    out = {
        "truth": truth,
        "mean": est.mean(),
        "bias": err.mean(),
        "bias_se": sd / math.sqrt(R),
        "variance": sd**2,
        "sd": sd,
        "mean_stderr": se.mean(),
        "mse": mse,
        "mse_se": mse_se,
        "rmse": rmse,
        "rmse_se": mse_se / (2 * rmse) if rmse > 0 else float("nan"),
        "rel_rmse": rmse / abs(truth) if truth != 0 else float("nan"),
        "coverage": covered.mean(),
        "coverage_se": math.sqrt(max(covered.mean() * (1 - covered.mean()), 1e-12) / R),
        "replications": R,
    }
    if runtimes is not None:
        out["runtime"] = float(np.mean(runtimes))
    if payoff_evals is not None:
        out["payoff_evals"] = float(np.mean(payoff_evals))
    return out


def loglog_slope(x, y) -> dict[str, float]:
    """OLS fit of log y = a + b log x; returns slope b with SE and 95% CI.

    The CI treats the points as independent, which is approximately true
    when each point comes from independent replications (as in our
    convergence study), but it ignores the uncertainty of each RMSE.
    """
    x = np.log(np.asarray(x, dtype=float))
    y = np.log(np.asarray(y, dtype=float))
    res = stats.linregress(x, y)
    tcrit = stats.t.ppf(0.975, max(len(x) - 2, 1))
    return {
        "slope": res.slope,
        "slope_se": res.stderr,
        "slope_lo": res.slope - tcrit * res.stderr,
        "slope_hi": res.slope + tcrit * res.stderr,
        "intercept": res.intercept,
        "r2": res.rvalue**2,
    }


def paired_mse_comparison(est_a, est_b, truth: float) -> dict[str, float]:
    """Paired comparison of two estimators evaluated on the same replications.

    Each replication r produces both theta_a,r and theta_b,r from the same
    base normals, so the squared errors are positively correlated and the
    difference d_r = e_a,r^2 - e_b,r^2 has smaller variance than for
    independent runs. Returns mean(d), its SE, a 95% CI and the MSE ratio.
    A CI excluding 0 is evidence (at roughly the 5% level, for this single
    comparison) that the MSEs differ; no multiple-testing correction is applied.
    """
    a = (np.asarray(est_a, float) - truth) ** 2
    b = (np.asarray(est_b, float) - truth) ** 2
    d = a - b
    R = d.size
    se = d.std(ddof=1) / math.sqrt(R)
    tcrit = stats.t.ppf(0.975, R - 1)
    return {
        "mse_a": a.mean(), "mse_b": b.mean(), "mse_ratio": a.mean() / b.mean(),
        "diff": d.mean(), "diff_se": se, "diff_lo": d.mean() - tcrit * se, "diff_hi": d.mean() + tcrit * se,
    }


def theoretical_rmse(bias: float, var1: float, n) -> np.ndarray:
    """RMSE(N) = sqrt(bias^2 + var1 / N) from exact moments."""
    n = np.asarray(n, dtype=float)
    return np.sqrt(bias**2 + var1 / n)

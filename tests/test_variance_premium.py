"""Phase 8B: point-in-time helpers of scripts/phase8_variance_premium_study.py (windows, no look-ahead, tests)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
vp = pytest.importorskip("phase8_variance_premium_study")


def test_windows_cover_exactly_30_calendar_days():
    dates = pd.Series(pd.bdate_range("2024-01-01", "2024-06-28"))
    dates = dates[dates != pd.Timestamp("2024-05-27")].reset_index(drop=True)        # a holiday
    n = vp.windows(dates)
    d = dates.to_numpy().astype("datetime64[D]")
    for t in range(len(d) - 30):
        end = t + n[t]
        assert d[end] <= d[t] + np.timedelta64(30, "D")                                  # inside the horizon
        if end + 1 < len(d):
            assert d[end + 1] > d[t] + np.timedelta64(30, "D")                           # and maximal
    assert set(n[: len(d) - 30]) <= {19, 20, 21, 22}


def test_payoff_regression_uses_only_known_payoffs():
    """Corrupting every payoff not yet known at origin i must not change the forecast at i."""
    rng = np.random.default_rng(0)
    N = 400
    d = pd.DataFrame({"origin": np.arange(N), "end": np.arange(N) + 21, "P": rng.standard_normal(N), "s": rng.standard_normal(N)})
    ph, pb = vp.oos_payoff_forecasts(d, ["s"])
    i = 350
    d2 = d.copy()
    unknown = d2.end > d2.origin[i]
    d2.loc[unknown, "P"] = 1e6
    ph2, pb2 = vp.oos_payoff_forecasts(d2, ["s"])
    assert ph[i] == pytest.approx(ph2[i]) and pb[i] == pytest.approx(pb2[i])
    # payoff j is known at origin i iff j + 21 <= i, so 250 known payoffs first exist at i = 250 + 20
    assert np.isnan(ph[: vp.MIN_CAL + 20]).all() and np.isfinite(ph[vp.MIN_CAL + 20])


def test_lagged_signal_uses_previous_day():
    d = pd.DataFrame({"origin": np.arange(300), "end": np.arange(300) - 1000, "P": np.arange(300.0), "s": np.arange(300.0) * 2})
    ph0, _ = vp.oos_payoff_forecasts(d, ["s"], lag=0)
    ph1, _ = vp.oos_payoff_forecasts(d, ["s"], lag=1)
    assert np.isnan(ph1[0])
    # P_j = j and s_j = 2j. Lag 0 fits P = s/2 and predicts 299 from s_299 = 598.
    # Lag 1 regresses P_j on s_{j-1} = 2(j-1), i.e. P = s/2 + 1, and must predict from s_298 = 596: 299.
    # Using s_299 by mistake would give 300.
    assert ph0[-1] == pytest.approx(299.0) and ph1[-1] == pytest.approx(299.0)


def test_holm_and_clark_west():
    np.testing.assert_allclose(vp.holm(np.array([0.01, 0.04, 0.03])), [0.03, 0.06, 0.06])
    rng = np.random.default_rng(1)
    y = rng.standard_normal(3000)
    z_good, p_good = vp.clark_west(y, np.zeros_like(y), y + 0.3 * rng.standard_normal(3000))
    z_bad, p_bad = vp.clark_west(y, np.zeros_like(y), 0.5 * rng.standard_normal(3000))
    assert p_good < 1e-6 and p_bad > 0.05

"""Smoke tests of the experiment engine (small sizes)."""
import numpy as np
import pandas as pd

from mc_greeks import PAPER_PARAMS, CashDigitalCall, EuropeanCall, IntervalDigital
from mc_greeks.experiments import Spec, build_specs, run_replications, summarize


def test_replications_independent_and_reproducible():
    f = IntervalDigital(100, 110)
    specs = [Spec("Malliavin", "delta"), Spec("FD-CRN-ctr", "delta", 1.0), Spec("FD-IND-ctr", "delta", 1.0)]
    a = run_replications(f, PAPER_PARAMS, specs, [500], 5, seed=[1, 2], extra={"param": 0.3}, with_exact=False)
    b = run_replications(f, PAPER_PARAMS, specs, [500], 5, seed=[1, 2], extra={"param": 0.3}, with_exact=False)
    pd.testing.assert_series_equal(a["value"], b["value"])  # same seed -> same numbers
    for _, g in a.groupby("method"):
        assert g["value"].nunique() == 5  # replications differ (regression: extra columns must not clobber 'value')
    assert (a["param"] == 0.3).all()


def test_equal_budget_sample_sizes():
    f = CashDigitalCall(100)
    specs = [Spec("Malliavin", "gamma"), Spec("FD-CRN-ctr", "gamma", 2.0), Spec("FD-CRN-ctr", "delta", 1.0)]
    raw = run_replications(f, PAPER_PARAMS, specs, [], 3, seed=5, budget=6000, with_exact=False)
    n = raw.groupby(["method", "greek"])["n"].first()
    assert n[("Malliavin", "gamma")] == 6000 and n[("FD-CRN-ctr", "gamma")] == 2000 and n[("FD-CRN-ctr", "delta")] == 3000
    assert (raw["payoff_evals"] == 6000).all()


def test_inapplicable_recorded_not_estimated():
    specs, skipped = build_specs(IntervalDigital(100, 110), PAPER_PARAMS, ["delta", "gamma"], ["Pathwise", "Malliavin"])
    assert all(s.method == "Malliavin" for s in specs)
    assert {(s["method"], s["greek"]) for s in skipped} == {("Pathwise", "delta"), ("Pathwise", "gamma")}


def test_summary_with_exact_theory():
    c = EuropeanCall(100)
    raw = run_replications(c, PAPER_PARAMS, [Spec("Pathwise", "delta")], [4000], 40, seed=9)
    s = summarize(raw, ["payoff", "greek", "method", "n"]).iloc[0]
    assert abs(s["exact_bias"]) < 1e-10
    assert 0.5 < s["var_ratio_emp_over_exact"] < 1.6  # chi-square band for 39 df is roughly [0.55, 1.55]
    assert np.isfinite(s["rmse_theory"])

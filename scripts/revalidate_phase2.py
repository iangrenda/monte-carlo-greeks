"""Phase-3 audit: revalidate the phase-2 Asian results under correct cost accounting.

    python revalidate_phase2.py                 # ~20-30 min (re-runs the adaptive variants)
    python revalidate_phase2.py --analysis-only

1. Re-runs the three phase-2 adaptive variants on the research grid with
   ``accounting="charged"``, recording the pilot's measured wall time.
2. Compares every strategy on a time-normalised basis,
   wnMSE = corrected MSE x mean *measured* wall time per estimate (pilot
   included for the adaptive variants). For unbiased MC estimators MSE ~ 1/time,
   so wnMSE is budget-free. The time-normalised regret is
   sqrt(wnMSE / min over fixed strategies of wnMSE).
3. Recomputes RQMC confidence-interval coverage with the Student-t quantile on
   (scrambles - 1) degrees of freedom instead of z = 1.96.

The stored phase-2 fixed-strategy runs are reused unchanged (their measured
runtimes are in fixed_raw.csv.gz). Output: results/asian/research_revalidation/.
"""
import argparse
import math

import numpy as np
import pandas as pd
from scipy import stats

from mc_greeks.asian import rqmc_layout
from mc_greeks.asian_experiments import (ADAPTIVE_VARIANTS, ASIAN_DIR, ASIAN_RESEARCH, CostModel, run_adaptive,
                                         summarize_runs)

SRC = ASIAN_DIR / "research"
OUT = ASIAN_DIR / "research_revalidation"
REPS = 20


def simulate(cfg):
    OUT.mkdir(parents=True, exist_ok=True)
    cost = CostModel(pd.read_csv(SRC / "cost_calibration.csv"))
    rows = []
    for b_i, B in enumerate(cfg.budgets):
        for s in cfg.scenarios():
            # new seed family (stage 30) so these runs are independent of the stored ones
            rows.append(run_adaptive(s, cost, B, REPS, cfg.pilot_fraction, [cfg.seed, 30, b_i, s.sid],
                                     variants=tuple(ADAPTIVE_VARIANTS), accounting="charged"))
            print(f"budget {B} scenario {s.sid + 1}/36", flush=True)
    pd.concat(rows).to_csv(OUT / "adaptive_charged_raw.csv.gz", index=False)


def analyse(cfg):
    scen = pd.read_csv(SRC / "scenarios.csv")
    refs = pd.read_csv(SRC / "references.csv")
    fixed = pd.read_csv(SRC / "fixed_raw.csv.gz")
    ad = pd.read_csv(OUT / "adaptive_charged_raw.csv.gz")
    ad_run = ad.copy()
    ad_run["runtime"] = ad_run["total_wall"]  # charge pilot + evaluation wall time
    summ = summarize_runs(pd.concat([fixed, ad_run.drop(columns=["total_wall"])]), refs).merge(scen, on="sid")
    summ["wn_mse"] = summ["mse_corrected"] * summ["runtime"]
    key = ["sid", "budget", "greek"]
    fx = summ[~summ.strategy.str.startswith("adaptive")]
    best = fx.groupby(key)["wn_mse"].min().rename("best_fixed_wn_mse")
    summ = summ.merge(best, on=key)
    summ["tn_regret"] = np.sqrt(summ["wn_mse"] / summ["best_fixed_wn_mse"])
    summ["time_over_budget"] = summ["runtime"] / summ["budget"]
    summ.to_csv(OUT / "summary_time_normalised.csv", index=False)

    def gm(x):
        return float(np.exp(np.log(x).mean()))

    strategies = ["CPW-first/RQMC", "CPW-first/MC", "FD-smooth/RQMC", "LR/MC"] + list(ADAPTIVE_VARIANTS)
    tab = (summ[summ.strategy.isin(strategies)].groupby(["budget", "greek", "strategy"])
           .agg(gm_tn_regret=("tn_regret", gm), worst_tn_regret=("tn_regret", "max"),
                median_time_over_budget=("time_over_budget", "median")).reset_index())
    tab.to_csv(OUT / "time_normalised_regret.csv", index=False)

    acc = ad.merge(scen, on="sid")
    acc["charged/nominal"] = acc.pilot_charged / acc.pilot_nominal
    acc["pilot_wall/nominal"] = acc.pilot_wall / acc.pilot_nominal
    acc["total_wall/B"] = acc.total_wall / acc.budget
    acc.groupby(["strategy", "budget", "m"])[["charged/nominal", "pilot_wall/nominal", "total_wall/B"]].median() \
        .reset_index().to_csv(OUT / "pilot_accounting.csv", index=False)

    # RQMC CI coverage: z = 1.96 (as used in phase 2) vs t_{R-1}
    cost = CostModel(pd.read_csv(SRC / "cost_calibration.csv"))
    rq = fixed[fixed.sampler == "RQMC"].merge(scen[["sid", "m"]], on="sid").merge(
        refs.melt(id_vars="sid", value_vars=["ref_delta", "ref_gamma", "ref_vega"], var_name="g", value_name="ref")
        .assign(greek=lambda d: d.g.str[4:]).drop(columns="g"), on=["sid", "greek"])
    rq["n_req"] = [cost.paths(b, m, me, "RQMC", g) for b, m, me, g in zip(rq.budget, rq.m, rq.method, rq.greek)]
    rq["n_scr"] = [rqmc_layout(n)[1] for n in rq.n_req]
    rq["err"] = (rq.value - rq.ref).abs()
    rq["cov_z"] = rq.err <= 1.96 * rq.stderr
    rq["cov_t"] = rq.err <= stats.t.ppf(0.975, rq.n_scr - 1) * rq.stderr
    mc = fixed[fixed.sampler == "MC"].merge(
        refs.melt(id_vars="sid", value_vars=["ref_delta", "ref_gamma", "ref_vega"], var_name="g", value_name="ref")
        .assign(greek=lambda d: d.g.str[4:]).drop(columns="g"), on=["sid", "greek"])
    mc_cov = ((mc.value - mc.ref).abs() <= 1.96 * mc.stderr).groupby(mc.method).mean()
    cov = rq.groupby("method")[["cov_z", "cov_t"]].mean()
    cov["mean_scrambles"] = rq.groupby("method").n_scr.mean()
    cov = cov.join(mc_cov.rename("mc_cov_z"), how="outer")
    cov.reset_index().to_csv(OUT / "ci_coverage_revalidated.csv", index=False)
    print(tab.to_string())
    print(cov.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis-only", action="store_true")
    a = ap.parse_args()
    if not a.analysis_only:
        simulate(ASIAN_RESEARCH)
    analyse(ASIAN_RESEARCH)

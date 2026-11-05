"""Build every figure from saved results (results/tables, results/experiment_data)."""
import pandas as pd

from mc_greeks import visualization as viz
from mc_greeks.config import DATA, TABLES


def _t(name):
    p = TABLES / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


def _r(name):
    p = DATA / f"{name}.csv.gz"
    return pd.read_csv(p) if p.exists() else None


def figures_a():
    traj = _r("A_paper_trajectories")
    if traj is not None:
        viz.plot_paper_trajectories(traj, "interval")
        viz.plot_localization(traj)
    rq = _t("A_rqmc_vs_mc")
    if rq is not None:
        viz.plot_rqmc(rq)


def figures_b():
    raw, summ = _r("B_convergence_raw"), _t("B_convergence_summary")
    if raw is None:
        return
    for i, g in enumerate(("delta", "gamma", "vega"), start=1):
        viz.plot_estimate_bands(raw, "interval", g, f"fig0{i}_interval_{g}_convergence")
    s = summ[summ.greek != "price"]
    viz.plot_rmse_grid(s, "n", "fig04_rmse_vs_n", "RMSE vs sample size (fixed, tuned FD bumps)", "Paths $N$")
    viz.plot_rmse_vs_cost(s, "fig05_rmse_vs_runtime")
    viz.plot_runtime(s, "figB_runtime_vs_n")
    for pname in ("interval", "call"):
        viz.plot_bias_variance(s, pname, f"figB_bias_variance_{pname}")


def figures_c():
    s = _t("C_bump_summary")
    if s is not None:
        viz.plot_bump(s, "fig06_bump_size_rmse")


def figures_sweeps():
    d = _t("D_volatility_summary")
    if d is not None:
        viz.plot_sweep(d, r"volatility $\sigma$", "fig07_variance_vs_volatility",
                       r"Estimator variance across volatility ($N$ fixed; dotted: exact theory)", metric="variance")
        viz.plot_sweep(d, r"volatility $\sigma$", "figD_rel_rmse_vs_volatility", "Relative RMSE across volatility")
    e = _t("E_moneyness_summary")
    if e is not None:
        viz.plot_sweep(e, r"moneyness $K/S_0$", "fig08_rmse_vs_moneyness", "Relative RMSE across strike (cash digital, call)")
        viz.plot_sweep(e, r"moneyness $K/S_0$", "figE_abs_rmse_vs_moneyness", "Absolute RMSE across strike", metric="rmse")
    f = _t("F_maturity_summary")
    if f is not None:
        viz.plot_sweep(f, r"maturity $T$ [years]", "fig09_rmse_vs_maturity", "Relative RMSE across maturity", logx=True)
        viz.plot_sweep(f, r"maturity $T$ [years]", "figF_variance_vs_maturity",
                       "Estimator variance across maturity (dotted: exact theory)", metric="variance", logx=True)
    g = _t("G_equal_budget_summary")
    if g is not None:
        viz.plot_equal_budget(g, "fig10_smooth_vs_discontinuous_equal_budget")


def main():
    figures_a()
    figures_b()
    figures_c()
    figures_sweeps()
    print("figures written to results/figures")


if __name__ == "__main__":
    main()

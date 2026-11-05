"""Publication-style figures. Every function plots data produced by an
experiment in :mod:`mc_greeks.experiments` (read back from results/), never
synthetic data.

Encoding: each method keeps a fixed colour (validated categorical palette,
CVD-checked) *and* a fixed marker; CRN finite differences are solid, IND
finite differences dashed, so identity never relies on colour alone.
Theory curves (exact bias/variance by quadrature) are thin dotted lines.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import FIGURES  # noqa: E402

STYLE = {
    "Malliavin": dict(color="#2a78d6", marker="o", ls="-", label="Malliavin weight"),
    "Malliavin-loc": dict(color="#008300", marker="D", ls="-", label="Localised Malliavin (call)"),
    "Pathwise": dict(color="#1baf7a", marker="^", ls="-", label="Pathwise (call)"),
    "FD-CRN-ctr": dict(color="#eb6834", marker="s", ls="-", label="FD central, CRN"),
    "FD-CRN-fwd": dict(color="#eda100", marker="v", ls="-", label="FD forward, CRN"),
    "FD-IND-ctr": dict(color="#6250d6", marker="s", ls="--", label="FD central, independent"),
    "FD-IND-fwd": dict(color="#e87ba4", marker="v", ls="--", label="FD forward, independent"),
    "MC": dict(color="#2a78d6", marker="o", ls="-", label="Plain MC (price)"),
}
GREEK_TEX = {"price": "Price $V$", "delta": r"Delta $\Delta$", "gamma": r"Gamma $\Gamma$", "vega": r"Vega $\mathcal{V}$"}
PAYOFF_TEX = {"interval": r"Interval digital $1\{100\leq S_T\leq 110\}$", "digital": r"Cash digital $1\{S_T>K\}$",
              "call": r"European call $(S_T-K)^+$"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "font.size": 9.5, "axes.titlesize": 10, "axes.labelsize": 9.5,
    "legend.fontsize": 8, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 1.6, "lines.markersize": 4.5, "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "mathtext.fontset": "cm",
})


def _save(fig, name: str) -> list[Path]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    paths = [FIGURES / f"{name}.png", FIGURES / f"{name}.pdf"]
    for p in paths:
        fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return paths


def _style(m):
    s = STYLE[m]
    out = dict(color=s["color"], marker=s["marker"], ls=s["ls"], label=s["label"])
    if m == "Pathwise":  # often coincides with CRN-FD at tiny bumps: draw hollow and on top
        out.update(markerfacecolor="none", markersize=8, markeredgewidth=1.4, zorder=5, lw=1.2)
    return out


def _legend_below(fig, axes, ncol=4):
    handles, labels = {}, {}
    for ax in np.ravel(axes):
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, h)
    fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, -0.02))


# --- paper-style trajectories (Experiment A) -----------------------------------------

def plot_paper_trajectories(traj: pd.DataFrame, payoff="interval", name_prefix="figA") -> list[Path]:
    """(a) Malliavin running estimate on a tight axis, as in FLLLT Figs. 1-3;
    (b) the same together with CRN central finite differences (h = 1% S0 or
    0.01 vol) on a wide axis. One sequence each: illustrative, not error bars."""
    out = []
    for i, g in enumerate(("delta", "gamma", "vega"), start=1):
        d = traj[(traj.payoff == payoff) & (traj.greek == g)]
        truth = d.truth.iloc[0]
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
        for ax, subset in zip(axes, (d[d.method == "Malliavin"], d)):
            for (m, smp), dd in subset.groupby(["method", "sampler"]):
                st = STYLE[m]
                ax.plot(dd.n, dd.estimate, color=st["color"], ls="-" if smp.startswith("Sobol") else ":",
                        lw=1.4 if smp.startswith("Sobol") else 1.1, label=f"{st['label']}, {smp}")
            ax.axhline(truth, color=INK, lw=0.9, ls="--", label=f"Analytical {truth:.6g}")
            ax.set_xscale("log")
            lo, hi = np.nanpercentile(subset[subset.n >= 500].estimate, [0.5, 99.5])
            pad = 0.3 * (hi - lo) + 1e-12
            ax.set_ylim(min(lo, truth) - pad, max(hi, truth) + pad)
            ax.set_xlabel("Paths $N$ (running estimate along one sequence)")
            ax.legend(loc="best", fontsize=7.5)
        axes[0].set_ylabel(GREEK_TEX[g])
        axes[0].set_title("(a) Malliavin weight (as in the paper's figure)")
        axes[1].set_title(r"(b) with CRN central finite differences")
        fig.suptitle(f"{GREEK_TEX[g]} of the {PAYOFF_TEX[payoff]}", y=1.02)
        fig.tight_layout()
        out += _save(fig, f"{name_prefix}{i}_paper_{payoff}_{g}_trajectory")
    return out


def plot_localization(traj: pd.DataFrame) -> list[Path]:
    d = traj[(traj.payoff == "call") & (traj.greek == "gamma") & (traj.method != "FD-CRN-ctr")
             & (traj.sampler.str.startswith("Sobol"))]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for m, dd in d.groupby("method"):
        ax.plot(dd.n, dd.estimate, **{k: v for k, v in _style(m).items() if k != "marker"})
    ax.axhline(d.truth.iloc[0], color=INK, lw=0.9, ls="--", label="Analytical")
    ax.set_xscale("log")
    ax.set_xlabel("Number of simulated paths $N$ (scrambled Sobol, running estimate)")
    ax.set_ylabel(r"Call $\Gamma$")
    ax.set_title(r"Call Gamma ($K=100$): global vs localised Malliavin weight ($\delta=10$)")
    ax.legend()
    return _save(fig, "figA4_call_gamma_localization")


def plot_rqmc(rq: pd.DataFrame) -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharex=True)
    for ax, g in zip(axes, ("delta", "gamma", "vega")):
        for smp, dd in rq[rq.greek == g].groupby("sampler"):
            ax.loglog(dd.n, dd.rmse, marker="o" if smp.startswith("Sobol") else "s",
                      color="#2a78d6" if smp.startswith("Sobol") else "#eb6834", label=smp)
        ax.set_title(f"Malliavin {GREEK_TEX[g]}")
        ax.set_xlabel("$N$")
    axes[0].set_ylabel("RMSE over replications")
    _legend_below(fig, axes, ncol=2)
    fig.suptitle("Interval digital: randomised QMC (scrambled Sobol) vs pseudorandom MC", y=1.02)
    fig.tight_layout()
    return _save(fig, "figA5_rqmc_vs_mc")


# --- Experiment B --------------------------------------------------------------------

def plot_estimate_bands(raw: pd.DataFrame, payoff: str, greek: str, fig_name: str) -> list[Path]:
    """Mean and 2.5-97.5% replication band of the estimate vs N (Figures 1-3)."""
    d = raw[(raw.payoff == payoff) & (raw.greek == greek)]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    methods = [m for m in STYLE if m in d.method.unique()]
    for k, m in enumerate(methods):
        dm = d[d.method == m].groupby("n")["value"]
        q = dm.quantile([0.025, 0.5, 0.975]).unstack()
        x = q.index.to_numpy() * (1 + 0.04 * (k - len(methods) / 2))  # small dodge
        st = STYLE[m]
        ax.errorbar(x, q[0.5], yerr=[q[0.5] - q[0.025], q[0.975] - q[0.5]], color=st["color"], marker=st["marker"],
                    ls=st["ls"], lw=1.2, capsize=2, label=st["label"])
    truth = d.truth.iloc[0]
    ax.axhline(truth, color=INK, lw=0.9, ls="--", label="Analytical")
    ax.set_xscale("log")
    spread = d.groupby("method")["value"].quantile(0.975) - d.groupby("method")["value"].quantile(0.025)
    good = d[d.method.isin(spread[spread < 50 * spread.min()].index)]
    lo, hi = good.value.quantile([0.01, 0.99])
    ax.set_ylim(lo - 0.15 * (hi - lo), hi + 0.15 * (hi - lo))
    ax.set_xlabel("Paths per estimate $N$")
    ax.set_ylabel(GREEK_TEX[greek])
    ax.set_title(f"{GREEK_TEX[greek]}, {PAYOFF_TEX[payoff]}: median and 95% replication band")
    ax.legend(ncol=2, loc="best")
    return _save(fig, fig_name)


def plot_rmse_grid(summ: pd.DataFrame, x: str, fig_name: str, title: str, xlabel: str,
                   theory: bool = True, relative: bool = True) -> list[Path]:
    """Rows = payoffs, columns = Greeks, RMSE (relative) vs x on log-log axes."""
    pays = [p for p in ("interval", "digital", "call") if p in summ.payoff.unique()]
    fig, axes = plt.subplots(len(pays), 3, figsize=(10.5, 2.7 * len(pays) + 0.6), squeeze=False)
    for i, pname in enumerate(pays):
        for j, g in enumerate(("delta", "gamma", "vega")):
            ax = axes[i, j]
            d = summ[(summ.payoff == pname) & (summ.greek == g)]
            for m in [m for m in STYLE if m in d.method.unique()]:
                dm = d[d.method == m].sort_values(x)
                scale = np.abs(dm.truth) if relative else 1.0
                ax.loglog(dm[x], dm.rmse / scale, **_style(m))
                if theory and dm.rmse_theory.notna().all():
                    ax.loglog(dm[x], dm.rmse_theory / scale, color=STYLE[m]["color"], ls=":", lw=1.0, marker=None)
            if i == 0:
                ax.set_title(GREEK_TEX[g])
            if j == 0:
                ax.set_ylabel(f"{pname}\n{'relative ' if relative else ''}RMSE")
            if i == len(pays) - 1:
                ax.set_xlabel(xlabel)
    _legend_below(fig, axes, ncol=4)
    fig.suptitle(title + ("  (dotted: exact theory)" if theory else ""), y=1.0)
    fig.tight_layout(rect=(0, 0.06, 1, 0.98))
    return _save(fig, fig_name)


def plot_rmse_vs_cost(summ: pd.DataFrame, fig_name: str) -> list[Path]:
    """RMSE against wall-clock time and against payoff evaluations (interval digital + call)."""
    pays = ["interval", "call"]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.4), squeeze=False)
    for i, pname in enumerate(pays):
        for j, g in enumerate(("delta", "gamma", "vega")):
            ax = axes[i, j]
            d = summ[(summ.payoff == pname) & (summ.greek == g)]
            for m in [m for m in STYLE if m in d.method.unique()]:
                dm = d[d.method == m].sort_values("runtime")
                ax.loglog(dm.runtime * 1e3, dm.rmse / abs(dm.truth.iloc[0]), **_style(m))
            if i == 0:
                ax.set_title(GREEK_TEX[g])
            if j == 0:
                ax.set_ylabel(f"{pname}\nrelative RMSE")
            if i == 1:
                ax.set_xlabel("Mean wall-clock time per estimate [ms]")
    _legend_below(fig, axes, ncol=4)
    fig.suptitle("Accuracy-cost frontier: relative RMSE vs measured runtime (lower-left is better)", y=1.0)
    fig.tight_layout(rect=(0, 0.06, 1, 0.98))
    return _save(fig, fig_name)


def plot_runtime(summ: pd.DataFrame, fig_name: str) -> list[Path]:
    d = summ[(summ.payoff == "interval")]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    for m in [m for m in STYLE if m in d.method.unique() and m != "MC"]:
        dm = d[(d.method == m) & (d.greek == ("gamma" if m != "Pathwise" else "delta"))].sort_values("n")
        if dm.empty:
            continue
        axes[0].loglog(dm.n, dm.runtime * 1e3, **_style(m))
        axes[1].semilogx(dm.n, dm.runtime / dm.n * 1e9, **_style(m))
    axes[0].set_ylabel("Wall-clock per estimate [ms]")
    axes[1].set_ylabel("Wall-clock per path [ns]")
    for ax in axes:
        ax.set_xlabel("Paths $N$")
    _legend_below(fig, axes, ncol=4)
    fig.suptitle(r"Runtime of the Gamma estimators, interval digital (includes normal generation)", y=1.0)
    fig.tight_layout(rect=(0, 0.08, 1, 0.98))
    return _save(fig, fig_name)


def plot_bias_variance(summ: pd.DataFrame, payoff: str, fig_name: str) -> list[Path]:
    d = summ[summ.payoff == payoff]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.8), squeeze=False)
    for j, g in enumerate(("delta", "gamma", "vega")):
        for m in [m for m in STYLE if m in d[d.greek == g].method.unique()]:
            dm = d[(d.greek == g) & (d.method == m)].sort_values("n")
            ax = axes[0, j]
            ax.errorbar(dm.n, dm.bias, yerr=1.96 * dm.bias_se, color=STYLE[m]["color"], marker=STYLE[m]["marker"],
                        ls=STYLE[m]["ls"], lw=1.1, capsize=2, label=STYLE[m]["label"])
            axes[1, j].loglog(dm.n, dm.variance, **_style(m))
        axes[0, j].axhline(0, color=INK, lw=0.8)
        axes[0, j].set_xscale("log")
        axes[0, j].set_yscale("symlog", linthresh=max(1e-12, d[d.greek == g].bias_se.median()))
        axes[0, j].set_title(GREEK_TEX[g])
        axes[1, j].set_xlabel("Paths $N$")
    axes[0, 0].set_ylabel("Bias (95% CI)")
    axes[1, 0].set_ylabel("Variance of estimator")
    _legend_below(fig, axes, ncol=4)
    fig.suptitle(f"Bias and variance vs $N$: {PAYOFF_TEX[payoff]}", y=1.0)
    fig.tight_layout(rect=(0, 0.07, 1, 0.98))
    return _save(fig, fig_name)


# --- Experiment C --------------------------------------------------------------------

def plot_bump(summ: pd.DataFrame, fig_name: str) -> list[Path]:
    pays = ["interval", "digital", "call"]
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 8.6), squeeze=False)
    for i, pname in enumerate(pays):
        for j, g in enumerate(("delta", "gamma", "vega")):
            ax = axes[i, j]
            d = summ[(summ.payoff == pname) & (summ.greek == g)]
            for m in [m for m in STYLE if m in d.method.unique()]:
                dm = d[d.method == m].sort_values("h_rel")
                ax.loglog(dm.h_rel, dm.rmse, **_style(m))
                ax.loglog(dm.h_rel, dm.rmse_theory, color=STYLE[m]["color"], ls=":", lw=1.0)
            if i == 0:
                ax.set_title(GREEK_TEX[g])
            if j == 0:
                ax.set_ylabel(f"{pname}\nRMSE")
            if i == 2:
                ax.set_xlabel("bump $h/S_0$" if g != "vega" else r"vol bump $h_\sigma$")
    _legend_below(fig, axes, ncol=4)
    n = int(summ.n.iloc[0])
    fig.suptitle(f"Finite-difference RMSE vs bump size, $N={n:,}$ paths per bumped price (dotted: exact theory)", y=1.0)
    fig.tight_layout(rect=(0, 0.05, 1, 0.98))
    return _save(fig, fig_name)


# --- Experiments D-F -----------------------------------------------------------------

def plot_sweep(summ: pd.DataFrame, xlabel: str, fig_name: str, title: str, metric: str = "rel_rmse",
               logx: bool = False) -> list[Path]:
    pays = [p for p in ("interval", "digital", "call") if p in summ.payoff.unique()]
    fig, axes = plt.subplots(len(pays), 3, figsize=(10.5, 2.7 * len(pays) + 0.6), squeeze=False)
    for i, pname in enumerate(pays):
        for j, g in enumerate(("delta", "gamma", "vega")):
            ax = axes[i, j]
            d = summ[(summ.payoff == pname) & (summ.greek == g)]
            for m in [m for m in STYLE if m in d.method.unique()]:
                dm = d[d.method == m].sort_values("param")
                y = dm[metric]
                ax.plot(dm.param, y, **_style(m))
                if metric == "variance":
                    ax.plot(dm.param, dm.exact_sd**2, color=STYLE[m]["color"], ls=":", lw=1.0)
            ax.set_yscale("log")
            if logx:
                ax.set_xscale("log")
            if i == 0:
                ax.set_title(GREEK_TEX[g])
            if j == 0:
                ax.set_ylabel(f"{pname}\n" + {"rel_rmse": "relative RMSE", "variance": "estimator variance",
                                               "rmse": "RMSE"}[metric])
            if i == len(pays) - 1:
                ax.set_xlabel(xlabel)
    _legend_below(fig, axes, ncol=4)
    fig.suptitle(title, y=1.0)
    fig.tight_layout(rect=(0, 0.06, 1, 0.98))
    return _save(fig, fig_name)


# --- Experiment G --------------------------------------------------------------------

def plot_equal_budget(summ: pd.DataFrame, fig_name: str) -> list[Path]:
    pays = ["interval", "digital", "call"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    methods = [m for m in STYLE if m in summ.method.unique() and m != "MC"]
    w = 0.8 / len(methods)
    for ax, g in zip(axes, ("delta", "gamma", "vega")):
        for k, m in enumerate(methods):
            d = summ[(summ.greek == g) & (summ.method == m)].set_index("payoff").reindex(pays)
            x = np.arange(len(pays)) + (k - len(methods) / 2 + 0.5) * w
            ax.bar(x, d.rmse_ratio_vs_malliavin, width=w * 0.92, color=STYLE[m]["color"], label=STYLE[m]["label"],
                   hatch="///" if STYLE[m]["ls"] == "--" else None, edgecolor="white", linewidth=0.5)
        ax.axhline(1.0, color=INK, lw=0.9)
        ax.set_yscale("log")
        ax.set_xticks(range(len(pays)), pays)
        ax.set_title(GREEK_TEX[g])
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("RMSE / RMSE(Malliavin)\n(equal payoff-evaluation budget)")
    _legend_below(fig, axes, ncol=4)
    b = int(summ.payoff_evals.max())
    fig.suptitle(f"Smooth vs discontinuous payoffs at an equal budget of {b:,} payoff evaluations\n"
                 "ratio < 1: more accurate than the global Malliavin weight; missing bar: estimator inapplicable",
                 y=1.04)
    fig.tight_layout(rect=(0, 0.08, 1, 0.98))
    return _save(fig, fig_name)

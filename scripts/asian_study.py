"""Asian digital study (phase 2): estimator comparison (H1, H2, H4) and
pilot-based adaptive selection (H3, H5).

    python asian_study.py --config quick       # ~1 min pipeline check
    python asian_study.py --config research    # full study (~20-30 min)
    python asian_study.py --config research --analysis-only

Outputs go to results/asian/<config>/ (tables, raw data, figures). Earlier results in
results/tables and results/figures are not touched.
"""
import argparse
import json
import math
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mc_greeks import visualization as viz  # noqa: F401  (applies the house plotting style)
from mc_greeks.asian_experiments import (ADAPTIVE_VARIANTS, ASIAN_CONFIGS, ASIAN_DIR, CANDIDATES, GREEKS3, HEURISTICS, CostModel,
                                         calibrate_costs, label, reference, run_adaptive, run_fixed,
                                         summarize_runs)
from mc_greeks.experiments import environment_info

FIG = ASIAN_DIR / "figures"
FIXED = [label(m, s) for m, s in CANDIDATES]
COLORS = {"CPW-first": "#2a78d6", "CPW-last": "#1baf7a", "LR": "#eb6834", "Malliavin-avg": "#6250d6",
          "FD-raw": "#e87ba4", "FD-smooth": "#eda100"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def simulate(cfg):
    ASIAN_DIR.mkdir(parents=True, exist_ok=True)
    scen = cfg.scenarios()
    pd.DataFrame([dict(sid=s.sid, m=s.m, moneyness=s.moneyness, sigma=s.sigma, T=s.T, S0=s.S0, r=s.r) for s in scen]
                 ).to_csv(ASIAN_DIR / "scenarios.csv", index=False)
    t0 = time.time()
    log("calibrating per-path costs")
    cal = calibrate_costs(cfg.ms, seed=cfg.seed)
    cal.to_csv(ASIAN_DIR / "cost_calibration.csv", index=False)
    cost = CostModel(cal)
    log(f"references for {len(scen)} scenarios")
    refs = pd.DataFrame([reference(s, cfg, [cfg.seed, 1, s.sid]) for s in scen])
    refs.to_csv(ASIAN_DIR / "references.csv", index=False)
    fixed, adapt = [], []
    for b_i, B in enumerate(cfg.budgets):
        for s in scen:
            fixed.append(run_fixed(s, cost, B, cfg.reps, [cfg.seed, 2, b_i, s.sid]))
            adapt.append(run_adaptive(s, cost, B, cfg.reps, cfg.pilot_fraction, [cfg.seed, 3, b_i, s.sid]))
            log(f"budget {B}s scenario {s.sid + 1}/{len(scen)} done")
    fixed, adapt = pd.concat(fixed), pd.concat(adapt)
    fixed.to_csv(ASIAN_DIR / "fixed_raw.csv.gz", index=False)
    adapt.to_csv(ASIAN_DIR / "adaptive_raw.csv.gz", index=False)
    meta = {"config": cfg.to_dict(), "environment": environment_info(), "wall_seconds": time.time() - t0}
    (ASIAN_DIR / "meta.json").write_text(json.dumps(meta, indent=2, default=str))


def boot_gm(x, rng, n=2000):
    """Geometric mean of x with a percentile bootstrap 95% CI over scenarios."""
    lx = np.log(np.asarray(x, dtype=float))
    if lx.size == 0:
        return np.nan, np.nan, np.nan
    bs = [lx[rng.integers(0, len(lx), len(lx))].mean() for _ in range(n)]
    return math.exp(lx.mean()), math.exp(np.percentile(bs, 2.5)), math.exp(np.percentile(bs, 97.5))


def analyse(cfg):
    scen = pd.read_csv(ASIAN_DIR / "scenarios.csv")
    refs = pd.read_csv(ASIAN_DIR / "references.csv")
    fixed = pd.read_csv(ASIAN_DIR / "fixed_raw.csv.gz")
    adapt = pd.read_csv(ASIAN_DIR / "adaptive_raw.csv.gz")
    summ = summarize_runs(pd.concat([fixed, adapt]), refs).merge(scen, on="sid")
    summ["rmse_c"] = np.sqrt(summ["mse_corrected"])
    summ.to_csv(ASIAN_DIR / "summary.csv", index=False)
    rng = np.random.default_rng(cfg.seed)

    # reference quality
    chk = refs[[c for c in refs if c.startswith("check_z_")]].stack()
    ref_tab = pd.DataFrame({"n_checks": [len(chk)], "mean_z": [chk.mean()], "sd_z": [chk.std()],
                            "max_abs_z": [chk.abs().max()], "share_abs_z_gt_3": [(chk.abs() > 3).mean()],
                            "max_ref_se_over_rmse": [summ["ref_se_over_rmse"].max()],
                            "median_ref_se_over_rmse": [summ["ref_se_over_rmse"].median()]})
    ref_tab.to_csv(ASIAN_DIR / "reference_quality.csv", index=False)

    fx = summ[summ.strategy.isin(FIXED)]
    key = ["sid", "budget", "greek"]
    oracle = fx.loc[fx.groupby(key)["mse_corrected"].idxmin(), key + ["strategy", "rmse_c"]]
    oracle = oracle.rename(columns={"strategy": "oracle_choice", "rmse_c": "oracle_rmse"})
    summ = summ.merge(oracle, on=key)
    summ["regret"] = summ["rmse_c"] / summ["oracle_rmse"]
    summ.to_csv(ASIAN_DIR / "summary.csv", index=False)

    # H1 / H4: equal-budget MSE ratios vs CPW-first/MC
    base = fx[fx.strategy == "CPW-first/MC"].set_index(key)["mse_corrected"]
    rows = []
    for (b, g, strat), d in fx.groupby(["budget", "greek", "strategy"]):
        r = d.set_index(key)["mse_corrected"] / base.reindex(d.set_index(key).index)
        for mm in sorted(d.m.unique()):
            rr = r[d.set_index(key).m == mm].dropna()
            if rr.empty:
                continue
            gm, lo, hi = boot_gm(rr.clip(lower=1e-12), rng)
            rows.append({"budget": b, "greek": g, "strategy": strat, "m": mm, "gm_mse_ratio_vs_CPWfirstMC": gm,
                         "ci_lo": lo, "ci_hi": hi, "min": rr.min(), "max": rr.max(), "n_scen": len(rr)})
    h1 = pd.DataFrame(rows)
    h1.to_csv(ASIAN_DIR / "H1_H4_mse_ratios.csv", index=False)

    # H2: oracle choices by regime
    oc = oracle.merge(scen, on="sid")
    h2 = oc.groupby(["budget", "greek", "oracle_choice"]).size().rename("count").reset_index()
    h2.to_csv(ASIAN_DIR / "H2_oracle_choices.csv", index=False)
    oc.to_csv(ASIAN_DIR / "H2_oracle_by_scenario.csv", index=False)

    # H3 / H5: adaptive vs fixed baselines, out of sample
    sids = scen.sid.to_numpy()
    perm = np.random.default_rng(cfg.seed + 1).permutation(sids)
    splits = {"random half": (set(perm[: len(sids) // 2]), set(perm[len(sids) // 2:])),
              f"train m<{max(cfg.ms)}, test m={max(cfg.ms)}": (set(scen.sid[scen.m < max(cfg.ms)]),
                                                                set(scen.sid[scen.m == max(cfg.ms)]))}
    if len(cfg.ms) < 2:
        splits.pop(f"train m<{max(cfg.ms)}, test m={max(cfg.ms)}")
    rows, picks = [], []
    for split, (train, test) in splits.items():
        for (b, g), d in summ.groupby(["budget", "greek"]):
            tr = d[d.sid.isin(train) & d.strategy.isin(FIXED)]
            best = tr.groupby("strategy")["regret"].apply(lambda x: np.log(x).mean()).idxmin()
            te = d[d.sid.isin(test)]
            cands = {**{f"fixed: {s}": s for s in FIXED}, **{f"heuristic: {k}": label(*v) for k, v in HEURISTICS.items()},
                     f"best fixed on train ({best})": best, **{v: v for v in ADAPTIVE_VARIANTS}}
            for name, strat in cands.items():
                x = te[te.strategy == strat].set_index("sid")["regret"]
                if x.empty:
                    continue
                gm, lo, hi = boot_gm(x, rng)
                rows.append({"split": split, "budget": b, "greek": g, "strategy": name, "gm_regret": gm,
                             "ci_lo": lo, "ci_hi": hi, "worst_regret": x.max(), "n_test": len(x)})
            bf = te[te.strategy == best].set_index("sid")
            for v in ADAPTIVE_VARIANTS:
              ad = te[te.strategy == v].set_index("sid")
              diff = ad["mse_corrected"] - bf["mse_corrected"]
              se = np.sqrt(ad["mse_se"] ** 2 + bf["mse_se"] ** 2)
              picks.append({"split": split, "budget": b, "greek": g, "variant": v, "best_fixed_train": best,
                          "adaptive_better_point": int((diff < 0).sum()), "adaptive_worse_point": int((diff > 0).sum()),
                          "adaptive_better_2se": int((diff < -2 * se).sum()), "adaptive_worse_2se": int((diff > 2 * se).sum()),
                          "n_test": len(diff)})
    h3 = pd.DataFrame(rows)
    h3.to_csv(ASIAN_DIR / "H3_regret_out_of_sample.csv", index=False)
    pd.DataFrame(picks).to_csv(ASIAN_DIR / "H3_adaptive_vs_best_fixed.csv", index=False)

    # selection behaviour of the adaptive rule
    ch = adapt.merge(oracle, on=key).merge(scen, on="sid")
    sel = ch.groupby(["strategy", "budget", "greek"]).apply(lambda d: pd.Series({
        "share_chose_oracle": (d.chosen == d.oracle_choice).mean(),
        "most_common_choice": d.chosen.value_counts().index[0],
        "share_most_common": d.chosen.value_counts(normalize=True).iloc[0],
        "n_distinct_choices": d.chosen.nunique()}), include_groups=False).reset_index()
    sel.to_csv(ASIAN_DIR / "H5_selection_behaviour.csv", index=False)
    ch.groupby(["strategy", "budget", "greek", "chosen"]).size().rename("count").reset_index().to_csv(
        ASIAN_DIR / "H5_choice_counts.csv", index=False)
    figures(summ, h1, h3, refs, cfg)
    log("analysis written to results/asian/")


def figures(summ, h1, h3, refs, cfg):
    FIG.mkdir(parents=True, exist_ok=True)
    B = max(cfg.budgets)
    # F1: MSE ratio vs CPW-first/MC by m (large budget)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, g in zip(axes, GREEKS3):
        d = h1[(h1.budget == B) & (h1.greek == g) & (h1.strategy != "CPW-first/MC")]
        strats = [s for s in FIXED if s in d.strategy.unique()]
        for k, s in enumerate(strats):
            ds = d[d.strategy == s].sort_values("m")
            x = np.arange(len(ds)) + (k - len(strats) / 2) * 0.08
            meth, smp = s.split("/")
            ax.errorbar(x, ds.gm_mse_ratio_vs_CPWfirstMC, yerr=[ds.gm_mse_ratio_vs_CPWfirstMC - ds.ci_lo, ds.ci_hi - ds.gm_mse_ratio_vs_CPWfirstMC],
                        color=COLORS[meth], marker="o" if smp == "MC" else "s", mfc="white" if smp == "RQMC" else COLORS[meth],
                        ls="-" if smp == "MC" else "--", lw=1.1, capsize=2, label=s)
        ax.axhline(1, color="#1f1f1e", lw=0.9)
        ax.set_yscale("log")
        ax.set_xticks(range(len(cfg.ms)), [f"m={m}" for m in cfg.ms])
        ax.set_title({"delta": r"Delta $\Delta$", "gamma": r"Gamma $\Gamma$", "vega": r"Vega $\mathcal{V}$"}[g])
    axes[0].set_ylabel("MSE / MSE(CPW-first, MC)\n(equal compute budget; geo-mean over scenarios)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.12))
    fig.suptitle(f"Asian digital: estimator MSE relative to CPW-first/MC at a budget of {B*1e3:.0f} ms (95% bootstrap CI over scenarios)", y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"figH1_mse_ratio_by_m.{ext}", bbox_inches="tight")
    plt.close(fig)

    # F2: oracle choice map
    oc = summ.drop_duplicates(["sid", "budget", "greek"])
    fig, axes = plt.subplots(len(cfg.budgets), 3, figsize=(12, 1.2 + 2.6 * len(cfg.budgets)), squeeze=False)
    strat_codes = {s: i for i, s in enumerate(FIXED)}
    for i, b in enumerate(cfg.budgets):
        for j, g in enumerate(GREEKS3):
            d = oc[(oc.budget == b) & (oc.greek == g)].sort_values(["m", "moneyness", "sigma", "T"])
            ax = axes[i, j]
            for k, row in enumerate(d.itertuples()):
                meth = row.oracle_choice.split("/")[0]
                ax.scatter(k, 0, s=60, color=COLORS[meth], marker="o" if row.oracle_choice.endswith("MC") else "s")
            ax.set_yticks([])
            ax.set_xticks(range(len(d)), [f"{r.m}|{r.moneyness}|{r.sigma}|{r.T}" for r in d.itertuples()], rotation=90, fontsize=5)
            ax.set_title(f"{g}, budget {b*1e3:.0f} ms", fontsize=9)
    handles = [plt.Line2D([], [], color=c, marker="o", ls="", label=k) for k, c in COLORS.items()]
    handles += [plt.Line2D([], [], color="gray", marker="s", ls="", label="square = RQMC")]
    fig.legend(handles=handles, loc="lower center", ncol=7, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Oracle (best fixed estimator, evaluation only) per scenario: labels m | K/S0 | sigma | T", y=1.0)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"figH2_oracle_choices.{ext}", bbox_inches="tight")
    plt.close(fig)

    # F3: out-of-sample regret
    for split in h3.split.unique():
        d = h3[h3.split == split]
        fig, axes = plt.subplots(len(cfg.budgets), 3, figsize=(12, 1.5 + 3.4 * len(cfg.budgets)), squeeze=False)
        for i, b in enumerate(cfg.budgets):
            for j, g in enumerate(GREEKS3):
                dd = d[(d.budget == b) & (d.greek == g)].sort_values("gm_regret")
                dd = pd.concat([dd[~dd.strategy.str.startswith("fixed")], dd[dd.strategy.str.startswith("fixed")].head(3)]).sort_values("gm_regret")
                ax = axes[i, j]
                y = np.arange(len(dd))
                col = ["#2a78d6" if s == "adaptive (pilot)" else "#6da7ec" if s.startswith("adaptive") else "#1f1f1e" if s.startswith("best") else "#9a9890" for s in dd.strategy]
                ax.barh(y, dd.gm_regret - 1, left=1, color=col, height=0.6)
                ax.errorbar(dd.gm_regret, y, xerr=[dd.gm_regret - dd.ci_lo, dd.ci_hi - dd.gm_regret], fmt="none", ecolor="#1f1f1e", lw=0.8, capsize=2)
                ax.set_yticks(y, [s.replace("heuristic: ", "") for s in dd.strategy], fontsize=6.5)
                ax.set_xscale("log")
                ax.axvline(1, color="#1f1f1e", lw=0.8)
                ax.set_title(f"{g}, budget {b*1e3:.0f} ms", fontsize=9)
                if i == len(cfg.budgets) - 1:
                    ax.set_xlabel("RMSE / oracle RMSE (test scenarios, geo-mean, 95% CI)")
        fig.suptitle(f"Out-of-sample regret, split: {split} (blue = adaptive, black = best fixed on train)", y=1.0)
        fig.tight_layout()
        tag = "random" if split.startswith("random") else "extrapolate_m"
        for ext in ("png", "pdf"):
            fig.savefig(FIG / f"figH3_regret_{tag}.{ext}", bbox_inches="tight")
        plt.close(fig)

    # F4: reference cross-check z-scores
    z = refs[[c for c in refs if c.startswith("check_z_")]].stack()
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(z, bins=20, color="#2a78d6", edgecolor="white")
    ax.set_xlabel("z = (CPW-last MC check - reference) / combined SE")
    ax.set_ylabel("count")
    ax.set_title(f"Reference cross-check ({len(z)} scenario-Greeks)")
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"figH0_reference_check.{ext}", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", choices=sorted(ASIAN_CONFIGS), default="quick")
    ap.add_argument("--analysis-only", action="store_true")
    a = ap.parse_args()
    cfg = ASIAN_CONFIGS[a.config]
    ASIAN_DIR = ASIAN_DIR / cfg.name  # one output folder per configuration (provenance)
    FIG = ASIAN_DIR / "figures"
    if not a.analysis_only:
        simulate(cfg)
    analyse(cfg)

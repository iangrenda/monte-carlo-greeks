"""Phase 4, Part B: confirmatory test of the pre-registered cost-based selection rule.

    python phase4_cost_validation.py              # full run (~45-60 min)
    python phase4_cost_validation.py --quick      # pipeline check (~2 min; NOT confirmatory)
    python phase4_cost_validation.py --analysis-only

Reads configs/phase4_cost_rule.json, which was frozen and committed before any
run (commit 5f657d4); nothing in this script tunes the rule. Output:
results/phase4_cost_rule/<run>/.
"""
import argparse
import json
import math
import platform
import subprocess
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mc_greeks import visualization as viz  # noqa: F401
from mc_greeks.asian import AsianDigital
from mc_greeks.asian_qmc import (CONSTRUCTIONS, SAMPLERS, calibrate_rqmc, pps_for, reference_estimates,
                                 timed_estimate, warm_cache)
from mc_greeks.config import RESULTS, ROOT
from mc_greeks.experiments import environment_info
from mc_greeks.models import BlackScholesParams
from mc_greeks.simulation import make_generator

CFG = json.loads((ROOT / "configs" / "phase4_cost_rule.json").read_text())
COL = {"MC": "#eb6834", "RQMC-seq": "#9a9890", "RQMC-bb": "#1baf7a", "RQMC-pca": "#2a78d6"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def provenance() -> dict:
    def git(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception as exc:  # pragma: no cover
            return f"unavailable ({exc})"
    return {**environment_info(), "git_commit": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain")), "machine": platform.machine(),
            "config_file": "configs/phase4_cost_rule.json"}


def scenarios(quick=False):
    g = CFG["grid"]
    ms = [8, 24] if quick else g["m"]
    ks = g["moneyness_K_over_S0"][:1] if quick else g["moneyness_K_over_S0"]
    ss = g["sigma"][:1] if quick else g["sigma"]
    Ts = g["T"][:1] if quick else g["T"]
    out, i = [], 0
    for m in ms:
        for k in ks:
            for s in ss:
                for T in Ts:
                    out.append(dict(sid=i, m=m, moneyness=k, sigma=s, T=T, r=g["r"], S0=g["S0"]))
                    i += 1
    return out


def objs(sc):
    return AsianDigital(sc["moneyness"] * sc["S0"], int(sc["m"])), BlackScholesParams(sc["S0"], sc["r"], sc["sigma"], sc["T"])


def rule_threshold(cal, m):
    a, b = cal[("pca", m)]
    return CFG["rule"]["n_scrambles"] * (a + CFG["rule"]["n_min_points"] * b)


def simulate(out, quick):
    out.mkdir(parents=True, exist_ok=True)
    seed = CFG["seed"]
    budgets = CFG["budgets_seconds"][:2] if quick else CFG["budgets_seconds"]
    reps = 4 if quick else CFG["replications"]
    scen = scenarios(quick)
    pd.DataFrame(scen).to_csv(out / "scenarios.csv", index=False)
    ms = sorted({s["m"] for s in scen})
    meta = {"provenance": provenance(), "quick": quick, "budgets": budgets, "reps": reps}

    log("pooled calibration (amortised one-off setup)")
    t0 = time.perf_counter()
    cal = calibrate_rqmc(tuple(ms), seed=seed + CFG["rule"]["cost_model"]["seed_offset"], pooled=True)
    meta["calibration_seconds"] = time.perf_counter() - t0
    rows = [dict(m=m, a=cal[("pca", m)][0], b=cal[("pca", m)][1], mc_sec_per_path=cal[("mc", m)][1],
                 rule_threshold_s=rule_threshold(cal, m)) for m in ms]
    pd.DataFrame(rows).to_csv(out / "calibration.csv", index=False)
    decisions = [dict(m=m, budget=B, rule_choice="RQMC-pca" if B >= rule_threshold(cal, m) else "MC") for m in ms for B in budgets]
    pd.DataFrame(decisions).to_csv(out / "rule_decisions.csv", index=False)
    log("rule decisions: " + ", ".join(f"m={d['m']} B={d['budget']*1e3:.0f}ms->{d['rule_choice']}" for d in decisions))

    log(f"references for {len(scen)} scenarios")
    refs = []
    for sc in scen:
        f, p = objs(sc)
        pts = (2**18 if f.m <= 24 else 2**16) if not quick else 2**12
        t0 = time.perf_counter()
        r = reference_estimates(f, p, make_generator([seed, 1, sc["sid"]]), pts, 32 if not quick else 16,
                                check_paths=2**20 if not quick else 2**16)
        refs.append({"sid": sc["sid"], "ref_points": pts, "ref_seconds": time.perf_counter() - t0, **r})
    pd.DataFrame(refs).to_csv(out / "references.csv", index=False)

    log("warm-cache timed runs")
    rows, setup = [], []
    for b_i, B in enumerate(budgets):
        for sc in scen:
            f, p = objs(sc)
            dt = p.T / f.m
            setup.append(dict(sid=sc["sid"], m=f.m, budget=B, warm_build_seconds=warm_cache(CONSTRUCTIONS, f.m, dt)))
            for g_i, g in enumerate(CFG["greeks"]):
                for s_i, s in enumerate(SAMPLERS):
                    pps = pps_for(B, cal, s, f.m)
                    for r in range(reps):
                        e = timed_estimate(g, f, p, B, make_generator([seed, 2, b_i, sc["sid"], g_i, s_i, r]), s,
                                           points_per_scramble=pps)
                        rows.append(dict(sid=sc["sid"], budget=B, greek=g, strategy=s, rep=r, value=e.value,
                                         stderr=e.stderr, ci=e.ci_halfwidth, wall=e.wall, n_paths=e.n_paths,
                                         units=e.n_units, pps=pps, cache="warm"))
        log(f"warm: budget {B * 1e3:.0f} ms done")
    pd.DataFrame(setup).to_csv(out / "setup_costs.csv", index=False)

    log("cold-start runs (matrix construction inside the timed region)")
    cold_sc = {}
    for sc in scen:
        cold_sc.setdefault(sc["m"], sc)  # first scenario per m
    for b_i, B in enumerate(budgets):
        for sc in cold_sc.values():
            f, p = objs(sc)
            for s_i, s in enumerate(SAMPLERS):
                pps = pps_for(B, cal, s, f.m)
                for r in range(min(reps, 10)):
                    e = timed_estimate("delta", f, p, B, make_generator([seed, 5, b_i, sc["sid"], s_i, r]), s,
                                       points_per_scramble=pps, cache={})
                    rows.append(dict(sid=sc["sid"], budget=B, greek="delta", strategy=s, rep=r, value=e.value,
                                     stderr=e.stderr, ci=e.ci_halfwidth, wall=e.wall, n_paths=e.n_paths,
                                     units=e.n_units, pps=pps, cache="cold"))
    pd.DataFrame(rows).to_csv(out / "runs_raw.csv.gz", index=False)
    meta["wall_seconds_total"] = None
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))


def boot_logratio(e_mc, e_pca, wall_mc, wall_pca, rng, n=2000):
    """95% bootstrap CI of log(wn_MC / wn_PCA) resampling replications independently."""
    a, b = np.asarray(e_mc) ** 2, np.asarray(e_pca) ** 2
    ia = rng.integers(0, len(a), (n, len(a)))
    ib = rng.integers(0, len(b), (n, len(b)))
    lr = np.log(a[ia].mean(1) * wall_mc) - np.log(b[ib].mean(1) * wall_pca)
    return np.percentile(lr, 2.5), np.percentile(lr, 97.5)


def gm(x):
    return float(np.exp(np.mean(np.log(np.asarray(x, float)))))


def analyse(out):
    scen = pd.read_csv(out / "scenarios.csv")
    refs = pd.read_csv(out / "references.csv")
    raw = pd.read_csv(out / "runs_raw.csv.gz")
    cal = pd.read_csv(out / "calibration.csv").set_index("m")
    dec = pd.read_csv(out / "rule_decisions.csv")
    rng = np.random.default_rng(CFG["seed"] + 99)
    truth = refs.melt(id_vars="sid", value_vars=[f"ref_{g}" for g in CFG["greeks"]], var_name="k", value_name="truth")
    truth["greek"] = truth.k.str[4:]
    rse = refs.melt(id_vars="sid", value_vars=[f"ref_se_{g}" for g in CFG["greeks"]], var_name="k", value_name="ref_se")
    rse["greek"] = rse.k.str[7:]
    x = raw.merge(truth[["sid", "greek", "truth"]], on=["sid", "greek"]).merge(rse[["sid", "greek", "ref_se"]], on=["sid", "greek"])
    x = x.merge(scen[["sid", "m"]], on="sid")
    x["err"] = x.value - x.truth
    x["over"] = x.wall / x.budget

    # overruns (all runs)
    ov = x.groupby(["cache", "budget", "m", "strategy"]).over.agg(
        median="median", p95=lambda v: v.quantile(0.95), max="max", share_over_1_10=lambda v: (v > 1.10).mean()).reset_index()
    ov.to_csv(out / "overruns.csv", index=False)

    w = x[x.cache == "warm"]
    s = w.groupby(["sid", "m", "budget", "greek", "strategy"]).agg(
        mse=("err", lambda e: float((e**2).mean())), ref_se=("ref_se", "first"), wall=("wall", "mean"),
        coverage=("ci", "size"), n=("err", "size")).reset_index()
    cov = w.assign(c=(w.err.abs() <= w.ci)).groupby(["sid", "budget", "greek", "strategy"]).c.mean().rename("coverage")
    s = s.drop(columns="coverage").merge(cov.reset_index(), on=["sid", "budget", "greek", "strategy"])
    s["mse_c"] = np.maximum(s.mse - s.ref_se**2, 1e-300)
    s["wn"] = s.mse_c * s.wall
    s["ref_share"] = s.ref_se**2 / s.mse
    key = ["sid", "budget", "greek"]
    orc = s.loc[s.groupby(key).wn.idxmin(), key + ["strategy", "wn"]].rename(columns={"strategy": "oracle", "wn": "wn_oracle"})
    s = s.merge(orc, on=key)
    s["regret"] = np.sqrt(s.wn / s.wn_oracle)
    s.to_csv(out / "summary.csv", index=False)

    # strategies
    s = s.merge(dec, on=["m", "budget"])
    rule = s[s.strategy == s.rule_choice].assign(policy="frozen cost-model rule")
    heur = s[s.strategy == np.where(s.m <= 12, "RQMC-pca", "MC")].assign(policy="frequency heuristic (PCA iff m<=12)")
    pca = s[s.strategy == "RQMC-pca"].assign(policy="best fixed from training (always PCA)")
    mc = s[s.strategy == "MC"].assign(policy="always MC")
    seq = s[s.strategy == "RQMC-seq"].assign(policy="always RQMC-seq")
    bb = s[s.strategy == "RQMC-bb"].assign(policy="always RQMC-bb")
    pol = pd.concat([rule, heur, pca, mc, seq, bb])
    reg = pol.groupby(["policy", "budget", "greek"]).regret.agg(gm_regret=gm, worst="max", n="size").reset_index()
    boots = []
    for (pname, B, g), d in pol.groupby(["policy", "budget", "greek"]):
        lr = np.log(d.regret.to_numpy())
        bs = [lr[rng.integers(0, len(lr), len(lr))].mean() for _ in range(1000)]
        boots.append(dict(policy=pname, budget=B, greek=g, ci_lo=math.exp(np.percentile(bs, 2.5)), ci_hi=math.exp(np.percentile(bs, 97.5))))
    reg = reg.merge(pd.DataFrame(boots), on=["policy", "budget", "greek"])
    reg.to_csv(out / "policy_regret.csv", index=False)

    # C2: significant MC-vs-PCA cells
    rows = []
    for (sid, B, g), d in w[w.strategy.isin(["MC", "RQMC-pca"])].groupby(["sid", "budget", "greek"]):
        dm, dp = d[d.strategy == "MC"], d[d.strategy == "RQMC-pca"]
        lo, hi = boot_logratio(dm.err, dp.err, dm.wall.mean(), dp.wall.mean(), rng)
        m = int(scen.set_index("sid").loc[sid, "m"])
        choice = dec[(dec.m == m) & (dec.budget == B)].rule_choice.iloc[0]
        better = "RQMC-pca" if lo > 0 else "MC" if hi < 0 else "ambiguous"
        rows.append(dict(sid=sid, m=m, budget=B, greek=g, lr_lo=lo, lr_hi=hi, better=better, rule=choice,
                         correct=(better == choice) if better != "ambiguous" else np.nan))
    c2 = pd.DataFrame(rows)
    c2.to_csv(out / "decision_accuracy.csv", index=False)

    # crossover budgets
    ratio = s[s.strategy.isin(["MC", "RQMC-pca"])].pivot_table(index=["sid", "m", "greek", "budget"], columns="strategy", values="wn")
    ratio["lr"] = np.log(ratio["MC"] / ratio["RQMC-pca"])
    xs = []
    for (sid, m, g), d in ratio.reset_index().groupby(["sid", "m", "greek"]):
        d = d.sort_values("budget")
        b, y = np.log(d.budget.to_numpy()), d.lr.to_numpy()
        cross = np.nan
        flag = "PCA better at all budgets" if (y > 0).all() else "MC better at all budgets" if (y < 0).all() else "crossover"
        for i in range(len(y) - 1):
            if y[i] <= 0 < y[i + 1]:
                cross = math.exp(b[i] + (0 - y[i]) * (b[i + 1] - b[i]) / (y[i + 1] - y[i]))
                break
        xs.append(dict(sid=sid, m=m, greek=g, crossover_s=cross, status=flag, rule_threshold_s=cal.loc[m, "rule_threshold_s"]))
    xo = pd.DataFrame(xs)
    xo.to_csv(out / "crossover.csv", index=False)

    # cold vs warm (delta, one scenario per m)
    cw = x.groupby(["cache", "sid", "m", "budget", "strategy"]).filter(lambda d: True)
    cold_sids = x[x.cache == "cold"].sid.unique()
    cw = x[(x.greek == "delta") & x.sid.isin(cold_sids)].groupby(["cache", "m", "budget", "strategy"]).agg(
        mse=("err", lambda e: float((e**2).mean())), wall=("wall", "mean"), units=("units", "mean"),
        over_p95=("over", lambda v: v.quantile(0.95))).reset_index()
    cw.to_csv(out / "cold_vs_warm.csv", index=False)

    # verdict
    rr = reg[reg.policy == "frozen cost-model rule"]
    c1 = bool((rr.gm_regret <= 1.25).all())
    sig = c2.dropna(subset=["correct"])
    c2_share = float(sig.correct.mean()) if len(sig) else float("nan")
    c2_ok = bool(c2_share >= 0.90)
    piv = reg.pivot_table(index=["budget", "greek"], columns="policy", values="gm_regret")
    c3_share = float(((piv["frozen cost-model rule"] <= piv["frequency heuristic (PCA iff m<=12)"] + 1e-12) &
                      (piv["frozen cost-model rule"] <= piv["best fixed from training (always PCA)"] + 1e-12)).mean())
    verdict = {"C1_all_cells_gm_regret_le_1.25": c1, "C1_max_gm_regret": float(rr.gm_regret.max()),
               "C2_share_correct_among_significant": c2_share, "C2_n_significant": int(len(sig)),
               "C2_n_ambiguous": int(c2.correct.isna().sum()), "C2_pass": c2_ok,
               "C3_share_cells_no_worse_than_both": c3_share, "C3_pass": bool(c3_share >= 2 / 3),
               "VERDICT": "PASS" if (c1 and c2_ok) else "FAIL"}
    (out / "verdict.json").write_text(json.dumps(verdict, indent=2))
    log(json.dumps(verdict))
    figures(out, s, reg, xo, cw, ov)


def figures(out, s, reg, xo, cw, ov):
    fd = out / "figures"
    fd.mkdir(exist_ok=True)
    greeks = CFG["greeks"]
    ms = sorted(s.m.unique())
    # F1: MSE vs measured time per m (delta, gamma, vega), medians over scenarios
    fig, axes = plt.subplots(len(greeks), len(ms), figsize=(3.2 * len(ms), 8), squeeze=False)
    for i, g in enumerate(greeks):
        for j, m in enumerate(ms):
            ax = axes[i, j]
            d = s[(s.greek == g) & (s.m == m)]
            for st in COL:
                dd = d[d.strategy == st].groupby("budget").agg(wall=("wall", "median"), mse=("mse_c", "median")).reset_index()
                ax.loglog(dd.wall * 1e3, dd.mse, marker="o", color=COL[st], label=st)
            ax.set_title(f"{g}, m={m}", fontsize=9)
            if i == len(greeks) - 1:
                ax.set_xlabel("measured wall time [ms]")
            if j == 0:
                ax.set_ylabel("MSE (median over 8 scenarios)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Phase 4 validation grid: MSE against measured computational time", y=1.0)
    fig.tight_layout()
    fig.savefig(fd / "fig4B1_mse_vs_time.png", bbox_inches="tight"); fig.savefig(fd / "fig4B1_mse_vs_time.pdf", bbox_inches="tight")
    plt.close(fig)
    # F2: MC/PCA wn ratio across budgets + rule threshold (crossover)
    fig, axes = plt.subplots(1, len(greeks), figsize=(11, 3.6), sharey=True)
    r = s[s.strategy.isin(["MC", "RQMC-pca"])].pivot_table(index=["sid", "m", "greek", "budget"], columns="strategy", values="wn").reset_index()
    r["ratio"] = r["MC"] / r["RQMC-pca"]
    cal = pd.read_csv(out / "calibration.csv").set_index("m")
    cols = dict(zip(ms, ["#2a78d6", "#1baf7a", "#eda100", "#e34948"]))
    for ax, g in zip(axes, greeks):
        for m in ms:
            d = r[(r.greek == g) & (r.m == m)].groupby("budget").ratio.apply(gm)
            ax.loglog(d.index * 1e3, d.values, marker="o", color=cols[m], label=f"m={m}")
            ax.axvline(cal.loc[m, "rule_threshold_s"] * 1e3, color=cols[m], ls=":", lw=1)
        ax.axhline(1, color="#1f1f1e", lw=0.9)
        ax.set_title(g)
        ax.set_xlabel("budget [ms] (dotted: frozen rule threshold)")
    axes[0].set_ylabel("wnMSE(MC) / wnMSE(PCA-RQMC)\n(>1: PCA better; geo-mean over scenarios)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fd / "fig4B2_mc_over_pca_ratio.png", bbox_inches="tight"); fig.savefig(fd / "fig4B2_mc_over_pca_ratio.pdf", bbox_inches="tight")
    plt.close(fig)
    # F3: crossover by m
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for k, g in enumerate(greeks):
        d = xo[(xo.greek == g) & xo.crossover_s.notna()]
        ax.scatter(np.log2(d.m) + (k - 1) * 0.08, d.crossover_s * 1e3, s=18, label=f"{g} (empirical, per scenario)")
    th = xo.drop_duplicates("m")
    ax.plot(np.log2(th.m), th.rule_threshold_s * 1e3, "k--", marker="s", label="frozen rule threshold")
    ax.set_yscale("log")
    ax.set_xticks(np.log2(ms), [str(m) for m in ms])
    ax.set_xlabel("monitoring dates m")
    ax.set_ylabel("crossover budget [ms]")
    ax.set_title("Budget where PCA-RQMC overtakes MC")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fd / "fig4B3_crossover_by_m.png", bbox_inches="tight"); fig.savefig(fd / "fig4B3_crossover_by_m.pdf", bbox_inches="tight")
    plt.close(fig)
    # F4: regret of policies
    pols = ["frozen cost-model rule", "frequency heuristic (PCA iff m<=12)", "best fixed from training (always PCA)", "always MC"]
    fig, axes = plt.subplots(1, len(greeks), figsize=(12, 3.6), sharey=True)
    for ax, g in zip(axes, greeks):
        for k, pn in enumerate(pols):
            d = reg[(reg.policy == pn) & (reg.greek == g)].sort_values("budget")
            xx = np.arange(len(d)) + (k - 1.5) * 0.15
            ax.errorbar(xx, d.gm_regret, yerr=[d.gm_regret - d.ci_lo, d.ci_hi - d.gm_regret], marker="o", capsize=2, label=pn)
        ax.set_yscale("log")
        ax.axhline(1.25, color="#e34948", ls=":", lw=1)
        ax.set_xticks(range(len(d)), [f"{b * 1e3:.0f}" for b in d.budget])
        ax.set_xlabel("budget [ms]")
        ax.set_title(g)
    axes[0].set_ylabel("regret vs oracle (geo-mean, 95% CI)\n(red dotted: C1 threshold 1.25)")
    axes[0].legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(fd / "fig4B4_policy_regret.png", bbox_inches="tight"); fig.savefig(fd / "fig4B4_policy_regret.pdf", bbox_inches="tight")
    plt.close(fig)
    # F5: cold vs warm
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    p = cw.pivot_table(index=["m", "budget", "strategy"], columns="cache", values=["wall", "mse"]).reset_index()
    for st in ["RQMC-bb", "RQMC-pca"]:
        d = p[p.strategy == st]
        axes[0].scatter(d[("wall", "warm")] * 1e3, d[("wall", "cold")] * 1e3, color=COL[st], label=st, s=18)
        axes[1].scatter(d.budget * 1e3 + 0 * d.m, d[("mse", "cold")] / d[("mse", "warm")], color=COL[st], s=18, label=st)
    lim = [1, 300]
    axes[0].plot(lim, lim, "k:", lw=1)
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("warm wall [ms]"); axes[0].set_ylabel("cold wall [ms]")
    axes[1].set_xscale("log"); axes[1].set_yscale("log"); axes[1].axhline(1, color="k", lw=0.8)
    axes[1].set_xlabel("budget [ms]"); axes[1].set_ylabel("MSE cold / MSE warm (Delta)")
    axes[0].legend(fontsize=7)
    fig.suptitle("Cold start (matrix built inside the timed region) vs warm cache", y=1.02)
    fig.tight_layout()
    fig.savefig(fd / "fig4B5_cold_vs_warm.png", bbox_inches="tight"); fig.savefig(fd / "fig4B5_cold_vs_warm.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--analysis-only", action="store_true")
    a = ap.parse_args()
    out = RESULTS / "phase4_cost_rule" / ("quick" if a.quick else "confirmatory")
    t0 = time.time()
    if not a.analysis_only:
        simulate(out, a.quick)
        meta = json.loads((out / "meta.json").read_text())
        meta["wall_seconds_total"] = time.time() - t0
        (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    analyse(out)

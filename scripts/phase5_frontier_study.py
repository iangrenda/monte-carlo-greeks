"""Phase 5A: dense-grid stress test of the frozen Phase-4 cost-model rule near the crossover.

    python scripts/phase5_frontier_study.py                    # primary ("quiet"), ~35 min
    python scripts/phase5_frontier_study.py --config contended # 8 background busy processes, ~15 min
    python scripts/phase5_frontier_study.py --quick            # pipeline check only (NOT evidence)
    python scripts/phase5_frontier_study.py --analysis-only [--config ...]

Protocol: configs/phase5_frontier.json, committed before the first run. The rule
is the unmodified rule of configs/phase4_cost_rule.json; nothing here tunes it.
Output: results/phase5_frontier/<config>/.
"""
import argparse
import json
import math
import platform
import subprocess
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mc_greeks import visualization as viz  # noqa: F401
from mc_greeks.asian import AsianDigital
from mc_greeks.asian_qmc import calibrate_rqmc, pps_for, reference_estimates, timed_estimate, warm_cache
from mc_greeks.config import RESULTS, ROOT
from mc_greeks.experiments import environment_info
from mc_greeks.models import BlackScholesParams
from mc_greeks.simulation import make_generator

CFG = json.loads((ROOT / "configs" / "phase5_frontier.json").read_text())
RULE = CFG["rule_under_test"]
SAMPLERS = CFG["samplers"]
GREEKS = CFG["greeks"]
NBOOT = 2000


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def provenance(config_file) -> dict:
    def git(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception as exc:  # pragma: no cover
            return f"unavailable ({exc})"
    import os
    return {**environment_info(), "git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
            "machine": platform.machine(), "logical_cpus": os.cpu_count(), "config_file": config_file,
            "config_last_commit": git("log", "-1", "--format=%H", "--", config_file)}


def scenarios(quick=False):
    g = CFG["grid"]
    out, i = [], 0
    for m in sorted(int(k) for k in g["budgets_ms_by_m"]):
        for k in g["moneyness_K_over_S0"]:
            for s in g["sigma"]:
                for T in g["T"]:
                    out.append(dict(sid=i, m=m, moneyness=k, sigma=s, T=T, r=g["r"], S0=g["S0"]))
                    i += 1
    if quick:
        seen, keep = set(), []
        for sc in out:
            if sc["m"] not in seen:
                seen.add(sc["m"])
                keep.append(sc)
        out = keep
    return out


def budgets_for(m, quick=False):
    b = [x / 1000.0 for x in CFG["grid"]["budgets_ms_by_m"][str(m)]]
    return b[:2] if quick else b


def objs(sc):
    return (AsianDigital(sc["moneyness"] * sc["S0"], int(sc["m"])),
            BlackScholesParams(sc["S0"], sc["r"], sc["sigma"], sc["T"]))


def threshold(cal, m):
    a, b = cal[("pca", m)]
    return RULE["n_scrambles"] * (a + RULE["n_min_points"] * b)


def start_load(n):
    return [subprocess.Popen([sys.executable, "-c", "while True: pass"]) for _ in range(n)]


def simulate(out, config, quick):
    out.mkdir(parents=True, exist_ok=True)
    seed = CFG["seed"]
    c = CFG["configurations"][config]
    reps = 4 if quick else c["replications"]
    scen = scenarios(quick)
    pd.DataFrame(scen).to_csv(out / "scenarios.csv", index=False)
    ms = sorted({s["m"] for s in scen})
    c_i = list(CFG["configurations"]).index(config)
    meta = {"provenance": provenance("configs/phase5_frontier.json"), "configuration": config, "quick": quick,
            "reps": reps, "background_load_processes": c["background_load_processes"]}

    log(f"references for {len(scen)} scenarios (no background load)")
    refs = []
    for sc in scen:
        f, p = objs(sc)
        pts = (2**18 if f.m <= 24 else 2**16) if not quick else 2**12
        t0 = time.perf_counter()
        r = reference_estimates(f, p, make_generator([seed, 1, sc["sid"]]), pts, 32 if not quick else 16,
                                check_paths=2**20 if not quick else 2**16)
        refs.append({"sid": sc["sid"], "ref_points": pts, "ref_seconds": time.perf_counter() - t0, **r})
    pd.DataFrame(refs).to_csv(out / "references.csv", index=False)

    load = start_load(c["background_load_processes"])
    try:
        if load:
            time.sleep(3.0)
        log("primary pooled calibration")
        t0 = time.perf_counter()
        cal = calibrate_rqmc(tuple(ms), seed=seed + RULE["calibration_seed_offset"] + 1000 * c_i, pooled=True)
        meta["calibration_seconds"] = time.perf_counter() - t0
        pd.DataFrame([dict(m=m, a=cal[("pca", m)][0], b=cal[("pca", m)][1], mc_sec_per_path=cal[("mc", m)][1],
                           rule_threshold_s=threshold(cal, m)) for m in ms]).to_csv(out / "calibration.csv", index=False)
        dec = [dict(m=m, budget=B, rule_choice="RQMC-pca" if B >= threshold(cal, m) else "MC")
               for m in ms for B in budgets_for(m, quick)]
        pd.DataFrame(dec).to_csv(out / "rule_decisions.csv", index=False)
        log("thresholds [ms]: " + ", ".join(f"m={m}: {threshold(cal, m) * 1e3:.2f}" for m in ms))

        def recalibrate(tag, n):
            rows = []
            for k in range(n):
                ck = calibrate_rqmc(tuple(ms), seed=seed + 50_000 + 1000 * c_i + 100 * tag + k, pooled=True)
                rows += [dict(when="before" if tag == 0 else "after", k=k, m=m, a=ck[("pca", m)][0], b=ck[("pca", m)][1],
                              rule_threshold_s=threshold(ck, m)) for m in ms]
            return rows

        n_before = 3 if quick else CFG["calibration_noise"]["repeats"]
        log(f"{n_before} repeat calibrations (noise study)")
        recal = recalibrate(0, n_before)

        log("timed runs")
        rows = []
        for m in ms:
            for b_i, B in enumerate(budgets_for(m, quick)):
                for sc in [s for s in scen if s["m"] == m]:
                    f, p = objs(sc)
                    warm_cache(("pca",), f.m, p.T / f.m)
                    pps = {s: pps_for(B, cal, s, f.m) for s in SAMPLERS}
                    for g_i, g in enumerate(GREEKS):
                        for r in range(reps):
                            for s_i, s in enumerate(SAMPLERS):  # back to back
                                e = timed_estimate(g, f, p, B, make_generator([seed, 2, c_i, b_i, sc["sid"], g_i, s_i, r]),
                                                   s, points_per_scramble=pps[s])
                                rows.append(dict(sid=sc["sid"], m=m, budget=B, greek=g, strategy=s, rep=r, value=e.value,
                                                 stderr=e.stderr, ci=e.ci_halfwidth, wall=e.wall, n_paths=e.n_paths,
                                                 units=e.n_units, pps=pps[s]))
                log(f"m={m} budget {B * 1e3:.0f} ms done")
        pd.DataFrame(rows).to_csv(out / "runs_raw.csv.gz", index=False)
        recal += recalibrate(1, 2 if quick else 5)
        pd.DataFrame(recal).to_csv(out / "calibration_repeats.csv", index=False)
    finally:
        for pr in load:
            pr.kill()
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))


def crossing(budgets, y):
    """First budget where y goes from <= 0 to > 0, linear in log budget; nan if none."""
    b = np.log(budgets)
    for i in range(len(y) - 1):
        if y[i] <= 0 < y[i + 1]:
            return math.exp(b[i] + (0 - y[i]) * (b[i + 1] - b[i]) / (y[i + 1] - y[i]))
    return float("nan")


def analyse(out):
    scen = pd.read_csv(out / "scenarios.csv")
    refs = pd.read_csv(out / "references.csv").set_index("sid")
    raw = pd.read_csv(out / "runs_raw.csv.gz")
    cal = pd.read_csv(out / "calibration.csv").set_index("m")
    dec = pd.read_csv(out / "rule_decisions.csv").set_index(["m", "budget"]).rule_choice
    recal = pd.read_csv(out / "calibration_repeats.csv")
    rng = np.random.default_rng(CFG["seed"] + 99)
    raw["truth"] = [refs.at[s, f"ref_{g}"] for s, g in zip(raw.sid, raw.greek)]
    raw["ref_se"] = [refs.at[s, f"ref_se_{g}"] for s, g in zip(raw.sid, raw.greek)]
    raw["err"] = raw.value - raw.truth
    raw["over"] = raw.wall / raw.budget
    raw["covered"] = raw.err.abs() <= raw.ci

    ov = raw.groupby(["m", "budget", "strategy"]).agg(
        median=("over", "median"), p95=("over", lambda v: v.quantile(0.95)), max=("over", "max"),
        share_over_1_10=("over", lambda v: (v > 1.10).mean()), units_median=("units", "median"),
        pps=("pps", "first"), coverage=("covered", "mean")).reset_index()
    ov.to_csv(out / "overruns.csv", index=False)

    s = raw.groupby(["sid", "m", "budget", "greek", "strategy"]).agg(
        mse=("err", lambda e: float((e**2).mean())), ref_se=("ref_se", "first"), wall=("wall", "mean"),
        coverage=("covered", "mean"), n=("err", "size")).reset_index()
    s["mse_c"] = np.maximum(s.mse - s.ref_se**2, 1e-300)
    s["wn"] = s.mse_c * s.wall
    s["ref_share"] = s.ref_se**2 / s.mse
    s.to_csv(out / "summary.csv", index=False)

    # pooled log ratios with bootstrap, per (m, greek) across the budget grid
    cells, cross, boot_lr = [], [], {}
    for (m, g), d in raw.groupby(["m", "greek"]):
        budgets = np.sort(d.budget.unique())
        sids = np.sort(d.sid.unique())
        lr = np.zeros(len(budgets))
        lrb = np.zeros((NBOOT, len(budgets)))
        for j, B in enumerate(budgets):
            for sid in sids:
                part = {}
                for st in SAMPLERS:
                    dd = d[(d.budget == B) & (d.sid == sid) & (d.strategy == st)]
                    e2, w, rse2 = dd.err.to_numpy() ** 2, dd.wall.to_numpy(), float(dd.ref_se.iloc[0]) ** 2
                    idx = rng.integers(0, len(e2), (NBOOT, len(e2)))
                    part[st] = (math.log(max(e2.mean() - rse2, 1e-300) * w.mean()),
                                np.log(np.maximum(e2[idx].mean(1) - rse2, 1e-300) * w[idx].mean(1)))
                lr[j] += (part["MC"][0] - part["RQMC-pca"][0]) / len(sids)
                lrb[:, j] += (part["MC"][1] - part["RQMC-pca"][1]) / len(sids)
        boot_lr[(m, g)] = (budgets, lr, lrb)
        for j, B in enumerate(budgets):
            choice = dec.loc[(m, B)]
            sign = 0.5 if choice == "MC" else -0.5
            rho, rb = math.exp(sign * lr[j]), np.exp(sign * lrb[:, j])
            lo, hi = np.percentile(rb, [2.5, 97.5])
            cls = "correct" if hi < 1 else "wrong_material" if lo > 1.10 else "wrong_negligible" if lo > 1 else "indistinguishable"
            cells.append(dict(m=m, greek=g, budget=B, rule_choice=choice, lr=lr[j], lr_lo=np.percentile(lrb[:, j], 2.5),
                              lr_hi=np.percentile(lrb[:, j], 97.5), rho=rho, rho_lo=lo, rho_hi=hi, cls=cls))
        xb = np.array([crossing(budgets, y) for y in lrb])
        cross.append(dict(m=m, greek=g, crossover_s=crossing(budgets, lr), ci_lo=np.nanpercentile(xb, 2.5) if np.isfinite(xb).any() else np.nan,
                          ci_hi=np.nanpercentile(xb, 97.5) if np.isfinite(xb).any() else np.nan,
                          share_boot_with_crossing=float(np.isfinite(xb).mean()),
                          status=("PCA better on whole grid" if (lr > 0).all() else "MC better on whole grid" if (lr < 0).all() else "crossover"),
                          rule_threshold_s=cal.loc[m, "rule_threshold_s"]))
    cells = pd.DataFrame(cells)
    cells.to_csv(out / "decision_cells.csv", index=False)
    cross = pd.DataFrame(cross)
    cross.to_csv(out / "crossover.csv", index=False)

    # per-scenario decisions (finer view; same bootstrap construction as Phase 4 C2)
    rows = []
    for (sid, m, B, g), d in raw.groupby(["sid", "m", "budget", "greek"]):
        dm, dp = d[d.strategy == "MC"], d[d.strategy == "RQMC-pca"]
        a, b = dm.err.to_numpy() ** 2, dp.err.to_numpy() ** 2
        ia, ib = rng.integers(0, len(a), (NBOOT, len(a))), rng.integers(0, len(b), (NBOOT, len(b)))
        lb = np.log(np.maximum(a[ia].mean(1), 1e-300) * dm.wall.mean()) - np.log(np.maximum(b[ib].mean(1), 1e-300) * dp.wall.mean())
        lo, hi = np.percentile(lb, [2.5, 97.5])
        better = "RQMC-pca" if lo > 0 else "MC" if hi < 0 else "ambiguous"
        choice = dec.loc[(m, B)]
        point = math.log(a.mean() * dm.wall.mean()) - math.log(b.mean() * dp.wall.mean())
        rows.append(dict(sid=sid, m=m, budget=B, greek=g, rule=choice, better=better, lr=point,
                         wrong_significant=(better != "ambiguous" and better != choice),
                         rho=math.exp((0.5 if choice == "MC" else -0.5) * point)))
    per_sc = pd.DataFrame(rows)
    per_sc.to_csv(out / "decision_per_scenario.csv", index=False)

    # equal measured runtime
    eq = []
    for (sid, m, g), d in s.groupby(["sid", "m", "greek"]):
        pm, pp = (d[d.strategy == st].sort_values("wall") for st in SAMPLERS)
        lo, hi = max(pm.wall.min(), pp.wall.min()), min(pm.wall.max(), pp.wall.max())
        for B in np.sort(d.budget.unique()):
            if lo <= B <= hi:
                lm = np.interp(math.log(B), np.log(pm.wall), np.log(pm.mse_c))
                lp = np.interp(math.log(B), np.log(pp.wall), np.log(pp.mse_c))
                eq.append(dict(sid=sid, m=m, greek=g, wall=B, log_mse_mc_over_pca=lm - lp))
    eq = pd.DataFrame(eq)
    eqs = eq.groupby(["m", "greek", "wall"]).log_mse_mc_over_pca.agg(
        gm_ratio=lambda v: math.exp(v.mean()), n="size").reset_index() if len(eq) else pd.DataFrame()
    eqs.to_csv(out / "equal_runtime_mse.csv", index=False)

    # calibration noise
    cn = []
    for (when, k), d in recal.groupby(["when", "k"]):
        th = d.set_index("m").rule_threshold_s
        alt = np.array([("RQMC-pca" if r.budget >= th[r.m] else "MC") for r in cells.itertuples()])
        flip = alt != cells.rule_choice.to_numpy()
        rho_alt = np.exp(np.where(alt == "MC", 0.5, -0.5) * cells.lr.to_numpy())
        cn.append(dict(when=when, k=k, flips=int(flip.sum()), cells=len(cells), gm_rho=float(np.exp(np.log(rho_alt).mean())),
                       max_rho=float(rho_alt.max()), **{f"threshold_ms_m{m}": th[m] * 1e3 for m in th.index}))
    cn = pd.DataFrame(cn)
    cn.to_csv(out / "calibration_noise.csv", index=False)
    th_stats = recal.groupby("m").rule_threshold_s.agg(["mean", "std", "min", "max"]).assign(primary=cal.rule_threshold_s)
    th_stats["cv"] = th_stats["std"] / th_stats["mean"]
    th_stats.to_csv(out / "threshold_noise.csv")

    # comparators on the same data (rho relative to the better-on-average of the two is not used; report gm wn regret vs pair-oracle too)
    pol = []
    for name in ("frozen rule", "always MC", "always RQMC-pca"):
        ch = cells.rule_choice if name == "frozen rule" else pd.Series("MC" if name == "always MC" else "RQMC-pca", index=cells.index)
        rho = np.exp(np.where(ch == "MC", 0.5, -0.5) * cells.lr)
        loss_vs_best = np.maximum(rho, 1.0)  # pooled-cell regret; mildly optimistic near ties
        for (m, g), idx in cells.groupby(["m", "greek"]).groups.items():
            pol.append(dict(policy=name, m=m, greek=g, gm_rho=float(np.exp(np.log(rho[idx]).mean())),
                            gm_regret_vs_pair_best=float(np.exp(np.log(loss_vs_best[idx]).mean())), worst_rho=float(rho[idx].max())))
    pol = pd.DataFrame(pol)
    pol.to_csv(out / "policy_summary.csv", index=False)

    rule = pol[pol.policy == "frozen rule"]
    wrong = cells.cls.isin(["wrong_material", "wrong_negligible"])
    gap = cells.apply(lambda r: (r.rule_choice == "RQMC-pca") and (r.lr < 0), axis=1)
    sig = per_sc[per_sc.better != "ambiguous"]
    verdict = {
        "D1_no_wrong_material_cell": bool((cells.cls != "wrong_material").all()),
        "D1_n_wrong_material": int((cells.cls == "wrong_material").sum()),
        "D2_gm_rho_le_1.05_every_m_greek": bool((rule.gm_rho <= 1.05).all()),
        "D2_max_gm_rho": float(rule.gm_rho.max()),
        "D3_share_cells_wrong": float(wrong.mean()), "n_cells": int(len(cells)),
        "class_counts": cells.cls.value_counts().to_dict(),
        "cells_rule_pca_but_point_estimate_favours_mc": int(gap.sum()),
        "worst_cell_rho": float(cells.rho.max()),
        "per_scenario_significant": int(len(sig)), "per_scenario_wrong_significant": int(sig.wrong_significant.sum()),
        "per_scenario_worst_rho_when_wrong": float(sig[sig.wrong_significant].rho.max()) if sig.wrong_significant.any() else None,
        "max_abs_reference_check_z": float(np.abs(refs[[f"check_z_{g}" for g in GREEKS]].to_numpy()).max()),
        "max_ref_share_of_mse": float(s.ref_share.max()),
    }
    verdict["VERDICT"] = "ROBUST" if verdict["D1_no_wrong_material_cell"] and verdict["D2_gm_rho_le_1.05_every_m_greek"] else "FRAGILE NEAR THE FRONTIER"
    (out / "verdict.json").write_text(json.dumps(verdict, indent=2))
    log(json.dumps(verdict))
    figures(out, boot_lr, cells, cross, recal, cal, ov)


def figures(out, boot_lr, cells, cross, recal, cal, ov):
    fd = out / "figures"
    fd.mkdir(exist_ok=True)
    ms = sorted({k[0] for k in boot_lr})
    col = {"delta": "#2a78d6", "gamma": "#e34948", "vega": "#1baf7a"}
    fig, axes = plt.subplots(1, len(ms), figsize=(4.3 * len(ms), 3.8), squeeze=False)
    for ax, m in zip(axes[0], ms):
        for g in GREEKS:
            b, lr, lrb = boot_lr[(m, g)]
            lo, hi = np.percentile(lrb, [2.5, 97.5], axis=0)
            ax.plot(b * 1e3, np.exp(lr), marker="o", ms=3.5, color=col[g], label=g)
            ax.fill_between(b * 1e3, np.exp(lo), np.exp(hi), color=col[g], alpha=0.15, lw=0)
        th = recal[recal.m == m].rule_threshold_s * 1e3
        ax.axvspan(th.min(), th.max(), color="#9a9890", alpha=0.25, lw=0)
        ax.axvline(cal.loc[m, "rule_threshold_s"] * 1e3, color="#1f1f1e", ls="--", lw=1)
        ax.axhline(1, color="#1f1f1e", lw=0.8)
        ax.set_yscale("log")
        ax.set_title(f"m = {m}")
        ax.set_xlabel("budget [ms]")
    axes[0, 0].set_ylabel("wnMSE(MC) / wnMSE(PCA-RQMC)\n(>1: PCA better; pooled over 8 scenarios, 95% CI)")
    axes[0, 0].legend(fontsize=7, title="dashed: rule threshold\ngrey: recalibration range", title_fontsize=6.5)
    fig.tight_layout()
    fig.savefig(fd / "fig5A1_ratio_dense_grid.png", bbox_inches="tight"); fig.savefig(fd / "fig5A1_ratio_dense_grid.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(ms), figsize=(4.3 * len(ms), 3.6), squeeze=False, sharey=True)
    for ax, m in zip(axes[0], ms):
        for k, g in enumerate(GREEKS):
            d = cells[(cells.m == m) & (cells.greek == g)].sort_values("budget")
            ax.errorbar(d.budget * 1e3 * (1 + 0.012 * (k - 1)), d.rho, yerr=[d.rho - d.rho_lo, d.rho_hi - d.rho], fmt="o", ms=3.5,
                        capsize=2, color=col[g], label=g)
        ax.axhline(1, color="#1f1f1e", lw=0.8)
        ax.axhline(1.10, color="#e34948", ls=":", lw=1)
        ax.axvline(cal.loc[m, "rule_threshold_s"] * 1e3, color="#1f1f1e", ls="--", lw=1)
        ax.set_title(f"m = {m}")
        ax.set_xlabel("budget [ms]")
    axes[0, 0].set_ylabel("decision loss rho of the frozen rule\n(RMSE factor vs the other sampler; <1 = right choice)")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fd / "fig5A2_decision_loss.png", bbox_inches="tight"); fig.savefig(fd / "fig5A2_decision_loss.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for k, g in enumerate(GREEKS):
        d = cross[cross.greek == g]
        axes[0].errorbar(np.log2(d.m) + 0.06 * (k - 1), d.crossover_s * 1e3,
                         yerr=np.nan_to_num(np.clip([(d.crossover_s - d.ci_lo) * 1e3, (d.ci_hi - d.crossover_s) * 1e3], 0, None)),
                         fmt="o", capsize=3, color=col[g],
                         label=f"{g}: empirical crossover (95% CI)")
    t = recal.groupby("m").rule_threshold_s.agg(["min", "max"])
    axes[0].errorbar(np.log2(t.index), cal.loc[t.index, "rule_threshold_s"] * 1e3,
                     yerr=np.clip([(cal.loc[t.index, "rule_threshold_s"] - t["min"]) * 1e3, (t["max"] - cal.loc[t.index, "rule_threshold_s"]) * 1e3], 0, None),
                     fmt="s--", color="#1f1f1e", capsize=3, label="rule threshold (recalibration range)")
    axes[0].set_yscale("log"); axes[0].set_xticks(np.log2(ms), [str(m) for m in ms])
    axes[0].set_xlabel("monitoring dates m"); axes[0].set_ylabel("budget [ms]"); axes[0].legend(fontsize=7)
    for st, c in (("MC", "#eb6834"), ("RQMC-pca", "#2a78d6")):
        d = ov[ov.strategy == st]
        axes[1].scatter(d.budget * 1e3, d.p95, color=c, s=16, label=f"{st}: p95")
        axes[1].scatter(d.budget * 1e3, d["max"], color=c, s=16, marker="x", label=f"{st}: max")
    axes[1].axhline(1.10, color="#e34948", ls=":", lw=1)
    axes[1].set_xscale("log"); axes[1].set_xlabel("budget [ms]"); axes[1].set_ylabel("measured wall / budget"); axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fd / "fig5A3_crossover_and_overruns.png", bbox_inches="tight"); fig.savefig(fd / "fig5A3_crossover_and_overruns.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="quiet", choices=list(CFG["configurations"]))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--analysis-only", action="store_true")
    a = ap.parse_args()
    out = RESULTS / "phase5_frontier" / ("quick" if a.quick else a.config)
    t0 = time.time()
    if not a.analysis_only:
        simulate(out, a.config, a.quick)
        meta = json.loads((out / "meta.json").read_text())
        meta["wall_seconds_total"] = time.time() - t0
        (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    analyse(out)

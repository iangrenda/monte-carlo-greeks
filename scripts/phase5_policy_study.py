"""Phase 5B: sampling-schedule policies under wall-clock budgets.

    python scripts/phase5_policy_study.py            # full run, ~25 min
    python scripts/phase5_policy_study.py --quick    # pipeline check only (NOT evidence)
    python scripts/phase5_policy_study.py --analysis-only

Protocol: configs/phase5_sampling_policy.json (committed before the run).
Output: results/phase5_sampling_policy/<run>/.
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

from mc_greeks import budget_policy as bp
from mc_greeks import visualization as viz  # noqa: F401
from mc_greeks.asian import AsianDigital
from mc_greeks.asian_qmc import calibrate_rqmc, pps_for, reference_estimates, timed_estimate, warm_cache
from mc_greeks.config import RESULTS, ROOT
from mc_greeks.joint_greeks import pca_factor
from mc_greeks.models import BlackScholesParams
from mc_greeks.simulation import make_generator
from phase5_frontier_study import provenance

CFG = json.loads((ROOT / "configs" / "phase5_sampling_policy.json").read_text())
PC = CFG["prior_constants"]
GREEKS = CFG["greeks"]
STRATS = list(CFG["strategies"])
NBOOT = 2000
COL = {"MC-base": "#f0a58a", "MC-look": "#eb6834", "PCA-base": "#9a9890", "PCA-fixed64": "#c9a227", "PCA-fixed512": "#8a6d00",
       "PCA-opt-mse": "#2a78d6", "PCA-opt-ci": "#7fb2ea", "PCA-shift-mse": "#1baf7a", "PCA-shift-ci": "#8fd9bd"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def scenarios(quick=False):
    g = CFG["grid"]
    out, i = [], 0
    for m in g["m"]:
        for k in g["moneyness_K_over_S0"]:
            for s in g["sigma"]:
                out.append(dict(sid=i, m=m, moneyness=k, sigma=s, T=g["T"], r=g["r"], S0=g["S0"]))
                i += 1
    if quick:
        out = [sc for sc in out if sc["moneyness"] == g["moneyness_K_over_S0"][0] and sc["sigma"] == g["sigma"][0]]
    return out


def objs(sc):
    return (AsianDigital(sc["moneyness"] * sc["S0"], int(sc["m"])),
            BlackScholesParams(sc["S0"], sc["r"], sc["sigma"], sc["T"]))


def plan_for(name, B, cm):
    kw = dict(alpha=PC["alpha"], r_min=PC["r_min"], margin=PC["margin"])
    if name == "MC-look":
        return bp.mc_schedule(B, cm, margin=PC["margin"])
    if name.startswith("PCA-fixed"):
        return bp.fixed_n_schedule(B, cm, int(name[9:]), "owen", PC["r_min"], PC["margin"])
    rand = "shift" if "shift" in name else "owen"
    return bp.optimise_schedule(B, cm, rand, objective=name.rsplit("-", 1)[1], n_min=PC["n_min"], **kw)


def run_strategy(name, g, f, p, B, rng, cm, cal4, L):
    """One timed estimate. Returns a row dict (planning is inside the measured wall time)."""
    if name in ("MC-base", "PCA-base"):
        s = "MC" if name == "MC-base" else "RQMC-pca"
        pps = pps_for(B, cal4, s, f.m)
        e = timed_estimate(g, f, p, B, rng, s, points_per_scramble=pps)
        return dict(value=e.value, stderr=e.stderr, ci=e.ci_halfwidth, wall=e.wall, n_paths=e.n_paths, units=e.n_units,
                    n_per_unit=pps or e.n_paths // e.n_units, feasible=True, planned_units=np.nan, pred_cost=np.nan)
    t0 = time.perf_counter()
    plan = plan_for(name, B, cm)
    e = bp.execute(plan, bp.make_evaluator(f, p, g), 1, L, f.m, p.T / f.m, B, rng, t0=t0)
    return dict(value=float(e.value[0]), stderr=float(e.stderr[0]), ci=float(e.ci_halfwidth[0]), wall=e.wall, n_paths=e.n_paths,
                units=e.n_units, n_per_unit=e.n_per_unit, feasible=e.feasible, planned_units=plan.units, pred_cost=plan.predicted_cost)


def simulate(out, quick):
    out.mkdir(parents=True, exist_ok=True)
    seed = CFG["seed"]
    scen = scenarios(quick)
    pd.DataFrame(scen).to_csv(out / "scenarios.csv", index=False)
    ms = sorted({s["m"] for s in scen})
    budgets = CFG["budgets_seconds"][1:3] if quick else CFG["budgets_seconds"]
    reps = 4 if quick else CFG["replications"]
    meta = {"provenance": provenance("configs/phase5_sampling_policy.json"), "quick": quick, "reps": reps, "budgets": budgets}

    log("references")
    refs = []
    for sc in scen:
        f, p = objs(sc)
        pts = (2**18 if f.m <= 24 else 2**16) if not quick else 2**12
        r = reference_estimates(f, p, make_generator([seed, 1, sc["sid"]]), pts, 32 if not quick else 16,
                                check_paths=2**20 if not quick else 2**16)
        refs.append({"sid": sc["sid"], "ref_points": pts, **r})
    pd.DataFrame(refs).to_csv(out / "references.csv", index=False)

    log("calibration")
    t0 = time.perf_counter()
    cal4 = calibrate_rqmc(tuple(ms), seed=seed + 7, pooled=True)
    meta["phase4_calibration_seconds"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    cms, crow = {}, []
    for m in ms:
        for g_i, g in enumerate(GREEKS):
            cm = bp.calibrate_cost_model(m, g, seed=seed + 11 + 100 * m + g_i, repeats=3 if quick else 7)
            cms[(m, g)] = cm
            crow.append(dict(m=m, greek=g, a4=cal4[("pca", m)][0], b4=cal4[("pca", m)][1],
                             c0_owen=cm.rq["owen"][0], a_owen=cm.rq["owen"][1], b_owen=cm.rq["owen"][2],
                             c0_shift=cm.rq["shift"][0], a_shift=cm.rq["shift"][1], b_shift=cm.rq["shift"][2],
                             mc0=cm.mc[0], mc_u=cm.mc[1], mc_c=cm.mc[2], fit_err_owen=cm.fit_error["owen"],
                             fit_err_shift=cm.fit_error["shift"], fit_err_mc=cm.fit_error["mc"],
                             phase4_threshold_s=10 * (cal4[("pca", m)][0] + 64 * cal4[("pca", m)][1]),
                             owen_threshold_s=10 * (cm.rq["owen"][1] + 64 * cm.rq["owen"][2]),
                             shift_threshold_s=10 * (cm.rq["shift"][1] + 64 * cm.rq["shift"][2])))
    meta["new_calibration_seconds"] = time.perf_counter() - t0
    pd.DataFrame(crow).to_csv(out / "calibration.csv", index=False)

    def block(tag, scen_list, greeks, budget_list, strategies, n_rep, label):
        rows = []
        for b_i, B in enumerate(budget_list):
            for sc in scen_list:
                f, p = objs(sc)
                dt = p.T / f.m
                warm_cache(("pca",), f.m, dt)
                L = pca_factor(f.m, dt)
                for g in greeks:
                    g_i = GREEKS.index(g)
                    for r in range(n_rep):
                        for s_i, s in enumerate(strategies):  # back to back within a replication
                            row = run_strategy(s, g, f, p, B, make_generator([seed, tag, b_i, sc["sid"], g_i, STRATS.index(s), r]),
                                               cms[(f.m, g)], cal4, L)
                            rows.append(dict(part=label, sid=sc["sid"], m=f.m, budget=B, greek=g, strategy=s, rep=r, **row))
            log(f"{label}: budget {B * 1e3:.2f} ms done")
        return rows

    rows = block(2, scen, GREEKS, budgets, STRATS, reps, "main")
    sw = CFG["cliff_sweep"]
    sweep_b = list(np.geomspace(0.002, 0.064, 6 if quick else 24))
    first = {}
    for sc in scen:
        first.setdefault(sc["m"], sc)
    rows += block(3, list(first.values()), [sw["greek"]], sweep_b, sw["strategies"], reps, "sweep")
    pd.DataFrame(rows).to_csv(out / "runs_raw.csv.gz", index=False)
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))


def boot_gm_ratio(d, num, den, rng):
    """Geometric mean over (sid, greek) cells of wn_num / wn_den with a 95% bootstrap CI
    (replications resampled independently within each cell and strategy). None if no common cell."""
    lr, lrb, n = 0.0, np.zeros(NBOOT), 0
    for _, c in d.groupby(["sid", "greek"]):
        a, b = c[c.strategy == num], c[c.strategy == den]
        if len(a) == 0 or len(b) == 0 or not (a.feasible.all() and b.feasible.all()):
            continue
        parts = []
        for x in (a, b):
            e2, w, r2 = x.err.to_numpy() ** 2, x.wall.to_numpy(), float(x.ref_se.iloc[0]) ** 2
            idx = rng.integers(0, len(e2), (NBOOT, len(e2)))
            parts.append((math.log(max(e2.mean() - r2, 1e-300) * w.mean()),
                          np.log(np.maximum(e2[idx].mean(1) - r2, 1e-300) * w[idx].mean(1))))
        lr += parts[0][0] - parts[1][0]
        lrb += parts[0][1] - parts[1][1]
        n += 1
    if n == 0:
        return None
    lo, hi = np.percentile(lrb / n, [2.5, 97.5])
    return dict(gm_ratio=math.exp(lr / n), ci_lo=math.exp(lo), ci_hi=math.exp(hi), n_cells=n)


def analyse(out):
    refs = pd.read_csv(out / "references.csv").set_index("sid")
    raw = pd.read_csv(out / "runs_raw.csv.gz")
    cal = pd.read_csv(out / "calibration.csv")
    rng = np.random.default_rng(CFG["seed"] + 99)
    raw["truth"] = [refs.at[s, f"ref_{g}"] for s, g in zip(raw.sid, raw.greek)]
    raw["ref_se"] = [refs.at[s, f"ref_se_{g}"] for s, g in zip(raw.sid, raw.greek)]
    raw["err"] = raw.value - raw.truth
    raw["over"] = raw.wall / raw.budget
    raw["covered"] = (raw.err.abs() <= raw.ci).where(raw.feasible)
    raw["pred_ratio"] = raw.wall / raw.pred_cost
    main, sweep = raw[raw.part == "main"], raw[raw.part == "sweep"]

    def q95(v):
        return v.quantile(0.95)

    ov = main.groupby(["m", "budget", "strategy"]).agg(
        feasible=("feasible", "mean"), median=("over", "median"), p95=("over", q95), max=("over", "max"),
        miss_1_10=("over", lambda v: (v > 1.10).mean()), miss_1_25=("over", lambda v: (v > 1.25).mean()),
        units_median=("units", "median"), units_min=("units", "min"), n_per_unit=("n_per_unit", "median"),
        coverage=("covered", "mean"), pred_ratio_median=("pred_ratio", "median"), pred_ratio_p95=("pred_ratio", q95)).reset_index()
    ov.loc[ov.feasible == 0, ["median", "p95", "max", "miss_1_10", "miss_1_25"]] = np.nan
    ov.to_csv(out / "overruns_and_coverage.csv", index=False)

    s = main.groupby(["sid", "m", "budget", "greek", "strategy"]).agg(
        feasible=("feasible", "all"), mse=("err", lambda e: float((e**2).mean())), ref_se=("ref_se", "first"),
        wall=("wall", "mean"), coverage=("covered", "mean")).reset_index()
    s["mse_c"] = np.maximum(s.mse - s.ref_se**2, 1e-300)
    s["wn"] = s.mse_c * s.wall
    s.loc[~s.feasible, ["mse", "mse_c", "wn"]] = np.nan
    s.to_csv(out / "summary.csv", index=False)

    cov = main[main.feasible].groupby("strategy").agg(coverage=("covered", "mean"), n=("covered", "size"),
                                                         units_median=("units", "median"), units_min=("units", "min")).reset_index()
    cov["mc_se"] = np.sqrt(cov.coverage * (1 - cov.coverage) / cov.n)
    cov.to_csv(out / "coverage_pooled.csv", index=False)

    pairs = [("PCA-opt-mse", "PCA-base"), ("PCA-opt-ci", "PCA-opt-mse"), ("PCA-shift-mse", "PCA-opt-mse"), ("PCA-shift-ci", "PCA-shift-mse"),
             ("PCA-fixed64", "PCA-opt-mse"), ("PCA-fixed512", "PCA-opt-mse"), ("MC-look", "MC-base"),
             ("PCA-opt-mse", "MC-look"), ("PCA-shift-mse", "MC-look"), ("PCA-base", "MC-base")]
    pr = []
    for (m, B), d in main.groupby(["m", "budget"]):
        for num, den in pairs:
            r = boot_gm_ratio(d, num, den, rng)
            if r:
                pr.append(dict(m=m, budget=B, num=num, den=den, **r))
    for num, den in pairs:
        r = boot_gm_ratio(main.assign(sid=main.sid * 1000 + (main.budget * 1e4).round().astype(int)), num, den, rng)
        if r:
            pr.append(dict(m=0, budget=np.nan, num=num, den=den, **r))
    pr = pd.DataFrame(pr)
    pr.to_csv(out / "pairwise_wn_ratios.csv", index=False)

    # composite policies
    th = cal.groupby("m").agg(p4=("phase4_threshold_s", "first")).p4
    cg = cal.set_index(["m", "greek"])
    comp = []
    for (sid, m, B, g), d in s.groupby(["sid", "m", "budget", "greek"]):
        d = d.set_index("strategy")
        def pick(name):
            return d.loc[name] if (name in d.index and bool(d.loc[name, "feasible"])) else None
        choices = {"phase4": "PCA-base" if B >= th[m] else "MC-base"}
        for rand, nm in (("owen", "PCA-opt-mse"), ("shift", "PCA-shift-mse")):
            want = B >= cg.loc[(m, g), f"{rand}_threshold_s"] and pick(nm) is not None
            choices[f"feasible-{rand}"] = nm if want else ("MC-look" if pick("MC-look") is not None else None)
        best = d[d.feasible & d.index.isin(["MC-look", "PCA-opt-mse", "PCA-shift-mse"])].wn.min()
        for pol, ch in choices.items():
            row = pick(ch) if ch else None
            comp.append(dict(policy=pol, sid=sid, m=m, budget=B, greek=g, choice=ch or "infeasible",
                             wn=np.nan if row is None else row.wn, mse_c=np.nan if row is None else row.mse_c,
                             wall=np.nan if row is None else row.wall, best_lookahead_wn=best))
    comp = pd.DataFrame(comp)
    comp["regret_vs_best_lookahead"] = np.sqrt(comp.wn / comp.best_lookahead_wn)
    comp.to_csv(out / "composite_policies.csv", index=False)
    cs = comp.groupby(["policy", "m", "budget"]).agg(
        choice=("choice", lambda v: "/".join(sorted(set(v)))), gm_wn=("wn", lambda v: math.exp(np.log(v).mean()) if v.notna().all() else np.nan),
        gm_mse=("mse_c", lambda v: math.exp(np.log(v).mean()) if v.notna().all() else np.nan), wall=("wall", "mean"),
        gm_regret=("regret_vs_best_lookahead", lambda v: math.exp(np.log(v).mean()) if v.notna().all() else np.nan)).reset_index()
    cs.to_csv(out / "composite_summary.csv", index=False)

    # cliff sweep
    sweep = sweep.copy()
    cl = sweep.groupby(["m", "strategy", "budget"]).agg(feasible=("feasible", "all"), mse=("err", lambda e: float((e**2).mean())),
                                                         wall=("wall", "mean"), n_per_unit=("n_per_unit", "median"),
                                                         units=("units", "median"), p95_over=("over", q95)).reset_index()
    cl.loc[~cl.feasible, ["mse", "wall"]] = np.nan
    cl.to_csv(out / "cliff_sweep.csv", index=False)
    cliff = []
    for (m, st), d in cl.groupby(["m", "strategy"]):
        d = d.dropna(subset=["mse"]).sort_values("budget")
        if len(d) > 1:
            jump = (d.mse.to_numpy()[1:] / d.mse.to_numpy()[:-1])
            cliff.append(dict(m=m, strategy=st, max_adjacent_mse_ratio=float(jump.max()), n_increases=int((jump > 1).sum()),
                              n_steps=len(jump), first_feasible_budget=float(d.budget.min())))
    cliff = pd.DataFrame(cliff)
    cliff.to_csv(out / "cliff_summary.csv", index=False)

    look = [x for x in STRATS if x not in ("MC-base", "PCA-base")]
    lo = ov[ov.strategy.isin(look) & (ov.feasible > 0)]
    r8 = cov[cov.strategy.isin(["PCA-fixed64", "PCA-fixed512", "PCA-opt-mse", "PCA-shift-mse"])]
    allp = pr[pr.m == 0].set_index(["num", "den"])
    piv = cliff.pivot_table(index="m", columns="strategy", values="max_adjacent_mse_ratio")
    verdict = {
        "S1_lookahead_max_cell_miss_share_1_10": float(lo.miss_1_10.max()),
        "S1_lookahead_cells_with_miss_share_over_5pct": int((lo.miss_1_10 > 0.05).sum()), "S1_lookahead_cells": int(len(lo)),
        "S1_lookahead_max_overrun": float(lo["max"].max()),
        "S1_pass": bool((lo.miss_1_10 <= 0.05).all()),
        "S1_baseline_PCA_cells_miss_over_5pct": int((ov[ov.strategy == "PCA-base"].miss_1_10 > 0.05).sum()),
        "S1_baseline_PCA_max_overrun": float(ov[ov.strategy == "PCA-base"]["max"].max()),
        "S2_min_pooled_coverage_rmin8": float(r8.coverage.min()), "S2_pass": bool((r8.coverage >= 0.93).all()),
        "S2_pooled_coverage": {r.strategy: round(float(r.coverage), 4) for r in cov.itertuples()},
        "S3_opt_over_base": allp.loc[("PCA-opt-mse", "PCA-base")].to_dict() if ("PCA-opt-mse", "PCA-base") in allp.index else None,
        "S4_shift_over_owen": allp.loc[("PCA-shift-mse", "PCA-opt-mse")].to_dict() if ("PCA-shift-mse", "PCA-opt-mse") in allp.index else None,
        "S4_feasible_cells": {st: int((ov[ov.strategy == st].feasible > 0).sum()) for st in ("PCA-opt-mse", "PCA-shift-mse")},
        "S5_max_adjacent_mse_ratio": piv.round(3).to_dict() if len(piv) else None,
    }
    (out / "verdict.json").write_text(json.dumps(verdict, indent=2, default=float))
    log(json.dumps(verdict, default=float))
    figures(out, ov, s, cl, cs)


def figures(out, ov, s, cl, cs):
    fd = out / "figures"
    fd.mkdir(exist_ok=True)
    ms = sorted(ov.m.unique())
    fig, axes = plt.subplots(1, len(ms), figsize=(4.3 * len(ms), 3.7), squeeze=False, sharey=True)
    for ax, m in zip(axes[0], ms):
        for st in STRATS:
            d = ov[(ov.m == m) & (ov.strategy == st)].dropna(subset=["p95"]).sort_values("budget")
            ax.plot(d.budget * 1e3, d.p95, marker="o", ms=3.5, color=COL[st], label=st, ls="--" if "base" in st else "-")
        ax.axhline(1.10, color="#e34948", ls=":", lw=1)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(f"m = {m}"); ax.set_xlabel("budget [ms]")
    axes[0, 0].set_ylabel("95th percentile of measured wall / budget\n(missing point: plan infeasible, nothing run)")
    axes[0, 0].legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(fd / "fig5B1_overruns.png", bbox_inches="tight"); fig.savefig(fd / "fig5B1_overruns.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(ms), figsize=(4.3 * len(ms), 3.7), squeeze=False)
    for ax, m in zip(axes[0], ms):
        for st in STRATS:
            d = s[(s.m == m) & (s.strategy == st) & (s.greek == "delta")].dropna(subset=["mse_c"])
            d = d.groupby("budget").agg(wall=("wall", "mean"), mse=("mse_c", lambda v: math.exp(np.log(v).mean()))).reset_index()
            ax.loglog(d.wall * 1e3, d.mse, marker="o", ms=3.5, color=COL[st], label=st, ls="--" if "base" in st else "-")
        ax.set_title(f"Delta, m = {m}"); ax.set_xlabel("mean measured wall time [ms]")
    axes[0, 0].set_ylabel("MSE (geometric mean over 4 scenarios)")
    axes[0, 0].legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(fd / "fig5B2_mse_vs_time.png", bbox_inches="tight"); fig.savefig(fd / "fig5B2_mse_vs_time.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, len(ms), figsize=(4.3 * len(ms), 6.2), squeeze=False, sharex=True)
    for j, m in enumerate(ms):
        for st in CFG["cliff_sweep"]["strategies"]:
            d = cl[(cl.m == m) & (cl.strategy == st)].dropna(subset=["mse"]).sort_values("budget")
            axes[0, j].loglog(d.budget * 1e3, d.mse, marker="o", ms=3, color=COL[st], label=st, ls="--" if "base" in st else "-")
            if st != "MC-look":
                axes[1, j].step(d.budget * 1e3, d.n_per_unit, where="post", color=COL[st], label=st, ls="--" if "base" in st else "-")
        axes[0, j].set_title(f"Delta, m = {m} (one scenario)")
        axes[1, j].set_yscale("log", base=2); axes[1, j].set_xlabel("nominal budget [ms]")
    axes[0, 0].set_ylabel("MSE at the nominal budget"); axes[1, 0].set_ylabel("points per replicate N")
    axes[0, 0].legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(fd / "fig5B3_cliff_sweep.png", bbox_inches="tight"); fig.savefig(fd / "fig5B3_cliff_sweep.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(ms), figsize=(4.3 * len(ms), 3.6), squeeze=False, sharey=True)
    for ax, m in zip(axes[0], ms):
        for pol, c in (("phase4", "#9a9890"), ("feasible-owen", "#2a78d6"), ("feasible-shift", "#1baf7a")):
            d = cs[(cs.m == m) & (cs.policy == pol)].dropna(subset=["gm_mse"]).sort_values("budget")
            ax.loglog(d.wall * 1e3, d.gm_mse, marker="o", ms=3.5, color=c, label=pol)
        ax.set_title(f"m = {m}"); ax.set_xlabel("mean measured wall time [ms]")
    axes[0, 0].set_ylabel("MSE, composite policies\n(geo-mean over scenarios and Greeks, per-Greek units)")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fd / "fig5B4_composite_policies.png", bbox_inches="tight"); fig.savefig(fd / "fig5B4_composite_policies.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--analysis-only", action="store_true")
    a = ap.parse_args()
    out = RESULTS / "phase5_sampling_policy" / ("quick" if a.quick else "study")
    t0 = time.time()
    if not a.analysis_only:
        simulate(out, a.quick)
        meta = json.loads((out / "meta.json").read_text())
        meta["wall_seconds_total"] = time.time() - t0
        (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    analyse(out)

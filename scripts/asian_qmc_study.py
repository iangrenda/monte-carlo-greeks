"""Phase 3: dimension reduction for RQMC Greeks of the Asian digital, under real
wall-clock budgets; fixed vs regime-aware vs pilot-based selection.

    python asian_qmc_study.py --config quick       # ~2 min pipeline check
    python asian_qmc_study.py --config research    # ~35-45 min
    python asian_qmc_study.py --config research --analysis-only

Estimator: CPW-first (level variable Z_1 integrated out exactly) for Delta, Gamma, Vega.
Samplers: MC; RQMC with sequential, Brownian-bridge or PCA construction of the
residual path. Outputs: results/asian_qmc/<config>/.
"""
import argparse
import itertools
import json
import math
import time
from dataclasses import asdict, dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mc_greeks import visualization as viz  # noqa: F401  (house style)
from mc_greeks.asian import AsianDigital
from mc_greeks.asian_qmc import (CONSTRUCTIONS, SAMPLERS, adaptive_timed, calibrate_rqmc, construction_matrix,
                                 explained_variance, functional_share, increments_from_residual, pps_for,
                                 reference_estimates, timed_estimate)
from mc_greeks.asian import cpw_first_samples
from mc_greeks.config import RESULTS
from mc_greeks.experiments import environment_info
from mc_greeks.models import BlackScholesParams
from mc_greeks.simulation import make_generator
from mc_greeks.statistics import loglog_slope

GREEKS3 = ("delta", "gamma", "vega")
COL = {"MC": "#eb6834", "RQMC-seq": "#9a9890", "RQMC-bb": "#1baf7a", "RQMC-pca": "#2a78d6"}
ADAPTIVE = {"adaptive (MC-rate prediction)": "mc", "adaptive (N^-2 RQMC prediction)": "n2"}


@dataclass(frozen=True)
class QMCConfig:
    name: str
    ms: tuple = (4, 12, 52)
    moneyness: tuple = (0.92, 1.0, 1.08)
    sigmas: tuple = (0.12, 0.28)
    Ts: tuple = (0.4, 1.5)
    r: float = 0.03
    budgets: tuple = (0.005, 0.05)   # seconds of real wall-clock time per estimate
    reps: int = 30
    seed: int = 2026_1110
    ref_points: int = 2**16
    ref_scrambles: int = 32
    check_paths: int = 2**20
    conv_log2: tuple = (6, 7, 8, 9, 10, 11, 12, 13, 14)
    conv_scrambles: int = 32

    def scenarios(self):
        return [dict(sid=i, m=m, moneyness=k, sigma=s, T=T, r=self.r, S0=100.0)
                for i, (m, k, s, T) in enumerate(itertools.product(self.ms, self.moneyness, self.sigmas, self.Ts))]


CONFIGS = {
    "quick": QMCConfig("quick", ms=(4, 12), moneyness=(1.0,), sigmas=(0.2,), Ts=(1.0,), budgets=(0.005,), reps=6,
                       ref_points=2**12, ref_scrambles=16, check_paths=2**17, conv_log2=(6, 8, 10), conv_scrambles=8),
    "research": QMCConfig("research"),
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def obj(sc):
    p = BlackScholesParams(sc["S0"], sc["r"], sc["sigma"], sc["T"])
    return AsianDigital(sc["moneyness"] * sc["S0"], int(sc["m"])), p


def write_calibration(cal, path):
    rows = [dict(kind=k[0], m=k[1], sec_per_unit_overhead=v[0], sec_per_point=v[1]) for k, v in cal.items() if k[0] != "raw"]
    rows += [dict(kind=f"raw-{c}", m=k[1], n=n, median_sec_per_unit=t) for k, v in cal.items() if k[0] == "raw"
             for c, d in v.items() for n, t in d.items()]
    pd.DataFrame(rows).to_csv(path, index=False)


def rerun_timed(cfg, out):
    """Re-run only the calibration-dependent parts (RQMC fixed runs, adaptive runs)
    with the pooled cost model. References, the convergence study and the MC
    fixed runs are kept. First-pass files are preserved as *_v1."""
    import shutil
    for name in ("calibration.csv", "fixed_raw.csv.gz", "adaptive_raw.csv.gz"):
        src, dst = out / name, out / name.replace(".csv", "_v1.csv")
        if src.exists() and not dst.exists():
            shutil.copy(src, dst)
    log("pooled calibration")
    cal = calibrate_rqmc(cfg.ms, seed=cfg.seed + 7, pooled=True)
    write_calibration(cal, out / "calibration.csv")
    old = pd.read_csv(out / "fixed_raw_v1.csv.gz")
    fixed, adapt = [old[old.strategy == "MC"].to_dict("records")], []
    fixed = fixed[0]
    for b_i, B in enumerate(cfg.budgets):
        for sc in cfg.scenarios():
            f, p = obj(sc)
            for g_i, g in enumerate(GREEKS3):
                for s_i, s in enumerate(SAMPLERS):
                    if s == "MC":
                        continue
                    pps = pps_for(B, cal, s, f.m)
                    for r in range(cfg.reps):
                        e = timed_estimate(g, f, p, B, make_generator([cfg.seed, 12, b_i, sc["sid"], g_i, s_i, r]), s,
                                           points_per_scramble=pps)
                        fixed.append(dict(sid=sc["sid"], budget=B, greek=g, strategy=s, rep=r, value=e.value,
                                          stderr=e.stderr, ci=e.ci_halfwidth, wall=e.wall, n_paths=e.n_paths,
                                          units=e.n_units, pps=pps))
                for a_i, (name, rate) in enumerate(ADAPTIVE.items()):
                    for r in range(cfg.reps):
                        e, info = adaptive_timed(g, f, p, B, cal, make_generator([cfg.seed, 13, b_i, sc["sid"], g_i, a_i, r]),
                                                 make_generator([cfg.seed, 14, b_i, sc["sid"], g_i, a_i, r]), rqmc_rate=rate)
                        adapt.append(dict(sid=sc["sid"], budget=B, greek=g, strategy=name, rep=r, value=e.value,
                                          stderr=e.stderr, ci=e.ci_halfwidth, wall=e.wall, n_paths=e.n_paths,
                                          units=e.n_units, chosen=info["chosen"], pilot_wall=info["pilot_wall"],
                                          pilot_exhausted=info["pilot_exhausted"]))
            log(f"re-run: budget {B * 1e3:.0f} ms, scenario {sc['sid'] + 1}")
    pd.DataFrame(fixed).to_csv(out / "fixed_raw.csv.gz", index=False)
    pd.DataFrame(adapt).to_csv(out / "adaptive_raw.csv.gz", index=False)


def simulate(cfg, out):
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    scen = pd.DataFrame(cfg.scenarios())
    scen.to_csv(out / "scenarios.csv", index=False)

    log("structure of the constructions")
    rows = []
    for m in cfg.ms:
        for c in CONSTRUCTIONS:
            for k in (1, 2, 4):
                if k <= m - 1:
                    rows.append(dict(m=m, construction=c, k=k, path_var_share=explained_variance(c, m, 1.0 / m, k),
                                     average_var_share=functional_share(c, m, 1.0 / m, k)))
    pd.DataFrame(rows).to_csv(out / "construction_structure.csv", index=False)

    log("calibrating wall-clock cost model")
    cal = calibrate_rqmc(cfg.ms, seed=cfg.seed)
    write_calibration(cal, out / "calibration.csv")

    log("fixed-N convergence study (mechanism)")
    conv = []
    for m in cfg.ms:
        f, p = AsianDigital(100.0, m), BlackScholesParams(100.0, cfg.r, 0.2, 1.0)
        ref = reference_estimates(f, p, make_generator([cfg.seed, 9, m]), cfg.ref_points, cfg.ref_scrambles,
                                  check_paths=2**16, greeks=GREEKS3)
        for c in CONSTRUCTIONS:
            L = construction_matrix(c, m, p.T / m)
            for lg in cfg.conv_log2:
                n = 2**lg
                rng = make_generator([cfg.seed, 10, m, CONSTRUCTIONS.index(c), lg])
                from scipy import special
                from scipy.stats import qmc
                ests = {g: [] for g in GREEKS3}
                for _ in range(cfg.conv_scrambles):
                    sob = qmc.Sobol(d=m - 1, scramble=True, seed=rng)
                    Z = increments_from_residual(special.ndtri(np.clip(sob.random(n), 1e-16, 1 - 1e-16)) @ L.T, p.T / m)
                    for g in GREEKS3:
                        ests[g].append(cpw_first_samples(f, p, Z, g).mean())
                for g in GREEKS3:
                    e = np.array(ests[g])
                    conv.append(dict(m=m, construction=c, n=n, greek=g, rmse=float(np.sqrt(np.mean((e - ref[f"ref_{g}"]) ** 2))),
                                     ref_se=ref[f"ref_se_{g}"], truth=ref[f"ref_{g}"]))
        # MC benchmark at the same N: exact per-path variance estimate from a large sample
        rng = make_generator([cfg.seed, 11, m])
        Zb = rng.standard_normal((2**16, m))
        for g in GREEKS3:
            v = cpw_first_samples(f, p, Zb, g).var(ddof=1)
            for lg in cfg.conv_log2:
                conv.append(dict(m=m, construction="MC", n=2**lg, greek=g, rmse=math.sqrt(v / 2**lg),
                                 ref_se=np.nan, truth=np.nan))
    pd.DataFrame(conv).to_csv(out / "convergence_fixed_n.csv", index=False)

    log(f"references for {len(scen)} scenarios")
    refs = []
    for sc in cfg.scenarios():
        f, p = obj(sc)
        refs.append({"sid": sc["sid"], **reference_estimates(f, p, make_generator([cfg.seed, 1, sc["sid"]]),
                                                             cfg.ref_points, cfg.ref_scrambles, cfg.check_paths)})
    pd.DataFrame(refs).to_csv(out / "references.csv", index=False)

    log("timed fixed-sampler runs and adaptive runs")
    fixed, adapt = [], []
    for b_i, B in enumerate(cfg.budgets):
        for sc in cfg.scenarios():
            f, p = obj(sc)
            for g_i, g in enumerate(GREEKS3):
                for s_i, s in enumerate(SAMPLERS):
                    pps = pps_for(B, cal, s, f.m)
                    for r in range(cfg.reps):
                        e = timed_estimate(g, f, p, B, make_generator([cfg.seed, 2, b_i, sc["sid"], g_i, s_i, r]), s,
                                           points_per_scramble=pps)
                        fixed.append(dict(sid=sc["sid"], budget=B, greek=g, strategy=s, rep=r, value=e.value,
                                          stderr=e.stderr, ci=e.ci_halfwidth, wall=e.wall, n_paths=e.n_paths,
                                          units=e.n_units, pps=pps))
                for a_i, (name, rate) in enumerate(ADAPTIVE.items()):
                    for r in range(cfg.reps):
                        e, info = adaptive_timed(g, f, p, B, cal, make_generator([cfg.seed, 3, b_i, sc["sid"], g_i, a_i, r]),
                                                 make_generator([cfg.seed, 4, b_i, sc["sid"], g_i, a_i, r]), rqmc_rate=rate)
                        adapt.append(dict(sid=sc["sid"], budget=B, greek=g, strategy=name, rep=r, value=e.value,
                                          stderr=e.stderr, ci=e.ci_halfwidth, wall=e.wall, n_paths=e.n_paths,
                                          units=e.n_units, chosen=info["chosen"], pilot_wall=info["pilot_wall"],
                                          pilot_exhausted=info["pilot_exhausted"]))
            log(f"budget {B * 1e3:.0f} ms, scenario {sc['sid'] + 1}/{len(scen)}")
    pd.DataFrame(fixed).to_csv(out / "fixed_raw.csv.gz", index=False)
    pd.DataFrame(adapt).to_csv(out / "adaptive_raw.csv.gz", index=False)
    meta = {"config": asdict(cfg), "environment": environment_info(), "wall_seconds": time.time() - t_start}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))


def gm(x):
    x = np.asarray(x, float)
    return float(np.exp(np.log(x).mean()))


def boot(x, rng, n=2000):
    lx = np.log(np.asarray(x, float))
    if lx.size == 0:
        return np.nan, np.nan, np.nan
    bs = [lx[rng.integers(0, len(lx), len(lx))].mean() for _ in range(n)]
    return math.exp(lx.mean()), math.exp(np.percentile(bs, 2.5)), math.exp(np.percentile(bs, 97.5))


def analyse(cfg, out):
    scen = pd.read_csv(out / "scenarios.csv")
    refs = pd.read_csv(out / "references.csv")
    raw = pd.concat([pd.read_csv(out / "fixed_raw.csv.gz"), pd.read_csv(out / "adaptive_raw.csv.gz")])
    rng = np.random.default_rng(cfg.seed)

    chk = refs[[c for c in refs if c.startswith("check_z_")]].stack()
    pd.DataFrame({"n_checks": [len(chk)], "mean_z": [chk.mean()], "sd_z": [chk.std()], "max_abs_z": [chk.abs().max()]}
                 ).to_csv(out / "reference_quality.csv", index=False)

    rows = []
    rl = refs.set_index("sid")
    for (sid, B, g, s), d in raw.groupby(["sid", "budget", "greek", "strategy"]):
        truth, rse = rl.loc[sid, f"ref_{g}"], rl.loc[sid, f"ref_se_{g}"]
        err = d.value - truth
        mse = float((err**2).mean())
        rows.append(dict(sid=sid, budget=B, greek=g, strategy=s, truth=truth, ref_se=rse, bias=err.mean(),
                         bias_se=err.std(ddof=1) / math.sqrt(len(d)), mse=mse, mse_c=max(mse - rse**2, 1e-300),
                         rmse=math.sqrt(mse), mean_wall=d.wall.mean(), wall_over_budget=float((d.wall / B).median()),
                         coverage=float((err.abs() <= d.ci).mean()), mean_units=d.units.mean(), mean_paths=d.n_paths.mean(),
                         reps=len(d)))
    summ = pd.DataFrame(rows).merge(scen, on="sid")
    summ["wn_mse"] = summ.mse_c * summ.mean_wall  # time-normalised MSE (MSE at 1 s of compute, MC scaling)
    key = ["sid", "budget", "greek"]
    fx = summ[summ.strategy.isin(SAMPLERS)]
    orc = fx.loc[fx.groupby(key).wn_mse.idxmin(), key + ["strategy", "wn_mse"]].rename(
        columns={"strategy": "oracle", "wn_mse": "oracle_wn"})
    summ = summ.merge(orc, on=key)
    summ["regret"] = np.sqrt(summ.wn_mse / summ.oracle_wn)
    summ.to_csv(out / "summary.csv", index=False)

    # Q1: does dimension reduction recover the RQMC advantage at large m?
    base = summ[summ.strategy == "MC"].set_index(key).wn_mse
    q1 = []
    for (B, g, m, s), d in summ[summ.strategy.isin(SAMPLERS)].groupby(["budget", "greek", "m", "strategy"]):
        ratio = d.set_index(key).wn_mse / base.reindex(d.set_index(key).index)
        v, lo, hi = boot(ratio, rng)
        q1.append(dict(budget=B, greek=g, m=m, strategy=s, gm_wnmse_ratio_vs_MC=v, ci_lo=lo, ci_hi=hi,
                       n_better_than_MC=int((ratio < 1).sum()), n=len(ratio), median_coverage=d.coverage.median(),
                       min_coverage=d.coverage.min(), median_wall_over_budget=d.wall_over_budget.median()))
    pd.DataFrame(q1).to_csv(out / "Q1_wnmse_ratio_vs_mc.csv", index=False)
    summ.groupby(["budget", "m", "greek", "oracle"]).sid.nunique().rename("count").reset_index().to_csv(
        out / "oracle_choices.csv", index=False)

    # Q2: fixed vs regime-aware vs adaptive, out of sample
    sids = scen.sid.to_numpy()
    perm = np.random.default_rng(cfg.seed + 1).permutation(sids)
    splits = {"random half": (set(perm[: len(sids) // 2]), set(perm[len(sids) // 2:]))}
    if len(cfg.ms) > 1:
        mx = max(cfg.ms)
        splits[f"train m<{mx}, test m={mx}"] = (set(scen.sid[scen.m < mx]), set(scen.sid[scen.m == mx]))
    calib = pd.read_csv(out / "calibration.csv")
    pca_rows = calib[calib.kind == "pca"].set_index("m")
    cal_a, cal_b = pca_rows.sec_per_unit_overhead.to_dict(), pca_rows.sec_per_point.to_dict()
    q2 = []
    for split, (train, test) in splits.items():
        for (B, g), d in summ.groupby(["budget", "greek"]):
            tr, te = d[d.sid.isin(train)], d[d.sid.isin(test)]
            reg = {s: gm(tr[tr.strategy == s].regret) for s in SAMPLERS}
            best_fixed = min(reg, key=reg.get)
            # regime-aware: sampler X if m <= threshold else Y; thresholds below the largest training m
            tms = sorted(tr.m.unique())
            best_rule, best_val = None, np.inf
            for thr in tms[:-1]:
                for X, Y in itertools.product(SAMPLERS, SAMPLERS):
                    sel = pd.concat([tr[(tr.m <= thr) & (tr.strategy == X)], tr[(tr.m > thr) & (tr.strategy == Y)]])
                    v = gm(sel.regret)
                    if v < best_val:
                        best_rule, best_val = (thr, X, Y), v
            strategies = {f"fixed: {s}": te[te.strategy == s] for s in SAMPLERS}
            strategies[f"best fixed on train ({best_fixed})"] = te[te.strategy == best_fixed]
            if best_rule:
                thr, X, Y = best_rule
                strategies[f"regime rule on train (m<={thr}: {X}; else {Y})"] = pd.concat(
                    [te[(te.m <= thr) & (te.strategy == X)], te[(te.m > thr) & (te.strategy == Y)]])
            strategies["a-priori rule (always RQMC-pca)"] = te[te.strategy == "RQMC-pca"]
            # POST-HOC (formulated after inspecting these results): pilot-free cost-model rule.
            # Use PCA-RQMC iff 10 scramblings of >= 64 points fit in the budget under the
            # calibrated cost model t(n) = a + b n; else MC. Uses only the calibration.
            pca_ok = {mm: B >= 10 * (cal_a[mm] + 64 * cal_b[mm]) for mm in te.m.unique()}
            strategies["POST-HOC cost-model rule (PCA if 10x(a+64b) <= B, else MC)"] = pd.concat(
                [te[(te.m == mm) & (te.strategy == ("RQMC-pca" if ok else "MC"))] for mm, ok in pca_ok.items()])
            for name in ADAPTIVE:
                strategies[name] = te[te.strategy == name]
            oracle_te = te.drop_duplicates("sid")
            strategies["oracle (evaluation only)"] = oracle_te.assign(regret=1.0)
            for name, sel in strategies.items():
                v, lo, hi = boot(sel.regret, rng)
                q2.append(dict(split=split, budget=B, greek=g, strategy=name, gm_regret=v, ci_lo=lo, ci_hi=hi,
                               worst=sel.regret.max(), n_test=sel.sid.nunique()))
    pd.DataFrame(q2).to_csv(out / "Q2_rules_out_of_sample.csv", index=False)

    ad = pd.read_csv(out / "adaptive_raw.csv.gz").merge(orc, on=key)
    ad["hit"] = ad.chosen == ad.oracle
    ad["pilot_over_B"] = ad.pilot_wall / ad.budget
    ad["total_over_B"] = ad.wall / ad.budget
    ad.groupby(["strategy", "budget", "greek"]).agg(
        share_chose_oracle=("hit", "mean"), pilot_wall_over_budget=("pilot_over_B", "median"),
        total_wall_over_budget=("total_over_B", "median"), share_pilot_exhausted=("pilot_exhausted", "mean"),
        most_common_choice=("chosen", lambda x: x.value_counts().index[0])).reset_index().to_csv(
        out / "adaptive_behaviour.csv", index=False)
    ad.groupby(["strategy", "budget", "greek", "chosen"]).size().rename("count").reset_index().to_csv(
        out / "adaptive_choices.csv", index=False)
    figures(cfg, out, summ)
    log(f"analysis written to {out}")


def figures(cfg, out, summ):
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    # F1: structure
    st = pd.read_csv(out / "construction_structure.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2), sharey=True)
    for ax, col, title in zip(axes, ["path_var_share", "average_var_share"],
                              ["path variance (trace of C)", "variance of the path average"]):
        d = st[st.k == 1]
        for i, c in enumerate(CONSTRUCTIONS):
            dd = d[d.construction == c].sort_values("m")
            ax.bar(np.arange(len(dd)) + (i - 1) * 0.27, dd[col], width=0.25, color=COL[f"RQMC-{c}"], label=c)
        ax.set_xticks(range(len(cfg.ms)), [f"m={m}" for m in cfg.ms])
        ax.set_title(f"share of {title}\ncarried by the first Sobol coordinate", fontsize=9)
        ax.set_ylim(0, 1.05)
    axes[0].legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"figQ0_construction_structure.{ext}", bbox_inches="tight")
    plt.close(fig)
    # F2: fixed-N convergence
    cv = pd.read_csv(out / "convergence_fixed_n.csv")
    fig, axes = plt.subplots(len(GREEKS3), len(cfg.ms), figsize=(3.4 * len(cfg.ms), 7.5), squeeze=False)
    slopes = []
    for i, g in enumerate(GREEKS3):
        for j, m in enumerate(cfg.ms):
            ax = axes[i, j]
            for s, c in [("MC", "MC"), ("seq", "RQMC-seq"), ("bb", "RQMC-bb"), ("pca", "RQMC-pca")]:
                d = cv[(cv.greek == g) & (cv.m == m) & (cv.construction == s)].sort_values("n")
                ax.loglog(d.n, d.rmse, marker="o", color=COL[c], label=c, ls="--" if s == "MC" else "-")
                if s != "MC":
                    fit = loglog_slope(d.n, d.rmse)
                    slopes.append(dict(greek=g, m=m, construction=s, slope=fit["slope"], lo=fit["slope_lo"], hi=fit["slope_hi"]))
            ax.set_title(f"{g}, m={m}", fontsize=9)
            if i == len(GREEKS3) - 1:
                ax.set_xlabel("points per scrambling N")
            if j == 0:
                ax.set_ylabel("RMSE (one scrambling)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("CPW-first: RMSE vs N at fixed sample size (32 scramblings per point)", y=1.0)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"figQ1_convergence_fixed_n.{ext}", bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(slopes).to_csv(out / "convergence_slopes.csv", index=False)
    # F3: time-normalised MSE ratio vs MC, by m
    q1 = pd.read_csv(out / "Q1_wnmse_ratio_vs_mc.csv")
    fig, axes = plt.subplots(len(cfg.budgets), 3, figsize=(11, 3.3 * len(cfg.budgets)), squeeze=False)
    for i, B in enumerate(cfg.budgets):
        for j, g in enumerate(GREEKS3):
            ax = axes[i, j]
            for k, s in enumerate(["RQMC-seq", "RQMC-bb", "RQMC-pca"]):
                d = q1[(q1.budget == B) & (q1.greek == g) & (q1.strategy == s)].sort_values("m")
                x = np.arange(len(d)) + (k - 1) * 0.12
                ax.errorbar(x, d.gm_wnmse_ratio_vs_MC, yerr=[d.gm_wnmse_ratio_vs_MC - d.ci_lo, d.ci_hi - d.gm_wnmse_ratio_vs_MC],
                            color=COL[s], marker="o", capsize=2, label=s)
            ax.axhline(1, color="#1f1f1e", lw=0.9)
            ax.set_yscale("log")
            ax.set_xticks(range(len(cfg.ms)), [f"m={m}" for m in cfg.ms])
            ax.set_title(f"{g}, wall-clock budget {B * 1e3:.0f} ms", fontsize=9)
            if j == 0:
                ax.set_ylabel("time-normalised MSE / MC\n(geo-mean over scenarios, 95% CI)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Does dimension reduction recover the RQMC advantage at large m? (<1: RQMC beats MC at equal time)", y=1.0)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"figQ2_rqmc_vs_mc_by_m.{ext}", bbox_inches="tight")
    plt.close(fig)
    # F4: rules out of sample
    q2 = pd.read_csv(out / "Q2_rules_out_of_sample.csv")
    for split in q2.split.unique():
        d0 = q2[q2.split == split]
        fig, axes = plt.subplots(len(cfg.budgets), 3, figsize=(12, 1.2 + 3.3 * len(cfg.budgets)), squeeze=False)
        for i, B in enumerate(cfg.budgets):
            for j, g in enumerate(GREEKS3):
                d = d0[(d0.budget == B) & (d0.greek == g)].sort_values("gm_regret")
                ax = axes[i, j]
                y = np.arange(len(d))
                cols = ["#2a78d6" if s.startswith("adaptive") else "#1f1f1e" if ("train" in s) else "#9a9890" for s in d.strategy]
                ax.barh(y, d.gm_regret - 1, left=1, color=cols, height=0.6)
                ax.errorbar(d.gm_regret, y, xerr=[d.gm_regret - d.ci_lo, d.ci_hi - d.gm_regret], fmt="none",
                            ecolor="#1f1f1e", lw=0.8, capsize=2)
                ax.set_yticks(y, d.strategy, fontsize=6)
                ax.set_xscale("log")
                ax.axvline(1, color="#1f1f1e", lw=0.8)
                ax.set_title(f"{g}, {B * 1e3:.0f} ms", fontsize=9)
                if i == len(cfg.budgets) - 1:
                    ax.set_xlabel("time-normalised RMSE / oracle (test scenarios)")
        fig.suptitle(f"Out-of-sample regret, split: {split} (blue = adaptive, black = learned on train)", y=1.0)
        fig.tight_layout()
        tag = "random" if split.startswith("random") else "extrapolate_m"
        for ext in ("png", "pdf"):
            fig.savefig(fig_dir / f"figQ3_rules_{tag}.{ext}", bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", choices=sorted(CONFIGS), default="quick")
    ap.add_argument("--analysis-only", action="store_true")
    ap.add_argument("--rerun-timed", action="store_true",
                    help="re-run RQMC fixed + adaptive runs with the pooled cost model (keeps references, MC runs)")
    a = ap.parse_args()
    cfg = CONFIGS[a.config]
    out = RESULTS / "asian_qmc" / cfg.name
    if a.rerun_timed:
        rerun_timed(cfg, out)
    elif not a.analysis_only:
        simulate(cfg, out)
    analyse(cfg, out)

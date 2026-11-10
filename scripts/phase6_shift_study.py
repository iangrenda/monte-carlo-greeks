"""Phase 6A: holdout confirmation of the digital-shift result, with a second LMS implementation.

    python scripts/phase6_shift_study.py                     # holdout, ~20 min
    python scripts/phase6_shift_study.py --config portable   # reduced run for a second machine
    python scripts/phase6_shift_study.py --quick             # pipeline check only (NOT evidence)
    python scripts/phase6_shift_study.py --analysis-only [--config ...]

Protocol: configs/phase6_shift_holdout.json (committed before the run).
Output: results/phase6_shift/<config>[-<env tag>]/.
"""
import argparse
import json
import math
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import special
from scipy.stats import qmc

from mc_greeks import budget_policy as bp
from mc_greeks import joint_greeks as jg
from mc_greeks import visualization as viz  # noqa: F401
from mc_greeks.asian import AsianDigital
from mc_greeks.asian_qmc import increments_from_residual, mc_block, reference_estimates
from mc_greeks.config import RESULTS, ROOT
from mc_greeks.models import BlackScholesParams
from mc_greeks.simulation import make_generator
from phase5_frontier_study import provenance

CFG = json.loads((ROOT / "configs" / "phase6_shift_holdout.json").read_text())
SCH = CFG["scheduler"]
G3 = list(jg.GREEKS)
RANDS = ["owen", "lms", "shift"]                      # "owen" = SciPy LMS+shift (historical key)
LABEL = {"owen": "SciPy LMS+shift", "lms": "own LMS+shift (cached)", "shift": "digital shift only", "MC": "MC"}
COL = {"owen": "#9a9890", "lms": "#2a78d6", "shift": "#1baf7a", "MC": "#eb6834"}
EPS = np.finfo(float).eps
NBOOT = 2000


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def scenarios(cfg_name, quick):
    g = CFG["grid"]
    out, i = [], 0
    for m in g["m"]:
        for k in g["moneyness_K_over_S0"]:
            for s in g["sigma"]:
                for T in g["T"]:
                    out.append(dict(sid=i, m=m, moneyness=k, sigma=s, T=T, r=g["r"], S0=g["S0"]))
                    i += 1
    if quick or cfg_name == "portable":
        keep = CFG["configurations"]["portable"]["scenario_filter"]
        out = [sc for sc in out if sc["sigma"] == keep["sigma"] and sc["T"] == keep["T"]]
        if quick:
            out = out[::2]
    return out


def objs(sc):
    return (AsianDigital(sc["moneyness"] * sc["S0"], int(sc["m"])),
            BlackScholesParams(sc["S0"], sc["r"], sc["sigma"], sc["T"]))


def uniforms(rand, d, n, rng):
    if rand == "owen":
        return qmc.Sobol(d=d, scramble=True, seed=rng).random(n)
    if rand == "lms":
        return bp.lms_uniforms(bp.sobol_direction_ints(d, n.bit_length() - 1), n, rng)
    return bp.shifted_uniforms(bp.sobol_int_points(d, n), rng)


def simulate(out, cfg_name, quick):
    out.mkdir(parents=True, exist_ok=True)
    c = CFG["configurations"][cfg_name]
    seed = CFG["seed"]
    scen = scenarios(cfg_name, quick)
    pd.DataFrame(scen).to_csv(out / "scenarios.csv", index=False)
    ms = sorted({s["m"] for s in scen})
    meta = {"provenance": provenance("configs/phase6_shift_holdout.json"), "configuration": cfg_name, "quick": quick,
            "env_tag": os.environ.get("PHASE6_ENV_TAG", "local")}

    log(f"references and scales for {len(scen)} scenarios")
    refs = []
    for sc in scen:
        f, p = objs(sc)
        pts = (2**18 if f.m <= 24 else 2**16) if not quick else 2**12
        r = reference_estimates(f, p, make_generator([seed, 1, sc["sid"]]), pts, 32 if not quick else 16,
                                check_paths=2**20 if not quick else 2**16)
        sc_ = jg.risk_scales(f, p, make_generator([seed, 9, sc["sid"]]), 1 << 16)
        refs.append({"sid": sc["sid"], **r, **{f"scale_{g}": sc_[k] for k, g in enumerate(G3)}})
    pd.DataFrame(refs).to_csv(out / "references.csv", index=False)

    log("part 1: equal point counts (hardware-independent)")
    R1 = 8 if quick else c["equal_n_replicates"]
    log2n = CFG["equal_n"]["log2n"][:3] if quick else CFG["equal_n"]["log2n"]
    rows = []
    for sc in scen:
        f, p = objs(sc)
        dt = p.T / f.m
        L = jg.pca_factor(f.m, dt)
        for lg in log2n:
            for r_i, rand in enumerate(RANDS):
                rng = make_generator([seed, 2, sc["sid"], lg, r_i])
                for r in range(R1):
                    xi = special.ndtri(np.clip(uniforms(rand, f.m - 1, 2**lg, rng), EPS, 1 - EPS))
                    mu = jg.cpw_first_joint(f, p, increments_from_residual(xi @ L.T, dt)).mean(axis=0)
                    rows.append(dict(sid=sc["sid"], m=f.m, log2n=lg, rand=rand, rep=r, **dict(zip(G3, mu))))
        log(f"  scenario {sc['sid']} done")
    pd.DataFrame(rows).to_csv(out / "equal_n.csv.gz", index=False)

    log("part 2: cost models")
    t0 = time.perf_counter()
    cms = {m: bp.calibrate_cost_model(m, "joint", seed=seed + 11 + m, repeats=3 if quick else 7, randomisations=RANDS)
           for m in ms}
    meta["calibration_seconds"] = time.perf_counter() - t0
    pd.DataFrame([dict(m=m, **{f"{r}_{n}": v for r in RANDS for n, v in zip(("c0", "a", "b"), cm.rq[r])},
                       mc0=cm.mc[0], mc_u=cm.mc[1], mc_c=cm.mc[2], **{f"fit_err_{k}": v for k, v in cm.fit_error.items()})
                  for m, cm in cms.items()]).to_csv(out / "calibration.csv", index=False)

    log("part 2: equal measured time")
    budgets = CFG["timed"]["budgets_seconds"][1:3] if quick else CFG["timed"]["budgets_seconds"]
    reps = 4 if quick else c["timed_replications"]
    strategies = ["MC"] + RANDS
    rows = []
    for b_i, B in enumerate(budgets):
        for sc in scen:
            f, p = objs(sc)
            dt = p.T / f.m
            L = jg.pca_factor(f.m, dt)
            ev = bp.make_evaluator(f, p, "joint")
            cm = cms[f.m]
            for r in range(reps):
                for s_i, st in enumerate(strategies):          # back to back within a replication
                    rng = make_generator([seed, 3, b_i, sc["sid"], s_i, r])
                    t0 = time.perf_counter()
                    if st == "MC":
                        pl = bp.mc_schedule(B, cm, block_max=mc_block(f.m), margin=SCH["margin"])
                    else:
                        pl = bp.optimise_schedule(B, cm, st, SCH["alpha"], SCH["objective"], SCH["r_min"], margin=SCH["margin"])
                    e = bp.execute(pl, ev, 3, L, f.m, dt, B, rng, t0=t0)
                    rows.append(dict(sid=sc["sid"], m=f.m, budget=B, strategy=st, rep=r, feasible=e.feasible, wall=e.wall,
                                     units=e.n_units, n_per_unit=e.n_per_unit, pred_cost=pl.predicted_cost,
                                     **{f"v_{g}": e.value[k] for k, g in enumerate(G3)},
                                     **{f"ci_{g}": e.ci_halfwidth[k] for k, g in enumerate(G3)}))
        log(f"  budget {B * 1e3:.0f} ms done")
    pd.DataFrame(rows).to_csv(out / "timed.csv.gz", index=False)
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))


def gm(v):
    v = np.asarray(v, float)
    return float(np.exp(np.log(v).mean())) if len(v) and np.all(v > 0) and np.all(np.isfinite(v)) else float("nan")


def analyse(out):
    refs = pd.read_csv(out / "references.csv").set_index("sid")
    scale = {s: refs.loc[s, [f"scale_{g}" for g in G3]].to_numpy(float) for s in refs.index}
    truth = {s: refs.loc[s, [f"ref_{g}" for g in G3]].to_numpy(float) for s in refs.index}
    rse = {s: refs.loc[s, [f"ref_se_{g}" for g in G3]].to_numpy(float) for s in refs.index}
    rng = np.random.default_rng(CFG["seed"] + 99)
    verdict = {}

    # ---- part 1 -------------------------------------------------------------------------------
    en = pd.read_csv(out / "equal_n.csv.gz")
    var_rows, boots = [], {}
    for (sid, m, lg, rand), d in en.groupby(["sid", "m", "log2n", "rand"]):
        X = d[G3].to_numpy()
        v = X.var(axis=0, ddof=1) / scale[sid] ** 2
        idx = rng.integers(0, len(X), (NBOOT, len(X)))
        boots[(sid, lg, rand)] = (X[idx].var(axis=1, ddof=1) / scale[sid] ** 2).mean(axis=1)
        var_rows.append(dict(sid=sid, m=m, log2n=lg, rand=rand, loss=float(v.mean()), **{f"var_{g}": v[k] for k, g in enumerate(G3)}))
    V = pd.DataFrame(var_rows)
    V.to_csv(out / "equal_n_variances.csv", index=False)
    pairs = [("lms", "owen"), ("shift", "lms"), ("shift", "owen")]
    rat = []
    for (m, lg), d in V.groupby(["m", "log2n"]):
        for a, b in pairs:
            sids = sorted(d.sid.unique())
            lr = np.mean([math.log(d[(d.sid == s) & (d.rand == a)].loss.iloc[0] / d[(d.sid == s) & (d.rand == b)].loss.iloc[0]) for s in sids])
            lb = np.mean([np.log(boots[(s, lg, a)] / boots[(s, lg, b)]) for s in sids], axis=0)
            lo, hi = np.percentile(lb, [2.5, 97.5])
            per_greek = {g: gm([d[(d.sid == s) & (d.rand == a)][f"var_{g}"].iloc[0] / d[(d.sid == s) & (d.rand == b)][f"var_{g}"].iloc[0] for s in sids]) for g in G3}
            rat.append(dict(m=m, log2n=lg, num=a, den=b, gm_ratio=math.exp(lr), ci_lo=math.exp(lo), ci_hi=math.exp(hi), n_scen=len(sids),
                            **{f"ratio_{g}": v for g, v in per_greek.items()}))
    rat = pd.DataFrame(rat)
    rat.to_csv(out / "equal_n_ratios.csv", index=False)
    slopes = []
    for (sid, m, rand), d in V.groupby(["sid", "m", "rand"]):
        d = d.sort_values("log2n")
        slopes.append(dict(sid=sid, m=m, rand=rand, slope=float(np.polyfit(d.log2n * math.log(2), np.log(d.loss), 1)[0])))
    slopes = pd.DataFrame(slopes)
    slopes.groupby(["m", "rand"]).slope.agg(["median", "min", "max"]).to_csv(out / "equal_n_slopes.csv")
    e1 = rat[(rat.num == "lms") & (rat.den == "owen")]
    e2 = rat[(rat.num == "shift") & (rat.den == "lms")]
    big = e2[e2.log2n >= 10]
    verdict["A1_own_vs_scipy_lms_gm_range"] = [float(e1.gm_ratio.min()), float(e1.gm_ratio.max())]
    verdict["A1_cells_ci_excluding_1"] = int(((e1.ci_lo > 1) | (e1.ci_hi < 1)).sum())
    verdict["A1_n_cells"] = int(len(e1))
    verdict["A1_pass"] = bool(((e1.gm_ratio >= 0.9) & (e1.gm_ratio <= 1.1)).all())
    verdict["A2_shift_over_lms_at_N_ge_2^10"] = {f"m{r.m}_N2^{r.log2n}": [round(r.gm_ratio, 3), round(r.ci_lo, 3), round(r.ci_hi, 3)] for r in big.itertuples()}
    verdict["A2_pass"] = bool((big.ci_lo > 1).all())
    verdict["slopes_median"] = slopes.groupby(["rand"]).slope.median().round(3).to_dict()

    # ---- part 2 -------------------------------------------------------------------------------
    tm = pd.read_csv(out / "timed.csv.gz")
    cal = pd.read_csv(out / "calibration.csv")
    E = np.stack([tm[f"v_{g}"].to_numpy() for g in G3], 1) - np.stack([truth[s] for s in tm.sid])
    for k, g in enumerate(G3):
        tm[f"e_{g}"] = E[:, k]
        tm[f"cov_{g}"] = np.where(tm.feasible, np.abs(E[:, k]) <= tm[f"ci_{g}"], np.nan)
    tm["over"] = tm.wall / tm.budget
    S, bt = [], {}
    for (sid, m, B, st), d in tm.groupby(["sid", "m", "budget", "strategy"]):
        row = dict(sid=sid, m=m, budget=B, strategy=st, feasible=bool(d.feasible.all()), wall=d.wall.mean(),
                   p95_over=float(d.over.quantile(0.95)), miss_1_10=float((d.over > 1.10).mean()),
                   n_per_unit=float(d.n_per_unit.median()), units=float(d.units.median()))
        if row["feasible"]:
            e = d[[f"e_{g}" for g in G3]].to_numpy()
            mse = np.maximum((e**2).mean(0) - rse[sid] ** 2, 1e-300) / scale[sid] ** 2
            row.update(loss=float(mse.mean()), wn=float(mse.mean()) * row["wall"],
                       **{f"cov_{g}": float(d[f"cov_{g}"].mean()) for g in G3})
            idx = rng.integers(0, len(e), (NBOOT, len(e)))
            bt[(sid, B, st)] = ((e[idx] ** 2).mean(1) / scale[sid] ** 2).mean(1) * row["wall"]
        S.append(row)
    S = pd.DataFrame(S)
    S.to_csv(out / "timed_summary.csv", index=False)
    rows = []
    for (m, B), d in S.groupby(["m", "budget"]):
        for a, b in [("shift", "owen"), ("lms", "shift"), ("lms", "owen"), ("lms", "MC"), ("shift", "MC"), ("owen", "MC")]:
            sids = [s for s in d.sid.unique() if (s, B, a) in bt and (s, B, b) in bt]
            if not sids:
                rows.append(dict(m=m, budget=B, num=a, den=b, gm_ratio=np.nan, ci_lo=np.nan, ci_hi=np.nan, n_scen=0))
                continue
            lr = np.mean([math.log(d[(d.sid == s) & (d.strategy == a)].wn.iloc[0] / d[(d.sid == s) & (d.strategy == b)].wn.iloc[0]) for s in sids])
            lb = np.mean([np.log(bt[(s, B, a)] / bt[(s, B, b)]) for s in sids], axis=0)
            lo, hi = np.percentile(lb, [2.5, 97.5])
            rows.append(dict(m=m, budget=B, num=a, den=b, gm_ratio=math.exp(lr), ci_lo=math.exp(lo), ci_hi=math.exp(hi), n_scen=len(sids)))
    TR = pd.DataFrame(rows)
    TR.to_csv(out / "timed_ratios.csv", index=False)
    feas = S.groupby(["m", "budget", "strategy"]).feasible.mean().unstack()
    feas.to_csv(out / "timed_feasibility.csv")
    cov = tm[tm.feasible].groupby("strategy")[[f"cov_{g}" for g in G3]].mean()
    cov.to_csv(out / "timed_coverage.csv")
    so = TR[(TR.num == "shift") & (TR.den == "owen") & (TR.budget <= 0.020)]
    ls = TR[(TR.num == "lms") & (TR.den == "shift")]
    verdict["A3_shift_over_scipy_lms_le_20ms"] = {f"m{r.m}_{r.budget * 1e3:.0f}ms": [round(r.gm_ratio, 3), round(r.ci_lo, 3), round(r.ci_hi, 3)] if r.n_scen else "SciPy LMS infeasible" for r in so.itertuples()}
    verdict["A3_pass"] = bool(((so.ci_hi < 1) | (so.n_scen == 0)).all())
    verdict["A4_lms_over_shift"] = {f"m{r.m}_{r.budget * 1e3:.0f}ms": [round(r.gm_ratio, 3), round(r.ci_lo, 3), round(r.ci_hi, 3)] for r in ls.itertuples() if r.n_scen}
    verdict["A4_gm_over_cells"] = gm(ls.gm_ratio.dropna())
    verdict["A4_pass"] = bool((ls.dropna(subset=["gm_ratio"]).ci_hi <= 1.1).all() and verdict["A4_gm_over_cells"] <= 1.0)
    rq = S[S.strategy.isin(RANDS) & S.feasible]
    verdict["A5_coverage"] = cov.round(4).to_dict(orient="index")
    verdict["A5_min_pooled_coverage_rqmc"] = float(cov.loc[[r for r in RANDS if r in cov.index]].min().min())
    verdict["A5_max_cell_miss_share"] = float(S[S.feasible].miss_1_10.max())
    verdict["A5_pass"] = bool(verdict["A5_min_pooled_coverage_rqmc"] >= 0.93 and verdict["A5_max_cell_miss_share"] <= 0.05)
    verdict["infeasible_cells"] = {f"{st}": int((S[S.strategy == st].feasible == False).sum()) for st in ["MC"] + RANDS}  # noqa: E712
    verdict["cost_ms"] = {f"m{r.m}": {"a_scipy": round(r.owen_a * 1e3, 4), "a_lms": round(r.lms_a * 1e3, 4), "a_shift": round(r.shift_a * 1e3, 4),
                                      "b_us_scipy": round(r.owen_b * 1e6, 3), "b_us_lms": round(r.lms_b * 1e6, 3), "b_us_shift": round(r.shift_b * 1e6, 3)}
                          for r in cal.itertuples()}
    verdict["max_abs_reference_check_z"] = float(np.abs(refs[[f"check_z_{g}" for g in G3]].to_numpy()).max())
    (out / "verdict.json").write_text(json.dumps(verdict, indent=2, default=float))
    log(json.dumps(verdict, default=float))
    figures(out, V, S, cal)


def figures(out, V, S, cal):
    fd = out / "figures"
    fd.mkdir(exist_ok=True)
    ms = sorted(V.m.unique())
    fig, axes = plt.subplots(1, len(ms), figsize=(4.3 * len(ms), 3.7), squeeze=False, sharey=False)
    for ax, m in zip(axes[0], ms):
        for rand in RANDS:
            d = V[(V.m == m) & (V.rand == rand)].groupby("log2n").loss.apply(gm)
            ax.loglog(2.0 ** d.index, d.values, marker="o", ms=3.5, color=COL[rand], label=LABEL[rand])
        ax.set_title(f"m = {m}"); ax.set_xlabel("points per replicate N")
    axes[0, 0].set_ylabel("risk-vector loss per replicate\n(variance / plain-MC per-path variance; geo-mean)")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fd / "fig6A1_equal_n.png", bbox_inches="tight"); fig.savefig(fd / "fig6A1_equal_n.pdf", bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, len(ms), figsize=(4.3 * len(ms), 3.7), squeeze=False)
    for ax, m in zip(axes[0], ms):
        for st in ["MC"] + RANDS:
            d = S[(S.m == m) & (S.strategy == st) & S.feasible].groupby("budget").agg(wall=("wall", "mean"), loss=("loss", gm)).reset_index()
            ax.loglog(d.wall * 1e3, d.loss, marker="o", ms=3.5, color=COL[st], label=LABEL[st])
        ax.set_title(f"m = {m}"); ax.set_xlabel("mean measured wall time [ms]")
    axes[0, 0].set_ylabel("risk-vector loss (geo-mean over scenarios)\nmissing point: plan infeasible")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fd / "fig6A2_equal_time.png", bbox_inches="tight"); fig.savefig(fd / "fig6A2_equal_time.pdf", bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    x = np.arange(len(cal))
    for k, r in enumerate(RANDS):
        axes[0].bar(x + (k - 1) * 0.27, cal[f"{r}_a"] * 1e3, 0.27, color=COL[r], label=LABEL[r])
        axes[1].bar(x + (k - 1) * 0.27, cal[f"{r}_b"] * 1e6, 0.27, color=COL[r], label=LABEL[r])
    for ax, t in zip(axes, ("set-up per replicate a [ms]", "cost per point b [µs]")):
        ax.set_xticks(x, [f"m = {m}" for m in cal.m]); ax.set_ylabel(t)
    axes[0].set_yscale("log"); axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fd / "fig6A3_costs.png", bbox_inches="tight"); fig.savefig(fd / "fig6A3_costs.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="holdout", choices=list(CFG["configurations"]))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--analysis-only", action="store_true")
    a = ap.parse_args()
    tag = os.environ.get("PHASE6_ENV_TAG")
    name = "quick" if a.quick else a.config + (f"-{tag}" if tag else "")
    out = RESULTS / "phase6_shift" / name
    t0 = time.time()
    if not a.analysis_only:
        simulate(out, a.config, a.quick)
        meta = json.loads((out / "meta.json").read_text())
        meta["wall_seconds_total"] = time.time() - t0
        (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    analyse(out)

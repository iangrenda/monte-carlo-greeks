"""Phase 6B: can a deterministic (training-free) joint rotation replace gradient-PCA training?

    python scripts/phase6_training_free_study.py            # holdout, ~15 min
    python scripts/phase6_training_free_study.py --quick    # pipeline check only (NOT evidence)
    python scripts/phase6_training_free_study.py --analysis-only

Protocol: configs/phase6_training_free.json (committed before the run; the primary candidate
was chosen on development data, results/phase6_training_free/dev/).
Output: results/phase6_training_free/<run>/.
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
from scipy import special
from scipy.stats import qmc

from mc_greeks import budget_policy as bp
from mc_greeks import joint_greeks as jg
from mc_greeks import training_free as tf
from mc_greeks import visualization as viz  # noqa: F401
from mc_greeks.asian import AsianDigital
from mc_greeks.asian_qmc import increments_from_residual, mc_block, reference_estimates
from mc_greeks.config import RESULTS, ROOT
from mc_greeks.models import BlackScholesParams
from mc_greeks.simulation import make_generator
from phase5_frontier_study import provenance

CFG = json.loads((ROOT / "configs" / "phase6_training_free.json").read_text())
SH = json.loads((ROOT / "configs" / "phase6_shift_holdout.json").read_text())
SCH = CFG["timed"]["scheduler"]
G3 = list(jg.GREEKS)
PRIMARY = CFG["primary"]
TF_KINDS = CFG["fixed_n"]["training_free"]
EPS = np.finfo(float).eps
NBOOT = 2000


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def scenarios(quick):
    g = SH["grid"]
    out, i = [], 0
    for m in g["m"]:
        for k in g["moneyness_K_over_S0"]:
            for s in g["sigma"]:
                for T in g["T"]:
                    out.append(dict(sid=i, m=m, moneyness=k, sigma=s, T=T, r=g["r"], S0=g["S0"]))
                    i += 1
    return out[::8] if quick else out


def objs(sc):
    return (AsianDigital(sc["moneyness"] * sc["S0"], int(sc["m"])),
            BlackScholesParams(sc["S0"], sc["r"], sc["sigma"], sc["T"]))


def scramble_means(f, p, L, n, R, rng):
    dt = p.T / f.m
    out = np.empty((R, 3))
    for r in range(R):
        xi = special.ndtri(np.clip(qmc.Sobol(d=f.m - 1, scramble=True, seed=rng).random(n), EPS, 1 - EPS))
        out[r] = jg.cpw_first_joint(f, p, increments_from_residual(xi @ L.T, dt)).mean(axis=0)
    return out


def setup_rotation(kind, f, p, rng):
    """Build a rotation inside the caller's timed region. Returns U (None = path PCA)."""
    if kind == "pca":
        return None
    if kind == "trained-256":
        return jg.train_rotation(f, p, 256, rng)[0]
    return tf.training_free_rotation(kind.removeprefix("tf-"), f, p)[0]


def simulate(out, quick):
    out.mkdir(parents=True, exist_ok=True)
    seed = CFG["seed"]
    scen = scenarios(quick)
    pd.DataFrame(scen).to_csv(out / "scenarios.csv", index=False)
    ms = sorted({s["m"] for s in scen})
    meta = {"provenance": provenance("configs/phase6_training_free.json"), "quick": quick}

    log("references and scales")
    refs = []
    for sc in scen:
        f, p = objs(sc)
        pts = (2**18 if f.m <= 24 else 2**16) if not quick else 2**12
        r = reference_estimates(f, p, make_generator([seed, 1, sc["sid"]]), pts, 32 if not quick else 16,
                                check_paths=2**20 if not quick else 2**16)
        s_ = jg.risk_scales(f, p, make_generator([seed, 9, sc["sid"]]), 1 << 16)
        refs.append({"sid": sc["sid"], **r, **{f"scale_{g}": s_[k] for k, g in enumerate(G3)}})
    pd.DataFrame(refs).to_csv(out / "references.csv", index=False)

    log("rotations, diagnostics and set-up times")
    fx = CFG["fixed_n"]
    n_inst = 1 if quick else fx["trained_instances"]
    rot, diag, cost = {}, [], []
    for sc in scen:
        f, p = objs(sc)
        Ghi = jg.gradient_covariances(f, p, fx["oracle_n_train"] if not quick else 2048, make_generator([seed, 2, sc["sid"]]))
        Uhi, _ = jg.joint_rotation(Ghi)
        Gj = sum(Ghi[g] / np.trace(Ghi[g]) for g in G3) / 3
        ev, EV = np.linalg.eigh(Gj)
        lam = ev[::-1]
        rot[(sc["sid"], "trained-oracle", 0)] = Uhi
        for i in range(n_inst):
            rot[(sc["sid"], "trained-256", i)] = jg.train_rotation(f, p, 256, make_generator([seed, 3, sc["sid"], i]))[0]
        for kd in TF_KINDS:
            U, _ = tf.training_free_rotation(kd, f, p)
            rot[(sc["sid"], f"tf-{kd}", 0)] = U
            cap = np.diag(U.T @ Gj @ U)
            diag.append(dict(sid=sc["sid"], m=f.m, kind=kd, cos_top=abs(U[:, 0] @ Uhi[:, 0]),
                             captured_k1=cap[0] / lam[0], cum_capture_k3=cap[:3].sum() / lam[:3].sum(),
                             cum_capture_k10=cap[:10].sum() / lam[:10].sum()))
        for kd in ["trained-256"] + [f"tf-{k}" for k in TF_KINDS]:
            ts = []
            for j in range(11):
                rng = make_generator([seed, 4, sc["sid"], j])
                t0 = time.perf_counter()
                setup_rotation(kd, f, p, rng)
                ts.append(time.perf_counter() - t0)
            cost.append(dict(sid=sc["sid"], m=f.m, kind=kd, setup_ms=float(np.median(ts[1:])) * 1e3))
    pd.DataFrame(diag).to_csv(out / "diagnostics.csv", index=False)
    pd.DataFrame(cost).to_csv(out / "setup_costs.csv", index=False)

    log("part 1: fixed N")
    R1 = 8 if quick else fx["scrambles"]
    rows = []
    for sc in scen:
        f, p = objs(sc)
        dt = p.T / f.m
        for lg in (fx["log2n"][:1] if quick else fx["log2n"]):
            for (sid, kind, i), U in [(k, v) for k, v in rot.items() if k[0] == sc["sid"]] + [((sc["sid"], "pca", 0), None)]:
                k_i = sorted({"pca", "trained-oracle", "trained-256", *[f"tf-{k}" for k in TF_KINDS]}).index(kind)
                mu = scramble_means(f, p, jg.rotated_factor(U, f.m, dt), 2**lg, R1, make_generator([seed, 5, sid, lg, k_i, i]))
                rows += [dict(sid=sid, m=f.m, log2n=lg, kind=kind, inst=i, scr=r, **dict(zip(G3, mu[r]))) for r in range(R1)]
        log(f"  scenario {sc['sid']} done")
    pd.DataFrame(rows).to_csv(out / "fixed_n.csv.gz", index=False)

    log("part 2: one-off valuation under a wall-clock budget (set-up charged)")
    tcfg = CFG["timed"]
    budgets = tcfg["budgets_seconds"][1:3] if quick else tcfg["budgets_seconds"]
    pd.DataFrame(timed_part(scen, budgets, quick, seed, list(range(len(budgets))))).to_csv(out / "timed.csv.gz", index=False)
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))


def timed_part(scen, budgets, quick, seed, budget_index, seed_offset=0):
    """Part 2. budget_index gives each budget's index in the protocol (seeds depend on it);
    seed_offset != 0 gives fresh random streams (used for a declared re-run)."""
    tcfg = CFG["timed"]
    ms = sorted({s["m"] for s in scen})
    cms = {m: bp.calibrate_cost_model(m, "joint", seed=seed + 11 + m + seed_offset, repeats=3 if quick else 7,
                                      randomisations=(SCH["randomisation"],)) for m in ms}
    reps = 4 if quick else tcfg["replications"]
    strategies = tcfg["strategies"]
    rows = []
    for b_i, B in zip(budget_index, budgets):
        for sc in scen:
            f, p = objs(sc)
            dt = p.T / f.m
            ev_ = bp.make_evaluator(f, p, "joint")
            cm = cms[f.m]
            for r in range(reps):
                for s_i, st in enumerate(strategies):
                    rng = make_generator([seed, 6 + seed_offset, b_i, sc["sid"], s_i, r])
                    t0 = time.perf_counter()
                    if st == "MC":
                        pl = bp.mc_schedule(B, cm, block_max=mc_block(f.m), margin=SCH["margin"])
                        L, setup = None, 0.0
                    else:
                        U = setup_rotation(st, f, p, rng)
                        L = jg.rotated_factor(U, f.m, dt)
                        setup = time.perf_counter() - t0
                        pl = bp.optimise_schedule(B - setup, cm, SCH["randomisation"], SCH["alpha"], SCH["objective"], SCH["r_min"],
                                                  margin=SCH["margin"]) if B > setup else bp.Plan("infeasible", None, 0, 0, math.nan, math.nan, False)
                        if not pl.feasible:   # fall back to MC with whatever time is left
                            pl = bp.mc_schedule(max(B - setup, 0.0), cm, block_max=mc_block(f.m), margin=SCH["margin"])
                    e = bp.execute(pl, ev_, 3, L, f.m, dt, B, rng, t0=t0)
                    rows.append(dict(sid=sc["sid"], m=f.m, budget=B, strategy=st, rep=r, feasible=e.feasible, method=e.method,
                                     setup=setup, wall=e.wall, units=e.n_units, n_per_unit=e.n_per_unit,
                                     **{f"v_{g}": e.value[k] for k, g in enumerate(G3)},
                                     **{f"ci_{g}": e.ci_halfwidth[k] for k, g in enumerate(G3)}))
        log(f"  budget {B * 1e3:.0f} ms done")
    return rows


def rerun_timed(out, budgets):
    """Declared re-run of whole budget blocks of part 2 (fresh calibration and random streams).
    Writes timed_rerun.csv.gz and a note; the original timed.csv.gz is kept unchanged."""
    scen = pd.read_csv(out / "scenarios.csv").to_dict(orient="records")
    allb = CFG["timed"]["budgets_seconds"]
    idx = [allb.index(B) for B in budgets]
    t0 = time.time()
    rows = timed_part(scen, budgets, False, CFG["seed"], idx, seed_offset=100)
    pd.DataFrame(rows).to_csv(out / "timed_rerun.csv.gz", index=False)
    (out / "timed_rerun_meta.json").write_text(json.dumps({"provenance": provenance("configs/phase6_training_free.json"),
        "budgets_rerun": budgets, "wall_seconds": time.time() - t0,
        "reason": "robustness re-run of the 50 ms block. During the original block the computer slept inside one run (wall/budget = 1.3e5; its estimate is still valid). The re-run (fresh calibration, fresh streams, seed offset 100, machine held awake) reproduced the other anomalies (set-up about 3x the idle cost, 15-32% of RQMC runs over 1.10 x budget), so they are systematic, not caused by the sleep. The primary analysis uses the original pre-registered data; the re-run is reported as a robustness check."},
        indent=2, default=str))


def gm(v):
    v = np.asarray(v, float)
    return float(np.exp(np.log(v).mean())) if len(v) and np.all(v > 0) and np.all(np.isfinite(v)) else float("nan")


def timed_tables(tm, truth, rse, scale, rng):
    """Per-cell summaries and paired ratio table for part 2. Runs without an estimate (set-up
    left no feasible schedule) are counted in n_infeasible and excluded from the loss."""
    tm = tm.copy()
    E = np.stack([tm[f"v_{g}"].to_numpy() for g in G3], 1) - np.stack([truth[s] for s in tm.sid])
    for k, g in enumerate(G3):
        tm[f"e_{g}"] = E[:, k]
        tm[f"cov_{g}"] = np.where(tm.feasible, np.abs(E[:, k]) <= tm[f"ci_{g}"], np.nan)
    tm["over"] = tm.wall / tm.budget
    S, bt = [], {}
    for (sid, m, B, st), d in tm.groupby(["sid", "m", "budget", "strategy"]):
        ok = d[d.feasible.astype(bool)]
        e = ok[[f"e_{g}" for g in G3]].to_numpy()
        mse = np.maximum((e**2).mean(0) - rse[sid] ** 2, 1e-300) / scale[sid] ** 2
        S.append(dict(sid=sid, m=m, budget=B, strategy=st, loss=float(mse.mean()), wall=d.wall.mean(), setup_ms=d.setup.median() * 1e3,
                      method="|".join(sorted(set(d.method))), p95_over=float(d.over.quantile(0.95)), miss=float((d.over > 1.10).mean()),
                      n_infeasible=int((~d.feasible.astype(bool)).sum()), **{f"cov_{g}": float(ok[f"cov_{g}"].mean()) for g in G3}))
        idx = rng.integers(0, len(e), (NBOOT, len(e)))
        bt[(sid, B, st)] = ((e[idx] ** 2).mean(1) / scale[sid] ** 2).mean(1)
    S = pd.DataFrame(S)
    rows = []
    for (m, B), d in S.groupby(["m", "budget"]):
        sids = sorted(d.sid.unique())
        for a, b in [(f"tf-{PRIMARY}", "trained-256"), (f"tf-{PRIMARY}", "pca"), ("trained-256", "pca"), ("tf-ones", "pca"), ("pca", "MC")]:
            lr = np.mean([math.log(d[(d.sid == s) & (d.strategy == a)].loss.iloc[0] / d[(d.sid == s) & (d.strategy == b)].loss.iloc[0]) for s in sids])
            lb = np.mean([np.log(bt[(s, B, a)] / bt[(s, B, b)]) for s in sids], axis=0)
            lo, hi = np.percentile(lb, [2.5, 97.5])
            rows.append(dict(m=m, budget=B, num=a, den=b, gm_ratio=math.exp(lr), ci_lo=math.exp(lo), ci_hi=math.exp(hi)))
    return S, pd.DataFrame(rows), tm


def analyse(out):
    refs = pd.read_csv(out / "references.csv").set_index("sid")
    scale = {s: refs.loc[s, [f"scale_{g}" for g in G3]].to_numpy(float) for s in refs.index}
    truth = {s: refs.loc[s, [f"ref_{g}" for g in G3]].to_numpy(float) for s in refs.index}
    rse = {s: refs.loc[s, [f"ref_se_{g}" for g in G3]].to_numpy(float) for s in refs.index}
    rng = np.random.default_rng(CFG["seed"] + 99)
    verdict = {}

    fx = pd.read_csv(out / "fixed_n.csv.gz")
    L_, B_ = {}, {}
    for (sid, lg, kind), d in fx.groupby(["sid", "log2n", "kind"]):
        X = d[G3].to_numpy() - d.groupby("inst")[G3].transform("mean").to_numpy()
        dof = len(d) - d.inst.nunique()
        v = (X**2).sum(0) / dof / scale[sid] ** 2
        L_[(sid, lg, kind)] = (float(v.mean()), v)
        # bootstrap: resample scramblings within instance
        parts = []
        for _, x in d.groupby("inst"):
            Y = x[G3].to_numpy()
            idx = rng.integers(0, len(Y), (NBOOT, len(Y)))
            parts.append((Y[idx].var(axis=1, ddof=1) * (len(Y) - 1), len(Y) - 1))
        B_[(sid, lg, kind)] = (sum(p_[0] for p_ in parts) / sum(p_[1] for p_ in parts) / scale[sid] ** 2).mean(1)
    rows = []
    kinds = sorted(fx.kind.unique())
    msid = fx.groupby("sid").m.first()
    for den in ("trained-oracle", "trained-256", "pca"):
        for (m, lg), _ in fx.groupby(["m", "log2n"]):
            sids = sorted(msid[msid == m].index)
            for kind in kinds:
                if kind == den:
                    continue
                lr = np.mean([math.log(L_[(s, lg, kind)][0] / L_[(s, lg, den)][0]) for s in sids])
                lb = np.mean([np.log(B_[(s, lg, kind)] / B_[(s, lg, den)]) for s in sids], axis=0)
                lo, hi = np.percentile(lb, [2.5, 97.5])
                per = {f"ratio_{g}": gm([L_[(s, lg, kind)][1][k] / L_[(s, lg, den)][1][k] for s in sids]) for k, g in enumerate(G3)}
                rows.append(dict(m=m, log2n=lg, kind=kind, den=den, gm_ratio=math.exp(lr), ci_lo=math.exp(lo), ci_hi=math.exp(hi), **per))
    FR = pd.DataFrame(rows)
    FR.to_csv(out / "fixed_n_ratios.csv", index=False)
    # pooled over N per m (geometric mean of the per-N ratios; CI from the averaged bootstrap)
    pooled = []
    for (m, kind, den), d in FR.groupby(["m", "kind", "den"]):
        sids = sorted(msid[msid == m].index)
        lgs = sorted(d.log2n.unique())
        lb = np.mean([np.log(B_[(s, lg, kind)] / B_[(s, lg, den)]) for s in sids for lg in lgs], axis=0)
        lo, hi = np.percentile(lb, [2.5, 97.5])
        pooled.append(dict(m=m, kind=kind, den=den, gm_ratio=gm(d.gm_ratio), ci_lo=math.exp(lo), ci_hi=math.exp(hi)))
    PO = pd.DataFrame(pooled)
    PO.to_csv(out / "fixed_n_pooled.csv", index=False)
    pr = PO[(PO.kind == f"tf-{PRIMARY}")]
    b1 = pr[pr.den == "trained-oracle"]
    verdict["B1_primary_over_oracle_by_m"] = {int(r.m): [round(r.gm_ratio, 3), round(r.ci_lo, 3), round(r.ci_hi, 3)] for r in b1.itertuples()}
    verdict["B1_pass"] = bool(((b1.gm_ratio <= 1.10) & (b1.ci_hi <= 1.25)).all())
    b2 = pr[pr.den == "trained-256"]
    verdict["B2_primary_over_trained256_by_m"] = {int(r.m): [round(r.gm_ratio, 3), round(r.ci_lo, 3), round(r.ci_hi, 3)] for r in b2.itertuples()}
    verdict["B2_pass"] = bool((b2.gm_ratio < 1).all())
    b4 = PO[(PO.kind == "tf-ones") & (PO.den == "trained-oracle")]
    verdict["B4_ones_over_oracle_by_m"] = {int(r.m): [round(r.gm_ratio, 3), round(r.ci_lo, 3), round(r.ci_hi, 3)] for r in b4.itertuples()}
    verdict["B4_pass"] = bool((b4.ci_lo > 1.25).all())
    verdict["all_vs_pca_by_m"] = {f"{r.kind}_m{r.m}": round(r.gm_ratio, 3) for r in PO[PO.den == "pca"].itertuples()}
    verdict["all_vs_oracle_by_m"] = {f"{r.kind}_m{r.m}": round(r.gm_ratio, 3) for r in PO[PO.den == "trained-oracle"].itertuples()}
    dg = pd.read_csv(out / "diagnostics.csv")
    verdict["diagnostics_min"] = dg.groupby("kind")[["cos_top", "captured_k1", "cum_capture_k3", "cum_capture_k10"]].min().round(5).to_dict(orient="index")
    sc_ = pd.read_csv(out / "setup_costs.csv")
    verdict["setup_ms_median"] = sc_.groupby(["m", "kind"]).setup_ms.median().round(3).unstack().to_dict(orient="index")

    tm = pd.read_csv(out / "timed.csv.gz")
    S, TR, tm = timed_tables(tm, truth, rse, scale, rng)
    S.to_csv(out / "timed_summary.csv", index=False)
    TR.to_csv(out / "timed_ratios.csv", index=False)
    b3 = TR[(TR.num == f"tf-{PRIMARY}") & (TR.den == "trained-256")]
    verdict["B3_primary_over_trained256_one_off"] = {f"m{r.m}_{r.budget * 1e3:.0f}ms": [round(r.gm_ratio, 3), round(r.ci_lo, 3), round(r.ci_hi, 3)] for r in b3.itertuples()}
    verdict["B3_pass"] = bool((b3.gm_ratio <= 1.0).all())
    b3p = TR[(TR.num == f"tf-{PRIMARY}") & (TR.den == "pca")]
    verdict["B3_primary_over_pca_one_off"] = {f"m{r.m}_{r.budget * 1e3:.0f}ms": [round(r.gm_ratio, 3), round(r.ci_lo, 3), round(r.ci_hi, 3)] for r in b3p.itertuples()}
    agg = S.groupby(["strategy", "m", "budget"]).n_infeasible.sum()
    verdict["timed_runs_without_estimate"] = {f"{k[0]}_m{k[1]}_{k[2] * 1e3:.0f}ms": int(v) for k, v in agg[agg > 0].items()}
    fb = tm[tm.method == "MC"].groupby(["strategy", "m", "budget"]).size()
    verdict["timed_runs_fallen_back_to_MC"] = {f"{k[0]}_m{k[1]}_{k[2] * 1e3:.0f}ms": int(v) for k, v in fb.items() if k[0] != "MC"}
    if (out / "timed_rerun.csv.gz").exists():
        rr = pd.read_csv(out / "timed_rerun.csv.gz")
        S2, TR2, _ = timed_tables(pd.concat([tm[~tm.budget.isin(rr.budget.unique())][[c for c in rr.columns]], rr], ignore_index=True),
                                  truth, rse, scale, rng)
        TR2.to_csv(out / "timed_ratios_with_rerun.csv", index=False)
        S2.to_csv(out / "timed_summary_with_rerun.csv", index=False)
        x = TR2[(TR2.num == f"tf-{PRIMARY}") & (TR2.den == "trained-256") & TR2.budget.isin(rr.budget.unique())]
        verdict["robustness_rerun_B3"] = {f"m{r.m}_{r.budget * 1e3:.0f}ms": [round(r.gm_ratio, 3), round(r.ci_lo, 3), round(r.ci_hi, 3)] for r in x.itertuples()}
        verdict["robustness_rerun_max_miss_share"] = float(S2[S2.budget.isin(rr.budget.unique())].miss.max())
    verdict["timed_coverage"] = tm[tm.feasible].groupby("strategy")[[f"cov_{g}" for g in G3]].mean().round(4).to_dict(orient="index")
    verdict["timed_max_miss_share"] = float(S.miss.max())
    verdict["timed_methods"] = S.groupby(["strategy"]).method.agg(lambda v: "|".join(sorted(set("|".join(v).split("|"))))).to_dict()
    verdict["max_abs_reference_check_z"] = float(np.abs(refs[[f"check_z_{g}" for g in G3]].to_numpy()).max())
    (out / "verdict.json").write_text(json.dumps(verdict, indent=2, default=float))
    log(json.dumps(verdict, default=float))
    figures(out, FR, S)


def figures(out, FR, S):
    fd = out / "figures"
    fd.mkdir(exist_ok=True)
    ms = sorted(FR.m.unique())
    d = FR[FR.den == "trained-oracle"]
    kinds = [k for k in ["pca", "trained-256"] + [f"tf-{k}" for k in TF_KINDS] if k in set(d.kind)]
    fig, axes = plt.subplots(1, len(ms), figsize=(4.6 * len(ms), 3.8), squeeze=False, sharey=True)
    for ax, m in zip(axes[0], ms):
        x = d[d.m == m]
        lgs = sorted(x.log2n.unique())
        for j, lg in enumerate(lgs):
            y = x[x.log2n == lg].set_index("kind")
            ax.bar(np.arange(len(kinds)) + (j - (len(lgs) - 1) / 2) * 0.25, [y.gm_ratio.get(k, np.nan) for k in kinds], 0.25, label=f"N = 2^{lg}")
        ax.axhline(1, color="#1f1f1e", lw=0.8)
        ax.set_yscale("log"); ax.set_xticks(range(len(kinds)), kinds, rotation=35, ha="right", fontsize=7); ax.set_title(f"m = {m}")
    axes[0, 0].set_ylabel("risk-vector loss / loss with the well-trained rotation\n(fixed N; geo-mean over 8 scenarios)")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fd / "fig6B1_fixed_n.png", bbox_inches="tight"); fig.savefig(fd / "fig6B1_fixed_n.pdf", bbox_inches="tight")
    plt.close(fig)
    col = {"MC": "#eb6834", "pca": "#9a9890", "trained-256": "#e34948", f"tf-{PRIMARY}": "#2a78d6", "tf-ones": "#1baf7a"}
    fig, axes = plt.subplots(1, len(ms), figsize=(4.6 * len(ms), 3.8), squeeze=False)
    for ax, m in zip(axes[0], ms):
        for st, c in col.items():
            x = S[(S.m == m) & (S.strategy == st)].groupby("budget").agg(wall=("wall", "mean"), loss=("loss", gm)).reset_index()
            ax.loglog(x.wall * 1e3, x.loss, marker="o", ms=3.5, color=c, label=st)
        ax.set_title(f"m = {m}"); ax.set_xlabel("mean measured wall time, set-up included [ms]")
    axes[0, 0].set_ylabel("risk-vector loss, one-off valuation")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fd / "fig6B2_one_off.png", bbox_inches="tight"); fig.savefig(fd / "fig6B2_one_off.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--analysis-only", action="store_true")
    ap.add_argument("--rerun-timed-budgets", type=float, nargs="+", default=None,
                    help="declared re-run of whole budget blocks of part 2 (seconds), e.g. 0.05")
    a = ap.parse_args()
    out = RESULTS / "phase6_training_free" / ("quick" if a.quick else "holdout")
    t0 = time.time()
    if a.rerun_timed_budgets:
        rerun_timed(out, a.rerun_timed_budgets)
    elif not a.analysis_only:
        simulate(out, a.quick)
        meta = json.loads((out / "meta.json").read_text())
        meta["wall_seconds_total"] = time.time() - t0
        (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    analyse(out)

"""Phase 5C + 5D: joint Delta/Gamma/Vega estimation and the economics of transform reuse.

    python scripts/phase5_joint_study.py            # full run, ~35 min
    python scripts/phase5_joint_study.py --quick    # pipeline check only (NOT evidence)
    python scripts/phase5_joint_study.py --analysis-only

Protocol: configs/phase5_joint.json (committed before the run).
Output: results/phase5_joint/<run>/.

Parts
  C1  fixed-N, reference-free: variance of every Greek under path PCA, Greek-specific,
      equal-weight joint and weighted joint rotations (same scrambled points for all Greeks).
  C2  wall-clock: one shared simulation for the vector vs three separate simulations
      sharing the same total budget.
  D1  break-even: M valuations with total time M*B; the trained rotation gets
      B - C_train/M per valuation.
  D2  transfer: a rotation trained on one contract applied to neighbouring contracts.
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
from mc_greeks import visualization as viz  # noqa: F401
from mc_greeks.asian import AsianDigital
from mc_greeks.asian_qmc import increments_from_residual, mc_block, reference_estimates
from mc_greeks.config import RESULTS, ROOT
from mc_greeks.models import BlackScholesParams
from mc_greeks.simulation import make_generator
from phase5_frontier_study import provenance

CFG = json.loads((ROOT / "configs" / "phase5_joint.json").read_text())
SCH = CFG["scheduler"]
G3 = list(jg.GREEKS)
WSETS = CFG["weight_sets"]
NBOOT = 2000
EPS = np.finfo(float).eps


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


def wdict(w):
    return dict(zip(G3, w))


def scramble_means(f, p, L, n, R, rng):
    """(R, 3) scramble means of (Delta, Gamma, Vega) from the same Owen-scrambled points."""
    dt = p.T / f.m
    out = np.empty((R, 3))
    for r in range(R):
        xi = special.ndtri(np.clip(qmc.Sobol(d=f.m - 1, scramble=True, seed=rng).random(n), EPS, 1 - EPS))
        out[r] = jg.cpw_first_joint(f, p, increments_from_residual(xi @ L.T, dt)).mean(axis=0)
    return out


def plan(kind, B, cm):
    """kind 'MC' or 'RQMC'. RQMC falls back to MC when no RQMC schedule fits; MC may be infeasible."""
    if kind == "RQMC":
        pl = bp.optimise_schedule(B, cm, SCH["randomisation"], SCH["alpha"], SCH["objective"], SCH["r_min"], margin=SCH["margin"])
        if pl.feasible:
            return pl
    return bp.mc_schedule(B, cm, block_max=mc_block(cm.m), margin=SCH["margin"]) if B > 0 else bp.Plan("infeasible", None, 0, 0, math.nan, math.nan, False)


def run_shared(kind, f, p, B, rng, cm, L):
    t0 = time.perf_counter()
    pl = plan(kind, B, cm)
    e = bp.execute(pl, bp.make_evaluator(f, p, "joint"), 3, L, f.m, p.T / f.m, B, rng, t0=t0)
    return dict(zip([f"v_{g}" for g in G3], e.value), **dict(zip([f"se_{g}" for g in G3], e.stderr)),
                **dict(zip([f"ci_{g}" for g in G3], e.ci_halfwidth)), wall=e.wall, n_paths=e.n_paths, units=e.n_units,
                n_per_unit=e.n_per_unit, method=e.method, feasible=e.feasible)


def run_separate(kind, f, p, B, rng, cms, Ls):
    """Three independent simulations, each with B/3. Ls: Greek -> construction matrix."""
    row, wall, n_paths, methods, feas = {}, 0.0, 0, [], True
    for g in G3:
        t0 = time.perf_counter()
        pl = plan(kind, B / 3.0, cms[g])
        e = bp.execute(pl, bp.make_evaluator(f, p, g), 1, Ls[g], f.m, p.T / f.m, B / 3.0, rng, t0=t0)
        row.update({f"v_{g}": float(e.value[0]), f"se_{g}": float(e.stderr[0]), f"ci_{g}": float(e.ci_halfwidth[0])})
        wall += e.wall
        n_paths += e.n_paths
        methods.append(e.method)
        feas &= e.feasible
    return dict(row, wall=wall, n_paths=n_paths, units=np.nan, n_per_unit=np.nan, method="/".join(methods), feasible=feas)


def simulate(out, quick):
    out.mkdir(parents=True, exist_ok=True)
    seed = CFG["seed"]
    scen = scenarios(quick)
    pd.DataFrame(scen).to_csv(out / "scenarios.csv", index=False)
    ms = sorted({s["m"] for s in scen})
    meta = {"provenance": provenance("configs/phase5_joint.json"), "quick": quick}
    n_train = CFG["n_train"]

    log("references and scales")
    refs = []
    for sc in scen:
        f, p = objs(sc)
        pts = (2**18 if f.m <= 24 else 2**16) if not quick else 2**12
        r = reference_estimates(f, p, make_generator([seed, 1, sc["sid"]]), pts, 32 if not quick else 16,
                                check_paths=2**20 if not quick else 2**16)
        s = jg.risk_scales(f, p, make_generator([seed, 9, sc["sid"]]), 1 << 16)
        refs.append({"sid": sc["sid"], **r, **{f"scale_{g}": s[k] for k, g in enumerate(G3)}})
    pd.DataFrame(refs).to_csv(out / "references.csv", index=False)

    log("cost models")
    t0 = time.perf_counter()
    cms = {(m, w): bp.calibrate_cost_model(m, w, seed=seed + 11 + 100 * m + i, repeats=3 if quick else 7)
           for m in ms for i, w in enumerate(G3 + ["joint"])}
    meta["calibration_seconds"] = time.perf_counter() - t0
    pd.DataFrame([dict(m=m, what=w, **{f"{r}_{n}": v for r in bp.RANDOMISATIONS for n, v in zip(("c0", "a", "b"), cm.rq[r])},
                       mc0=cm.mc[0], mc_u=cm.mc[1], mc_c=cm.mc[2], **{f"fit_err_{k}": v for k, v in cm.fit_error.items()})
                  for (m, w), cm in cms.items()]).to_csv(out / "calibration.csv", index=False)

    # rotations: n_inst independent trainings of each kind per scenario
    n_inst = 2 if quick else CFG["training_instances"]
    kinds = {"joint": None, **{f"spec-{g}": wdict([float(h == g) for h in G3]) for g in G3},
             **{f"joint-{k}": wdict(w) for k, w in WSETS.items() if k != "equal"}}
    rot, trow, spec = {}, [], []
    for sc in scen:
        f, p = objs(sc)
        for i in range(n_inst):
            for k_i, (kind, w) in enumerate(kinds.items()):
                greeks = [g for g in G3 if w is None or w[g] != 0.0]
                U, sec, Gs = jg.train_rotation(f, p, n_train, make_generator([seed, 3, sc["sid"], i, k_i]), w, greeks)
                rot[(sc["sid"], kind, i)] = U
                trow.append(dict(sid=sc["sid"], m=f.m, kind=kind, inst=i, train_seconds=sec, n_train=n_train))
                if kind == "joint":
                    tops = {g: np.linalg.eigh(Gs[g])[1][:, -1] for g in G3}
                    spec.append(dict(sid=sc["sid"], m=f.m, inst=i,
                                     **{f"top_share_{g}": jg.effective_rank(Gs[g])["top_share"] for g in G3},
                                     **{f"part_ratio_{g}": jg.effective_rank(Gs[g])["participation_ratio"] for g in G3},
                                     cos_delta_gamma=abs(tops["delta"] @ tops["gamma"]), cos_delta_vega=abs(tops["delta"] @ tops["vega"]),
                                     cos_gamma_vega=abs(tops["gamma"] @ tops["vega"]),
                                     cos_joint_pca1=abs(U[0, 0]), cos_delta_pca1=abs(tops["delta"][0])))
    pd.DataFrame(trow).to_csv(out / "training_costs.csv", index=False)
    pd.DataFrame(spec).to_csv(out / "gradient_spectra.csv", index=False)
    tmed = pd.DataFrame(trow).groupby(["sid", "kind"]).train_seconds.median()

    # --- C1: fixed N --------------------------------------------------------------------------
    log("C1 fixed-N")
    c1 = CFG["fixed_n"]
    R1 = 8 if quick else c1["scrambles_per_instance"]
    rows = []
    for sc in scen:
        f, p = objs(sc)
        dt = p.T / f.m
        for lg in (c1["log2n"][:1] if quick else c1["log2n"]):
            for k_i, kind in enumerate(["pca"] + list(kinds)):
                for i in range(n_inst):
                    L = jg.rotated_factor(None if kind == "pca" else rot[(sc["sid"], kind, i)], f.m, dt)
                    mu = scramble_means(f, p, L, 2**lg, R1, make_generator([seed, 4, sc["sid"], lg, k_i, i]))
                    rows += [dict(sid=sc["sid"], m=f.m, log2n=lg, kind=kind, inst=i, scr=r, **dict(zip(G3, mu[r]))) for r in range(R1)]
    pd.DataFrame(rows).to_csv(out / "c1_scramble_means.csv.gz", index=False)

    # --- C2: shared vs separate under a total wall-clock budget -------------------------------
    log("C2 shared vs separate (timed)")
    c2 = CFG["timed"]
    budgets = c2["total_budgets_seconds"][1:3] if quick else c2["total_budgets_seconds"]
    reps = 4 if quick else c2["replications"]
    rows = []
    strategies = ["shared-MC", "shared-PCA", "shared-GPCA", "separate-MC", "separate-PCA", "separate-GPCA"]
    for b_i, B in enumerate(budgets):
        for sc in scen:
            f, p = objs(sc)
            dt = p.T / f.m
            Lp = jg.pca_factor(f.m, dt)
            cmg = {g: cms[(f.m, g)] for g in G3}
            for r in range(reps):
                i = r % n_inst
                Lj = jg.rotated_factor(rot[(sc["sid"], "joint", i)], f.m, dt)
                Ls = {g: jg.rotated_factor(rot[(sc["sid"], f"spec-{g}", i)], f.m, dt) for g in G3}
                for s_i, st in enumerate(strategies):
                    rng = make_generator([seed, 5, b_i, sc["sid"], s_i, r])
                    kind = "MC" if st.endswith("MC") else "RQMC"
                    if st.startswith("shared"):
                        row = run_shared(kind, f, p, B, rng, cms[(f.m, "joint")], Lj if st.endswith("GPCA") else Lp)
                    else:
                        row = run_separate(kind, f, p, B, rng, cmg, Ls if st.endswith("GPCA") else {g: Lp for g in G3})
                    rows.append(dict(sid=sc["sid"], m=f.m, budget=B, strategy=st, rep=r, inst=i, **row))
        log(f"C2: total budget {B * 1e3:.0f} ms done")
    pd.DataFrame(rows).to_csv(out / "c2_runs.csv.gz", index=False)

    # --- D1: break-even ------------------------------------------------------------------------
    log("D1 break-even")
    d1 = CFG["break_even"]
    Bs = d1["per_valuation_budgets_seconds"][:1] if quick else d1["per_valuation_budgets_seconds"]
    Ms = d1["M"][::3] if quick else d1["M"]
    reps = 4 if quick else d1["replications"]
    rows = []
    for b_i, B in enumerate(Bs):
        for sc in scen:
            f, p = objs(sc)
            dt = p.T / f.m
            Lp = jg.pca_factor(f.m, dt)
            cmg = {g: cms[(f.m, g)] for g in G3}
            cj = float(tmed[(sc["sid"], "joint")])
            cs = float(sum(tmed[(sc["sid"], f"spec-{g}")] for g in G3))
            for r in range(reps):
                i = r % n_inst
                Lj = jg.rotated_factor(rot[(sc["sid"], "joint", i)], f.m, dt)
                Ls = {g: jg.rotated_factor(rot[(sc["sid"], f"spec-{g}", i)], f.m, dt) for g in G3}
                for s_i, (st, kind) in enumerate((("MC", "MC"), ("PCA", "RQMC"))):
                    row = run_shared(kind, f, p, B, make_generator([seed, 6, b_i, sc["sid"], s_i, r]), cms[(f.m, "joint")], Lp)
                    rows.append(dict(sid=sc["sid"], m=f.m, budget=B, strategy=st, M=np.inf, rep=r, train_charge=0.0, sim_budget=B, **row))
                for M_i, M in enumerate(Ms):
                    for s_i, (st, c) in enumerate((("GPCA-joint", cj), ("GPCA-separate", cs))):
                        charge = 0.0 if M is None else c / M
                        Bs_ = B - charge
                        rng = make_generator([seed, 7, b_i, sc["sid"], M_i, s_i, r])
                        if Bs_ <= 0:
                            row = dict({f"v_{g}": np.nan for g in G3}, wall=0.0, n_paths=0, units=0, n_per_unit=0, method="infeasible", feasible=False)
                        elif st == "GPCA-joint":
                            row = run_shared("RQMC", f, p, Bs_, rng, cms[(f.m, "joint")], Lj)
                        else:
                            row = run_separate("RQMC", f, p, Bs_, rng, cmg, Ls)
                        rows.append(dict(sid=sc["sid"], m=f.m, budget=B, strategy=st, M=np.inf if M is None else M, rep=r,
                                         train_charge=charge, sim_budget=Bs_, **row))
        log(f"D1: per-valuation budget {B * 1e3:.0f} ms done")
    pd.DataFrame(rows).to_csv(out / "d1_runs.csv.gz", index=False)

    # --- D2: transfer --------------------------------------------------------------------------
    log("D2 transfer")
    d2 = CFG["transfer"]
    R2 = 8 if quick else d2["scrambles_per_instance"]
    n2 = 2 ** d2["log2n"]
    rows, arow = [], []
    bases = {}
    for sc in scen:
        bases.setdefault(sc["m"], sc)
    targets = ([("K", x) for x in d2["K_factors"]] + [("sigma", x) for x in d2["sigma_factors"]] + [("T", x) for x in d2["T_factors"]]
               + [("regime", tuple(x)) for x in d2["regime_K_sigma_factors"]])
    if quick:
        targets = targets[::5]
    for base in bases.values():
        fb, pb = objs(base)
        Ub = [rot[(base["sid"], "joint", i)] for i in range(n_inst)]
        for t_i, (axis, x) in enumerate(targets):
            kf, sf, Tf = (x, 1, 1) if axis == "K" else (1, x, 1) if axis == "sigma" else (1, 1, x) if axis == "T" else (x[0], x[1], 1)
            ft = AsianDigital(fb.K * kf, fb.m)
            pt = BlackScholesParams(pb.S0, pb.r, pb.sigma * sf, pb.T * Tf)
            dt = pt.T / ft.m
            scale = jg.risk_scales(ft, pt, make_generator([seed, 8, base["sid"], t_i, 0]), 1 << 16)
            fresh = [jg.train_rotation(ft, pt, n_train, make_generator([seed, 8, base["sid"], t_i, 1, i]))[0] for i in range(n_inst)]
            for i in range(n_inst):
                arow.append(dict(base_sid=base["sid"], m=fb.m, axis=axis, factor=str(x), inst=i,
                                 **{f"align_k{k}": jg.top_alignment(Ub[i], fresh[i], k) for k in (1, 2, 3)}))
            for k_i, (kind, Us) in enumerate((("pca", [None] * n_inst), ("transferred", Ub), ("fresh", fresh))):
                for i in range(n_inst):
                    mu = scramble_means(ft, pt, jg.rotated_factor(Us[i], ft.m, dt), n2, R2, make_generator([seed, 8, base["sid"], t_i, 2, k_i, i]))
                    rows += [dict(base_sid=base["sid"], m=fb.m, axis=axis, factor=str(x), kind=kind, inst=i, scr=r,
                                  **dict(zip(G3, mu[r])), **{f"scale_{g}": scale[j] for j, g in enumerate(G3)}) for r in range(R2)]
    pd.DataFrame(rows).to_csv(out / "d2_scramble_means.csv.gz", index=False)
    pd.DataFrame(arow).to_csv(out / "d2_alignment.csv", index=False)
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))


# ------------------------------------------------------------------------------------------------

def pooled_var(d):
    """Within-instance variance of scramble means per Greek (3,), and the pooled covariance (3, 3)."""
    X = d[G3].to_numpy() - d.groupby("inst")[G3].transform("mean").to_numpy()
    dof = len(d) - d.inst.nunique()
    C = X.T @ X / dof
    return np.diag(C), C


def boot_loss_ratio(dn, dd, scales, w, rng, n=NBOOT):
    """Bootstrap (scramblings resampled within instance) of loss_num / loss_den."""
    def losses(d):
        out = np.zeros(n)
        tot = 0
        for _, x in d.groupby("inst"):
            v = x[G3].to_numpy()
            idx = rng.integers(0, len(v), (n, len(v)))
            out += (v[idx].var(axis=1, ddof=1) / scales**2 * w).sum(axis=1) * (len(v) - 1)
            tot += len(v) - 1
        return out / tot
    r = losses(dn) / losses(dd)
    return np.percentile(r, [2.5, 97.5])


def gm(v):
    v = np.asarray(v, float)
    return float(np.exp(np.log(v).mean())) if len(v) and np.all(np.isfinite(v)) and np.all(v > 0) else float("nan")


def analyse(out):
    refs = pd.read_csv(out / "references.csv").set_index("sid")
    rng = np.random.default_rng(CFG["seed"] + 99)
    scale = {sid: refs.loc[sid, [f"scale_{g}" for g in G3]].to_numpy(float) for sid in refs.index}
    truth = {sid: refs.loc[sid, [f"ref_{g}" for g in G3]].to_numpy(float) for sid in refs.index}
    rse = {sid: refs.loc[sid, [f"ref_se_{g}" for g in G3]].to_numpy(float) for sid in refs.index}
    wsets = {k: np.array(v, float) for k, v in WSETS.items()}
    headline = {}

    # --- C1 ---------------------------------------------------------------------------------
    c1 = pd.read_csv(out / "c1_scramble_means.csv.gz")
    rows, corr = [], []
    for (sid, m, lg), d in c1.groupby(["sid", "m", "log2n"]):
        base = d[d.kind == "pca"]
        vb, _ = pooled_var(base)
        for kind, x in d.groupby("kind"):
            v, C = pooled_var(x)
            sd = np.sqrt(v)
            corr.append(dict(sid=sid, m=m, log2n=lg, kind=kind, corr_delta_gamma=C[0, 1] / sd[0] / sd[1],
                             corr_delta_vega=C[0, 2] / sd[0] / sd[2], corr_gamma_vega=C[1, 2] / sd[1] / sd[2]))
            row = dict(sid=sid, m=m, log2n=lg, kind=kind, **{f"var_ratio_{g}": v[k] / vb[k] for k, g in enumerate(G3)})
            for wn, w in wsets.items():
                row[f"loss_{wn}"] = float((w * v / scale[sid] ** 2).sum())
                lo, hi = boot_loss_ratio(x, base, scale[sid], w, rng, 500)
                row[f"loss_ratio_vs_pca_{wn}"] = row[f"loss_{wn}"] / float((w * vb / scale[sid] ** 2).sum())
                row[f"lo_{wn}"], row[f"hi_{wn}"] = lo, hi
            rows.append(row)
    c1s = pd.DataFrame(rows)
    c1s.to_csv(out / "c1_fixed_n.csv", index=False)
    pd.DataFrame(corr).to_csv(out / "c1_error_correlations.csv", index=False)
    agg = c1s.groupby(["m", "log2n", "kind"]).agg(**{c: (c, gm) for c in c1s.columns if c.startswith(("var_ratio_", "loss_ratio_vs_pca_"))}).reset_index()
    agg.to_csv(out / "c1_fixed_n_gm.csv", index=False)
    # weight mismatch: loss under weights w of the rotation trained for w' relative to the equal-weight joint rotation
    mm = []
    for (sid, m, lg), d in c1s.groupby(["sid", "m", "log2n"]):
        d = d.set_index("kind")
        for wn in wsets:
            for kind in d.index:
                mm.append(dict(sid=sid, m=m, log2n=lg, eval_weights=wn, kind=kind, loss_over_equal_joint=d.loc[kind, f"loss_{wn}"] / d.loc["joint", f"loss_{wn}"]))
    mm = pd.DataFrame(mm).groupby(["m", "eval_weights", "kind"]).loss_over_equal_joint.agg(gm=gm, min="min", max="max").reset_index()
    mm.to_csv(out / "c1_weight_mismatch.csv", index=False)
    spec = pd.read_csv(out / "gradient_spectra.csv")
    spec.groupby("m").agg(["median", "min"]).drop(columns=["sid", "inst"], errors="ignore").to_csv(out / "gradient_spectra_summary.csv")
    headline["C1_gm_loss_ratio_joint_vs_pca_equal"] = {f"m{m}_N2^{lg}": round(v, 4) for (m, lg), v in
                                                        agg[agg.kind == "joint"].set_index(["m", "log2n"]).loss_ratio_vs_pca_equal.items()}
    headline["C1_weight_mismatch_range_of_gm"] = [float(mm[mm.kind != "pca"].gm.min()), float(mm[mm.kind != "pca"].gm.max())]
    headline["C1_min_top_share"] = float(spec[[f"top_share_{g}" for g in G3]].min().min())
    headline["C1_min_cos_between_greek_top_directions"] = float(spec[["cos_delta_gamma", "cos_delta_vega", "cos_gamma_vega"]].min().min())

    # --- timed helper ------------------------------------------------------------------------
    def timed_summary(df, keys):
        E = np.stack([df[f"v_{g}"].to_numpy() for g in G3], axis=1) - np.stack([truth[s] for s in df.sid])
        df = df.assign(**{f"e_{g}": E[:, k] for k, g in enumerate(G3)})
        rows = []
        for key, d in df.groupby(keys, dropna=False):
            key = key if isinstance(key, tuple) else (key,)
            sid = d.sid.iloc[0]
            row = dict(zip(keys, key), feasible=bool(d.feasible.all()), wall=d.wall.mean(), n=len(d), method="|".join(sorted(set(d.method))),
                       over_p95=float((d.wall / d.sim_budget).quantile(0.95)) if "sim_budget" in d else float((d.wall / d.budget).quantile(0.95)))
            if row["feasible"]:
                e = d[[f"e_{g}" for g in G3]].to_numpy()
                mse = np.maximum((e**2).mean(axis=0) - rse[sid] ** 2, 1e-300)
                row.update({f"mse_{g}": mse[k] for k, g in enumerate(G3)})
                row.update({f"loss_{wn}": float((w * mse / scale[sid] ** 2).sum()) for wn, w in wsets.items()})
                C = np.corrcoef(e, rowvar=False)
                row.update(corr_dg=C[0, 1], corr_dv=C[0, 2], corr_gv=C[1, 2])
                if all(f"ci_{g}" in d for g in G3):
                    row.update({f"cov_{g}": float((np.abs(d[f"e_{g}"]) <= d[f"ci_{g}"]).mean()) for g in G3})
                idx = rng.integers(0, len(e), (NBOOT, len(e)))
                row["_boot"] = ((e[idx] ** 2).mean(axis=1) / scale[sid] ** 2).astype(np.float32)  # (NBOOT, 3) normalised MSE
            rows.append(row)
        return pd.DataFrame(rows)

    def ratio_table(S, keys, num, den, by, wn="equal", time_normalised=False):
        """gm over scenarios of loss_num / loss_den with bootstrap CI; grouped by ``by``."""
        w = wsets[wn]
        out_rows = []
        for key, d in S.groupby(by, dropna=False):
            key = key if isinstance(key, tuple) else (key,)
            lr, lb, n = 0.0, np.zeros(NBOOT), 0
            for sid, x in d.groupby("sid"):
                a, b = x[x[keys] == num], x[x[keys] == den]
                if len(a) != 1 or len(b) != 1 or not (a.feasible.iloc[0] and b.feasible.iloc[0]):
                    continue
                a, b = a.iloc[0], b.iloc[0]
                ta, tb = (a.wall, b.wall) if time_normalised else (1.0, 1.0)
                lr += math.log(a[f"loss_{wn}"] * ta) - math.log(b[f"loss_{wn}"] * tb)
                lb += np.log(np.maximum((a["_boot"] * w).sum(axis=1), 1e-300) * ta) - np.log(np.maximum((b["_boot"] * w).sum(axis=1), 1e-300) * tb)
                n += 1
            if n:
                lo, hi = np.percentile(lb / n, [2.5, 97.5])
                out_rows.append(dict(zip(by, key), num=num, den=den, weights=wn, time_normalised=time_normalised,
                                     gm_ratio=math.exp(lr / n), ci_lo=math.exp(lo), ci_hi=math.exp(hi), n_scen=n))
        return out_rows

    # --- C2 ---------------------------------------------------------------------------------
    c2 = pd.read_csv(out / "c2_runs.csv.gz")
    S2 = timed_summary(c2, ["sid", "m", "budget", "strategy"])
    S2.drop(columns="_boot").to_csv(out / "c2_summary.csv", index=False)
    rt = []
    for num, den in (("shared-PCA", "separate-PCA"), ("shared-MC", "separate-MC"), ("shared-GPCA", "separate-GPCA"),
                     ("shared-GPCA", "shared-PCA"), ("shared-PCA", "shared-MC"), ("separate-GPCA", "separate-PCA")):
        for wn in wsets:
            rt += ratio_table(S2, "strategy", num, den, ["m", "budget"], wn)
        rt += ratio_table(S2, "strategy", num, den, ["m", "budget"], "equal", time_normalised=True)
    rt = pd.DataFrame(rt)
    rt.to_csv(out / "c2_ratios.csv", index=False)
    c2g = S2[S2.feasible].groupby(["m", "budget", "strategy"]).agg(
        loss=("loss_equal", gm), wall=("wall", "mean"), over_p95=("over_p95", "max"), method=("method", lambda v: "|".join(sorted(set(v)))),
        corr_dg=("corr_dg", "median"), corr_dv=("corr_dv", "median"), corr_gv=("corr_gv", "median"),
        cov_delta=("cov_delta", "mean"), cov_gamma=("cov_gamma", "mean"), cov_vega=("cov_vega", "mean")).reset_index()
    c2g.to_csv(out / "c2_by_cell.csv", index=False)
    e = rt[(rt.weights == "equal") & ~rt.time_normalised]
    for num, den in (("shared-PCA", "separate-PCA"), ("shared-MC", "separate-MC"), ("shared-GPCA", "shared-PCA")):
        x = e[(e.num == num) & (e.den == den)]
        headline[f"C2_loss_ratio_{num}_over_{den}_range"] = [float(x.gm_ratio.min()), float(x.gm_ratio.max())]
    headline["C2_max_p95_overrun"] = float(S2.over_p95.max())

    # --- D1 ---------------------------------------------------------------------------------
    d1 = pd.read_csv(out / "d1_runs.csv.gz")
    d1["label"] = d1.strategy + "@" + d1.M.astype(str)
    S1 = timed_summary(d1, ["sid", "m", "budget", "strategy", "M", "label"])
    S1.drop(columns="_boot").to_csv(out / "d1_summary.csv", index=False)
    be, tab = [], []
    for st in ("GPCA-joint", "GPCA-separate"):
        for M in sorted(S1[S1.strategy == st].M.unique()):
            for den in ("PCA@inf", "MC@inf"):
                tab += [dict(r, strategy=st, M=M) for r in ratio_table(S1, "label", f"{st}@{M}", den, ["m", "budget"])]
    tab = pd.DataFrame(tab)
    feas = S1[S1.strategy.str.startswith("GPCA")].groupby(["m", "budget", "strategy", "M"]).feasible.mean().rename("share_feasible").reset_index()
    tab = tab.merge(feas, on=["m", "budget", "strategy", "M"], how="outer")
    tab.to_csv(out / "d1_ratios.csv", index=False)
    for (m, B, st), d in tab[tab.den == "PCA@inf"].groupby(["m", "budget", "strategy"]):
        d = d.sort_values("M")
        ok = d[(d.share_feasible == 1.0) & (d.gm_ratio < 1.0)]
        sig = d[(d.share_feasible == 1.0) & (d.ci_hi < 1.0)]
        be.append(dict(m=m, budget=B, strategy=st, break_even_M_point=float(ok.M.min()) if len(ok) else np.nan,
                       break_even_M_significant=float(sig.M.min()) if len(sig) else np.nan,
                       ratio_at_M1=float(d[d.M == 1].gm_ratio.iloc[0]) if (d.M == 1).any() and np.isfinite(d[d.M == 1].gm_ratio.iloc[0]) else np.nan,
                       feasible_at_M1=float(d[d.M == 1].share_feasible.iloc[0]) if (d.M == 1).any() else np.nan,
                       ratio_amortised=float(d[np.isinf(d.M)].gm_ratio.iloc[0]) if np.isinf(d.M).any() else np.nan))
    be = pd.DataFrame(be)
    tc = pd.read_csv(out / "training_costs.csv").groupby(["m", "kind"]).train_seconds.median().unstack()
    be["train_ms_joint"] = be.m.map(tc["joint"]) * 1e3
    be["train_ms_separate"] = be.m.map(tc[[f"spec-{g}" for g in G3]].sum(axis=1)) * 1e3
    be.to_csv(out / "d1_break_even.csv", index=False)
    headline["D1_break_even"] = be.round(4).to_dict(orient="records")

    # --- D2 ---------------------------------------------------------------------------------
    d2 = pd.read_csv(out / "d2_scramble_means.csv.gz")
    al = pd.read_csv(out / "d2_alignment.csv").groupby(["m", "axis", "factor"])[["align_k1", "align_k2", "align_k3"]].median().reset_index()
    rows = []
    w = wsets["equal"]
    for (bs, m, axis, fac), d in d2.groupby(["base_sid", "m", "axis", "factor"]):
        sc = d[[f"scale_{g}" for g in G3]].iloc[0].to_numpy(float)
        L = {k: float((w * pooled_var(x)[0] / sc**2).sum()) for k, x in d.groupby("kind")}
        row = dict(m=m, axis=axis, factor=fac, transferred_over_fresh=L["transferred"] / L["fresh"], transferred_over_pca=L["transferred"] / L["pca"],
                   fresh_over_pca=L["fresh"] / L["pca"])
        for nm, (a, b) in (("tf", ("transferred", "fresh")), ("tp", ("transferred", "pca"))):
            row[f"{nm}_lo"], row[f"{nm}_hi"] = boot_loss_ratio(d[d.kind == a], d[d.kind == b], sc, w, rng, 1000)
        rows.append(row)
    d2s = pd.DataFrame(rows).merge(al, on=["m", "axis", "factor"])
    alpha, a_ = SCH["alpha"], []
    # model-based retraining break-even: transferred needs rho^(1/alpha) times the simulation time of a fresh rotation
    for r in d2s.itertuples():
        ctrain = float(tc.loc[r.m, "joint"])
        extra = max(r.transferred_over_fresh, 1.0) ** (1.0 / alpha) - 1.0
        a_.append({f"retrain_break_even_M_at_{int(B * 1e3)}ms": (ctrain / (B * extra) if extra > 1e-9 else np.inf)
                   for B in CFG["break_even"]["per_valuation_budgets_seconds"]})
    d2s = pd.concat([d2s, pd.DataFrame(a_)], axis=1)
    d2s.to_csv(out / "d2_transfer.csv", index=False)
    headline["D2_transferred_over_fresh_range"] = [float(d2s.transferred_over_fresh.min()), float(d2s.transferred_over_fresh.max())]
    headline["D2_transferred_over_pca_range"] = [float(d2s.transferred_over_pca.min()), float(d2s.transferred_over_pca.max())]
    headline["D2_n_targets_transferred_worse_than_fresh_significant"] = int((d2s.tf_lo > 1).sum())
    headline["D2_n_targets_transferred_worse_than_pca_significant"] = int((d2s.tp_lo > 1).sum())
    headline["D2_n_targets"] = int(len(d2s))
    headline["max_abs_reference_check_z"] = float(np.abs(refs[[f"check_z_{g}" for g in G3]].to_numpy()).max())
    (out / "headline.json").write_text(json.dumps(headline, indent=2, default=float))
    log(json.dumps(headline, default=float))
    figures(out, agg, c2g, rt, tab, d2s)


def figures(out, agg, c2g, rt, tab, d2s):
    fd = out / "figures"
    fd.mkdir(exist_ok=True)
    ms = sorted(agg.m.unique())
    # C1: loss ratio vs path PCA for each rotation
    fig, axes = plt.subplots(1, len(ms), figsize=(5 * len(ms), 3.6), squeeze=False, sharey=True)
    kinds = ["joint"] + [f"spec-{g}" for g in G3] + [k for k in agg.kind.unique() if k.startswith("joint-")]
    for ax, m in zip(axes[0], ms):
        d = agg[agg.m == m]
        lgs = sorted(d.log2n.unique())
        for j, lg in enumerate(lgs):
            x = d[d.log2n == lg].set_index("kind").loss_ratio_vs_pca_equal
            ax.bar(np.arange(len(kinds)) + (j - (len(lgs) - 1) / 2) * 0.25, [x.get(k, np.nan) for k in kinds], 0.25, label=f"N = 2^{lg}")
        ax.axhline(1, color="#1f1f1e", lw=0.8)
        ax.set_yscale("log"); ax.set_xticks(range(len(kinds)), kinds, rotation=35, ha="right", fontsize=7); ax.set_title(f"m = {m}")
    axes[0, 0].set_ylabel("risk-vector loss / loss with path PCA\n(equal weights; geo-mean over scenarios)")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fd / "fig5C1_fixed_n_loss.png", bbox_inches="tight"); fig.savefig(fd / "fig5C1_fixed_n_loss.pdf", bbox_inches="tight")
    plt.close(fig)
    # C2: loss vs measured total time
    col = {"shared-MC": "#eb6834", "shared-PCA": "#2a78d6", "shared-GPCA": "#1baf7a", "separate-MC": "#eb6834", "separate-PCA": "#2a78d6", "separate-GPCA": "#1baf7a"}
    fig, axes = plt.subplots(1, len(ms), figsize=(5 * len(ms), 3.8), squeeze=False)
    for ax, m in zip(axes[0], ms):
        for st in col:
            d = c2g[(c2g.m == m) & (c2g.strategy == st)].sort_values("budget")
            ax.loglog(d.wall * 1e3, d.loss, marker="o", ms=3.5, color=col[st], ls="-" if st.startswith("shared") else "--", label=st)
        ax.set_title(f"m = {m}"); ax.set_xlabel("measured total wall time for (Delta, Gamma, Vega) [ms]")
    axes[0, 0].set_ylabel("risk-vector loss (equal weights)")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fd / "fig5C2_shared_vs_separate.png", bbox_inches="tight"); fig.savefig(fd / "fig5C2_shared_vs_separate.pdf", bbox_inches="tight")
    plt.close(fig)
    # D1: loss ratio vs PCA against M
    fig, axes = plt.subplots(1, len(ms), figsize=(5 * len(ms), 3.8), squeeze=False, sharey=True)
    cb = ["#2a78d6", "#e34948", "#1baf7a", "#eda100"]
    for ax, m in zip(axes[0], ms):
        k = 0
        for (B, st), d in tab[(tab.m == m) & (tab.den == "PCA@inf")].groupby(["budget", "strategy"]):
            d = d.dropna(subset=["gm_ratio"]).sort_values("M")
            x = np.where(np.isinf(d.M), 300, d.M)
            ax.errorbar(x, d.gm_ratio, yerr=[d.gm_ratio - d.ci_lo, d.ci_hi - d.gm_ratio], marker="o", ms=3.5, capsize=2, color=cb[k % 4],
                        ls="-" if st == "GPCA-joint" else "--", label=f"{st}, B = {B * 1e3:.0f} ms")
            k += 1
        ax.axhline(1, color="#1f1f1e", lw=0.8)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(f"m = {m}")
        ax.set_xticks([1, 2, 5, 10, 25, 50, 100, 300], ["1", "2", "5", "10", "25", "50", "100", "inf"])
        ax.set_xlabel("number of valuations M sharing one training")
    axes[0, 0].set_ylabel("loss / loss of path PCA at the same total time\n(<1: training has paid for itself)")
    axes[0, 0].legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(fd / "fig5D1_break_even.png", bbox_inches="tight"); fig.savefig(fd / "fig5D1_break_even.pdf", bbox_inches="tight")
    plt.close(fig)
    # D2: transfer
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, axis in zip(axes, ("K", "sigma", "T")):
        for m, c in zip(ms, ("#2a78d6", "#e34948")):
            d = d2s[(d2s.m == m) & (d2s.axis == axis)].assign(x=lambda t: t.factor.astype(float)).sort_values("x")
            ax.errorbar(d.x, d.transferred_over_fresh, yerr=[d.transferred_over_fresh - d.tf_lo, d.tf_hi - d.transferred_over_fresh],
                        marker="o", ms=3.5, capsize=2, color=c, label=f"m = {m}: transferred / fresh")
            ax.plot(d.x, d.transferred_over_pca, marker="s", ms=3.5, ls=":", color=c, label=f"m = {m}: transferred / path PCA")
        ax.axhline(1, color="#1f1f1e", lw=0.8); ax.set_yscale("log")
        if (d2s.axis == axis).any():
            ax.set_xscale("log")
        ax.set_xlabel(f"{axis} of the new contract / {axis} of the training contract")
    axes[0].set_ylabel("risk-vector loss ratio at fixed N (equal weights)")
    axes[0].legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(fd / "fig5D2_transfer.png", bbox_inches="tight"); fig.savefig(fd / "fig5D2_transfer.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--analysis-only", action="store_true")
    a = ap.parse_args()
    out = RESULTS / "phase5_joint" / ("quick" if a.quick else "study")
    t0 = time.time()
    if not a.analysis_only:
        simulate(out, a.quick)
        meta = json.loads((out / "meta.json").read_text())
        meta["wall_seconds_total"] = time.time() - t0
        (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    analyse(out)

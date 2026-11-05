"""Phase 4, Part C/D: Greek-aware gradient-PCA transforms for RQMC Greeks of the Asian digital.

    python greek_aware_study.py --quick          # pipeline check (not for conclusions)
    python greek_aware_study.py                  # full study (~40-60 min)
    python greek_aware_study.py --analysis-only

Everything is fixed a priori (no tuning on evaluation scenarios): n_train = 1024
gradient samples, trace-normalised equal weights for the joint transform,
budgets 10 ms and 50 ms. Independent random streams for training (family 1),
evaluation of gradient energy (2), references (3), fixed-N runs (4), n_train
sweep (5) and timed runs (6). Output: results/phase4_greek_aware/<run>/.
"""
import argparse
import itertools
import json
import math
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mc_greeks import visualization as viz  # noqa: F401
from mc_greeks.asian import AsianDigital
from mc_greeks.asian_qmc import (calibrate_rqmc, construction_matrix, pps_for, reference_estimates, residual_cov,
                                 timed_estimate)
from mc_greeks.config import RESULTS
from mc_greeks.greek_aware_qmc import (GREEKS, build_transform, capture_curve, path_gradient_cov, rqmc_fixed_n,
                                       timed_with_transform)
from mc_greeks.models import BlackScholesParams
from mc_greeks.simulation import make_generator
from mc_greeks.statistics import loglog_slope

KINDS = ("seq", "bb", "pca", "gpca-delta", "gpca-gamma", "gpca-vega", "gpca-joint")
COL = {"seq": "#9a9890", "bb": "#1baf7a", "pca": "#2a78d6", "gpca-delta": "#eb6834", "gpca-gamma": "#6250d6",
       "gpca-vega": "#e87ba4", "gpca-joint": "#1f1f1e", "MC": "#eda100"}
FULL = dict(ms=(12, 52, 104), moneyness=(0.94, 1.06), sigmas=(0.22, 0.38), Ts=(1.0,), r=0.02, n_train=1024,
            log2n=(7, 8, 9, 10, 11, 12, 13), scrambles=16, budgets=(0.010, 0.050), reps=20, seed=2026_1130,
            sweep_ntrain=(32, 128, 512, 2048, 8192), sweep_log2n=11, h_eval=8192)
QUICK = dict(FULL, ms=(12,), moneyness=(1.06,), sigmas=(0.22,), log2n=(7, 9), scrambles=8, budgets=(0.010,), reps=3,
             sweep_ntrain=(64, 1024), h_eval=1024)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def scen(cfg):
    return [dict(sid=i, m=m, moneyness=k, sigma=s, T=T, r=cfg["r"], S0=100.0)
            for i, (m, k, s, T) in enumerate(itertools.product(cfg["ms"], cfg["moneyness"], cfg["sigmas"], cfg["Ts"]))]


def objs(sc):
    return AsianDigital(sc["moneyness"] * sc["S0"], int(sc["m"])), BlackScholesParams(sc["S0"], sc["r"], sc["sigma"], sc["T"])


def strategy_name(kind, mode):
    return f"{kind} ({mode})"


def simulate(cfg, out):
    from phase4_cost_validation import provenance  # same provenance record
    out.mkdir(parents=True, exist_ok=True)
    S = scen(cfg)
    pd.DataFrame(S).to_csv(out / "scenarios.csv", index=False)
    seed = cfg["seed"]
    meta = {"config": cfg, "provenance": provenance()}
    t0 = time.perf_counter()
    cal = calibrate_rqmc(cfg["ms"], seed=seed + 7, pooled=True)
    meta["calibration_seconds"] = time.perf_counter() - t0
    setup, spectra, capture, fixedn, sweep, timed, refs = [], [], [], [], [], [], []
    for sc in S:
        f, p = objs(sc)
        m, dt = f.m, p.T / f.m
        sid = sc["sid"]
        log(f"scenario {sid + 1}/{len(S)}: m={m}, K/S0={sc['moneyness']}, sigma={sc['sigma']}")
        r = reference_estimates(f, p, make_generator([seed, 3, sid]), 2**18 if m <= 24 else 2**16, 32,
                                check_paths=2**20 if len(S) > 1 else 2**16, greeks=("price",) + GREEKS)
        refs.append({"sid": sid, **r})
        # transforms (training stream 1); setup measured cold
        T = {}
        for k_i, kind in enumerate(KINDS):
            T[kind] = build_transform(kind, f, p, n_train=cfg["n_train"], rng=make_generator([seed, 1, sid, k_i]))
            setup.append(dict(sid=sid, m=m, kind=kind, setup_seconds=T[kind].setup_seconds))
        # spectra: path covariance vs gradient covariances
        wC = np.sort(np.linalg.eigvalsh(residual_cov(m, dt)))[::-1]
        for j, v in enumerate(wC / wC.sum()):
            spectra.append(dict(sid=sid, m=m, matrix="path covariance C", k=j + 1, share=v))
        for kind in KINDS[3:]:
            w = np.clip(T[kind].eigenvalues, 0, None)
            for j, v in enumerate(w / w.sum()):
                spectra.append(dict(sid=sid, m=m, matrix=f"gradient cov ({kind})", k=j + 1, share=v))
        # capture curves on an independent evaluation stream (2)
        for g_i, g in enumerate(GREEKS):
            H = path_gradient_cov(f, p, g, cfg["h_eval"], make_generator([seed, 2, sid, g_i]))
            for kind in KINDS:
                c = capture_curve(T[kind].L, H)
                for k in (1, 2, 4, 8):
                    if k <= m - 1:
                        capture.append(dict(sid=sid, m=m, greek=g, kind=kind, k=k, capture=c[k - 1]))
                d99 = int(np.searchsorted(c, 0.99) + 1)
                capture.append(dict(sid=sid, m=m, greek=g, kind=kind, k=-1, capture=d99))  # k=-1 row: dim for 99%
        # fixed-N convergence (stream 4), all Greeks from the same points
        for k_i, kind in enumerate(KINDS):
            for lg in cfg["log2n"]:
                est = rqmc_fixed_n(f, p, T[kind].L, 2**lg, cfg["scrambles"], make_generator([seed, 4, sid, k_i, lg]))
                for g in GREEKS:
                    e = est[g] - r[f"ref_{g}"]
                    fixedn.append(dict(sid=sid, m=m, kind=kind, n=2**lg, greek=g, mse=float(np.mean(e**2)),
                                       var=float(est[g].var(ddof=1)), ref_se=r[f"ref_se_{g}"]))
        # n_train sweep (stream 5): gradient-noise sensitivity
        if m >= 52:
            for nt in cfg["sweep_ntrain"]:
                for g_i, g in enumerate(GREEKS):
                    Tg = build_transform(f"gpca-{g}", f, p, n_train=nt, rng=make_generator([seed, 5, sid, g_i, nt]))
                    est = rqmc_fixed_n(f, p, Tg.L, 2**cfg["sweep_log2n"], cfg["scrambles"],
                                       make_generator([seed, 5, sid, g_i, nt, 1]), greeks=(g,))[g]
                    H = path_gradient_cov(f, p, g, cfg["h_eval"], make_generator([seed, 2, sid, g_i]))
                    sweep.append(dict(sid=sid, m=m, greek=g, n_train=nt, mse=float(np.mean((est - r[f"ref_{g}"]) ** 2)),
                                      capture1=float(capture_curve(Tg.L, H)[0]), setup_seconds=Tg.setup_seconds))
        # timed runs (stream 6)
        for b_i, B in enumerate(cfg["budgets"]):
            for g_i, g in enumerate(GREEKS):
                for r_i in range(cfg["reps"]):
                    base = [seed, 6, sid, b_i, g_i, r_i]
                    e = timed_estimate(g, f, p, B, make_generator(base + [0]), "MC")
                    timed.append(dict(sid=sid, m=m, budget=B, greek=g, strategy="MC", value=e.value, wall=e.wall,
                                      setup=0.0, units=e.n_units))
                    for kind in ("pca", f"gpca-{g}", "gpca-joint"):
                        k_i = KINDS.index(kind)
                        # amortised: transform already built (its setup reported, not charged)
                        e = timed_with_transform(g, f, p, B, make_generator(base + [1, k_i]), T[kind].L,
                                                 pps_for(B, cal, "RQMC-pca", m))
                        timed.append(dict(sid=sid, m=m, budget=B, greek=g, strategy=strategy_name(kind.replace(g, "g") if kind == f"gpca-{g}" else kind, "amortised"),
                                          value=e.value, wall=e.wall, setup=T[kind].setup_seconds, units=e.n_units))
                        # one-off: build the transform inside the budget, then estimate with what is left
                        t1 = time.perf_counter()
                        Tb = build_transform(kind, f, p, n_train=cfg["n_train"], rng=make_generator(base + [2, k_i]))
                        rem = max(B - (time.perf_counter() - t1), 0.0)
                        e = timed_with_transform(g, f, p, rem, make_generator(base + [3, k_i]), Tb.L,
                                                 pps_for(max(rem, 1e-4), cal, "RQMC-pca", m))
                        timed.append(dict(sid=sid, m=m, budget=B, greek=g, strategy=strategy_name(kind.replace(g, "g") if kind == f"gpca-{g}" else kind, "one-off"),
                                          value=e.value, wall=time.perf_counter() - t1, setup=Tb.setup_seconds,
                                          units=e.n_units))
    for name, rows in (("references", refs), ("setup_costs", setup), ("spectra", spectra), ("capture", capture),
                       ("fixed_n", fixedn), ("ntrain_sweep", sweep)):
        pd.DataFrame(rows).to_csv(out / f"{name}.csv", index=False)
    pd.DataFrame(timed).to_csv(out / "timed_raw.csv.gz", index=False)
    meta["wall_seconds_total"] = time.perf_counter() - t0
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))


def gm(x):
    return float(np.exp(np.mean(np.log(np.asarray(x, float)))))


def boot_gm(x, rng, n=2000):
    lx = np.log(np.asarray(x, float))
    bs = [lx[rng.integers(0, len(lx), len(lx))].mean() for _ in range(n)]
    return math.exp(lx.mean()), math.exp(np.percentile(bs, 2.5)), math.exp(np.percentile(bs, 97.5))


def analyse(cfg, out):
    rng = np.random.default_rng(cfg["seed"] + 5)
    S = pd.read_csv(out / "scenarios.csv")
    refs = pd.read_csv(out / "references.csv")
    chk = refs[[c for c in refs if c.startswith("check_z_")]].stack()
    res = {"reference_check": {"n": int(len(chk)), "mean_z": float(chk.mean()), "sd_z": float(chk.std()), "max_abs_z": float(chk.abs().max())}}

    # H1: effective dimension proxies
    cap = pd.read_csv(out / "capture.csv")
    d99 = cap[cap.k == -1].rename(columns={"capture": "dim99"})
    h1 = cap[cap.k == 1].pivot_table(index=["m", "greek"], columns="kind", values="capture", aggfunc="median")
    h1.to_csv(out / "H1_capture_k1_median.csv")
    d99.pivot_table(index=["m", "greek"], columns="kind", values="dim99", aggfunc="median").to_csv(out / "H1_dim99_median.csv")

    # H2: fixed-N MSE ratios vs standard PCA and slopes
    fx = pd.read_csv(out / "fixed_n.csv")
    fx["mse_c"] = np.maximum(fx.mse - fx.ref_se**2, 1e-300)
    base = fx[fx.kind == "pca"].set_index(["sid", "n", "greek"]).mse_c
    fx["ratio_vs_pca"] = fx.mse_c.to_numpy() / base.reindex(fx.set_index(["sid", "n", "greek"]).index).to_numpy()
    rows = []
    nmax = fx.n.max()
    for (m, g, kind), d in fx[fx.n == nmax].groupby(["m", "greek", "kind"]):
        v, lo, hi = boot_gm(d.ratio_vs_pca, rng)
        rows.append(dict(m=m, greek=g, kind=kind, n=nmax, gm_mse_ratio_vs_pca=v, ci_lo=lo, ci_hi=hi,
                         n_better=int((d.ratio_vs_pca < 1).sum()), n_scen=len(d)))
    pd.DataFrame(rows).to_csv(out / "H2_fixed_n_ratio_vs_pca.csv", index=False)
    slopes = []
    for (sid, m, g, kind), d in fx.groupby(["sid", "m", "greek", "kind"]):
        fit = loglog_slope(d.n, np.sqrt(d.mse_c))
        slopes.append(dict(sid=sid, m=m, greek=g, kind=kind, slope=fit["slope"], lo=fit["slope_lo"], hi=fit["slope_hi"]))
    sl = pd.DataFrame(slopes)
    sl.to_csv(out / "H2_slopes_per_scenario.csv", index=False)
    sl.groupby(["m", "greek", "kind"]).slope.agg(["median", "min", "max"]).reset_index().to_csv(out / "H2_slopes_summary.csv", index=False)

    # joint objective: geometric mean over Greeks of the MSE ratio vs PCA, per scenario (H4)
    h4 = fx[fx.n == nmax].copy()
    h4["kind2"] = np.where(h4.kind == "gpca-" + h4.greek, "gpca-specific", h4.kind)
    j = h4[h4.kind2.isin(["pca", "gpca-specific", "gpca-joint", "bb"])].groupby(["sid", "m", "kind2"]).ratio_vs_pca.apply(gm).reset_index()
    rows = []
    for (m, kind), d in j.groupby(["m", "kind2"]):
        v, lo, hi = boot_gm(d.ratio_vs_pca, rng)
        rows.append(dict(m=m, kind=kind, joint_objective=v, ci_lo=lo, ci_hi=hi))
    pd.DataFrame(rows).to_csv(out / "H4_joint_objective.csv", index=False)

    # H3: wall-clock with setup charged vs amortised
    tm = pd.read_csv(out / "timed_raw.csv.gz")
    tr = refs.melt(id_vars="sid", value_vars=[f"ref_{g}" for g in GREEKS], var_name="k", value_name="truth")
    tr["greek"] = tr.k.str[4:]
    rs = refs.melt(id_vars="sid", value_vars=[f"ref_se_{g}" for g in GREEKS], var_name="k", value_name="ref_se")
    rs["greek"] = rs.k.str[7:]
    tm = tm.merge(tr[["sid", "greek", "truth"]], on=["sid", "greek"]).merge(rs[["sid", "greek", "ref_se"]], on=["sid", "greek"])
    tm["e2"] = (tm.value - tm.truth) ** 2
    tm["over"] = tm.wall / tm.budget
    s = tm.groupby(["sid", "m", "budget", "greek", "strategy"]).agg(
        mse=("e2", "mean"), ref_se=("ref_se", "first"), wall=("wall", "mean"),
        wall_p95_over_B=("over", lambda v: float(v.quantile(0.95))), setup=("setup", "mean")).reset_index()
    s["mse_c"] = np.maximum(s.mse - s.ref_se**2, 1e-300)
    s["wn"] = s.mse_c * s.wall
    ref = s[s.strategy == "pca (amortised)"].set_index(["sid", "budget", "greek"]).wn
    s["ratio_vs_pca_amortised"] = s.wn.to_numpy() / ref.reindex(s.set_index(["sid", "budget", "greek"]).index).to_numpy()
    s.to_csv(out / "timed_summary.csv", index=False)
    rows = []
    for (B, g, m, st), d in s.groupby(["budget", "greek", "m", "strategy"]):
        v, lo, hi = boot_gm(d.ratio_vs_pca_amortised, rng)
        rows.append(dict(budget=B, greek=g, m=m, strategy=st, gm_wn_ratio_vs_pca_amortised=v, ci_lo=lo, ci_hi=hi,
                         n_better=int((d.ratio_vs_pca_amortised < 1).sum()), n=len(d), median_wall_over_B=float((d.wall / B).median()),
                         p95_wall_over_B=float(d.wall_p95_over_B.max())))
    pd.DataFrame(rows).to_csv(out / "H3_timed_ratio_vs_pca.csv", index=False)

    # H5: n_train sweep
    try:
        sw = pd.read_csv(out / "ntrain_sweep.csv")
    except pd.errors.EmptyDataError:
        sw = pd.DataFrame()
    if len(sw):
        pca_n = fx[(fx.kind == "pca") & (fx.n == 2 ** cfg["sweep_log2n"])][["sid", "greek", "mse_c"]].rename(columns={"mse_c": "pca_mse"})
        sw = sw.merge(pca_n, on=["sid", "greek"], how="left")
        sw["ratio_vs_pca"] = sw.mse / sw.pca_mse
        sw.to_csv(out / "H5_ntrain_sweep.csv", index=False)
    setup = pd.read_csv(out / "setup_costs.csv")
    setup.groupby(["m", "kind"]).setup_seconds.median().unstack().to_csv(out / "setup_costs_median.csv")
    (out / "headline.json").write_text(json.dumps(res, indent=2))
    figures(cfg, out, fx, sl, s, sw, setup)
    log(f"analysis written to {out}")


def figures(cfg, out, fx, sl, s, sw, setup):
    fd = out / "figures"
    fd.mkdir(exist_ok=True)
    ms = sorted(fx.m.unique())
    # eigenvalue decay
    sp = pd.read_csv(out / "spectra.csv")
    fig, axes = plt.subplots(1, len(ms), figsize=(4 * len(ms), 3.4), squeeze=False)
    for ax, m in zip(axes[0], ms):
        d = sp[sp.m == m].groupby(["matrix", "k"]).share.median().reset_index()
        for mat, dd in d.groupby("matrix"):
            kind = mat.split("(")[-1].rstrip(")") if "(" in mat else "pca"
            ax.semilogy(dd.k, dd.share, marker=".", color=COL.get(kind, "k"), label=mat, lw=1.2)
        ax.set_xlim(0.5, min(20, m - 1) + 0.5)
        ax.set_title(f"m={m}")
        ax.set_xlabel("eigenvalue index k")
    axes[0, 0].set_ylabel("eigenvalue / trace (median over scenarios)")
    axes[0, 0].legend(fontsize=6.5)
    fig.suptitle("Eigenvalue decay: path covariance vs Greek gradient covariances", y=1.02)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(fd / f"fig4C1_eigen_decay.{e}", bbox_inches="tight")
    plt.close(fig)
    # capture by k
    cap = pd.read_csv(out / "capture.csv")
    cap = cap[cap.k > 0]
    fig, axes = plt.subplots(len(GREEKS), len(ms), figsize=(3.6 * len(ms), 8), squeeze=False)
    for i, g in enumerate(GREEKS):
        for j, m in enumerate(ms):
            ax = axes[i, j]
            d = cap[(cap.greek == g) & (cap.m == m)].groupby(["kind", "k"]).capture.median().reset_index()
            for kind, dd in d.groupby("kind"):
                ax.plot(dd.k, dd.capture, marker="o", color=COL[kind], label=kind, lw=1.1)
            ax.set_xscale("log", base=2)
            ax.set_ylim(0, 1.02)
            ax.set_title(f"{g}, m={m}", fontsize=9)
            if j == 0:
                ax.set_ylabel("gradient energy in first k coords")
            if i == len(GREEKS) - 1:
                ax.set_xlabel("k")
    axes[0, 0].legend(fontsize=6.5)
    fig.suptitle("Fraction of Greek-integrand gradient energy captured by the first k Sobol coordinates (independent sample)", y=1.0)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(fd / f"fig4C2_capture_k.{e}", bbox_inches="tight")
    plt.close(fig)
    # MSE vs N
    fig, axes = plt.subplots(len(GREEKS), len(ms), figsize=(3.6 * len(ms), 8), squeeze=False)
    for i, g in enumerate(GREEKS):
        for j, m in enumerate(ms):
            ax = axes[i, j]
            d = fx[(fx.greek == g) & (fx.m == m)].groupby(["kind", "n"]).mse_c.apply(gm).reset_index()
            for kind, dd in d.groupby("kind"):
                ax.loglog(dd.n, dd.mse_c, marker="o", color=COL[kind], label=kind, lw=1.4 if kind.startswith("gpca") else 1.0,
                          ls="-" if kind.startswith("gpca") else "--")
            ax.set_title(f"{g}, m={m}", fontsize=9)
            if j == 0:
                ax.set_ylabel("MSE (geo-mean over scenarios)")
            if i == len(GREEKS) - 1:
                ax.set_xlabel("points per scrambling N")
    axes[0, 0].legend(fontsize=6.5)
    fig.suptitle("Fixed-N MSE: path constructions vs Greek-aware transforms (16 scramblings)", y=1.0)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(fd / f"fig4C3_mse_vs_n.{e}", bbox_inches="tight")
    plt.close(fig)
    # slopes
    fig, ax = plt.subplots(figsize=(10, 3.6))
    kinds = list(KINDS)
    for i, g in enumerate(GREEKS):
        for j, kind in enumerate(kinds):
            d = sl[(sl.greek == g) & (sl.kind == kind)]
            x = i * (len(kinds) + 1) + j
            ax.scatter(np.full(len(d), x), d.slope, color=COL[kind], s=10)
            ax.errorbar(x, d.slope.median(), yerr=[[d.slope.median() - d.slope.min()], [d.slope.max() - d.slope.median()]],
                        color=COL[kind], fmt="s", capsize=2)
    ax.set_xticks([i * (len(kinds) + 1) + 3 for i in range(3)], GREEKS)
    ax.axhline(-0.5, color="k", ls=":", lw=0.8)
    ax.axhline(-1.0, color="k", ls="--", lw=0.8)
    ax.set_ylabel("fitted RMSE slope vs N (per scenario)")
    handles = [plt.Line2D([], [], color=COL[k], marker="s", ls="", label=k) for k in kinds]
    ax.legend(handles=handles, fontsize=7, ncol=4)
    ax.set_title("Empirical convergence rates (dotted: MC rate -1/2; dashed: -1)")
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(fd / f"fig4C4_slopes.{e}", bbox_inches="tight")
    plt.close(fig)
    # MSE vs time (timed)
    fig, axes = plt.subplots(len(GREEKS), len(ms), figsize=(3.6 * len(ms), 8), squeeze=False)
    for i, g in enumerate(GREEKS):
        for j, m in enumerate(ms):
            ax = axes[i, j]
            d = s[(s.greek == g) & (s.m == m)].groupby(["strategy", "budget"]).agg(w=("wall", "median"), mse=("mse_c", gm)).reset_index()
            for st, dd in d.groupby("strategy"):
                kind = st.split(" ")[0]
                c = COL.get(kind.replace("gpca-g", "gpca-delta"), "k") if kind != "MC" else COL["MC"]
                ax.loglog(dd.w * 1e3, dd.mse, marker="o", ls="-" if "amort" in st or st == "MC" else ":", color=c, label=st, lw=1.1)
            ax.set_title(f"{g}, m={m}", fontsize=9)
            if i == len(GREEKS) - 1:
                ax.set_xlabel("measured wall time [ms]")
    axes[0, 0].legend(fontsize=6)
    fig.suptitle("MSE vs measured time (solid: transform setup amortised; dotted: setup charged)", y=1.0)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(fd / f"fig4C5_mse_vs_time.{e}", bbox_inches="tight")
    plt.close(fig)
    # joint vs specific
    h4 = pd.read_csv(out / "H4_joint_objective.csv")
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for k_i, kind in enumerate(["bb", "gpca-specific", "gpca-joint"]):
        d = h4[h4.kind == kind].sort_values("m")
        x = np.arange(len(d)) + (k_i - 1) * 0.15
        ax.errorbar(x, d.joint_objective, yerr=[d.joint_objective - d.ci_lo, d.ci_hi - d.joint_objective],
                    marker="o", capsize=2, color=COL.get(kind, "#eb6834"), label=kind)
    ax.axhline(1, color="k", lw=0.8)
    ax.set_yscale("log")
    ax.set_xticks(range(len(ms)), [f"m={m}" for m in ms])
    ax.set_ylabel("geo-mean over Greeks of MSE / MSE(PCA)\nat the largest N (95% CI over scenarios)")
    ax.set_title("Joint vs Greek-specific transforms (<1: better than path PCA)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(fd / f"fig4C6_joint_vs_specific.{e}", bbox_inches="tight")
    plt.close(fig)
    # setup costs
    fig, ax = plt.subplots(figsize=(6, 3.4))
    d = setup.groupby(["m", "kind"]).setup_seconds.median().reset_index()
    for kind, dd in d.groupby("kind"):
        ax.semilogy(dd.m, dd.setup_seconds * 1e3, marker="o", color=COL[kind], label=kind)
    ax.set_xlabel("m")
    ax.set_ylabel("transform construction [ms] (median)")
    ax.set_title(f"Setup cost (gradient PCA: n_train = {cfg['n_train']})")
    ax.legend(fontsize=7)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(fd / f"fig4C7_setup_costs.{e}", bbox_inches="tight")
    plt.close(fig)
    # n_train sweep
    if len(sw):
        fig, axes = plt.subplots(1, len(GREEKS), figsize=(11, 3.4), sharey=True)
        for ax, g in zip(axes, GREEKS):
            d = sw[sw.greek == g].groupby(["m", "n_train"]).ratio_vs_pca.apply(gm).reset_index()
            for m, dd in d.groupby("m"):
                ax.loglog(dd.n_train, dd.ratio_vs_pca, marker="o", label=f"m={m}")
            ax.axhline(1, color="k", lw=0.8)
            ax.set_title(g)
            ax.set_xlabel("gradient training sample n_train")
        axes[0].set_ylabel(f"MSE / MSE(PCA) at N=2^{cfg['sweep_log2n']}")
        axes[0].legend(fontsize=7)
        fig.suptitle("Gradient noise: effect of the training sample size", y=1.02)
        fig.tight_layout()
        for e in ("png", "pdf"):
            fig.savefig(fd / f"fig4C8_ntrain_sweep.{e}", bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--analysis-only", action="store_true")
    a = ap.parse_args()
    cfg = QUICK if a.quick else FULL
    out = RESULTS / "phase4_greek_aware" / ("quick" if a.quick else "study")
    if not a.analysis_only:
        simulate(cfg, out)
    analyse(cfg, out)

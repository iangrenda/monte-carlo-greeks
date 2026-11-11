"""Phase 7E: from Chronos-2 volatility-curve forecasts to uncertainty in Asian-digital values and Greeks.

    HF_HUB_OFFLINE=1 ~/.venvs/chronos/Scripts/python scripts/phase7_forecast_greeks_study.py forecast    # Chronos-2 + GARCH curves
    ~/.venvs/mc-greeks/Scripts/python scripts/phase7_forecast_greeks_study.py propagate                  # repricing + analysis
    ... add --quick to either for a plumbing check (NOT evidence); 'analyse' re-runs the analysis only

Protocol: configs/phase7_forecast_greeks.json (committed before any curve was propagated).
Physical-measure scenario analysis: values under forecast volatility curves, not option prices.
Output: results/phase7_forecast_greeks/<run>/ (curve_forecasts.npz is regenerated, not committed).
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import phase7_vol_forecast_study as V  # noqa: E402  (data loading, trailing RV, GARCH, Chronos pipeline)

CFG = json.loads((ROOT / "configs" / "phase7_forecast_greeks.json").read_text())
LEV = np.array(V.CHRONOS_Q)
M, BLOCK, H = 12, 5, 60
CTX = 512
GREEKS = ("price", "delta", "gamma", "vega")
COPULAS = ("comonotone", "independent", "empirical")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def outdir(quick):
    d = ROOT / "results" / "phase7_forecast_greeks" / ("quick" if quick else "study")
    d.mkdir(parents=True, exist_ok=True)
    return d


# --- stage 1: forecasts (Chronos environment) ----------------------------------------------------

def forecast_stage(out, quick):
    df, sha = V.load()
    r = df.r.to_numpy()
    n = len(r)
    ly = np.log(V.trailing_rv(r, BLOCK))
    origins = np.arange(CTX + BLOCK - 2, n, BLOCK)                    # needs ly[t-511..t] finite
    origins = origins[np.isfinite(ly[origins - CTX + 1])]
    if quick:
        origins = origins[:30]
    log(f"{len(origins)} forecast origins, {df.observation_date.iloc[origins[0]].date()} .. {df.observation_date.iloc[origins[-1]].date()}")
    stamps = pd.bdate_range("2000-01-03", periods=CTX)
    ctx = pd.concat([pd.DataFrame({"id": t, "timestamp": stamps, "target": ly[t - CTX + 1:t + 1]}) for t in origins], ignore_index=True)
    t0 = time.perf_counter()
    fc = V.pipeline().predict_df(ctx, prediction_length=H, quantile_levels=list(LEV), id_column="id", timestamp_column="timestamp",
                                 target="target", cross_learning=False, batch_size=256)
    sec = time.perf_counter() - t0
    fc = fc.sort_values(["id", "timestamp"])
    fc["step"] = fc.groupby("id").cumcount() + 1
    fc = fc[fc.step % BLOCK == 0]
    logq = np.stack([fc[fc.id == t][[str(q) for q in LEV]].to_numpy() for t in origins])          # (n_orig, 12, 21)
    realised = np.full((len(origins), M), np.nan)
    for i, t in enumerate(origins):
        if t + H <= n - 1:
            rr = r[t + 1:t + H + 1].reshape(M, BLOCK)
            realised[i] = np.sqrt(252.0 / BLOCK * (rr**2).sum(axis=1))
    garch = np.full(len(origins), np.nan)
    theta = np.array([1e-6, 0.08, 0.9])
    for i, t in enumerate(origins):
        if i % 4 == 0:
            theta = V.garch_fit(r[max(0, t - 999):t + 1], theta)
        garch[i] = math.sqrt(252 * V.garch_forecast(theta, r[max(0, t - 999):t + 1], H))
    np.savez_compressed(out / "curve_forecasts.npz", origins=origins, dates=df.observation_date.iloc[origins].dt.strftime("%Y-%m-%d").to_numpy(),
                        logq=logq, realised=realised, garch=garch)
    meta = {"stage": "forecast", "data_sha256": sha, "chronos_seconds": sec, "n_origins": int(len(origins)),
            "git_commit": V.git("rev-parse", "HEAD"), "config_last_commit": V.git("log", "-1", "--format=%H", "--", "configs/phase7_forecast_greeks.json"),
            "quick": quick}
    (out / "meta_forecast.json").write_text(json.dumps(meta, indent=2))


# --- stage 2: propagation (research environment) -----------------------------------------------------

def quantile_curves(logq_t, U):
    """Block volatilities (n, 12) from uniforms U (n, 12) through the per-block forecast quantiles."""
    q = np.maximum.accumulate(logq_t, axis=1)                         # enforce monotone quantiles
    Uc = np.clip(U, LEV[0], LEV[-1])
    return np.exp(np.stack([np.interp(Uc[:, j], LEV, q[j]) for j in range(M)], axis=1))


def pit(logq_t, sig):
    q = np.maximum.accumulate(logq_t, axis=1)
    return np.clip(np.array([np.interp(math.log(sig[j]), q[j], LEV) for j in range(M)]), 0.005, 0.995)


def sample_u(kind, n, rng, chol=None):
    from scipy.stats import norm
    if kind == "comonotone":
        return np.repeat(rng.random((n, 1)), M, axis=1)
    if kind == "independent":
        return rng.random((n, M))
    return norm.cdf(rng.standard_normal((n, M)) @ chol.T)


def _one_origin(args):
    """All curves for one evaluation origin (both strikes, all copulas). Deterministic given its seeds."""
    from scipy import special
    from scipy.stats import qmc
    from mc_greeks import term_structure as ts
    from mc_greeks.asian import AsianDigital
    from mc_greeks.simulation import make_generator
    i, origin, date, logq_i, realised_i, garch_i, past_z, n_rep = args
    seed = CFG["seed"]
    T, r0, S0 = H / 252.0, CFG["contract"]["r"], CFG["contract"]["S0"]
    strikes = {"atm": S0, "otm": 1.05 * S0}
    eps = np.finfo(float).eps
    rng = make_generator([seed, 1, int(origin)])
    # common random numbers: with the 'naive' construction the standard-normal increments Z do not
    # depend on the curve (tests/test_term_structure.py), so every curve at this origin uses the same Z
    p1 = ts.TSParams(S0, r0, T, (1.0,) * M)
    L0 = ts.construction_matrix("naive", p1)
    Zs = []
    for s in range(8):
        xi = special.ndtri(np.clip(qmc.Sobol(M - 1, scramble=True, seed=rng).random(4096), eps, 1 - eps))
        Zs.append(ts.increments_from_residual(xi @ L0.T, p1))
    Zall = np.concatenate(Zs)                                          # (8 * 4096, 12); split by scrambling after

    def values(curves, K, bucket=False):
        f = AsianDigital(K, M)
        res = np.empty((len(curves), 4, 8))
        bv = None
        for c_i, sig in enumerate(curves):
            o = ts.cpw_first(f, ts.TSParams(S0, r0, T, tuple(sig)), Zall, bucket=bucket and c_i == 0)
            for g_i, g in enumerate(GREEKS):
                res[c_i, g_i] = o[g].reshape(8, -1).mean(axis=1)
            if bucket and c_i == 0:
                bv = o["bucket_vega"].mean(axis=0)
        return res, bv

    med = quantile_curves(logq_i, np.full((1, M), 0.5))[0]
    flat = np.full(M, math.sqrt(np.mean(med**2)))
    chol = None
    if len(past_z) >= 52:
        C = 0.9 * np.corrcoef(past_z.T) + 0.1 * np.eye(M)
        chol = np.linalg.cholesky(C)
    rows, refs = [], []
    for k_name, K in strikes.items():
        base, bv = values([med, flat, np.full(M, garch_i), realised_i], K, bucket=True)
        ref = dict(origin=int(origin), date=date, strike=k_name, n_past=len(past_z),
                   sigma_med_mean=float(med.mean()), sigma_realised_mean=float(realised_i.mean()), sigma_garch=float(garch_i))
        for j, nm in enumerate(("median", "flat", "garch", "realised")):
            for g_i, g in enumerate(GREEKS):
                ref[f"{nm}_{g}"] = float(base[j, g_i].mean())
                ref[f"{nm}_{g}_se"] = float(base[j, g_i].std(ddof=1) / math.sqrt(8))
        refs.append(ref)
        for c_i, kind in enumerate(COPULAS):
            if kind == "empirical" and chol is None:
                continue
            crng = make_generator([seed, 2, int(origin), c_i])
            big = quantile_curves(logq_i, sample_u(kind, 4000, crng, chol))
            sd_lin = float(math.sqrt(max(bv @ np.cov(big.T) @ bv, 0.0)))
            curves = quantile_curves(logq_i, sample_u(kind, n_rep, crng, chol))
            v = values(curves, K)[0].mean(axis=2)                          # (n_rep, 4)
            row = dict(origin=int(origin), date=date, strike=k_name, copula=kind, n_curves=n_rep, sd_lin=sd_lin)
            for g_i, g in enumerate(GREEKS):
                row[f"{g}_sd"] = float(v[:, g_i].std(ddof=1))
                row[f"{g}_q05"], row[f"{g}_q50"], row[f"{g}_q95"] = np.quantile(v[:, g_i], [0.05, 0.5, 0.95]).tolist()
            row["mc_se_price_median_curve"] = ref["median_price_se"]
            row["covered_price"] = bool(row["price_q05"] <= ref["realised_price"] <= row["price_q95"])
            row["covered_delta"] = bool(row["delta_q05"] <= ref["realised_delta"] <= row["delta_q95"])
            rows.append(row)
    return rows, refs


def propagate_stage(out, quick, workers=8):
    from concurrent.futures import ProcessPoolExecutor
    from scipy.stats import norm
    z = np.load(out / "curve_forecasts.npz", allow_pickle=True)
    origins, dates, logq, realised, garch = z["origins"], z["dates"], z["logq"], z["realised"], z["garch"]
    months = pd.Series(pd.to_datetime(dates)).dt.to_period("M")
    ok = ~np.isnan(realised).any(axis=1)
    ev = [i for i in range(len(origins)) if ok[i] and dates[i] >= "2019-01-01" and (i == 0 or months[i] != months[i - 1])]
    if quick:
        ev = ev[:4]
    z_pit = np.full((len(origins), M), np.nan)
    for i in np.where(ok)[0]:
        z_pit[i] = norm.ppf(pit(logq[i], realised[i]))
    n_rep = 40 if quick else 200
    jobs = []
    for i in ev:
        past = [k for k in range(len(origins)) if ok[k] and origins[k] + H <= origins[i]]   # outcome observed before origin i
        jobs.append((i, origins[i], dates[i], logq[i], realised[i], garch[i], z_pit[past], n_rep))
    rows, refs = [], []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for k, (r_, f_) in enumerate(ex.map(_one_origin, jobs)):
            rows += r_
            refs += f_
            if (k + 1) % 10 == 0 or k + 1 == len(jobs):
                log(f"{k + 1}/{len(jobs)} origins done")
    pd.DataFrame(rows).to_csv(out / "bands.csv", index=False)
    pd.DataFrame(refs).to_csv(out / "reference_values.csv", index=False)
    (out / "meta_propagate.json").write_text(json.dumps({"stage": "propagate", "n_eval_origins": len(ev), "quick": quick, "workers": workers,
                                                         "seconds": time.perf_counter() - t0, "git_commit": V.git("rev-parse", "HEAD")}, indent=2))


def analyse(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    B = pd.read_csv(out / "bands.csv")
    Rf = pd.read_csv(out / "reference_values.csv")
    B["ratio_lin"] = B.sd_lin / B.price_sd
    v = {}
    e1 = B.groupby(["strike", "copula"]).ratio_lin.apply(lambda x: float(((x >= 0.8) & (x <= 1.25)).mean()))
    v["E1_share_ratio_in_0.8_1.25"] = {f"{k[0]}_{k[1]}": round(x, 3) for k, x in e1.items()}
    v["E1_ratio_median"] = {f"{k[0]}_{k[1]}": round(float(x), 3) for k, x in B.groupby(["strike", "copula"]).ratio_lin.median().items()}
    v["E1_pass"] = bool((e1 >= 0.8).all())
    w = B.pivot_table(index=["origin", "strike"], columns="copula", values="price_sd").reset_index()
    e2 = (w.comonotone / w.independent).groupby(w.strike).median()
    v["E2_median_sd_comonotone_over_independent"] = {k: round(float(x), 3) for k, x in e2.items()}
    v["E2_pass"] = bool((e2 >= 2).all())
    cov = B.groupby(["strike", "copula"]).agg(cov_price=("covered_price", "mean"), cov_delta=("covered_delta", "mean"), n=("covered_price", "size"))
    v["E3_coverage_90pct_band"] = {f"{k[0]}_{k[1]}": {"price": round(float(r.cov_price), 3), "delta": round(float(r.cov_delta), 3), "n": int(r.n)} for k, r in cov.iterrows()}
    emp = cov.xs("empirical", level="copula") if "empirical" in cov.index.get_level_values(1) else None
    v["E3_pass"] = bool(emp is not None and ((emp.cov_price >= 0.80) & (emp.cov_price <= 0.97)).all())
    for nm in ("flat", "garch"):
        for g in ("price", "delta"):
            rel = (Rf[f"median_{g}"] - Rf[f"{nm}_{g}"]) / Rf[f"{nm}_{g}"].abs()
            v[f"E4_median_vs_{nm}_{g}_rel_diff"] = {s: {"median_abs": round(float(rel[Rf.strike == s].abs().median()), 4),
                                                      "p90_abs": round(float(rel[Rf.strike == s].abs().quantile(0.9)), 4)} for s in ("atm", "otm")}
    v["mc_se_vs_band"] = {s: round(float((B[(B.strike == s) & (B.copula == "independent")].mc_se_price_median_curve /
                                          B[(B.strike == s) & (B.copula == "independent")].price_sd).median()), 4) for s in ("atm", "otm")}
    (out / "verdict.json").write_text(json.dumps(v, indent=2))
    log(json.dumps(v))
    fd = out / "figures"
    fd.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), squeeze=False)
    for ax, s in zip(axes[:, 0], ("atm", "otm")):
        d = B[(B.strike == s)]
        rf = Rf[Rf.strike == s].set_index("date")
        for kind, c in (("comonotone", "#9a9890"), ("empirical", "#2a78d6"), ("independent", "#1baf7a")):
            x = d[d.copula == kind].set_index("date")
            ax.fill_between(pd.to_datetime(x.index), x.price_q05, x.price_q95, color=c, alpha=0.3, lw=0, label=f"90% band, {kind}")
        ax.plot(pd.to_datetime(rf.index), rf.realised_price, "k.", ms=3, label="value under the realised curve")
        ax.plot(pd.to_datetime(rf.index), rf.median_price, color="#e34948", lw=0.8, label="median curve")
        ax.set_title(f"Asian digital ({s.upper()}), model value under Chronos-2 curve uncertainty"); ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(fd / "fig7E1_bands.png", bbox_inches="tight"); fig.savefig(fd / "fig7E1_bands.pdf", bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for kind, c in (("comonotone", "#9a9890"), ("empirical", "#2a78d6"), ("independent", "#1baf7a")):
        x = B[B.copula == kind].ratio_lin
        ax.hist(np.log(x), bins=40, alpha=0.5, color=c, label=kind)
    ax.axvline(math.log(0.8), color="k", ls=":"); ax.axvline(math.log(1.25), color="k", ls=":")
    ax.set_xlabel("log(sd first-order / sd resampled), price"); ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fd / "fig7E2_linearisation.png", bbox_inches="tight"); fig.savefig(fd / "fig7E2_linearisation.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["forecast", "propagate", "analyse"])
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    out = outdir(a.quick)
    if a.stage == "forecast":
        forecast_stage(out, a.quick)
    elif a.stage == "propagate":
        propagate_stage(out, a.quick)
        if not a.quick:
            analyse(out)
        else:
            b = pd.read_csv(out / "bands.csv")
            print("quick run:", len(b), "band rows; NaN:", bool(b.isna().any().any()), "(analysis deliberately not run)")
    else:
        analyse(out)

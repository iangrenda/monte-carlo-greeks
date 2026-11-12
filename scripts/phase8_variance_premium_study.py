"""Phase 8B: do volatility forecasts predict realised-minus-implied variance (a VIX-based variance-swap payoff)?

    HF_HUB_OFFLINE=1 ~/.venvs/chronos/Scripts/python scripts/phase8_variance_premium_study.py forecast   # all forecasts
    ~/.venvs/chronos/Scripts/python scripts/phase8_variance_premium_study.py analyse                      # tests + figures
    ... --quick: plumbing check (NOT evidence; computes no scores)

Protocol: configs/phase8_variance_premium.json (committed before any forecast was scored).
Research only; frictionless; not tradable as stated. Output: results/phase8_variance_premium/<run>/
(forecasts.csv.gz contains VIX levels and is not committed).
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import phase7_vol_forecast_study as V  # noqa: E402  (FRED loader, trailing RV, GARCH, Chronos pipeline)

CFG = json.loads((ROOT / "configs" / "phase8_variance_premium.json").read_text())
FORECASTERS = ["hist", "ewma", "garch", "har", "chronos", "chronos-vix", "comb"]
CTX, LAGS, NBOOT, MIN_CAL = 512, 21, 2000, 250


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def windows(dates):
    """n_t: number of returns in (t, t + 30 calendar days], from the trading calendar."""
    d = dates.to_numpy().astype("datetime64[D]")
    end = np.searchsorted(d, d + np.timedelta64(30, "D"), side="right") - 1
    return end - np.arange(len(d))                                     # end index - t


def forecast_stage(out, quick):
    out.mkdir(parents=True, exist_ok=True)
    df, sha = V.load()
    r = df.r.to_numpy()
    n = len(r)
    nt = windows(df.observation_date)
    c2 = np.concatenate([[0.0], np.cumsum(r**2)])
    # a window is complete only if date_t + 30 calendar days is inside the data (searchsorted clamps at the end)
    dd = df.observation_date.to_numpy().astype("datetime64[D]")
    complete = dd + np.timedelta64(30, "D") <= dd[-1]
    realised = np.array([c2[t + 1 + nt[t]] - c2[t + 1] if complete[t] else np.nan for t in range(n)])
    origins = np.arange(CTX + 22, n)
    origins = origins[complete[origins]]
    if quick:
        origins = origins[-60:]
    log(f"{len(origins)} origins {df.observation_date.iloc[origins[0]].date()} .. {df.observation_date.iloc[origins[-1]].date()}")
    avg = lambda k, t: float(np.mean(r[t - k + 1:t + 1] ** 2))
    ev = V.ewma_var(r)
    F = {k: np.full(len(origins), np.nan) for k in FORECASTERS}
    theta = np.array([1e-6, 0.08, 0.9])
    # HAR on the log of the average daily variance in the window
    feats = lambda s: [1.0, math.log(max(r[s] ** 2, 1e-10)), math.log(avg(5, s)), math.log(avg(22, s))]
    coef, s_e = None, None
    for i, t in enumerate(origins):
        m = nt[t]
        F["hist"][i] = avg(21, t) * m
        F["ewma"][i] = ev[t] * m
        if i % 21 == 0:
            theta = V.garch_fit(r[max(0, t - 999):t + 1], theta)
            past = [s for s in range(22, t) if s + nt[s] <= t]             # windows ended by t
            X = np.array([feats(s) for s in past])
            y = np.log(realised[past] / nt[past])
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            s_e = float(np.std(y - X @ coef, ddof=4))
        w, a, b = theta
        s2 = V._garch_filter(theta, r[max(0, t - 999):t + 1])[-1]
        lr = w / (1 - a - b)
        F["garch"][i] = float(np.sum(lr + (a + b) ** np.arange(m) * (s2 - lr)))
        F["har"][i] = math.exp(coef @ feats(t) + 0.5 * s_e**2) * m
    log("classical forecasts done")
    for name, vix in (("chronos", False), ("chronos-vix", True)):
        t0 = time.perf_counter()
        Q = V.chronos_quantiles(df, 21, origins, vix)                    # step-21 quantiles of 21-day RV (annualised vol)
        F[name] = V.mean_square_from_quantiles(Q) / 252 * nt[origins]
        log(f"{name}: {time.perf_counter() - t0:.0f} s")
    F["comb"] = (F["garch"] + F["har"] + F["chronos-vix"]) / 3
    iv = (df.VIXCLS.to_numpy()[origins] / 100) ** 2 * 30 / 365
    res = pd.DataFrame({"origin": origins, "date": df.observation_date.iloc[origins].dt.strftime("%Y-%m-%d").to_numpy(),
                        "n": nt[origins], "end": origins + nt[origins], "rv": realised[origins], "iv": iv,
                        "vix": df.VIXCLS.to_numpy()[origins], **{f"F_{k}": v for k, v in F.items()}})
    res.to_csv(out / "forecasts.csv.gz", index=False)
    meta = {"data_sha256": sha, "first_date": str(df.observation_date.iloc[0].date()), "last_date": str(df.observation_date.iloc[-1].date()),
            "git_commit": V.git("rev-parse", "HEAD"), "config_last_commit": V.git("log", "-1", "--format=%H", "--", "configs/phase8_variance_premium.json"),
            "quick": quick, "n_origins": int(len(origins))}
    import torch, chronos
    meta.update(python=sys.version.split()[0], torch=torch.__version__, chronos=getattr(chronos, "__version__", "?"), numpy=np.__version__, pandas=pd.__version__)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))


# --- statistics ----------------------------------------------------------------------------------

def nw_var(x, lags=LAGS):
    """Newey-West long-run variance of the mean of x (Bartlett kernel)."""
    u = x - x.mean()
    n = len(u)
    s = u @ u / n
    for k in range(1, lags + 1):
        s += 2 * (1 - k / (lags + 1)) * (u[k:] @ u[:-k]) / n
    return max(s, 1e-300) / n


def clark_west(y, f_bench, f_model, lags=LAGS):
    """One-sided Clark-West test of nested forecasts: H1 = the larger model improves MSPE."""
    adj = (y - f_bench) ** 2 - ((y - f_model) ** 2 - (f_bench - f_model) ** 2)
    z = adj.mean() / math.sqrt(nw_var(adj, lags))
    return float(z), float(stats.norm.sf(z))


def block_boot_idx(n, rng, block=LAGS):
    k = int(math.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=k)
    return (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i])
        adj[i] = min(run, 1.0)
    return adj


def oos_payoff_forecasts(d, cols, lag=0):
    """Expanding OLS of P on [1, s...] using only origins whose window ended by t; plus the no-signal mean."""
    P = d.P.to_numpy()
    S = d[cols].to_numpy() if cols else np.zeros((len(d), 0))
    if lag:
        S = np.vstack([np.full((lag, S.shape[1]), np.nan), S[:-lag]])
    ends, org = d.end.to_numpy(), d.origin.to_numpy()
    ph, pb = np.full(len(d), np.nan), np.full(len(d), np.nan)
    for i in range(len(d)):
        ok = np.where(ends[:i] <= org[i])[0]                              # payoffs known at origin i
        ok = ok[np.isfinite(S[ok]).all(axis=1)] if S.shape[1] else ok
        if len(ok) < MIN_CAL or not np.isfinite(S[i]).all():
            continue
        X = np.column_stack([np.ones(len(ok)), S[ok]])
        beta, *_ = np.linalg.lstsq(X, P[ok], rcond=None)
        ph[i] = beta @ np.concatenate([[1.0], S[i]])
        pb[i] = P[ok].mean()
    return ph, pb


def analyse(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = pd.read_csv(out / "forecasts.csv.gz", parse_dates=["date"])
    d["P"] = d.rv - d.iv
    for k in FORECASTERS:
        d[f"s_{k}"] = d[f"F_{k}"] - d.iv
    hold0 = pd.Timestamp(CFG["periods"]["holdout"][0])
    post0 = pd.Timestamp(CFG["periods"]["post_release_subperiod"][0])
    hold_start_idx = int(d.origin[d.date >= hold0].min())
    period = np.where(d.date >= hold0, "holdout", np.where(d.end < hold_start_idx, "development", "purged"))
    d["period"] = period
    rng = np.random.default_rng(CFG["seed"])
    v = {"n_origins": {p: int((d.period == p).sum()) for p in ("development", "holdout", "purged")}}

    # V5: variance risk premium sanity
    v["V5_mean_payoff_annualised"] = {}
    for p in ("development", "holdout"):
        x = d.P[d.period == p].to_numpy()
        t = x.mean() / math.sqrt(nw_var(x))
        v["V5_mean_payoff_annualised"][p] = {"mean": round(float(x.mean() * 365 / 30), 5), "t_hac": round(float(t), 2),
                                             "p_one_sided": round(float(stats.norm.cdf(t)), 4), "share_negative": round(float((x < 0).mean()), 3)}
    v["V5_pass"] = all(x["p_one_sided"] < 0.05 for x in v["V5_mean_payoff_annualised"].values())

    # forecast loss and calibration
    ql = lambda rv, f: rv / f - np.log(rv / f) - 1
    v["qlike"] = {p: {k: round(float(ql(d.rv[m], d[f"F_{k}"][m]).mean()), 4) for k in FORECASTERS}
                  for p, m in (("development", d.period == "development"), ("holdout", d.period == "holdout"), ("post_release", d.date >= post0))}

    # V1, V2: out-of-sample payoff predictability
    rows, pv1, pv2 = [], [], []
    for lag in (0, 1):
        for k in FORECASTERS:
            ph, pb = oos_payoff_forecasts(d, [f"s_{k}"], lag)
            d[f"Ph_{k}_lag{lag}"], d[f"Pb_lag{lag}"] = ph, pb
            for p in ("development", "holdout"):
                m = (d.period == p).to_numpy() & np.isfinite(ph)
                y, fb, fm = d.P.to_numpy()[m], pb[m], ph[m]
                r2 = 1 - np.sum((y - fm) ** 2) / np.sum((y - fb) ** 2)
                z, pcw = clark_west(y, fb, fm)
                hit_m, hit_b = np.mean(np.sign(fm) == np.sign(y)), np.mean(np.sign(fb) == np.sign(y))
                diffs = []
                for _ in range(NBOOT):
                    idx = block_boot_idx(len(y), rng)
                    diffs.append(np.mean(np.sign(fm[idx]) == np.sign(y[idx])) - np.mean(np.sign(fb[idx]) == np.sign(y[idx])))
                diffs = np.array(diffs)
                p_hit = float(np.mean(diffs <= 0))
                rows.append(dict(lag=lag, forecaster=k, period=p, n=int(m.sum()), oos_r2=float(r2), cw_z=z, cw_p=pcw,
                                 hit_signal=float(hit_m), hit_nosignal=float(hit_b), hit_diff=float(hit_m - hit_b), hit_p=p_hit))
    T = pd.DataFrame(rows)
    h0 = T[(T.lag == 0) & (T.period == "holdout")].set_index("forecaster").loc[FORECASTERS]
    T.loc[h0.index.map(lambda k: T[(T.lag == 0) & (T.period == "holdout") & (T.forecaster == k)].index[0]), "cw_p_holm"] = holm(h0.cw_p.to_numpy())
    T.loc[h0.index.map(lambda k: T[(T.lag == 0) & (T.period == "holdout") & (T.forecaster == k)].index[0]), "hit_p_holm"] = holm(h0.hit_p.to_numpy())
    T.to_csv(out / "predictability.csv", index=False)
    hh = T[(T.lag == 0) & (T.period == "holdout")]
    v["V1"] = {r.forecaster: {"oos_r2": round(r.oos_r2, 4), "cw_p": round(r.cw_p, 4), "cw_p_holm": round(r.cw_p_holm, 4)} for r in hh.itertuples()}
    v["V1_pass"] = bool((hh.cw_p_holm < 0.05).any())
    v["V2"] = {r.forecaster: {"hit_signal": round(r.hit_signal, 3), "hit_nosignal": round(r.hit_nosignal, 3), "p_holm": round(r.hit_p_holm, 4)} for r in hh.itertuples()}
    v["V2_pass"] = bool((hh.hit_p_holm < 0.05).any())
    hl = T[(T.lag == 1) & (T.period == "holdout")]
    v["robustness_lagged_V1"] = {r.forecaster: {"oos_r2": round(r.oos_r2, 4), "cw_p": round(r.cw_p, 4)} for r in hl.itertuples()}
    dv = T[(T.lag == 0) & (T.period == "development")]
    v["development_oos_r2"] = {r.forecaster: round(r.oos_r2, 4) for r in dv.itertuples()}

    # V3: Chronos-VIX beyond GARCH (nested)
    pg, _ = oos_payoff_forecasts(d, ["s_garch"])
    pgc, _ = oos_payoff_forecasts(d, ["s_garch", "s_chronos-vix"])
    m = (d.period == "holdout").to_numpy() & np.isfinite(pg) & np.isfinite(pgc)
    z, p = clark_west(d.P.to_numpy()[m], pg[m], pgc[m])
    v["V3"] = {"cw_z": round(z, 3), "cw_p": round(p, 4),
               "oos_r2_vs_garch_only": round(float(1 - np.sum((d.P.to_numpy()[m] - pgc[m]) ** 2) / np.sum((d.P.to_numpy()[m] - pg[m]) ** 2)), 4)}
    v["V3_pass"] = bool(p < 0.05)

    # V4: regimes (ex-ante VIX terciles from development)
    cuts = np.quantile(d.vix[d.period == "development"], [1 / 3, 2 / 3])
    d["regime"] = np.digitize(d.vix, cuts)
    reg = []
    for p in ("development", "holdout"):
        for g in range(3):
            for k in FORECASTERS:
                m = (d.period == p).to_numpy() & (d.regime == g).to_numpy() & np.isfinite(d[f"Ph_{k}_lag0"]).to_numpy()
                if m.sum() < 30:
                    continue
                y, fm, fb = d.P.to_numpy()[m], d[f"Ph_{k}_lag0"].to_numpy()[m], d["Pb_lag0"].to_numpy()[m]
                reg.append(dict(period=p, regime=["low", "mid", "high"][g], forecaster=k, n=int(m.sum()),
                                oos_r2=float(1 - np.sum((y - fm) ** 2) / np.sum((y - fb) ** 2)), mean_P_annualised=float(y.mean() * 365 / 30)))
    R = pd.DataFrame(reg)
    R.to_csv(out / "regimes.csv", index=False)
    v["V4_vix_cuts"] = [round(float(c), 2) for c in cuts]
    v["V4_holdout_oos_r2"] = {f"{r.regime}_{r.forecaster}": round(r.oos_r2, 3) for r in R[R.period == "holdout"].itertuples()}

    # calibration of 80% intervals for RV (empirical log-ratio quantiles from completed past windows)
    cal = {}
    for k in FORECASTERS:
        lr = np.log(d.rv / d[f"F_{k}"]).to_numpy()
        ends, org = d.end.to_numpy(), d.origin.to_numpy()
        cov = np.full(len(d), np.nan)
        for i in range(len(d)):
            ok = np.where(ends[:i] <= org[i])[0][-500:]
            if len(ok) >= 250:
                lo, hi = np.quantile(lr[ok], [0.1, 0.9])
                cov[i] = float(lo <= lr[i] <= hi)
        cal[k] = {p: round(float(np.nanmean(cov[(d.period == p).to_numpy()])), 3) for p in ("development", "holdout")}
    v["calibration_80pct_empirical"] = cal

    # frictionless illustration (no hypothesis)
    first = d.groupby(d.date.dt.to_period("M")).head(1)
    ill = {}
    for k in ("garch", "chronos-vix", "comb"):
        x = first[(first.period == "holdout") & np.isfinite(first[f"Ph_{k}_lag0"])]
        pos = np.where(x[f"Ph_{k}_lag0"] > 0, 1.0, -1.0)
        pnl, short = pos * x.P.to_numpy() * 365 / 30, -x.P.to_numpy() * 365 / 30
        ill[k] = {"months": int(len(x)), "share_long": round(float((pos > 0).mean()), 3), "mean": round(float(pnl.mean()), 5),
                  "sharpe_monthly": round(float(pnl.mean() / pnl.std(ddof=1)), 3), "worst": round(float(pnl.min()), 5),
                  "always_short_mean": round(float(short.mean()), 5), "always_short_sharpe_monthly": round(float(short.mean() / short.std(ddof=1)), 3),
                  "always_short_worst": round(float(short.min()), 5)}
    v["illustration_frictionless_holdout_monthly"] = ill
    (out / "verdict.json").write_text(json.dumps(v, indent=2, default=float))
    log(json.dumps({k: v[k] for k in v if k.endswith("pass")}))

    fd = out / "figures"
    fd.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    ann = 365 / 30
    axes[0].plot(d.date, d.iv * ann, color="#1f1f1e", lw=0.8, label="implied variance (VIX², annualised)")
    axes[0].plot(d.date, d.rv * ann, color="#e34948", lw=0.6, label="realised variance over the next 30 days (annualised)")
    axes[0].plot(d.date, d["F_garch"] * ann, color="#2a78d6", lw=0.6, label="GARCH forecast")
    axes[0].set_yscale("log"); axes[0].legend(fontsize=7); axes[0].axvline(hold0, color="k", ls=":")
    axes[1].plot(d.date, d.P * ann, color="#9a9890", lw=0.6, label="payoff P = realised - implied (annualised)")
    axes[1].plot(d.date, d["Ph_comb_lag0"] * ann, color="#1baf7a", lw=0.8, label="out-of-sample forecast of P (comb)")
    axes[1].axhline(0, color="k", lw=0.6); axes[1].axvline(hold0, color="k", ls=":"); axes[1].legend(fontsize=7)
    axes[1].set_ylim(np.nanpercentile(d.P * ann, 0.5), np.nanpercentile(d.P * ann, 99.5))
    fig.tight_layout()
    fig.savefig(fd / "fig8B1_variance_premium.png", bbox_inches="tight"); fig.savefig(fd / "fig8B1_variance_premium.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["forecast", "analyse"])
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    out = ROOT / "results" / "phase8_variance_premium" / ("quick" if a.quick else "study")
    if a.stage == "forecast":
        forecast_stage(out, a.quick)
        if a.quick:
            f = pd.read_csv(out / "forecasts.csv.gz")
            print("quick run:", len(f), "rows; NaN in forecasts:", bool(f.filter(like="F_").isna().any().any()),
                  "n range", int(f.n.min()), int(f.n.max()), "(no scores computed)")
    else:
        if a.quick:
            sys.exit("analysis is not run in quick mode")
        analyse(out)

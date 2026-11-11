"""Phase 7D: walk-forward forecasts of S&P 500 realised volatility, Chronos-2 vs classical models.

Runs in the separate environment with PyTorch and chronos-forecasting:

    ~/.venvs/chronos/Scripts/python scripts/phase7_vol_forecast_study.py            # full run
    ~/.venvs/chronos/Scripts/python scripts/phase7_vol_forecast_study.py --quick    # plumbing check (NOT evidence)
    ~/.venvs/chronos/Scripts/python scripts/phase7_vol_forecast_study.py --analysis-only

Protocol: configs/phase7_vol_forecast.json (committed before any forecast was computed).
Data: FRED SP500 and VIXCLS, downloaded to data/fred/ (not committed). Output: results/phase7_vol_forecast/<run>/.
"""
import argparse
import hashlib
import json
import math
import platform
import subprocess
import time
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, signal, stats

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs" / "phase7_vol_forecast.json").read_text())
DATA = ROOT / "data" / "fred"
HORIZONS = CFG["target"]["horizons"]
CTX = 512
CHRONOS_Q = [0.01, 0.05] + [round(0.1 * k, 2) for k in range(1, 10)] + [0.95, 0.99]
CHRONOS_Q = sorted(set(CHRONOS_Q + [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]))   # the model's 21 native levels
LEVELS = CFG["interval_forecasts"]["levels"]
CLASSICAL = ["ewma", "garch", "har"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --- data -----------------------------------------------------------------------------------------

def fetch(series):
    DATA.mkdir(parents=True, exist_ok=True)
    f = DATA / f"{series}.csv"
    if not f.exists():
        req = urllib.request.Request(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}",
                                     headers={"User-Agent": "mc-greeks-research/0.1"})
        f.write_bytes(urllib.request.urlopen(req, timeout=60).read())
    return f


def load():
    fs, fv = fetch("SP500"), fetch("VIXCLS")
    sha = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (fs, fv)}
    s = pd.read_csv(fs, na_values=["."], parse_dates=["observation_date"]).dropna().reset_index(drop=True)
    v = pd.read_csv(fv, na_values=["."], parse_dates=["observation_date"]).dropna()
    s = pd.merge_asof(s, v, on="observation_date", direction="backward")
    s["r"] = np.log(s.SP500).diff()
    s = s.dropna().reset_index(drop=True)
    return s, sha


def trailing_rv(r, h):
    """sqrt(252/h * sum of the last h squared returns), aligned so element t includes r_t."""
    c = np.concatenate([[0.0], np.cumsum(r**2)])
    out = np.full(len(r), np.nan)
    out[h - 1:] = np.sqrt(252.0 / h * (c[h:] - c[:-h]))
    return out


# --- classical forecasters (variance per day, unannualised, unless stated) ------------------------

def ewma_var(r, lam=0.94):
    v = np.empty(len(r))
    s2 = r[:21].var()
    for t in range(len(r)):
        s2 = lam * s2 + (1 - lam) * r[t] ** 2        # sigma^2_{t+1 | t}
        v[t] = s2
    return v


def _garch_filter(theta, r):
    """sigma^2_t for t = 0..len(r) (the last entry is the one-step forecast), sigma^2_0 = sample variance."""
    w, a, b = theta
    x = w + a * r**2                                   # drives sigma^2_{t+1}
    s0 = r.var()
    s2, _ = signal.lfilter([1.0], [1.0, -b], x, zi=[b * s0])
    return np.concatenate([[s0], s2])


def garch_nll(theta, r):
    w, a, b = theta
    if w <= 0 or a < 0 or b < 0 or a + b >= 0.9999:
        return 1e10
    s2 = _garch_filter(theta, r)[:-1]
    return 0.5 * np.sum(np.log(s2) + r**2 / s2)


def garch_fit(r, x0):
    res = optimize.minimize(garch_nll, x0, args=(r,), method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-8, "maxiter": 4000})
    return res.x


def garch_forecast(theta, r, h):
    """Average daily variance over the next h days given returns up to t (r[-1] = r_t)."""
    w, a, b = theta
    s2 = _garch_filter(theta, r)[-1]                   # sigma^2_{t+1 | t}
    lr = w / (1 - a - b)
    k = np.arange(h)
    return float(np.mean(lr + (a + b) ** k * (s2 - lr)))


def classical_forecasts(df, h, origins, quick):
    r = df.r.to_numpy()
    n = len(r)
    out = {k: np.full(len(origins), np.nan) for k in ("hist", "ewma", "garch", "har", "har_s")}
    rv_h, rv1, rv5, rv22 = trailing_rv(r, h), np.sqrt(252) * np.maximum(np.abs(r), 1e-4), trailing_rv(r, 5), trailing_rv(r, 22)
    ev = ewma_var(r)
    theta, coef, s_res = np.array([1e-6, 0.08, 0.9]), None, None
    target = np.full(n, np.nan)
    target[: n - h] = rv_h[h:]                         # RV over (t, t+h] stored at t
    for i, t in enumerate(origins):
        out["hist"][i] = rv_h[t] ** 2
        out["ewma"][i] = 252 * ev[t]
        if i % 21 == 0:
            theta = garch_fit(r[max(0, t - 999):t + 1], theta)
            past = np.arange(22, t - h + 1)           # days whose target is observed by t
            X = np.column_stack([np.ones(len(past)), np.log(rv1[past]), np.log(rv5[past]), np.log(rv22[past])])
            yv = np.log(target[past])
            coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
            s_res = float(np.std(yv - X @ coef, ddof=4))
        out["garch"][i] = 252 * garch_forecast(theta, r[max(0, t - 999):t + 1], h)
        mu = coef @ [1.0, math.log(rv1[t]), math.log(rv5[t]), math.log(rv22[t])]
        out["har"][i] = math.exp(2 * mu + 2 * s_res**2)
    return out, target


# --- Chronos-2 ------------------------------------------------------------------------------------

_PIPE = None


def pipeline():
    """Load amazon/chronos-2 once. The weights are cached locally after the first download; set
    HF_HUB_OFFLINE=1 to skip the Hub's online checks (they retry with back-off when offline)."""
    global _PIPE
    if _PIPE is None:
        from chronos import Chronos2Pipeline
        _PIPE = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
    return _PIPE


def chronos_quantiles(df, h, origins, with_vix):
    pipe = pipeline()
    y = trailing_rv(df.r.to_numpy(), h)
    vix = df.VIXCLS.to_numpy() / 100
    stamps = pd.bdate_range("2000-01-03", periods=CTX)
    frames = []
    for t in origins:
        d = {"id": t, "timestamp": stamps, "target": y[t - CTX + 1:t + 1]}
        if with_vix:
            d["vix"] = vix[t - CTX + 1:t + 1]
        frames.append(pd.DataFrame(d))
    ctx = pd.concat(frames, ignore_index=True)
    fc = pipe.predict_df(ctx, prediction_length=h, quantile_levels=CHRONOS_Q, id_column="id", timestamp_column="timestamp",
                         target="target", cross_learning=False, batch_size=256)
    fc = fc.sort_values(["id", "timestamp"])
    last = fc.groupby("id").tail(1).set_index("id").loc[origins]
    return np.column_stack([last[str(q)].to_numpy() for q in CHRONOS_Q])        # (n_origins, 21), volatility units


def mean_square_from_quantiles(Q):
    """E[Y^2] from 21 quantiles: piecewise-linear quantile function, constant beyond 0.01 / 0.99."""
    u = np.linspace(0.0, 1.0, 981)
    vals = np.array([np.interp(np.clip(u, 0.01, 0.99), CHRONOS_Q, q) for q in Q])
    return (vals**2).mean(axis=1)


# --- run --------------------------------------------------------------------------------------------

def git(*a):
    try:
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as exc:  # pragma: no cover
        return f"unavailable ({exc})"


def simulate(out, quick):
    out.mkdir(parents=True, exist_ok=True)
    df, sha = load()
    import torch
    import chronos
    meta = {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
            "chronos": getattr(chronos, "__version__", "unknown"), "numpy": np.__version__, "pandas": pd.__version__,
            "git_commit": git("rev-parse", "HEAD"), "config_last_commit": git("log", "-1", "--format=%H", "--", "configs/phase7_vol_forecast.json"),
            "data_sha256": sha, "first_date": str(df.observation_date.iloc[0].date()), "last_date": str(df.observation_date.iloc[-1].date()),
            "quick": quick}
    rows = []
    for h in HORIZONS:
        first = CTX + 22
        start = int(np.searchsorted(df.observation_date.to_numpy(), np.datetime64("2019-01-02")))
        origins = np.arange(max(first, start), len(df) - h)
        if quick:
            origins = origins[:40]
        log(f"h={h}: {len(origins)} origins from {df.observation_date.iloc[origins[0]].date()} to {df.observation_date.iloc[origins[-1]].date()}")
        cl, target = classical_forecasts(df, h, origins, quick)
        t0 = time.perf_counter()
        qc = chronos_quantiles(df, h, origins, False)
        t1 = time.perf_counter()
        qv = chronos_quantiles(df, h, origins, True)
        t2 = time.perf_counter()
        meta[f"chronos_seconds_h{h}"] = [t1 - t0, t2 - t1]
        d = pd.DataFrame({"h": h, "origin": origins, "date": df.observation_date.iloc[origins].dt.date.astype(str).to_numpy(),
                          "rv": target[origins], "vix": df.VIXCLS.iloc[origins].to_numpy() / 100,
                          **{f"var_{k}": v for k, v in cl.items() if k != "har_s"}})
        d["var_chronos"] = mean_square_from_quantiles(qc)
        d["var_chronos-vix"] = mean_square_from_quantiles(qv)
        for j, q in enumerate(CHRONOS_Q):
            d[f"q_chronos_{q}"] = qc[:, j]
            d[f"q_chronos-vix_{q}"] = qv[:, j]
        rows.append(d)
        log(f"h={h} done")
    pd.concat(rows, ignore_index=True).to_csv(out / "forecasts.csv.gz", index=False)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))


# --- analysis -----------------------------------------------------------------------------------------

def qlike(rv2, f):
    x = rv2 / f
    return x - np.log(x) - 1


def pinball(y, q, tau):
    d = y - q
    return np.maximum(tau * d, (tau - 1) * d)


def dm(d, lag):
    """Diebold-Mariano: mean of d over its Newey-West (Bartlett, lag) standard error; two-sided p."""
    d = d - 0.0
    n = len(d)
    u = d - d.mean()
    s = u @ u / n
    for k in range(1, lag + 1):
        s += 2 * (1 - k / (lag + 1)) * (u[k:] @ u[:-k]) / n
    se = math.sqrt(max(s, 1e-300) / n)
    z = d.mean() / se
    return float(z), float(2 * stats.norm.sf(abs(z)))


def analyse(out):
    F = pd.read_csv(out / "forecasts.csv.gz", parse_dates=["date"])
    models = ["hist", "ewma", "garch", "har", "chronos", "chronos-vix"]
    rows, cov, tests = [], [], []
    for h, d in F.groupby("h"):
        d = d.sort_values("origin").reset_index(drop=True)
        rv2 = d.rv.to_numpy() ** 2
        L = {mname: qlike(rv2, d[f"var_{mname}"].to_numpy()) for mname in models}
        M = {mname: (rv2 - d[f"var_{mname}"].to_numpy()) ** 2 for mname in models}
        if h == 21:
            L["vix"], M["vix"] = qlike(rv2, d.vix.to_numpy() ** 2), (rv2 - d.vix.to_numpy() ** 2) ** 2
        # interval forecasts
        P, C = {}, {}
        for mname in models:
            if mname.startswith("chronos"):
                Qs = np.column_stack([d[f"q_{mname}_{q}"] for q in LEVELS])
            else:
                pt = np.sqrt(d[f"var_{mname}"].to_numpy())
                lr = np.log(d.rv.to_numpy() / pt)
                Qs = np.full((len(d), len(LEVELS)), np.nan)
                for i in range(len(d)):
                    past = lr[max(0, i - h - 500 + 1):max(0, i - h + 1)]       # origins whose target was observed by origin i
                    if len(past) >= 250:
                        Qs[i] = pt[i] * np.exp(np.quantile(past, LEVELS))
            P[mname] = np.mean([pinball(d.rv.to_numpy(), Qs[:, k], tau) for k, tau in enumerate(LEVELS)], axis=0)
            C[mname] = (d.rv.to_numpy() >= Qs[:, 0]) & (d.rv.to_numpy() <= Qs[:, -1])
            C[mname] = np.where(np.isnan(Qs[:, 0]), np.nan, C[mname])
        for pname, (a, b) in CFG["periods"].items():
            mask = (d.date >= pd.Timestamp(a)) & ((d.date <= pd.Timestamp(b)) if b != "end" else True)
            mask = np.asarray(mask)
            for mname in L:
                rows.append(dict(h=h, period=pname, model=mname, n=int(mask.sum()), qlike=float(L[mname][mask].mean()),
                                 mse=float(M[mname][mask].mean()),
                                 pinball=float(np.nanmean(P[mname][mask])) if mname in P else np.nan,
                                 pinball_n=int(np.isfinite(P[mname][mask]).sum()) if mname in P else 0,
                                 coverage80=float(np.nanmean(C[mname][mask])) if mname in C else np.nan))
            for a_ in ("chronos", "chronos-vix"):
                for b_ in [x for x in L if x != a_]:
                    z, p = dm(L[a_][mask] - L[b_][mask], h - 1)
                    both = mask & np.isfinite(P.get(a_, np.full(len(d), np.nan))) & np.isfinite(P.get(b_, np.full(len(d), np.nan)))
                    zp, pp = dm(P[a_][both] - P[b_][both], h - 1) if (b_ in P and both.sum() > 50) else (np.nan, np.nan)
                    tests.append(dict(h=h, period=pname, model=a_, versus=b_, qlike_diff=float((L[a_][mask] - L[b_][mask]).mean()),
                                      dm_z=z, dm_p=p, pinball_dm_z=zp, pinball_dm_p=pp))
    S, T = pd.DataFrame(rows), pd.DataFrame(tests)
    S.to_csv(out / "scores.csv", index=False)
    T.to_csv(out / "dm_tests.csv", index=False)

    v = {}
    full = S[S.period == "full"].set_index(["h", "model"])
    Tf = T[T.period == "full"].set_index(["h", "model", "versus"])
    v["F1"] = {f"h{h}": [round(full.loc[(h, "chronos"), "qlike"], 4), round(full.loc[(h, "hist"), "qlike"], 4),
                         round(Tf.loc[(h, "chronos", "hist"), "dm_p"], 4)] for h in HORIZONS}
    v["F1_pass"] = bool(all(full.loc[(h, "chronos"), "qlike"] < full.loc[(h, "hist"), "qlike"] and Tf.loc[(h, "chronos", "hist"), "dm_p"] < 0.05 for h in HORIZONS))
    best = {h: min(CLASSICAL, key=lambda mm: full.loc[(h, mm), "qlike"]) for h in HORIZONS}
    v["F2"] = {f"h{h}": {"best_classical": best[h], "chronos_over_best": round(full.loc[(h, "chronos"), "qlike"] / full.loc[(h, best[h]), "qlike"], 4),
                         "dm_p": round(Tf.loc[(h, "chronos", best[h]), "dm_p"], 4)} for h in HORIZONS}
    v["F2_pass"] = bool(all(full.loc[(h, "chronos"), "qlike"] <= 1.05 * full.loc[(h, best[h]), "qlike"] for h in HORIZONS))
    v["F3"] = {"h21": [round(full.loc[(21, "chronos-vix"), "qlike"], 4), round(full.loc[(21, "chronos"), "qlike"], 4),
                       round(Tf.loc[(21, "chronos-vix", "chronos"), "dm_p"], 4)]}
    v["F3_pass"] = bool(full.loc[(21, "chronos-vix"), "qlike"] < full.loc[(21, "chronos"), "qlike"] and Tf.loc[(21, "chronos-vix", "chronos"), "dm_p"] < 0.05)
    v["F4"] = {f"h{h}": round(full.loc[(h, "chronos"), "coverage80"], 4) for h in HORIZONS}
    v["F4_pass"] = bool(all(0.75 <= full.loc[(h, "chronos"), "coverage80"] <= 0.85 for h in HORIZONS))
    post = S[S.period == "post-release"].set_index(["h", "model"])
    v["F5"] = {f"h{h}": {"full_sign": float(np.sign(full.loc[(h, "chronos"), "qlike"] - full.loc[(h, best[h]), "qlike"])),
                         "post_sign": float(np.sign(post.loc[(h, "chronos"), "qlike"] - post.loc[(h, best[h]), "qlike"])),
                         "post_n": int(post.loc[(h, "chronos"), "n"])} for h in HORIZONS}
    v["F5_pass"] = bool(all(v["F5"][f"h{h}"]["full_sign"] == v["F5"][f"h{h}"]["post_sign"] for h in HORIZONS))
    (out / "verdict.json").write_text(json.dumps(v, indent=2))
    log(json.dumps(v))
    figures(out, F, S)


def figures(out, F, S):
    fd = out / "figures"
    fd.mkdir(exist_ok=True)
    col = {"hist": "#9a9890", "ewma": "#eda100", "garch": "#e34948", "har": "#1baf7a", "chronos": "#2a78d6", "chronos-vix": "#7fb2ea", "vix": "#1f1f1e"}
    fig, axes = plt.subplots(len(HORIZONS), 1, figsize=(11, 3.4 * len(HORIZONS)), squeeze=False)
    for ax, h in zip(axes[:, 0], HORIZONS):
        d = F[F.h == h].sort_values("date")
        ax.plot(d.date, d.rv, color="#1f1f1e", lw=0.7, label=f"realised vol, next {h} days")
        ax.fill_between(d.date, d["q_chronos_0.1"], d["q_chronos_0.9"], color=col["chronos"], alpha=0.25, lw=0, label="Chronos-2 80% interval")
        ax.plot(d.date, d["q_chronos_0.5"], color=col["chronos"], lw=0.8, label="Chronos-2 median")
        ax.plot(d.date, np.sqrt(d.var_garch), color=col["garch"], lw=0.8, label="GARCH(1,1)")
        ax.set_yscale("log"); ax.set_ylabel("annualised volatility"); ax.legend(fontsize=7, ncol=4); ax.set_title(f"h = {h} trading days")
    fig.tight_layout()
    fig.savefig(fd / "fig7D1_forecasts.png", bbox_inches="tight"); fig.savefig(fd / "fig7D1_forecasts.pdf", bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, len(HORIZONS), figsize=(11, 3.6), squeeze=False)
    for ax, h in zip(axes[0], HORIZONS):
        d = S[(S.h == h)].pivot_table(index="period", columns="model", values="qlike")
        d = d.reindex(list(CFG["periods"]))
        d.plot.bar(ax=ax, color=[col[c] for c in d.columns], width=0.85, legend=(h == HORIZONS[0]))
        ax.set_ylabel("mean QLIKE (lower is better)"); ax.set_title(f"h = {h}"); ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(fd / "fig7D2_qlike_by_period.png", bbox_inches="tight"); fig.savefig(fd / "fig7D2_qlike_by_period.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--analysis-only", action="store_true")
    a = ap.parse_args()
    out = ROOT / "results" / "phase7_vol_forecast" / ("quick" if a.quick else "study")
    t0 = time.time()
    if not a.analysis_only:
        simulate(out, a.quick)
        meta = json.loads((out / "meta.json").read_text())
        meta["wall_seconds_total"] = time.time() - t0
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
    if a.quick:
        f = pd.read_csv(out / "forecasts.csv.gz")
        print("quick run: columns", len(f.columns), "rows", len(f), "any NaN in forecasts:",
              bool(f.filter(like="var_").isna().any().any()), "(scores deliberately not computed)")
    else:
        analyse(out)

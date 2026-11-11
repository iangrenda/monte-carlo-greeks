"""Phase 7D feasibility: does Chronos-2 run here, and is it sensible on a volatility-like series
whose true forecast distribution is known?

Run with the separate environment that has PyTorch and chronos-forecasting:

    ~/.venvs/chronos/Scripts/python scripts/phase7_chronos_smoke.py

Synthetic data (no market data): log-volatility x_t follows a Gaussian AR(1),
    x_t = mu + phi (x_{t-1} - mu) + eta eps_t,
so the h-step forecast of x is N(mu + phi^h (x_T - mu), eta^2 (1 - phi^(2h)) / (1 - phi^2)).
Chronos-2 forecasts sigma_t = exp(x_t) zero-shot (quantiles are monotone-invariant, so the true
sigma quantiles are exp of the x quantiles). Reported: quantile-loss of Chronos-2 vs the true
conditional quantiles and vs a naive "last value" forecast, and coverage of its 80% interval.
This is a smoke test of plumbing and gross calibration only; it says nothing about real markets.
Output: results/phase7_chronos/smoke/.
"""
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

from chronos import Chronos2Pipeline

OUT = Path(__file__).resolve().parents[1] / "results" / "phase7_chronos" / "smoke"
SEED, N_SERIES, CONTEXT, H = 20270601, 64, 512, 21
MU, PHI, ETA = np.log(0.2), 0.97, 0.08
Q = [0.1, 0.25, 0.5, 0.75, 0.9]


def simulate(rng, n, length):
    x = np.empty((n, length))
    x[:, 0] = MU + rng.standard_normal(n) * ETA / np.sqrt(1 - PHI**2)
    for t in range(1, length):
        x[:, t] = MU + PHI * (x[:, t - 1] - MU) + ETA * rng.standard_normal(n)
    return x


def pinball(y, q, tau):
    d = y - q
    return np.maximum(tau * d, (tau - 1) * d)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    x = simulate(rng, N_SERIES, CONTEXT + H)
    ctx, fut = x[:, :CONTEXT], x[:, CONTEXT:]
    ts = pd.date_range("2020-01-01", periods=CONTEXT, freq="B")
    df = pd.DataFrame({"id": np.repeat(np.arange(N_SERIES), CONTEXT), "timestamp": np.tile(ts, N_SERIES),
                       "target": np.exp(ctx).ravel()})
    t0 = time.perf_counter()
    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
    load_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    fc = pipe.predict_df(df, prediction_length=H, quantile_levels=Q, id_column="id", timestamp_column="timestamp", target="target")
    pred_s = time.perf_counter() - t0
    fc = fc.sort_values(["id", "timestamp"])
    qcols = [str(q) for q in Q] if str(Q[0]) in fc.columns else [c for c in fc.columns if c not in ("id", "timestamp", "target_name", "predictions")]
    chron = np.stack([fc[c].to_numpy().reshape(N_SERIES, H) for c in qcols], axis=-1)       # (n, H, q) in sigma units
    h = np.arange(1, H + 1)
    m = MU + PHI**h * (ctx[:, -1:] - MU)
    s = ETA * np.sqrt((1 - PHI ** (2 * h)) / (1 - PHI**2))
    true_q = np.exp(m[..., None] + s[None, :, None] * norm.ppf(Q))                           # (n, H, q)
    naive_q = np.repeat(np.exp(ctx[:, -1:])[..., None], len(Q), axis=-1) * np.ones((1, H, 1))
    y = np.exp(fut)[..., None]
    loss = {name: float(np.mean([pinball(y[..., 0], arr[..., k], tau) for k, tau in enumerate(Q)]))
            for name, arr in (("chronos2", chron), ("true_conditional", true_q), ("naive_last_value", naive_q))}
    cover = float(np.mean((y[..., 0] >= chron[..., 0]) & (y[..., 0] <= chron[..., -1])))
    med_err = np.abs(np.log(chron[..., 2]) - m).mean(axis=0)                                  # log-median error by horizon
    res = {"python": platform.python_version(), "torch": torch.__version__, "load_seconds": load_s, "predict_seconds": pred_s,
           "series": N_SERIES, "context": CONTEXT, "horizon": H, "quantile_columns_used": qcols,
           "mean_pinball_loss": loss, "loss_ratio_chronos_over_true": loss["chronos2"] / loss["true_conditional"],
           "loss_ratio_naive_over_true": loss["naive_last_value"] / loss["true_conditional"],
           "coverage_80pct_interval": cover, "abs_log_median_error_h1_h5_h21": [float(med_err[0]), float(med_err[4]), float(med_err[-1])]}
    (OUT / "smoke.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

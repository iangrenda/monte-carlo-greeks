# Phase 8 research design: from Greek estimators to an options research engine

**Final question.** Do probabilistic volatility forecasts improve financially meaningful option
trading or hedging decisions, once market-implied volatility, forecast uncertainty,
transaction costs and tail risk are accounted for?

**Status of this phase.** Research only. There is no live trading, no broker connection and
no order submission. Nothing here is investment advice. Backtested results, if any are ever
produced, are paper results, not account profits.

## A. Audit of the repository (2026-11-11)

**Starting point.** `main` at `da88a2c4de121ee2754c582fc38fbf3853931ac2` (PRs #1–#6 merged).
The Phase 8 work is on `phase8/variance-premium`. The environment is Python 3.12.10 on
Windows 11, research venv `~/.venvs/mc-greeks`, Chronos venv `~/.venvs/chronos`
(`requirements-chronos.txt`).

**Tests.** `pytest`: 416 passed locally, and GitHub Actions passed on `da88a2c`.

### A.1 Point-in-time alignment of the Phase 7 data

Checked in the code (`scripts/phase7_vol_forecast_study.py`,
`scripts/phase7_forecast_greeks_study.py`):

| Item | Finding |
|---|---|
| Origin information | Every forecast at origin t uses returns r_1..r_t (close-to-close up to day t) and, for the VIX covariate, VIX closes up to day t. Targets start at r_{t+1}. |
| GARCH | Fitted on up to the last 1,000 returns ending at t; refitted every 21 origins. No future data. |
| HAR | Regressed on days s whose target window ended by t (s + h ≤ t). No overlap with the target. |
| Baseline intervals | Built from forecast errors of origins i − h − 500 + 1 … i − h, all observed by i. |
| 7E copula | Estimated from forecast origins k with origin_k + 60 ≤ origin_i (outcomes fully observed). |
| VIX timing | VIXCLS is the VIX close at 16:15 ET; the S&P 500 close is at 16:00. Using VIX_t at day t's close is a 15-minute peek into the first target return. Negligible for forecasting, but Phase 8 also runs every test with all information lagged one day. |
| Data vintage | FRED serves the last 10 years of SP500 (the licence), so the start date moves over time. Phase 7 recorded SHA-256 hashes. Index closes are not revised. |

**Verdict.** No look-ahead in Phase 7, apart from the 15-minute VIX timing.

### A.2 Phase 7 forecasting metrics recomputed from saved output

Recomputed with new code from `results/phase7_vol_forecast/study/forecasts.csv.gz` (local;
not committed):

| QLIKE | hist | EWMA | GARCH | HAR | Chronos-2 | Chronos-2 + VIX |
|---|---|---|---|---|---|---|
| h = 5 | 0.9192 | 0.5004 | 0.4449 | 0.4705 | 0.4843 | 0.4525 |
| h = 21 | 0.6214 | 0.5908 | 0.5015 | 0.5158 | 0.5303 | 0.5015 |

Chronos-2's 80% interval coverage is 0.7294 (h = 5) and 0.7497 (h = 21). All of these
match the Phase 7 report exactly.

### A.3 Risk-neutral valuation versus physical-measure analysis

| Where | Measure | What it is |
|---|---|---|
| Phases 1–6 (all estimators) | risk-neutral (drift r, given σ) | Model prices and Greeks of contracts under Black–Scholes. They are *valuations* for an assumed σ, not market prices. The Asian and basket digitals are research contracts, not exchange-traded. |
| 7A–7B | risk-neutral with a deterministic σ(t) | Numerical study of estimators. |
| 7D | physical | Forecasts of realised volatility (what will happen). |
| 7E | hybrid | Risk-neutral drift r with physical forecasts of σ(t): "model value if volatility follows the forecast". Scenario analysis, not a price. |
| Phase 8 implied variance (VIX², option IVs) | risk-neutral | What the market charges. |

The gap between the last two rows is the subject of Phase 8. On average, implied variance
exceeds realised variance (the **variance risk premium**), which is compensation for crash
and volatility risk. A forecast can be accurate and still give no trading advantage if it
does not predict *realised minus implied*.

### A.4 Dependence, intervals and calibration (known limitations carried forward)

* **Chronos-2 calibration.** Intervals are too narrow (80% nominal covers 73–75%); lower
  quantiles can be negative on the raw volatility scale; crash-time tails explode. Phase 7E
  forecasts log-volatility to stay positive.
* **Dependence across horizons.** Chronos-2 gives marginal forecasts per step. Independence
  across blocks gave 58–60% coverage of a nominal 90% band; an empirical Gaussian copula from
  past errors gave 89%. Any Phase 8 scenario distribution must model this dependence
  (Part F).
* **Confidence intervals.**
  * Phase 7D used Diebold–Mariano tests with Newey–West variances (lag h − 1); overlapping
    targets make naive standard errors too small.
  * Phase 7E's copula used overlapping past forecast windows, so its correlation estimate is
    noisier than its sample size suggests.
* **Realised-variance proxy.** Daily squared close-to-close returns, with no intraday data.
  This is noisy but unbiased for close-to-close variance, which is also what a
  close-settled variance contract would pay.

### A.5 Numerical accuracy is not a tradable advantage

Phases 1–6 make *given-model* quantities cheaper and more precise. A better estimator of
the Delta of a model contract changes nothing about whether that model's price differs from
the market's. A trading advantage needs three things:
1. an ex-ante signal that predicts *market* outcomes (here, realised minus implied variance);
2. a way to act on it at observed prices (bid/ask, hedge instruments, costs);
3. an edge that survives tail risk and is confirmed out of sample.

Phase 7 established none of these. Its best result is that Chronos-2 + VIX matches GARCH as
a forecaster of realised volatility.

For exchange-traded S&P 500 options (European, cash-settled), prices and Greeks are
available in closed form. The Monte Carlo machinery is not needed to price them and will be
used only where it adds something: verification, scenario distributions, uncertainty
propagation, and non-standard payoffs (Part H).

## B. Plan and execution order

1. **Variance-risk-premium study (public data only).**
   * Protocol: `configs/phase8_variance_premium.json`, frozen before scoring.
   * Script: `scripts/phase8_variance_premium_study.py`.
   * This is the gate. If no forecaster predicts realised-minus-implied variance out of
     sample, there is no public-data support for the Part E strategies.
2. **Option data schema and engine with unit tests, on synthetic fixtures only.**
   * `src/mc_greeks/market_data.py`: canonical schema, CSV ingestion, quote validation.
   * `src/mc_greeks/implied_vol.py`: Black-76 / Black–Scholes European pricing on the
     forward, implied-volatility inversion, Greeks.
   * `src/mc_greeks/options_backtest.py`: fills at bid/ask, multiplier, fees, slippage,
     cash settlement, hedge ledger, reconciliation.
3. **Data availability.** Determine which historical executable option data can actually be
   obtained (§C below). No purchase without explicit approval.
4. **A small backtest** (Parts E–G), only with licensed point-in-time data and a protocol
   frozen on development data.
5. **A completely separate out-of-sample evaluation.**
6. **Paper-trading plan**, only after credible net results.

## C. Data availability (initial findings, 2026-11-11)

| Source | What it offers (as described by the vendor) | Notes |
|---|---|---|
| WRDS / OptionMetrics IvyDB US | daily SPX option bid/ask, implied volatilities, Greeks, zero curve, dividend yields | **Cornell access not confirmed by search.** Check the Cornell Library database list or ask a business librarian. If available, this is the standard academic source. |
| Cboe DataShop, "Option Quotes End of Day with Calcs" | bid/ask at 15:45 and the close, from 2012 | Paid. Index underlying bid/ask needs a separate Cboe index licence. |
| HistoricalOptionData.com, "ALLSPX" | end-of-day SPX bid/ask from 1990 (OPRA source) | Paid; price not listed on the page. |
| ORATS | historical bid/ask quotes, IVs and Greeks for US options | Paid; price tiers by user type. |

Sources: [Cboe DataShop EOD with calcs](https://datashop.cboe.com/option-quotes-end-of-day-with-calcs),
[HistoricalOptionData ALLSPX](https://www.historicaloptiondata.com/product/allspx-sp-500),
[ORATS historical quotes](https://blog.orats.com/historical-options-quotes-and-greeks),
[WRDS OptionMetrics](https://wrds-www.wharton.upenn.edu/pages/about/data-vendors/optionmetrics).
The vendor details have not been verified beyond these pages.

**Rule.** No market data is scraped, and no licensed data is committed or redistributed.
The engine is tested on clearly labelled synthetic fixtures; synthetic data never produce
"historical" results.

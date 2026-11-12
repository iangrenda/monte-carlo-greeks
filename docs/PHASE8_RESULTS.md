# Phase 8 results (first stage): the variance-premium gate, and the options engine

**Research only; not investment advice.** Nothing here was traded, paper-traded or connected to
a broker. "P&L" figures are frictionless illustrations, not account results.

## 1. The question this stage answers

Before buying option data or building strategies, test the cheapest necessary condition: does
any volatility forecast predict **realised minus implied** variance out of sample? If not,
strategies that trade "forecast volatility minus implied volatility" have no support from
public data.

**Set-up** (`configs/phase8_variance_premium.json`, frozen in `4559901` before any score was
computed; `scripts/phase8_variance_premium_study.py`):
* **Data.** FRED S&P 500 closes and VIX closes, 2016-11-14 to 2026-11-09 (SHA-256 in
  `results/phase8_variance_premium/study/meta.json`).
* **Horizons aligned exactly.** The VIX refers to the next **30 calendar days** and is
  annualised on a 365-day basis, so implied total variance is IV_t = (VIX_t/100)²·30/365.
  Realised total variance RV_t is the sum of squared close-to-close returns from day t to the
  last close within date_t + 30 days (19–22 returns; the exchange calendar fixes the count in
  advance).
* **Payoff.** P_t = RV_t − IV_t: a long 30-day variance swap struck at VIX², before costs. A
  frictionless approximation; the VIX is not a traded strike.
* **Signals.** s_t = F_t − IV_t for seven forecasters:
  * historical, EWMA, GARCH(1,1), HAR;
  * Chronos-2, and Chronos-2 with the VIX as a past covariate;
  * an equal-weight combination of GARCH, HAR and Chronos-2 + VIX.
* **Out-of-sample payoff forecast.** P̂_t = a + b·s_t, fitted only on payoffs already known at
  t. The "no signal" benchmark is the mean of those known payoffs.
* **Splits.** 1,955 daily origins from 2018-12-31 to 2026-10-09.
  * Development: 1,240 origins (the first 250+ are used only for calibration).
  * Holdout: 696 origins from 2024-01-01.
  * 19 development origins whose windows reach into the holdout are purged.
* **Inference.** Newey–West (21 lags) and moving-block bootstrap (block 21), because daily
  origins overlap; Holm correction across the seven forecasters.

## 2. Results

**V1: is the payoff predictable out of sample? No.**

| Holdout, 2024-01 to 2026-10 | out-of-sample R² vs no-signal | Clark–West p | Holm-adjusted |
|---|---|---|---|
| historical | −0.032 | 0.52 | 1.00 |
| EWMA | −0.039 | 0.47 | 1.00 |
| GARCH(1,1) | −0.007 | 0.64 | 1.00 |
| HAR | +0.007 | 0.13 | 0.88 |
| Chronos-2 | −0.000 | 0.62 | 1.00 |
| Chronos-2 + VIX | −0.000 | 0.61 | 1.00 |
| combination | −0.000 | 0.62 | 1.00 |

With all information lagged one day (removing the 15-minute VIX timing advantage) the
picture is the same: best R² +0.007 (HAR), p = 0.17.

**V2: direction. No improvement.** The no-signal benchmark is right 86.1% of the time,
because the payoff is negative in most months and the benchmark always predicts negative.
* The Chronos-2 and combination forecasts never predicted a positive payoff in the holdout, so
  they matched the benchmark exactly.
* The historical, EWMA and GARCH forecasts sometimes predicted positive and did worse
  (79.7–84.1%).

**V3: does Chronos-2 + VIX add to GARCH? No** (Clark–West p = 0.69; out-of-sample R² −0.019
relative to GARCH alone).

**V5: is there a variance risk premium?** Yes in the typical month, but the mean is not
significant in development:

| | mean payoff (annualised variance) | HAC t | share of months with RV < IV |
|---|---|---|---|
| development | −0.0072 | −0.70 | 84.5% |
| holdout | −0.0091 | −1.99 (p = 0.023) | 86.1% |

* The median implied volatility over the sample was 18.0% against a median realised
  volatility of 13.3%.
* The development mean is dragged toward zero by March 2020, where single windows paid up to
  +0.84 in annualised variance (realised volatility far above the VIX).
* As written, V5 fails on the development period. This is the textbook shape of the premium:
  a small, frequent gain for the seller and rare, very large losses.

**V4: regimes.** Holdout out-of-sample R² by ex-ante VIX tercile (cut-offs 17.1 and 22.8
from development): GARCH +0.003, +0.003 and −0.032 (low, middle, high); Chronos-2 + VIX and
the combination ≈ 0 in every tercile. No regime shows predictability.

**Forecast quality and calibration** (30-calendar-day total variance, QLIKE):

| | historical | EWMA | GARCH | HAR | Chronos-2 | Chronos-2 + VIX | combination |
|---|---|---|---|---|---|---|---|
| development | 0.698 | 0.674 | 0.580 | 0.619 | 0.560 | **0.511** | 0.514 |
| holdout | 0.447 | 0.395 | 0.315 | **0.296** | 0.461 | 0.478 | 0.342 |
| post-release (2025-11 on) | 0.221 | 0.180 | **0.154** | 0.181 | 0.257 | 0.323 | 0.216 |

* Chronos-2 + VIX was the best forecaster in development and among the worst in the holdout.
* The empirically calibrated 80% intervals covered 82–89% in development but only 60–74% in
  the holdout. The calm 2024–2026 period was harder to calibrate than the development errors
  suggested.

**A Chronos-2 weakness reappears.** Chronos-2's crash-time upper tails (Phase 7D) made its
variance forecasts explode in March 2020. Payoff regressions calibrated on that period gave
absurd development forecasts (out-of-sample R² −8,768 for Chronos-2 and −21,487 for
Chronos-2 + VIX). In the holdout the fitted slope is near zero, so the forecast collapses to
the benchmark.

**Frictionless illustration (no hypothesis).** On the first trading day of each holdout month
(34 months), the "long if P̂ > 0, else short" rule was short in every month for GARCH,
Chronos-2 + VIX and the combination. It is therefore identical to always selling variance:
* mean +0.0086 annualised variance per month, monthly Sharpe ratio 0.24;
* worst month −0.18 annualised variance.

That is the premium itself, not a forecast. It comes with exactly the crash exposure that
compensates it, and it is not tradable as stated (no costs, no traded strike).

**Kill criterion: triggered.** V1 fails for every forecaster. The public data give no support
to strategies that trade forecast-minus-implied variance. Better volatility forecasts,
including Chronos-2, did not translate into better predictions of what the market's implied
variance would turn out to over- or under-state.

## 3. Interpretation

1. **Accuracy is not edge.** HAR and GARCH forecast realised variance well (QLIKE about 0.3),
   yet their forecasts minus VIX² predict the payoff no better than its historical mean.
   The VIX already contains what these forecasts know, and more.
2. **The only robust pattern is the unconditional premium.** Selling variance earned on
   average and lost badly in crashes. Harvesting it is a risk-taking decision with large
   tail exposure, not a forecasting skill. It needs no model from this repository.
3. **Consequence for Parts E–G.** Strategy 1 ("long volatility when the forecast exceeds
   implied") has no empirical support here. The forecasts almost never exceeded implied
   variance, and when the simpler models said so, they were wrong more often than not. A
   real option backtest would need an independent rationale, for example hedging-cost
   reduction rather than directional volatility bets, and a new frozen protocol.

## 4. Infrastructure built (tested on synthetic fixtures only)

| Module | What it does | Tests |
|---|---|---|
| `src/mc_greeks/implied_vol.py` | Black-76 pricing on the forward; spot Delta, Gamma, Vega; no-arbitrage bounds; Brent implied volatility; forward from put–call parity | matches the analytical Black–Scholes module to 1e-12; parity; finite-difference Greeks; implied-volatility round trip; bounds |
| `src/mc_greeks/market_data.py` | Canonical point-in-time quote schema; CSV ingestion; time to AM/PM settlement; validation flags (missing, crossed, zero ask, negative bid, bad strike, expired, stale, wide, below intrinsic); point-in-time contract selection; **labelled synthetic** chains | each defect flagged; CSV round trip; selection ignores later snapshots |
| `src/mc_greeks/options_backtest.py` | Cash-flow ledger: buys at ask + slippage and sells at bid − slippage, commissions, multipliers, cash settlement, hedge proxy with its own unit delta, ratio and costs, daily delta hedging, mark-to-market reconciliation | hand-checked histories: long call to settlement (+$378.70), short put expiring (+$294.35), round-trip cost, a 4-day delta-hedged call (−$505.65, with the day-1 mark) |

Tests: 16 for the engine, 4 point-in-time tests for the study helpers. The full suite: 436
passed.

## 5. Data availability

See `docs/PHASE8_RESEARCH_DESIGN.md` §C.
* **The cheapest credible route** is WRDS / OptionMetrics, *if* Cornell's subscription
  includes it. That needs checking with the Cornell Library; a web search did not confirm it.
* **Paid alternatives:** Cboe DataShop end-of-day quotes (from 2012), HistoricalOptionData
  ALLSPX (from 1990), ORATS.
* No data was bought or downloaded.

## 6. Not done in this stage, and why

* **Parts E–G** (strategy protocols, a backtest with inference) need licensed executable
  option quotes. The gate result also removes the main strategy's rationale.
* **Part F** (uncertainty-aware no-trade rules) presupposes a signal worth acting on.
* **Part I** (dashboard) would currently display only synthetic positions.

These remain designed but not run.

## 7. Limitations

* One index over ten years, with daily closes only. The holdout covers 34 months: enough to
  reject a strong effect, not a small one.
* VIX² is only approximately a variance-swap strike: discrete strikes, the 30-day
  interpolation, and the jump-related convexity difference.
* Realised variance from daily closes misses intraday moves; it is unbiased for
  close-to-close variance.
* Chronos-2 was used zero-shot only. Fine-tuning was not tested.

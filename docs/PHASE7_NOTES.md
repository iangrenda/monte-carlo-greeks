# Phase 7 (in progress): volatility term structures, then Chronos-2

**Goal.** Test whether time-varying volatility forecasts, eventually from Chronos-2, can be
used in the Monte Carlo Greek machinery of Phases 1–6, and whether they improve anything that
can be measured. The numerical part comes first, so the work stands even if the forecasting
part adds nothing.

**Status.**
* 7A–7C, the numerical generalisation, is done and validated.
* The first study (7B) has been run.
* Chronos-2 runs on this machine and has been compared with classical volatility forecasts on
  ten years of S&P 500 data (7D, §3b).

## 1. Mathematics (7A)

**Model.** dS_t = r S_t dt + σ(t) S_t dW_t, with σ(t) = σ_j on the j-th of m equal intervals.
The intervals are the Asian monitoring periods. The curve is fixed when the simulation
starts, so it is *deterministic*: a forecast is an input, not a random quantity. With
a_j = σ_j√Δt and Z_j i.i.d. N(0, 1),

  ln S_{t_i} = ln S₀ + μ_i + Σ_{j≤i} a_j Z_j,  μ_i = Σ_{j≤i} (r − σ_j²/2)Δt.

**European digital.** ln S_T is normal with variance V = Σ σ_j² Δt. So every Black–Scholes
digital formula holds with σ²T replaced by V. The likelihood-ratio (Malliavin) Delta weight
is the score of that normal density with respect to S₀:

  Δ = e^{−rT} E[ 1{S_T > K} · X / (S₀ V) ],  X = Σ_j a_j Z_j = ∫₀ᵀ σ(t) dW_t.

For a flat curve this is the familiar W_T/(S₀σT). It is tested for unbiasedness on four
curve shapes.

**Asian digital, CPW-first.** Every monitored price contains the same factor e^{a₁Z₁}, so
S_{t_i} = e^{a₁Z₁}R_i with R_i = S₀ exp(μ_i + B_i), where B_i = Σ_{j=2}^{i} a_j Z_j does not
depend on Z₁. Conditioning on Z₂…Z_m (Rao–Blackwell):

  P(A ≥ K | Z₂…Z_m) = Φ(δ),  δ = ln(Ā/K)/a₁,  Ā = mean_i R_i.

Per path:
* Delta = Dφ(δ)/(S₀a₁);
* Gamma = −Dφ(δ)(δ + a₁)/(S₀a₁)²;
* bucket vegas, by differentiating δ with Z held fixed:

  ∂/∂σ_k: Dφ(δ)·[c_k W_k − 1{k=1}·δ√Δt]/a₁,  c_k = −σ_kΔt + 1{k≥2}√Δt Z_k,  W_k = Σ_{i≥k} R_i/ΣR.

The m bucket vegas cost O(m) more work per path, from the same simulation. Their sum is the
parallel Vega, which for a flat curve equals the Phase-2 CPW-first Vega.

**Why bucket vegas matter for forecasting.** If the forecast curve has error ε (a vector), the
price error is approximately vegaᵀε, with variance vegaᵀ Σ_ε vega. The bucket-vega vector is
therefore exactly how forecast uncertainty per horizon propagates into price and Greek
uncertainty (§4).

**Path constructions.** B is a Brownian motion on the *variance clock* τ_k = Σ_{j=2}^{k} a_j².
Any L with LLᵀ = min(τ_k, τ_l) gives the same law:
* sequential;
* Brownian bridge on τ;
* **curve-aware PCA** (eigenvectors of that covariance);
* **naive PCA** (the constant-σ PCA of the increments, scaled by a_j; exact in law, but its
  ordering ignores the curve);
* the Greek-aware rotations of Phases 4–6, trained by finite-difference gradients or
  training-free (the Phase-6 weighted design-point surrogate, on the variance clock).

**Validation** (`tests/test_term_structure.py`, 66 tests). A flat curve reproduces the Phase-2
estimators and the Phase-3/6 constructions exactly. The European closed forms reduce to
Black–Scholes and match finite differences. The one-date Asian equals the European. Every
per-path Greek, including each bucket vega, equals the finite-difference derivative of the
per-path smoothed price. Estimators are unbiased against the raw payoff and under every
construction.

## 2. Study 7B: does the efficiency survive a volatility curve?

Protocol `configs/phase7_term_structure.json`, frozen in `9618248` before the run.
* 40 contracts: m ∈ {12, 52}; 5 shapes (flat, rising, falling, a single event spike at 3×,
  and a post-shock curve decaying from 2σ̄ to σ̄); σ̄ ∈ {0.2, 0.4}; K/S₀ ∈ {0.95, 1.05}.
* Fixed N (2¹⁰, 2¹²) with 32 SciPy LMS+shift scramblings; reference-free.
* Loss: mean over (Δ, Γ, parallel Vega) of variance / plain-MC per-path variance.
* Run: `python scripts/phase7_term_structure_study.py` (6.5 min). Output:
  `results/phase7_term_structure/study/`.

Joint loss relative to curve-aware PCA (geometric mean over 4 contracts and 2 N):

| m | shape | naive | Brownian bridge | sequential | gradient PCA, trained | gradient PCA, training-free |
|---|---|---|---|---|---|---|
| 12 | flat | 1.00 | 60 | 1,138 | 0.39 | 0.35 |
| 12 | rising | 0.28 | 72 | 427 | 0.16 | 0.12 |
| 12 | falling | 4.8 | 30 | 1,471 | 0.75 | 0.76 |
| 12 | event | 117 | 23 | 347 | 0.29 | 0.40 |
| 12 | reverting | 3.9 | 37 | 1,286 | 1.06 | 0.85 |
| 52 | flat | 0.98 | 79 | 347 | 0.16 | 0.16 |
| 52 | rising | 0.11 | 24 | 45 | 0.06 | 0.05 |
| 52 | falling | 11.4 | 127 | 1,235 | 0.32 | 0.39 |
| 52 | event | 36 | 47 | 150 | 0.12 | 0.20 |
| 52 | reverting | 17.9 | 107 | 1,153 | 0.51 | 0.48 |

**Findings.**

1. **Ignoring the curve can be catastrophic or helpful.** Reusing the constant-σ
   construction costs 36–130× for an event spike and 4–18× for front-loaded curves, but is
   **3–10× better** than curve-aware PCA for a rising curve. Ordering directions by *path
   variance* is not the right criterion: with a rising curve, curve-aware PCA puts its first
   coordinate where the variance is (late), while the Asian average and the conditioning
   care about the whole path.
2. **Greek-aware rotations are the robust choice.** They are never materially worse than
   curve-aware PCA (worst cell 1.06). Against whichever of curve-aware or naive PCA is
   better in each cell, the trained rotation gains 1.3–8.6× and the training-free one
   1.2–6.2×. The one exception is the post-shock curve at m = 12, where the trained
   rotation is 1.06× (no gain) and the training-free one 0.85×. Their gain is smallest
   for front-loaded curves at m = 12 (falling 0.75, post-shock 0.85–1.06), which is the
   shape closest to what a volatility forecast after a shock looks like.
3. **Training-free nearly matches training**, except for the event spike, where it is
   1.4–1.7× worse (geometric mean; at m = 52 and N = 2¹⁰ up to 2.1×). One very
   high-volatility bucket makes the single-point linearisation less accurate. The
   training-free rotation took 4–9 ms to build; finite-difference training on 4,096 paths
   took 0.1 s (m = 12) and 3 s (m = 52). These timings are informational: a PyTorch install
   was running at the same time.
4. **Bucket vegas gain less.** The joint (Δ, Γ, Vega) rotation reduces the bucket-vega loss
   too (0.17–0.81 relative to PCA), but always less than the joint loss, and not at all for
   the post-shock curve at m = 12 (1.11).
5. **Brownian bridge and sequential constructions are far worse** (15–2,950× in single
   cells). That is consistent with Phases 3–4, larger here because CPW-first plus PCA
   leaves very little variance.

**Hypotheses (pre-registered).**

| | hypothesis | outcome |
|---|---|---|
| T1 | naive = PCA for flat curves, worse for the event spike | the event part holds (×36–130, intervals far above 1). As written it **fails**: I required an *exact* ratio of 1 for flat curves, but the arms use independent random streams, so the ratio is 0.94–1.07 with intervals containing 1. That criterion was mis-specified. |
| T2 | Greek-aware ≤ 0.5× PCA in every cell | **fails**: falling and post-shock curves at m = 12 (0.75–1.06) and post-shock at m = 52 (0.48–0.51) |
| T3 | training-free ≤ 1.25× trained in every cell | **fails** for the event spike (1.37 and 1.71 pooled) |
| T4 | bucket gain < 1 and smaller than the joint gain | smaller everywhere; **fails** the "< 1" part in one cell (post-shock, m = 12: 1.11) |

## 3. Chronos-2 feasibility (7D)

* **Installation.** A separate environment (`~/.venvs/chronos`: Python 3.12.10, CPU PyTorch
  2.15.0, chronos-forecasting 2.3.2) installs and runs on this laptop. Windows Application
  Control did not block PyTorch. The research environment is untouched.
* **Test.** `~/.venvs/chronos/Scripts/python scripts/phase7_chronos_smoke.py` downloads
  `amazon/chronos-2` (120M parameters; 16 s to load) and forecasts 64 synthetic
  log-AR(1) volatility series with 512-day contexts, 21 days ahead (1.5 s on the CPU).
* **Results.**
  * Quantile (pinball) loss is 1.11× that of the *true* conditional quantiles; a "last
    value" forecast is 1.60×.
  * Its 80% interval covers 77% of outcomes.
  * Its median drifts from the true conditional median as the horizon grows (log error 0.02
    at 1 day, 0.11 at 21 days).
* This shows the plumbing works and the model is roughly calibrated on a clean
  mean-reverting process. It says nothing about real markets. Output:
  `results/phase7_chronos/smoke/smoke.json`.

## 3b. Real data: S&P 500 realised volatility (7D)

**Set-up.**
* Protocol `configs/phase7_vol_forecast.json`, frozen in `b53bed9` before any forecast was
  scored. A 40-origin plumbing check computed no scores and was deleted.
* Data: FRED SP500 (2016-11-14 to 2026-11-09, 2,511 trading days) and VIXCLS; SHA-256 in
  `meta.json`.
* Each trading day from 2019-01-02 forecasts realised volatility over the next h = 5 and 21
  days (1,970 and 1,954 origins).
* Models: Chronos-2 zero-shot on the trailing realised-volatility series (with and without the
  VIX as a past-only covariate; cross-learning off) against random walk, EWMA, GARCH(1,1) and
  log-HAR.
* Command:
  `HF_HUB_OFFLINE=1 ~/.venvs/chronos/Scripts/python scripts/phase7_vol_forecast_study.py`
  (4.5 min). Results in `results/phase7_vol_forecast/study/`. The per-day forecast file
  (which contains VIX levels) is regenerated by the script and not committed.

**QLIKE, full period** (lower is better; p-values are Diebold–Mariano with Newey–West
variance, Chronos-2 against each model):

| | h = 5 | Chronos-2 vs it (p) | h = 21 | Chronos-2 vs it (p) |
|---|---|---|---|---|
| random walk | 0.919 | better (< 0.001) | 0.621 | better, n.s. (0.15) |
| EWMA | 0.500 | n.s. (0.31) | 0.591 | n.s. (0.52) |
| GARCH(1,1) | **0.445** | **worse (0.003)** | 0.502 | n.s. (0.70) |
| HAR | 0.471 | n.s. (0.39) | 0.516 | n.s. (0.83) |
| Chronos-2 | 0.484 | | 0.530 | |
| Chronos-2 + VIX | 0.453 | better (< 0.001) | **0.502** | n.s. (0.49) |
| VIX itself (risk-neutral; reference) | — | | 0.430 | n.s. (0.11) |

**Findings.**

1. **Zero-shot Chronos-2 is a reasonable volatility forecaster, not a better one.** It is far
   better than the random walk at 5 days and statistically tied with EWMA and HAR. It is
   significantly worse than GARCH(1,1) at 5 days (9% higher QLIKE). At 21 days it is 6% worse
   than GARCH, which is not significant.
2. **The VIX as a covariate closes the gap.** Chronos-2 + VIX ties GARCH at both horizons
   (0.453 vs 0.445; 0.502 vs 0.502). At 5 days the gain over plain Chronos-2 is highly
   significant. At 21 days, the pre-registered horizon, it is not (p = 0.49).
3. **Its intervals are too narrow.** The 80% intervals cover 73% (5 days) and 75% (21 days).
   The empirically calibrated intervals of the classical models cover about 80%.
4. **It does not respect the nature of volatility.**
   * Lower quantiles can be negative, which is impossible for a volatility.
   * In a crash it extrapolates the spike. On 2020-03-16 its 21-day forecast had a median of
     166% and a 99th percentile of **31,800%** annualised (realised: 64%).
   * Twelve forecasts had a 99th percentile above 200%.
   * These tails make the variance MSE useless (3,132 against 0.007 for GARCH). QLIKE is
     barely affected.
   * Any curve fed into the pricer from Chronos-2 must be guarded: forecast log-volatility, or
     clip.
5. **No sign that pretraining data flattered it.** In the post-release year (2025-11 on,
   which the model cannot have seen) it is again worse than GARCH. At 21 days that is
   significant (p = 0.002), and it is also worse than HAR (p = 0.009).
6. **The VIX itself had the lowest 21-day QLIKE (0.430).** That is despite its risk premium:
   QLIKE penalises under-prediction more than over-prediction, so an upward-biased forecast
   is cheap.

**Hypotheses (pre-registered).**

| | hypothesis | outcome |
|---|---|---|
| F1 | Chronos-2 beats the random walk at both horizons (p < 0.05) | **fails** at 21 days (p = 0.15); holds at 5 |
| F2 | within 5% of the best classical model at both horizons | **fails**: 8.9% and 5.8% above GARCH |
| F3 | the VIX covariate helps at 21 days (p < 0.05) | **fails** (p = 0.49); it does help at 5 days |
| F4 | 80% interval coverage in [0.75, 0.85] | **fails**: 0.73 and 0.7497 |
| F5 | post-release ranking matches the full period | **passes**: worse than GARCH in both |

**Caveats.**
* Daily squared returns are a noisy proxy for true variance. There is no intraday data on
  FRED.
* The classical models' intervals need 250 past forecast errors, so their first year of
  origins (including the 2020 crash) has no interval score. Pinball comparisons are
  therefore made on common origins only.
* One index, one decade, zero-shot only. Fine-tuning, which drove the gains in Professor
  Xin's paper, was not tried.
* Forecasting log-volatility (which fixes the negative quantiles) was not pre-registered. It
  would need its own holdout.

## 3c. From forecast uncertainty to Greek uncertainty (7E)

**Question.** If the volatility curve over the next 60 trading days comes from Chronos-2, how
uncertain are an Asian digital's model value and Greeks? And is that uncertainty band
accurate and calibrated? This is physical-measure scenario analysis: values under forecast
volatility, not option prices.

**Design** (`configs/phase7_forecast_greeks.json`, frozen in `0048268` before any curve was
propagated).
* Contract: 12 monitoring dates, one every 5 trading days (T = 60/252), strikes at S₀ and
  1.05 S₀.
* Chronos-2 forecasts the **log** 5-day realised volatility (so curves stay positive) 60 days
  ahead, every 5th trading day (399 forecasts, 27 s). The step-5j quantiles give each 5-day
  block's forecast distribution.
* Chronos-2 does not say how blocks move *together*, so curves were sampled under three
  dependence assumptions:
  * **comonotone** (all blocks at the same quantile);
  * **independent**;
  * an **empirical Gaussian copula** whose correlation is estimated only from forecast errors
    already observed at the time (at least 52 needed, so 78 of 92 dates).
* Each of 200 curves per assumption was repriced with the CPW-first estimators (8 × 4,096
  scrambled points). Every curve at a date reuses the same random numbers: the "naive"
  construction makes the normals curve-independent (tested). Simulation error is
  0.01–0.04% of the band width.
* Each band is checked against the value under the volatility path that actually happened.
* Run: `HF_HUB_OFFLINE=1 ~/.venvs/chronos/Scripts/python scripts/phase7_forecast_greeks_study.py forecast`,
  then `~/.venvs/mc-greeks/Scripts/python scripts/phase7_forecast_greeks_study.py propagate`
  (8 min with 8 processes). Results in `results/phase7_forecast_greeks/study/`.

**Results** (92 monthly dates, 2019-01 to 2026-08).

| | at the money | 5% out of the money |
|---|---|---|
| 90% band width / value: comonotone, empirical, independent | 21%, 10%, 7% | 239%, 153%, 97% |
| 90% band width / Delta: comonotone, empirical, independent | 215%, 110%, 74% | 114%, 56%, 32% |
| coverage of the realised-curve value: comonotone, empirical, independent | 92%, 89%, 58% | 92%, 89%, 60% |
| first-order band within ±25% of resampled: comonotone, empirical, independent | 64%, 78%, 82% | 25%, 60%, 75% |

1. **The dependence assumption dominates the band.** The comonotone band is 3.5× (at the
   money) and 2.0× (out of the money) the independent one (median sd ratio). Chronos-2
   provides none of this information, so it has to be modelled.
2. **Assuming independence is badly overconfident.** Its 90% band covered the realised-curve
   value only 58–60% of the time. The empirical copula, estimated from past forecast errors
   only, covered 89%, close to the nominal 90%. Comonotone covered 92%.
3. **Volatility uncertainty matters far more for Delta than for the price at the money.**
   The empirical-copula band is ±5% of the price but about ±55% of Delta. The Delta of an
   Asian digital scales roughly like 1/σ.
4. **The first-order (bucket-vega) approximation is not reliable enough.**
   * It is within ±25% of the resampled width for 78–82% of dates at the money (independent
     or empirical copula).
   * Out of the money it is within ±25% for only 25–75% of dates, and for 45% under the
     empirical copula in high-volatility periods.
   * It overstates the comonotone out-of-the-money band by a median factor of 1.45.
   * Out-of-the-money digitals are strongly nonlinear in volatility. Full repricing (cheap
     here) is needed.
5. **Curve shape versus level.**
   * The Chronos-2 median curve versus a flat curve with the same total variance changes the
     at-the-money price by only 0.3% (median), but Delta by 3.7%. Out of the money, the price
     changes by 7.2% (90th percentile 18%) and Delta by 1.8%.
   * The *level* matters more. The Chronos-2 median volatility is about equal to realised
     (median ratio 0.99), while GARCH's flat forecast is 1.35× realised. GARCH forecasts mean
     variance, which right-skew inflates.
   * The median Delta difference between the two at the money is 33%. This also explains
     §3b: QLIKE penalises under-forecasts more, so the higher GARCH forecast scores better on
     it.

**Hypotheses (pre-registered).**

| | hypothesis | outcome |
|---|---|---|
| E1 | first-order band within ±25% of resampled for ≥ 80% of dates, every copula and strike | **fails** (passes only at the money with independent blocks, 82%) |
| E2 | comonotone / independent sd ≥ 2 | **passes** (3.5 and 2.0) |
| E3 | empirical-copula 90% band covers 80–97% | **passes** (89% for both strikes) |
| E4 | descriptive: median curve vs flat, and vs GARCH | reported above |

**Caveats.**
* The realised curve is measured from daily returns, which are noisy, so coverage mixes
  forecast error with measurement noise.
* The empirical copula's correlation uses overlapping past forecasts.
* One index, monthly dates (92, of which 78 have the empirical copula).
* Physical measure only: these are not prices.

## 4. Next steps (not started)

* **7D follow-up.** Chronos-2 on log-volatility, and fine-tuned on a training window, each
  pre-registered and tested on a later holdout.
* **7E follow-up.** A cheaper second-order propagation (using Delta's sensitivity to each
  block's volatility as well as the bucket vegas), and the same exercise for a hedging
  decision: how much does Delta-hedging P&L depend on the volatility curve used?
* **7F, hedging (optional, needs option data).** Whether forecast-based curves improve
  delta-hedging P&L against implied-volatility or constant-σ hedges. Option prices are
  risk-neutral; the forecasts are not. That gap must be handled explicitly (e.g. by
  scenario analysis rather than pricing).
* **Study follow-up.** A version of the training-free rotation that handles single-bucket
  spikes (e.g. linearising at more than one point), tested on fresh contracts.

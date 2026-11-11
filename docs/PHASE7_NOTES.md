# Phase 7 (in progress): volatility term structures, then Chronos-2

**Goal.** Test whether time-varying volatility forecasts, eventually from Chronos-2, can be
used in the Monte Carlo Greek machinery of Phases 1–6, and whether they improve anything that
can be measured. The numerical part comes first, so the work stands even if the forecasting
part adds nothing.

**Status.**
* 7A–7C, the numerical generalisation, is done and validated.
* The first study (7B) has been run.
* Chronos-2 runs on this machine (7D, feasibility only).
* No market data has been used yet.

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

## 4. Next steps (not started)

* **7D, real data.** Daily index prices (e.g. FRED's S&P 500 and VIX series). Target:
  realised volatility over 1–21 days. Walk-forward comparison of Chronos-2 zero-shot with
  EWMA, GARCH(1,1) and historical volatility, on volatility loss functions (QLIKE, MSE of
  variance) and interval coverage. Real-world (physical) measure only.
* **7E, forecast uncertainty → Greek uncertainty.** Chronos-2's quantiles give a per-horizon
  spread of σ(t). Through the bucket vegas this becomes an uncertainty band on Asian-digital
  prices and Greeks, to compare with resampling curves from the forecast quantiles.
* **7F, hedging (optional, needs option data).** Whether forecast-based curves improve
  delta-hedging P&L against implied-volatility or constant-σ hedges. Option prices are
  risk-neutral; the forecasts are not. That gap must be handled explicitly (e.g. by
  scenario analysis rather than pricing).
* **Study follow-up.** A version of the training-free rotation that handles single-bucket
  spikes (e.g. linearising at more than one point), tested on fresh contracts.

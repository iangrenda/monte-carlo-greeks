# Research findings (phase 2: Asian digital, adaptive estimator selection)

Every number in this file comes from an executed run, and the result files
are named next to it. "Research grid" means `results/asian/research/`
(36 scenarios, 2 budgets, 40 replications; 38 min wall-clock).
"Confirmation grid" means `results/asian/confirm/` (fresh scenarios and
seeds; see §6).

## 0. Validation log (commands, outcomes, runtimes)

| What | Command | Result | Wall time |
|---|---|---|---|
| Unit tests, phase 1 | `python -m pytest` (repo root) | 142 passed | 2.2 s |
| Unit tests, after phase 2 | `python -m pytest` | **175 passed** | 3.4 s |
| Paper benchmarks, two routes | inline script (closed form vs lognormal-density quadrature) | identical to 10 digits: 0.1682376255, −0.0013348588, −0.0003887569, −0.7775137088 | < 1 s |
| MC consistency, paper example | inline, 20 × 200k paths per Greek | z-scores mean −0.39…0.28, sd 0.96–1.16, max 2.80 | 0.9 s |
| Exact FD variance laws | inline quadrature | ratios 27 (h⁻³), 81 (h⁻⁴), 3.0 (h⁻¹), as predicted; 3 benign IntegrationWarnings | < 1 s |
| Phase-1 reproducibility | `python full_experiment.py --config research` re-run | accuracy tables bit-identical; timings ±30% | ~5 min |
| Asian m = 1 reduction | `tests/test_asian.py` | CPW estimators equal the closed-form digital Greeks; LR and Malliavin-avg weights equal the European weights path by path | — |
| Asian m = 2 exactness | `tests/test_asian.py` + an inline 1-D quadrature check | all estimators within 2.4 SE of quadrature references (~40 checks) | — |
| Rao–Blackwell identity | inline quadrature over Z₁ | CPW-first Δ, Γ = E[LR │ Z₂..Z_m] to ~1e-9; Vega differs (as derived) | — |
| Asian study, quick (development) | `python asian_study.py --config quick` | pipeline check; used to specify the "lite" variant | 23 s |
| Asian study, research | `python asian_study.py --config research` | §1–5 | 38 min |
| Asian study, confirmation | `python asian_study.py --config confirm` | §6 | see §6 |

## 1. Reference quality

`reference_quality.csv`, `figures/figH0_reference_check.png`

* **Cross-check z-scores.** The CPW-last MC cross-check (2^20 paths, a
  different estimator, sampler and seed) agrees with the CPW-first RQMC
  references in all 108 scenario-Greek pairs: mean z = −0.01, sd 1.00,
  max |z| = 2.74, none above 3.
* **Reference error vs method RMSE.** The reference SE is a median 1.1% of
  each strategy's RMSE (max 20%). It is subtracted in the corrected MSE.

## 2. H1: conditional smoothing (supported for CPW-first; refuted for CPW-last)

`H1_H4_mse_ratios.csv`, `figures/figH1_mse_ratio_by_m.png`

* **Per-path variance, LR vs CPW-first.** The ratio Var(LR)/Var(CPW-first)
  is at least **5.5 (Δ)** and **15.5 (Γ)** in every research setting
  (medians 30 and 54). This is guaranteed by the Rao–Blackwell identity
  (derivations II.5.1). For **Vega**, where no guarantee exists, the ratio
  is ≥ 5.1 (median 654).
* **Equal-budget MSE vs CPW-first/MC** (geometric mean over 12 scenarios
  per m, 20 ms budget):
  * LR/MC is 36–675× worse.
  * FD-raw/MC is 2.8–1 140× worse. FD-raw improves relative to CPW-first as
    m grows (Γ: 1 140× at m = 4, 2.8× at m = 52).
  * FD-smooth/MC is 1.4–5.4× worse.
  * Malliavin-avg (Δ only) is 4.9–58× worse.
* **CPW-last (last-step conditioning) is a poor smoother for averages.** Its
  MSE is 13–63× (Δ) and 118–142 000× (Γ) worse than CPW-first. It has
  higher per-path variance than plain LR in **46% (Δ) and 90% (Γ)** of
  settings. *Conditioning does not automatically help a differentiated
  estimator.* The variable that is integrated out matters: Z_m moves A by
  only 1/m, so Φ(d) stays nearly an indicator, and its pathwise derivative
  is spiky.
* **A genuinely non-LR Malliavin weight helps.** The anticipating weight
  (u ≡ 1) has lower variance than LR in 97% of Delta settings (median ratio
  0.29), and its advantage grows with m. It still loses to CPW-first.

## 3. H4: RQMC on top of smoothing (supported for small m, reversed for m = 52)

CPW-first/RQMC vs CPW-first/MC, MSE ratio at an equal calibrated budget:

| m | 2 ms (Δ / Γ / 𝒱) | 20 ms |
|---|---|---|
| 4 | 0.010 / 0.038 / 0.018 | **0.0012 / 0.0022 / 0.0026** |
| 12 | 0.84 / 0.81 / 0.40 | 0.28 / 0.54 / 0.28 |
| 52 | **1.72 / 2.23 / 1.38** (worse) | **1.72 / 2.02 / 1.64** (worse) |

* With sequential path construction, the RQMC gain collapses as m grows. At
  m = 52 it *loses*, because a scrambled-Sobol path costs ~2× a
  pseudorandom one here.
* This is a statement about our construction. A Brownian bridge or PCA
  (Albieri et al. 2025) is expected to restore gains, but that is untested.
* **Cost-model caveat.** Measured wall time exceeded the calibrated cost by
  a median ×1.08 for MC estimates but ×1.75 for RQMC estimates: per-call
  scramble overhead is not captured by per-path calibration. In wall-clock
  terms, the RQMC advantages above are therefore **overstated by roughly
  1.6×**. The m = 4 conclusion survives; the m = 12 gains would shrink to
  about break-even for Δ and Γ.

## 4. H2: the best estimator changes with the regime (supported, but the change is essentially one switch)

`H2_oracle_by_scenario.csv`, `figures/figH2_oracle_choices.png`

Oracle (best fixed) choice, counted over 12 scenarios per cell:

* **m = 4.** CPW-first/RQMC in 71 of 72 cells (both budgets, all Greeks).
* **m = 12.** Mostly CPW-first/RQMC (55/72). The rest are FD-smooth/RQMC
  (11) and CPW-first/MC (6), which are within noise of each other.
* **m = 52.** Mostly CPW-first/MC (47/72). The others are FD-smooth/MC (9),
  FD-raw (8), CPW-first/RQMC (3), CPW-last MC or RQMC (3) and Malliavin-avg (2).
  Many are near-ties: the oracle is a minimum over noisy MSEs.
* **Greek and budget** matter much less than m.

So the regime dependence is real, but it reduces largely to **"use RQMC
when m is small, MC when m is large, always with CPW-first."** Moneyness,
σ and T did not change the winning *family*.

## 5. H3 and H5: pilot-based adaptive selection (H3 refuted for the full selector; H5 supported with a specific mechanism)

`H3_regret_out_of_sample.csv`, `H3_adaptive_vs_best_fixed.csv`,
`H5_selection_behaviour.csv`, `figures/figH3_regret_*.png`

Regret is RMSE / oracle RMSE on the test scenarios, as a geometric mean
with a 95% bootstrap CI over scenarios.

**Random-half split, test = 18 scenarios:**

| Greek, budget | Best fixed on train (CPW-first/RQMC) | Adaptive (10 cands, 10% pilot) | Adaptive-lite (3 cands, 5% pilot) |
|---|---|---|---|
| Δ, 2 ms | 1.18 [1.10, 1.27] | 7.72 [5.9, 9.9] | 1.30 [1.12, 1.53] |
| Γ, 2 ms | 1.29 [1.12, 1.49] | 37.0 [12, 106] | 1.48 [1.31, 1.70] |
| 𝒱, 2 ms | 1.13 [1.04, 1.24] | 7.69 [6.2, 9.8] | 2.02 [1.47, 2.90] |
| Δ, 20 ms | 1.15 [1.06, 1.26] | 3.73 [2.2, 6.4] | 1.23 [1.07, 1.46] |
| Γ, 20 ms | 1.17 [1.07, 1.29] | 20.8 [5.9, 74] | **1.07 [0.99, 1.16]** |
| 𝒱, 20 ms | 1.16 [1.06, 1.27] | 4.36 [2.8, 6.6] | 1.30 [1.08, 1.65] |

**Extrapolation split (train m ∈ {4, 12}, test m = 52; 12 test scenarios):**

| Greek, budget | Best fixed on train (CPW-first/RQMC) | Adaptive | Adaptive-lite | Heuristic "always CPW-first/MC" |
|---|---|---|---|---|
| Δ, 2 ms | 1.39 | 7.96 | **1.18** | 1.06 |
| Γ, 2 ms | 1.66 | 284 | **1.45** | 1.11 |
| 𝒱, 2 ms | 1.33 | 6.76 | **1.21** | 1.13 |
| Δ, 20 ms | 1.35 | 9.61 | **1.14** | 1.03 |
| Γ, 20 ms | 1.42 | 428 | **1.18** | 1.00 |
| 𝒱, 20 ms | 1.30 | 8.69 | **1.07** | 1.02 |

* **The full selector is far worse than every sensible fixed rule** (H3
  refuted). It loses to the best fixed estimator in 15–18 of 18 test
  scenarios (by > 2 SE in 8–16 of them).
* **Why (H5, diagnosed).** It picks a rare-event estimator whose pilot
  happened to contain **no nonzero per-path terms**, so its sample variance
  was 0 and its predicted MSE was 0. CPW-last (MC or RQMC) was chosen in
  22–40% of replications and accounts for **90–100% of the selector's total
  squared error** for each Greek. For those choices the realized/predicted MSE ratio is
  infinite. It is matched to the oracle only 24–56% of the time. A
  10-candidate pilot at 2 ms leaves ~15–150 paths per candidate.
* **The restricted "lite" selector** (3 smooth candidates, 5% pilot) is
  roughly as good as the best fixed rule on the random split: better for Γ
  at 20 ms (1.07 vs 1.17), worse for 𝒱 at 2 ms (2.02 vs 1.13), otherwise
  CIs overlap.
  * Under **distribution shift** (extrapolating to m = 52) it beats the rule
    learned on m ≤ 12 in all 6 cells. In head-to-head counts it is better
    by > 2 SE in 1–4 of 12 scenarios per cell and never worse by > 2 SE.
  * The reason: CPW-first/RQMC, learned on small m, is the wrong choice at
    m = 52, and the pilot notices.
  * But the simple heuristic "always CPW-first/MC" does even better there
    (1.00–1.13).
  * Caveat: the lite candidate set was chosen after the quick development
    run (disjoint scenarios), so it reflects a design decision informed by
    data, not a blind one.
* **Bottom line for H3.** Adaptivity pays only as a guard against a fixed
  rule that was learned on a different regime, and only with a small,
  well-behaved candidate set. In this problem the candidate families are not
  "similarly efficient": one family dominates. That is the situation where
  selection can only lose (pilot cost plus selection noise).

## 6. Confirmation run: guarded selector on fresh scenarios

**Design.** The "guarded" selector (all 10 candidates, 10% pilot) excludes
any candidate whose pilot had fewer than 10 nonzero per-path terms. It was
specified *after* the §5 failure analysis, so it is evaluated only on a
**fresh grid**:

* m ∈ {4, 12, 52}, K/S₀ ∈ {0.95, 1.05}, σ ∈ {0.15, 0.25}, T = 0.5 (none of
  these σ, K or T values occur in the quick or research grids);
* budget 5 ms, 30 replications, seed 20261105;
* 5 min wall-clock (`results/asian/confirm/`).

**References.**

* The CPW-last cross-check gives z mean −0.52, sd 0.82, with 1 of 36 above
  |z| = 3.
* The shift is entirely in Δ (mean z −0.97 over 12 scenarios).
* A third, light-tailed check (CPW-first MC, 2²¹ fresh paths) on six
  scenarios, including the worst one, agreed with the references
  (z = −1.56 … +0.97, mixed signs). We conclude the references are sound,
  and that the CPW-last Δ check is right-skewed (it usually falls below
  the mean, with an underestimated SE). That is the same heavy-tail
  property that defeats the pilot.

**Results** (geometric-mean regret, 95% bootstrap CI over test scenarios;
test sets are small, with 6 and 4 scenarios, so the CIs are wide):

| Split, Greek | Best fixed on train (CPW-first/RQMC) | Adaptive (no guard) | **Adaptive-guarded** | Adaptive-lite |
|---|---|---|---|---|
| random, Δ | 1.09 [1.01, 1.19] | 3.84 | 1.56 [1.17, 2.2] | 1.16 |
| random, Γ | 1.26 [1.07, 1.50] | 30.2 | 3.27 [1.33, 12.3] | 1.12 |
| random, 𝒱 | 1.18 [1.03, 1.38] | 3.72 | 1.35 [1.23, 1.48] | 1.31 |
| m = 52 held out, Δ | 1.07 | 11.4 | 1.36 | 1.17 |
| m = 52 held out, Γ | 1.64 | 524 | 3.61 | **1.16** |
| m = 52 held out, 𝒱 | 1.42 | 9.37 | **1.29** | **1.28** |

* **The guard removes the gross failure.** Regret falls from 3.7–524 to
  1.3–3.6. The guarded rule chose a rare-event estimator (CPW-last, FD-raw)
  in only 10 of 1 080 selections, against 22–40% unguarded.
* **It still does not beat the best fixed rule on the random split** (all
  three Greeks).
  * For Γ, a residual heavy tail remains (worst scenario regret 72.8): a few
    remaining FD-raw/CPW-last picks passed the ≥ 10 rule with
    underestimated variance.
  * Under the m = 52 shift it wins for 𝒱 and loses for Δ and Γ.
* **Adaptive-lite again tracks the best fixed rule** (1.12–1.31 vs
  1.09–1.26 on the random split), and beats it under the m = 52 shift for
  Γ and 𝒱. But "always CPW-first/MC" is better still at m = 52 (1.00–1.20).
* **Verdict on H3.** Even with a variance guard, pilot-based selection over
  a heterogeneous candidate set did **not** outperform a well-chosen fixed
  estimator in our experiments. A small candidate set of near-equivalent,
  well-behaved estimators makes selection *safe* (regret ≈ best fixed) and
  sometimes helpful under regime shift, but not reliably better. Plug-in
  variance from a small pilot is not a trustworthy basis for choosing
  heavy-tailed estimators: a zero-variance pilot is the extreme case, and
  underestimated tails are the milder one.

**What would be needed to change this verdict** (not done):

* robust pilot statistics (e.g. heavy-tail-aware upper confidence bounds on
  variance, or median-of-means);
* sequential allocation (successive halving) instead of one pilot;
* pre-screening by a theory-based variance proxy (e.g. the Rao–Blackwell
  ordering of II.5.1, which already rules out LR in favour of CPW-first for
  Δ and Γ).

## 7. Estimator-level limitations seen in the data

* **CI coverage.** CPW-first/MC has a median coverage of 0.95, but 3% of
  rows fall below 0.85.
  * All of the lowest are one rare-event regime: K/S₀ = 0.9, σ = 0.1,
    T = 0.25, m = 52, where the digital almost surely pays.
  * The estimates are unbiased there (|bias| < 1 SE), but with ≤ 1 500 paths
    the per-path distribution is so skewed that the normal-theory SE is
    underestimated.
  * RQMC intervals from 8 scramblings use z = 1.96 rather than t₇ = 2.36,
    which adds under-coverage (median 0.925).
* **Calibrated cost vs wall clock.** See §3. The calibration should be done
  per call size, including the scramble overhead.
* **Single model.** All results are for GBM, where CPW-first conditioning is
  exact and closed form. Under stochastic volatility the conditional
  probability is no longer explicit.

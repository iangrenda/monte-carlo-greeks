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
| Phase-2 revalidation | `python revalidate_phase2.py` | §8 | ~25 min |
| Phase 3, first pass | `python asian_qmc_study.py --config research` | superseded for RQMC/adaptive by the re-run (§9.2) | 142 min |
| Phase 3, re-run with pooled cost model | `python asian_qmc_study.py --config research --rerun-timed` | §9 | 17 min |
| Unit tests, after phase 3 | `python -m pytest` | **221 passed** | 2.9 s |

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

> **Erratum (phase 3, §8).**
> * The regrets in this section charge the selectors only their *nominal*
>   pilot budget. Measured pilot wall time was 6–400× larger.
> * With measured time charged, every adaptive variant is worse than the
>   best fixed estimator. The statements below that adaptive-lite is
>   "roughly as good as the best fixed rule" and "beats the rule learned on
>   m ≤ 12 under the m = 52 shift" are **retracted**.
> * The negative conclusion for the full selector is strengthened.

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

> **Erratum (phase 3, §8).** The same nominal-pilot accounting applies here.
> Read the adaptive regrets as lower bounds.

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

---

# Phase 3

## 8. Budget-accounting audit and revalidation of phase 2

Sources: `scripts/revalidate_phase2.py` (re-run of the three adaptive
variants with corrected accounting, 20 replications, new seed family; about
25 min), output in `results/asian/research_revalidation/`. A read-only
pilot audit was also run on six scenarios.

**Defect.**

* The phase-2 selector gave each candidate at least 16 pilot paths and then
  charged the final run the *nominal* pilot fraction.
* The calibration is per path, so it misses per-call overhead, mainly
  constructing ≥ 8 Owen-scrambled Sobol generators per RQMC candidate.

Measured (`pilot_accounting.csv`, medians):

| | 2 ms budget: m = 4 / 12 / 52 | 20 ms budget: m = 4 / 12 / 52 |
|---|---|---|
| calibrated pilot cost / nominal (full selector) | 0.98 / 0.97 / **2.47** | 0.99 / 0.98 / 0.97 |
| measured pilot wall time / nominal | **78 / 135 / 301** | 7.8 / 14.5 / 40 |
| measured total wall time / B | **9.5 / 15.5 / 33** | 1.4 / 2.4 / 5.0 |

**Fixed estimators also missed their nominal budgets** (medians of measured
time / B, from the stored phase-2 runs):

* **MC:** 0.5–2.0.
* **RQMC:** 0.85–8.3. The worst case, 8.3, is m = 52 at 2 ms, where
  per-scramble overhead dominates.

**Fix.** `adaptive_select(..., accounting="charged")` is now the default.
The pilot's measured wall time is recorded, and "nominal" reproduces the
stored phase-2 numbers bit for bit (verified at 1e-16).

**Revalidation (time-normalised).** wnMSE = MSE × measured mean time,
pilot included. The regret is √(wnMSE / best fixed wnMSE), as a geometric
mean over the 36 research scenarios (`time_normalised_regret.csv`):

| | 2 ms: Δ / Γ / 𝒱 | 20 ms: Δ / Γ / 𝒱 |
|---|---|---|
| CPW-first/RQMC (phase-2 "best fixed") | 2.30 / 2.48 / 2.09 | **1.17 / 1.18 / 1.14** |
| CPW-first/MC | **2.16 / 1.64 / 1.85** | 4.69 / 3.83 / 4.02 |
| adaptive (10 cands) | 24.8 / 78.1 / 21.7 | 5.0 / 17.2 / 4.7 |
| adaptive-guarded | 10.7 / 12.4 / 9.4 | 3.8 / 8.3 / 3.7 |
| adaptive-lite | 4.1 / 4.6 / 5.8 | 1.84 / 1.69 / 1.61 |

At m = 52 only, adaptive-lite at 20 ms scores 2.1 / 2.2 / 1.8 against
1.6 / 1.6 / 1.5 for CPW-first/RQMC and 1.0 / 1.0 / 1.0 for CPW-first/MC.

**What changes.**

1. Every adaptive variant is worse than the best fixed estimator at both
   budgets once measured time is charged. The phase-2 statements favouring
   adaptive-lite are retracted (errata in §5 and §6).
2. At the 2 ms budget, RQMC actually used 4–5× its budget. Time-normalised,
   it no longer beats plain MC there. The phase-2 RQMC advantage at small
   budgets was partly an accounting artefact.
3. **Unchanged.** CPW-first dominates LR, CPW-last and FD (all MC-based
   comparisons are at most mildly affected). CPW-first/RQMC is the best
   fixed choice at 20 ms. MC is best at m = 52.

**Confidence intervals** (`ci_coverage_revalidated.csv`). Phase 2 used
z = 1.96 with R ≈ 10 scramblings; t_{R−1} is correct. Mean coverage moves
as follows:

* CPW-first/RQMC: 0.904 → **0.933**.
* LR/RQMC: 0.914 → 0.944.
* FD-smooth/RQMC: 0.68 → 0.71. It remains poor because the interval ignores
  the O(h²) bias.

MC coverage (z) is 0.94 for CPW-first and LR, and 0.69 for FD-raw (bias
again). Phase 3 uses t-intervals throughout.

## 9. Phase 3: dimension reduction for RQMC under real wall-clock budgets

**Sources.** `scripts/asian_qmc_study.py --config research` (first pass,
142 min), then `--rerun-timed`, which re-ran the RQMC fixed runs and the
adaptive runs with a pooled cost model (17 min; see "Calibration defect"
below). Output is in `results/asian_qmc/research/`. First-pass files are kept
as `*_v1`.

**Setup.**

* 36 out-of-sample scenarios; m ∈ {4, 12, 52}; real wall-clock budgets of
  5 ms and 50 ms; 30 replications.
* Estimator: CPW-first.
* Samplers: MC, and RQMC with sequential, Brownian-bridge (BB) or PCA
  construction of the residual path. The level variable Z₁ is integrated
  out exactly in every case (Part III).

**What is established vs new.**

* **Established.** BB and PCA constructions (Moskowitz & Caflisch 1996;
  Acworth, Broadie & Glasserman 1998). Their use with conditional pathwise
  Greeks (Albieri et al. 2025, with a Brownian bridge). The effective-dimension
  explanation.
* **New here.** A controlled comparison:
  * under *measured wall-clock* budgets that include scrambling overhead;
  * with the level-direction conditioning kept exact under every
    construction;
  * of fixed, regime-aware and pilot-based rules, evaluated out of sample.
* **Not claimed.** Novelty of any method.

### 9.1 Validation

* **References.** PCA-RQMC with 32 × 2¹⁶ points. Over 144 independent
  plain-MC checks, z has mean −0.03, sd 1.01, and max 3.36. One |z| > 3 in
  144 is consistent with chance; the expected count is about 0.4.
* **Tests (all pass).**
  * Constructions: LLᵀ = C to ≤ 1e-13 for m = 2…52.
  * Path law: W = √Δ Z₁ + B has covariance min(t_i, t_j), exactly and by
    sampling.
  * Conditioning identity: the reconstructed residual equals B, and Z₁ has
    no effect.
  * Unbiasedness: every sampler/construction agrees with quadrature at
    m = 2 and with large MC at m = 12.
  * Budgets: timed runs reach the budget, use t-intervals, and are
    reproducible at zero budget.
* **Budget adherence.** Measured time / B has median 1.01–1.09, except at
  5 ms and m = 52. There even the minimum RQMC work (4 scramblings × 16
  points) overruns by 18%; it is charged in full.
* **Coverage** of the 95% intervals (t_{R−1} for RQMC): median 0.93–0.97 for
  every sampler; minimum 0.80 over 30 replications (binomial SE ≈ 0.04).

### 9.2 Calibration defect found and fixed during phase 3

* **What went wrong.** The first pass fitted the cost model t(n) = a + b·n
  separately per construction from two timings. Timing noise gave PCA a
  per-scrambling overhead 2–3× that of the others, although all three run
  identical code. At 5 ms, PCA therefore got **16** points per scrambling
  where sequential and BB got **2 048** (m = 4).
  * That made PCA look 4–16× *worse* than MC at m ≤ 12. It was an artefact.
* **Fix.** One pooled least-squares cost model per m, from four sizes ×
  9 repeats, shared by all constructions. Raw medians per construction agree
  within noise (`calibration.csv`). Every construction now uses the same
  points per scrambling. A regression test enforces this.
* **Status.** All numbers below are from the corrected re-run.

### 9.3 Mechanism: convergence at fixed N

`convergence_fixed_n.csv`, `convergence_slopes.csv`, `figures/figQ1_*`.
These are log-log slopes of RMSE against points per scrambling (2⁶…2¹⁴,
32 scramblings each). Plain MC is −0.5.

| | m = 4 (seq / BB / PCA) | m = 12 | m = 52 |
|---|---|---|---|
| Δ | −1.09 / −1.10 / −1.19 | −0.63 / −0.83 / −1.10 | **−0.48 / −0.65 / −1.01** |
| Γ | −1.05 / −1.03 / −1.14 | −0.58 / −0.85 / −1.09 | −0.49 / −0.64 / −0.99 |
| 𝒱 | −1.07 / −1.12 / −1.08 | −0.62 / −0.84 / −1.11 | −0.48 / −0.64 / −1.04 |

With sequential construction the RQMC rate decays to the MC rate as m
grows. **PCA keeps an approximately N⁻¹ rate at m = 52.** BB is in between.

This matches the structure of the constructions. The share of the variance
of the path average carried by the first Sobol coordinate is:

| | seq | BB | PCA |
|---|---|---|---|
| m = 4 | 0.64 | 0.86 | 0.99 |
| m = 12 | 0.24 | 0.78 | 0.99 |
| m = 52 | 0.06 | 0.76 | 0.99 |

### 9.4 Q1: does dimension reduction recover the RQMC advantage at large m?

`Q1_wnmse_ratio_vs_mc.csv`, `figures/figQ2_*`. Time-normalised MSE relative
to MC, as a geometric mean over 12 scenarios per m with a 95% bootstrap CI.
The count beside each value is the number of scenarios where RQMC beats MC.

| Budget | Construction | m = 4 (Δ / Γ / 𝒱) | m = 12 | m = 52 |
|---|---|---|---|---|
| 50 ms | PCA | 2e-6 / 3e-6 / 6e-6 (12/12 each) | 4e-5 / 1e-4 / 5e-5 (12/12) | **0.014 / 0.043 / 0.012 (12/12)** |
| 50 ms | BB | 3e-5 / 6e-5 / 4e-5 | 0.009 / 0.030 / 0.008 | 0.88 / 1.98 / 0.89 (6, 1, 6 of 12) |
| 50 ms | seq | 9e-5 / 2e-4 / 1e-4 | 0.26 / 0.74 / 0.29 | 3.9 / 3.9 / 3.7 (0/12) |
| 5 ms | PCA | 1e-4 / 2e-4 / 2e-4 (12/12) | 0.012 / 0.038 / 0.018 (12/12) | **17 / 43 / 14 (0/12)** |
| 5 ms | BB | 0.001 / 0.002 / 0.001 | 0.42 / 1.43 / 0.56 | 77 / 100 / 51 |
| 5 ms | seq | 0.003 / 0.007 / 0.003 | 3.3 / 8.0 / 4.8 | 122 / 126 / 80 |

**Answer: yes, conditionally.**

* With PCA, the RQMC advantage at m = 52 is recovered at a 50 ms budget:
  time-normalised MSE is 23–86× lower than MC (RMSE 5–9× lower) in all
  36 scenario-Greeks.
* The Brownian bridge recovers only parity at m = 52.
* At a 5 ms budget and m = 52, every construction loses to MC by an order
  of magnitude. The ~0.8 ms per-scrambling overhead leaves room only for the
  minimum of 4 scramblings × 16 points.
* So the obstacle at large m is no longer effective dimension once PCA is
  used. It is the **fixed per-randomisation cost** of RQMC relative to the
  budget.

### 9.5 Q2: fixed vs regime-aware vs pilot-based selection

`Q2_rules_out_of_sample.csv`, `adaptive_behaviour.csv`, `figures/figQ3_*`.

**Oracle choices.**

* 5 ms: PCA for every m ≤ 12 scenario (72/72); MC for every m = 52 scenario
  (36/36).
* 50 ms: PCA in 106 of 108 scenario-Greeks; BB twice, both at m = 4,
  within noise.

Regret (time-normalised RMSE / oracle; geometric mean; 95% CI over test
scenarios):

| Split, budget | Always PCA | Best fixed on train | Regime rule on train | Adaptive (MC-rate) | Adaptive (N⁻² RQMC) |
|---|---|---|---|---|---|
| random, 5 ms (Δ / Γ / 𝒱) | 1.73 / 2.07 / 1.67 | same (PCA) | **1.00 / 1.00 / 1.00** (m ≤ 12: PCA, else MC) | 31 / 24 / 26 | 24 / 23 / 23 |
| random, 50 ms | **1.00** | **1.00** (PCA) | **1.00** | 14 / 10 / 14 | 7.2 / 5.8 / 6.1 |
| held-out m = 52, 5 ms | 4.2 / 6.5 / 3.8 | 4.2 / 6.5 / 3.8 | 4.2 / 6.5 / 3.8 (cannot learn a threshold above 12) | 5.3 / 5.0 / 4.9 | 4.9 / 5.0 / 4.5 |
| held-out m = 52, 50 ms | **1.00** | **1.00** | **1.00** | 12 / 5.7 / 12 | 9.6 / 6.0 / 10.8 |

* **Simple rules beat pilot-based selection decisively.** This holds in 11
  of 12 split×budget×Greek cells: by factors of 5–30 in the 9 cells outside
  held-out m = 52 at 5 ms, and narrowly for Δ and 𝒱 inside it.
* **Why the selectors fail.**
  * At 5 ms the pilot alone used a median of 1.5× the whole budget. The
    four candidates' minimum RQMC pilots (4 scramblings each) don't fit,
    and the budget was exhausted in 77–87% of runs.
  * At 50 ms the pilot used 17% of B. The MC-rate prediction still chose
    plain MC 61–65% of the time: a tiny RQMC pilot (16–64 points per
    scrambling) shows almost none of the rate advantage. The optimistic
    N⁻² variant chose PCA 61–67% of the time but still had regret 6–7.
  * Pilot estimates of RQMC variance at small n do not extrapolate to large
    n.
* **The one cell where nothing works well** is held-out m = 52 at 5 ms.
  Every learned rule extrapolates "PCA", which is wrong there. The selectors
  are comparable: better for Γ (5.0 vs 6.5), slightly worse for Δ and 𝒱.
* **Post-hoc cost-model rule** (formulated *after* seeing these results, so
  it carries no confirmatory weight). "Use PCA-RQMC iff 10 scramblings of
  ≥ 64 points fit in B under the calibrated t(n) = a + b·n, else MC." It uses
  only the machine calibration, with no pilot and no training data, and
  matches the oracle in all cells, including the m = 52 extrapolation. It is
  a hypothesis for a future confirmatory grid, not a finding.

### 9.6 Answers and limitations

* **Does dimension reduction recover the RQMC advantage at large m?** Yes,
  with PCA, once the budget is large compared with the per-randomisation
  overhead. The Brownian bridge does so only partly at m = 52. Sequential
  construction does not.
* **Do simple rules still beat pilot-based selection?** Yes. "Always PCA"
  at the larger budget, and a regime rule (or the post-hoc cost-model rule)
  at the smaller one, beat both pilot-based selectors in 11 of 12 cells. The
  exception is Γ in held-out m = 52 at 5 ms, where every strategy is poor.
* **Limitations.**
  * Only one estimator (CPW-first), one model (GBM) and one payoff.
  * Two budgets, one machine (Windows, single-threaded NumPy/SciPy).
  * The SciPy Sobol scrambler overhead is implementation-specific: a
    faster digital-shift or reusable scrambler would move the 5 ms frontier.
  * The pilot design is one of many. Sequential allocation and successive
    halving were not tested.
  * The oracle is a minimum over noisy wnMSEs (30 replications), so regret
    near 1 is within noise.
  * wnMSE assumes MSE ∝ 1/time when normalising. Budgets were met to
    within 1–9%, so the normalisation changes little.

---

## 10. Phase 4 (see separate documents)

* `PHASE4_VALIDATION.md`: revalidation of phases 1–3.
* `PHASE4_COST_RULE_VALIDATION.md`: pre-registered confirmation of the §9.5
  cost rule (PASS, with qualifications).
* `PHASE4_REPORT.md`: Greek-aware gradient PCA, joint multi-Greek transform,
  and cost accounting. Also a consolidated report of phases 1–4.

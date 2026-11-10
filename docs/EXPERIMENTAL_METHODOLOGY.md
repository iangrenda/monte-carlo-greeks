# Experimental methodology (phase 2: Asian digital)

> **Erratum (Phase 6, 2026-11-10).** Where this document says "Owen scrambling" or "Owen-scrambled" for the RQMC randomisation, read **SciPy's LMS+shift**: `scipy.stats.qmc.Sobol(scramble=True)` applies Matoušek's random linear matrix scrambling followed by a random digital shift, not Owen's nested uniform scrambling. Every estimate stays unbiased and every number reported here is unchanged; only the label was wrong. Citations of Owen's or Liu & Owen's papers are unaffected. See `docs/PHASE6_REPORT.md`.

Phase 1's methodology for the European study is in `experimental_design.md`
and is unchanged. Here "MC" means plain Monte Carlo with pseudorandom
numbers, and "RQMC" means randomised quasi-Monte Carlo (scrambled Sobol
points).

## Contract, model, grid

* **Contract.** `e^{−rT}·1{(1/m)Σ S_{t_i} ≥ K}`, with equally spaced
  t_i = iT/m. GBM is simulated exactly on the grid.
* **Grid.** 36 scenarios: m ∈ {4, 12, 52}, K/S₀ ∈ {0.9, 1.0, 1.1},
  σ ∈ {0.1, 0.3}, T ∈ {0.25, 1.0}, with S₀ = 100 and r = 0.05
  (`results/asian/research/scenarios.csv`).
* **Greeks.** Δ, Γ and 𝒱 (per unit σ).

## Candidates (10)

| Family | MC | RQMC |
|---|---|---|
| LR (= FLLLT weight, a = 1/t₁) | ✔ | ✔ |
| Malliavin-avg (anticipating, Δ only) | ✔ | — |
| CPW-last | ✔ | ✔ |
| CPW-first | ✔ | ✔ |
| FD-raw (CRN, central) | ✔ | — |
| FD-smooth (CRN, central, on the CPW-first price) | ✔ | ✔ |

**FD bumps** are fixed rules, set before the study:

* FD-raw: h_Δ = 0.25·S₀σ√T, h_Γ = 0.5·S₀σ√T, h_𝒱 = 0.1σ.
* FD-smooth: h_Δ = 0.05·S₀σ√T, h_Γ = 0.1·S₀σ√T, h_𝒱 = 0.05σ.

At σ = 0.2, T = 1, the FD-raw coefficients equal the phase-1 European
optima.

**RQMC.** Owen-scrambled Sobol points with sequential construction. Each
estimate uses points-per-scramble = 2^k and at least 8 scramblings, sized
to fill the budget; the SE comes from the spread across scramblings.

## Budget and cost

* **Calibration.** Per-path seconds for each (method, sampler, Greek, m)
  are calibrated once at the start of the run: median of 5 timings of
  2^14 paths, saved as `cost_calibration.csv`.
* **Budget.** A budget B (2 ms or 20 ms) buys N = B / c paths. Every
  strategy, adaptive ones included, spends the same B.
* **Wall-clock.** Actual run times are recorded alongside, to check the
  calibration.
* Single thread; hardware as in `meta.json`.

## References ("truth")

* **Primary.** CPW-first with RQMC: 32 independent scramblings × 2^15
  points (≈ 1.05 M paths). Every Greek is computed from the same points,
  and its SE is the spread across scramblings.
* **Independent cross-check.** CPW-last with plain MC, 2^20 paths: a
  different conditioning, a different sampler and a different seed. We
  report z = (check − ref)/√(SE_check² + SE_ref²).
* **Error accounting.** The MSE against the reference is corrected by
  subtracting SE_ref², because the reference error is independent of the
  runs. The ratio SE_ref/RMSE is reported for every row.
* **Exactness checks.** For m = 2, 1-D quadrature gives near-exact values;
  these test all estimators in `tests/test_asian.py`. For m = 1 the closed
  forms apply.

## Replications and seeds

* **Replications.** 40 independent replications per (scenario, budget,
  Greek, strategy).
* **Seeds.** The master seed is 20261104. Streams are derived as
  `SeedSequence([seed, stage, budget_index, scenario, candidate, greek, rep])`.
  Pilot and evaluation runs of the adaptive selector use **different**
  streams, and fixed-strategy runs use a third family.
* **Development data.** The quick configuration (`results/asian/quick`) has
  σ = 0.2 only, so it shares no scenario with the research grid. It was used
  to debug the pipeline and to specify the "adaptive-lite" variant before the
  research run.

## Adaptive selector (deployable)

1. **Pilot.** Spend a fraction f of B (10% default, 5% for the "lite"
   variant), split equally across the J applicable candidates.
2. **Predict** each candidate's MSE at the remaining budget:
   * unbiased MC: s²/N;
   * RQMC: SE_pilot²·n_pilot/N. This deliberately assumes the MC rate, so
     it is conservative;
   * FD: bias_hat² + s²/N, with bias_hat = (θ̂(2h) − θ̂(h))/3 from CRN pilot
     runs.
3. **Choose** the minimum, and run it with the remaining budget on **fresh**
   random numbers.

Variants:

* "adaptive (pilot)": all 10 candidates, f = 10%.
* "adaptive-lite": {CPW-first MC, CPW-first RQMC, FD-smooth RQMC},
  f = 5%.

## Baselines

* Each fixed candidate.
* Heuristics: always CPW-first/RQMC; always CPW-first/MC; always LR/MC; and
  always FD-raw/MC (bump & revalue).
* **Best fixed on train.** The candidate with the smallest geometric-mean
  regret on the training scenarios, applied unchanged to the test
  scenarios.
* **Oracle** (evaluation only). For each scenario, the fixed candidate with
  the lowest measured corrected MSE. It uses the evaluation data, so it is
  not deployable and is mildly optimistic, because the minimum of noisy
  MSEs is biased down.

## Splits and metrics

* **Random half.** Train and test are a seeded random permutation of the
  36 scenarios.
* **Extrapolation.** Train on m ∈ {4, 12} and test on m = 52.
* **Regret.** RMSE / oracle RMSE per test scenario, aggregated by geometric
  mean, with a 95% percentile bootstrap over test scenarios (2 000
  resamples).
* **Head-to-head.** For the adaptive variant vs best-fixed-on-train, we
  count the test scenarios where the adaptive MSE is lower or higher by
  more than 2 combined SEs. Runs are independent, not paired.
* **Per-replication metrics.** Bias ± SE, variance, MSE ± SE, coverage of
  ±1.96·SE intervals, mean paths and mean wall time.
* **Selection behaviour.** The share of selections that match the oracle,
  and the choice counts.

## Known limitations of the design

* **Oracle bias.** The oracle is a minimum over noisy estimates (40
  replications, relative MSE SE ≈ 20–30%), so a regret near 1 is within noise.
* **Within-noise ties.** "Best fixed on train" is chosen on 18 scenarios.
  With few clear winners, ties between near-equivalent candidates are
  decided by noise.
* **Conservative RQMC prediction.** The RQMC MSE prediction assumes the MC
  rate, so it tends to undervalue RQMC at large budgets.
* **Sequential construction only.** No Brownian bridge or PCA, so RQMC
  results understate what the literature achieves at large m.
* **Timing.** Timings are machine-specific. Costs are calibrated once, and
  the calibration itself has ±10–30% noise.
* **Superseded.** The phase-2 per-path cost calibration omits per-call
  overhead. See the phase-3 audit (`RESEARCH_FINDINGS.md` §8). Phase 3 uses
  measured wall-clock budgets instead.

---

# Phase 3: path constructions and wall-clock budgets

Script: `scripts/asian_qmc_study.py`. Output: `results/asian_qmc/<config>/`.
Mathematics: `mathematical_derivations.md` Part III.

* **Estimator.** CPW-first only, with the level variable Z₁ integrated out
  exactly, for Δ, Γ and 𝒱. Phases 1–2 already established it as the
  dominant estimator family.
* **Samplers.**
  * MC: sequential construction, which is equivalent in law under MC.
  * RQMC-seq, RQMC-bb, RQMC-pca: Owen-scrambled Sobol points in dimension
    m − 1, used for the residual path.
* **Out-of-sample grid (36 scenarios).** m ∈ {4, 12, 52},
  K/S₀ ∈ {0.92, 1.00, 1.08}, σ ∈ {0.12, 0.28}, T ∈ {0.4, 1.5}, r = 0.03.
  None of these K, σ, T or r values occur in any earlier grid.
* **Budgets.** Real wall-clock time, B ∈ {5 ms, 50 ms} per estimate.
  * Estimation stops when the clock passes B (block / scrambling
    granularity), with a minimum of 2 MC blocks or 4 scramblings.
  * The measured time of every estimate is recorded.
  * RQMC points per scrambling: the largest power of two such that 10
    scramblings fit, under the measured cost t(n) = a + b·n, fitted per
    (construction, m) at the start of the run.
* **Replications.** 30 per (scenario, budget, Greek, strategy).
* **Seeds.** Master seed 20261110. Separate seed families for references
  (1), fixed runs (2), adaptive pilots (3), adaptive final runs (4), the
  convergence study (9–11) and the train/test permutation (seed + 1).
* **References.** PCA-RQMC with 32 × 2¹⁶ points, all Greeks from the same
  points. Independent check: CPW-first plain MC with 2²⁰ paths per Greek,
  reported as z-scores.
* **Mechanism study.** At fixed N (not time), m ∈ {4, 12, 52}, K = S₀,
  σ = 0.2, T = 1: 32 scramblings at N = 2⁶…2¹⁴, plus the exact MC RMSE at the
  same N. Reports log-log slopes.
* **Metrics.**
  * Bias, MSE (corrected for the reference variance), coverage of the 95%
    interval (t_{R−1} for RQMC), and measured wall time / B.
  * Time-normalised MSE, wnMSE = MSE × mean measured time.
  * Regret = √(wnMSE / min over the four fixed samplers), i.e. the RMSE ratio
    at equal measured time.
* **Strategies compared on test scenarios.**
  * The four fixed samplers.
  * The best fixed sampler on the training scenarios.
  * A regime-aware rule learned on training: sampler X if m ≤ threshold,
    else Y, with the threshold below the largest training m.
  * An a-priori rule, "always RQMC-pca".
  * Two pilot-based selectors. Each gets a 10% pilot split over the four
    candidates, uses its measured pilot time, and runs the final estimate
    on fresh random numbers. Their predicted MSE at the remaining time T
    differs:
    * SE²·t/T, the MC rate (conservative for RQMC);
    * SE²·(t/T)² for RQMC (an optimistic N⁻² variance rate).
  * The oracle, the best fixed sampler per scenario by measured wnMSE. It
    is for evaluation only and is optimistic, since it is a minimum over
    noisy estimates.
* **Splits.** A random half of the scenarios (seeded), and train
  m ∈ {4, 12} / test m = 52. In the second split the regime rule can only
  place its threshold at m = 4. Aggregation is by geometric mean with a
  2 000-resample bootstrap 95% CI over test scenarios.

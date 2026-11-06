# Phase 4, Part B: confirmatory validation of the cost-based selection rule

**Verdict: PASS** on both pre-registered criteria, with important
qualifications (§5).

## 1. What was tested, and how it was protected against overfitting

* **The rule** (formulated post hoc in phase 3, §9.5): use PCA-RQMC iff the
  budget satisfies B ≥ 10·(a_m + 64·b_m), else plain MC. Here (a_m, b_m) is
  the pooled per-scrambling cost model t(n) = a + b·n, calibrated on the
  evaluation machine at the start of the run.
* **Frozen before any phase-4 run.** The rule, its comparators, the grid,
  budgets, seeds, replications, metrics and the success criteria are in
  `configs/phase4_cost_rule.json`. That file was committed and pushed as
  **`5f657d4`** before the validation script existed. The config is
  byte-identical at analysis time (`git diff 5f657d4 -- configs/` is empty).
* **Unseen grid.** 32 scenarios: m ∈ {8, 24, 52, 104}, K/S₀ ∈ {0.97, 1.03},
  σ ∈ {0.18, 0.33}, T ∈ {0.7, 2.0}, r = 0.04. None of these values occur in
  the phase-2 or phase-3 grids. Budgets were 5, 10, 20, 50 and 100 ms of
  measured wall time, with 3 Greeks × 4 samplers × 20 replications.
* **Run.** `python scripts/phase4_cost_validation.py` took 43 min. Output is
  in `results/phase4_cost_rule/confirmatory/`. The meta file records git
  `5f657d4` + uncommitted working tree: the runner and docs were not yet
  committed; the config was.
* **Nothing was tuned.** The rule, its 10 / 64 constants and the grid were
  not changed after any result was seen. The `--quick` pipeline check
  (2 scenarios) is stored separately and is not confirmatory.

## 2. Validation of the measurement itself

* **References.** PCA-RQMC with 32 × 2¹⁸ points (m ≤ 24) or 32 × 2¹⁶. Over
  128 independent plain-MC checks, z has mean 0.02, sd 1.00 and max 2.68.
* **Coverage** of the 95% intervals: median 0.95 for every sampler; minimum
  0.75–0.80 over 20 replications.
* **Calibration** took 5.6 s and is reported as amortised one-off setup.

| m | a [ms] | b [µs/point] | rule threshold [ms] |
|---|---|---|---|
| 8 | 0.35 | 0.32 | 3.7 |
| 24 | 0.69 | 1.38 | 7.7 |
| 52 | 0.97 | 4.36 | 12.5 |
| 104 | 2.06 | 8.96 | 26.3 |

## 3. Results against the pre-registered criteria

| Criterion | Result |
|---|---|
| **C1** Geometric-mean regret ≤ 1.25 in every (budget, Greek) cell | ✅ max **1.019** (15 cells) |
| **C2** Picks the better of MC / PCA-RQMC in ≥ 90% of significant cells | ✅ **450/450 = 100%**. Significance means the 95% bootstrap CI of log(wnMSE_MC/wnMSE_PCA) excludes 0; 30 cells were ambiguous |
| **C3** No worse than both comparators in ≥ 2/3 of cells | ✅ 15/15 |

Geometric-mean regret against the four-sampler oracle (`policy_regret.csv`):

| Policy | 5 ms | 10 ms | 20 ms | 50 ms | 100 ms |
|---|---|---|---|---|---|
| **frozen cost-model rule** (Δ / Γ / 𝒱) | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.01 | 1.00 / 1.02 / 1.00 | 1.00 / 1.01 / 1.01 | 1.00 / 1.00 / 1.00 |
| always PCA (best fixed on phase-3 training) | 2.67 / 3.47 / 2.63 | 2.08 / 2.46 / 2.18 | 1.50 / 1.72 / 1.55 | 1.00 / 1.01 / 1.01 | 1.00 / 1.00 / 1.00 |
| frequency heuristic (PCA iff m ≤ 12) | 1.00 / 1.00 / 1.00 | 1.32 / 1.22 / 1.32 | 2.29 / 1.71 / 2.23 | 4.76 / 3.14 / 4.21 | 7.64 / 4.60 / 6.57 |
| always MC | 1.85 / 1.74 / 1.77 | 3.4 / 3.0 / 3.1 | 7.8 / 5.5 / 6.5 | 17.7 / 10.7 / 13.8 | 30.2 / 16.4 / 21.9 |
| always RQMC-bb | 6.7–8.7 | 6.3–7.6 | 6.3–7.9 | 6.0–8.2 | 7.8–11.1 |
| always RQMC-seq | 10.5–13.7 | 12–13.5 | 13–17 | 18–25 | 24–34 |

The two comparators fail in opposite regimes:

* **Always PCA** fails at small budgets with large m, where per-scrambling
  overhead dominates.
* **The frequency heuristic** fails at large budgets. It ignores the budget
  and keeps MC at m ≥ 24.

Only a rule that uses *both* m (through the cost model) and B gets both
right.

**Oracle choices** (`summary.csv`) follow a single frontier:

* PCA at every budget for m = 8.
* MC below ≈ 7–10 ms for m = 24, below ≈ 15–20 ms for m = 52, and below
  ≈ 35–50 ms for m = 104.
* PCA above those budgets.

The sequential and bridge constructions are never the oracle.

## 4. Figures (`results/phase4_cost_rule/confirmatory/figures/`)

* `fig4B1_mse_vs_time`: MSE against measured time, per m and Greek.
* `fig4B2_mc_over_pca_ratio`: the wnMSE(MC)/wnMSE(PCA) ratio across budgets,
  with the frozen thresholds dotted.
* `fig4B3_crossover_by_m`: the empirical crossover budget per scenario vs
  the rule threshold.
* `fig4B4_policy_regret`: out-of-sample regret of the policies, with 95% CIs.
* `fig4B5_cold_vs_warm`: cold-start vs warm-cache wall time and MSE.

## 5. Qualifications (read before relying on the rule)

1. **The threshold is systematically early at large m.** The empirical
   crossover budget (where wnMSE_MC = wnMSE_PCA, interpolated per scenario)
   compares to the rule threshold as follows:

   | m | empirical crossover: median [range] | rule threshold |
   |---|---|---|
   | 24 | 7.3 ms [6.2, 11.1] | 7.7 ms |
   | 52 | 15.9 ms [14.3, 24.1] | 12.5 ms |
   | 104 | 38.6 ms [33.4, 67.9] | 26.3 ms |

   At m ≥ 52 the rule switches to PCA about 1.3–2.6× too early (Γ is the
   furthest off). The pre-registered budget grid (5/10/20/50/100 ms) has no
   point between threshold and crossover at m = 52 or 104, so this never
   produced a wrong decision here. **The pass is partly owed to the grid's
   coarseness.**

   Near the crossover the two samplers are close in efficiency by
   definition, so mistakes there should cost little regret. That was not
   measured. A finer budget grid near the frontier is the natural next
   confirmatory test. Changing the constant 10 based on this figure would
   be post hoc and would need its own pre-registration.
2. **Budgets are not honoured by RQMC at small B.**

   | | MC | RQMC (any construction) |
   |---|---|---|
   | p95 wall / B at 5 ms | 1.25 | **2.5–2.6** |
   | p95 wall / B at 10 ms | 1.08 | 1.5 |
   | share of runs > 1.10 × B, at 5 and 10 ms | 31% / 3% | **100% / 100%** |
   | max wall / B | — | **4.6** |

   The minimum of 4 scramblings × 16 points cannot fit for larger m. These
   overruns are **charged** in the time-normalised MSE: the analysis uses
   mean measured time. But they mean "a 5 ms RQMC estimate" is not
   actually achievable at m ≥ 24. The rule avoids RQMC in most of these
   cells, which is part of why it succeeds.
3. **Cold vs warm.** With the construction matrix built inside the timed
   region:
   * wall time rises 1.2–1.8× for BB/PCA only at 5–10 ms with m ∈ {52, 104};
   * everywhere else the wall ratio is 0.98–1.03.

   Building L costs ≤ 6.5 ms (BB, m = 104), so it matters only when it is
   comparable to the budget. Cold/warm MSE ratios from 10 replications
   scatter between 0.4 and 3 with no systematic direction; they are too
   noisy to support any claim. In the main grid the matrices were built
   before timing (amortised; build times are in `setup_costs.csv`).
4. **Machine and implementation dependence.** The threshold comes from this
   machine's SciPy Sobol scrambler cost. The rule *form* is portable; its
   numbers must be recalibrated per machine, which is how it is specified.
5. **Scope.** Only CPW-first Greeks of the arithmetic Asian digital under
   GBM, and one PCA-vs-MC dichotomy.

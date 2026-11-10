# Phase 5A: stress test of the cost-model rule near the MC / RQMC crossover

**Verdict: FRAGILE NEAR THE FRONTIER**, in both machine configurations. The
rule is right away from the crossover and materially wrong just above its own
threshold at m = 52 and m = 104.

The rule tested is the unmodified Phase-4 rule: *use PCA-RQMC iff the budget
B ≥ 10·(a_m + 64·b_m), else plain MC*, with (a_m, b_m) calibrated at the start
of the run. Nothing was tuned in this phase.

## 1. Protocol and how it was protected

* **Frozen first.** `configs/phase5_frontier.json` was committed as `8b05116`
  before the first run. The run's metadata records that commit as the
  config's last change. (The metadata flag `git_dirty` is true because the
  run's own output files already existed when it was read.)
* **New contracts.** 24 scenarios: m ∈ {24, 52, 104}, K/S₀ ∈ {0.95, 1.05},
  σ ∈ {0.22, 0.40}, T ∈ {0.5, 1.5}, r = 0.02. None of these values was used
  in Phases 2–4.
* **Dense budgets** inside the gap Phase 4 skipped: 6–12 ms (m = 24, 7
  values), 10–30 ms (m = 52, 8 values), 24–70 ms (m = 104, 9 values).
  72 (m, budget, Greek) cells.
* **Two configurations of one machine.** *Quiet*: 50 replications.
  *Contended*: 8 busy background processes on 16 logical CPUs, 20
  replications, cost model recalibrated under the load. No second computer
  was available, so "more than one hardware configuration" is only partly met.
* **Scheduler.** The unchanged Phase-3/4 scheduler. MC and RQMC were run back
  to back within each replication so machine drift hits both alike.
* **Commands.** `python scripts/phase5_frontier_study.py` (39 min) and
  `--config contended` (23 min). Output: `results/phase5_frontier/{quiet,contended}/`.

**Decision loss.** For each cell, pooled over the 8 scenarios,
ρ = √(wnMSE of the rule's choice / wnMSE of the other sampler), where
wnMSE = MSE × mean measured time. ρ is an RMSE factor: ρ = 1.5 means the
rule's choice needs 1.5² ≈ 2.3 times the time for the same accuracy. No
per-cell "oracle" is used, so there is no winner's-curse bias. Classes, fixed
in advance from the 95% bootstrap interval of ρ: *correct* (upper < 1),
*wrong, material* (lower > 1.10), *wrong, negligible* (lower > 1),
*indistinguishable* (otherwise).

## 2. Measurement checks

* **References.** 96 independent plain-MC checks: z has mean −0.01, standard
  deviation 1.03, maximum 3.0. Reference variance is at most 0.025% of any
  measured MSE.
* **Coverage** of the 95% intervals: 0.948 (MC) and 0.951 (RQMC) on average;
  lowest single cell 0.82 over 50 replications.

## 3. Results

### 3.1 Where the crossover is, against where the rule switches

| | m | rule threshold | empirical crossover, 95% CI: Delta | Gamma | Vega |
|---|---|---|---|---|---|
| quiet | 24 | 5.5 ms | below 6 ms | ≈ 6 ms | below 6 ms |
| | 52 | 12.1 ms | 14.0 [13.7, 14.2] | 15.8 [15.6, 16.4] | 14.2 [14.0, 14.4] |
| | 104 | 28.7 ms | 34.9 [34.4, 35.6] | 43.8 [43.2, 44.6] | 34.9 [34.4, 35.5] |
| contended | 24 | 6.7 ms | 6.7 [6.6, 6.8] | 6.9 [6.8, 7.0] | 6.7 [6.6, 6.8] |
| | 52 | 11.8 ms | 12.1 [11.9, 12.4] | 13.7 [13.4, 16.1] | 12.4 [12.0, 12.7] |
| | 104 | 21.3 ms | 26.5 [25.8, 27.5] | 35.1 [34.4, 36.3] | 26.6 [25.6, 27.9] |

* The crossover is sharply determined (intervals of a few percent).
* **H5A1 confirmed.** The threshold is early by a factor 1.0–1.3 at m = 52 and
  1.2–1.65 at m = 104. It is about right at m = 24.
* **Gamma crosses later than Delta and Vega**, by 12–32%. One threshold per m
  cannot be right for all three Greeks.
* The machine was *faster* under background load (crossovers 14–24% lower).
  Presumably the load kept the CPU out of a low-power state. Both the
  threshold and the crossover moved, but not by the same factor.

### 3.2 Wrong decisions and what they cost

| | correct | indistinguishable | wrong, negligible | wrong, material |
|---|---|---|---|---|
| quiet (72 cells) | 61 | 4 | 1 | **6** |
| contended (72 cells) | 59 | 7 | 0 | **6** |

The material cells, all "rule chose RQMC, MC was better":

| configuration | m | Greek | budget | ρ [95% CI] |
|---|---|---|---|---|
| quiet | 52 | Gamma | 14 ms | 1.77 [1.63, 1.95] |
| quiet | 104 | Delta | 32 ms | 1.52 [1.37, 1.67] |
| quiet | 104 | Gamma | 32 ms | **2.63 [2.41, 2.89]** |
| quiet | 104 | Gamma | 36 ms | 1.59 [1.46, 1.74] |
| quiet | 104 | Gamma | 40 ms | 1.50 [1.37, 1.64] |
| quiet | 104 | Vega | 32 ms | 1.52 [1.38, 1.67] |
| contended | 52 | Gamma | 12 ms | 1.71 [1.48, 1.95] |
| contended | 104 | Delta | 24 ms | 1.48 [1.26, 1.71] |
| contended | 104 | Gamma | 24 ms | **2.78 [2.38, 3.29]** |
| contended | 104 | Gamma | 28 ms | 1.53 [1.30, 1.77] |
| contended | 104 | Gamma | 32 ms | 1.61 [1.38, 1.88] |
| contended | 104 | Vega | 24 ms | 1.36 [1.16, 1.57] |

* **H5A2 refuted.** The errors in the gap are not negligible: an RMSE factor
  of 1.5–2.8, i.e. 2–8 times the computing time for the same accuracy.
* Per scenario (quiet): 486 of 576 scenario cells show a significant
  difference between the samplers; the rule is wrong in 44 (9.1%): 1 of 156
  at m = 24, 8 of 163 at m = 52, 35 of 167 at m = 104. Worst single ρ = 3.15.
* **Criteria.** D1 (no material cell): **fail**, 6 cells. D2 (geometric-mean
  ρ ≤ 1.05 over each budget range): pass (max 0.72 quiet, 0.96 contended).
  D3: 9.7% and 8.3% of cells wrong.
* **Still better than a fixed sampler** over these ranges. Geometric-mean
  regret against the better of the two per cell (quiet): rule 1.00–1.23,
  always RQMC 1.00–1.82, always MC 1.15–5.7.

### 3.3 MSE at equal measured runtime

Interpolating each sampler's MSE to the same measured time gives the same
picture without the "MSE × time" normalisation. Quiet, m = 104, Gamma:
MSE_MC / MSE_RQMC = 0.11 at 32 ms, 0.28 at 36 ms, 0.42 at 40 ms, 1.15 at 46 ms,
2.7 at 70 ms. So at 32 ms RQMC has about 9 times the MSE of MC at the same
measured time, while the rule selects it.

### 3.4 Sensitivity to cost-calibration noise

The threshold is a function of one calibration lasting about 4 seconds. We
repeated it 30 times per configuration.

| quiet | m = 24 | m = 52 | m = 104 |
|---|---|---|---|
| threshold used | 5.5 ms | 12.1 ms | 28.7 ms |
| 30 repeats: mean (range) | 6.3 (5.8–7.9) | 10.0 (8.2–14.3) | 22.5 (19.8–29.4) |
| coefficient of variation | 7% | 14% | 11% |
| median of 25 repeats just before the timed runs | 6.2 | 9.7 | 21.7 |
| median of 5 repeats after the timed runs | 6.0 | 8.5 | 20.4 |

* The calibration that was actually used sat at the **high (slow) end**: the
  machine sped up by roughly 30% during the session.
* That was lucky for the rule. A typical recalibration would have switched to
  RQMC at 21.5 ms for m = 104. Replaying the 30 thresholds on the measured
  data: the median recalibration flips **15 of the 72 decisions**, and its
  worst cell has ρ = 7.9 (RQMC chosen for Gamma at 24 ms, m = 104).
* Contended: CV 7–8%; the median recalibration flips 3 decisions; worst-cell
  ρ between 1.7 and 5.3.

**So the reported 6 material cells are a favourable case.** The rule's error
near the frontier is driven as much by *when* the machine was calibrated as by
the constants 10 and 64.

### 3.5 Budget overruns

| | MC | PCA-RQMC |
|---|---|---|
| quiet: worst cell p95 of wall/B | 1.06 | 1.12 |
| quiet: worst cell share of runs > 1.10·B | 0.2% | 21% |
| quiet: max wall/B | 1.12 | 1.23 |
| contended: worst cell share > 1.10·B | 0.2% | 59% |
| contended: max wall/B | 1.44 | 2.38 |

Overruns are charged (the analysis uses measured time). They are far smaller
than in Phase 4 because these budgets are above the scheduler's minimum
workload; Phase 5B addresses the small-budget case.

## 4. Why the rule is early

At the threshold the scheduler runs 10 scramblings of 64 points. With the
quiet calibration, the set-up of one scrambling costs a = 2.42 ms at m = 104,
and its 64 points cost 64·b = 0.45 ms. **About 84% of RQMC's time at the
threshold is scrambler set-up** (82% at m = 52, 87% at m = 24). RQMC therefore
processes roughly a sixth of the points MC does. Its faster convergence only
makes up for that once N is a few hundred, and more for Gamma, whose integrand
is rougher. The constant "64 points" was chosen in Phase 3 on m ≤ 52 and does
not grow with dimension.

The MSE ratio also moves in steps (figure 5A1): it jumps each time the
scheduler's power-of-two N doubles. The frontier is a staircase, not a line.

## 5. What this does and does not establish

* It establishes that the frozen rule, as specified, makes materially wrong
  choices in a band above its threshold at m ≥ 52, and that its threshold is
  unstable under recalibration.
* It does **not** provide a corrected rule. Refitting the constants to these
  data and then calling the result "validated" would be circular. A revised
  rule needs development data and a fresh holdout.
* Phase 5B suggests the better remedy is to remove the cause: with a cheaper
  randomisation the per-replicate overhead falls by 50–100×, and on the 5B
  grid RQMC beats MC at every budget tested, so there is no frontier to
  locate (`PHASE5_SAMPLING_POLICY.md`).
* Scope: one machine, CPW-first Greeks of the arithmetic Asian digital under
  GBM, SciPy's Sobol' implementation.

## 6. Figures (`results/phase5_frontier/<config>/figures/`)

* `fig5A1_ratio_dense_grid`: wnMSE(MC)/wnMSE(RQMC) across the dense grid with
  95% bands; rule threshold (dashed) and recalibration range (grey).
* `fig5A2_decision_loss`: ρ of the frozen rule per cell, with the 1.10 line.
* `fig5A3_crossover_and_overruns`: crossover intervals against the threshold
  and its recalibration range; overrun distribution.

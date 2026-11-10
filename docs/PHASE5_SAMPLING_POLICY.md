# Phase 5B: budget-feasible sampling schedules

> **Erratum (Phase 6, 2026-11-10).** Where this document says "Owen scrambling" or "Owen-scrambled" for the RQMC randomisation, read **SciPy's LMS+shift**: `scipy.stats.qmc.Sobol(scramble=True)` applies Matoušek's random linear matrix scrambling followed by a random digital shift, not Owen's nested uniform scrambling. Every estimate stays unbiased and every number reported here is unchanged; only the label was wrong. Citations of Owen's or Liu & Owen's papers are unaffected. See `docs/PHASE6_REPORT.md`.

**Summary.** Planning from a cost model and stopping by look-ahead removes
the budget overruns. It does not, by itself, make RQMC more accurate. What
does is replacing SciPy's per-replicate scrambling by a random digital shift
of one cached Sobol' set: 11 times lower time-normalised MSE on this grid, and
RQMC then beats plain MC at every budget tested, down to 2 ms.

Code: `src/mc_greeks/budget_policy.py`. Protocol:
`configs/phase5_sampling_policy.json`, committed as `9bbcfc2` before the run.
Run: `python scripts/phase5_policy_study.py` (22 min). Output:
`results/phase5_sampling_policy/study/`.

## 1. The problem with the old scheduler

The Phase-3/4 scheduler picks the largest power-of-two N such that 10
scramblings fit, but never fewer than 16 points and never fewer than 4
scramblings, and checks the clock only *after* each scrambling. So:

* at small budgets it schedules a minimum workload it could know in advance
  will not fit;
* every run overshoots by up to one scrambling.

## 2. The schedule as an optimisation problem

Cost model, fitted per (m, Greek, randomisation) at the start of the run from
timings of the real executor (relative least squares, medians of 7 repeats):

  cost_RQMC(R, N) = c₀ + R·(a + b·N),  cost_MC(K, n_b) = c₀′ + K·(u + c·n_b).

Working error model: Var = κ·N^{−α}/R with α = 2, taken from the Phase-4
fixed-N study (development data, not this grid).

  minimise  N^{−α}/R  (or t²_{R−1}·N^{−α}/R for the "ci" objective)
  subject to c₀ + R(a + bN) ≤ 0.95·B,  R ≥ R_min,  N a power of two ≥ 16.

Solved by enumeration (`optimise_schedule`; the tests compare it with brute
force). `PHASE5_MATHEMATICS.md` §7 shows why the optimum is "largest N that
leaves R_min replicates" and why the "ci" objective settles near R ≈ 4.

**Feasibility.** If no (R, N) satisfies the constraint the plan is
*infeasible*: nothing is run and the estimate is returned as "infeasible".
A composite policy then falls back to MC, and if two minimal MC blocks do not
fit either, reports that the request cannot be met.

**Look-ahead stopping.** A unit of work is started only if elapsed time plus
its predicted cost is within B. Two units always run (a standard error needs
two). Stopping depends only on the clock, never on simulated values, so
estimates stay unbiased. A hard wall-clock guarantee is impossible on a
general-purpose operating system; overruns are measured and reported.

**Constants fixed before the run:** α = 2, R_min = 8, margin 5%, N ≥ 16.

## 3. Strategies compared

| name | description |
|---|---|
| MC-base, PCA-base | Phase-3/4 schedulers, unchanged |
| MC-look | MC blocks sized from the cost model, look-ahead stopping |
| PCA-fixed64, PCA-fixed512 | fixed N, R from the cost model, feasible iff R ≥ 8 |
| PCA-opt-mse | the optimisation above, R_min = 8, Owen scrambling |
| PCA-opt-ci | "ci" objective (R ≥ 3), Owen scrambling |
| PCA-shift-mse, PCA-shift-ci | the same two with random digital shifts |

**The digital shift.** One unscrambled Sobol' set is cached as 30-bit
integers k. A replicate draws s uniformly from {0, …, 2³⁰−1}^d and v ~ U(0,1)^d
and uses u = ((k XOR s) + v)/2³⁰. XOR with a uniform s makes each coordinate
uniform on the integers; adding v makes it uniform on [0,1). The XOR permutes
dyadic intervals, so the shifted set is still a digital net with the same
equidistribution. This is a textbook randomisation (e.g. Dick &
Pillichshammer 2010), not a contribution of this project. Tests check
uniformity, unbiasedness and the one-point-per-interval net property. In
theory it lacks the extra variance-rate gain Owen scrambling has for smooth
integrands; here both show the same rate.

Grid: 12 new scenarios (m ∈ {24, 52, 104}; K/S₀ ∈ {0.96, 1.04};
σ ∈ {0.25, 0.45}; T = 1.2; r = 0.01), budgets 2, 5, 10, 20, 50 ms, three
Greeks, 30 replications; plus a 24-budget sweep (2–64 ms) on one scenario per
m. Reference checks: max |z| = 2.70, standard deviation 0.95.

## 4. Results against the pre-registered hypotheses

### S1. Deadlines — supported

Share of runs exceeding 1.10·B (worst cells):

| budget | PCA-base m = 24 / 52 / 104 | all look-ahead strategies (worst) |
|---|---|---|
| 2 ms | 100% / 100% / 100% | 4% |
| 5 ms | 33% / 91% / 100% | 2% |
| 10 ms | 3% / 21% / 86% | 4% |
| 20 ms | 0% / 1% / 13% | 0% |
| 50 ms | 2% / 0% / 0% | 1% |

* Look-ahead strategies: 77 feasible (m, budget, strategy) cells, none with
  more than 5% of runs late; worst cell 4.4%; largest single overrun 1.39×.
* PCA-base: 9 of 15 cells above 5%; 95th percentile 5.2× the budget at
  m = 104 and 2 ms; largest overrun 7.3×. MC-base is late in 26–29% of runs
  at 2 ms.
* No look-ahead strategy ever ran a plan predicted not to fit. Owen
  scrambling with R_min = 8 was infeasible in 6 of 15 cells (all of 2 ms;
  5 ms at m ≥ 52; 10 ms at m = 104) and said so.

### S2. Interval coverage — supported

Pooled coverage of nominal 95% intervals is 0.948–0.952 for all nine
strategies. That includes the "ci" objective, which typically uses only 3–7
replicates (t-quantiles 4.3–2.4). The lowest single cells are 0.91 (PCA-base)
and 0.92 (PCA-shift-ci), within sampling error of 0.95 for 360 runs.

### S3. Is the optimised Owen schedule more accurate than the old one? — not supported

Time-normalised MSE, PCA-opt-mse / PCA-base, geometric mean over the 100
cells where both ran: **1.06 [0.99, 1.15]**. Cell by cell it ranges from 0.25
to 7.1. The new schedule fixes feasibility; with the same randomisation it is
not more efficient. Fixed N is clearly worse than optimised N: 2.5× (N = 512)
and 66× (N = 64).

### S4. Digital shifts — strongly supported, and larger than expected

| measured | m = 24 | m = 52 | m = 104 |
|---|---|---|---|
| set-up per replicate, Owen (a) | 0.59 ms | 1.10 ms | 2.14 ms |
| set-up per replicate, shift (a) | 0.024 ms | 0.012 ms | < 0.001 ms |
| cost per point, Owen / shift (b) | 0.94 / 0.80 µs | 2.6 / 2.1 µs | 5.4 / 5.2 µs |

* Feasible in **15 of 15** cells, against 9 for Owen.
* Time-normalised MSE, shift / Owen (same objective): **0.09 [0.08, 0.09]**
  over 100 cells. The gap closes as the budget grows: at 50 ms it is
  0.85 [0.69, 1.04] for m = 24 and 0.33–0.37 for m = 52 and 104.
* The gain is explained by throughput: at m = 104 and 50 ms the shift affords
  N = 1024 where Owen affords 512, and variance ∝ N^{−2} predicts the
  observed factor ≈ 0.3. Per point the two randomisations are equally good.
* **Against MC:** PCA-shift-mse has lower time-normalised MSE than MC-look in
  every one of the 15 cells. The closest is m = 104 at 2 ms, ratio
  0.68 [0.56, 0.85]; at 50 ms the ratios are 0.02 or smaller.
* By contrast, Owen RQMC at its smallest feasible budget is 5–10× *worse*
  than MC (m = 24 at 5 ms, m = 52 at 10 ms, m = 104 at 20 ms), which is the
  Phase-5A frontier seen from the other side.

**Implication.** The MC/RQMC crossover studied in Phases 3, 4 and 5A is
largely a property of the scrambler's set-up cost, not of RQMC. With a cheap
randomisation the crossover moves below 2 ms on this grid.

### "ci" versus "mse" objective

Fewer, larger replicates are more efficient: opt-ci / opt-mse =
0.14 [0.13, 0.15] with Owen scrambling and 0.57 [0.54, 0.60] with shifts,
with coverage intact (S2). The price is a wider error bar per unit of
accuracy and too few replicates to estimate a covariance matrix.

### S5. The scramble-size cliff — inconclusive

Largest increase of MSE between adjacent budgets on the 24-point sweep
(a monotone curve would give ≤ 1):

| m | PCA-base | PCA-opt-mse | PCA-shift-mse |
|---|---|---|---|
| 24 | 1.47 | 1.48 | 1.42 |
| 52 | 1.73 | 1.62 | 1.89 |
| 104 | 2.16 | 1.23 | 1.85 |

With 30 replications an MSE ratio has roughly ±40% noise, so only the m = 104
difference is suggestive. The new schedules have fewer upward steps (3–5 of
23 against 6–8), but the staircase in N is inherent to power-of-two sample
sizes and is still visible in figure 5B3.

### A flaw found in the new MC schedule

MC-look is **1.15 [1.09, 1.20] times worse** than MC-base in time-normalised
MSE, and about 2.1–2.2× worse at 50 ms for m ≥ 52. It uses blocks of up to
4096 paths (2048 at m = 104, about 2 MB of normals), which is slower per path than
the cache-sized blocks of the old scheduler. Its deadline behaviour is
better; its throughput is not. Phases 5C/5D cap MC blocks at the old
cache-friendly size. That change was made on these data and is applied there
to different scenarios.

### Composite policies

Regret against the best look-ahead strategy in each cell:

| policy | regret range over the 15 (m, budget) cells |
|---|---|
| Phase-4 pipeline (old rule, old schedulers) | 1.13 – 13.3 |
| feasible-owen (rule form, new Owen schedule, MC fallback) | 1.16 – 12.9 |
| feasible-shift (rule form with the shift cost model) | 1.00 in 13 cells; 1.07; 1.30 |

The 1.30 is m = 104 at 2 ms: the rule form "B ≥ 10(a + 64b)" with the shift
cost model says MC (threshold 3.3 ms) while the shift was already better.
So the rule form remains slightly conservative even here. Its constants were
never validated for this randomisation; this variant is exploratory.

## 5. Recommendations taken forward

1. Use random digital shifts for small budgets; Owen scrambling's advantage
   did not appear at any budget tested.
2. Plan from the cost model, stop by look-ahead, report infeasibility.
3. Keep R_min = 8 when a covariance matrix of several Greeks is wanted; the
   "ci" objective is the efficient choice for a single number.
4. Cap MC blocks at cache size.

## 6. Limitations

* One machine and SciPy's Sobol' implementation. A faster Owen scrambler
  would shrink the gap; the comparison is of implementations as much as of
  methods.
* The cost model's median relative fit error is 6–12% for RQMC and up to 25%
  for MC at m = 104. Phase 5A showed machine speed drifting by 30% within a
  session. Look-ahead stopping, not the fit, is what protects the deadline.
* α = 2 is an empirical constant for PCA-type constructions of this
  integrand.
* The digital shift was tested on dimensions 23–103 and N ≤ 8192 only.

## 7. Figures (`results/phase5_sampling_policy/study/figures/`)

* `fig5B1_overruns`: 95th percentile of measured time over budget.
* `fig5B2_mse_vs_time`: MSE against measured time, Delta.
* `fig5B3_cliff_sweep`: MSE and N along the fine budget sweep.
* `fig5B4_composite_policies`: MSE against time for the composite policies.

# Phase 5C and 5D: joint Greeks and the economics of transform reuse

**Summary.**

* Estimating Delta, Gamma and Vega from one shared simulation beats three
  separate simulations at the same total time: by ≈ 3× for MC, 5–16× for path
  PCA, 2.6–37× for gradient PCA (risk-vector loss).
* One joint rotation serves all three Greeks and all weightings tested.
* With shared paths, digital shifts and 256 training paths, **training paid
  for itself within a single valuation** in every cell tested (break-even
  M\* = 1), including m = 104 at 10 ms where training takes 44% of the budget.
* A rotation trained on one contract worked on all 34 neighbouring contracts
  tested, always far better than path PCA and statistically
  indistinguishable from retraining in 30 of them.

Code: `src/mc_greeks/joint_greeks.py`, `budget_policy.py`. Protocol:
`configs/phase5_joint.json`, committed as `578048d` before the run.
Run: `python scripts/phase5_joint_study.py` (22 min). Output:
`results/phase5_joint/study/`.

## 1. Set-up

* **Contracts.** 8 new scenarios: m ∈ {24, 104}, K/S₀ ∈ {0.98, 1.06},
  σ ∈ {0.24, 0.38}, T = 0.9, r = 0.015.
* **Scheduler.** Chosen from Phase 5B on a different grid: random digital
  shifts, "mse" objective, at least 8 replicates, look-ahead stopping; MC
  blocks capped at cache size.
* **Loss.** L = Σ_g w_g·MSE_g/s_g², with s_g the per-path standard deviation
  of the plain-MC CPW-first estimator (2¹⁶ independent paths). MSE_g/s_g² is
  the reciprocal of the number of plain-MC paths giving the same accuracy.
  It is unit-free and never divides by a Greek near zero.
* **Weightings.** Equal (⅓, ⅓, ⅓) and three tilted ones (0.8 on one Greek).
* **Training.** 256 gradient paths per rotation, 5 independently trained
  instances per scenario; evaluation streams are independent of training
  streams. Timed runs cycle through the instances, so training variability
  is included.
* **Checks.** References: max |z| = 1.90 against plain MC. Pooled 95%
  coverage per Greek is 0.941–0.956 for all six timed strategies. Largest
  overrun 1.13× the budget; at most 0.8% of runs above 1.10×.

## 2. The shared-path interface

`cpw_first_joint(f, p, Z)` returns an (N, 3) array. The exponentials, the path
average, δ and φ(δ) are computed once. Tests verify agreement with the three
single-Greek formulas to 12 digits, and that a joint run and three
single-Greek runs on the same points give identical estimates.
`run_rqmc` returns the estimate vector and its 3×3 covariance estimated across
replicates.

## 3. Results, 5C

### 3.1 The gradient covariances are rank one and aligned

| | m = 24 | m = 104 |
|---|---|---|
| top-eigenvalue share of G, Delta / Gamma (minimum) | 0.999 / 0.998 | 0.999 / 0.999 |
| top-eigenvalue share of G, Vega (minimum) | 0.968 | 0.993 |
| \|cos\| between top directions of any two Greeks (minimum) | 0.9987 | 0.9999 |
| \|cos\| between that direction and path PCA's first (median) | 0.993 | 0.993 |

This is the structure derived in `PHASE5_MATHEMATICS.md` §5–6: every Greek is
a function of the one scalar δ(B). The last row shows how small the
correction is. The Greek-aware direction is only about 7° away from path
PCA's first direction, yet that rotation is worth a factor 3–10 in variance.

### 3.2 Fixed N: joint versus specific versus path PCA

Risk-vector loss relative to path PCA (geometric mean over scenarios; no
reference needed, since every estimator is unbiased):

| rotation | m = 24: N = 2⁸ / 2¹⁰ / 2¹² | m = 104: N = 2⁸ / 2¹⁰ / 2¹² |
|---|---|---|
| joint, equal weights | 0.19 / 0.30 / 0.32 | 0.10 / 0.09 / 0.10 |
| Delta-specific | 0.21 / 0.29 / 0.28 | 0.11 / 0.10 / 0.09 |
| Gamma-specific | 0.19 / 0.27 / 0.31 | 0.13 / 0.10 / 0.09 |
| Vega-specific | 0.23 / 0.30 / 0.31 | 0.12 / 0.09 / 0.09 |

* Gradient PCA lowers the loss 3–5× at m = 24 and about 10× at m = 104.
* By Greek, with the joint rotation at m = 24, the variance ratio against
  path PCA is 0.12–0.18 for Gamma, 0.18–0.25 for Delta and 0.34–0.66 for Vega.
* **J2 supported.** Under each of the four weightings, the loss of every
  trained rotation relative to the equal-weight joint one has geometric mean
  between 0.89 and 1.12. Single scenarios scatter between 0.54 and 1.97,
  which is the noise of a variance ratio from 80 scramblings. The weights do
  not matter, because the rotation barely depends on them.

### 3.3 Shared against separate simulations at equal total time

Loss ratio shared / separate (equal weights, geometric mean over 4 scenarios,
range over the five total budgets 6–150 ms; every interval excludes 1):

| | m = 24 | m = 104 |
|---|---|---|
| MC | 0.30 – 0.42 | 0.28 – 0.37 |
| path PCA | 0.06 – 0.18 | 0.11 – 0.20 |
| gradient PCA (joint rotation vs three specific) | 0.03 – 0.38 | 0.03 – 0.27 |

* **J1 supported.** For MC the factor is about ⅓, as it must be: the shared
  run has three times the paths for each Greek. For RQMC it is larger
  because variance falls faster than 1/N.
* The shared and separate runs use the same total measured time (within 4%
  in 9 of 10 cells, 10% in one), so this is not an artefact of ignoring
  saved work.

### 3.4 Joint rotation against path PCA and MC under a wall-clock budget (training amortised)

| total budget | m = 24: GPCA / PCA | m = 104: GPCA / PCA | m = 104: PCA / MC |
|---|---|---|---|
| 6 ms | 0.27 [0.20, 0.34] | 0.08 [0.06, 0.10] | 0.20 [0.16, 0.25] |
| 15 ms | 0.15 [0.12, 0.19] | 0.14 [0.11, 0.17] | 0.24 [0.19, 0.29] |
| 30 ms | 0.29 [0.22, 0.37] | 0.12 [0.09, 0.15] | 0.09 [0.07, 0.11] |
| 60 ms | 0.30 [0.24, 0.37] | 0.15 [0.11, 0.20] | 0.06 [0.05, 0.08] |
| 150 ms | 0.17 [0.13, 0.22] | 0.03 [0.03, 0.04] | 0.02 [0.02, 0.03] |

* **J3 supported**: 3–7× lower loss at m = 24 and 7–30× at m = 104.
* The same holds under the tilted weightings (ratios 0.03–0.54).
* Path PCA beats MC in every cell; at m = 24 the ratio is below 0.01.
* The ratios are not monotone in the budget. That is the power-of-two
  staircase of Phase 5B, not noise in the ranking.

### 3.5 The error covariance

Correlation of estimation errors over 40 runs per scenario (range, over the
10 cells, of the median across 4 scenarios):

| | Delta–Gamma | Delta–Vega | Gamma–Vega |
|---|---|---|---|
| shared MC | −0.20 to 0.01 | −0.03 to 0.00 | −0.14 to 0.12 |
| shared path PCA | −0.49 to 0.12 | −0.16 to 0.16 | −0.40 to 0.24 |
| shared gradient PCA | −0.09 to 0.45 | −0.18 to 0.17 | **−0.82 to −0.30** |
| separate (any) | ≈ 0 | ≈ 0 | ≈ 0 |

After the gradient rotation the remaining Gamma and Vega errors are strongly
negatively correlated. For a portfolio quantity aᵀĝ the variance is aᵀΣa, so
the off-diagonal term matters: it helps when Gamma and Vega enter with the
same sign and hurts when they enter with opposite signs. Reporting three
separate standard errors would hide this. The covariance returned by
`run_rqmc` makes it available.

## 4. Results, 5D

### 4.1 Break-even: how many valuations before training pays?

Deployment model: M valuations, total time M·B. A trained strategy simulates
with B − C_train/M each. Measured training cost (256 paths, median):

| | m = 24 | m = 104 |
|---|---|---|
| one joint rotation | 0.33 ms | 4.4 ms |
| three Greek-specific rotations | 0.51 ms | 9.5 ms |

Loss relative to shared path PCA at the same total time, [95% interval]:

| strategy | cell | M = 1 | M = 2 | M = 10 | amortised |
|---|---|---|---|---|---|
| joint, shared | m = 24, 10 ms | 0.13 [0.10, 0.16] | 0.11 | 0.14 | 0.12 |
| joint, shared | m = 24, 50 ms | 0.30 [0.24, 0.38] | 0.25 | 0.33 | 0.33 |
| joint, shared | m = 104, 10 ms | **0.18 [0.15, 0.22]** | 0.09 | 0.07 | 0.06 |
| joint, shared | m = 104, 50 ms | 0.23 [0.18, 0.29] | 0.20 | 0.17 | 0.17 |
| specific, separate | m = 24, 10 ms | 8.7 | 8.3 | 7.5 | 7.7 |
| specific, separate | m = 24, 50 ms | 0.65 | 0.60 | 0.63 | 0.69 |
| specific, separate | m = 104, 10 ms | 461 | 1.13 | 0.52 | 0.52 |
| specific, separate | m = 104, 50 ms | 1.90 | 1.53 | 1.39 | 1.54 |

* **J4, joint rotation: break-even M\* = 1 in all four cells**, significant in
  each. Even at m = 104 and 10 ms, where training consumes 4.4 of the 10 ms,
  the remaining 5.6 ms with the rotation beats 10 ms of path PCA by a factor
  5.6 in loss. The hypothesised "larger M\* at small budgets and large m" did
  not appear within this grid. The cost of the charge is visible (0.18 at
  M = 1 against 0.06 amortised) but never reverses the ranking.
* **Three specific rotations with three simulations is a poor design.** It
  never breaks even at m = 24 / 10 ms or m = 104 / 50 ms, needs M = 5 at
  m = 104 / 10 ms, and at M = 1 there its training (9.5 ms) leaves so little
  that it falls back to MC (loss 461×).
* Against plain MC the joint strategy at M = 1 has loss ratios of 0.05 or less.

**How this relates to Phase 4.** Phase 4 found one-off gradient PCA 9–16×
*worse* than path PCA at m = 104 and 50 ms. Three things differ here, and the
experiment does not separate them: (i) 256 training paths instead of 1024
(Phase 4 itself found 32 sufficient); (ii) one joint rotation and one
simulation instead of one per Greek; (iii) digital shifts with a cost-model
schedule, which removed the scramble-size cliff that Phase 4 diagnosed as the
cause. Phase 4's result stands for its configuration; it was a property of
that pipeline, not of gradient PCA.

### 4.2 Transfer to neighbouring contracts

A rotation trained on a base contract (one per m) was applied, unchanged, to
17 other contracts per m: strike × {0.9 … 1.1}, volatility × {0.5 … 2},
maturity × {0.5 … 2}, and two regime changes (strike × 0.9 with volatility
× 2; strike × 1.1 with volatility × 0.5). Fixed N = 1024, 80 scramblings per
arm, fresh rotations trained on independent streams.

| over the 34 targets | range |
|---|---|
| loss, transferred / path PCA | 0.014 – 0.47 (all significantly below 1) |
| loss, transferred / freshly trained | 0.56 – 1.64 |
| targets where transferred is significantly worse than fresh | 2 (1.64 and 1.48, both m = 24, strike × 0.9) |
| targets where transferred is significantly *better* than fresh | 2 |
| \|cos\| between transferred and fresh top directions | 1.000 in every case |

* **J5 partly supported.** Transfer keeps essentially the whole benefit. The
  predicted growth of the loss with distance in K, σ or T is **not** visible:
  the ratios scatter around 1 with no trend, and 2 + 2 significant
  deviations among 34 two-sided tests is close to what chance gives.
* The reason is the last row. The dominant direction does not depend on the
  contract within this family.
* **Retraining.** No tested move made retraining clearly worthwhile. The
  model-based retraining break-evens in `d2_transfer.csv` are computed from
  point estimates that are mostly noise around 1 and should not be used.

**A practical reading.** For this contract family a risk engine should
train one joint rotation per monitoring frequency m, store it (an
(m−1)×(m−1) matrix), and reuse it across strikes, volatilities and
maturities. Training is cheap enough that even a single valuation repays it.

## 5. Limitations

* One contract family (arithmetic Asian digital, GBM, CPW-first). The
  rank-one, contract-independent direction is what makes joint rotations and
  transfer work; Phase 5E tests a contract without the single scalar.
* Since the top direction is contract-independent, a rotation with *no*
  training (for example aligned with the gradient of the path average, in the
  spirit of Imai & Tan's linear transformation) might do as well. That was
  not tested.
* Transfer was tested at fixed N only, not under a wall-clock budget, and
  only for the same m. A rotation cannot be reused across different m.
* 4 scenarios per m; fixed-N variance ratios carry roughly ±30% noise.
* The comparison with Phase 4 changes three things at once (§4.1).
* One machine; timings are specific to it.

## 6. Figures (`results/phase5_joint/study/figures/`)

* `fig5C1_fixed_n_loss`: loss of each rotation relative to path PCA.
* `fig5C2_shared_vs_separate`: loss against measured total time.
* `fig5D1_break_even`: loss relative to path PCA against M.
* `fig5D2_transfer`: transferred / fresh and transferred / PCA along K, σ, T.

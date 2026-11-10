# Phase 6 report: a validated digital-shift baseline and a training-free joint rotation

Phase 6 was narrowed to two deliverables:

* **6A.** Independently confirm the Phase-5 digital-shift result and find
  out why it happens.
* **6B.** Find out whether the trained joint gradient-PCA rotation can be
  replaced by a deterministic one.

Derivations are in [`PHASE6_MATHEMATICS.md`](PHASE6_MATHEMATICS.md).

## 0. Summary

| | finding |
|---|---|
| **Erratum** | SciPy's `scramble=True` is Matoušek's linear matrix scrambling plus a digital shift (LMS+shift), not Owen scrambling as Phases 3–5 said. Numbers unchanged; label corrected. |
| **6A confirmed** | On 24 new contracts the digital shift again beats SciPy-scrambled Sobol' at equal measured time: 0.015–0.57× the time-normalised loss at budgets ≤ 20 ms. SciPy's version could not run at all in 6 of 15 (m, budget) cells. |
| **6A cause** | Mostly SciPy's per-replicate engine construction. My own LMS+shift on cached direction numbers has the same variance (ratio 0.90–1.07) and recovers most of the gap. |
| **6A refuted** | The theory-based expectation that scrambling beats a bare shift per point did not hold for these integrands: variance ratios went both ways (0.68–1.23) and all three randomisations converge at the same rate (variance ∝ N⁻²·⁰). The bare shift stays cheapest, and my LMS was 1.9× less efficient per unit time. |
| **6B confirmed** | A deterministic rotation (second-order expansion at the design point, Greek weights integrated by quadrature) matches a rotation trained on 32,768 gradient paths: 0.98–1.07× its loss at fixed N. Training can be eliminated without loss of accuracy. |
| **6B refuted** | It does not beat the practical 256-path training, which was already near-optimal on the holdout (0.94–1.01×), and it is not cheaper to build (1.2–3.2 ms against 0.3–3.3 ms). One-off under a wall-clock budget the two are indistinguishable in 9 of 12 cells. |
| **6B confirmed** | The leading direction alone is not enough. A rotation that gets it right (|cos| ≥ 0.998) but keeps path-PCA order for the rest loses 1.46–1.76×. |

**Practical baseline after Phase 6.** Shifted Sobol' with a schedule planned
from a cost model; one joint rotation per monitoring frequency, either
trained on about 256 gradient paths or built deterministically. The choice
between the two is about reproducibility and explanation, not accuracy or
speed.

## 1. Set-up and pre-registration

| | |
|---|---|
| starting point | `main` after PR #3 merged (`8e79300`) |
| branch | `phase6/shift-holdout-training-free` |
| protocols frozen | `configs/phase6_shift_holdout.json` and `configs/phase6_training_free.json`, commit `959db40`, before either holdout run |
| holdout contracts | 24: m ∈ {16, 40, 80}, K/S₀ ∈ {0.93, 1.07}, σ ∈ {0.17, 0.35}, T ∈ {0.75, 1.75}, r = 0.035. None of these values occurs in any earlier grid. |
| 6B development data | the eight Phase-5C contracts, already used, for choosing the primary candidate (`results/phase6_training_free/dev/`) |
| machine | the same Windows laptop as Phases 4–5 (16 logical CPUs), Python 3.12.10, NumPy 2.4.3, SciPy 1.16.2 |
| tests | 349 pass (24 new) |

**Second environment: not yet run.** The workflow
`.github/workflows/phase6-portable.yml` runs a reduced 6A study on a GitHub
Ubuntu runner and uploads the results. It needs the branch on GitHub and a
manual trigger. Until then, every timed number below comes from one machine.
The equal-N results do not depend on hardware.

## 2. Phase 6A: the digital shift on a holdout

Commands: `python scripts/phase6_shift_study.py` (25 min). Output:
`results/phase6_shift/holdout/`.

Three randomisations of the same Sobol' points, all with the path-PCA
construction and all three Greeks on shared points:

* **SciPy LMS+shift**, a new SciPy engine per replicate. Phases 4–5 called
  this "Owen"; the config key `owen` is kept.
* **Own LMS+shift**, the same randomisation on cached direction numbers.
* **Digital shift only**, as in Phase 5.

### 2.1 Equal point counts (hardware-independent)

Variance ratio of replicate means (risk-vector loss, geometric mean over 8
contracts, 100 replicates each, 95% bootstrap interval):

| m | N | own / SciPy LMS | shift / LMS |
|---|---|---|---|
| 16 | 2⁶ … 2¹⁴ | 0.90 – 1.04 | 0.57 – 1.00 (shift better in 3 of 5) |
| 40 | 2⁶ … 2¹⁴ | 0.99 – 1.07 | 0.84 – 1.34 |
| 80 | 2⁶ … 2¹⁴ | 0.95 – 1.07 | 1.07 – 1.24 (LMS better in 3 of 5) |

* **A1 (implementation check): pass.** In 15 of 15 cells the interval
  contains 1. The two implementations are the same randomisation.
* **A2 (scrambling beats a shift per point at N ≥ 2¹⁰): refuted.**
  * At m = 16 the shift was significantly *better* at N = 2¹⁰ and 2¹⁴
    (0.73, 0.68).
  * At m = 80 it was worse at N = 2¹² (1.23 [1.12, 1.36]).
  * Elsewhere the two are indistinguishable.
* **Convergence.** All three randomisations converge at the same rate: the
  median slope of log variance against log N is −2.01 (SciPy LMS), −1.99
  (own LMS) and −1.98 (shift). So the faster rate that scrambling can give
  smooth integrands did not appear at these dimensions (15–79) and sample
  sizes (N ≤ 16,384).

### 2.2 Equal measured time

Same scheduler as Phase 5 (look-ahead stopping, at least 8 replicates,
power-of-two N, α = 2), each arm with its own cost model; 50 replications;
budgets 2–50 ms. Time-normalised loss ratios, [95% CI]:

| m | budget | shift / SciPy LMS | own LMS / SciPy LMS | own LMS / shift |
|---|---|---|---|---|
| 16 | 2 ms | SciPy infeasible | SciPy infeasible | 4.73 [4.08, 5.51] |
| | 5 | 0.13 [0.11, 0.15] | 0.14 [0.12, 0.16] | 1.05 [0.90, 1.22] |
| | 10 | 0.42 [0.36, 0.49] | 0.71 [0.60, 0.82] | 1.67 [1.43, 1.94] |
| | 20 | 0.57 [0.49, 0.67] | 0.75 [0.65, 0.88] | 1.32 [1.12, 1.56] |
| | 50 | 0.84 [0.73, 0.97] | 1.15 [0.99, 1.32] | 1.36 [1.18, 1.58] |
| 40 | 2–5 | SciPy infeasible | SciPy infeasible | 2.67 – 6.60 |
| | 10 | 0.015 [0.013, 0.017] | 0.015 | 1.05 [0.91, 1.21] |
| | 20 | 0.18 [0.16, 0.21] | 0.22 | 1.22 [1.05, 1.42] |
| | 50 | 0.60 [0.52, 0.70] | 0.79 | 1.31 [1.13, 1.52] |
| 80 | 2–10 | SciPy infeasible | SciPy infeasible | 1.48 – 6.00 |
| | 20 | 0.025 [0.022, 0.029] | 0.053 | 2.09 [1.80, 2.42] |
| | 50 | 0.35 [0.30, 0.41] | 0.34 | 0.97 [0.83, 1.14] |

* **A3 (Phase 5B replicates): pass.** The shift beats SciPy LMS in every
  cell at ≤ 20 ms where SciPy could run. SciPy LMS could not fit 8
  replicates of 16 points in 6 of the 15 cells; the other two arms ran
  everywhere.
* **A4 (own LMS at least as efficient as the shift): refuted.** It is
  1.91× worse on geometric average, significantly worse in 12 of 15 cells
  and never significantly better.
  * Cause: set-up per replicate 0.10–0.16 ms against 0.01–0.06 ms, and at
    m = 16 a 28% higher cost per point.
  * Because N is a power of two, a small extra cost often halves N (e.g.
    m = 40 at 5 ms: N = 128 against 256), which quadruples the variance.
* **Decomposition of the Phase-5 gap.** SciPy LMS → own LMS removes most of
  it (ratios 0.015–0.75 at ≤ 20 ms); own LMS → shift removes the rest
  (about 1.9×). Since the shift lost nothing per point (A2), it is the best
  of the three here.
* **A5 (validity): coverage passes, deadline criterion fails.**
  * 95% coverage is 0.945–0.955 for every arm and Greek.
  * In some contract cells at 2 ms, 6–14% of RQMC runs exceeded 1.10× the
    budget (worst: own LMS, m = 80, 7 of 50 runs). At ≥ 5 ms the worst cell
    is 10%. Phase 5B's "at most 4.4%" does not carry over to 2 ms budgets on
    this grid.

### 2.3 What 6A establishes

The Phase-5 recommendation (shift, not SciPy scrambling) holds on new
contracts. The reason is now known: SciPy rebuilds a scrambled engine per
replicate, which costs 0.3–1.8 ms. A cheaper scrambling does not beat the
shift, because on these integrands scrambling buys nothing per point.

That last statement is integrand-specific. The unit test with a smooth
5-dimensional product function found the shift clearly worse
(`tests/test_budget_policy.py`). For smoother, lower-dimensional integrands
or much larger N, scrambling's rate advantage should reappear.

## 3. Phase 6B: a training-free joint rotation

Commands: `python scripts/phase6_tf_dev.py` (development, already-used
contracts) and `python scripts/phase6_training_free_study.py` (holdout).
Output: `results/phase6_training_free/{dev,holdout}/`.

### 3.1 Development stage (Phase-5C contracts, not evidence)

* Every deterministic first direction was almost exactly the trained one
  (|cos| ≥ 0.9999), but first-order rotations were 1.5–1.8× worse than a
  well-trained rotation. **The ordering of the remaining directions
  matters**, even though they carry only about 0.1% of the gradient energy.
* A second-order (Taylor) expansion of the gradient fixed part of this
  (1.12–1.33×).
* Adding the Greek weights h_g′(δ)² by Gauss–Hermite quadrature, expanded at
  the design point, fixed the rest (0.99× and 1.06×).

That candidate, "weighted-dominant", was frozen as primary before the
holdout.

### 3.2 Holdout: fixed N

Loss relative to a rotation trained on 32,768 gradient paths ("oracle"),
pooled over N = 2⁸, 2¹⁰, 2¹² (geometric mean of 8 contracts, 64 SciPy-LMS
replicates; 95% bootstrap interval):

| rotation | m = 16 | m = 40 | m = 80 | set-up (median, ms) m = 16 / 40 / 80 |
|---|---|---|---|---|
| path PCA | 4.81 | 7.41 | 11.70 | — |
| "ones" (first-order) | 1.46 [1.30, 1.63] | 1.62 [1.48, 1.77] | 1.76 [1.62, 1.91] | 0.03 / 0.03 / 0.03 |
| Taylor at the design point | 1.12 [1.00, 1.26] | 0.98 | 1.11 [1.02, 1.21] | 0.25 / 0.66 / 2.18 |
| weighted at the origin | 1.10 | 0.97 | 0.98 | 1.00 / 1.39 / 2.99 |
| **weighted at the design point (primary)** | **1.07 [0.96, 1.19]** | **0.98 [0.90, 1.07]** | **1.01 [0.93, 1.11]** | 1.17 / 1.53 / 3.19 |
| trained, 256 paths | 1.08 [1.00, 1.20] | 0.97 [0.91, 1.05] | 1.08 [1.01, 1.17] | 0.34 / 1.06 / 3.34 |

* **B1 (training can be eliminated): pass.** The primary rotation is within
  7% of the oracle at every m, with interval upper bounds ≤ 1.19.
* **B2 (better than 256-path training): not supported.** Ratios are 0.98,
  1.01 and 0.94, with intervals [0.89, 1.06], [0.93, 1.08] and [0.86, 1.00].
  The development hint (1.37× at m = 104) did not replicate at m = 80, where
  256-path training was within 8% of the oracle.
* **B4 (leading direction is not enough): pass.** "ones" is 1.46–1.76× the
  oracle (lower bounds 1.30–1.62), although its first column has
  |cos| ≥ 0.998 with the oracle's and captures ≥ 99.7% of the top eigenvalue.
* **Benefit over path PCA.** Every second-order rotation, trained or not,
  has 0.08–0.23× the loss of path PCA, i.e. 4.5–12×.

### 3.3 Holdout: one-off valuation, set-up charged

Every construction or training happened inside the timed region, and the
schedule used whatever budget was left. Shift scheduler as in Phase 5;
40 replications; loss ratio of the primary rotation to 256-path training:

| | 5 ms | 10 ms | 20 ms | 50 ms |
|---|---|---|---|---|
| m = 16 | 1.74 [1.32, 2.15] | 1.04 [0.86, 1.24] | 0.84 [0.70, 0.98] | 1.02 [0.86, 1.23] |
| m = 40 | 1.01 [0.85, 1.20] | 0.99 [0.83, 1.19] | 1.16 [0.95, 1.39] | 1.01 [0.84, 1.23] |
| m = 80 | 0.015 (training did not fit) | 0.89 [0.72, 1.07] | 0.91 [0.77, 1.08] | 1.06 [0.89, 1.29] |

* **B3 (primary ≤ trained one-off in every cell): fails.** The two are
  indistinguishable in 9 of 12 cells.
  * **m = 16, 5 ms:** the deterministic rotation is worse. It costs 1.4 ms
    to build against 0.4 ms for training, and at m = 16 that difference is a
    sizable share of 5 ms.
  * **m = 80, 5 ms:** 256-path training costs about 3.5 ms. In 13 of 320
    runs it left no time for any estimate, and in 54 the scheduler fell back
    to MC. The deterministic rotation (2.7 ms) left an estimate in 319 of
    320 runs.
* Against path PCA, the primary rotation's one-off loss is 0.08–0.48×.
* Coverage 0.943–0.953 for every strategy.

### 3.4 Deviations and anomalies in the timed part of 6B (reported in full)

1. **The computer slept during one run of the 50 ms block.** That run took
   6,318 s instead of 50 ms; the block took 2 h 8 min instead of about
   4 min. The laptop was on battery in the "Silent" power plan, and the run
   was continuing in the background after my turn had ended.
   * The run's estimate is still valid (look-ahead stopping ended it after
     waking). Losses are not time-normalised in 6B, so only that run's wall
     time is wrong.
2. **I re-ran the 50 ms block** with fresh calibration and random streams,
   with the machine held awake (`results/phase6_training_free/holdout/timed_rerun.csv.gz`).
   * At first I attributed the slow set-up times and overruns in that block
     to the sleep. **The re-run reproduced them**, so that attribution was
     wrong and has been corrected in the re-run's metadata.
   * The primary analysis uses the original pre-registered data. The re-run
     agrees: 0.98, 1.11 and 0.97 at m = 16 / 40 / 80, all intervals
     containing 1.
3. **Set-up is slower right after large simulations.**
   * In the 50 ms block, rotation construction took about 3× its idle cost
     (e.g. 10.5 ms against 3.2 ms at m = 80, for training and deterministic
     construction alike).
   * In isolation, building the same rotation took 3.5 ms idle and 6.6 ms
     right after RQMC runs of 2,048 points, plausibly cache eviction or
     OpenBLAS thread state (`eigh`).
   * The cost model does not see this. In some contract cells of the 50 ms
     block, up to 90% of runs (85% in the re-run) exceeded 1.10× the budget,
     though medians stayed at 1.07–1.10.
   * **For a one-off valuation, set-up cost is not a constant.**
4. **Book-keeping.** A run with no estimate records wall time 0 instead of
   its set-up time. This affects reported mean times only, not losses. The
   frozen runner was left unchanged.

### 3.5 What 6B establishes

The question "is gradient-covariance training necessary?" now has a precise
answer for this contract family.

* **No.** A closed-form surrogate built from the gradient and Hessian of
  ln Ā at the design point, with the Greek weights integrated exactly in
  one dimension, reproduces the well-trained rotation.
* **It is not a shortcut.** It costs about as much as 256-path training,
  and 256-path training is already enough here.
* Its value is that it is deterministic (no training noise, reproducible,
  explainable from the payoff's structure) and that it shows *why* the
  trained rotation works. The leading direction is the gradient of the
  average along the most likely exercise path; the rest are ordered by the
  curvature of ln Ā, weighted by where each Greek's integrand lives.

## 4. Hypothesis scorecard

| | hypothesis | outcome |
|---|---|---|
| A1 | own LMS ≡ SciPy LMS | **pass** |
| A2 | scrambling beats a bare shift per point (N ≥ 2¹⁰) | **refuted** (mixed both ways; equal rates) |
| A3 | shift beats SciPy LMS per unit time ≤ 20 ms | **pass** |
| A4 | own LMS at least as efficient as the shift | **refuted** (1.9× worse) |
| A5 | coverage ≥ 0.93; ≤ 5% late runs per cell | coverage pass; **deadline fail** at 2 ms (max 14%) |
| A6 | second environment reproduces | **not run** |
| B1 | deterministic ≈ oracle-trained (≤ 1.10) | **pass** |
| B2 | deterministic beats 256-path training | **not supported** (indistinguishable) |
| B3 | deterministic ≤ trained one-off in every cell | **fail** (indistinguishable in 9/12; worse 1, better 2) |
| B4 | first direction alone is not enough | **pass** |

## 5. Limitations

* **One machine** for all timings (A6 not run). The equal-N and fixed-N
  results are machine-independent.
* **One contract family** (CPW-first Asian digitals under GBM). The
  deterministic rotation uses this payoff's structure (one scalar δ) and
  does not transfer as written to the basket or to other payoffs.
* **The primary candidate was chosen** among seven on development data. The
  holdout confirms it but did not test the selection procedure itself.
* **The deterministic construction is not optimised.** Only the top few
  eigenvectors matter, so a cheaper partial eigensolver could cut its cost.
  Not tested.
* **Owen (2003)'s variance equivalence of linear scrambling** is cited from
  secondary knowledge; no new literature search was made for Phase 6, and no
  method here is claimed as new.
* **Deviation in 6B's timed part** (§3.4); the conclusion is the same with
  or without the re-run.

## 6. Reproducing

```bash
python scripts/phase6_shift_study.py
```

```bash
python scripts/phase6_tf_dev.py
```

```bash
python scripts/phase6_training_free_study.py
```

```bash
python scripts/phase6_training_free_study.py --rerun-timed-budgets 0.05
```

All take `--analysis-only`. The first takes `--config portable` (the
second-environment configuration); set `PHASE6_ENV_TAG` to label the output.

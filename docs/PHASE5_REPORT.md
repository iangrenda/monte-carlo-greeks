# Phase 5 report: budget-feasible joint Monte Carlo Greek estimation

## 1. The question and the short answer

**Question.** Can a computational method choose between ordinary Monte Carlo
(MC), path-PCA randomised QMC and joint Greek-aware PCA while respecting a
real computing budget, accounting for set-up costs, and staying accurate over
repeated valuations?

**Answer, for the contract family studied.** Yes, and the choice turned out
simpler than the question assumed:

1. Use randomised QMC with **random digital shifts**, not per-replicate
   scrambling. Then RQMC beat MC at every budget tested (2–150 ms), so the
   MC-versus-RQMC decision that Phases 3–5A tried to model largely disappears.
2. **Plan the schedule from a cost model and stop by look-ahead.** Budgets are
   then respected (at most 4.4% of runs late in any cell).
3. **Estimate Delta, Gamma and Vega together** on shared paths with **one
   joint gradient-PCA rotation**. Its training cost was repaid within a single
   valuation in every cell tested, and it transferred unchanged to
   neighbouring contracts.

A fixed method won. That is the finding. The cost-based selection rule of
Phase 4, tested harder, proved fragile exactly where it matters.

What is established and what is new: every method used here is published
(conditional smoothing, PCA constructions, gradient PCA, digital shifts). The
contribution is the controlled, pre-registered measurement of how they
combine under wall-clock budgets, set-up charges and reuse.

## 2. Starting point and environment

| | |
|---|---|
| starting commit | `main` at `c4e5fda6f5d231f2b45892fc3d8b9e16fbe3c5b4` |
| branch | `phase5/budget-feasible-joint-greeks` |
| Phase 4 revalidation | 271 tests passed locally; CI on `main` green; the saved Phase-4 verdict (max regret 1.019, 450/450) recomputed exactly from `runs_raw.csv.gz` |
| machine | Windows 11, Intel64 Family 6 Model 197, 16 logical CPUs |
| software | Python 3.12.10, NumPy 2.4.3, SciPy 1.16.2, pandas 2.3.3, Matplotlib 3.10.7 |
| tests now | 325 pass (54 new) |

Every `meta.json` records the commit, the config's last commit, versions and
hardware. No Phase 1–4 result file was modified.

## 3. Pre-registration and data separation

| part | config (frozen in commit) | contracts | used for |
|---|---|---|---|
| 5A frontier | `phase5_frontier.json` (`8b05116`) | 24, grid A | testing the unmodified Phase-4 rule |
| 5B schedules | `phase5_sampling_policy.json` (`9bbcfc2`) | 12, grid B | comparing schedules; choosing one |
| 5C/5D joint, reuse | `phase5_joint.json` (`578048d`) | 8, grid C (+34 transfer targets) | testing the chosen schedule and joint rotations |
| 5E basket | `phase5_basket.json` (`9bbcfc2`) | 12, two-asset | generalisation |

* The grids share no strike, volatility, maturity or rate value with each
  other or with Phases 2–4.
* Each config was committed before its study ran. Only `--quick` pipeline
  checks ran earlier; their output was deleted.
* Choices made on one grid were applied only to a later grid: the scheduler
  for 5C/5D (from 5B), the MC block cap (from 5B), α = 2 (from Phase 4).
* Training, evaluation and reference streams use disjoint seeds.
* Script fixes after a freeze (one plotting bug, one dtype bug) are separate
  commits and changed no protocol.

## 4. The ideas, briefly

Full derivations are in [`PHASE5_MATHEMATICS.md`](PHASE5_MATHEMATICS.md).

1. **Greeks are derivatives of expectations.** We need a random variable
   whose mean is the derivative.
2. **Likelihood ratio = Malliavin weight** for terminal payoffs: move the
   derivative onto the density, giving payoff × score.
3. **Conditioning.** Integrating out the first Brownian step turns the
   digital's jump into the smooth Φ(δ). By Rao–Blackwell this never raises
   variance, and it makes pathwise differentiation legal.
4. **Brownian covariance and PCA.** The path is Lξ with LLᵀ = C. Every such
   L is unbiased; they differ in which input coordinate matters most, which is
   what QMC cares about.
5. **Gradient PCA.** Rotate the inputs by the eigenvectors of E[∇f∇fᵀ]. Any
   orthogonal rotation leaves the distribution unchanged.
6. **Joint estimation.** The three Greeks share the expensive path work and
   one rotation. Their errors form a vector with a covariance matrix.
7. **Scheduling as constrained optimisation.** Minimise predicted variance
   over (R, N) subject to predicted cost ≤ budget; infeasible means say so.
8. **Set-up versus marginal cost.** Training pays when
   loss_trained(B − C_train/M) < loss_plain(B).

## 5. Findings

### 5A. The Phase-4 rule near its own threshold — fragile

[Details](PHASE5_COST_FRONTIER.md).

* 72 cells on a dense budget grid, two machine configurations. In each,
  **6 cells are materially wrong**: the rule picks RQMC while MC is 1.4–2.8×
  better in RMSE. All lie just above the threshold at m = 52 and 104.
* The threshold is early by up to 1.3× (m = 52) and 1.65× (m = 104). Gamma
  crosses 12–32% later than Delta and Vega.
* **Calibration noise.** The threshold moves 7–14% between repeat
  calibrations, and the machine sped up about 30% within one session. The
  calibration used was favourable; a typical one would have flipped 15 of 72
  decisions with a worst cell near 8× in RMSE.
* Away from the gap the rule is right (61 and 59 of 72 cells clearly correct)
  and beats both fixed samplers on average.
* Cause: at the threshold about 84% of RQMC's time is scrambler set-up.

### 5B. Feasible schedules — deadlines fixed; the randomisation matters more

[Details](PHASE5_SAMPLING_POLICY.md).

* Look-ahead schedules: at most 4.4% of runs late in any cell, largest
  overrun 1.39×. Old RQMC scheduler: 100% late at 2 ms, up to 7.3×.
* Coverage of 95% intervals: 0.948–0.952 for all nine strategies.
* **Negative:** the optimised schedule with Owen scrambling is not more
  accurate than the old one (1.06 [0.99, 1.15]).
* **Digital shifts:** per-replicate set-up falls from 0.6–2.1 ms to under
  0.03 ms; time-normalised MSE 0.09× that of Owen scrambling; better than MC
  in all 15 cells.
* **Negative:** my new MC schedule is 1.15× less efficient than the old one
  (blocks too large). Corrected by a block cap for later phases.
* The "scramble-size cliff" comparison was inconclusive.

### 5C. Joint estimation — sharing is worth 3× to 37×

[Details](PHASE5_JOINT_GREEKS.md).

* Shared against three separate simulations, equal total time: loss ratio
  0.28–0.42 (MC), 0.06–0.20 (path PCA), 0.03–0.38 (gradient PCA).
* Gradient covariances are rank one (top share ≥ 0.968) and aligned across
  Greeks (|cos| ≥ 0.9987). The joint rotation is as good as any
  Greek-specific or re-weighted one (0.89–1.12).
* Amortised, joint gradient PCA has 3–7× lower loss than path PCA at m = 24
  and 7–30× at m = 104, under measured wall-clock budgets.
* Gamma and Vega errors are strongly negatively correlated (−0.3 to −0.8)
  after the rotation; the estimator returns the full covariance.

### 5D. Reuse — training pays at once and transfers

* **Break-even M\* = 1** for the joint rotation in all four (m, budget)
  cells, e.g. loss 0.18 [0.15, 0.22] of path PCA's at m = 104 and 10 ms with
  the 4.4 ms training charged in full.
* Three specific rotations with three simulations often never break even.
* Transfer: on all 34 neighbouring contracts the transferred rotation is
  significantly better than path PCA (0.014–0.47) and indistinguishable from
  retraining in 30; no trend with distance.
* This differs from Phase 4's "one-off can be 16× worse". Fewer training
  paths, shared paths and digital shifts all changed; the experiment does
  not separate their contributions.

### 5E. Generalisation — holds with a smaller gain

[Details](PHASE5_GENERALISATION.md).

* Two-asset basket Asian digital, new conditional estimator with
  root-finding, validated against per-path finite differences and the exact
  Black–Scholes limit.
* Joint rotation over six Greeks: 2.5–5.6× lower loss than path PCA.
* Delta and Gamma stay rank one (I predicted otherwise); Vega does not.
* The strict per-Greek criterion for the joint rotation failed in 6 of 36
  cells; the average cost is 7%.
* Fixed N only. Finite-difference training costs 10–38 ms.

## 6. Hypothesis scorecard

| | hypothesis | outcome |
|---|---|---|
| H5A1 | rule is early at m ≥ 52 | confirmed |
| H5A2 | those errors are negligible | **refuted** |
| S1 | look-ahead meets deadlines | supported |
| S2 | coverage ≥ 0.93 | supported |
| S3 | optimised schedule more accurate | **not supported** |
| S4 | shifts feasible at smaller budgets | supported; also far more accurate per unit time |
| S5 | smaller cliff | inconclusive |
| J1 | sharing beats separate | supported |
| J2 | joint rotation robust to weights | supported |
| J3 | joint rotation beats path PCA, amortised | supported |
| J4 | finite break-even, larger at small budget / large m | break-even is 1 everywhere; the predicted growth **did not appear** |
| J5 | transfer works, degrades with distance | works; degradation **not observed** |
| E1 | basket G rank one | Delta, Gamma yes; **Vega no** |
| E2 | same-asset directions aligned | yes, and cross-asset too |
| E3 | joint within 1.25 of specific for each Greek | **failed** in 6 of 36 cells (mean 1.07) |
| E4 | still better than path PCA | yes, by less |

## 7. A method-selection framework, as far as the evidence goes

For CPW-first Greeks of Asian digitals on this kind of hardware:

1. Calibrate a cost model once (≈ 5 s per (m, evaluator)); treat it as
   approximate.
2. Use shifted-Sobol' RQMC with the path generated through a stored joint
   rotation for that m. Train the rotation if none is stored: the cost is
   repaid in one valuation.
3. Choose (R, N) by the constrained optimisation; R ≥ 8 if a covariance
   matrix is wanted.
4. Stop by look-ahead; if no schedule fits, fall back to MC; if that does not
   fit, report infeasible.
5. Compute all requested Greeks on the shared paths and report the error
   covariance.

Step 2 replaces the Phase-4 selection rule. On the 5B grid a rule of the old
form with the shift cost model chose MC in one cell (m = 104, 2 ms) where
RQMC was 1.3× better, so even there the rule adds nothing.

## 8. Limitations

* **One machine.** Every timing result is specific to it and to SciPy's
  Sobol' code. The 5A "second configuration" is the same machine under load.
  A faster Owen scrambler would narrow the gap to digital shifts.
* **One model and one payoff family**: GBM; digitals on arithmetic averages;
  conditioning leaves one scalar. 5E widens this only slightly.
* **The digital-shift result rests on one grid** (12 contracts, 5 budgets)
  plus its use in 5C/5D. It has not had its own confirmatory holdout in the
  sense 5A gave the Phase-4 rule.
* **Budgets of 2–150 ms.** Below 2 ms nothing was tested.
* **Small scenario counts** (4 per cell in 5B/5C/5D, 2 in 5E): geometric
  means over scenarios are stable, single-scenario ratios are noisy.
* **Three changes at once** between Phase 4's and Phase 5's one-off result.
* **Transfer at fixed N only**, same m only.
* **Basket:** fixed N, finite-difference training, no timed study.
* **Cost model** fit error 6–25%; machine drift of 30% observed.
* No new literature search was made in Phase 5; novelty is not claimed for
  any method.

## 9. What a broader publication claim would need

1. Independent hardware (at least two machines, ideally a server CPU) and a
   second Sobol'/scrambling implementation.
2. A pre-registered holdout for the shift-based pipeline as a whole,
   including budgets below 2 ms where a crossover with MC must exist.
3. An ablation separating training size, shared paths and randomisation in
   the break-even result.
4. Contracts without a single conditioning scalar: barriers, worst-of
   baskets, stochastic volatility (where the conditional step needs a
   different construction).
5. Analytic gradients and a timed study for the basket.
6. A test of training-free rotations (for example Imai & Tan's linear
   transformation), since the dominant direction looks contract-independent.
7. A targeted literature search for joint-Greek rotations and for
   budget-constrained RQMC scheduling, before any claim of novelty.

## 10. Reproducing

```bash
python scripts/phase5_frontier_study.py
```

```bash
python scripts/phase5_frontier_study.py --config contended
```

```bash
python scripts/phase5_policy_study.py
```

```bash
python scripts/phase5_joint_study.py
```

```bash
python scripts/phase5_basket_study.py
```

Each takes `--quick` (pipeline check, not evidence) and `--analysis-only`.
Timed results will differ on other machines; fixed-N results (5C §3.2,
5D transfer, 5E) are reproducible from the seeds on the same software stack.

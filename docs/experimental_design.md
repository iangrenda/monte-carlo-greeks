# Experimental design

All numbers in `results/` come from `scripts/full_experiment.py --config research`
unless a file name says otherwise. The `quick` configuration runs the same
code with smaller N and fewer replications, for demonstration only.

## Baseline

Black–Scholes with `S0 = 100, r = 0.10, σ = 0.20, T = 1` (FLLLT §4). The payoffs are:

* `interval`: `1{100 ≤ S_T ≤ 110}` (the paper's example),
* `digital`: `1{S_T > K}`, K = 100 unless swept,
* `call`: `(S_T − K)⁺`, K = 100 unless swept.

## Estimators

| code name | description | evals/path |
|---|---|---|
| `Malliavin` | global weight π (FLLLT §4) | 1 |
| `Malliavin-loc` | localised, δ = 10 (FLLLT pp. 409–410); call only | 1 |
| `Pathwise` | IPA; call Δ and 𝒱 only | 1 |
| `FD-CRN-ctr`, `FD-CRN-fwd` | central / forward differences, common random numbers | 2 (Δ, 𝒱), 3 (Γ) |
| `FD-IND-ctr`, `FD-IND-fwd` | same, independent random numbers per bumped price | 2, 3 |

Combinations that are mathematically invalid (pathwise on digitals, pathwise
Gamma, localisation on digitals) are **not run**. They are listed in
`results/tables/inapplicable_estimators.csv`.

## Random numbers and replications

* Each experiment has a master seed (`config.SEEDS`). Sub-runs use
  `SeedSequence([master, i, j, ...])`, and replications are
  `SeedSequence.spawn` children of a sub-run. The PCG64 generator is used throughout.
* **One replication = one base vector Z of N standard normals.** Every
  estimator in the replication uses this same Z, which makes comparisons
  between methods **paired** (see `B_paired_vs_malliavin.csv`). CRN finite
  differences use Z for every bumped price. IND finite differences use Z for
  the first bumped price and fresh draws from the replication's stream for
  the others, so their bumped prices are mutually independent.
* Replications are independent of each other.
* **Tuning vs evaluation.** FD bumps are tuned in Experiment C (seed
  `C_bump`) by minimising the empirical RMSE over the bump grid. Experiments
  B and D–G use these tuned bumps with **different seeds**, so the bump
  choice is never confirmed on the data that selected it. The bumps are
  tuned at the baseline and N = 50 000 only, and expressed relative to
  `S0` (Δ, Γ) or as absolute vol (𝒱, capped at σ/2). Away from the
  baseline they are therefore not re-optimised; that is a deliberate
  feature of the design, mirroring how a practitioner fixes a bump rule.

## Metrics

For R replications `θ̂_1..R` and analytical `θ`: mean, bias (± SE `sd/√R`),
variance, MSE (± SE), RMSE (± delta-method SE), relative RMSE `RMSE/|θ|`,
mean reported standard error, and **coverage** of the nominal 95% interval
`θ̂ ± 1.96·SE_r`. Each row also carries the estimator's **exact** bias and
per-path variance (quadrature, `estimators.exact_estimator_moments`), from
which `rmse_theory = √(bias² + var₁/N)` and the ratio
`var_ratio_emp_over_exact` are derived. This ratio should scatter around 1,
within roughly ±20% for R = 200, unless the estimator is heavy-tailed.

**Cost.** We record both

* workload: payoff evaluations = N × (bumped prices per path), and
* wall-clock time per estimate: the pro-rated time to draw the base normals,
  plus the method's own extra draws, plus compute time, measured with
  `time.perf_counter` in a single thread.
* All estimators are evaluated in **blocks of 16 384 paths**
  (`estimators.blocked_mean_var`; block means and variances are merged
  with Chan's stable update). Without blocking, a single NumPy pass over
  150 000 paths was 3–4× slower *per path* than over 50 000 on the
  development machine, because the temporaries no longer fit in cache.
  That would have penalised methods that use more paths for the same
  budget (Malliavin uses N = B, central FD uses B/2). With blocking, runtime
  is proportional to work (verified: ×3.0 time for ×3 N).

The equal-budget comparison (G) gives every method the same number of payoff
evaluations B (N = B for Malliavin and pathwise, B/2 for FD Δ and 𝒱, B/3 for
FD Γ).

## Experiments

| | what varies | N | R | output |
|---|---|---|---|---|
| A | paper replication: N = 10 000 table, running estimates (Sobol and pseudo), RQMC vs MC for N = 2¹⁰…2¹⁷, localisation | 10 000 / 2^k | 100 | `A_*` |
| B | N ∈ {1k, 2.5k, 5k, 10k, 25k, 50k, 100k, 200k}, all methods, 3 payoffs | grid | 200 | `B_*` |
| C | FD bump: h/S0 ∈ {1e-4 … 0.3} (Δ, Γ), h_σ ∈ {1e-4 … 0.15} (𝒱), CRN vs IND, fwd vs ctr | 50 000 | 200 | `C_*`, `tuned_bumps.json` |
| D | σ ∈ {0.10, 0.15, 0.20, 0.30, 0.40, 0.60} | 50 000 | 200 | `D_*` |
| E | K/S0 ∈ {0.80 … 1.20}, cash digital and call | 50 000 | 200 | `E_*` |
| F | T ∈ {0.05, 0.10, 0.25, 0.5, 1, 2} | 50 000 | 200 | `F_*` |
| G | payoff type at an equal budget of 150 000 payoff evaluations | budget | 200 | `G_*` |

## Statistical caveats

* Convergence slopes are OLS fits of log RMSE on log N over the grid. Their
  confidence intervals ignore the sampling error of each RMSE point and
  should be read as descriptive.
* Paired MSE differences come with a t-interval for a **single**
  comparison. We run many comparisons and apply no multiplicity correction,
  so we describe differences by effect size (MSE ratios), not by
  "significance".
* RQMC error bars come from independent scramblings. The single-sequence
  running estimates (A) are illustrative trajectories, like the paper's
  figures, and are not error estimates.
* Timings depend on hardware and on NumPy's vectorised kernels. They are
  reported for one machine (see each `*_meta.json`). Ratios between methods
  are more portable than absolute times.

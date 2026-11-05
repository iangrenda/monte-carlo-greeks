# Project audit: `monte-carlo-greeks` (phase 1 state, audited 2026-11-04)

**Scope.** The repository at `C:\Users\iamfl\OneDrive\Desktop\greeks\monte-carlo-greeks`
(the desktop folder is named `greeks`). The audit covers everything built
in phase 1 (European options): code, tests, results and documents.

**Version control.** There is **no git repository**, so there is no
history to inspect, and the change history in §G is reconstructed from the
session that produced the code. Recommendation 5 addresses this.

**Execution environment.** Python 3.12.10 in a venv at
`C:\Users\iamfl\.venvs\mc-greeks`, with numpy 2.4.3, scipy 1.16.2,
pandas 2.3.3 and matplotlib 3.10.7. The venv lives outside the repository:
Windows Application Control blocked newer wheels, and the original
temporary path exceeded MAX_PATH. Hardware: Intel x86-64, 16 logical cores;
all runs single-threaded.

---

## A. Existing capabilities (verified by running, not just by reading)

| Kind | Items |
|---|---|
| Model | Black–Scholes GBM, exact terminal sampling (`simulation.py`) |
| Payoffs | European call, cash digital, interval digital (`payoffs.py`) |
| Benchmarks | Closed-form price, Δ, Γ, 𝒱 for all three payoffs (`analytical.py`) |
| Estimators | Global Malliavin weights (= LR here); localised Malliavin (call); pathwise (call Δ, 𝒱); FD central/forward × CRN/IND (`malliavin.py`, `pathwise.py`, `finite_difference.py`) |
| Samplers | Pseudorandom (PCG64, SeedSequence trees); scrambled Sobol (1-D) |
| Exact theory | Exact bias and per-path variance of every estimator by Gaussian quadrature (`estimators.exact_estimator_moments`) |
| Experiments | A paper replication (+ RQMC, localisation); B convergence; C bump size; D σ; E K; F T; G equal budget (`experiments.py`) |
| Statistics | Bias ± SE, variance, MSE ± SE, RMSE, coverage, log-log slopes, paired MSE differences (`statistics.py`) |
| Figures | 21 figures (PNG + PDF) in `results/figures` |
| Tests | 142 in phase 1; 175 after phase 2; all passing |
| Documents | derivations, experimental design, research report, README, 4 executed notebooks |

## B. Mathematical correctness

| Check | Result | Evidence |
|---|---|---|
| Discount factor exp(−rT) in every estimator | correct | `test_discounting_consistent` (r = 0.3) |
| Δ/Γ/𝒱 scaling; 𝒱 per unit σ | correct | `test_vega_is_per_unit_vol`; `𝒱 = x²σTΓ` identity test |
| FD signs and bump definitions | correct | `test_stencil_shapes`; exact bias ratios 4:1 (central), 2:1 (forward) |
| Malliavin weights (FLLLT §4) | correct | quadrature `E[φπ]` = closed form to 1e-7, 30 payoff×parameter×Greek cases |
| Gaussian simulation | correct | mean/variance/martingale tests; exact log-map |
| Variance and CI formulas | correct | coverage of an unbiased Gaussian = 0.95 ± 0.01 |
| Discontinuities | handled correctly | pathwise refused on digitals and call Γ (`InapplicableEstimatorError`) |
| CRN implementation | correct | per-path identity for a linear payoff; manual recomputation |
| Benchmarks for the paper example | correct to 10 digits | see §B.1 |

### B.1 Independent verification of the paper benchmarks (re-run for this audit)

Two different routes agree to all printed digits:

* **Closed form** (`analytical.greeks`): price 0.1682376255, Δ −0.0013348588,
  Γ −0.0003887569, 𝒱 −0.7775137088.
* **Density integration.** `e^{−rT}∫_{100}^{110}` of the lognormal density
  (SciPy quad, relative tolerance 1e-13), with Greeks from
  Richardson-extrapolated differences of that integral. Same four values to
  10 digits.

The brief's targets (0.1682376255, −0.0013348588, −0.0003887569, −0.777513709)
match.

### B.2 Statistical consistency of the MC estimators (re-run)

Command: an inline script, 20 independent runs of N = 200 000 per Greek
(seed 12345). It took 0.9 s wall. Results for z = (estimate − exact)/SE:

| | mean z | sd z | max abs z |
|---|---|---|---|
| price | −0.39 | 1.05 | 2.80 |
| Δ | −0.21 | 1.04 | 2.20 |
| Γ | 0.28 | 1.16 | 1.79 |
| 𝒱 | −0.09 | 0.96 | 1.95 |

These are consistent with N(0, 1): the estimators are unbiased and the
reported SEs are accurate. Gamma shows no negative, unrealistic or unstable
behaviour for the Malliavin estimator. Noisy and wrong-signed Gamma
estimates do occur, as expected, for FD with small bumps; that is
documented in Experiment A.

### B.3 Finite-difference variance laws (re-run)

Exact N·Var by quadrature:

* CRN digital Γ: h = 0.01 → 30 151, h = 0.03 → 1 117. Ratio 27 = 3³, as
  O(h⁻³) predicts.
* IND digital Γ: ratio 81 = 3⁴.
* CRN call Γ: ratio 3.0 = 3¹.

Three SciPy `IntegrationWarning`s appear at h ≤ 0.03. The values still
follow the theoretical scaling, so the warnings are benign here. However,
the phase-1 research runs used `python -W ignore`, which hides all
warnings. That is a weakness (§C).

### B.4 Malliavin = likelihood ratio for terminal European payoffs

`log S_T ~ N(log x + (r − σ²/2)T, σ²T)`. The score with respect to x is

```
∂_x log f(S_T; x) = (log S_T − log x − (r−σ²/2)T)/(σ²T x) = σW_T/(σ²T x) = W_T/(xσT),
```

which is exactly the FLLLT Delta weight. Likewise `f_xx/f` is the Gamma
weight, and `∂_σ log f` is the Vega weight (all three verified in
`mathematical_derivations.md` §2.5).

**Implications.**

1. Throughout phase 1, "Malliavin estimator" and "likelihood-ratio estimator
   (Broadie & Glasserman 1996)" are the *same numbers*.
2. Benhamou (2003) shows that the LR weight has the smallest total variance
   among admissible weights. There is therefore nothing to gain by
   "optimising the Malliavin weight" in this setting.
3. Phase 1's conclusion "Malliavin beats FD on digitals" is a restatement of
   the known LR-vs-FD comparison. It is a correct replication, not a new
   finding.
4. Malliavin calculus differs from LR only when the density is unavailable
   or when non-adapted directions are used. Phase 2 implements one such
   weight for the Asian digital (II.3).

The phase-1 docs state this (§2.5), but the README headline and figure
legends say "Malliavin weight" without the qualifier.

### B.5 Open discrepancies

The paper's "exact" call Vega (33.320063) differs from the closed form
(33.322460) by 7e-5 relative. It is reported, not explained. There is no
evidence that our value is wrong: it agrees with the closed form to 1e-7
and with independent quadrature.

## C. Computational correctness

* **Vectorisation and memory.** NumPy throughout. Estimators run in
  16 384-path blocks with Chan's stable merge (`blocked_mean_var`), so memory
  is bounded and wall-clock time is linear in N (verified ×3.0 for ×3 N).
  ✔
* **Reproducibility.** One master seed per experiment and SeedSequence
  children per replication. Re-running the full research pipeline gave
  **bit-identical** accuracy tables (0 change in every non-timing column). ✔
* **Weakness: order-dependent random streams.** The extra draws for IND
  finite differences come from the replication's generator in spec order.
  Adding or removing a method changes the IND numbers. They remain
  reproducible, but the coupling is fragile.
* **Weakness: timing noise not quantified.** Runtimes moved by up to 30%
  between identical reruns. Time ratios are reported as point values with
  no uncertainty, so claims like "1.5–2.0× slower" carry about ±15%
  machine noise.
* **Weakness: warnings suppressed.** `-W ignore` in the research runs hides
  quadrature warnings. They proved benign (B.3), but the practice should go.
* **Numerical stability.** Variance merges are stable (tested with a mean of
  1e6); Sobol points are clipped away from {0, 1}; quadrature is split at
  discontinuities. ✔

## D. Research quality

1. **Replication counts.** 200 replications per setting is adequate for
   RMSE, with relative SE ≈ 5–8%. Coverage SEs are ≈ 1.5%. ✔
2. **Convergence evidence.** Slopes are fitted with a *fixed* bump, which is
   documented. Asymptotic FD rates (e.g. N^{−2/5}) were derived but never
   tested with N-dependent bumps. ✗ minor.
3. **Uncertainty in figures.** RMSE SEs are in the tables but **not drawn**
   in Figures 4, 6, 8 and 9 (no error bars or bands). Small differences,
   such as pathwise vs CRN-FD on the call, look decisive in the figures but
   are within noise. ✗
4. **Fairness of comparisons.**
   * FD bumps were tuned on separate seeds ✔, but only at the baseline. They
     were then transferred unchanged to the σ/K/T sweeps, which handicaps FD
     at short maturities (coverage 0). This is documented, but it still
     drives several headline ratios. ✗
   * The equal-budget comparison does not credit Malliavin/LR for
     computing all Greeks from one simulation (conservative, documented).
5. **Overfitted hyperparameters.** The bump choice minimises a noisy RMSE
   curve. The empirical optimum matches the exact-theory optimum in 32 of
   36 cases, so the risk is low. ✔
6. **Analytical benchmarks.** Verified independently (B.1). ✔
7. **Significance language.** Paired-difference CIs come without
   multiplicity correction. The report avoids "significant" and uses effect
   sizes. ✔
8. **Most serious limitation: no practical case.** Every phase-1 payoff has
   a closed form, so the study shows *relative* estimator quality in a
   setting where Monte Carlo is unnecessary. In one dimension, the RQMC
   N^{−1} rate and the Malliavin/LR advantage are the textbook best case. This
   motivates phase 2 (path-dependent, no closed form).
9. **Novelty.** Phase 1 is a careful replication plus a standard
   comparative study. None of it is new relative to FLLLT (1999),
   Broadie & Glasserman (1996), Benhamou (2003) and Glasserman (2004).

## E. Component status

| Component | Status | Evidence | Issues | Recommended action |
|---|---|---|---|---|
| Closed-form benchmarks | complete | 10-digit independent check; 40+ tests | none | — |
| Exact terminal simulation | complete | moment and martingale tests | none | — |
| Malliavin/LR European weights | complete | quadrature to 1e-7; MC z-scores | named "Malliavin" without "= LR" in headline material | relabel (done in phase-2 docs) |
| Localised Malliavin (call) | complete | quadrature unbiasedness; 42× variance cut | δ = 10 untuned; call only | adaptive δ is a possible extension (see gaps) |
| Pathwise | complete | validity enforced; tests | — | — |
| FD CRN/IND | complete | stencil, bias-order and CRN tests | bump rule not rescaled with σ, T | scale bumps with S₀σ√T (done for phase 2) |
| Exact moments by quadrature | complete | empirical/exact variance ratio 0.67–1.28 over ~1 400 rows | warnings hidden | stop using `-W ignore` |
| RQMC | partially complete | 1-D only; slope ≈ −1 | trivial dimension | path-dependent RQMC (phase 2) |
| Experiments A–G | complete | reproducible, bit-identical | no error bars in figures; timing noise unquantified | add bands; repeat timing runs |
| Statistical framework | complete | coverage and statistics tests | — | — |
| Path-dependent payoffs | missing in phase 1 | — | — | **implemented in phase 2** (`asian.py`) |
| Variance reduction beyond CRN/localisation | missing | — | no control or antithetic variates | future work |
| Version control | missing | no `.git` | no history or provenance | `git init` and commit (not done without your go-ahead) |
| Notebooks | complete | executed via nbconvert | thin wrappers over results | — |

## F. Five highest-priority improvements

1. **Move to a problem without a closed form.** Path-dependent,
   discontinuous payoffs, where estimator choice matters in practice.
   *Done in phase 2.*
2. **State the Malliavin = LR equivalence everywhere it matters**, and test
   a genuinely non-LR Malliavin weight. *Done in phase 2: II.3, the
   anticipating Asian Delta weight.*
3. **Show uncertainty in every comparative figure**, and repeat timing
   measurements to put error bars on time ratios. *Partly done in phase 2:
   bootstrap CIs over scenarios on the new figures; phase-1 figures
   unchanged.*
4. **Use bump rules that scale with the problem** (S₀σ√T) instead of fixed
   percentages tuned at one point. *Done in phase 2; phase-1 sweeps not
   re-run.*
5. **Put the repository under git**, stop suppressing warnings, and add a
   one-command reproduction check (the phase-2 runner records config and
   environment in `meta.json`). *Git not initialised: it is your call.*

## G. What changed in this audit pass

* **Validation re-run.** Tests, benchmarks, MC z-scores and variance laws
  (§B). Commands are listed in `docs/RESEARCH_FINDINGS.md` §0.
* **No phase-1 code or results modified.** The phase-2 code is additive
  (`asian.py`, `asian_experiments.py`, `scripts/asian_study.py`,
  `tests/test_asian.py`), and phase-2 outputs go to `results/asian/<config>/`.

---

## H. Phase-3 audit addendum (2026-11-05)

| Finding | Severity | Evidence | Action |
|---|---|---|---|
| Phase-2 adaptive selector charged only the nominal pilot budget. The 16-path minimum made the calibrated pilot cost up to 2.5× nominal, and the measured wall time was 6–400× nominal (up to 33× the whole budget) because of per-call and scrambler overhead. | **high** (it changed conclusions) | `results/asian/research_revalidation/pilot_accounting.csv` | `accounting="charged"` is the default. Pilot wall time is recorded. The phase-2 adaptive-lite claims are retracted (`RESEARCH_FINDINGS.md` §5 errata, §8). |
| Phase-2 fixed runs missed their nominal budgets (measured/B: MC 0.5–2.0, RQMC 0.85–8.3) | medium | stored `fixed_raw.csv.gz` runtimes | Phase 3 uses measured wall-clock budgets. Phase-2 conclusions were revalidated time-normalised. |
| RQMC confidence intervals used z = 1.96 with about 10 scramblings | medium | coverage 0.904 (z) vs 0.933 (t) | t_{R−1} in phase 3 |
| Phase-3 first pass: per-construction two-point cost fits made PCA's scramble size 16 vs 2 048 for the others (noise) | high for the 5 ms comparison | `calibration_v1.csv`, `fixed_raw_v1.csv.gz` | Pooled least-squares cost model, plus a regression test. The calibration-dependent runs were re-run. |

Tests: 221 passing (`python -m pytest`), 46 of them new in phase 3.

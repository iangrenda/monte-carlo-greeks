# Phase 5E: does the joint rotation survive a two-asset basket?

**Summary.** Mostly yes, with a weaker gain. On a two-asset basket Asian
digital, one joint rotation over six Greeks still lowers the risk-vector loss
2.5–5.6× against path PCA (the one-asset contract gave 3–10×). Delta and Gamma
keep a rank-one gradient structure, which I had predicted would break. Vega
does not: its gradient covariance needs two directions. Training with
finite-difference gradients is expensive, so the reuse economics are worse.

This is a fixed-N study. It was not run under wall-clock budgets.

Code: `src/mc_greeks/basket.py`. Protocol: `configs/phase5_basket.json`,
committed as `9bbcfc2` before the run. Run:
`python scripts/phase5_basket_study.py` (2 min). Output:
`results/phase5_basket/study/`.

## 1. Contract and estimator

Payoff 1{w₁·mean_i S¹_{t_i} + w₂·mean_i S²_{t_i} ≥ K} under correlated GBM.
The derivation is in `PHASE5_MATHEMATICS.md` §9:

* integrate out the common first-step direction Z₀ = (Z¹+Z²)/√(2(1+ρ));
* the conditional probability is Φ(−z\*), where z\* solves
  a₁e^{s₁z} + a₂e^{s₂z} = K (Newton; no closed form when σ₁ ≠ σ₂);
* the Greeks follow from the implicit function theorem.

The one-asset CPW-first formula is **not** reused; it would be wrong here.
The method is conditional QMC smoothing as in Zhang & Wang (2020) and
Xiao & Wang (2018). No novelty is claimed, and no new literature search was
made for this phase beyond the Phase-4 review.

**Validation** (`tests/test_basket.py`, 12 tests):

* The residual covariance has rank 2m − 1 with the expected null vector, and
  matches simulation; the residual is uncorrelated with Z₀.
* The root satisfies A(z\*) = K to 11 digits.
* Each per-path Greek equals the finite-difference derivative of the
  per-path smoothed price (relative error below 5·10⁻⁶), for both assets.
* With m = 1 and w₂ = 0 the estimator's expectation, computed by exact
  Gauss–Hermite quadrature, equals the Black–Scholes digital price, Delta,
  Gamma and Vega to 8 digits.
* With w₂ = 0 it agrees in expectation with the one-asset CPW-first estimator.
* Swapping the assets swaps the Greeks.
* In the study itself: smoothed and raw digital prices agree (|z| ≤ 0.66 in
  all 12 scenarios); 1 296 RQMC means agree with independent plain MC
  (max |z| = 3.0).

## 2. Design

12 scenarios: m ∈ {12, 26} (dimension 23 or 51), ρ ∈ {0, 0.5, 0.9},
K ∈ {100, 106}, S = (100, 100), σ = (0.2, 0.4), w = (½, ½), T = 1, r = 0.02.
Six Greeks: Delta, Gamma and Vega of each asset. Gradients by central finite
differences of the smooth integrands; 256 training paths; 3 training
instances; 32 Owen scramblings each at N = 2¹⁰ and 2¹².

Rotations: path PCA; each Greek's own; joint over asset 1's three Greeks;
joint over all six.

## 3. Results against the pre-registered hypotheses

### E1. Rank one? — holds for Delta and Gamma, fails for Vega

| top-eigenvalue share of G | minimum | median |
|---|---|---|
| Delta (either asset) | 0.963 | 0.991 – 0.996 |
| Gamma (either asset) | 0.962 | 0.991 – 0.995 |
| Vega, asset 1 | **0.558** | 0.740 |
| Vega, asset 2 | 0.649 | 0.797 |

For Vega the top *two* directions carry at least 98% (participation ratio
1.3–1.9). I expected Delta and Gamma to need two directions as well, because
z\* depends on two path averages. They do not: ∇z\* is a weighted sum of the
two assets' average-gradients with weights ã_j/A′ that vary little from path
to path, so it is again close to one fixed vector. My prediction in the
mathematics note was wrong on this point.

### E2. Aligned directions? — yes, including across assets

* Same asset: |cos| ≥ 0.92 (medians 0.97–1.00).
* Different assets (Delta₁ vs Delta₂, and so on): medians 0.98–1.00; the
  lowest single value is 0.874 (Vega₁ vs Vega₂, ρ = 0).
* The shared direction is no longer path PCA's first direction when the
  assets are uncorrelated: |cos| ≈ 0.3–0.45 at ρ = 0, against 0.97–0.99 at
  ρ ≥ 0.5. At ρ = 0 path PCA has two equally important leading directions
  (one per asset) and no reason to pick the mixture the payoff needs.

### E3. One joint rotation for six Greeks — criterion not met, cost small

Variance of each Greek under the six-Greek rotation, relative to that Greek's
own rotation (geometric mean over K and N, 36 (m, ρ, Greek) cells):

* overall geometric mean **1.07**;
* range 0.67–1.48; **6 of 36 cells exceed the pre-set 1.25** (all Delta or
  Gamma; four of the six at m = 12);
* by Greek type: Delta 1.10, Gamma 1.19–1.21, Vega 0.91.

So the criterion "at most 1.25 for any single Greek" fails. The individual
ratios carry about ±30% noise (several are well below 1, which is impossible
in expectation by much), so the honest reading is: a joint rotation costs
roughly 10–20% in variance for Delta and Gamma, nothing for Vega, and some
cells may be worse. For Vega its own rotation is no better than the joint
one, consistent with a two-directional G that a single leading eigenvector
describes poorly.

### E4. Still better than path PCA? — yes, by less

Six-Greek loss, joint rotation / path PCA:

| | ρ = 0 | ρ = 0.5 | ρ = 0.9 |
|---|---|---|---|
| m = 12 (N = 2¹⁰ / 2¹²) | 0.40 / 0.40 | 0.25 / 0.26 | 0.37 / 0.40 |
| m = 26 | 0.27 / 0.22 | 0.16 / 0.18 | 0.20 / 0.23 |

That is 2.5–5.6×. The one-asset contract in Phase 5C gave 3–5× at m = 24 and
about 10× at m = 104. By Greek (specific rotation / PCA): Delta 0.16,
Gamma 0.09–0.21, Vega 0.43. Vega gains least, as in the one-asset case.

### E5. Reuse economics — model-based only

| | m = 12 | m = 26 |
|---|---|---|
| training: 256 paths, finite-difference gradients | 9.9 ms | 38 ms |
| cost per simulated point | 0.9 µs | 1.7 µs |
| break-even M at 10 ms per valuation (model) | 1.0 – 1.7 | 2.6 – 4.1 |
| break-even M at 50 ms per valuation (model) | 0.2 – 0.35 | 0.5 – 0.8 |

* The break-even is computed from the fixed-N loss ratio and variance ∝ N⁻².
  It is an estimate, not a timed experiment.
* Training costs as much as or more than a 10 ms valuation, so a one-off
  valuation at 10 ms cannot afford it (m = 26 not at all). Analytic gradients
  would cut the training cost by roughly the dimension (23–51×), which would
  restore the one-asset picture. They were not implemented.

## 4. Conclusion for the research question

1. Gradient covariances stay near rank one for Delta and Gamma, not for Vega.
2. Delta, Gamma and Vega of both assets share a dominant direction.
3. One joint rotation remains efficient: about 7% worse on average than
   Greek-specific rotations, though the strict per-Greek criterion failed.
4. The more complicated model weakens the advantage over path PCA
   (2.5–5.6× instead of 3–10×) but does not remove it.
5. Reuse is still worthwhile from 1–4 valuations, limited by the cost of
   finite-difference gradients.

## 5. Limitations

* Fixed N only; no wall-clock experiment, no digital shifts, no schedule.
* One volatility pair, equal weights, equal spots, two assets.
* No transfer experiment for the basket.
* 2 scenarios per (m, ρ); variance ratios are noisy.
* Still GBM, still a digital on an arithmetic average, still one
  conditioning scalar after smoothing. A contract with a genuinely
  two-dimensional discontinuity (for example a worst-of barrier) is the
  natural harder test.

Figure: `results/phase5_basket/study/figures/fig5E1_basket`.

# Monte Carlo Greeks: A Comparative Study of Finite Differences, Pathwise Differentiation, and Malliavin Calculus

*Computational study based on Fournié, Lasry, Lebuchoux, Lions & Touzi (1999).*

All numbers below come from `python scripts/full_experiment.py --config research`
(tables in `results/tables/`, figures in `results/figures/`). Machine:
Intel x86-64 (16 logical cores, single-threaded runs), Windows 11, Python 3.12.10,
NumPy 2.4.3, SciPy 1.16.2. The full run takes about 5 minutes.

---

## Abstract

We study three ways of estimating the Delta, Gamma and Vega of European
options by Monte Carlo in the Black–Scholes model: finite differences with
independent or common random numbers (IND/CRN), pathwise differentiation,
and the integration-by-parts ("Malliavin") weights of Fournié et al. (1999),
both global and localised. Each estimator is compared with closed-form
benchmarks over 200 independent replications. We also derive each
estimator's exact bias and variance by quadrature, which reproduces the
empirical variances to within sampling error in ≈1 400 configurations.

* We replicate the paper's Section 4 digital-option results. Its "exact"
  values agree with ours to the digits it reports, except small
  discrepancies in the Vegas (relative 3·10⁻⁶ for the digital, 7·10⁻⁵ for
  the call).
* For the paper's interval digital, the Malliavin estimator dominates. At
  an equal budget of 150 000 payoff evaluations its RMSE is **18–21×
  smaller** than the best finite difference for Delta and Gamma, and
  **4.5×** smaller for Vega, at ≈1.5–2.0× the wall-clock cost per estimate.
* The advantage is **not universal**:
  * For the smooth call, pathwise and CRN finite differences beat the
    *global* Malliavin weight by 2.0–3.5× in RMSE.
  * The paper's *localised* Malliavin estimator restores competitiveness
    for the call: best in 54 of the 57 call settings swept.
  * For the cash digital, CRN finite differences beat Malliavin for
    low-volatility Delta and Vega, deep-ITM Delta, and short-maturity Vega.
* CRN is essential for smooth payoffs. For discontinuous payoffs at
  RMSE-optimal bumps, CRN and IND give similar accuracy, because the
  optimal bumps are large.

---

## 1. Introduction

Greeks drive hedging and risk limits, so they are needed accurately and
often. When prices come from Monte Carlo, the obvious approach is
bump-and-reprice. It has a deterministic truncation bias that grows with
the bump `h`, and a Monte Carlo variance that grows as `h → 0`. The
problem is worst for payoffs with discontinuities: a digital's per-path
finite difference is zero except on a rare event, where it is huge.
Fournié et al. (1999) showed that Greeks can be written as
`E[payoff × weight]`, where the weight comes from Malliavin integration by
parts and does not depend on the payoff. The payoff is never
differentiated. This project tests empirically **when this matters and when
it does not**, rather than assuming the answer.

## 2. Mathematical background

Under the risk-neutral measure `dS = rS dt + σS dW`, the price is
`V = e^{−rT} E[φ(S_T)]` with the exact solution
`S_T = S_0 exp((r−σ²/2)T + σ√T Z)`. We sample this exactly, with no time
discretisation. The Greeks studied are `Δ = ∂V/∂S₀`, `Γ = ∂²V/∂S₀²` and
`𝒱 = ∂V/∂σ` (per unit σ). Three payoffs:

* interval digital `1{100 ≤ S_T ≤ 110}` (the paper's example),
* cash digital `1{S_T > K}`,
* European call `(S_T − K)⁺`.

All closed-form benchmarks are derived in `docs/mathematical_derivations.md` §3.

## 3. Methods

| method | formula (per path, times `e^{−rT}`) | valid for | bias |
|---|---|---|---|
| Malliavin (FLLLT §4) | `φ(S_T)·π`, with `π_Δ = W_T/(S₀σT)`, `π_𝒱 = W_T²/(σT) − W_T − 1/σ`, `π_Γ = π_𝒱/(S₀²σT)` | any bounded measurable φ | 0 |
| Localised Malliavin (FLLLT pp. 409–410) | smooth part of `(s−K)⁺` near K differentiated pathwise, compact remainder × π | call | 0 |
| Pathwise | `φ'(S_T) ∂S_T/∂θ` | Lipschitz φ, first-order Greeks (call Δ, 𝒱) | 0 |
| FD-CRN / FD-IND | central / forward stencils on bumped prices; same / independent normals | any | O(h²) / O(h) |

**Intuition for the weights.** Moving a derivative from `φ` onto the
Gaussian density is integration by parts: `E[g'(W)] = E[g(W) W/T]`. For a
digital, `φ'` is a Dirac mass that simulation never hits. After integration
by parts the same information is spread over every path through the
smooth weight `W_T/(S₀σT)`, which can be averaged at the usual `N^{-1/2}`
rate. In the one-dimensional European case these weights coincide with the
likelihood-ratio weights of Broadie & Glasserman (1996). Malliavin
calculus is what extends them to path-dependent payoffs and general
diffusions; we do not test those cases here (derivations §2.5).

**Where pathwise fails.** Pathwise digital Greeks would be identically 0,
and pathwise call Gamma would miss the whole boundary term. These cases are
marked *inapplicable* (`results/tables/inapplicable_estimators.csv`) and
never reported as numbers.

## 4. Experimental design

Full details are in `docs/experimental_design.md`. In short:

* 200 replications per setting (100 in the paper replication).
* In each replication, all estimators share the same base normals, which
  gives paired comparisons.
* FD bumps are tuned on separate seeds (Experiment C, N = 50 000) and then
  fixed, relative to S₀ or in vol units.
* Estimators run in cache-sized blocks so that wall-clock time is
  proportional to work.
* Cost is measured as payoff evaluations and as wall-clock time.
* Metrics: bias ± SE, variance, RMSE ± SE, relative RMSE, and 95%
  confidence-interval coverage. Each is compared with exact
  quadrature-based bias, variance and RMSE.

**Validation of the machinery.**

* 142 unit tests pass. They include quadrature checks that `E[φπ]` equals
  each closed-form Greek to 10⁻⁷, and a check that dropping the `−1/σ`
  term in the Vega weight is detected at >20 SE.
* Across all ≈1 400 summary rows, the ratio of empirical to exact
  estimator variance has median 1.00 and range 0.67–1.28. This is
  consistent with χ²₁₉₉ sampling error and the heavy tails of some FD
  estimators.

## 5. Results

### 5.1 Replication of FLLLT Section 4 (Experiment A)

**Benchmarks.** Our analytical interval-digital values are `V = 0.16823763`,
`Δ = −0.00133486`, `Γ = −0.00038876`, `𝒱 = −0.77751371`. The paper's
"exact" column gives `−0.001335, −0.000389, −0.777516`.

**N = 10 000 table.** Scrambled Sobol (one randomisation, then RMSE over 100
randomisations), `results/tables/A_paper_table.csv`:

| Greek | exact (ours) | paper MC-Malliavin | our Malliavin, Sobol (RMSE over 100 scramblings) | our Malliavin, pseudo-random (RMSE) | paper MC-FD | our CRN-FD, Sobol, h = 1 (RMSE) | our CRN-FD, Sobol, h = 0.1 (RMSE) |
|---|---|---|---|---|---|---|---|
| Δ digital | −0.00133486 | −0.001335 | −0.00133453 (1.6e-6) | (3.9e-5) | −0.003167 | (6.4e-5) | (6.5e-4) |
| Γ digital | −0.00038876 | −0.000389 | −0.00038869 (2.4e-7) | (8.0e-6) | **+0.099532** | (2.6e-4) | (2.3e-2) |
| 𝒱 digital | −0.777514 | −0.778695 | −0.777386 (4.7e-4) | (1.6e-2) | −0.542902 | (6.4e-3) | (4.7e-2; h_σ = 0.001) |

* **Malliavin estimates.** With low-discrepancy points our Malliavin
  estimates are at least as close to the truth as the paper's.
* **The paper's FD Gamma.** The paper's finite-difference Gamma of
  `+0.0995` (the wrong sign, 250× too large) is not reproduced with a 1%
  bump. It *is* typical of a 0.1% bump: there our single Sobol run gives
  `+0.036`, and individual pseudo-random runs have SD 0.078 around the
  truth. The paper does not state its bump, so this is a consistency
  check, not an exact replication.
* **Paper figures.** Figures `figA1–A3` reproduce the qualitative content
  of the paper's Figs. 1–3: the Malliavin running estimate settles within
  ~1% by N ≈ 2 000 with Sobol points. A single sequence is illustrative,
  not an error bar.

**Localisation (FLLLT Fig. 6).** For the call Gamma (K = 100, δ = 10),
localisation cuts the exact per-path variance from 0.0217 to 0.00052, a 42×
reduction (`A_exact_variances.csv`, `figA4`). It cuts the call-Vega
variance 14× (86 835 → 6 097).

**RQMC (extension).** With scrambled Sobol, the RMSE of the Malliavin
estimators falls like N^−0.99 to N^−1.00 (95% CIs about [−1.1, −0.88]),
against N^−0.49 to N^−0.50 for pseudo-random sampling (`A_rqmc_slopes.csv`,
`figA5`). At N = 2¹⁷ that is 160–300× lower RMSE. The integrand is
one-dimensional, the best case for QMC, so this should not be extrapolated
to path-dependent problems. Gamma and Vega have identical relative errors
because `π_Γ ∝ π_𝒱`.

### 5.2 Convergence with sample size (Experiment B)

Figure 4 (`fig04_rmse_vs_n`) shows relative RMSE against N for every payoff,
Greek and method. Dotted lines are exact theory and agree with the
simulations throughout. Fitted log-log slopes (`B_convergence_slopes.csv`):

| estimator | slope range (all payoffs/Greeks) | reading |
|---|---|---|
| Malliavin, localised, pathwise | −0.48 to −0.51 | unbiased: textbook N^−1/2 |
| CRN central FD, call Δ/𝒱 (h → 1e-4 S₀) | −0.48 to −0.50 | bias negligible: N^−1/2 |
| CRN/IND FD on digitals (fixed h) | −0.28 to −0.49 | bend as bias takes over, e.g. CRN-fwd digital Δ −0.29 |
| IND forward FD, call Γ | −0.17 | plateau at the O(h) bias |

With a fixed bump, finite differences do *not* have a convergence rate in
the asymptotic sense. They converge to `|bias|`. That is why FD confidence
intervals under-cover at large N (Figure `figB_bias_variance_*`). For
example, IND forward call Γ has coverage 0.19 at N = 50 000.

Relative RMSE at N = 50 000 (`B_convergence_summary.csv`):

| payoff | Greek | Malliavin | best FD | pathwise | localised |
|---|---|---|---|---|---|
| interval | Δ | **1.2%** | 18% (IND-ctr) | n/a | n/a |
| interval | Γ | **0.96%** | 10% (IND-ctr) | n/a | n/a |
| interval | 𝒱 | **0.96%** | 3.2% (CRN-ctr) | n/a | n/a |
| digital | Δ | **0.74%** | 1.1% (CRN-ctr) | n/a | n/a |
| digital | Γ | **1.8%** | 7.4% (CRN-fwd) | n/a | n/a |
| digital | 𝒱 | **1.8%** | 2.3% (CRN-ctr) | n/a | n/a |
| call | Δ | 0.95% | 0.36% (CRN) | 0.36% | **0.33%** |
| call | Γ | 4.1% | 0.83% (CRN-ctr) | n/a | **0.56%** |
| call | 𝒱 | 4.1% | 1.0% (CRN) | 1.0% | **0.99%** |

### 5.3 Bump size and common random numbers (Experiment C)

Figure 6 (`fig06_bump_size_rmse`) reproduces the bias–variance trade-off
almost exactly as the theory of derivations §4 predicts.

* **IND** RMSE explodes like 1/h (Δ, 𝒱) and 1/h² (Γ) as h → 0, for every payoff.
* **CRN, call Δ and 𝒱.** RMSE is flat as h → 0 (bounded variance): the
  optimal bump is the smallest tested (10⁻⁴ S₀), and CRN-FD becomes the
  pathwise estimator.
* **CRN, digitals.** RMSE still diverges as h → 0, more slowly (√(1/h) for
  Δ, h^{−3/2} for Γ). The RMSE-optimal bumps are large: 5% of S₀ for Δ and
  10% for Γ (`C_optimal_bumps.csv`).
* **CRN vs IND at those bumps.** The two give almost identical RMSE there,
  because two bumped indicators 10 apart are barely correlated. CRN helps
  digitals mainly by making small bumps *usable*, not by improving the
  optimum.
* **Tuning check.** The empirically optimal bump agrees with the
  theory-optimal bump on the grid in 32 of 36 cases. Two mismatches are
  adjacent grid points. The other two (CRN call Δ and 𝒱) lie where the
  RMSE curve is flat over 10⁻⁴–10⁻² S₀, so the choice is immaterial.

### 5.4 Volatility (Experiment D)

Figure 7 (`fig07_variance_vs_volatility`, variance with exact-theory lines):

* **Malliavin weights become noisy at low volatility, as predicted.** The
  variance of `π_Δ` scales like `1/σ²` and that of `π_Γ` like `1/σ⁴`. For
  the cash digital at σ = 0.10, CRN-FD beats global Malliavin for Δ (RMSE
  ratio 0.69) and 𝒱 (0.69). At σ ≥ 0.15, Malliavin wins Δ, and its lead
  grows with σ (FD/Malliavin = 2.4 at σ = 0.6).
* For Γ of the cash digital, Malliavin is best at every σ, partly because
  the fixed FD bump becomes very biased at low σ (coverage 0 at σ = 0.10).
* For the interval digital, Malliavin wins at every σ against the better
  of CRN and IND finite differences: by 9–90× (Δ, Γ) and 2.4–8× (𝒱).
* For the call, global Malliavin is never best. The localised estimator is
  best in 17 of 18 settings; CRN-FD and pathwise in the remaining one
  (𝒱, σ = 0.3, within noise).

### 5.5 Moneyness (Experiment E)

Figure 8 (`fig08_rmse_vs_moneyness`; absolute errors in `figE_abs_rmse_vs_moneyness`):

* **Cash digital.** Malliavin is best for Γ at every strike (FD 1.5–4.4×
  worse). For Δ, CRN-FD is marginally better deep in the money (K/S₀ = 0.8:
  0.99×; 0.9: 0.90×). For 𝒱, CRN-FD is better out of the money
  (K/S₀ = 1.1: 0.55×; 1.2: 0.87×). There the true Vega crosses zero
  (−0.22 at 1.1, +0.49 at 1.2), so relative errors are inflated for every
  method and the absolute RMSEs are the meaningful comparison.
* **Call.** Global Malliavin is 1.8–6.5× worse than pathwise or CRN-FD for
  Δ and 𝒱 at every strike. Localisation brings it back to the best or
  equal-best. For deep-ITM call Γ (K/S₀ = 0.8), global Malliavin's
  relative RMSE is 18% versus 1.0% localised. The polynomial weight
  multiplies a large payoff, exactly the case FLLLT designed localisation
  for.
* **Coverage.** CRN-FD coverage for call Γ falls to 0.005 at K/S₀ = 0.8 and
  to 0.23–0.58 at the other off-the-money strikes except K/S₀ = 0.95,
  because the bump tuned at K = S₀ is biased elsewhere.

### 5.6 Maturity (Experiment F)

Figure 9 (`fig09_rmse_vs_maturity`; variance in `figF_variance_vs_maturity`):

* **Relative error of Malliavin is stable in T.** The weights carry
  `1/T` factors, but the true digital Greeks grow as T → 0 too. Malliavin
  relative RMSE for digital Δ stays at 0.6–0.85% from T = 0.05 to 2.
* **Fixed bump rules fail at short maturity.** With bumps tuned at T = 1,
  every FD estimator of digital/interval Δ and Γ has coverage 0 for
  T ≤ 0.25, and RMSE up to 49× Malliavin's. This is mostly a
  *bump-rule* failure: a 5–10% bump is comparable to the T = 0.05
  standard deviation of S_T (≈4.5% of S₀). Re-tuning h per maturity would
  narrow the gap. Experiment C shows how much the optimum moves.
* **Exception: digital Vega at short maturity.** CRN-FD *beats* Malliavin
  (0.55× at T = 0.05, 0.65× at T = 0.1, 0.83–0.91× at T = 0.25–0.5). The
  vega weight `W²/(σT) − W − 1/σ` has an O(1/σ) variance that does not
  shrink with T, while the vol bump of 0.02 stays well scaled.

### 5.7 Smooth vs discontinuous payoffs at equal budget (Experiment G)

Every method gets 150 000 payoff evaluations (`G_equal_budget_summary.csv`,
Figure 10). RMSE ratios to the global Malliavin estimator (< 1 means
better than Malliavin):

| | interval Δ / Γ / 𝒱 | digital Δ / Γ / 𝒱 | call Δ / Γ / 𝒱 |
|---|---|---|---|
| FD-CRN central | 21 / 21 / 4.5 | 2.3 / 7.5 / 1.7 | 0.49 / 0.38 / 0.39 |
| FD-IND central | 21 / 18 / 9.1 | 3.4 / 10 / 3.9 | 2.1 / 1.5 / 1.2 |
| FD-CRN forward | 61 / 73 / 11 | 8.5 / 6.9 / 4.3 | 0.49 / 2.1 / 0.39 |
| Pathwise | n/a | n/a | 0.35 / n/a / 0.29 |
| Localised Malliavin | n/a | n/a | **0.32 / 0.16 / 0.29** |

Paired MSE differences (`G_paired_vs_malliavin.csv`) have 95% CIs that
exclude zero for every entry in this table.

**Wall-clock time per estimate**, relative to global Malliavin:

| estimator | relative time |
|---|---|
| CRN-FD | 0.51–0.67 |
| IND-FD | 0.80–0.91 |
| pathwise | 0.94–0.99 |
| localised Malliavin | 1.40–1.45 |

On efficiency, i.e. 1/(MSE × time), the global Malliavin estimator is the
most efficient for every digital Greek. Its margin over the next best is
1.8× (digital 𝒱), 3.7× (digital Δ), 25× (digital Γ) and 13–300×
(interval). On the call it is 8–33× *less* efficient than the best
estimator, which is pathwise for Δ and 𝒱 and localised Malliavin for Γ.

## 6. Discussion

**Answers to the research questions.**

1. *Digitals.* The Malliavin weight is the most accurate estimator for the
   interval digital in every configuration we ran, often by an order of
   magnitude. For the cash digital it is usually best, but not always
   (§5.4–5.6).
2. *Error decay.* Unbiased estimators show RMSE ∝ N^−1/2, with fitted
   slopes −0.48 to −0.51. Fixed-bump FD shows N^−1/2 only until it hits
   its bias floor. With RQMC the weights reach ≈N^−1 in this
   one-dimensional problem.
3. *Bump size.* Classic bias–variance U-curves. For digitals the optimum
   is large (5–10% of S₀), which makes FD estimators fragile when
   parameters move.
4. *CRN.* Decisive for smooth payoffs: it turns FD into the pathwise
   estimator, with over 1 000× lower RMSE than IND at h = 10⁻⁴ S₀. For
   discontinuous payoffs it matters mostly at small h, and at the optimum
   it gains little.
5. *σ, K, T.*
   * Low σ inflates the Malliavin weights. For the cash digital, FD can
     then win Δ and 𝒱.
   * Short T is the hardest regime for fixed-bump FD, while Malliavin's
     relative error stays stable.
   * Deep-ITM calls are where global weights do worst.
6. *Computational advantage.* Per estimate, the weight costs ≈1.5–2.0×
   a CRN-FD estimate at the same budget. One simulation also gives price,
   Δ, Γ and 𝒱 together; we time each Greek separately, so this sharing is
   not credited. The large accuracy gains on digitals dominate this cost.
7. *Where alternatives win.*
   * Pathwise and CRN-FD on smooth calls, by 2–3.5×.
   * CRN-FD on low-vol, deep-ITM and short-maturity cash-digital Δ/𝒱.
   * Localised Malliavin on calls (54 of 57 swept settings).
8. *Equal budgets.* Accounting for the extra payoff evaluations of FD
   strengthens the case for weights on discontinuous payoffs. On the call,
   pathwise and CRN-FD keep their lead even at equal budget.

**Contrary to the initial hypothesis.** "Malliavin is better for
discontinuous payoffs" holds strongly for the interval digital. For the
plain cash digital it holds only conditionally. That payoff has a single
jump and a large O(1) Delta, so a well-tuned CRN central difference with
a 5% bump is within a factor of 1–2.5 of Malliavin for Δ and 𝒱 in most
settings. The interval digital is harder for FD because its true Delta is
tiny (two opposite jumps nearly cancel). The FD estimator pays the full
noise of each jump but estimates a small difference.

## 7. Limitations

* **Model.** Black–Scholes with exact sampling and one-dimensional
  European payoffs. Here Malliavin = likelihood ratio, so we have not
  tested the setting (path dependence, unknown densities) that motivates
  Malliavin calculus.
* **Bumps.** FD bumps were tuned once (baseline, N = 50 000) and
  transferred as fixed rules. Re-tuning per configuration would improve FD
  in Experiments D–F, notably at short maturities. A practitioner rarely
  has this luxury, but the comparison is not "best-possible FD".
* **Localisation width.** The localisation width δ = 10 (the paper's
  value) was not tuned.
* **Rates.** Slopes are fitted over a finite grid with a fixed bump; they
  are not asymptotic rates.
* **Timing.** Timings are for one machine and NumPy implementation;
  ratios are more portable than absolute times.
* **Paper's FD settings.** The paper's FD bump sizes and its exact QMC
  generator are not stated. We used scrambled Sobol (SciPy), so our
  trajectories are not the paper's.

## 8. Future research

* Localisation for digitals: smooth the indicator near each jump, and
  choose the localising function optimally.
* Antithetic variates, and control variates using the analytical price.
* Path-dependent payoffs (Asian, barrier), where Malliavin weights differ
  from LR weights. FLLLT give Asian examples.
* Adaptive bump selection, e.g. pilot runs or the asymptotic h* ∝ N^(−1/5)
  for CRN digital Δ.
* Mixed pathwise–LR Gamma for the call.
* QMC with Brownian-bridge constructions in higher dimension.
* Stochastic volatility (Heston) and multi-asset models.
* Optimal allocation of a simulation budget across Greeks.

## 9. References

* Black, F., Scholes, M. (1973). The pricing of options and corporate liabilities. *J. Political Economy* 81, 637–654.
* Broadie, M., Glasserman, P. (1996). Estimating security price derivatives using simulation. *Management Science* 42, 269–285.
* Chan, T. F., Golub, G. H., LeVeque, R. J. (1983). Algorithms for computing the sample variance: analysis and recommendations. *American Statistician* 37, 242–247.
* Fournié, E., Lasry, J.-M., Lebuchoux, J., Lions, P.-L., Touzi, N. (1999). Applications of Malliavin calculus to Monte Carlo methods in finance. *Finance and Stochastics* 3, 391–412.
* Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering*. Springer.
* Glynn, P. W. (1989). Optimization of stochastic systems via simulation. *Proc. 1989 Winter Simulation Conference*, 90–105.
* L'Ecuyer, P., Perron, G. (1994). On the convergence rates of IPA and FDC derivative estimators. *Operations Research* 42, 643–656.
* Owen, A. B. (1997). Scrambled net variance for integrals of smooth functions. *Annals of Statistics* 25, 1541–1562.

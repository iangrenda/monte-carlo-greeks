# Budget-aware and Greek-aware Monte Carlo sensitivity estimation

*Technical report covering phases 1–4 of `monte-carlo-greeks`, with a focus
on phase 4.* It is written for a reader who knows probability, multivariate
normal distributions, conditional expectation, covariance matrices and
eigenvalues, but not stochastic calculus.

---

## 1. The problem: Greeks by simulation

An option's value is an expectation, `V(θ) = E[payoff(θ)]`. Here θ collects
the model inputs: today's price S₀, the volatility σ, and so on. Traders
hedge with the derivatives of V:

* **Delta** `Δ = ∂V/∂S₀`;
* **Gamma** `Γ = ∂²V/∂S₀²`;
* **Vega** `𝒱 = ∂V/∂σ`.

When V has no formula, it is estimated by averaging simulated payoffs.
Estimating its *derivatives* is harder. The naive method, bump-and-reprice,
estimates `[V(θ + h) − V(θ − h)]/(2h)` from two simulations. That has a bias
from h and a variance that explodes as h → 0. It is especially bad for
payoffs that jump, like a digital option paying 1 if the price ends above
K and 0 otherwise: a small bump changes almost no simulated payoffs, but
those it does change, it changes by 1/h.

## 2. The starting point: Fournié et al. (1999)

Fournié, Lasry, Lebuchoux, Lions and Touzi showed that a Greek can be
written as `E[payoff × π]`, where the **weight** π:

* comes from an integration-by-parts formula on the space of Brownian paths
  (Malliavin calculus);
* does not depend on the payoff;
* never requires differentiating the payoff.

In the one-dimensional Black–Scholes case that is just Gaussian integration
by parts, `E[g'(W)] = E[g(W)·W/T]`. Phase 1 showed that for terminal
European payoffs these weights *coincide* with the classical
likelihood-ratio weights, which are variance-optimal in their class
(Benhamou 2003).

## 3. What phases 1–3 established

**Phase 1: European options.** We replicated the paper's Section 4
(interval digital Greeks to 10 digits), the localisation trick, and the
benchmark table. The Malliavin = LR weights beat finite differences on
digitals, and lose to pathwise / CRN finite differences on smooth calls.

**Phase 2: a harder contract.** The arithmetic Asian digital pays
`e^{−rT}·1{average of m monitored prices ≥ K}` and has no formula.

* **Conditioning on the first increment (CPW-first).** Write the Brownian
  path at the dates as `W_{t_i} = √Δ·Z₁ + B_i`. The "level" variable Z₁
  multiplies every monitored price by the same factor, so
  `P(payoff = 1 | B)` has a closed form `Φ(·)`. Integrating Z₁ out
  analytically turns a discontinuous payoff into a smooth function of B.
* **This conditional estimator is the Rao–Blackwellisation** of the
  likelihood-ratio estimator for Δ and Γ, so it never has higher variance.
  We proved this and checked it to 1e-9.
* **Adaptive selection failed.** A pilot-based adaptive selector did not
  beat a well-chosen fixed estimator. Phase 3 later showed its pilot had
  been badly under-charged.

**Phase 3: randomised quasi-Monte Carlo (RQMC).**

* *Background.* Sobol points fill space more evenly than random points, so
  smooth integrands can converge at nearly N⁻¹ instead of N^{−1/2}.
* *But* this works only if the integrand depends mostly on the first few
  coordinates: low **effective dimension**.
* *How the path is built from those coordinates matters:*
  * sequential construction (one increment per coordinate);
  * Brownian bridge (endpoint first);
  * PCA (largest-variance direction first).
* *Results.* With CPW-first and PCA, the fixed-N convergence rate at m = 52
  was ≈ N^{−1.0}; sequential construction degraded to the MC rate. Under
  *measured* wall-clock budgets, PCA-RQMC beat MC at 50 ms but lost at 5 ms,
  because each randomisation costs ~1 ms of overhead.
* *Audit.* Phase 3 also exposed that phase 2 had charged the adaptive
  pilot 6–400× too little time, and retracted the affected claims.

## 4. Phase 4, Part B: is the cost-based rule real?

**The rule.** Phase 3 suggested, after looking at its own data: *use PCA-RQMC
if the budget allows 10 randomisations of at least 64 points, otherwise
MC.* A rule found by inspecting data can simply be fitted to noise. So:

1. We **froze** the rule, its comparators, a new grid of contracts and
   budgets, and pass/fail criteria. They were committed to git (`5f657d4`)
   before writing the validation code.
2. We ran it on 32 contracts (m = 8–104 monitoring dates; volatilities,
   strikes, maturities and rates never used before) at 5–100 ms budgets,
   with 20 independent repetitions each.

**Result: the rule passed.**

* Its worst average regret against an after-the-fact oracle was 1.02
  (criterion ≤ 1.25).
* It chose the better of MC and PCA-RQMC in all 450 cells where the
  difference was statistically clear.
* The two natural alternatives each fail badly in one regime: "always PCA"
  at small budgets, "PCA only for few dates" at large ones.

**Qualifications.**

* The rule's threshold sits 1.3–2.6× below the true crossover budget at
  m ≥ 52. The budget grid happened not to probe that gap.
* At 5 ms, RQMC estimates overran their budget by up to 4.6×.

(`PHASE4_COST_RULE_VALIDATION.md`.)

## 5. Phase 4, Part C: dimension reduction aimed at the Greek

### 5.1 Idea

PCA orders directions by how much they move the **path**. A Greek integrand
may care about different directions.

Write the smoothed integrand as `f_g(ξ)`, with ξ a vector of independent
standard normals that the Sobol points drive. Then
`G_g = E[∇f_g ∇f_gᵀ]` measures how sensitive f_g is, on average, to each
direction. Its leading eigenvector is the single direction along which f_g
changes most.

Rotating coordinates so that this direction comes first, then the next, and
so on, should lower the integrand's effective dimension. This is
**gradient PCA**. It is established (Xiao & Wang 2017/2018; Liu & Owen
2023's active subspaces; Imai & Tan's LT method), and **we do not claim it
as new**.

### 5.2 Why the rotation is legitimate

If U is orthogonal and ξ ~ N(0, I), then Uξ has covariance `UIUᵀ = I`, so
it is again N(0, I). The simulated path `B = L₀Uξ` therefore has exactly
the same distribution as before: every estimate stays unbiased. Only the
*assignment* of quasi-random coordinates to directions changes. The
conditioning on Z₁ is untouched, because the rotation acts only on the
residual path B.

### 5.3 Gradients

The CPW-first integrand depends on B through one smooth scalar,
`δ = ln(Ā/K)/(σ√Δ)`, with `Ā` the conditional average. So its gradient is
`h'(δ)·∂δ/∂B` plus, for Vega, one extra term. All three gradients are
written in closed form (derivations Part IV) and checked against finite
differences to ~1e-10.

### 5.4 One transform for three Greeks

Δ, Γ and 𝒱 have different units, so their gradient matrices are first
scaled to unit trace, then averaged. This gives one rotation that targets
all three. It is a heuristic: it maximises captured gradient energy, which
is *not* the same as minimising estimation error. Whether it helps is
tested.

### 5.5 Costs

Building a gradient transform costs a training sample of gradients (1 024
paths, fixed in advance) plus one eigendecomposition. We report results
both ways:

* **charged** to each estimate (one-off valuation);
* **amortised** (the same contract revalued many times).

## 6. Methods (Part C)

* **12 new evaluation contracts:** m ∈ {12, 52, 104},
  K/S₀ ∈ {0.94, 1.06}, σ ∈ {0.22, 0.38}, T = 1, r = 0.02.
* **Seven transforms:** sequential, bridge, path PCA, Δ-aware, Γ-aware,
  𝒱-aware, joint.
* **Independent random streams** for training the transforms, measuring
  their gradient energy, references, fixed-N runs and timed runs.
* **References:** PCA-RQMC with 32 × 2¹⁶ (2¹⁸ for m ≤ 24) points, each
  cross-checked by plain MC.
* **Fixed-N study:** N = 2⁷…2¹³ points × 16 independent scramblings.
* **Wall-clock study:** 10 and 50 ms, 20 repetitions.
* **Gradient-noise study:** n_train ∈ {32 … 8 192}.

* Run: `python scripts/greek_aware_study.py`, 16 min.

## 7. Results (Part C)

All ratios are geometric means over the 4 scenarios per m. Bracketed
values are 95% bootstrap intervals over scenarios. Tables are in
`results/phase4_greek_aware/study/`.

**Reference check.** 48 plain-MC cross-checks:

* the Greek z-scores have sd 0.8–1.3 and max |z| 2.6 (12 per Greek);
* one price check reached |z| = 3.6;
* reference error is ≤ 1.8% of even the smallest MSE.

**H1: effective dimension (supported, modestly by this metric).** Share of
the Greek's gradient energy carried by the first Sobol coordinate,
measured on an independent sample:

| | sequential | bridge | path PCA | gradient PCA (any) |
|---|---|---|---|---|
| m = 12 | 0.24 | 0.78 | 0.985 | 0.995–0.999 |
| m = 52 | 0.06 | 0.76 | 0.984 | 0.999 |
| m = 104 | 0.03 | 0.75 | 0.984 | 0.999 |

Coordinates needed for 99% of the gradient energy: sequential 10–82,
bridge 6, **path PCA 2, gradient PCA 1**.

The gradient covariance matrices are nearly rank one. Their second
eigenvalue is ≈ 10⁻³ of the trace, against ≈ 10⁻¹ for the path covariance
(`fig4C1`).

**H2: fixed-N error (supported, strongly).** MSE relative to path PCA at
N = 2¹³ points × 16 scramblings:

| Greek | m = 12 | m = 52 | m = 104 |
|---|---|---|---|
| Δ, Δ-aware | 0.18 | 0.19 | 0.07 |
| Γ, Γ-aware | 0.18 | 0.11 | 0.06 |
| 𝒱, 𝒱-aware | 0.34 | 0.33 | 0.09 |
| joint transform, average over Greeks (J) | 0.29 [0.17, 0.49] | 0.17 [0.07, 0.36] | 0.08 [0.05, 0.15] |

* Gradient PCA wins in 33–35 of 36 scenario-Greek cells per transform.
* A reference-free check agrees. Using only the variance across
  independent scramblings, the advantage is 4–10×.
* Fitted convergence slopes are ≈ −1 for both path PCA and gradient PCA
  (per-scenario medians −0.89 to −1.09; `fig4C4`). The gain is a constant
  factor, **not** a better rate.
* The 1.5% of gradient energy that path PCA leaves outside its first
  coordinate costs a factor of 3–20 in MSE.

**H4: joint transform (supported).** The joint transform is as good as the
Greek-specific ones: the J intervals overlap at every m. Any single Greek's
transform also works almost as well for the other two.

*Mechanism.* For Δ and Γ the integrand is a function of one scalar δ, so
`∇ψ_g = h_g'(δ)·∇δ`. Each `G_g = E[h_g'(δ)² ∇δ∇δᵀ]` is a reweighting of
the same direction field, and they share their leading eigenvectors. Vega
adds a second term, but it is nearly aligned too.

This is a structural property of CPW-first for this contract. It should not
be expected to hold for multi-asset or stochastic-volatility problems.

**H5a: noisy gradients (refuted here).** Even n_train = 32 training paths
give the full gain (`fig4C8`). At m = 52 the MSE ratio vs path PCA is
0.12–0.18 at n_train = 32 and 0.12–0.25 at 8 192. The leading direction is
so dominant that a tiny sample finds it.

**H3: error per unit wall-clock time, setup included (mixed).**
Time-normalised MSE relative to path PCA (amortised):

| Deployment | 10 ms: m = 12 / 52 / 104 | 50 ms: m = 12 / 52 / 104 |
|---|---|---|
| gradient PCA, **amortised** (transform reused) | 0.35 / 0.34 / 0.26 | 0.49 / 0.13 / 0.10 |
| joint, amortised | 0.40 / 0.39 / 0.27 | 0.43 / 0.13 / 0.09 |
| gradient PCA, **one-off** (setup charged) | 0.54 / 0.68 / 0.61 | 0.50 / 0.18 / **9.1** |
| joint, one-off | 1.22 / 0.78 / 0.77 | 0.39 / 0.17 / **16.4** |
| path PCA, one-off | 0.88 / 1.58 / 1.31 | 1.03 / 0.97 / **3.96** |
| plain MC | 764 / 0.05 / 0.03 | 10⁴ / 31 / 1.03 |

* **Amortised:** gradient PCA is 2–10× more efficient than path PCA
  everywhere. The joint transform gives three Greeks for one setup.
* **One-off:** setup costs, measured medians (`fig4C7`):

  | m | gradient PCA | joint | path PCA |
  |---|---|---|---|
  | 12 | 0.4–0.5 ms | 1.2 ms | 0.1 ms |
  | 52 | 3.7–5.3 ms | 9.2 ms | 1.2 ms |
  | 104 | 10–14 ms | 24 ms | 4 ms |

  * At 10 ms, the one-off variants still beat path PCA, but at m = 104
    their measured wall time is **2.7–3.9× the budget**, because setup
    alone exceeds it.
  * At 50 ms and m = 104, charging setup makes gradient PCA **9–16× worse**
    than amortised path PCA. Charging path PCA's own setup makes it 4×
    worse.

  *Mechanism, verified:* setup leaves 26–38 ms, so the power-of-two
  scramble size drops from 256 to 32–128 points. RQMC variance falls
  roughly like N⁻² per scrambling at fixed time, so smaller scramblings
  are disproportionately worse. This is a property of our scramble-sizing
  policy interacting with setup cost, not of the transform.
* **H5b: small budgets (supported, for one-off use).** With setup charged,
  the simpler path PCA (cheaper setup) or plain MC is preferable at small
  budgets and large m.
* **MC vs RQMC.** Plain MC is best at 10 ms for m ≥ 52, consistent with
  Part B's crossover.

## 8. Computational trade-offs

* **Repeated valuation** (risk systems, recalibration, scenario grids):
  train once per contract, or for a family of similar contracts, and reuse.
  Gradient PCA dominates path PCA by 2–10× in error per unit time, and one
  joint transform serves all three Greeks.
* **One-off valuation at a small budget:** setup (up to 24 ms for the joint
  transform at m = 104) can exceed the budget. Use path PCA or MC as Part
  B's rule prescribes, or cut training to n_train ≈ 32–128, which loses
  nothing here (H5a). Setup at that size is dominated by the
  eigendecompositions. *This last recommendation is a post-hoc
  observation, not a tested policy.*
* **The scramble-size cliff:** fixed power-of-two sizes make time-boxed
  RQMC sensitive to small losses of budget. Choosing the number of
  scramblings instead (keeping n_s) would remove the cliff, but was not
  tested.

## 9. Failure cases and limitations

* **One model, one contract family.** GBM, the arithmetic Asian digital,
  and the CPW-first estimator. The near-rank-one gradient structure that
  makes the joint transform free may not survive in multi-asset or
  stochastic-volatility models.
* **One machine.** Timings are from a single machine and SciPy's Sobol
  scrambler. Setup and overhead numbers do not transfer; ratios transfer
  better.
* **Small scenario counts.** 12 evaluation scenarios in Part C (4 per m).
  The confidence intervals are over scenarios and are wide.
* **Part B pass is grid-dependent.** It relied on a budget grid that did
  not probe the gap between the rule's threshold and the true crossover at
  m ≥ 52 (Part B §5).
* **Overruns.** RQMC overran small budgets: up to 4.6× in Part B and 3.9×
  in Part C one-off. These were charged in the metrics, but such budgets are
  not achievable with these RQMC settings.
* **The capture metric understates the effect.** Gradient energy in the
  first coordinate (DGSM) moved only from 0.984 to 0.999, while MSE fell
  3–20×. It is a coarse proxy, so we report errors directly.
* **Not tested:** alternative scramble-sizing policies, Brownian-bridge-based
  gradient transforms, and preintegrating along the gradient direction
  itself (Liu & Owen's active-subspace preintegration). The last replaces
  CPW-first's Z₁ and would need a different conditional formula.

## 10. What is known vs what this project shows

| | Status |
|---|---|
| Malliavin/LR weights; superiority on digitals | known (Fournié et al. 1999; Broadie & Glasserman 1996) |
| Conditional smoothing of Asian digital Greeks; QMC on the smoothed integrand | known (Zhang & Wang 2020; Xiao & Wang 2018) |
| Bridge/PCA dimension reduction | known (Moskowitz & Caflisch 1996; Acworth, Broadie & Glasserman 1998) |
| Gradient PCA / active subspaces / LT for QMC, including after preintegration and for hedging | known (Imai & Tan; Xiao & Wang 2017/2018; Liu & Owen 2023) |
| Rao–Blackwell identity CPW-first = E[LR \| Z₂..Z_m] for Δ, Γ (phase 2) | derived here explicitly; probably known to specialists |
| Under measured wall-clock budgets, RQMC needs PCA and a minimum budget; a cost-based MC/PCA rule **passed a pre-registered out-of-sample test** | new empirical evidence (narrow: one machine, one contract family) |
| One trace-normalised joint gradient transform matches Greek-specific transforms, because all CPW-first Greeks share one smooth scalar; gradient noise is irrelevant at n_train ≥ 32 | new empirical observation, with a mechanism; generality untested |
| Gradient PCA's advantage disappears or reverses when its setup is charged at small budgets / large m, via a scramble-size cliff | new empirical observation; policy-dependent |

**Bottom line.**

* *Methods:* none of the methods is new.
* *Contribution:* careful, reproducible, cost-honest evaluation:
  * a pre-registered confirmation of a selection rule;
  * the demonstration that a joint multi-Greek transform is essentially
    free for this estimator, with an explanation of why;
  * a clear account of when setup cost erases the benefit.
* *Publishability:* this is solid coursework or undergraduate-research
  material. Publication would need broader models, more machines, and a
  comparison against active-subspace preintegration.

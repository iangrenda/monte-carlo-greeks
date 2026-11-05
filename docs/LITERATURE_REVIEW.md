# Literature review: Monte Carlo Greeks after Fournié et al. (1999)

**Scope.** This review covers what came after the 1999 paper that is
relevant to (i) Greek estimation for discontinuous and path-dependent
payoffs and (ii) choosing among estimators.

**Verification.** Each entry is marked by how it was checked (searches made
on 2026-11-04):

* **V**: bibliographic details and abstract confirmed against a primary
  listing (publisher, arXiv, RePEc or INFORMS);
* **A**: full text or HTML read in the relevant parts;
* **T**: standard textbook material, cited from general knowledge and not
  re-checked during this review.

Claims marked V are limited to what the abstract or listing states.

---

## 1. Direct successors of the 1999 paper

**Fournié, Lasry, Lebuchoux, Lions (2001).** "Applications of Malliavin
calculus to Monte-Carlo methods in finance. II", *Finance and Stochastics* 5(2),
201–236, doi:10.1007/PL00013529. **[V]**

* Returns to the European Greek weights and answers the question of the
  **optimal weight functional in the sense of minimal variance**.
* Develops Malliavin representations of **conditional expectations**
  (used later for American-option regression).
* Introduces Girsanov transforms with anticipating drifts, giving variance
  reduction by **importance sampling**.

Consequence for us: weight optimisation and conditional expectations were
addressed by the original authors themselves within two years. We did not
read the full text (paywalled), so we do not claim specifics beyond the
abstract.

**Benhamou (2003).** "Optimal Malliavin weighting function for the
computation of the Greeks", *Mathematical Finance* 13(1), 37–53. **[V]**

* Shows that the **likelihood ratio is the weighting function with the
  smallest total variance** among admissible Malliavin weights for the
  Greeks it studies.
* Discusses when Malliavin weights are preferable to finite differences.

Consequence: for terminal European payoffs our "Malliavin" estimator *is*
the likelihood-ratio estimator (derivation in `PROJECT_AUDIT.md` §B.4). By
Benhamou's result it is already variance-optimal within its class, so there
is nothing to optimise in the weight itself.

**Chen & Glasserman (2007).** "Malliavin Greeks without Malliavin calculus",
*Stochastic Processes and their Applications* 117(11), 1689–1723. **[V]**

* Shows that, in several important cases, Malliavin estimators can be
  obtained by elementary means: pathwise and LR on the Euler scheme, then
  time-averaging.

Consequence: "Malliavin" vs "pathwise/LR" is often a difference in
derivation, not in the estimator.

**Broadie & Glasserman (1996).** *Management Science* 42, 269–285, which
introduces pathwise and LR Greeks; cited by FLLLT. **[V]**
**Glasserman (2004).** *Monte Carlo Methods in Financial Engineering*,
ch. 7 (pathwise, LR, mixed estimators; smoothing by conditioning). **[T]**

## 2. Conditioning and smoothing of discontinuous payoffs

**Glasserman & Staum (2001).** "Conditioning on one-step survival for
barrier option simulations", *Operations Research* 49(6), 923–937. **[V]**

* Samples each step conditionally on survival, so every path survives.
  This is a likelihood-weighted smoothing of discretely monitored barrier
  payoffs.
* Later work (arXiv:1804.03975; Frankfurt thesis 2022; arXiv:1906.11002)
  combines it with pathwise sensitivities to obtain stable barrier Greeks,
  including second order. **[V, abstracts]**

**Giles (2009).** "Vibrato Monte Carlo sensitivities", in *Monte Carlo and
Quasi-Monte Carlo Methods 2008*, Springer. **[V]**

* Conditions on the path up to the **penultimate time step**: pathwise for
  the dependence of Ŝ_{N−1} on the inputs, LR / analytic conditional
  expectation for the last step.
* For a digital payoff the conditional expectation is a normal CDF.
* Reported variance: O(h^{-1/2}) for discontinuous payoffs, versus O(h^{-1})
  for pure LR.

**This is essentially the "conditional smoothing over the final increment"
idea proposed for our project**, applied to time-discretised SDEs.

**Zhang & Wang (2020).** "Quasi-Monte Carlo-based conditional pathwise
method for option Greeks", *Quantitative Finance* 20(1), 49–67,
doi:10.1080/14697688.2019.1600714. **[V]**

* Takes a conditional expectation to smooth the discontinuous integrand,
  then differentiates pathwise (CPW).
* Proves that the interchange of differentiation and expectation is valid.
* For Asian and **binary Asian** options the resulting estimates are
  **infinitely differentiable**, which makes QMC efficient.

**Conditional smoothing of discretely monitored Asian digital Greeks,
combined with QMC, is therefore established prior art.**

**Yu & Wang (2023).** *Computational Economics* 62(1): conditional
Malliavin method + QMC for continuous-time Asian Greeks, including binary
Asians. **[V]**

**Albieri, Kucherenko, Scoleri, Bianchetti (2025).** "Effective
dimensionality reduction for Greeks computation using randomized QMC",
*Wilmott* 2025(137); arXiv:2504.11576. **[V]**

* Applies conditional pathwise and Chebyshev interpolation to discretely
  monitored **barrier and arithmetic Asian** options.
* Uses global sensitivity analysis (GSA) to show that these approaches
  reduce effective dimension, so RQMC with a Brownian bridge converges
  faster.

**Bayer, Siebenmorgen, Tempone (2018).** "Smoothing the payoff for efficient
computation of basket option prices", *Quantitative Finance* 18(3), 491–505. **[V]**

* Exact conditional expectation along one Gaussian direction mollifies the
  payoff without bias.
* Used with sparse grids, QMC and MC; the high-order method's advantage
  holds up to moderate dimension (25–35, depending on the version).

**L'Ecuyer, Puchhammer, Ben Abdellah (2022).** *INFORMS J. Computing* 34(3),
1729–1748. **[V]**

* Conditional Monte Carlo density estimators. With RQMC, rates near
  O(n^{-2}) are reported on examples, with conditions given under which
  such rates provably hold.
* Relevant because a digital Greek is a density-type quantity.

## 3. Unbiased estimators for discontinuities without full conditioning

**Liu & Hong (2011).** "Kernel estimation of the Greeks for options with
discontinuous payoffs", *Operations Research* 59(1), 96–108. **[V]**

* Extends pathwise to discontinuous payoffs and second-order Greeks, with
  kernel-smoothed estimators of the boundary term.
* The kernel bandwidth introduces a bias–variance trade-off, as finite
  differences do.

**Lyuu & Teng (2011).** "Unbiased and efficient Greeks of financial
options", *Finance and Stochastics* 15(1), 141–181. **[V]**

* Derives Greek formulas systematically, then estimates them by importance
  sampling.
* Reports unbiased Greeks for rainbow options and a path-dependent option.

**Lakshtanov (2026, preprint).** "Unbiased Monte Carlo Greeks for
Discontinuous Payoffs", arXiv:2609.06137, v1 5 Sep 2026, v2 5 Oct 2026.
**[V, A (HTML)]** *Not peer reviewed.*

* For F(Z, θ) with jumps on surfaces {g_i = 0} in normal space:
  `∂V/∂θ = E[∂_θF] + Σ_i E_w[ΔF · φ(u*) / ∂_u g · ∂_θ g]`.
* Z = u·v + w is split along a fixed direction v, and u* solves g = 0 by
  Newton's method (analytic for GBM).
* ΔF is the jump, obtained by two replays of the pricing kernel.
* Implemented in AADC. Reported results are within 0.1–4% of references
  for barrier options (GBM, GBM plus Hull–White) and a 3-asset Heston
  autocallable.
* What it reveals: the pathwise method loses exactly the surface term. This
  is the Leibniz-rule term of a *one-dimensional conditional integration
  along v*, so it is a *local* version of conditioning, evaluated only at
  the boundary.
* The precise relationship to Giles' vibrato and to CPW (which integrate the
  same Gaussian direction globally) is not discussed in the HTML text we
  read. A companion paper, Goloubentsev, Lakshtanov & Piterbarg (2024), is
  referenced by the author's code repository; we did not verify its
  content.

## 4. QMC and sensitivity analysis for Greeks

**Scoleri, Bianchetti, Kucherenko (2021).** "Application of quasi Monte Carlo
and global sensitivity analysis to option pricing and Greeks: finite
differences vs. AAD", *Wilmott* 2021(116), 66–83, doi:10.1002/wilm.10972
(extended arXiv:2602.14354). **[V]**

* With QMC (Sobol, with Brownian bridge or PCA), FD can match AAD accuracy,
  thanks to effective-dimension reduction.
* Relevance: QMC advantages depend on effective dimension. Our earlier
  one-dimensional European experiments are the trivial best case
  (dimension 1). Path-dependent problems are where the question is real.

## 5. Efficiency criteria and adaptive procedures

**Glynn & Whitt (1992).** "The asymptotic efficiency of simulation
estimators", *Operations Research* 40(3), 505–520. **[V]**

* The principled criterion for comparing unbiased estimators under a
  computing budget is **variance × expected cost per sample**.
* Our selection rule uses this criterion.

**Adaptive Monte Carlo.** Lapeyre & Lelong (arXiv:1001.3551) adapt
importance-sampling parameters by stochastic approximation **[V]**.
Adaptive stratification: arXiv:0809.1135 **[V]**. Capriotti
(arXiv:0808.2332) applies variance reduction to LR Greeks **[V]**. A 2026
multi-fidelity paper (arXiv:2605.23016) studies how limited pilot samples
distort estimator choices **[V]**; this is the effect our hypothesis H5
anticipates.

**What we did not find.** No paper we found studies **pilot-based
selection among Greek estimator families** (FD / LR-Malliavin / CPW /
RQMC), with pilot cost charged to the budget and out-of-sample evaluation.
That is the result of a limited search (about 20 queries), not proof of
absence. The statistical idea is two-stage ranking and selection: estimate
variances in a first stage, choose, then sample.

## 6. Summary table

| Topic | Established by | Status for this project |
|---|---|---|
| Malliavin weights for European Greeks | FLLLT 1999 | replicated (phase 1) |
| LR = minimum-variance weight | Benhamou 2003; FLLLT 2001 | verified algebraically; not a contribution |
| Localisation of IBP | FLLLT 1999 §4 | replicated for calls |
| Conditioning on the last step for digital Greeks | Giles 2009; Zhang & Wang 2020 | prior art: we re-implement as a baseline, not as novelty |
| CPW + QMC for Asian/barrier | Zhang & Wang 2020; Albieri et al. 2025 | prior art |
| Boundary-term unbiased pathwise | Liu & Hong 2011; Lakshtanov 2026 (preprint) | related; not implemented |
| Efficiency = variance × cost | Glynn & Whitt 1992 | used as the selection criterion |
| Pilot-based selection among Greek estimators, cost-charged, out-of-sample | not found | the empirical question of this phase |

## Sources consulted

* https://arxiv.org/abs/2609.06137 and https://arxiv.org/html/2609.06137
* https://ideas.repec.org/a/taf/quantf/v20y2020i1p49-67.html
* https://econpapers.repec.org/RePEc:spr:finsto:v:5:y:2001:i:2:p:201-236
* https://cris.unibo.it/handle/11585/844448 · https://arxiv.org/abs/2602.14354
* https://www.springerprofessional.de/vibrato-monte-carlo-sensitivities/3191776 · https://ora.ox.ac.uk (Giles, vibrato)
* https://ideas.repec.org/a/bla/mathfi/v13y2003i1p37-53.html
* https://arxiv.org/abs/2504.11576
* https://pubsonline.informs.org/doi/10.1287/opre.49.6.923.10018
* https://ideas.repec.org/a/inm/oropre/v59y2011i1p96-108.html
* https://www.arxiv.org/pdf/1607.05572
* https://web.stanford.edu/~glynn/papers/1992/GW92b.html
* https://ideas.repec.org/a/eee/spapps/v117y2007i11p1689-1723.html
* https://link.springer.com/article/10.1007/s00780-010-0137-5
* https://arxiv.org/abs/1906.04607
* https://ideas.repec.org/a/kap/compec/v62y2023i1d10.1007_s10614-022-10257-3.html
* https://arxiv.org/pdf/1804.03975v3 · https://arxiv.org/pdf/1001.3551 · https://arxiv.org/pdf/0809.1135 · https://arxiv.org/pdf/0808.2332 · https://arxiv.org/html/2605.23016

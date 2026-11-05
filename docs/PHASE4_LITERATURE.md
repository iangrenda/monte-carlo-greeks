# Phase 4 literature: integrand-aware dimension reduction for (R)QMC Greeks

Searches were made on 2026-11-05.

* **[V]** means the bibliographic details and abstract were confirmed on a
  publisher, arXiv or index page.
* **[S]** means the content is known only through secondary citations.

Abstracts were read; full texts were not, unless stated.

## 1. What is already published

**Path-generation dimension reduction**

* Moskowitz & Caflisch (1996): the Brownian bridge.
* Acworth, Broadie & Glasserman (1998): PCA.

Both concentrate the variance of the *path*, not of the integrand. [S]
(standard references, cited by every work below)

**Imai & Tan, the linear transformation (LT) method (2006 onward)**

* Chooses an orthogonal transform tailored to the *payoff*, not the path, to
  minimise truncation-sense effective dimension, using the integrand's
  gradient at selected points.
* It is exact for integrands of the form g(linear combination). Later
  generalised (GLT) and, in 2025, a linear-quadratic regression variant.
* [S, via arXiv:0710.5872, arXiv:0907.3092 and an econpapers record]

**Wang & Tan, "Pricing and hedging with discontinuous functions: QMC and
dimension reduction"** (*Management Science*, ≈2013; DOI prefix
10.1287/mnsc.1120.1568) [V abstract]

* For discontinuous payoffs, *where the discontinuities lie* matters more
  than effective dimension.
* Aligning discontinuities with coordinate axes gives large variance
  reduction for prices and **Greeks**.
* The bridge and PCA can make digital problems *harder* for QMC.

**Xiao & Wang, "Enhancing QMC simulation by minimizing effective dimension
for derivative pricing"** (*Computational Economics*, 2017/2019,
doi:10.1007/s10614-017-9732-2) [V]

* **Gradient PCA (GPCA):** "PCA on the gradients of the functions at sample
  points", giving an orthogonal transform that reduces effective dimension.
  Reported on exotic options and MBS.
* This is exactly the per-Greek transform of our Part C (eigenvectors of
  E[∇f ∇fᵀ]).

**Xiao & Wang (2018), "Conditional QMC methods and dimension reduction for
option pricing and hedging with discontinuous functions"** (*J. Comput.
Appl. Math.*) [S: index record, cited by arXiv:2603.01763]

* Conditional QMC (preintegration) to smooth the discontinuity, combined
  with dimension reduction, applied to **pricing and hedging**.
* The **PI_GPCA** method preintegrates first, then applies GPCA to the
  smoothed (d−1)-dimensional integrand.
* Our CPW-first + gradient PCA is an instance of PI_GPCA applied to Greek
  integrands.

**Zhang & Wang (2020), QMC-based conditional pathwise method for option
Greeks** (*Quantitative Finance* 20(1)) [V, phase 2]

* Conditioning gives infinitely differentiable Greek integrands for
  (binary) Asians, which makes QMC effective.
* Its follow-ups apply GPCA to these smoothed integrands; per
  arXiv:2109.05279, "GPCA method can be applied to reduce the effective
  dimension". [S]

**Liu & Owen (2023), "Preintegration via active subspace"** (*SIAM J. Numer.
Anal.* 61(2), 495–514, doi:10.1137/22M1479129) [V]

* Chooses the *preintegration direction* as the first principal component
  of the active-subspace (gradient covariance) decomposition.
* Proves that preintegration can reduce, but not increase, the variance of
  scrambled-net integration.
* On Asian options it is competitive with preintegrating the first Brownian
  principal component.
* Follow-up: Liu (2024, *SIAM J. Sci. Comput.*), constrained active
  subspaces.

**Active subspaces** (Constantine 2015, the general method) [S]: the
eigendecomposition of E[∇f ∇fᵀ].

**Derivative-based global sensitivity measures (DGSM)** [V abstracts]

* Sobol' & Kucherenko (2009); Kucherenko et al. (2009, *RESS* 94(7)).
* E[(∂f/∂x_i)²] bounds Sobol' total indices up to Poincaré constants and
  estimates effective dimension cheaply.
* Our "gradient energy captured by the first k coordinates" is a DGSM-type
  diagnostic.

**Recent work** [V]: Yu & Wang (arXiv:2603.01763, 2026) combine importance
sampling with active subspaces in QMC, and use PI_GPCA as a standard
comparator. Their sensitivity experiment treats Delta only.

## 2. What we therefore do **not** claim

* Gradient PCA / active-subspace rotations for QMC: Imai & Tan; Xiao & Wang.
* Gradient PCA after conditional smoothing (PI_GPCA), including for
  hedging: Xiao & Wang 2018.
* Conditional QMC smoothing of digital/Asian Greeks: Zhang & Wang 2020;
  Wang & Tan.
* Active-subspace choice of the preintegration direction: Liu & Owen 2023.

## 3. What we found no published treatment of (a limited search)

1. **One orthogonal transform optimised jointly for Δ, Γ and 𝒱** of the same
   contract, compared against per-Greek transforms and path PCA. Yu & Wang
   2026 treat Delta only; the GPCA papers optimise for one integrand at a
   time.
2. **Gradient PCA evaluated under measured wall-clock budgets**, with the
   gradient-sampling and eigendecomposition cost *charged* (one-off
   valuation) vs *amortised* (repeated valuation), against an un-trained
   path PCA.
3. **Pre-registered confirmation** of a cost-based MC-vs-RQMC selection rule
   on unseen contracts and budgets (Part B).

These are evaluation questions, not new methods. Any contribution is the
controlled empirical answer, and it can be negative. About 12 targeted
searches cannot prove absence, so the claims are framed as "we found no
published treatment".

## Sources

* https://pubsonline.informs.org/doi/fpi/10.1287/mnsc.1120.1568
* https://link.springer.com/article/10.1007/s10614-017-9732-2
* https://arxiv.org/abs/2202.02682 · https://par.nsf.gov/servlets/purl/10509471 · https://scholars.duke.edu/publication/1682997
* https://arxiv.org/html/2603.01763
* https://arxiv.org/pdf/0710.5872 · https://arxiv.org/pdf/0907.3092
* https://bvbvf1.bib-bvb.de/vufind/PrimoRecord/cdi_proquest_miscellaneous_903630402 · https://www.broda.co.uk/doc/ApplicationOfGSI_Kucherenko_RESS.pdf
* https://arxiv.org/html/2109.05279 · https://arxiv.org/html/2209.11337
* https://ideas.repec.org/a/taf/quantf/v20y2020i1p49-67.html

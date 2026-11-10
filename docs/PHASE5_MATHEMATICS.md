# Phase 5 mathematics, from probability and linear algebra

This note derives every formula Phase 5 relies on, assuming multivariate
normals, conditional expectation, eigen-decompositions and basic optimisation,
but no stochastic calculus. Results that are standard are marked as such; the
full earlier derivations are in `mathematical_derivations.md`.

Notation: Φ and φ are the standard normal cdf and pdf. D = e^{−rT} is the
discount factor.

## 1. A Greek is the derivative of an expectation

The price of a contract is an expectation, V(θ) = E[payoff(θ, Z)], where Z is
a vector of independent N(0,1) variables that drive the simulated path and θ
is a parameter: the spot x, or the volatility σ.

* Delta = ∂V/∂x, Gamma = ∂²V/∂x², Vega = ∂V/∂σ.

Monte Carlo estimates an expectation by an average. To estimate a Greek we
therefore need a random variable whose *expectation* is the derivative. There
are three classical routes:

1. **Finite differences.** Estimate V(θ + h) and V(θ − h) and divide. Biased
   in h and noisy.
2. **Pathwise.** Move the derivative inside: ∂V/∂θ = E[∂payoff/∂θ]. Valid only
   if the payoff is continuous in θ. A digital payoff 1{A ≥ K} jumps, its
   derivative is zero almost everywhere, and the method returns 0. Wrong.
3. **Likelihood ratio / Malliavin.** Move the derivative onto the density
   instead (section 2).

Our contract is an **Asian digital**: it pays 1 if the average of the stock
price over m monitoring dates is at least K. It is discontinuous, so route 2
fails, which is what makes it a good test case.

## 2. Likelihood ratio and Malliavin weights

Write the expectation as an integral against the density of the simulated
quantity S: V(θ) = ∫ g(s) p_θ(s) ds. If θ appears only in the density,

  ∂V/∂θ = ∫ g(s) ∂p_θ(s)/∂θ ds = E[ g(S) · ∂ ln p_θ(S)/∂θ ].

The factor ∂ ln p_θ/∂θ is the *score* from statistics. So a Greek is the
expectation of payoff × weight, and the payoff is never differentiated.

For a lognormal terminal price, S_T = x exp((r − σ²/2)T + σ√T Z), the score
with respect to x works out to Z/(xσ√T). This is the Delta weight of Fournié
et al. (1999), who derive it by Malliavin calculus (an integration by parts on
the space of Brownian paths). For payoffs of S_T alone the two derivations
give the same weight; Phase 1 verified this numerically.

The drawback: the weight has variance of order 1/(σ²T), and for path-dependent
payoffs with m dates the usable weight involves only the first time step, so
its variance grows like m. Phase 2 measured this.

## 3. Conditioning (Rao–Blackwell) fixes the discontinuity

**The tower rule.** For any random variables, E[Y] = E[ E[Y | 𝒢] ], and

  Var(Y) = Var(E[Y | 𝒢]) + E[Var(Y | 𝒢)]  ≥  Var(E[Y | 𝒢]).

Replacing Y by its conditional expectation never increases variance. This is
the Rao–Blackwell theorem.

**Applying it.** Let W_{t_i} be the Brownian motion at monitoring date
t_i = i·dt. Split off the first step:

  W_{t_i} = √dt · Z₁ + B_i,  B₁ = 0,  B_i = W_{t_i} − W_{t_1}.

Z₁ ~ N(0,1) is independent of the *residual path* B = (B₂, …, B_m), because
Brownian increments over disjoint intervals are independent. Every monitored
price contains the same factor e^{σ√dt·Z₁}:

  S_{t_i} = e^{sZ₁} R_i,  s = σ√dt,  R_i = x exp((r − σ²/2)t_i + σB_i).

So the average is A = e^{sZ₁} Ā with Ā = mean_i R_i > 0, and

  {A ≥ K} = {Z₁ ≥ −δ},  δ = ln(Ā/K)/s,
  P(A ≥ K | B) = Φ(δ).

The discontinuous indicator has become the smooth function Φ(δ(B)). Now the
pathwise route (route 2) is legal. Ā is proportional to x, so ∂δ/∂x = 1/(xs):

  Delta = D · E[ φ(δ)/(xs) ],
  Gamma = D · E[ −φ(δ)(δ + s)/(xs)² ],
  Vega  = D · E[ φ(δ) ( (∂ ln Ā/∂σ)/s − δ/σ ) ].

We call this estimator **CPW-first** (conditional pathwise, first increment
integrated out). It is established work (Zhang & Wang 2020). All three Greeks
are functions of the same scalar δ(B) and the same density φ(δ); only the
polynomial factor in front differs. Sections 5 and 6 depend on that.

## 4. Brownian covariance, PCA and why the ordering of inputs matters

The residual path B is a multivariate normal vector with covariance

  C_{kl} = dt · min(k, l),  k, l = 1, …, m − 1.

Any matrix L with L Lᵀ = C generates it: B = Lξ with ξ ~ N(0, I). Proof:
Cov(Lξ) = L I Lᵀ = C, and a linear map of a Gaussian vector is Gaussian.
Different L give the same distribution, so **every such L gives an unbiased
estimator with the same plain Monte Carlo variance**.

They differ for quasi-Monte Carlo. A Sobol' sequence fills the unit cube more
evenly than random points, but its evenness is best in the first few
coordinates. So we want the first coordinates of ξ to control most of what the
integrand does.

* **Sequential** (L = Cholesky): ξ_k is the k-th increment. Every coordinate
  matters about equally.
* **Path PCA**: C = VΛVᵀ, L = VΛ^{1/2}. The first coordinate moves the path
  along the eigenvector with the largest eigenvalue. For Brownian motion the
  first eigenvector is a slow half-sine wave and carries about 81% of the
  total variance, the second about 9%.

**Randomised QMC (RQMC).** A fixed Sobol' set gives one number and no error
estimate. Randomising it (Owen scrambling, or a random digital shift) makes
each point exactly uniform while keeping the evenness. R independent
randomisations give R independent unbiased estimates μ̂₁, …, μ̂_R. Their mean
is the estimate and their sample standard deviation over √R is its standard
error, with a t_{R−1} interval because R is small.

## 5. Standard PCA versus gradient-aware PCA

Path PCA looks only at the *path*. What matters for integration error is the
*integrand* f(ξ) = ψ(Lξ), where ψ is the CPW-first formula. A direction of ξ
is important if f changes a lot along it. Measure this with the matrix of
mean squared directional derivatives,

  G = E[ ∇f(ξ) ∇f(ξ)ᵀ ]  (symmetric, positive semi-definite).

For a unit vector u, uᵀGu = E[(∂f/∂u)²]. The eigenvector of G with the largest
eigenvalue is the direction of largest mean squared slope. Take
G = UΛUᵀ and use L' = LU. Then:

* L'L'ᵀ = L U Uᵀ Lᵀ = L Lᵀ = C, because U is orthogonal. The path distribution
  is untouched, so the estimator is still unbiased, whatever U is.
* The first Sobol' coordinate now drives the steepest direction of f.

This is **gradient PCA** (Xiao & Wang 2017), also known as the active-subspace
method (Constantine 2015). It is established; we do not claim it.

**Why G is nearly rank one here.** By the chain rule, with f = h(δ(B)),

  ∇_B f = h′(δ) · ∇_B δ,  (∇_B δ)_i = σR_i / (m Ā s).

∇_B δ is a vector of positive weights R_i/(mĀ) that sum to less than 1; it
varies only mildly from path to path. So ∇f is, to a good approximation, a
*random scalar* h′(δ) times a *nearly fixed vector*. Then G ≈ E[h′(δ)²]·vvᵀ, a
rank-one matrix. Vega has one extra term, which is why its G is slightly less
concentrated.

## 6. Three Greeks as one multivariate estimation problem

A risk system wants the vector g = (Delta, Gamma, Vega), not one number.

**Shared paths.** All three CPW-first formulas need the same expensive
quantities: the m exponentials R_i, the sum Ā, δ and φ(δ). Computing the
vector on one set of paths costs barely more than one Greek. Estimating them
in three separate simulations with a third of the time each wastes two thirds
of the work.

**The error is a random vector.** ĝ − g has a 3×3 covariance matrix Σ. Its
diagonal holds the three variances. Its off-diagonal entries matter whenever
the Greeks are combined: for a linear combination aᵀĝ (for example a
Taylor P&L estimate a = (ΔS, ½ΔS², Δσ)), Var(aᵀĝ) = aᵀΣa. With RQMC, Σ is
estimated by the sample covariance of the R replicate vectors, divided by R.

**A scalar loss.** To compare methods we need one number:

  L = Σ_g w_g · MSE_g / s_g²,  w_g ≥ 0, Σ w_g = 1.

The scales s_g make the terms unit-free. Dividing by the true Greek (relative
error) is unstable because Gamma and Vega change sign. We use s_g = the
per-path standard deviation of the plain Monte Carlo CPW-first estimator. Then
MSE_g/s_g² = 1/n_eq, where n_eq is the number of plain Monte Carlo paths that
would give the same accuracy. L is the weighted mean of 1/n_eq.

**One rotation for all three.** A simulation has one L, so one rotation U must
serve three integrands. We use the eigenvectors of

  G_w = Σ_g w_g · G_g / tr(G_g).

Dividing by the trace gives each Greek the same total "gradient energy" before
mixing, so units do not matter. By section 5 each G_g ≈ c_g·v_g v_gᵀ, and all
three v_g are close to the same vector ∇_B δ. So G_w is itself nearly rank one
*with the same top eigenvector for any weights*. That predicts two things,
both tested in Phase 5C: the joint rotation loses almost nothing against a
Greek-specific one, and changing w barely changes the rotation.

This alignment is a property of this contract (one conditioning scalar δ). It
is not a general theorem; Phase 5E tests a contract where it should weaken.

## 7. Choosing the sampler as a constrained optimisation

**Cost model.** Measured on the machine:

  cost_RQMC(R, N) = c₀ + R·(a + b·N),  cost_MC(n) ≈ c₀′ + c·n.

a is the set-up cost of one randomisation, b the cost per point.

**Error model.** For RQMC with R replicates of N points, the estimator's
variance is σ_N²/R, and empirically σ_N² ≈ κ·N^{−α} with α ≈ 2 for PCA
constructions (Phase 4). For MC, variance = s²/n.

**The schedule problem.**

  minimise   N^{−α}/R
  subject to c₀ + R(a + bN) ≤ B,  R ≥ R_min,  N ∈ {16, 32, 64, …}.

*Why R should be small.* Substitute the largest feasible R ≈ (B − c₀)/(a + bN):
the objective becomes N^{−α}(a + bN)/(B − c₀). Its derivative in N is negative
whenever α ≥ 1. So with α = 2 it always pays to put the time into more points
per replicate rather than more replicates. The optimum is the largest N that
still leaves R_min replicates.

*Why R cannot be too small.* The error bar uses a t_{R−1} quantile: 12.7 for
R = 2, 4.30 for R = 3, 2.36 for R = 8. A second objective makes this explicit
by minimising the squared half-width, t²_{R−1}·N^{−α}/R. Doubling N while
halving R multiplies it by (t_{R/2−1}/t_{R−1})²·2/2^α. With α = 2 that is
favourable only while the t-ratio squared is below 2, which stops near R ≈ 4.

**Feasibility.** If even R_min replicates of the smallest N violate the
constraint, the problem has no solution. The old scheduler ran them anyway and
overshot the budget. The new one reports infeasibility and falls back to MC,
whose smallest unit of work is far cheaper.

**The MC-versus-RQMC rule** of Phase 4 is the statement "use RQMC iff the
budget affords 10 replicates of 64 points". In the notation above:
B ≥ 10(a + 64b). It encodes the fact that below some N the N^{−α} advantage
has not yet overtaken the overhead a.

## 8. Set-up cost versus marginal cost

Training a gradient-PCA rotation costs C_train once (sample gradients, one
eigen-decomposition). M valuations then cost

  C_total(M) = C_train + Σ_{j=1}^M C_eval(j).

Fix the time per valuation at B, so the total is M·B. A method with set-up
cost has only B − C_train/M left for each simulation. Its loss is
ℓ_trained(B − C_train/M); the untrained competitor's is ℓ_plain(B). Training
pays once

  ℓ_trained(B − C_train/M) < ℓ_plain(B).

The **break-even M\*** is the smallest M satisfying this. For M = 1 the
set-up may exceed B entirely (infeasible). As M → ∞ the charge vanishes
(the amortised case). Phase 5D measures both sides of this inequality directly
with timed runs.

**Reuse across contracts.** By construction (see `joint_greeks.py`), every
rotation is stored relative to the PCA basis of the matrix min(k,l), which
does not depend on σ, K, T or dt. A rotation trained on one contract can be
applied to any other with the same m and remains unbiased. Whether it remains
*efficient* depends on whether the top eigenvector of G moves, which is an
empirical question (Phase 5D, transfer).

## 9. A two-asset basket: where the one-scalar structure breaks

Two correlated stocks with Brownian motions W¹, W², Corr = ρ. Payoff
1{A ≥ K} with A = w₁·mean_i S¹_{t_i} + w₂·mean_i S²_{t_i}.

**A common direction.** Let Z¹, Z² be the standardised first-step increments
and Z₀ = (Z¹ + Z²)/√(2(1+ρ)), which is N(0,1). By the regression formula for
jointly normal variables, any Gaussian Y splits as Cov(Y, Z₀)·Z₀ plus a part
independent of Z₀. Here Cov(W^j_{t_i}, Z₀) = c√dt with c = √((1+ρ)/2), for
both assets and all dates. So

  W^j_{t_i} = c√dt·Z₀ + X^j_i,  X independent of Z₀,
  Cov(X^j_i, X^l_k) = dt·(ρ_{jl}·min(i,k) − c²).

**Conditioning.** Given X,

  A(z) = a₁e^{s₁z} + a₂e^{s₂z},  s_j = σ_j c√dt > 0,
  a_j = w_j·mean_i x_j exp((r − σ_j²/2)t_i + σ_j X^j_i).

A(z) is strictly increasing, so {A ≥ K} = {Z₀ ≥ z\*} where A(z\*) = K, and
P(A ≥ K | X) = Φ(−z\*). When σ₁ ≠ σ₂ there is no closed form for z\*; we
solve by Newton's method on ln A(z) − ln K, which is increasing and convex
(a log-sum-exp), so Newton converges.

**Greeks by the implicit function theorem.** Differentiating A(z\*(θ), θ) = K
gives ∂z\*/∂θ = −(∂A/∂θ)/(∂A/∂z). With ã_j = a_j e^{s_j z\*} (so ã₁ + ã₂ = K)
and A′ = s₁ã₁ + s₂ã₂:

  Delta_j = D·φ(z\*)·ã_j/(x_j A′),
  Vega_j = D·φ(z\*)·(∂A/∂σ_j)/A′,
  Gamma_jj = D·[ z\*φ(z\*) z_x² − φ(z\*) z_xx ],  z_x = −ã_j/(x_j A′).

Each per-path formula is the exact derivative of the per-path smooth price
D·Φ(−z\*). The tests check this by finite differences path by path, to six
digits.

**Why rank one should fail.** z\* now depends on *two* scalars, a₁ and a₂,
which are driven by different parts of the path (asset 1's and asset 2's).
∇z\* is a combination of ∇a₁ and ∇a₂ with path-dependent coefficients
ã_j/A′, and Delta₁ carries the factor ã₁ while Delta₂ carries ã₂. So the
gradient covariance should have roughly two significant directions, and
Greeks of different assets need not share a top direction. Phase 5E measures
how much this costs a single joint rotation.

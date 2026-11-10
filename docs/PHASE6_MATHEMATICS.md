# Phase 6 mathematics

Two topics: what SciPy's "scrambling" actually is (6A), and how to build the
joint rotation without sampling gradients (6B). Notation follows
`PHASE5_MATHEMATICS.md`.

## 1. Three randomisations of one Sobol' point set

**Digital representation.** Write each coordinate of a Sobol' point with 30
binary digits, u = Σ_{i=1}^{30} x_i 2^{−i}. The first 2^k points in
coordinate j are x(n) = C_j·n (mod 2), where n is the point index written in
binary and C_j is a 30 × k binary *generating matrix*. In integers:

  x(n) = ⊕_{b : bit b of n is 1} V_j[b],

where V_j[b] is column b of C_j stored as a 30-bit integer and ⊕ is XOR.

**Digital shift.** Draw s uniformly from the 30-bit integers and use x(n) ⊕ s.
XOR with a uniform s makes every digit uniform, so each point is uniform.
Shifting every point by the same s keeps the *differences* between points, so
the stratification of the net survives.

**Linear matrix scrambling (Matoušek).** Draw a random lower-triangular
binary matrix M with ones on the diagonal, and use M·x(n) ⊕ s.
* Lower-triangular means output digit i depends only on input digits 1…i.
  Two points that agree in their first i digits still agree after scrambling,
  so every stratum of the net maps to a stratum: the net property survives.
* The diagonal ones make M invertible.
* Because M is linear over GF(2), M·x(n) = ⊕_b M·V_j[b]. **Scrambling the k
  columns of the generating matrix once is the same as scrambling every
  point.** That is the whole implementation trick of 6A.

**What SciPy does.** `qmc.Sobol(scramble=True)` builds a new engine for each
replicate. It draws d × 30 × 30 random bits for M, scrambles the direction
numbers and draws a shift. That construction, not the arithmetic, is the
0.4–2 ms per replicate measured in Phases 4–5.

**What `lms_uniforms` does.**
1. Cache the unscrambled columns V_j[b] once per (d, k).
2. Per replicate, draw M as 30 random 30-bit row masks per coordinate.
3. Apply M to the k cached columns, computing output digit i as the parity
   of (row_i AND column) with one vectorised popcount.
4. Generate all 2^k points by doubling: X[2^b : 2^{b+1}] = X[0 : 2^b] ⊕ W[b].

Steps 2–3 cost O(d·k) bit operations and step 4 costs one XOR per coordinate
of each point. The randomisation is the same as SciPy's. The tests confirm
this through equal replicate variances on a smooth test integrand (ratio
within 0.8–1.25 for 600 replicates each).

**Why this matters for the theory.** A digital shift only translates the
net. Matrix scrambling also randomises the digits *within* each stratum.
Owen's analysis of scrambled nets shows that this randomisation can improve
the variance rate for smooth integrands from about N⁻² to about N⁻³ (up to
log factors). Owen (2003, "Variance with alternative scramblings of digital
nets") shows that several linear scramblings share the variance of nested
uniform scrambling (bibliography [S]; not re-checked in this phase). A shift
alone does not have this property. So at equal N, LMS+shift should never be
worse than a shift and may be better (hypothesis A2).

**What the holdout showed.** The prediction failed for these integrands.
* At equal N the variance ratio of shift to LMS ranged from 0.57 to 1.34,
  with significant differences in both directions.
* All three randomisations converged at the same rate (variance ∝ N⁻²·⁰).

The rate advantage of scrambling is an asymptotic statement for smooth
integrands. At dimensions 15–79 and N ≤ 2¹⁴, after CPW-first smoothing and a
PCA construction, it did not appear. The 5-dimensional smooth test integrand
in the unit tests does show it.

## 2. A deterministic gradient covariance (6B)

**Target.** The trained joint rotation is the eigenbasis of

  G = (1/3) Σ_g G_g / tr(G_g),  G_g = E[∇_ξ f_g ∇_ξ f_gᵀ],  f_g = h_g(δ(Lξ)),

with ξ ~ N(0, I) and L the PCA factor. Each G_g is estimated from n_train
gradient samples. We want the same matrix without sampling.

**Step 1: the gradient is a scalar times a slowly varying vector.** By the
chain rule, ∇_ξ f_g = h_g′(δ)·∇_ξ δ and δ = (ln Ā − ln K)/s. Hence

  G_g = s⁻² · E[ h_g′(δ)² · ∇_ξ ln Ā · (∇_ξ ln Ā)ᵀ ].

**Step 2: linearise the gradient of ln Ā around a point B₀ = Lξ₀.**

  ∇_ξ ln Ā(Lξ) ≈ a + K(ξ − ξ₀),  a = Lᵀ∇_B ln Ā(B₀),  K = Lᵀ ∇²_B ln Ā(B₀) L.

Both derivatives are explicit. With w_i = R_i / Σ_k R_k:

  ∂ ln Ā / ∂B_i = σ w_i,  ∂² ln Ā / ∂B_i ∂B_j = σ² (w_i 1{i = j} − w_i w_j).

(Differentiate ln Σ R_k with R_k ∝ e^{σB_k}.) Write c = a − Kξ₀, so the
linearised gradient is c + Kξ.

**Step 3: linearise δ along the leading direction.** Let â = a/|a| and
t = âᵀξ ~ N(0, 1). Then δ ≈ δ₀ + (|a|/s)(t − t₀), with δ₀ = δ(B₀) and
t₀ = âᵀξ₀, so the weight h_g′(δ)² ≈ w_g(t) depends on ξ only through t.

**Step 4: integrate exactly over the other directions.** Decompose
ξ = t·â + η, where η is independent of t, has mean 0 and covariance
I − ââᵀ. Using E[η] = 0, E[ηηᵀ] = I − ââᵀ and independence:

  E[w_g (c + Kξ)(c + Kξ)ᵀ]
   = E[w_g] ccᵀ + E[w_g t] (c(Kâ)ᵀ + (Kâ)cᵀ)
     + K (E[w_g t²] ââᵀ + E[w_g] (I − ââᵀ)) Kᵀ.

Only three one-dimensional Gaussian expectations of w_g remain. They are
computed by 80-node Gauss–Hermite quadrature, and the matrix algebra is
O(m²) after K Kᵀ (O(m³)). The tests check this formula against a Monte Carlo
average of the same linearised quantities.

**The weights.** Up to constants that cancel after trace normalisation:

| Greek | h_g′(δ) |
|---|---|
| Delta, h ∝ φ(δ) | −δ φ(δ) |
| Gamma, h ∝ −φ(δ)(δ + s) | φ(δ)(δ(δ + s) − 1) |
| Vega, h ∝ φ(δ)(ℓ/s − δ/σ) | −δφ(δ)(ℓ/s − δ/σ) − φ(δ)/σ, with ℓ = ∂ ln Ā/∂σ frozen at B₀ |

**Choice of B₀: the design point.** The Greek integrands carry the factor
φ(δ), so they live near the exercise boundary δ = 0, i.e. Ā = K. The most
likely path on that boundary is

  B* = argmin ½ BᵀC⁻¹B subject to ln Ā(B) = ln K.

The Lagrange condition is C⁻¹B = λ∇ ln Ā(B). It is solved by the
Hasofer–Lind–Rackwitz–Fiessler iteration
B ← ((∇g)ᵀB − g) / ((∇g)ᵀ C ∇g) · C∇g with g = ln Ā − ln K, which converges
in a few steps here because ln Ā is nearly linear in B. The tests verify the
constraint, the Lagrange condition, and that nearby boundary points have a
larger Mahalanobis norm.

**The rotation.** Its columns are the eigenvectors of the resulting
deterministic G, ordered by decreasing eigenvalue. Any orthogonal U keeps
LUUᵀLᵀ = C, so the estimator stays unbiased whatever the quality of the
approximations. Approximations can only cost variance.

**Simpler candidates and why they fell short on development data.**
* *First-order* rotations ("ones", "origin", "dominant"): take only the
  leading direction (∝ Lᵀ∇ ln Ā at some point) and complete it with a
  Householder reflection.
  * Their first column matched the trained one almost perfectly
    (|cos| ≥ 0.9999), yet their loss was 1.5–1.8× that of a well-trained
    rotation.
  * So with G 99.9% rank one, the variance that RQMC still has depends on how
    the *remaining* 0.1% of gradient energy is ordered over the Sobol'
    coordinates.
* *Taylor without weights* (aaᵀ + KKᵀ): orders the remaining directions by
  the curvature of ln Ā but ignores the Greek weights. Development loss was
  1.12–1.33×.
* *Weighted at the origin*: the weights are evaluated far from the boundary
  for out-of-the-money contracts. Development loss was 1.15–1.25×.

**Relation to published work.**
* Using a gradient at a chosen point for the first column of an orthogonal
  transform is the linear-transformation method of Imai & Tan (2006).
* Design points are standard in reliability analysis and in importance
  sampling for options (Glasserman, Heidelberger & Shahabuddin 1999).
* Second-order (Hessian-based) approximations of active subspaces have been
  studied in the active-subspace literature.

The specific combination tested here (a quadrature-weighted second-order
surrogate of the *joint, trace-normalised* gradient covariance of
conditionally smoothed Greeks) was not found in the Phase-4 literature
search. That search was not repeated for Phase 6, so no novelty is claimed.

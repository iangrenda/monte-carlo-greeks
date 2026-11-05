# Mathematical derivations

This note derives every formula implemented in `src/mc_greeks`, says which
results come from Fournié, Lasry, Lebuchoux, Lions & Touzi (1999) (hereafter
**FLLLT**) and which are standard results or our own additions, and states
the conditions under which each estimator is valid.

---

## 0. Attribution: what comes from where

| Item | Source |
|---|---|
| Weighted-expectation representation of Greeks, weight independent of φ (Props. 3.1–3.3) | FLLLT §3 |
| Black–Scholes Delta / Gamma / Vega weights with `a(t) = 1/T` | FLLLT §4 (p. 405) |
| Interval digital `1_[100,110]`, `x=100, r=0.1, σ=0.2, T=1`; low-discrepancy experiments (Figs. 1–3) | FLLLT §4 |
| Localisation of the call Delta and Gamma (`H_δ`, `I_δ`, `F_δ`), δ = 10; N = 10 000 benchmark table | FLLLT §4 (pp. 409–410) |
| FD convergence rates n^(-1/4), n^(-1/3), n^(-1/2) with CRN (cited by FLLLT from Glynn 1989; L'Ecuyer & Perron 1994) | FLLLT §1, citing the literature |
| Closed-form Black–Scholes prices and Greeks | standard (Black–Scholes 1973; e.g. Hull, Glasserman 2004) |
| Pathwise and likelihood-ratio estimators | Broadie & Glasserman (1996); Glasserman (2004, ch. 7) |
| Elementary Gaussian integration-by-parts proofs of the weights (§2 below) | our independent check |
| Localised **Vega** for the call, exact-variance quadrature, bias/variance orders of FD for jump payoffs, all experiments B–G | this project |

---

## 1. Model and exact simulation

Under the risk-neutral measure, `dS_t = r S_t dt + σ S_t dW_t`, `S_0 = x`. Itô's
formula applied to `log S_t` gives the exact solution

```
S_T = x · exp((r − σ²/2) T + σ W_T),     W_T = √T · Z,   Z ~ N(0, 1).
```

For a European payoff `φ(S_T)` only `S_T` matters, and its law is known
exactly. So sampling `Z` and applying this map draws from the exact
distribution of `S_T`, and there is **no discretisation bias**. An
Euler–Maruyama scheme would add an O(Δt) weak error and extra cost, and buys
nothing here. The value is `V = D · E[φ(S_T)]` with `D = e^{−rT}`, and every
estimator below multiplies by the same `D`.

Two derivatives of the path are used repeatedly:

```
∂S_T/∂x = S_T / x  (= Y_T, the tangent process in FLLLT),     ∂²S_T/∂x² = 0,
∂S_T/∂σ = S_T (W_T − σT).
```

---

## 2. Gaussian integration by parts (the one-dimensional Malliavin identity)

**Lemma.** Let `W ~ N(0, T)` and let `g` be absolutely continuous with
`E|g'(W)| < ∞`, and `h` be C¹ with polynomial growth. Then

```
E[g'(W)]       = E[g(W) · W/T],
E[g'(W) h(W)]  = E[g(W) · (h(W) W/T − h'(W))].
```

*Proof.* The density `p(w) = (2πT)^{-1/2} e^{-w²/2T}` satisfies `p'(w) = −(w/T) p(w)`.
Integrating by parts,
`∫ g'(w) h(w) p(w) dw = −∫ g(w) (h p)'(w) dw = ∫ g(w) (h(w) w/T − h'(w)) p(w) dw`.
The boundary terms vanish because `p` decays faster than any polynomial. ∎

On Wiener space this is the duality `E[⟨DF, u⟩] = E[F δ(u)]` between the
Malliavin derivative `D` and the Skorohod integral `δ`, applied to
`F = g(W_T)` and the deterministic process `u ≡ 1/T`, for which
`δ(u) = ∫₀ᵀ (1/T) dW_t = W_T/T`. FLLLT use the same duality in full generality
(any diffusion, path functionals, anticipating integrands). The lemma above
is the special case needed when everything is a function of the single
Gaussian variable `W_T`.

The key point: after integration by parts **the derivative has moved off the
payoff onto the Gaussian density**. The result needs no `φ'`, so it remains
valid, by a density argument (FLLLT, proof of Prop. 3.2(ii)), for bounded
measurable `φ` such as indicators.

Write `g(w) = φ(S(w))` with `S(w) = x exp((r−σ²/2)T + σw)`. Then
`g'(w) = σ S φ'(S)` and `g''(w) = σ² S² φ''(S) + σ² S φ'(S)` (for smooth `φ`).

### 2.1 Delta

```
Δ = D ∂_x E[φ(S_T)] = D E[φ'(S_T) S_T / x] = D/(σx) · E[g'(W_T)] = D · E[ φ(S_T) · W_T/(x σ T) ].
```

**Agreement with FLLLT Prop. 3.2.** The proposition gives
`∇u(x) = E[φ(X_T) ∫₀ᵀ a(t) [σ⁻¹(X_t) Y_t]* dW_t]` for any `a` with
`∫₀ᵀ a = 1`. In Black–Scholes `σ(X_t) = σ S_t` and `Y_t = S_t/x`, so
`σ⁻¹(X_t) Y_t = 1/(σx)`. With `a = 1/T` this is `W_T/(xσT)`. ✔

### 2.2 Vega

```
𝒱 = D E[φ'(S_T) S_T (W_T − σT)] = D/σ · E[g'(W_T) (W_T − σT)].
```

Apply the lemma with `h(w) = (w − σT)/σ`, `h' = 1/σ`:

```
𝒱 = D · E[ φ(S_T) · ( W_T²/(σT) − W_T − 1/σ ) ].
```

This equals the FLLLT §4 formula (their Prop. 3.3 specialised to constant σ). ✔

### 2.3 Gamma

```
Γ = D E[φ''(S_T) S_T²] / x²     (because ∂²S_T/∂x² = 0)
  = D/(x²σ²) · E[g''(W) − σ g'(W)]
```

using `φ'' S² = g''/σ² − g'/σ`. Two applications of the lemma give
`E[g''] = E[g' W/T] = E[g (W²/T² − 1/T)]`, and one gives `E[g'] = E[g W/T]`. So

```
Γ = D · E[ φ(S_T) · (1/(x²σT)) · ( W_T²/(σT) − W_T − 1/σ ) ].
```

This equals the FLLLT §4 gamma formula. Comparing with 2.2 gives
`π_Γ = π_𝒱 / (x²σT)`, the Black–Scholes identity
**`𝒱 = x² σ T Γ`**, which holds for every European payoff. (It is tested in
`tests/test_analytical.py` and `tests/test_malliavin.py`.) ✔

### 2.4 Properties of the weights

* **Payoff independence.** `π` depends only on `(x, σ, T, W_T)`. One set of
  simulated `(S_T, W_T)` pairs gives price, Delta, Gamma and Vega at the
  cost of one payoff evaluation per path. FLLLT stress this point (p. 393).
* **Mean zero.** `E[π] = 0` (take `φ ≡ 1`).
* **Scaling.** `Var(π_Δ) = 1/(x²σ²T)` and
  `Var(π_𝒱) = E[(Z² − 1 − σ√T Z)²]/σ² = (2 + σ²T)/σ²`, using `W_T = √T Z`
  and `E[Z⁴] = 3`. So `Var(π_Γ) = (2 + σ²T)/(x⁴σ⁴T²)`.
  The weights blow up as `σ → 0` or `T → 0`. That alone does not make the
  estimator worse in *relative* terms, because the true Greeks of
  discontinuous payoffs blow up too (e.g. digital Δ ∝ 1/(xσ√T)). Experiments D
  and F measure the net effect.
* **Unboundedness.** The weights grow polynomially in `W_T`. For payoffs
  that are large where `|W_T|` is large (deep in-the-money calls), `φ · π`
  has a large variance. That is the motivation for localisation (§6).

### 2.5 Relation to the likelihood-ratio method (an honest caveat)

For this one-dimensional European problem the Malliavin weights coincide
exactly with the **likelihood-ratio (score-function)** weights of Broadie &
Glasserman (1996). With `log S_T ~ N(log x + (r−σ²/2)T, σ²T)`, the score
`∂_x log f(S_T; x) = W_T/(xσT) = π_Δ` and `f_xx/f = π_Γ`. So in this project
"Malliavin estimator" and "LR estimator" are the same numbers. What
Malliavin calculus adds is the ability to construct such weights when the
density of the underlying functional is unknown (path-dependent payoffs,
general diffusions; FLLLT §§3–4, Asian examples). We do not implement those
cases, so our experiments evaluate the **Black–Scholes European special
case** only.

---

## 3. Analytical benchmarks

Let `d₂(K) = [ln(x/K) + (r − σ²/2)T]/(σ√T)` and `d₁ = d₂ + σ√T`, and let
`Φ`, `ϕ` be the standard normal CDF and PDF. Useful derivatives:

```
∂d₂/∂x = 1/(xσ√T),        ∂d₂/∂σ = −d₂/σ − √T = −d₁/σ.
```

(For the second: `ln(x/K) + rT = d₂σ√T + σ²T/2`, so
`∂d₂/∂σ = −(ln(x/K)+rT)/(σ²√T) − √T/2 = −d₂/σ − √T`.)

**Cash-or-nothing digital call**, `V = D Φ(d₂)` because `P(S_T > K) = Φ(d₂)`:

```
Δ = D ϕ(d₂) / (xσ√T)
Γ = ∂_x Δ = D [ −d₂ ϕ(d₂)/(x²σ²T) − ϕ(d₂)/(x²σ√T) ] = −D ϕ(d₂) d₁ / (x²σ²T)
𝒱 = D ϕ(d₂) ∂d₂/∂σ = −D ϕ(d₂) d₁ / σ           (per unit of σ)
```

(using `ϕ'(d) = −d ϕ(d)`). As a check, `x²σT Γ = −D ϕ(d₂) d₁/σ = 𝒱`. ✔

**Interval digital.** `1{a ≤ S_T ≤ b} = 1{S_T > a} − 1{S_T > b}` almost surely,
because `P(S_T = a) = 0`. So every Greek is `digital(a) − digital(b)`.

**European call.** `V = xΦ(d₁) − KDΦ(d₂)`, `Δ = Φ(d₁)`, `Γ = ϕ(d₁)/(xσ√T)`,
`𝒱 = xϕ(d₁)√T`. These use the identity `x ϕ(d₁) = K D ϕ(d₂)`.

**Paper parameters** (`x=100, r=0.1, σ=0.2, T=1, a=100, b=110`): `d₂(100) = 0.4`,
`d₂(110) = −0.076551`, and

| | Price | Delta | Gamma | Vega (per unit σ) |
|---|---|---|---|---|
| ours | 0.16823763 | −0.00133486 | −0.00038876 | −0.77751371 |
| FLLLT table, "exact" column | — | −0.001335 | −0.000389 | −0.777516 |

For the call (`K = 100`): ours `Δ = 0.725747, Γ = 0.016661, 𝒱 = 33.322460`, while
the FLLLT table lists `0.725747, 0.016660, 33.320063`. Our values are the
standard closed forms and are confirmed in the tests against
Richardson-extrapolated finite differences of the price (agreement to 1e-7
relative). We cannot account for the paper's small differences in the
call Vega (relative 7e-5) and the digital Vega (relative 3e-6). They may
come from rounding or from how the "exact" column was computed. We report
them and do not adjust our benchmarks.

**Units.** Vega is `∂V/∂σ` per unit of σ. The value per volatility point
(Δσ = 0.01) is `𝒱/100`. The paper's −0.7775 is per unit, which matches our
convention.

---

## 4. Finite differences: truncation bias vs Monte Carlo variance

An FD estimator is a linear combination `θ̂ = (1/div) Σᵢ cᵢ V̂(θ + kᵢh)`.

**Bias** (deterministic): `E[θ̂] − θ` is the stencil applied to the exact
price function. Taylor expansion gives `O(h)` for forward differences,
`O(h²)` for central ones, and `O(h²)` for the central second difference.
Because it does not depend on how the `V̂` are coupled, it is the same for
CRN and IND. We compute it exactly from the closed-form price
(`fd_expected_value`).

**Variance** with N paths per bumped price:

| coupling | payoff | Var(Δ̂) | Var(Γ̂) |
|---|---|---|---|
| independent (IND) | any | `O(1/(N h²))` | `O(1/(N h⁴))` |
| common (CRN) | Lipschitz (call) | `O(1/N)` | `O(1/(N h))` (kink) |
| common (CRN) | jump (digital) | `O(1/(N h))` | `O(1/(N h³))` |

*Why the CRN digital rows hold.* For `φ = 1{s > K}`, the per-path
difference `φ(S_T(x+h)) − φ(S_T(x−h))` is 0 except on the event
`{S_T(x−h) ≤ K < S_T(x+h)}`, which has probability `≈ 2h · ∂_x P(S_T>K) = O(h)`.
On that event the per-path Delta term is `1/(2h)`, so its second moment is
`O(h)/h² = O(1/h)`. For Gamma the per-path term is `O(1/h²)` on an `O(h)`
event, giving `O(1/h³)`. For the call the per-path first difference is
bounded by `S_T/x` (Lipschitz), so its variance is O(1). The second
difference is nonzero only on an `O(h)` event, where it is `O(1/h)`.

**MSE-optimal bump and resulting rate.** Minimising `c₁ h^{2p} + c₂/(N h^q)`,
with `p` the bias order and `q` the variance exponent, gives
`h* ∝ N^{-1/(2p+q)}` and `RMSE* ∝ N^{-p/(2p+q)}`:

| estimator | p | q | RMSE rate |
|---|---|---|---|
| forward, IND | 1 | 2 | N^(−1/4)  (Glynn 1989, cited by FLLLT) |
| central, IND | 2 | 2 | N^(−1/3)  (cited by FLLLT) |
| central, CRN, smooth payoff | 2 | 0 | N^(−1/2) as h → 0 |
| central Δ, CRN, digital | 2 | 1 | N^(−2/5) |
| central Γ, CRN, digital | 2 | 3 | N^(−2/7) |
| central Γ, CRN, call | 2 | 1 | N^(−2/5) |
| central Γ, IND, any | 2 | 4 | N^(−1/4) |

These rates hold only when `h` shrinks with `N` at the optimal speed. In
Experiment B we keep `h` **fixed** at a value tuned in Experiment C. The
RMSE then falls like `N^(−1/2)` while variance dominates and flattens at
`|bias|` once the bias dominates. We report the fitted slopes with this
caveat.

**Exact variance by quadrature.** Every single-stream estimator (Malliavin,
pathwise, CRN-FD) is `mean(f(Z))` for an explicit `f`, so
`N·Var(θ̂) = E[f(Z)²] − E[f(Z)]²`. We compute it by adaptive quadrature,
splitting the integration range at the payoff's jump and kink points
mapped to `Z`-space. For IND-FD it is `Σᵢ cᵢ² Var(D φ(S_T(θᵢ)))/div²`. This
gives a noise-free theoretical `RMSE(N) = √(bias² + var₁/N)`, which the
figures overlay on the simulation results as dotted lines. It is an
independent check that the simulation code is right.

---

## 5. Pathwise (IPA) estimators: when they are valid

The pathwise estimator `∂_θ E[Dφ(S_T(θ))] = E[D φ'(S_T) ∂_θ S_T]` requires
moving the derivative inside the expectation. A sufficient condition is that
`θ ↦ φ(S_T(θ))` is almost surely differentiable and Lipschitz with an
integrable constant (dominated convergence). For the **call**
`φ'(s) = 1{s>K}` almost everywhere, and the condition holds for Delta and Vega:

```
Δ_PW = D E[1{S_T>K} S_T/x],       𝒱_PW = D E[1{S_T>K} S_T (W_T − σT)].
```

**Where it fails, and what we do about it:**

* *Digital payoffs.* `φ' = 0` almost everywhere, so the naive pathwise
  estimator is identically 0. The interchange fails because `φ` jumps; the
  true derivative is a boundary (Dirac) term. The code raises
  `InapplicableEstimatorError`, and the experiments record "inapplicable"
  instead of reporting 0.
* *Call Gamma.* Differentiating `1{S_T>K} S_T/x` again pathwise gives 0
  almost surely (`S_T/x` does not depend on `x`, and `∂_x 1{S_T>K} = 0` almost
  everywhere). The whole Gamma, `D (K/x)² f_{S_T}(K)`, is the missed
  boundary term. This is also marked inapplicable. Mixed pathwise/LR
  estimators fix it but are not pathwise estimators; they are listed as
  future work.

---

## 6. Localisation (FLLLT pp. 409–410)

For the call, split `(s − K)⁺ = G_δ(s) + F_δ(s)` with

```
H_δ(s) = clip((s − (K−δ))/(2δ), 0, 1),   G_δ(s) = ∫ H_δ = { 0, (s−K+δ)²/(4δ), s−K },
I_δ(s) = H_δ'(s) = 1{|s−K|<δ}/(2δ),      F_δ = (s−K)⁺ − G_δ  (zero outside (K−δ, K+δ)).
```

`G_δ` is C¹ with bounded `G_δ''`, so it can be differentiated pathwise.
`F_δ` is bounded with compact support, so the Malliavin weight is applied to
a variable that vanishes when `|W_T|` is large:

```
Δ = D E[H_δ(S_T) S_T/x]              + D E[F_δ(S_T) π_Δ]        (FLLLT)
Γ = D E[I_δ(S_T) (S_T/x)²]           + D E[F_δ(S_T) π_Γ]        (FLLLT)
𝒱 = D E[H_δ(S_T) S_T (W_T − σT)]     + D E[F_δ(S_T) π_𝒱]        (our extension, same decomposition)
```

The Gamma formula uses `∂²_x G_δ(S_T) = I_δ(S_T)(S_T/x)²`, since `∂²_x S_T = 0`.
At the paper's parameters with δ = 10, quadrature gives a per-path variance
of the call-Gamma estimator of **0.00052** (localised) versus **0.0217**
(global): a 42× reduction (`results/tables/A_exact_variances.csv`).

---

## 7. Estimator summary

| Estimator | Greeks | Payoffs | Bias | Per-path variance | Evals/path |
|---|---|---|---|---|---|
| Malliavin weight | Δ, Γ, 𝒱 | any bounded measurable φ (also calls) | 0 | O(1); grows like 1/(σ²T) (Δ), 1/(σ⁴T²) (Γ) | 1 (shared by all Greeks) |
| Localised Malliavin | Δ, Γ, 𝒱 | call | 0 | smaller than global | 1 |
| Pathwise | Δ, 𝒱 | Lipschitz (call) | 0 | O(1) | 1 |
| FD-CRN central/forward | Δ, Γ, 𝒱 | any | O(h²)/O(h) | see §4 | 2 (Δ, 𝒱), 3 (Γ) |
| FD-IND central/forward | Δ, Γ, 𝒱 | any | O(h²)/O(h) | O(1/h²), O(1/h⁴) | 2, 3 |

---

## References

* Black, F., Scholes, M. (1973). The pricing of options and corporate liabilities. *J. Political Economy* 81, 637–654.
* Broadie, M., Glasserman, P. (1996). Estimating security price derivatives using simulation. *Management Science* 42, 269–285.
* Fournié, E., Lasry, J.-M., Lebuchoux, J., Lions, P.-L., Touzi, N. (1999). Applications of Malliavin calculus to Monte Carlo methods in finance. *Finance and Stochastics* 3, 391–412.
* Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering*. Springer.
* Glynn, P. W. (1989). Optimization of stochastic systems via simulation. *Proc. 1989 Winter Simulation Conference*, 90–105.
* L'Ecuyer, P., Perron, G. (1994). On the convergence rates of IPA and FDC derivative estimators. *Operations Research* 42, 643–656.
* Nualart, D. (1995). *The Malliavin Calculus and Related Topics*. Springer.

---
---

# Part II: Discretely monitored Asian digital (phase 2)

> This file is also the `MATHEMATICAL_DERIVATIONS.md` requested for phase 2.
> Windows file names are case-insensitive, so the existing file is extended
> rather than duplicated.

## II.1 Contract, model and simulation

There are m equally spaced dates `t_i = iΔ`, with `Δ = T/m`. The average is
`A = (1/m) Σ_i S_{t_i}` and the payoff is `D·1{A ≥ K}`, with `D = e^{−rT}`.
Each path is driven by i.i.d. `Z_1..Z_m ~ N(0,1)`, with

```
W_i = √Δ (Z_1 + … + Z_i),     S_i = x · exp((r − σ²/2) t_i + σ W_i),     x = S_0.
```

This is the exact joint law of the GBM at the monitoring dates (there is no
discretisation error), and `S_i` is linear in `x`. No closed form exists for
the arithmetic Asian digital.

Notation: `s = σ√Δ` and `c = (r − σ²/2)Δ`.

## II.2 Likelihood ratio = FLLLT weight with a(t) = 1/t₁

**FLLLT Proposition 3.2.** For a path functional `φ(X_{t_1},…,X_{t_m})`,
`∇u(x) = E[φ ∫ a(t) σ⁻¹(X_t) Y_t dW_t]` holds for any `a ∈ Γ_m`, that is,
any `a` with `∫_0^{t_i} a = 1` for **every** i. These constraints force
`∫_{t_1}^{t_i} a = 0` for all i. The simplest admissible choice, and the
one implemented, is `a = 1/t_1` on `[0, t_1]` and 0 elsewhere. In
Black–Scholes `σ⁻¹(X)Y = 1/(σx)`, so the weight is
`W_{t_1}/(xσt_1) = Z_1/(x s)`.

**LR derivation.** The log-increments are independent normals and only the
first one depends on x, through its mean `ln x + c`. The score of
`ln S_1 ~ N(ln x + c, s²)` with respect to x is `Z_1/(x s)`. The two
weights are therefore **the same estimator**.

**Gamma.** `f_xx/f = score² + ∂_x score`. With `∂_x score = −(1 + s Z_1)/(x² s²)`
this gives

```
π_Γ = (Z_1² − 1 − s Z_1) / (x² s²).
```

**Vega.** Every increment `ln(S_i/S_{i−1}) ~ N(c, s²)` depends on σ. Writing
the increment as `c + s Z_i`, each contributes `−1/σ + Z_i²/σ − √Δ Z_i`, so

```
π_𝒱 = Σ_i [ (Z_i² − 1)/σ − √Δ Z_i ].
```

For m = 1 all three reduce to the European weights of Part I (tested).

**Variance growth.** `Var(π_Δ) = 1/(x²σ²Δ) = m/(x²σ²T)`. The Delta weight's
variance grows linearly in m, because a must live on the first monitoring
interval.

## II.3 An anticipating Malliavin Delta weight (u ≡ 1)

FLLLT's Asian example (§4, Remark 3.4) uses a non-adapted direction. Take
`u ≡ 1` on `[0, T]`. Then

```
D_u S_i = σ t_i S_i,
D_u A   = (σ/m) Σ t_i S_i  > 0,
D_u D_u A = (σ²/m) Σ t_i² S_i.
```

Since A is linear in x, `x ∂_x E[φ(A)] = E[φ'(A) A] = E[φ'(A) D_uA · G]` with
`G = A/D_uA`. By the duality `E[⟨Dφ(A), u⟩ G] = E[φ(A) δ(uG)]` and
`δ(uG) = G W_T − D_u G`, where
`D_u G = 1 − A·D_uD_uA/(D_uA)²`. Hence

```
Δ = (D/x) · E[ φ(A) · ( G W_T − 1 + A · D_uD_uA / (D_uA)² ) ].
```

For m = 1 this gives `W_T/(xσT)` (tested). G is smooth with moments of all
orders (`D_uA ≥ (σ t_1/m) Σ S_i`), so the density argument of FLLLT extends
the identity to bounded measurable φ.

This weight is **not** a likelihood-ratio weight of the discrete increments:
here Malliavin calculus produces something new. At m = 2 its unbiasedness
is confirmed against quadrature (`tests/test_asian.py`).

## II.4 Conditioning on the last increment (CPW-last, as in Giles' vibrato MC)

Let `H = mK − Σ_{i<m} S_i`. Then `A ≥ K ⇔ S_m ≥ H`, and
`S_m = S_{m−1} e^{c + s Z_m}` (with `S_0 = x` when m = 1), so

```
P(A ≥ K | F_{t_{m−1}}) = 1                                  if H ≤ 0,
                       = Φ(d),   d = [ln(S_{m−1}/H) + c]/s  if H > 0.
```

**Delta.** All `S_i ∝ x`, so `∂_x d = mK/(sxH)` and

```
Δ_last = D φ(d) · mK/(s x H)    (0 if H ≤ 0).
```

**Gamma.**
`∂²_x d = −mK (H − Σ_{i<m}S_i)/(s (xH)²)` and
`Γ_last = D[φ(d) ∂²_x d − d φ(d) (∂_x d)²]`.

**Vega.** With `N = ln S_{m−1} − ln H + c`,

```
∂_σN = (W_{m−1} − σt_{m−1}) + Σ_{i<m} S_i (W_i − σt_i)/H − σΔ,
∂_σd = ∂_σN/s − d/σ.
```

**Boundary H ↓ 0.** Then `d → +∞`, and each derivative carries
`φ(d)·H^{−k}`. This tends to 0, because
`φ(d) ~ exp(−(ln H)²/(2s²))` beats any power of H. So the conditional price
is C^∞ across H = 0 and its pathwise derivatives exist everywhere. For the
interchange of ∂ and E, take the mean-value theorem plus a dominating bound
`sup_θ φ(d)H^{−k} ≤ C_k(1 + S_{m−1}^{−k})`. This has all moments under the
lognormal law, locally uniformly in θ.

All derivatives are checked against per-path finite differences of the
conditional price (`tests/test_asian.py`).

## II.5 Conditioning on the first increment (CPW-first)

`Z_1` shifts every `W_i` by `√Δ Z_1`, so `S_i = e^{sZ_1} R_i` with

```
R_i = x exp((r − σ²/2) t_i + σ(W_i − √Δ Z_1)),
```

which does not depend on Z_1. Hence `A = e^{sZ_1} Ā` with
`Ā = mean_i R_i > 0`, and

```
P(A ≥ K | Z_2..Z_m) = P(Z_1 ≥ −δ) = Φ(δ),   δ = ln(Ā/K)/s.
```

This is smooth on all of ℝ^{m−1}, with no boundary region.

* `Ā ∝ x`, so `∂_xδ = 1/(xs)` and `∂²_xδ = −1/(x²s)`.
* `Δ_first = D φ(δ)/(xs)`.
* `Γ_first = −D φ(δ)(δ + s)/(xs)²`.
* `∂_σ ln Ā = Σ R_i (W_i − √Δ Z_1 − σt_i) / Σ R_i`.
* `𝒱_first = D φ(δ)[∂_σ ln Ā / s − δ/σ]`.

**Interchange.** `|Δ_first| ≤ D/(xs√(2π))` is bounded, and
`|δφ(δ)| ≤ (2πe)^{−1/2}`. The Vega term is bounded by
`C(1 + max_i|W_i| + σT)`, which is integrable locally uniformly in σ. So
dominated convergence applies.

For m = 1, Ā is deterministic, δ = d₂, and the estimator equals the
closed-form digital Greeks with zero variance (tested).

### II.5.1 CPW-first is the Rao–Blackwellised LR estimator for Δ and Γ

The LR Delta and Gamma weights (II.2) depend on `Z_1` only. Conditioning on
`(Z_2..Z_m)`, with `ζ = −δ` the threshold for `Z_1`:

```
E[1{Z_1 ≥ ζ} Z_1/(xs) | Z_2..Z_m]             = φ(ζ)/(xs)          = Δ_first / D,
E[1{Z_1 ≥ ζ} (Z_1² − 1 − sZ_1)/(x²s²) | …]    = (ζ − s) φ(ζ)/(xs)²  = Γ_first / D,
```

using `∫_ζ^∞ zφ = φ(ζ)` and `∫_ζ^∞ (z² − 1)φ = ζφ(ζ)`.

So **CPW-first Δ and Γ are exactly E[LR | Z_{2:m}]**. By the law of total
variance,

```
Var(LR) = Var(E[LR | Z_{2:m}]) + E[Var(LR | Z_{2:m})] ≥ Var(CPW-first)
```

in every scenario: the variance reduction is guaranteed and equals
`E[Var(LR | Z_{2:m})]`. This was verified numerically to ~1e-9 by
quadrature over Z_1, at m = 3, K = 103, σ = 0.3, T = 0.5, on five random
(Z_2, Z_3).

### II.5.2 The guarantee does not extend

* **Vega.** The LR Vega weight involves every Z_i. `E[LR_𝒱 | Z_{2:m}]`
  differs from 𝒱_first: they are two different unbiased estimators (checked
  numerically, with O(1) differences). Their ordering is an empirical
  question.
* **CPW-last vs LR.** Conditioning on `Z_{1:m−1}` leaves the LR weight
  `Z_1/(xs)` untouched: `E[LR_Δ | Z_{1:m−1}] = Z_1/(xs)·Φ(d)`. That is a
  "conditional LR" estimator, *not* CPW-last, which differentiates Φ(d)
  pathwise. Nothing guarantees that CPW-last beats LR.
* **Price.** Both conditionings Rao–Blackwellise `1{A ≥ K}`, so both reduce
  the variance of the price estimator. CPW-first removes more, because Z_1
  scales A multiplicatively, whereas Z_m enters A only through the last
  term (weight 1/m).

## II.6 Finite differences

* **FD-raw** is the central CRN difference of `D·1{A ≥ K}`. Its bias is O(h²),
  and its per-path variance is O(1/h) for Δ and 𝒱 and O(1/h³) for Γ (Part I §4).
* **FD-smooth** is the same stencil applied to the C^∞ CPW-first conditional
  price. Its bias is O(h²) and its CRN variance stays bounded as h → 0.

Bump rules are pre-specified and scale with `S_0 σ √T`
(`asian_experiments.bump`).

## II.7 Randomised QMC

* Owen-scrambled Sobol points are mapped by Φ⁻¹ to `(Z_1..Z_m)` with
  sequential path construction.
* For CPW-first, the first (best) Sobol coordinates drive `Z_2, Z_3, …`.
* Error bars come from ≥ 8 independent scramblings.
* A Brownian-bridge or PCA construction, which the RQMC literature uses to
  lower effective dimension (Albieri et al. 2025), was **not** implemented.
  Phase 3 (Part III) adds both.

---
---

# Part III: Path constructions for RQMC, wall-clock budgets (phase 3)

## III.1 Keeping the CPW-first conditioning exact

With `t_i = iΔ`, split the Brownian path at the monitoring dates into a
*level* part and a *residual* part:

```
W_{t_i} = √Δ Z_1 + B_i,       B_1 = 0,   B_i = W_{t_i} − W_{t_1}  (i = 2..m).
```

By independent increments, `(B_2,…,B_m)` is a Brownian motion observed at
`τ_k = kΔ` (k = 1..m−1), started at 0 and **independent of Z_1**. Its
covariance is `C_kl = Δ·min(k, l)`.

CPW-first (II.5) needs only

```
R_i = x exp((r − σ²/2) t_i + σ B_i),
Ā = mean_i R_i,
P(A ≥ K | B) = Φ(ln(Ā/K)/(σ√Δ)).
```

So the estimator is a fixed function `ψ(B)`, and the Greek is `E[ψ(B)]`
for **any** construction of B with the right law. Let `ξ ~ N(0, I_{m−1})`
and `B = Lξ`. Then `B ~ N(0, LLᵀ)`, so every L with `LLᵀ = C` gives:

* the same law of B, and hence the same law of `(Z_1, B)`, since Z_1 is not
  touched;
* the same law of W at the dates, with `Cov(W_{t_i}, W_{t_j}) = Δ + C_{i−1,j−1} = min(t_i, t_j)`
  (taking `C_{0,·} = 0`);
* the same conditional identity, hence an **unbiased** estimator for every
  construction.

Under plain MC the three constructions give *identically distributed*
estimators. Under (R)QMC they differ, because Sobol points are not
rotation-invariant: the construction decides which direction of path space
the best Sobol coordinates drive.

**Implementation.** `asian_qmc.increments_from_residual` turns
`B = Lξ` back into the increments that `cpw_first_samples` expects
(`Z_1 := 0`, unused, and `Z_{k+1} = (B_{k+1} − B_k)/√Δ`). The tests check
that the reconstructed residual path equals B to rounding, and that
changing `Z_1` does not change the estimator.

## III.2 The three constructions (established methods)

Let `n = m − 1`.

* **Sequential.** `L = √Δ · T`, with T the lower-triangular all-ones matrix:
  the cumulative sum of i.i.d. increments.
* **Brownian bridge** (Moskowitz & Caflisch 1996; Glasserman 2004 §3.1).
  `B_{τ_n} = √τ_n ξ_1` first. Then, for an interval `(a, b)` with both ends
  known, the midpoint `c = ⌊(a + b)/2⌋` is drawn as

  ```
  B_c = [(τ_b − τ_c) B_a + (τ_c − τ_a) B_b]/(τ_b − τ_a) + √((τ_c − τ_a)(τ_b − τ_c)/(τ_b − τ_a)) ξ_j,
  ```

  breadth-first, with `B_0 = 0` at τ = 0. These are the exact Gaussian
  conditional laws, so `LLᵀ = C`.
* **PCA** (Acworth, Broadie & Glasserman 1998).
  `C = VΛVᵀ`, `L = V Λ^{1/2}`, eigenvalues decreasing. Among all L with
  `LLᵀ = C`, this maximises the variance carried by the first k
  coordinates, for every k.

**Where the variance goes.** These are computed exactly
(`construction_structure.csv`). They give the share of path variance
`tr C` carried by ξ_1. The analogous share of `Var(mean B)`, the leading
linear driver of Ā, is in the same CSV.

| m | sequential | bridge | PCA |
|---|---|---|---|
| 4 | 0.50 | 0.78 | 0.84 |
| 12 | 0.17 | 0.70 | 0.81 |
| 52 | 0.04 | 0.67 | 0.81 |

Sequential construction spreads the important direction across all
coordinates as m grows. That is the mechanism behind phase 2's loss of the
RQMC advantage at m = 52. It is also why dimension reduction *might*
recover it. Whether it does is an empirical question, because the
integrand is not linear in B.

## III.3 Wall-clock budgets: unbiasedness, intervals and cost

**Stopping by time.** A timed estimator keeps adding i.i.d. units until
the elapsed time exceeds the budget. For MC a unit is a block of paths; for
RQMC it is an independent scrambling of a fixed number of points.

The number of units R is a stopping time, but it depends only on the clock,
never on the simulated values. Conditional on R the units are still i.i.d.
with mean θ, so `E[θ̂] = θ`. The minimum is 4 RQMC units or 2 MC blocks.

The clock is affected by system load, so R, and hence the variance, is
random and not exactly reproducible. The estimator's expectation is
unaffected.

**Confidence intervals.**

* **RQMC.** With R scramblings, the interval is `mean ± t_{R−1, 0.975} · sd/√R`.
  Phase 2 used 1.96 here; with R = 8 that gives nominal coverage of about
  0.91 rather than 0.95 under normality.
* **MC.** z = 1.96 on the per-path SE (thousands of paths).

**Points per scrambling.** n_s is the largest power of two such that 10
scramblings fit in the budget under the measured cost model
`t(n) = a + b·n`. Here `a` is the per-scrambling overhead, mostly building
the Owen-scrambled Sobol generator, and `b` is the per-point cost. Both are
fitted per (construction, m) by timing n = 2⁶ and 2¹² (`calibration.csv`).
This choice is a heuristic trade-off: larger n_s exploits the RQMC rate,
and more scramblings give a better SE.

**Time-normalised comparison.**

```
wnMSE = MSE × (mean measured wall time per estimate).
```

For MC this is budget-invariant. For RQMC it is favourable when the rate
exceeds N^{−1/2}, and we report it per budget. Regret is
`√(wnMSE / min over fixed samplers of wnMSE)`: the RMSE ratio at equal
measured time.

## III.4 Budget accounting of the phase-2 selector (audit)

The phase-2 selector:

1. gave each of J candidates `n_0 = max(16, ⌊fB/(J c)⌋)` pilot paths, with
   c the calibrated seconds per path;
2. charged the final run with `B − fB` (the *nominal* pilot).

**Measured on six representative research scenarios** (`m × σ`,
K/S₀ = 1, T = 1):

* **Charged cost.** The calibrated pilot cost Σ n_0 c was ≈ nominal, except
  at m = 52 with B = 2 ms, where the 16-path minimum made it **2.1–2.5×**
  nominal.
* **Measured pilot time.** The pilot's wall time was **6–400× nominal**,
  i.e. **0.6–27× the whole budget B**. The calibration is per path and
  misses per-call overhead (estimator dispatch, plus ≥ 8 Sobol scrambler
  constructions for each RQMC candidate).
* **Fixed runs also missed their nominal budgets.** Median measured/nominal
  was 0.5–2.0 for MC and 0.85–8.3 for RQMC: the RQMC overhead grows with m
  at small budgets.

**Fix (`asian_experiments.adaptive_select`).**

* `accounting="charged"` (default) gives the final run
  `B − Σ n_0 c`, floored at 0.
* The pilot's measured wall time is always recorded.
* `accounting="nominal"` reproduces the stored phase-2 numbers bit for bit;
  it was verified on one scenario at 1e-16.

**Phase 3 avoids the issue by construction:** every estimate, pilot
included, is time-boxed and charged its measured wall time.

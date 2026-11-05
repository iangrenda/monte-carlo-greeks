# Research gaps

This document builds on `LITERATURE_REVIEW.md`. Each direction is classed as
**solved**, **partly open** or **open**, by what the literature we verified
establishes.

## Solved (do not claim as contributions)

| Direction | Why it is solved |
|---|---|
| Malliavin/LR weights for European Greeks; superiority over FD for digitals | FLLLT 1999; Broadie & Glasserman 1996 |
| Optimal (minimum-variance) weight for European Greeks | FLLLT 2001; Benhamou 2003: the likelihood ratio is optimal |
| Localised integration by parts near the singularity | FLLLT 1999 §4 |
| Conditioning on the last step for digital/discontinuous Greeks | Giles 2009 (vibrato); Glasserman & Staum 2001 for barriers |
| Conditional pathwise + QMC for (binary) Asian and barrier Greeks | Zhang & Wang 2020; Albieri et al. 2025; Yu & Wang 2023 (continuous time) |
| Exact conditional smoothing + QMC/sparse grids for baskets | Bayer, Siebenmorgen & Tempone 2018 |
| Unbiased pathwise with explicit boundary terms | Liu & Hong 2011 (kernel); Lakshtanov 2026 preprint (exact, AAD) |
| Efficiency = variance × cost as the comparison criterion | Glynn & Whitt 1992 |

## Partly open

1. **Which conditioning variable to integrate out.** CPW papers condition on
   *some* Gaussian coordinate. The variance consequences of the choice
   (first increment vs last increment vs other directions), and the exact
   Rao–Blackwell relationship between CPW and LR for each Greek, are rarely
   spelled out.
   * Our derivation (II.5.1) shows CPW-first Δ, Γ = E[LR | Z_{2:m}], a
     variance guarantee, and that no such guarantee holds for 𝒱 or for
     last-step conditioning.
   * This is an elementary observation, probably known to specialists. We
     claim no novelty, only an explicit derivation and an empirical
     quantification.
2. **Anticipating Malliavin weights for *discrete* arithmetic Asians vs
   conditioning.** FLLLT and Benhamou give Asian Malliavin weights. How they
   compare at equal computational cost with CPW+RQMC across the number of
   monitoring dates m is partly open. We test one such weight (u ≡ 1).
3. **RQMC gains as a function of m with a smooth conditional integrand under
   sequential construction.** Albieri et al. study effective dimension with
   a Brownian bridge. We quantify how quickly the gain disappears *without*
   it. This is a limitation of our setup, reported as such.

## Open or underexplored (a feasible scope for this project)

4. **Pilot-based selection among Greek estimator families under a fixed
   budget, with pilot cost charged and out-of-sample evaluation.**
   * We found no paper that evaluates this for Greeks.
   * The statistical basis exists: two-stage ranking and selection, and
     Glynn–Whitt efficiency.
   * Open questions:
     * Does a deployable rule beat the best fixed estimator?
     * How large is the pilot penalty?
     * How does selection error grow with the number of candidates and
       shrink with the budget?
     * Does a rule trained on some contracts transfer to others (e.g. to
       larger m)?
   * **This is the primary question of phase 2.**
5. **Adaptive localisation width (FLLLT §4).** δ is a tuning parameter. A
   pilot rule for δ is straightforward, but FLLLT 2001 and Benhamou already
   treat weight optimality. Novelty is unclear without the full texts.
6. **Joint estimation of Δ, Γ, 𝒱 from one simulation.** This needs the
   covariance across Greeks and a budget split. It is underexplored for
   discontinuous path-dependent payoffs, and it is the natural follow-up to
   item 4.
7. **Learned control variates for conditionally smoothed estimators.** This
   needs expectation-corrected control variates for unbiasedness. A strong
   simple baseline exists: the geometric-Asian digital, which has closed-form
   Greeks. Feasible, but secondary.
8. **Boundary-corrected AD (Lakshtanov 2026) vs CPW.** For GBM Asian
   digitals the boundary point in Z_1 is analytic, so the boundary
   correction and CPW-first integrate the same direction. The relationship
   (local boundary term vs global conditional expectation) is unexplored in
   what we read. It would be a mathematically clean comparison, but the
   preprint is not yet peer reviewed.

## What this project can reasonably investigate

* **Primary.** Item 4, together with items 1–3 as the estimator-level
  substrate, on the discretely monitored Asian digital.
* **Secondary, if time allows.** Item 6 (multi-Greek allocation) or item 8
  (boundary term vs conditioning).
* **Out of scope now.** Item 7 (ML control variates), and full AAD
  frameworks.

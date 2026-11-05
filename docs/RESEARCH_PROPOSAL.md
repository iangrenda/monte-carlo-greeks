# Research proposal: adaptive Greek estimation for a discontinuous path-dependent option

## 1. Where things stand

| Layer | Content | Source |
|---|---|---|
| **Original paper replication** | Malliavin weights for European Δ, Γ, 𝒱; the interval-digital example; localisation for calls; the N = 10 000 benchmark table | FLLLT 1999, reproduced in phase 1 (`docs/research_report.md`) |
| **Established literature** | Malliavin = LR for terminal payoffs and LR is variance-optimal (Benhamou 2003); last-step conditioning for digitals (Giles 2009); conditional pathwise + QMC for binary Asians (Zhang & Wang 2020; Albieri et al. 2025); efficiency = variance × cost (Glynn & Whitt 1992) | `LITERATURE_REVIEW.md` |
| **Our implementation (verified)** | Phase 1 European study. Phase 2: discretely monitored Asian digital with six estimator families (LR/Malliavin, an anticipating Malliavin Δ weight, CPW-last, CPW-first, FD-raw, FD-smooth), each with MC or RQMC; m = 1 reductions, m = 2 quadrature references, per-path derivative checks | `src/mc_greeks/asian.py`, `tests/test_asian.py` |
| **Our proposed extension** | A pilot-based selector that chooses an estimator per (contract, Greek, budget) by predicted MSE, with pilot cost charged. Evaluated out of sample against every fixed method, heuristics, the best fixed method chosen on training contracts, and an oracle | `asian_experiments.py`, `scripts/asian_study.py` |
| **Empirical findings** | Reported only from executed runs | `RESEARCH_FINDINGS.md` |
| **Novelty** | See §5. Modest, and empirical rather than methodological | — |

## 2. Candidate extensions (Phase 8 assessment)

Scores use a 1–5 scale. For *difficulty* and *risk*, higher means harder or
riskier.

| | A. Adaptive Malliavin localisation | B. Conditional smoothing + learned control variates | C. Boundary-corrected AD for digitals | D. Joint multi-Greek allocation | **E. Pilot-based estimator selection with conditional smoothing** |
|---|---|---|---|---|---|
| Question | Choose δ by pilot | Regression/NN CVs on CPW estimators | Pathwise + boundary term vs CPW | Split budget across Δ, Γ, 𝒱 from shared paths | Does a deployable selector beat the best fixed estimator out of sample? |
| Math foundation | FLLLT §4; weight optimality | Conditional MC; CV theory (needs exact expectation of the CV) | Leibniz rule along a Gaussian direction; Lakshtanov 2026 | Covariance of joint estimators; multi-objective allocation | Two-stage ranking & selection; Glynn–Whitt efficiency; Rao–Blackwell (II.5.1) |
| Closest literature | FLLLT 2001; Benhamou 2003 | Capriotti 2008; geometric-Asian CVs (classical) | Lakshtanov 2026; Liu & Hong 2011 | Little found | Nothing found for Greeks; generic adaptive MC (Lapeyre & Lelong) |
| Already known | Optimal weights (European) | CVs for prices: classical | Unbiased boundary terms | — | CPW/RQMC superiority in many settings |
| Contribution | Small (call only; European) | Medium, if unbiased and robust | Medium: clean comparison, GBM only | Medium | Medium: an empirical study with negative results possible |
| Difficulty | 2 | 4 | 3 | 3 | 3 |
| Compute cost | low | medium | low | medium | medium (~25 min) |
| Main risks | low novelty (European, analytic) | leakage or bias in learned CVs; ML adds complexity without gain | preprint only; little beyond CPW for GBM | needs E first | the selector may simply lose to "always CPW-first/RQMC" |
| Validation | closed forms | references with CIs | m = 2 quadrature | references | quadrature (m = 2) + RQMC references with CIs; train/test split |
| Value for quant finance | low | medium | medium–high (AAD shops) | medium | medium: tells practitioners whether adaptivity is worth it |
| **Rank** | 5 | 4 | 2 | 3 | **1** |

**Chosen: E.** It is the preferred direction of the brief. The audit and
literature found no reason to override it. The estimators it needs (CPW,
RQMC, Malliavin) are themselves known, so the contribution is the
**selection study**, not a new estimator. C ranks second as a follow-up:
for GBM Asians its boundary point is analytic, so it could be compared
cheaply with CPW-first.

## 3. Hypotheses (falsifiable; all treated as open until tested)

* **H1.** Conditional smoothing lowers the equal-budget MSE of Asian-digital
  Greek estimators relative to the non-smoothed LR/Malliavin and FD-raw
  estimators.
  * Refined after the derivations: guaranteed for CPW-first Δ and Γ
    against LR (II.5.1).
  * Empirical for CPW-first 𝒱, for CPW-last, and against FD and the
    anticipating Malliavin weight.
* **H2.** The best estimator (oracle) changes across m, moneyness, σ, T, the
  budget and the Greek.
* **H3.** A pilot-based adaptive selector has lower out-of-sample RMSE at
  equal total budget (pilot included) than the best fixed estimator chosen
  on training contracts.
* **H4.** RQMC adds a further benefit on top of conditional smoothing, at
  least for some m.
* **H5.** The selector loses when the pilot is too small (small budget,
  many candidates, large m) or when candidates have similar efficiency.

## 4. Design summary

Details are in `EXPERIMENTAL_METHODOLOGY.md`.

* **Scenarios.** 36: m ∈ {4, 12, 52}, K/S₀ ∈ {0.9, 1, 1.1},
  σ ∈ {0.1, 0.3}, T ∈ {0.25, 1}, with r = 0.05 and S₀ = 100.
* **Budgets.** 2 ms and 20 ms of calibrated compute per estimate.
* **Replications.** 40 per scenario × strategy × Greek × budget.
* **References.** CPW-first + RQMC with 2^15 × 32 points, and an independent
  CPW-last MC cross-check with 2^20 paths.
* **Splits.** A random half of the scenarios for training and half for
  test; plus an extrapolation split (train m ∈ {4, 12}, test m = 52).

## 5. What would be new, honestly

* **New in our results (to our knowledge, after about 20 targeted
  searches).** An out-of-sample, cost-charged evaluation of pilot-based
  selection among Greek estimator families, including its failure modes.
  * A negative result, "a fixed CPW-first+RQMC rule is as good or better",
    would be equally informative.
* **Explicit but probably known.** The Rao–Blackwell identity
  CPW-first = E[LR | Z_{2:m}] for Δ and Γ, and the absence of an ordering for
  𝒱 and for last-step conditioning.
* **Not new.** CPW itself, RQMC with smoothing, the Malliavin Asian weights,
  and the LR = Malliavin equivalence.
* **Bar for publication.** None of this is publication-level on its own.
  It would need at least:
  * a richer model (Heston or local volatility) where conditioning is not
    exact;
  * a Brownian-bridge/PCA RQMC baseline;
  * comparison with boundary-corrected AD;
  * a theoretical analysis of the selection risk.

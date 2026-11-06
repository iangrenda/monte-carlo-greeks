# Phase 4, Part A: revalidation of the repository

**State audited.** Branch `main` at merge commit `872a224` (PR #1 merged
2026-11-05T19:14Z), on Python 3.12.10 with numpy 2.4.3, scipy 1.16.2 and
pandas 2.3.3. The machine is Windows 11, Intel x86-64, 16 logical cores, and
all runs are single-threaded. No code or result file was modified before
these checks.

Status key:

* ✅ passed
* ⚠️ passed with a caveat
* ❌ failed / corrected
* ❓ uncertain

## 1. Test suite

`python -m pytest -o addopts="" -W default` gave **221 passed in 10.3 s** ✅.
After phase-4 additions the count rises (see `PHASE4_COST_RULE_VALIDATION.md`
and the Greek-aware study).

## 2. Phase-3 constructions and conditioning identity

✅ **LLᵀ = C.** Recomputed independently for m ∈ {4, 12, 52, 104}, including
the new m = 104. Max |LLᵀ − C| was 0 to 1.5e-14 for the sequential, bridge
and PCA constructions.

✅ **Conditioning identity.** The tests cover reconstruction of B from the
increments, invariance to Z₁, path law by sampling, and unbiasedness
against quadrature at m = 2 and against large MC at m = 12. All pass.

## 3. Definitions

| Quantity | Implementation | Check |
|---|---|---|
| MSE | mean of (estimate − reference)² over replications | ✅ recomputed |
| Corrected MSE | max(MSE − SE_ref², 1e-300) | ✅. The floor was never reached (0 of 864 rows). SE_ref²/MSE has median 9e-7 and max 0.12, so the reference noise is negligible except in a few PCA cells at m = 4. |
| Time-normalised MSE | corrected MSE × mean measured wall time | ✅. Valid as budget-free only if MSE ∝ 1/time (MC). For RQMC it depends on the budget, which is why results are reported per budget. |
| Regret | √(wnMSE / min over the four fixed samplers of wnMSE) | ✅ recomputed, max abs error 2.4e-12 |
| Confidence intervals | MC: 1.96 × SE. RQMC: t₀.₉₇₅,R−1 × sd/√R over R independent scramblings | ✅ max abs error 9e-16 against the raw data |
| Reference uncertainty | SE across 32 independent scramblings of the PCA-RQMC reference | ✅. Cross-check: 144 plain-MC z-scores, mean −0.03, sd 1.01 |

## 4. Pilot accounting (`asian_experiments.adaptive_select`)

The code was reviewed.

✅ **What `accounting="charged"` charges.** It charges Σ n_path · c over
every pilot run actually executed, including FD's 2h runs, and floors the
remaining budget at 0. The pilot wall time is recorded, and `"nominal"`
reproduces phase 2 bit for bit.

⚠️ **Caveat 1: residual under-charging.** "Charged" still uses the per-path
calibration, which omits per-call overhead. That is why the phase-3
revalidation compared strategies on *measured* wall time, which is the
authoritative metric.

⚠️ **Caveat 2: the final run always does at least 16 paths.** Even after an
exhausted pilot, the final run executes ≥ 16 paths, a small uncharged
overrun. It is visible in the recorded `total_wall`.

## 5. Phase-3 results vs saved data

✅ **Q1 table recomputed.** I recomputed `Q1_wnmse_ratio_vs_mc.csv` from
`fixed_raw.csv.gz` + `references.csv` with standalone code that shares
nothing with the analysis script. All 72 rows agree with the saved table to
a relative difference of ≤ 4e-13.

✅ **Oracle and regret.** The oracle and regret columns of `summary.csv` agree
to 2e-12.

## 6. Caching, matrix construction, scrambling, timing noise

| Effect | Measurement | Consequence |
|---|---|---|
| Building L (cold) | seq < 0.02 ms. BB 1.5 ms (m = 52), 6.5 ms (m = 104). PCA 1.3 ms (m = 52), 2.7 ms (m = 104) | Phase 3 cached L by (construction, m, dt). The first replication per scenario built it inside the timed region. ⚠️ No measurable effect on the phase-3 results: rep-0 wall / other reps = 0.97–1.05, because time-boxing absorbs it by running fewer scramblings. It was not separately reported, though. Phase 4 measures cold vs warm explicitly. |
| Sobol scrambler construction | 0.21 ms (m = 4), 0.36 ms (m = 12), 1.4 ms (m = 52), 2.8 ms (m = 104) per scrambling | Dominates the per-scrambling overhead `a`. It is charged inside every timed estimate. It is SciPy-specific: a cheaper randomisation (e.g. a digital shift) would move the cost frontier. |
| Overruns | Phase 3 reported medians only (1.01–1.09). The **95th percentile of wall/B at 5 ms is 1.46** for RQMC (1.17 MC). 1.7% of all phase-3 estimates exceeded 1.5 × B (max 2.12 ×). | ⚠️ Under-reported in phase 3. The time-normalised metrics charge it, but the text understated it. Phase 4 reports p95 and the share above 1.10 × B. |
| Calibration noise | Repeated pooled calibrations of the same quantity differ by tens of % (e.g. Sobol init at m = 52: 0.83 ms in the phase-3 calibration vs 1.43 ms measured here) | ❓ The rule threshold 10(a + 64b) inherits this noise. Phase 4 calibrates fresh at the start of the run, as the frozen rule specifies, and reports it. |

## 7. Documentation reconciliation

❌ → ✅ **Derivations Part III.3** still described the per-construction
two-point calibration fit. It now describes the pooled least-squares
implementation, notes the superseded first pass (`*_v1`), and lists the
uncharged setup costs.

## 8. Remaining uncertainties

❓ **Machine dependence.** Timing results depend on the machine (single
Windows laptop, background load uncontrolled). All conclusions involving
wall time are machine-specific. Ratios are more portable than absolute
times.

❓ **Oracle bias.** The oracle is a minimum over noisy estimates (30
replications in phase 3, 20 in phase 4), so regret is biased slightly
upward for every non-oracle strategy.

❓ **No CI in phase 1–3.** Tests were run only locally before phase 4. A
GitHub Actions workflow is added in phase 4.

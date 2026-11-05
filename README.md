# Monte Carlo Greeks

**A comparative study of finite differences, pathwise differentiation and
Malliavin-calculus estimators for option sensitivities.**

Monte Carlo prices are easy, and Monte Carlo *Greeks* are not: bump-and-reprice
estimators trade truncation bias against simulation noise, and they break
down for discontinuous payoffs such as digital options. Fournié, Lasry,
Lebuchoux, Lions & Touzi (1999) showed that Greeks can be written as
`E[payoff × weight]`, where the weight comes from integration by parts on
Wiener space, does not depend on the payoff, and never requires
differentiating it.

This repository implements those estimators in the Black–Scholes model,
replicates the paper's numerical section, and runs a controlled statistical
comparison against finite differences (common vs independent random numbers)
and pathwise derivatives. It covers several payoffs, sample sizes, bump
sizes, volatilities, strikes and maturities, at equal computational budgets.
Every simulated quantity is checked against closed-form values, and every
estimator's variance against an exact quadrature value.

📄 **[Research report](docs/research_report.md)** · 🧮 **[Mathematical derivations](docs/mathematical_derivations.md)** · 🧪 **[Experimental design](docs/experimental_design.md)**

> **Note.** For terminal European payoffs in Black–Scholes, the Malliavin
> weights are *identical* to the likelihood-ratio (score) weights
> ([derivations §2.5](docs/mathematical_derivations.md)). "Malliavin weight"
> in the phase-1 results below means "Malliavin = LR weight".

## Phase 2: path-dependent digital Greeks and adaptive estimator selection

📋 [Audit](docs/PROJECT_AUDIT.md) · 📚 [Literature review](docs/LITERATURE_REVIEW.md) · 🧭 [Research gaps](docs/RESEARCH_GAPS.md) · 📝 [Proposal](docs/RESEARCH_PROPOSAL.md) · 🧪 [Methodology](docs/EXPERIMENTAL_METHODOLOGY.md) · 📊 **[Findings](docs/RESEARCH_FINDINGS.md)**

**Problem.** The discretely monitored Asian digital `e^{−rT}·1{(1/m)Σ S_{t_i} ≥ K}`
has no closed form. Phase 2 compares six estimator families on it, each
with plain MC or scrambled-Sobol RQMC:

* likelihood ratio (equal here to the FLLLT weight);
* an anticipating Malliavin Δ weight;
* conditional pathwise, integrating out the last increment;
* conditional pathwise, integrating out the first increment;
* bump-and-revalue on the raw payoff;
* bump-and-revalue on the smoothed payoff.

On top of these sits a pilot-based selector that picks an estimator per
contract under a fixed compute budget. It is evaluated out of sample,
against fixed rules and an oracle.

**Main results** (36 + 12 scenarios, m ∈ {4, 12, 52}; details and CIs in the
findings):

* **Rao–Blackwell guarantee.** Integrating out the *first* Gaussian increment
  gives a smooth, closed-form conditional estimator that is exactly the
  Rao–Blackwellised likelihood-ratio estimator for Δ and Γ. Its variance
  is therefore never higher, and in practice it was 5–3·10⁵× lower.
* **Last-step conditioning ("vibrato"-style) is a poor smoother for
  averages.** It was worse than plain LR in 46% (Δ) and 90% (Γ) of settings.
* **RQMC on top of smoothing.** Up to ~850× lower MSE at m = 4, but it loses
  at m = 52 with sequential construction (no Brownian bridge implemented).
* **A pilot-based adaptive selector did *not* beat a well-chosen fixed
  estimator.**
  * The naive version failed catastrophically, with up to 500× the oracle
    RMSE. Rare-event estimators produced zero-variance pilots and were
    selected.
  * A variance guard and a restricted candidate set make selection safe
    (about as good as the best fixed rule) but not reliably better.
* The idea of conditional smoothing itself is prior art (Giles 2009;
  Zhang & Wang 2020); the contribution here is the controlled,
  cost-charged evaluation and its failure analysis.

```bash
cd scripts
```

```bash
python asian_study.py --config quick
```

```bash
python asian_study.py --config research
```

```bash
python asian_study.py --config confirm
```

The research run takes about 40 minutes and the confirmation run about 5.
Outputs go to `results/asian/<config>/`, and phase-1 results are untouched.

---

## Key findings

All results come from 200 independent replications per setting.

| | Interval digital `1{100≤S_T≤110}` | Cash digital `1{S_T>K}` | European call |
|---|---|---|---|
| Most accurate | **Malliavin weight**, every Greek and setting tested | Malliavin weight in most settings | **Localised Malliavin** (54/57 settings), then pathwise ≈ CRN-FD |
| RMSE of best FD ÷ Malliavin, equal budget of 150k payoff evaluations | Δ 20.5×, Γ 18×, 𝒱 4.5× | Δ 2.3×, Γ 6.9×, 𝒱 1.7× | Δ 0.49×, Γ 0.38×, 𝒱 0.39× (FD better) |
| Where Malliavin loses | — | low σ (Δ, 𝒱), deep ITM (Δ), short T (𝒱) | everywhere vs pathwise, unless localised |

1. **Replication.** The paper's interval-digital benchmarks are reproduced
   (Δ = −0.00133486, Γ = −0.00038876, 𝒱 = −0.77751371). Its
   wrong-signed finite-difference Gamma (+0.0995) is consistent with a
   very small bump. The paper's localisation cuts call-Gamma variance 42×.
2. **Convergence.** Unbiased estimators converge like N^−1/2 (fitted slopes
   −0.48 to −0.51). Fixed-bump finite differences flatten at their bias,
   and their nominal 95% confidence intervals under-cover (down to 0%).
   With scrambled Sobol points the Malliavin estimators reach ≈N^−1.
3. **CRN** turns finite differences on the smooth call into the pathwise
   estimator (over 1 000× lower RMSE than independent draws at tiny bumps).
   For digitals the RMSE-optimal bump is large (5–10% of S₀), and CRN and
   IND end up similar.
4. **Hypothesis check.** "Malliavin is better for discontinuous payoffs" is
   strongly true for the interval digital and only conditionally true for
   the cash digital. For smooth payoffs the *global* weight is 2–3.5× worse
   than pathwise or CRN-FD; the paper's localisation fixes this.
5. **Cost.** A weighted estimate costs ≈1.5–2.0× a CRN-FD estimate in
   wall-clock time at equal workload. On digitals it is still 1.8–300× more
   efficient (1/(MSE·time)).

<p align="center">
<img src="results/figures/fig04_rmse_vs_n.png" width="88%"><br>
<em>Relative RMSE vs N. Dotted lines are the exact theoretical RMSE from quadrature.</em>
</p>
<p align="center">
<img src="results/figures/fig06_bump_size_rmse.png" width="88%"><br>
<em>Finite-difference RMSE vs bump size: CRN vs independent random numbers.</em>
</p>
<p align="center">
<img src="results/figures/fig10_smooth_vs_discontinuous_equal_budget.png" width="88%"><br>
<em>RMSE relative to the Malliavin estimator at an equal budget (log scale).</em>
</p>

## Model and estimators

Black–Scholes `dS = rS dt + σS dW`. `S_T = S₀ exp((r−σ²/2)T + σ√T Z)` is
sampled exactly, with no Euler discretisation. With `D = e^{−rT}` and
`W_T = √T Z`:

| Estimator | Per-path term | Valid for |
|---|---|---|
| Malliavin (paper §4) | `D φ(S_T) · W_T/(S₀σT)` (Δ)<br>`D φ(S_T) · (W_T²/(σT) − W_T − 1/σ)` (𝒱)<br>𝒱-weight `/(S₀²σT)` (Γ) | any bounded payoff |
| Localised Malliavin (paper §4) | pathwise on a smoothed kink + weight on a compact remainder | call |
| Pathwise | `D φ'(S_T) ∂S_T/∂θ` | Lipschitz payoffs, Δ and 𝒱 only |
| Finite differences | forward/central stencils, CRN or independent normals | any (biased) |

Invalid combinations are refused with `InapplicableEstimatorError`, never
reported as a misleading zero. Examples are a pathwise digital Delta and a
pathwise call Gamma.

**What comes from the paper, and what is ours.** The weights, the interval
digital experiment and the localisation are from Fournié et al. (1999).
The rest is this project: the finite-difference and pathwise baselines,
the exact-variance quadrature, the localised Vega, Experiments B–G and all
conclusions. In this one-dimensional European setting the Malliavin
weights coincide with Broadie–Glasserman likelihood-ratio weights; see
[derivations §2.5](docs/mathematical_derivations.md).

## Installation

```bash
git clone <this-repo> monte-carlo-greeks
```

```bash
cd monte-carlo-greeks
```

```bash
python -m venv .venv
```

Activate the venv (`source .venv/bin/activate`, or `.venv\Scripts\activate` on Windows), then:

```bash
pip install -r requirements.txt
```

```bash
pip install -e .
```

Requires Python ≥ 3.10. The direct dependencies are NumPy, SciPy, pandas and
Matplotlib; pytest is used for the tests.

## Reproducing the results

The experiment scripts run from the `scripts/` folder:

```bash
cd scripts
```

```bash
python full_experiment.py --config quick
```

```bash
python full_experiment.py --config research
```

The `quick` run takes about 30 s and is for demonstration. The `research`
run (≈5 min on one core) produces everything in `results/` and the report.
Individual studies:

```bash
python bump_size_study.py --config research
```

```bash
python reproduce_paper.py --config research
```

Run `bump_size_study.py` first, because it tunes the FD bumps that the
other studies use. The others are `convergence_study.py`,
`volatility_study.py`, `moneyness_study.py`, `maturity_study.py` and
`payoff_comparison.py`. Add `--figures-only` to any script to re-plot from
saved results. Seeds are fixed per experiment (`src/mc_greeks/config.py`),
so reruns give identical numbers on the same software stack.

Using the library directly:

```python
import numpy as np
from mc_greeks import PAPER_PARAMS, PAPER_PAYOFF, analytical, estimate

Z = np.random.default_rng(0).standard_normal(100_000)
est = estimate("Malliavin", "gamma", PAPER_PAYOFF, PAPER_PARAMS, Z)
print(est.value, "+/-", est.stderr, " exact:", analytical.greek(PAPER_PAYOFF, PAPER_PARAMS, "gamma"))
```

## Testing

```bash
pytest
```

175 tests cover:

* closed forms against high-order numerical derivatives;
* the Malliavin and pathwise weights against quadrature, to 10⁻⁷, with no
  Monte Carlo noise;
* Monte Carlo convergence at 4.5-SE tolerances, plus a test that a wrong
  weight is detected at >20 SE;
* common random numbers actually being shared, and independent draws
  actually being independent;
* the finite-difference bias orders and the O(1/h) CRN variance law;
* seed reproducibility, input validation, and pathwise estimators refused
  where invalid.

## Repository structure

```
src/mc_greeks/
  models.py            Black-Scholes parameters (validated dataclass)
  simulation.py        exact S_T sampling, seeded generators, scrambled Sobol
  payoffs.py           call, cash digital, interval digital (extensible)
  analytical.py        closed-form prices and Greeks
  malliavin.py         Malliavin weights, localised call estimator
  pathwise.py          pathwise estimators (validity enforced)
  finite_difference.py FD stencils, CRN and independent estimators, exact FD bias
  estimators.py        unified interface, workload accounting, blocked evaluation, exact moments
  statistics.py        replication statistics, coverage, log-log slopes, paired tests
  experiments.py       Experiments A-G
  asian.py             Asian digital: exact path simulation, LR/Malliavin, CPW-first/last, FD, RQMC
  asian_experiments.py phase-2 study: cost calibration, references, fixed runs, adaptive selectors
  visualization.py     figures
  config.py            quick / research configurations, seeds, paper table
scripts/               one script per experiment + full_experiment.py + figures.py
notebooks/             01 foundations, 02 paper replication, 03 comparison, 04 analysis
tests/                 pytest suite
results/               figures/ (PNG+PDF), tables/ (CSV), experiment_data/ (raw replications, metadata JSON)
docs/                  derivations, experimental design, research report
```

## Limitations

* Black–Scholes and one-dimensional European payoffs only. This is the
  setting where Malliavin weights equal likelihood-ratio weights, so it
  does not test the path-dependent cases that motivate Malliavin calculus.
* Finite-difference bumps are tuned once (at the baseline) and then held
  fixed as a rule. That is realistic, but it is not the best possible FD
  in every configuration, especially at short maturities.
* Timings are for one machine and a NumPy implementation.

## Future extensions

* Localisation for digital payoffs.
* Antithetic variates and control variates.
* Asian and barrier options with genuine Malliavin weights.
* Adaptive bump selection.
* Mixed pathwise/likelihood-ratio call Gamma.
* Heston and multi-asset models.
* Budget allocation across Greeks.

## References

* Fournié, E., Lasry, J.-M., Lebuchoux, J., Lions, P.-L., Touzi, N. (1999). Applications of Malliavin calculus to Monte Carlo methods in finance. *Finance and Stochastics* 3, 391–412.
* Broadie, M., Glasserman, P. (1996). Estimating security price derivatives using simulation. *Management Science* 42, 269–285.
* Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering*. Springer.
* Glynn, P. W. (1989). Optimization of stochastic systems via simulation. *Proc. Winter Simulation Conference*, 90–105.

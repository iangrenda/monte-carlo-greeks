"""Generate the four tutorial notebooks in notebooks/ (run, then execute with nbconvert).

The notebooks are thin: they call the package and read results/ so they never
diverge from the scripts.
"""
from pathlib import Path

import nbformat as nbf

NB = Path(__file__).resolve().parents[1] / "notebooks"
SETUP = """import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from mc_greeks import *
from mc_greeks.estimators import exact_estimator_moments, per_path_function
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
TABLES = ROOT / "results" / "tables"
pd.set_option("display.float_format", lambda v: f"{v:.6g}")"""


def nb(cells):
    n = nbf.v4.new_notebook()
    n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    n.cells = [nbf.v4.new_markdown_cell(c[1]) if c[0] == "md" else nbf.v4.new_code_cell(c[1]) for c in cells]
    return n


notebooks = {
    "01_mathematical_foundations.ipynb": [
        ("md", "# 01 · Mathematical foundations\n\nGaussian integration by parts turns a derivative of "
               "`E[φ(S_T)]` into `E[φ(S_T)·π]`. Here we check the weights *without simulation*, by quadrature, "
               "and look at why they work for an indicator payoff. Derivations: `docs/mathematical_derivations.md`."),
        ("code", SETUP),
        ("md", "## Exact check: E[D φ(S_T) π] equals the closed-form Greek\n`exact_estimator_moments` integrates "
               "the per-path term against the normal density, splitting at the payoff's jumps."),
        ("code", "p, f = PAPER_PARAMS, PAPER_PAYOFF\nrows = []\nfor g in ['delta', 'gamma', 'vega']:\n"
                 "    m = exact_estimator_moments('Malliavin', g, f, p)\n"
                 "    rows.append({'greek': g, 'E[D phi pi] (quadrature)': m['mean'], 'analytical': m['truth'],\n"
                 "                 'per-path variance': m['var1']})\npd.DataFrame(rows)"),
        ("md", "## The weight does not depend on the payoff\nThe same `π_Δ = W_T/(xσT)` gives the Delta of a call, "
               "a digital and the interval digital:"),
        ("code", "for pay in [EuropeanCall(100), CashDigitalCall(100), PAPER_PAYOFF]:\n"
                 "    m = exact_estimator_moments('Malliavin', 'delta', pay, p)\n"
                 "    print(f\"{pay.name:9s} quadrature {m['mean']:+.8f}   analytical {m['truth']:+.8f}\")"),
        ("md", "## Why finite differences struggle with a jump\nPer-path CRN central difference for the digital: "
               "it is 0 except on an O(h) event where it equals 1/(2h). N·Var therefore grows like 1/h."),
        ("code", "d = CashDigitalCall(100)\nfor h in [2.0, 0.5, 0.1, 0.02]:\n"
                 "    m = exact_estimator_moments('FD-CRN-ctr', 'delta', d, p, h=h)\n"
                 "    print(f'h={h:5}:  bias={m[\"bias\"]:+.2e}   N*Var={m[\"var1\"]:.3e}   h*N*Var={h*m[\"var1\"]:.3e}')"),
        ("md", "The last column is roughly constant, confirming the O(1/h) variance law."),
    ],
    "02_paper_replication.ipynb": [
        ("md", "# 02 · Replicating Fournié et al. (1999), Section 4\n\nThe paper used low-discrepancy sequences; we "
               "show scrambled Sobol and pseudorandom runs. Regenerate with `python scripts/reproduce_paper.py`."),
        ("code", SETUP),
        ("code", "t = pd.read_csv(TABLES / 'A_paper_table.csv')\n"
                 "t[t.sampler.str.startswith('Sobol')][['payoff','greek','method','h','exact','paper_exact','paper_fd',"
                 "'paper_malliavin','first_run','rmse']]"),
        ("md", "Running estimates (paper Figs. 1–3):"),
        ("code", "from IPython.display import Image, display\nfor g in ['delta','gamma','vega']:\n"
                 "    display(Image(str(ROOT / f'results/figures/figA{[\"delta\",\"gamma\",\"vega\"].index(g)+1}_paper_interval_{g}_trajectory.png')))"),
        ("code", "pd.read_csv(TABLES / 'A_rqmc_slopes.csv')[['greek','sampler','slope','slope_lo','slope_hi']]"),
    ],
    "03_estimator_comparison.ipynb": [
        ("md", "# 03 · Estimator comparison: convergence and bump size\n\nExperiments B and C."),
        ("code", SETUP),
        ("code", "s = pd.read_csv(TABLES / 'B_convergence_summary.csv')\n"
                 "s[(s.n == 50000) & (s.greek != 'price')][['payoff','greek','method','h','bias','sd','rmse','rel_rmse',"
                 "'coverage','runtime']]"),
        ("code", "pd.read_csv(TABLES / 'B_convergence_slopes.csv')[['payoff','greek','method','slope','slope_lo','slope_hi',"
                 "'slope_theory_on_grid']]"),
        ("code", "pd.read_csv(TABLES / 'C_optimal_bumps.csv')"),
        ("code", "from IPython.display import Image\nImage(str(ROOT / 'results/figures/fig06_bump_size_rmse.png'))"),
    ],
    "04_research_analysis.ipynb": [
        ("md", "# 04 · Research analysis: volatility, moneyness, maturity, payoff type\n\nExperiments D–G."),
        ("code", SETUP),
        ("code", "g = pd.read_csv(TABLES / 'G_equal_budget_summary.csv')\n"
                 "g[['payoff','greek','method','n','rmse','rmse_ratio_vs_malliavin','time_ratio_vs_malliavin','coverage']]"),
        ("code", "from IPython.display import Image, display\nfor f in ['fig07_variance_vs_volatility','fig08_rmse_vs_moneyness',"
                 "'fig09_rmse_vs_maturity','fig10_smooth_vs_discontinuous_equal_budget']:\n"
                 "    display(Image(str(ROOT / f'results/figures/{f}.png')))"),
        ("md", "Exact scaling of the Malliavin weight variance with σ and T (no simulation):"),
        ("code", "rows=[]\nfor T in [0.05, 0.25, 1.0]:\n  for s in [0.1, 0.2, 0.4]:\n"
                 "    q = BlackScholesParams(100, 0.1, s, T)\n"
                 "    m = exact_estimator_moments('Malliavin', 'delta', CashDigitalCall(100), q)\n"
                 "    rows.append({'T':T,'sigma':s,'delta':m['truth'],'sd per path':m['var1']**0.5,"
                 "'relative sd per path':m['var1']**0.5/abs(m['truth'])})\npd.DataFrame(rows)"),
        ("md", "For the K = S0 digital the *relative* per-path standard deviation of the Malliavin Delta stays "
               "between about 1.5 and 2.9 over this grid, although the weight itself scales like 1/(σ√T): the true "
               "Delta grows at the same rate. (The equivalent ratio for Gamma varies much more, because the "
               "digital Gamma is proportional to d1 and passes through zero.)"),
    ],
}

if __name__ == "__main__":
    NB.mkdir(exist_ok=True)
    for name, cells in notebooks.items():
        nbf.write(nb(cells), NB / name)
        print("wrote", NB / name)

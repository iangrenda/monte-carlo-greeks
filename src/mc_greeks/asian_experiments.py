"""Asian-digital study: cost calibration, references, fixed-estimator runs and a
pilot-based adaptive estimator selector. See docs/EXPERIMENTAL_METHODOLOGY.md.

Budget accounting. Every strategy receives the same budget B, in seconds of
*calibrated* compute. c_j = seconds per path for candidate j (method,
sampler, Greek, m), measured once by ``calibrate_costs`` on this machine and
treated as known: a practitioner would calibrate once per platform. Fixed
strategies use N_j = B / c_j paths. The adaptive strategy spends a pilot
fraction of B, and its final run uses the remaining budget. Actual
wall-clock times are recorded as well, to check the calibration.
"""

from __future__ import annotations

import itertools
import json
import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .asian import AsianDigital, asian_estimate, check_applicable, cpw_first_samples, cpw_last_samples, per_path, sobol_normal_blocks
from .config import RESULTS
from .models import BlackScholesParams
from .payoffs import InapplicableEstimatorError
from .simulation import make_generator
from .statistics import replication_summary

ASIAN_DIR = RESULTS / "asian"
GREEKS3 = ("delta", "gamma", "vega")
CANDIDATES = [  # (method, sampler)
    ("LR", "MC"), ("Malliavin-avg", "MC"), ("CPW-last", "MC"), ("CPW-first", "MC"),
    ("FD-raw", "MC"), ("FD-smooth", "MC"),
    ("LR", "RQMC"), ("CPW-last", "RQMC"), ("CPW-first", "RQMC"), ("FD-smooth", "RQMC"),
]
#: Reduced candidate set for the "adaptive-lite" variant. Specified after the quick
#: development run (sigma = 0.2 scenarios, disjoint from the research grid's sigma values).
LITE_CANDIDATES = [("CPW-first", "MC"), ("CPW-first", "RQMC"), ("FD-smooth", "RQMC")]
ADAPTIVE_VARIANTS = {  # name: (candidate list, pilot fraction or None = config value, min nonzero pilot terms)
    "adaptive (pilot)": (None, None, 0),
    "adaptive-lite (3 cands, 5% pilot)": (LITE_CANDIDATES, 0.05, 0),
    # Added AFTER the research run, from its failure analysis (degenerate pilots with zero
    # sample variance). Evaluated only on the fresh "confirm" grid; see RESEARCH_FINDINGS.md.
    "adaptive-guarded (all cands, >=10 nonzero)": (None, None, 10),
}
DEFAULT_FALLBACK = ("CPW-first", "MC")
HEURISTICS = {  # fixed rules a practitioner might adopt without data
    "always CPW-first/RQMC": ("CPW-first", "RQMC"),
    "always CPW-first/MC": ("CPW-first", "MC"),
    "always LR/MC (Malliavin)": ("LR", "MC"),
    "always FD-raw/MC (bump & revalue)": ("FD-raw", "MC"),
}


def label(method: str, sampler: str) -> str:
    return f"{method}/{sampler}"


def bump(method: str, greek: str, p: BlackScholesParams) -> float | None:
    """Pre-specified bump rules (not tuned on these experiments).

    Scaled with the spread of S_T (S0 sigma sqrt T) so they adapt to sigma and T.
    The coefficients for FD-raw reproduce, at sigma = 0.2 and T = 1, the
    RMSE-optimal digital bumps of the European study (5% Delta, 10% Gamma,
    0.02 vol). FD-smooth acts on a smooth function, so it uses smaller bumps.
    """
    if not method.startswith("FD"):
        return None
    width = p.S0 * p.sigma * math.sqrt(p.T)
    if method == "FD-raw":
        return {"delta": 0.25 * width, "gamma": 0.5 * width, "vega": 0.1 * p.sigma}[greek]
    return {"delta": 0.05 * width, "gamma": 0.1 * width, "vega": 0.05 * p.sigma}[greek]


def applicable(method: str, greek: str) -> bool:
    try:
        check_applicable(method, greek)
        return True
    except InapplicableEstimatorError:
        return False


@dataclass(frozen=True)
class Scenario:
    sid: int
    m: int
    moneyness: float
    sigma: float
    T: float
    S0: float = 100.0
    r: float = 0.05

    @property
    def params(self) -> BlackScholesParams:
        return BlackScholesParams(self.S0, self.r, self.sigma, self.T)

    @property
    def payoff(self) -> AsianDigital:
        return AsianDigital(self.moneyness * self.S0, self.m)


@dataclass(frozen=True)
class AsianConfig:
    name: str
    ms: tuple = (4, 12, 52)
    moneyness: tuple = (0.9, 1.0, 1.1)
    sigmas: tuple = (0.1, 0.3)
    Ts: tuple = (0.25, 1.0)
    budgets: tuple = (0.002, 0.02)          # seconds of calibrated compute per estimate
    reps: int = 40
    pilot_fraction: float = 0.10
    ref_points: int = 2**15                 # Sobol points per scramble for references
    ref_scrambles: int = 32
    check_paths: int = 2**20                # independent CPW-last MC cross-check
    seed: int = 2026_1104
    train_fraction: float = 0.5

    def scenarios(self) -> list[Scenario]:
        grid = itertools.product(self.ms, self.moneyness, self.sigmas, self.Ts)
        return [Scenario(i, *g) for i, g in enumerate(grid)]

    def to_dict(self):
        return asdict(self)


ASIAN_QUICK = AsianConfig(name="quick", ms=(4, 12), sigmas=(0.2,), Ts=(1.0,), budgets=(0.004,), reps=8,
                          ref_points=2**13, ref_scrambles=16, check_paths=2**17)
ASIAN_RESEARCH = AsianConfig(name="research")
#: Fresh confirmation grid: sigma, K and T values used in neither the quick nor the research
#: grid, new seed. Run after the research analysis to test the guarded selector out of sample.
ASIAN_CONFIRM = AsianConfig(name="confirm", ms=(4, 12, 52), moneyness=(0.95, 1.05), sigmas=(0.15, 0.25), Ts=(0.5,),
                            budgets=(0.005,), reps=30, seed=2026_1105)
ASIAN_CONFIGS = {"quick": ASIAN_QUICK, "research": ASIAN_RESEARCH, "confirm": ASIAN_CONFIRM}


# --- cost calibration --------------------------------------------------------------------

def calibrate_costs(ms, n: int = 2**14, repeats: int = 5, seed: int = 0) -> pd.DataFrame:
    """Median seconds per path for each (method, sampler, greek, m)."""
    rows = []
    rng = make_generator(seed)
    p = BlackScholesParams(100.0, 0.05, 0.2, 1.0)
    for m in ms:
        f = AsianDigital(100.0, m)
        for (meth, smp), g in itertools.product(CANDIDATES, GREEKS3):
            if not applicable(meth, g):
                continue
            ts = []
            for _ in range(repeats + 1):
                e = asian_estimate(meth, g, f, p, n, rng, smp, h=bump(meth, g, p))
                ts.append(e.runtime / e.n_paths)
            rows.append({"m": m, "method": meth, "sampler": smp, "greek": g, "sec_per_path": float(np.median(ts[1:]))})
    return pd.DataFrame(rows)


class CostModel:
    def __init__(self, df: pd.DataFrame):
        self.c = {(r.m, r.method, r.sampler, r.greek): r.sec_per_path for r in df.itertuples()}

    def __call__(self, m, method, sampler, greek) -> float:
        return self.c[(m, method, sampler, greek)]

    def paths(self, budget, m, method, sampler, greek) -> int:
        return max(16, int(budget / self(m, method, sampler, greek)))


# --- references --------------------------------------------------------------------------

def reference(sc: Scenario, cfg: AsianConfig, seed) -> dict:
    """High-accuracy reference: CPW-first with randomised QMC (independent scramblings).

    All Greeks are computed from the same scrambled points; each Greek's
    uncertainty is the SE over scramblings. Cross-check: CPW-last with plain MC
    (a different estimator and sampler), reported as a z-score.
    """
    p, f = sc.params, sc.payoff
    rng = make_generator(seed)
    means = {g: [] for g in ("price",) + GREEKS3}
    for _ in range(cfg.ref_scrambles):
        acc = {g: [0.0, 0] for g in means}
        for Zb in sobol_normal_blocks(cfg.ref_points, f.m, rng):
            Zb = np.roll(Zb, 1, axis=1)  # best Sobol coordinates go to Z_2, Z_3, ...
            for g in means:
                y = cpw_first_samples(f, p, Zb, g)
                acc[g][0] += y.sum()
                acc[g][1] += y.size
        for g in means:
            means[g].append(acc[g][0] / acc[g][1])
    out = {"sid": sc.sid}
    for g, v in means.items():
        v = np.array(v)
        out[f"ref_{g}"] = v.mean()
        out[f"ref_se_{g}"] = v.std(ddof=1) / math.sqrt(len(v))
    chk_rng = make_generator([seed, 1])
    for g in GREEKS3:
        e = asian_estimate("CPW-last", g, f, p, cfg.check_paths, chk_rng, "MC")
        out[f"check_{g}"] = e.value
        out[f"check_z_{g}"] = (e.value - out[f"ref_{g}"]) / math.hypot(e.stderr, out[f"ref_se_{g}"])
    return out


# --- fixed-estimator runs ------------------------------------------------------------------

def run_fixed(sc: Scenario, cost: CostModel, budget: float, reps: int, seed) -> pd.DataFrame:
    rows = []
    p, f = sc.params, sc.payoff
    for (meth, smp), g in itertools.product(CANDIDATES, GREEKS3):
        if not applicable(meth, g):
            continue
        n = cost.paths(budget, sc.m, meth, smp, g)
        h = bump(meth, g, p)
        for r in range(reps):
            rng = make_generator([*seed, CANDIDATES.index((meth, smp)), GREEKS3.index(g), r])
            e = asian_estimate(meth, g, f, p, n, rng, smp, h=h)
            rows.append({"sid": sc.sid, "budget": budget, "greek": g, "method": meth, "sampler": smp,
                         "strategy": label(meth, smp), "rep": r, "value": e.value, "stderr": e.stderr,
                         "n_paths": e.n_paths, "runtime": e.runtime, "h": h})
    return pd.DataFrame(rows)


# --- adaptive selection --------------------------------------------------------------------

def adaptive_select(sc: Scenario, greek: str, cost: CostModel, budget: float, pilot_fraction: float,
                    rng: np.random.Generator, candidates=None, min_nonzero: int = 0,
                    accounting: str = "charged") -> tuple[tuple[str, str], dict, float, dict]:
    """Pilot every applicable candidate on an equal share of the pilot budget and
    return the candidate with the smallest predicted MSE at the remaining budget.

    Predicted MSE at N paths:
      * unbiased MC estimators: s^2 / N, with s^2 the pilot per-path variance;
      * RQMC estimators: SE_pilot^2 * n_pilot / N. This assumes the MC rate N^-1 for
        the variance, which is conservative for RQMC (the true rate may be faster);
      * finite differences: bias_hat^2 + s^2 / N, with bias_hat = (theta(2h) - theta(h)) / 3
        from CRN pilot estimates (the O(h^2) Richardson estimate; it is noisy, and the
        noise inflates the prediction, i.e. errs against FD).
    Pilot cost (calibrated) is charged to the budget, including FD's extra 2h runs.

    Guard (``min_nonzero`` > 0): a candidate whose pilot produced fewer than
    ``min_nonzero`` nonzero per-path terms is excluded, because its sample variance
    is uninformative (often exactly 0 for rare-event estimators). If every
    candidate is excluded, DEFAULT_FALLBACK is used.

    Budget accounting (phase-3 audit, docs/RESEARCH_FINDINGS.md section 8):
      * ``accounting="nominal"`` (phase-2 behaviour, kept only to reproduce phase-2
        numbers): the final run receives budget - pilot_fraction * budget, even when
        the 16-path minimum makes the pilot cost more than its nominal share.
      * ``accounting="charged"`` (default): the final run receives
        budget - sum(n_pilot_paths * c), the calibrated cost of the pilot actually run,
        floored at 0.
    In both modes the returned ``info`` reports the nominal, charged and measured
    wall-clock pilot cost. Calibration is per path and omits per-call overheads
    (e.g. constructing a Sobol scrambler), so the wall-clock cost can be far larger.
    """
    if accounting not in ("nominal", "charged"):
        raise ValueError("accounting must be 'nominal' or 'charged'")
    p, f = sc.params, sc.payoff
    cands = [(m_, s_) for (m_, s_) in (candidates or CANDIDATES) if applicable(m_, greek)]
    pilot_budget = pilot_fraction * budget
    share = pilot_budget / len(cands)
    runs = []  # (candidate, estimate) for every pilot run, in order
    t_start = time.perf_counter()
    for meth, smp in cands:
        h = bump(meth, greek, p)
        c = cost(sc.m, meth, smp, greek)
        if meth.startswith("FD"):
            n0 = max(16, int(share / (2 * c)))  # two bump sizes share the slot
            e1 = asian_estimate(meth, greek, f, p, n0, rng, smp, h=h)
            e2 = asian_estimate(meth, greek, f, p, n0, rng, smp, h=2 * h)
            runs.append(((meth, smp), c, [e1, e2]))
        else:
            n0 = max(16, int(share / c))
            e = asian_estimate(meth, greek, f, p, n0, rng, smp, h=h)
            runs.append(((meth, smp), c, [e]))
    pilot_wall = time.perf_counter() - t_start
    charged = sum(e.n_paths * c for _, c, es in runs for e in es)
    remaining = budget - pilot_budget if accounting == "nominal" else max(budget - charged, 0.0)
    preds = {}
    for (meth, smp), c, es in runs:
        N = max(16, int(remaining / c))
        if meth.startswith("FD"):
            e1, e2 = es
            bias = (e2.value - e1.value) / 3.0
            var_n = e1.stderr**2 * e1.n_paths
            preds[(meth, smp)] = bias**2 + var_n / N if e1.n_nonzero >= min_nonzero else math.inf
        else:
            e = es[0]
            preds[(meth, smp)] = e.stderr**2 * e.n_paths / N if e.n_nonzero >= min_nonzero else math.inf
    best = min(preds, key=preds.get)
    if math.isinf(preds[best]):
        best = DEFAULT_FALLBACK
        preds.setdefault(best, math.inf)
    info = {"pilot_nominal": pilot_budget, "pilot_charged": charged, "pilot_wall": pilot_wall,
            "pilot_overshoot": charged > pilot_budget * 1.0001, "remaining": remaining, "accounting": accounting}
    return best, preds, remaining, info


def run_adaptive(sc: Scenario, cost: CostModel, budget: float, reps: int, pilot_fraction: float, seed,
                 variants=tuple(ADAPTIVE_VARIANTS), accounting: str = "nominal") -> pd.DataFrame:
    """Adaptive runs. ``accounting="nominal"`` reproduces the stored phase-2 results;
    use "charged" for corrected budgets. Pilot wall time is recorded either way."""
    rows = []
    p, f = sc.params, sc.payoff
    for v_i, (name, (cands, frac, min_nz)) in enumerate(ADAPTIVE_VARIANTS.items()):
      if name not in variants:
          continue
      for g in GREEKS3:
        for r in range(reps):
            pilot_rng = make_generator([*seed, v_i, 0, GREEKS3.index(g), r])
            (meth, smp), preds, remaining, info = adaptive_select(sc, g, cost, budget, frac or pilot_fraction, pilot_rng,
                                                                 cands, min_nz, accounting)
            eval_rng = make_generator([*seed, v_i, 1, GREEKS3.index(g), r])  # fresh, independent of the pilot
            n = cost.paths(remaining, sc.m, meth, smp, g)
            e = asian_estimate(meth, g, f, p, n, eval_rng, smp, h=bump(meth, g, p))
            rows.append({"sid": sc.sid, "budget": budget, "greek": g, "strategy": name,
                         "chosen": label(meth, smp), "rep": r, "value": e.value, "stderr": e.stderr,
                         "n_paths": e.n_paths, "runtime": e.runtime,
                         "pred_mse_chosen": preds[(meth, smp)], "pilot_wall": info["pilot_wall"],
                         "pilot_charged": info["pilot_charged"], "pilot_nominal": info["pilot_nominal"],
                         "total_wall": info["pilot_wall"] + e.runtime, "accounting": accounting})
    return pd.DataFrame(rows)


# --- evaluation ----------------------------------------------------------------------------

def summarize_runs(raw: pd.DataFrame, refs: pd.DataFrame) -> pd.DataFrame:
    """RMSE etc. per (sid, budget, greek, strategy), against the reference.

    The reference has its own error; we report the measured MSE and a
    corrected MSE = measured - ref_se^2 (the reference error is independent of
    the runs), plus the ratio ref_se / RMSE so the reader can see when the
    reference is not negligible.
    """
    out = []
    refs = refs.set_index("sid")
    for (sid, b, g, strat), d in raw.groupby(["sid", "budget", "greek", "strategy"], sort=False):
        truth, rse = refs.loc[sid, f"ref_{g}"], refs.loc[sid, f"ref_se_{g}"]
        s = replication_summary(d["value"], d["stderr"], truth, d["runtime"])
        s.update({"sid": sid, "budget": b, "greek": g, "strategy": strat, "ref_se": rse,
                  "mse_corrected": max(s["mse"] - rse**2, 0.0), "ref_se_over_rmse": rse / s["rmse"] if s["rmse"] > 0 else np.nan,
                  "mean_paths": d["n_paths"].mean()})
        out.append(s)
    return pd.DataFrame(out)

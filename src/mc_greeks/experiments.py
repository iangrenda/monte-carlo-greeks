"""Experiment drivers A-G.

Design (see docs/experimental_design.md):

* One replication = one base vector Z of N standard normals drawn from an
  independent child stream. *All* estimators in that replication are
  evaluated on the same Z (independent-random-number FD additionally draws
  its extra vectors from the same child stream). Estimators are therefore
  paired across methods, which sharpens method-vs-method comparisons,
  while replications are mutually independent.
* For every (method, Greek, bump) we also compute the estimator's exact
  bias and per-path variance by quadrature (``exact_estimator_moments``),
  giving a noise-free theoretical RMSE(N) = sqrt(bias^2 + var1/N) to compare
  with the empirical one.
* Runtime of an estimate = time to draw the normals it consumes (pro-rated
  share of the base draw + its own extra draws) + compute time. Workload is
  counted in payoff evaluations.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from . import analytical
from .config import (ALL_METHODS, DATA, DEFAULT_REL_BUMP, DEFAULT_VOL_BUMP, FD_METHODS, GREEKS, PAPER_TABLE,
                     SWEEP_METHODS, TABLES, ExperimentConfig)
from .estimators import (check_applicable, estimate, exact_estimator_moments, payoff_evals_per_path,
                         per_path_function)
from .models import PAPER_PARAMS, BlackScholesParams
from .payoffs import CashDigitalCall, EuropeanCall, InapplicableEstimatorError, IntervalDigital, Payoff
from .simulation import make_generator, pseudo_normals, sobol_normals
from .statistics import loglog_slope, paired_mse_comparison, replication_summary

BumpFn = Callable[[str, str, BlackScholesParams], float]


def paper_payoffs(K: float = 100.0) -> dict[str, Payoff]:
    return {"interval": IntervalDigital(100.0, 110.0), "digital": CashDigitalCall(K), "call": EuropeanCall(K)}


@dataclass(frozen=True)
class Spec:
    method: str
    greek: str
    h: float | None = None  # absolute bump (FD only)


def default_bumps(method: str, greek: str, p: BlackScholesParams) -> float:
    return DEFAULT_VOL_BUMP if greek == "vega" else DEFAULT_REL_BUMP * p.S0


def tuned_bump_fn(table: dict | None) -> BumpFn:
    """Bump rule from the tuned table {payoff: {method: {greek: h}}}.

    Delta/Gamma bumps are stored relative to S0, Vega bumps in vol units.
    Vega bumps are capped at sigma/2 so central differences stay valid when
    sigma is small (Experiment D).
    """
    def make(payoff_name: str) -> BumpFn:
        def f(method: str, greek: str, p: BlackScholesParams) -> float:
            if table and payoff_name in table and method in table[payoff_name] and greek in table[payoff_name][method]:
                h = table[payoff_name][method][greek]
            else:
                h = DEFAULT_VOL_BUMP if greek == "vega" else DEFAULT_REL_BUMP
            return min(h, 0.5 * p.sigma) if greek == "vega" else h * p.S0
        return f
    return make


def build_specs(payoff: Payoff, p: BlackScholesParams, greeks: Iterable[str], methods: Iterable[str],
                bump: BumpFn | None = None) -> tuple[list[Spec], list[dict]]:
    """Applicable specs, plus a record of the inapplicable combinations."""
    bump = bump or default_bumps
    specs, skipped = [], []
    for g in greeks:
        for m in methods:
            try:
                check_applicable(m, g, payoff)
            except InapplicableEstimatorError as exc:
                skipped.append({"payoff": payoff.name, "method": m, "greek": g, "reason": str(exc)})
                continue
            specs.append(Spec(m, g, bump(m, g, p) if m in FD_METHODS else None))
    return specs, skipped


def exact_for(spec: Spec, payoff: Payoff, p: BlackScholesParams, loc_width: float) -> dict:
    try:
        return exact_estimator_moments(spec.method, spec.greek, payoff, p, h=spec.h, loc_width=loc_width)
    except Exception as exc:  # quadrature failure must not kill an experiment; record it
        return {"mean": np.nan, "bias": np.nan, "var1": np.nan, "truth": analytical.greek(payoff, p, spec.greek),
                "error": repr(exc)}


def run_replications(payoff: Payoff, p: BlackScholesParams, specs: list[Spec], n_values: Iterable[int], reps: int,
                     seed, loc_width: float = 10.0, budget: int | None = None, extra: dict | None = None,
                     with_exact: bool = True) -> pd.DataFrame:
    """Raw replication results, one row per (n, replication, spec).

    If ``budget`` is given, each spec uses N = budget // (payoff evals per
    path) paths (an equal-computational-budget comparison); ``n_values`` is
    then ignored. ``seed`` may be an int or a list of ints (SeedSequence entropy).
    """
    extra = extra or {}
    exact = {s: exact_for(s, payoff, p, loc_width) for s in specs} if with_exact else {}
    n_values = [budget] if budget is not None else list(n_values)
    root = np.random.SeedSequence(seed)
    rows = []
    for n, child in zip(n_values, root.spawn(len(n_values))):
        n_spec = {s: (budget // payoff_evals_per_path(s.method, s.greek, p, s.h) if budget else n) for s in specs}
        n_max = max(n_spec.values())
        for r, rep_seed in enumerate(child.spawn(reps)):
            rng = make_generator(rep_seed)
            t0 = time.perf_counter()
            Z = rng.standard_normal(n_max)
            t_rng = time.perf_counter() - t0
            for s in specs:
                ns = n_spec[s]
                e = estimate(s.method, s.greek, payoff, p, Z[:ns], rng=rng, h=s.h, loc_width=loc_width)
                ex = exact.get(s, {})
                rows.append({
                    "payoff": payoff.name, "method": s.method, "greek": s.greek, "h": s.h, "n": ns,
                    "rep": r, "value": e.value, "stderr": e.stderr, "payoff_evals": e.payoff_evals,
                    "runtime": e.runtime + t_rng * ns / n_max,
                    "truth": analytical.greek(payoff, p, s.greek),
                    "exact_bias": ex.get("bias", np.nan), "exact_var1": ex.get("var1", np.nan),
                    "S0": p.S0, "r": p.r, "sigma": p.sigma, "T": p.T, **extra,
                })
    return pd.DataFrame(rows)


def summarize(raw: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Replication statistics per group, plus exact (quadrature) theory columns."""
    out = []
    for key, g in raw.groupby(by, dropna=False, sort=False):
        s = replication_summary(g["value"], g["stderr"], g["truth"].iloc[0], g["runtime"], g["payoff_evals"])
        row = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        row.update(s)
        n = g["n"].iloc[0]
        b, v = g["exact_bias"].iloc[0], g["exact_var1"].iloc[0]
        row.update({"exact_bias": b, "exact_sd": np.sqrt(v / n), "rmse_theory": np.sqrt(b**2 + v / n),
                    "var_ratio_emp_over_exact": s["variance"] / (v / n) if v > 0 else np.nan})
        out.append(row)
    return pd.DataFrame(out)


def paired_vs_reference(raw: pd.DataFrame, by: list[str], reference: str = "Malliavin") -> pd.DataFrame:
    """Paired MSE comparison of every method against ``reference`` within groups."""
    rows = []
    for key, g in raw.groupby(by, dropna=False, sort=False):
        ref = g[g["method"] == reference].sort_values("rep")
        if ref.empty:
            continue
        for m, gm in g.groupby("method"):
            if m == reference:
                continue
            gm = gm.sort_values("rep")
            res = paired_mse_comparison(gm["value"].to_numpy(), ref["value"].to_numpy(), g["truth"].iloc[0])
            rows.append({**dict(zip(by, key if isinstance(key, tuple) else (key,))), "method": m,
                         "reference": reference, **res})
    return pd.DataFrame(rows)


# --- persistence --------------------------------------------------------------------

def environment_info() -> dict:
    import matplotlib
    import scipy
    return {"python": platform.python_version(), "platform": platform.platform(), "processor": platform.processor(),
            "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}


def save(df: pd.DataFrame, name: str, raw: bool = False) -> Path:
    folder = DATA if raw else TABLES
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (f"{name}.csv.gz" if raw else f"{name}.csv")
    df.to_csv(path, index=False)
    return path


def save_meta(name: str, cfg: ExperimentConfig, **info) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    meta = {"experiment": name, "config": cfg.to_dict(), "environment": environment_info(), **info}
    (DATA / f"{name}_meta.json").write_text(json.dumps(meta, indent=2, default=str))


TUNED_PATH = TABLES / "tuned_bumps.json"


def load_tuned_bumps() -> dict | None:
    return json.loads(TUNED_PATH.read_text()) if TUNED_PATH.exists() else None


# --- Experiment A: paper replication ------------------------------------------------

def experiment_a(cfg: ExperimentConfig) -> dict[str, pd.DataFrame]:
    """Replicate Section 4: interval digital Greeks, the N = 10 000 table, and
    the running-estimate figures; plus RQMC-vs-MC and localisation extensions."""
    p, seed = PAPER_PARAMS, cfg.seeds["A_paper"]
    pays = {"interval": IntervalDigital(100, 110), "call": EuropeanCall(100)}
    n_seq = 2 ** int(np.ceil(np.log2(cfg.paper_n)))

    # A1: running estimates along one scrambled Sobol sequence and one pseudorandom stream
    traj = []
    zq = sobol_normals(n_seq, seed=seed)
    zp = pseudo_normals(make_generator([seed, 1]), n_seq)
    idx = np.unique(np.geomspace(50, n_seq, 400).astype(int))
    for pname, f in pays.items():
        methods = [("Malliavin", None), ("FD-CRN-ctr", "fd")]
        if pname == "call":
            methods.append(("Malliavin-loc", None))
        for g in GREEKS:
            for m, kind in methods:
                h = default_bumps(m, g, p) if kind else None
                fn = per_path_function(m, g, f, p, h=h, loc_width=cfg.loc_width)
                for sampler, z in (("Sobol (scrambled)", zq), ("pseudorandom", zp)):
                    run = np.cumsum(fn(z)) / np.arange(1, n_seq + 1)
                    traj.append(pd.DataFrame({"payoff": pname, "greek": g, "method": m, "sampler": sampler,
                                              "n": idx, "estimate": run[idx - 1],
                                              "truth": analytical.greek(f, p, g)}))
    traj = pd.concat(traj, ignore_index=True)

    # A2: the N = 10 000 table (single runs, as in the paper) + replication RMSE
    table_rows = []
    fd_hs = {"delta": (1.0, 0.1), "gamma": (1.0, 0.1), "vega": (0.01, 0.001)}
    for pname, f in pays.items():
        for g in GREEKS:
            exact = analytical.greek(f, p, g)
            pe, pfd, pml = PAPER_TABLE[(g, pname)]
            mall = "Malliavin-loc" if pname == "call" else "Malliavin"
            cands = [(mall, None), ("Malliavin", None)] if pname == "call" else [(mall, None)]
            cands += [("FD-CRN-ctr", h) for h in fd_hs[g]]
            for m, h in cands:
                for sampler in ("Sobol (scrambled)", "pseudorandom"):
                    vals = []
                    for r in range(cfg.paper_reps):
                        z = (sobol_normals(cfg.paper_n, seed=make_generator([seed, 7, r])) if sampler.startswith("Sobol")
                             else pseudo_normals(make_generator([seed, 8, r]), cfg.paper_n))
                        vals.append(estimate(m, g, f, p, z, h=h, loc_width=cfg.loc_width).value)
                    vals = np.array(vals)
                    table_rows.append({
                        "payoff": pname, "greek": g, "method": m, "h": h, "sampler": sampler, "n": cfg.paper_n,
                        "exact": exact, "paper_exact": pe, "paper_fd": pfd, "paper_malliavin": pml,
                        "first_run": vals[0], "mean": vals.mean(), "sd": vals.std(ddof=1),
                        "rmse": np.sqrt(np.mean((vals - exact) ** 2)), "replications": len(vals)})
    table = pd.DataFrame(table_rows)

    # A3 (extension): RQMC vs MC RMSE for the Malliavin estimators of the paper's digital
    f = pays["interval"]
    rq = []
    for k in cfg.rqmc_log2n:
        n = 2**k
        for g in GREEKS:
            fn = per_path_function("Malliavin", g, f, p)
            for sampler in ("Sobol (scrambled)", "pseudorandom"):
                vals = []
                for r in range(cfg.paper_reps):
                    z = (sobol_normals(n, seed=make_generator([seed, 100 + k, r])) if sampler.startswith("Sobol")
                         else pseudo_normals(make_generator([seed, 200 + k, r]), n))
                    vals.append(fn(z).mean())
                vals = np.array(vals)
                truth = analytical.greek(f, p, g)
                rq.append({"greek": g, "sampler": sampler, "n": n, "rmse": np.sqrt(np.mean((vals - truth) ** 2)),
                           "bias": vals.mean() - truth, "sd": vals.std(ddof=1), "truth": truth})
    rq = pd.DataFrame(rq)
    slopes = []
    for (g, s), d in rq.groupby(["greek", "sampler"]):
        fit = loglog_slope(d["n"], d["rmse"])
        slopes.append({"greek": g, "sampler": s, **fit})
    slopes = pd.DataFrame(slopes)

    # A4: exact per-path variance of the estimators in the paper's table (no noise)
    var_rows = []
    for pname, f in pays.items():
        for g in GREEKS:
            for m in (["Malliavin", "Malliavin-loc", "Pathwise"] if pname == "call" else ["Malliavin"]):
                try:
                    ex = exact_estimator_moments(m, g, f, p, loc_width=cfg.loc_width)
                except InapplicableEstimatorError:
                    continue
                var_rows.append({"payoff": pname, "greek": g, "method": m, "var1": ex["var1"],
                                 "sd_at_paper_n": np.sqrt(ex["var1"] / cfg.paper_n), "truth": ex["truth"]})
    varx = pd.DataFrame(var_rows)

    save(traj, "A_paper_trajectories", raw=True)
    save(table, "A_paper_table")
    save(rq, "A_rqmc_vs_mc")
    save(slopes, "A_rqmc_slopes")
    save(varx, "A_exact_variances")
    save_meta("A_paper", cfg)
    return {"trajectories": traj, "table": table, "rqmc": rq, "rqmc_slopes": slopes, "exact_var": varx}


# --- Experiment C: bump size (run before B, D-G; it produces the tuned bumps) ---------

def experiment_c(cfg: ExperimentConfig) -> dict[str, pd.DataFrame]:
    p, seed = PAPER_PARAMS, cfg.seeds["C_bump"]
    raws = []
    for i, (pname, f) in enumerate(paper_payoffs().items()):
        for j, g in enumerate(GREEKS):
            grid = cfg.vol_bumps if g == "vega" else cfg.rel_bumps
            for k, hrel in enumerate(grid):
                h = hrel if g == "vega" else hrel * p.S0
                specs = [Spec(m, g, h) for m in FD_METHODS]
                raw = run_replications(f, p, specs, [cfg.bump_n], cfg.bump_reps, [seed, i, j, k],
                                       extra={"h_rel": hrel})
                raws.append(raw)
    raw = pd.concat(raws, ignore_index=True)
    summ = summarize(raw, ["payoff", "greek", "method", "h_rel", "h", "n"])
    # choose h* by minimum empirical RMSE (tuning data only), record the theory optimum too
    best = {}
    rows = []
    for (pname, g, m), d in summ.groupby(["payoff", "greek", "method"]):
        emp = d.loc[d["rmse"].idxmin()]
        th = d.loc[d["rmse_theory"].idxmin()]
        best.setdefault(pname, {}).setdefault(m, {})[g] = float(emp["h_rel"])
        rows.append({"payoff": pname, "greek": g, "method": m, "h_rel_empirical": emp["h_rel"],
                     "rmse_empirical": emp["rmse"], "h_rel_theory": th["h_rel"], "rmse_theory_at_opt": th["rmse_theory"]})
    TUNED_PATH.parent.mkdir(parents=True, exist_ok=True)
    TUNED_PATH.write_text(json.dumps(best, indent=2))
    opt = pd.DataFrame(rows)
    save(raw, "C_bump_raw", raw=True)
    save(summ, "C_bump_summary")
    save(opt, "C_optimal_bumps")
    save_meta("C_bump", cfg, note="h_rel is h/S0 for delta/gamma and absolute vol bump for vega")
    return {"raw": raw, "summary": summ, "optimal": opt}


# --- Experiment B: convergence -------------------------------------------------------

def experiment_b(cfg: ExperimentConfig) -> dict[str, pd.DataFrame]:
    p, seed = PAPER_PARAMS, cfg.seeds["B_convergence"]
    bumps = tuned_bump_fn(load_tuned_bumps())
    raws, skipped = [], []
    for i, (pname, f) in enumerate(paper_payoffs().items()):
        specs, sk = build_specs(f, p, ("price",) + GREEKS, ("MC",) + ALL_METHODS, bumps(pname))
        skipped += sk
        raws.append(run_replications(f, p, specs, cfg.sample_sizes, cfg.replications, [seed, i],
                                     loc_width=cfg.loc_width))
    raw = pd.concat(raws, ignore_index=True)
    summ = summarize(raw, ["payoff", "greek", "method", "h", "n"])
    slopes = []
    for (pname, g, m), d in summ.groupby(["payoff", "greek", "method"]):
        fit = loglog_slope(d["n"], d["rmse"])
        fit_th = loglog_slope(d["n"], d["rmse_theory"]) if d["rmse_theory"].notna().all() else {"slope": np.nan}
        slopes.append({"payoff": pname, "greek": g, "method": m, "h": d["h"].iloc[0], **fit,
                       "slope_theory_on_grid": fit_th["slope"]})
    slopes = pd.DataFrame(slopes)
    paired = paired_vs_reference(raw, ["payoff", "greek", "n"])
    save(raw, "B_convergence_raw", raw=True)
    save(summ, "B_convergence_summary")
    save(slopes, "B_convergence_slopes")
    save(paired, "B_paired_vs_malliavin")
    save(pd.DataFrame(skipped).drop_duplicates(), "inapplicable_estimators")
    save_meta("B_convergence", cfg, bumps=load_tuned_bumps())
    return {"raw": raw, "summary": summ, "slopes": slopes, "paired": paired}


# --- Experiments D, E, F: parameter sweeps -------------------------------------------

def _sweep(cfg: ExperimentConfig, key: str, values, make_case) -> dict[str, pd.DataFrame]:
    """Generic sweep: ``make_case(value) -> list of (payoff_name, payoff, params)``."""
    bumps = tuned_bump_fn(load_tuned_bumps())
    seed = cfg.seeds[key]
    raws = []
    for i, v in enumerate(values):
        for j, (pname, f, p) in enumerate(make_case(v)):
            specs, _ = build_specs(f, p, GREEKS, SWEEP_METHODS, bumps(pname))
            raws.append(run_replications(f, p, specs, [cfg.sweep_n], cfg.sweep_reps, [seed, i, j],
                                         loc_width=cfg.loc_width, extra={"param": v}))
    raw = pd.concat(raws, ignore_index=True)
    summ = summarize(raw, ["payoff", "greek", "method", "param", "h", "n"])
    save(raw, f"{key}_raw", raw=True)
    save(summ, f"{key}_summary")
    save_meta(key, cfg, bumps=load_tuned_bumps())
    return {"raw": raw, "summary": summ}


def experiment_d(cfg: ExperimentConfig):
    return _sweep(cfg, "D_volatility", cfg.sigmas,
                  lambda s: [(n, f, PAPER_PARAMS.bumped(sigma=s)) for n, f in paper_payoffs().items()])


def experiment_e(cfg: ExperimentConfig):
    S0 = PAPER_PARAMS.S0
    return _sweep(cfg, "E_moneyness", cfg.moneyness,
                  lambda m: [("digital", CashDigitalCall(m * S0), PAPER_PARAMS),
                             ("call", EuropeanCall(m * S0), PAPER_PARAMS)])


def experiment_f(cfg: ExperimentConfig):
    return _sweep(cfg, "F_maturity", cfg.maturities,
                  lambda T: [(n, f, PAPER_PARAMS.bumped(T=T)) for n, f in paper_payoffs().items()])


# --- Experiment G: smooth vs discontinuous at equal computational budget -------------

def experiment_g(cfg: ExperimentConfig) -> dict[str, pd.DataFrame]:
    p, seed = PAPER_PARAMS, cfg.seeds["G_payoffs"]
    bumps = tuned_bump_fn(load_tuned_bumps())
    raws = []
    for i, (pname, f) in enumerate(paper_payoffs().items()):
        specs, _ = build_specs(f, p, GREEKS, ALL_METHODS, bumps(pname))
        raws.append(run_replications(f, p, specs, [], cfg.sweep_reps, [seed, i], loc_width=cfg.loc_width,
                                     budget=cfg.budget))
    raw = pd.concat(raws, ignore_index=True)
    summ = summarize(raw, ["payoff", "greek", "method", "h", "n"])
    summ["efficiency"] = 1.0 / (summ["mse"] * summ["runtime"])  # higher is better
    ref = summ[summ["method"] == "Malliavin"].set_index(["payoff", "greek"])
    summ["rmse_ratio_vs_malliavin"] = [r.rmse / ref.loc[(r.payoff, r.greek), "rmse"] for r in summ.itertuples()]
    summ["time_ratio_vs_malliavin"] = [r.runtime / ref.loc[(r.payoff, r.greek), "runtime"] for r in summ.itertuples()]
    paired = paired_vs_reference(raw, ["payoff", "greek"])
    save(raw, "G_payoffs_raw", raw=True)
    save(summ, "G_equal_budget_summary")
    save(paired, "G_paired_vs_malliavin")
    save_meta("G_payoffs", cfg, bumps=load_tuned_bumps(), budget_unit="payoff evaluations")
    return {"raw": raw, "summary": summ, "paired": paired}

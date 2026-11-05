"""Phase 3: path constructions for RQMC and wall-clock-budgeted estimation.

CPW-first (``asian.cpw_first_samples``) integrates out the *level* variable
Z_1 analytically. Write the Brownian path at the monitoring dates as

    W_{t_i} = sqrt(dt) Z_1 + B_i,    B_1 = 0,  B_i = W_{t_i} - W_{t_1}  (i >= 2),

where (B_2, ..., B_m) is a Brownian motion observed at tau_k = k dt
(k = 1..m-1), independent of Z_1. CPW-first is a function of B only. Hence any
linear construction

    B = L xi,   xi ~ N(0, I_{m-1}),   L L^T = C,   C_kl = dt * min(k, l),

gives exactly the same joint law of (Z_1, B), and the conditioning identity
P(A >= K | B) = Phi(ln(Abar/K)/(sigma sqrt dt)) is untouched. The constructions
differ only in *which* xi coordinate drives which direction of path space,
which is what matters for (R)QMC:

* sequential: L = sqrt(dt) * lower-triangular ones (the cumulative sum);
* Brownian bridge (BB): terminal value first, then recursive midpoints;
* PCA: L = V Lambda^{1/2}, eigenvectors of C by decreasing eigenvalue.

For plain MC all three give identical estimator distributions, so MC uses
the cheapest (sequential).

Budgets here are real wall-clock time: estimation stops when the budget is
spent, with block-level granularity. The stopping rule depends only on
elapsed time, never on the simulated values, so the estimator stays unbiased.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from scipy import special, stats
from scipy.stats import qmc

from .asian import AsianDigital, cpw_first_samples
from .models import BlackScholesParams

CONSTRUCTIONS = ("seq", "bb", "pca")
SAMPLERS = ("MC", "RQMC-seq", "RQMC-bb", "RQMC-pca")


# --- constructions -------------------------------------------------------------------------

def residual_cov(m: int, dt: float) -> np.ndarray:
    """Covariance of (B_2..B_m): C_kl = dt * min(k, l), k, l = 1..m-1."""
    k = np.arange(1, m)
    return dt * np.minimum.outer(k, k).astype(float)


def bridge_matrix(n: int, dt: float) -> np.ndarray:
    """Linear map L (n x n) of the Brownian-bridge construction on times dt*(1..n).

    Column j of L is the path produced when only xi_j = 1. Order of construction:
    the terminal point first, then midpoints of the remaining gaps,
    breadth-first (the standard bridge ordering).
    """
    times = dt * np.arange(1, n + 1)
    L = np.zeros((n, n))
    if n == 0:
        return L
    order = [(n - 1, -1, None)]  # (target, left neighbour or -1 for tau = 0, right neighbour or None)
    queue = [(-1, n - 1)]
    while queue:
        nxt = []
        for a, b in queue:
            if b - a > 1:
                c = (a + b) // 2          # a < c < b
                order.append((c, a, b))
                nxt += [(a, c), (c, b)]
        queue = nxt
    assert len(order) == n, (len(order), n)
    for j in range(n):
        xi = np.zeros(n)
        xi[j] = 1.0
        path = np.zeros(n)
        for k, (tgt, a, b) in enumerate(order):
            ta = 0.0 if a < 0 else times[a]
            Ba = 0.0 if a < 0 else path[a]
            if b is None:  # terminal point
                path[tgt] = math.sqrt(times[tgt]) * xi[k]
            else:
                tb, Bb, tc = times[b], path[b], times[tgt]
                mean = ((tb - tc) * Ba + (tc - ta) * Bb) / (tb - ta)
                sd = math.sqrt((tc - ta) * (tb - tc) / (tb - ta))
                path[tgt] = mean + sd * xi[k]
        L[:, j] = path
    return L


def construction_matrix(construction: str, m: int, dt: float) -> np.ndarray:
    """L with L L^T = residual_cov(m, dt), for the m-1 residual dates."""
    n = m - 1
    if construction == "seq":
        return math.sqrt(dt) * np.tril(np.ones((n, n)))
    if construction == "bb":
        return bridge_matrix(n, dt)
    if construction == "pca":
        w, V = np.linalg.eigh(residual_cov(m, dt))
        idx = np.argsort(w)[::-1]
        return V[:, idx] * np.sqrt(np.clip(w[idx], 0.0, None))
    raise ValueError(f"unknown construction {construction!r}")


def explained_variance(construction: str, m: int, dt: float, k: int) -> float:
    """Share of total path variance (trace of C) carried by the first k xi coordinates."""
    L = construction_matrix(construction, m, dt)
    return float((L[:, :k] ** 2).sum() / (L**2).sum())


def increments_from_residual(B: np.ndarray, dt: float) -> np.ndarray:
    """Map residual paths B (N, m-1) to the (N, m) increment matrix expected by
    ``cpw_first_samples``: Z_1 = 0 (never used), Z_{k+1} = (B_{k+1} - B_k)/sqrt(dt)."""
    N = B.shape[0]
    full = np.concatenate([np.zeros((N, 1)), B], axis=1)  # B_1 = 0, B_2..B_m
    Z = np.empty_like(full)
    Z[:, 0] = 0.0
    Z[:, 1:] = np.diff(full, axis=1) / math.sqrt(dt)
    return Z


# --- wall-clock budgeted estimation --------------------------------------------------------

@dataclass(frozen=True)
class TimedEstimate:
    value: float
    stderr: float
    n_paths: int
    n_units: int          # MC: blocks; RQMC: scramblings (independent units for the SE)
    wall: float           # measured wall-clock seconds, all overheads included
    ci_halfwidth: float   # 95%: t-quantile on (units - 1) df for RQMC, z for MC


def mc_block(m: int) -> int:
    """MC block size (power of two, ~2^15 normals) so a budget spans many blocks."""
    b = max(64, (1 << 15) // m)
    return 1 << (b.bit_length() - 1)


def timed_estimate(greek: str, f: AsianDigital, p: BlackScholesParams, budget: float, rng: np.random.Generator,
                   sampler: str, points_per_scramble: int | None = None, min_units: int = 4,
                   _L_cache: dict = {}) -> TimedEstimate:
    """CPW-first estimate of ``greek`` using (about) ``budget`` seconds of wall time.

    MC: blocks of i.i.d. paths until the budget is spent (at least ``min_units``
    blocks). SE from the per-path variance; CI with z = 1.96.

    RQMC-<construction>: independent Owen scramblings of ``points_per_scramble``
    Sobol points in dimension m-1, until the budget is spent (at least
    ``min_units``). Estimate = mean of scramble means; SE = sd / sqrt(R); CI with
    t_{R-1}. ``points_per_scramble`` must be a power of two.
    """
    m, dt = f.m, p.T / f.m
    t0 = time.perf_counter()
    if sampler == "MC":
        b = mc_block(m)
        n_tot, mean, m2, units = 0, 0.0, 0.0, 0
        while units < min_units or time.perf_counter() - t0 < budget:
            Z = rng.standard_normal((b, m))
            y = cpw_first_samples(f, p, Z, greek)
            mb = float(y.mean())
            m2b = float(((y - mb) ** 2).sum())
            d = mb - mean
            nn = n_tot + b
            mean += d * b / nn
            m2 += m2b + d * d * n_tot * b / nn
            n_tot = nn
            units += 1
        se = math.sqrt(m2 / (n_tot - 1) / n_tot)
        return TimedEstimate(mean, se, n_tot, units, time.perf_counter() - t0, 1.96 * se)
    if not sampler.startswith("RQMC-"):
        raise ValueError(f"unknown sampler {sampler!r}")
    cons = sampler.split("-", 1)[1]
    if points_per_scramble is None or points_per_scramble & (points_per_scramble - 1):
        raise ValueError("RQMC needs points_per_scramble = power of two")
    key = (cons, m, dt)
    if key not in _L_cache:
        _L_cache[key] = construction_matrix(cons, m, dt)
    L = _L_cache[key]
    eps = np.finfo(float).eps
    rows = (1 << 18) // max(m - 1, 1)
    blk = max(64, min(points_per_scramble, 1 << (rows.bit_length() - 1)))
    means = []
    while len(means) < min_units or time.perf_counter() - t0 < budget:
        sob = qmc.Sobol(d=m - 1, scramble=True, seed=rng)
        acc, cnt = 0.0, 0
        while cnt < points_per_scramble:
            k = min(blk, points_per_scramble - cnt)
            xi = special.ndtri(np.clip(sob.random(k), eps, 1 - eps))
            Z = increments_from_residual(xi @ L.T, dt)
            acc += float(cpw_first_samples(f, p, Z, greek).sum())
            cnt += k
        means.append(acc / cnt)
    R = len(means)
    v = np.array(means)
    se = float(v.std(ddof=1) / math.sqrt(R))
    return TimedEstimate(float(v.mean()), se, R * points_per_scramble, R, time.perf_counter() - t0,
                         float(stats.t.ppf(0.975, R - 1)) * se)


def choose_points_per_scramble(budget: float, sec_per_point: float, sec_per_scramble: float,
                               target_scrambles: int = 10, lo: int = 16, hi: int = 1 << 18) -> int:
    """Largest power of two such that ``target_scrambles`` scramblings fit in the budget
    under the measured cost model t(n) = sec_per_scramble + sec_per_point * n."""
    room = budget / target_scrambles - sec_per_scramble
    n = int(room / sec_per_point) if room > 0 else lo
    n = min(max(n, lo), hi)
    return 1 << (n.bit_length() - 1)


def functional_share(construction: str, m: int, dt: float, k: int) -> float:
    """Share of Var(mean_i B_i) explained by the first k xi coordinates.

    The average of the residual path is the leading linear driver of Abar (and
    hence of the CPW-first integrand), so this measures how well a construction
    aligns the important direction with the best Sobol coordinates.
    """
    L = construction_matrix(construction, m, dt)
    a = np.full(m - 1, 1.0 / m)  # B_1 = 0 contributes nothing to the average
    contrib = (a @ L) ** 2
    return float(contrib[:k].sum() / contrib.sum())


# --- experiment building blocks ------------------------------------------------------------

def calibrate_rqmc(ms, constructions=CONSTRUCTIONS, greek: str = "delta", repeats: int = 9, seed: int = 0,
                   pooled: bool = True, sizes=(64, 256, 1024, 4096)) -> dict:
    """Wall time per scrambling, t(n) = a + b n, per m.

    a is the per-scrambling overhead (mostly building the scrambled Sobol
    generator) and b the per-point cost. Medians over ``repeats`` at each n in
    ``sizes`` are fitted by least squares.

    pooled=True (default since the phase-3 re-run): one (a, b) per m shared by
    all constructions. The three constructions execute the same code (Sobol
    generation, inverse normal, one dense matrix product), so differences
    between per-construction fits are timing noise. In the first research pass
    that noise gave PCA 16 points per scrambling where the others got 2048
    (5 ms, m = 4), confounding the comparison. pooled=False reproduces the
    per-construction two-point fit used in that pass.
    Also returns MC seconds per path under key ("mc", m).
    """
    rng = np.random.default_rng(seed)
    p = BlackScholesParams(100.0, 0.03, 0.2, 1.0)
    out = {}
    use_sizes = sizes if pooled else (64, 4096)
    for m in ms:
        f = AsianDigital(100.0, m)
        pts = {c: {} for c in constructions}
        for cons in constructions:
            for n in use_sizes:
                vals = []
                for _ in range(repeats):
                    e = timed_estimate(greek, f, p, 0.0, rng, f"RQMC-{cons}", points_per_scramble=n, min_units=2)
                    vals.append(e.wall / e.n_units)
                pts[cons][n] = float(np.median(vals))
        if pooled:
            xs = np.array([n for c in constructions for n in use_sizes], float)
            ys = np.array([pts[c][n] for c in constructions for n in use_sizes])
            b, a = np.polyfit(xs, ys, 1)
            a, b = max(a, 0.0), max(b, 1e-12)
            for cons in constructions:
                out[(cons, m)] = (a, b)
        else:
            for cons in constructions:
                b = max((pts[cons][4096] - pts[cons][64]) / (4096 - 64), 1e-12)
                out[(cons, m)] = (max(pts[cons][64] - 64 * b, 0.0), b)
        vals = []
        for _ in range(repeats):
            e = timed_estimate(greek, f, p, 0.0, rng, "MC", min_units=4)
            vals.append(e.wall / e.n_paths)
        out[("mc", m)] = (0.0, float(np.median(vals)))
        out[("raw", m)] = pts  # medians per construction and n (diagnostics)
    return out


def pps_for(budget: float, cal: dict, sampler: str, m: int, target_scrambles: int = 10) -> int | None:
    if sampler == "MC":
        return None
    a, b = cal[(sampler.split("-", 1)[1], m)]
    return choose_points_per_scramble(budget, b, a, target_scrambles)


def reference_estimates(f: AsianDigital, p: BlackScholesParams, rng: np.random.Generator, points: int = 2**16,
                        scrambles: int = 32, check_paths: int = 2**20, greeks=("price", "delta", "gamma", "vega")) -> dict:
    """CPW-first + PCA-RQMC reference (all Greeks from the same points), plus an
    independent CPW-first plain-MC check with z-scores."""
    m, dt = f.m, p.T / f.m
    L = construction_matrix("pca", m, dt)
    eps = np.finfo(float).eps
    means = {g: [] for g in greeks}
    rows = (1 << 18) // max(m - 1, 1)
    blk = max(64, min(points, 1 << (rows.bit_length() - 1)))
    for _ in range(scrambles):
        sob = qmc.Sobol(d=m - 1, scramble=True, seed=rng)
        acc = {g: 0.0 for g in greeks}
        cnt = 0
        while cnt < points:
            k = min(blk, points - cnt)
            Z = increments_from_residual(special.ndtri(np.clip(sob.random(k), eps, 1 - eps)) @ L.T, dt)
            for g in greeks:
                acc[g] += float(cpw_first_samples(f, p, Z, g).sum())
            cnt += k
        for g in greeks:
            means[g].append(acc[g] / cnt)
    out = {}
    for g in greeks:
        v = np.array(means[g])
        out[f"ref_{g}"] = float(v.mean())
        out[f"ref_se_{g}"] = float(v.std(ddof=1) / math.sqrt(scrambles))
    for g in greeks:
        n_tot, mean, m2 = 0, 0.0, 0.0
        b = mc_block(m) * 8
        while n_tot < check_paths:
            y = cpw_first_samples(f, p, rng.standard_normal((b, m)), g)
            mb = float(y.mean()); m2b = float(((y - mb) ** 2).sum()); d = mb - mean; nn = n_tot + b
            mean += d * b / nn; m2 += m2b + d * d * n_tot * b / nn; n_tot = nn
        se = math.sqrt(m2 / (n_tot - 1) / n_tot)
        out[f"check_{g}"] = mean
        out[f"check_z_{g}"] = (mean - out[f"ref_{g}"]) / math.hypot(se, out[f"ref_se_{g}"])
    return out


def adaptive_timed(greek: str, f: AsianDigital, p: BlackScholesParams, budget: float, cal: dict,
                   rng_pilot: np.random.Generator, rng_eval: np.random.Generator, candidates=SAMPLERS,
                   pilot_fraction: float = 0.10, rqmc_rate: str = "mc") -> tuple[TimedEstimate, dict]:
    """Pilot-based selection under a real wall-clock budget.

    Each candidate gets pilot_fraction * budget / J seconds (but at least 2 MC
    blocks or 4 scramblings; the overshoot is *measured*, not hidden). The
    predicted variance at the remaining time T_rem, from the pilot SE^2 at the
    pilot wall time t_j, is:
      rqmc_rate="mc": SE^2 * t_j / T_rem for every candidate (conservative for RQMC);
      rqmc_rate="n2": SE^2 * (t_j / T_rem)^2 for RQMC candidates (an optimistic N^-2
                      variance rate), SE^2 * t_j / T_rem for MC.
    The final run uses T_rem = budget - measured pilot time (at least one minimal
    run if the pilot exhausted the budget) on fresh random numbers. The total wall
    time returned includes the pilot.
    """
    m = f.m
    t0 = time.perf_counter()
    share = pilot_fraction * budget / len(candidates)
    preds, pilots = {}, {}
    for s in candidates:
        pps = pps_for(share, cal, s, m, target_scrambles=4) if s != "MC" else None
        e = timed_estimate(greek, f, p, share, rng_pilot, s, points_per_scramble=pps,
                           min_units=4 if s != "MC" else 2)
        pilots[s] = e
    pilot_wall = time.perf_counter() - t0
    rem = budget - pilot_wall
    t_rem = max(rem, 1e-6)
    for s, e in pilots.items():
        ratio = e.wall / t_rem
        preds[s] = e.stderr**2 * (ratio**2 if (rqmc_rate == "n2" and s != "MC") else ratio)
    best = min(preds, key=preds.get)
    pps = pps_for(max(rem, 0.0), cal, best, m) if best != "MC" else None
    e = timed_estimate(greek, f, p, max(rem, 0.0), rng_eval, best, points_per_scramble=pps,
                       min_units=4 if best != "MC" else 2)
    total = time.perf_counter() - t0
    info = {"chosen": best, "pilot_wall": pilot_wall, "total_wall": total, "pilot_exhausted": rem <= 0,
            **{f"pred_{k}": v for k, v in preds.items()}}
    return TimedEstimate(e.value, e.stderr, e.n_paths, e.n_units, total, e.ci_halfwidth), info

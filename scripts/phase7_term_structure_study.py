"""Phase 7B: efficiency of CPW-first Greek estimation under volatility term structures.

    python scripts/phase7_term_structure_study.py            # ~10 min
    python scripts/phase7_term_structure_study.py --quick    # pipeline check only (NOT evidence)
    python scripts/phase7_term_structure_study.py --analysis-only

Protocol: configs/phase7_term_structure.json (committed before the run).
Output: results/phase7_term_structure/<run>/.
"""
import argparse
import json
import math
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import special
from scipy.stats import qmc

from mc_greeks import term_structure as ts
from mc_greeks import visualization as viz  # noqa: F401
from mc_greeks.asian import AsianDigital
from mc_greeks.config import RESULTS, ROOT
from mc_greeks.simulation import make_generator
from phase5_frontier_study import provenance

CFG = json.loads((ROOT / "configs" / "phase7_term_structure.json").read_text())
ARMS = list(CFG["arms"])
G3 = list(ts.JOINT)
EPS = np.finfo(float).eps
NBOOT = 2000


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def curve(shape, sbar, m, T):
    t = T * np.arange(1, m + 1) / m
    if shape == "flat":
        return np.full(m, sbar)
    if shape == "rising":
        return np.linspace(0.6 * sbar, 1.4 * sbar, m)
    if shape == "falling":
        return np.linspace(1.4 * sbar, 0.6 * sbar, m)
    if shape == "event":
        s = np.full(m, sbar)
        s[int(math.floor(0.6 * m))] = 3 * sbar
        return s
    if shape == "reverting":
        return sbar + sbar * np.exp(-3 * t / T)
    raise ValueError(shape)


def scenarios(quick):
    g = CFG["grid"]
    out, i = [], 0
    for m in g["m"]:
        for shape in g["shapes"]:
            for sb in g["sigma_bar"]:
                for k in g["moneyness_K_over_S0"]:
                    out.append(dict(sid=i, m=m, shape=shape, sigma_bar=sb, moneyness=k))
                    i += 1
    return [s for s in out if s["sigma_bar"] == 0.2 and s["moneyness"] == 0.95 and s["m"] == 12] if quick else out


def objs(sc):
    g = CFG["grid"]
    p = ts.TSParams(g["S0"], g["r"], g["T"], tuple(curve(sc["shape"], sc["sigma_bar"], sc["m"], g["T"])))
    return AsianDigital(sc["moneyness"] * g["S0"], sc["m"]), p


def means(f, p, L, n, R, rng):
    """(R, 3 + m): replicate means of (Delta, Gamma, Vega, bucket vegas) from the same points."""
    out = np.empty((R, 3 + p.m))
    for r in range(R):
        xi = special.ndtri(np.clip(qmc.Sobol(d=p.m - 1, scramble=True, seed=rng).random(n), EPS, 1 - EPS))
        o = ts.cpw_first(f, p, ts.increments_from_residual(xi @ L.T, p), bucket=True)
        out[r, :3] = [o[g].mean() for g in G3]
        out[r, 3:] = o["bucket_vega"].mean(axis=0)
    return out


def simulate(out, quick):
    out.mkdir(parents=True, exist_ok=True)
    seed = CFG["seed"]
    scen = scenarios(quick)
    pd.DataFrame(scen).to_csv(out / "scenarios.csv", index=False)
    R = 8 if quick else 32
    rows, scal, setup = [], [], []
    for sc in scen:
        f, p = objs(sc)
        Zs = make_generator([seed, 0, sc["sid"]]).standard_normal((1 << 16, p.m))
        o = ts.cpw_first(f, p, Zs, bucket=True)
        s = np.concatenate([[o[g].std(ddof=1) for g in G3], o["bucket_vega"].std(axis=0, ddof=1)])
        scal.append(dict(sid=sc["sid"], **{f"s{j}": v for j, v in enumerate(s)}))
        Lp = ts.construction_matrix("pca", p)
        Ls = {k: ts.construction_matrix(k, p) for k in ("seq", "bb", "naive", "pca")}
        t0 = time.perf_counter()
        Ls["gpca-trained"] = Lp @ ts.gradient_rotation(f, p, 512 if quick else 4096, make_generator([seed, 1, sc["sid"]]))
        t1 = time.perf_counter()
        Ls["gpca-free"] = Lp @ ts.weighted_rotation(f, p)
        t2 = time.perf_counter()
        setup.append(dict(sid=sc["sid"], m=p.m, trained_s=t1 - t0, free_s=t2 - t1))
        for lg in CFG["log2n"]:
            for a_i, arm in enumerate(ARMS):
                mu = means(f, p, Ls[arm], 2**lg, R, make_generator([seed, 2, sc["sid"], lg, a_i]))
                rows += [dict(sid=sc["sid"], log2n=lg, arm=arm, scr=r, **{f"v{j}": mu[r, j] for j in range(mu.shape[1])}) for r in range(R)]
        log(f"scenario {sc['sid']} ({sc['shape']}, m={sc['m']}) done")
    pd.DataFrame(rows).to_csv(out / "scramble_means.csv.gz", index=False)
    pd.DataFrame(scal).to_csv(out / "scales.csv", index=False)
    pd.DataFrame(setup).to_csv(out / "setup_costs.csv", index=False)
    (out / "meta.json").write_text(json.dumps({"provenance": provenance("configs/phase7_term_structure.json"), "quick": quick}, indent=2, default=str))


def gm(v):
    v = np.asarray(v, float)
    return float(np.exp(np.log(v).mean()))


def analyse(out):
    scen = pd.read_csv(out / "scenarios.csv").set_index("sid")
    sm = pd.read_csv(out / "scramble_means.csv.gz")
    sc = pd.read_csv(out / "scales.csv").set_index("sid")
    rng = np.random.default_rng(CFG["seed"] + 99)
    L, Bt = {}, {}
    for (sid, lg, arm), d in sm.groupby(["sid", "log2n", "arm"]):
        m = int(scen.loc[sid, "m"])
        X = d[[f"v{j}" for j in range(3 + m)]].to_numpy()
        s = sc.loc[sid, [f"s{j}" for j in range(3 + m)]].to_numpy(float)
        v = X.var(axis=0, ddof=1) / s**2
        idx = rng.integers(0, len(X), (NBOOT, len(X)))
        vb = X[idx].var(axis=1, ddof=1) / s**2
        L[(sid, lg, arm)] = (v[:3].mean(), v[3:].mean())
        Bt[(sid, lg, arm)] = (vb[:, :3].mean(1), vb[:, 3:].mean(1))
    rows = []
    for (m, shape), d in scen.groupby(["m", "shape"]):
        sids = list(d.index)
        for lg in CFG["log2n"]:
            for arm in ARMS:
                for den in ("pca", "gpca-trained"):
                    if arm == den:
                        continue
                    for li, lname in enumerate(("joint", "bucket")):
                        lr = np.mean([math.log(L[(s, lg, arm)][li] / L[(s, lg, den)][li]) for s in sids])
                        lb = np.mean([np.log(Bt[(s, lg, arm)][li] / Bt[(s, lg, den)][li]) for s in sids], axis=0)
                        lo, hi = np.percentile(lb, [2.5, 97.5])
                        rows.append(dict(m=m, shape=shape, log2n=lg, arm=arm, den=den, loss=lname,
                                         gm_ratio=math.exp(lr), ci_lo=math.exp(lo), ci_hi=math.exp(hi)))
    Rt = pd.DataFrame(rows)
    Rt.to_csv(out / "ratios.csv", index=False)
    pooled = Rt.groupby(["m", "shape", "arm", "den", "loss"]).agg(gm_ratio=("gm_ratio", gm), ci_lo=("ci_lo", gm), ci_hi=("ci_hi", gm)).reset_index()
    pooled.to_csv(out / "ratios_pooled_over_n.csv", index=False)

    def cell(arm, den, loss, shape=None):
        x = Rt[(Rt.arm == arm) & (Rt.den == den) & (Rt.loss == loss)]
        return x if shape is None else x[x["shape"] == shape]

    v = {}
    nv = cell("naive", "pca", "joint")
    v["T1_naive_over_pca_joint"] = {f"m{r.m}_{r.shape}_N2^{r.log2n}": [round(r.gm_ratio, 3), round(r.ci_lo, 3), round(r.ci_hi, 3)] for r in nv.itertuples()}
    ev = nv[nv["shape"] == "event"]
    v["T1_pass"] = bool((ev.ci_lo > 1).all() and np.allclose(nv[nv["shape"] == "flat"].gm_ratio, 1.0, atol=1e-9))
    for arm in ("gpca-trained", "gpca-free"):
        x = cell(arm, "pca", "joint")
        v[f"T2_{arm}_over_pca_joint"] = {f"m{r.m}_{r.shape}_N2^{r.log2n}": round(r.gm_ratio, 3) for r in x.itertuples()}
    t2 = pd.concat([cell(a, "pca", "joint") for a in ("gpca-trained", "gpca-free")])
    v["T2_max"] = float(t2.gm_ratio.max())
    v["T2_pass"] = bool((t2.gm_ratio <= 0.5).all())
    t3 = cell("gpca-free", "gpca-trained", "joint")
    v["T3_free_over_trained_joint"] = {f"m{r.m}_{r.shape}_N2^{r.log2n}": [round(r.gm_ratio, 3), round(r.ci_lo, 3), round(r.ci_hi, 3)] for r in t3.itertuples()}
    v["T3_pass"] = bool((t3.gm_ratio <= 1.25).all())
    tb, tj = cell("gpca-trained", "pca", "bucket"), cell("gpca-trained", "pca", "joint")
    j = tb.merge(tj, on=["m", "shape", "log2n"], suffixes=("_bucket", "_joint"))
    v["T4_trained_over_pca_bucket_vs_joint"] = {f"m{r.m}_{r.shape}_N2^{r.log2n}": [round(r.gm_ratio_bucket, 3), round(r.gm_ratio_joint, 3)] for r in j.itertuples()}
    v["T4_pass"] = bool((j.gm_ratio_bucket < 1).all() and (j.gm_ratio_bucket > j.gm_ratio_joint).all())
    others = Rt[(Rt.den == "pca") & Rt.arm.isin(["seq", "bb"])]
    v["seq_bb_over_pca_joint_range"] = {a: [round(float(others[(others.arm == a) & (others.loss == "joint")].gm_ratio.min()), 3),
                                            round(float(others[(others.arm == a) & (others.loss == "joint")].gm_ratio.max()), 3)] for a in ("seq", "bb")}
    st = pd.read_csv(out / "setup_costs.csv")
    v["setup_ms_median"] = {f"m{m}": {"trained_4096": round(float(d.trained_s.median() * 1e3), 2), "free": round(float(d.free_s.median() * 1e3), 3)} for m, d in st.groupby("m")}
    (out / "verdict.json").write_text(json.dumps(v, indent=2))
    log(json.dumps({k: v[k] for k in v if k.endswith("pass") or k == "T2_max"}))
    figures(out, pooled)


def figures(out, pooled):
    fd = out / "figures"
    fd.mkdir(exist_ok=True)
    ms = sorted(pooled.m.unique())
    shapes = list(CFG["grid"]["shapes"])
    arms = [a for a in ARMS if a != "pca"]
    fig, axes = plt.subplots(2, len(ms), figsize=(6 * len(ms), 7), squeeze=False, sharey="row")
    for i, loss in enumerate(("joint", "bucket")):
        for j, m in enumerate(ms):
            ax = axes[i, j]
            d = pooled[(pooled.m == m) & (pooled.den == "pca") & (pooled.loss == loss)]
            for k, arm in enumerate(arms):
                y = d[d.arm == arm].set_index("shape").reindex(shapes)
                ax.bar(np.arange(len(shapes)) + (k - 2) * 0.16, y.gm_ratio, 0.16, label=arm)
            ax.axhline(1, color="#1f1f1e", lw=0.8)
            ax.set_yscale("log"); ax.set_xticks(range(len(shapes)), shapes); ax.set_title(f"{loss} loss, m = {m}")
        axes[i, 0].set_ylabel("loss / loss with curve-aware PCA\n(fixed N, geo-mean over 4 scenarios and 2 N)")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fd / "fig7B1_constructions.png", bbox_inches="tight"); fig.savefig(fd / "fig7B1_constructions.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--analysis-only", action="store_true")
    a = ap.parse_args()
    out = RESULTS / "phase7_term_structure" / ("quick" if a.quick else "study")
    t0 = time.time()
    if not a.analysis_only:
        simulate(out, a.quick)
        meta = json.loads((out / "meta.json").read_text())
        meta["wall_seconds_total"] = time.time() - t0
        (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    analyse(out)

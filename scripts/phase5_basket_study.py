"""Phase 5E: does the joint Greek-aware rotation survive a two-asset basket Asian digital?

    python scripts/phase5_basket_study.py            # ~10 min
    python scripts/phase5_basket_study.py --quick    # pipeline check only (NOT evidence)
    python scripts/phase5_basket_study.py --analysis-only

Protocol: configs/phase5_basket.json (committed before the run). Fixed-N and
reference-free: every estimator is unbiased, so MSE = variance of scramble means.
Output: results/phase5_basket/<run>/.
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

from mc_greeks import basket as bk
from mc_greeks import visualization as viz  # noqa: F401
from mc_greeks.config import RESULTS, ROOT
from mc_greeks.joint_greeks import effective_rank
from mc_greeks.simulation import make_generator
from phase5_frontier_study import provenance

CFG = json.loads((ROOT / "configs" / "phase5_basket.json").read_text())
NAMES = list(bk.NAMES)
EPS = np.finfo(float).eps
SETS = {"joint3": ["delta1", "gamma11", "vega1"], "joint6": NAMES}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def scenarios(quick=False):
    g = CFG["grid"]
    out, i = [], 0
    for m in g["m"]:
        for rho in g["rho"]:
            for k in g["K"]:
                out.append(dict(sid=i, m=m, rho=rho, K=k))
                i += 1
    return [s for s in out if s["m"] == g["m"][0] and s["K"] == g["K"][0]] if quick else out


def objs(sc):
    g = CFG["grid"]
    return (bk.BasketAsianDigital(sc["K"], int(sc["m"]), g["w"][0], g["w"][1]),
            bk.TwoAssetParams(g["S"][0], g["S"][1], g["r"], g["sigma"][0], g["sigma"][1], sc["rho"], g["T"]))


def top_vec(G):
    return np.linalg.eigh(G)[1][:, -1]


def rotation(Gs, names):
    G = sum(Gs[n] / np.trace(Gs[n]) for n in names) / len(names)
    w, U = np.linalg.eigh(0.5 * (G + G.T))
    return U[:, np.argsort(w)[::-1]]


def scramble_means(f, p, L, n, R, rng):
    out = np.empty((R, len(NAMES) + 1))
    for r in range(R):
        xi = special.ndtri(np.clip(qmc.Sobol(d=L.shape[1], scramble=True, seed=rng).random(n), EPS, 1 - EPS))
        out[r] = bk.smoothed(f, p, xi @ L.T, with_price=True).mean(axis=0)
    return out


def simulate(out, quick):
    out.mkdir(parents=True, exist_ok=True)
    seed = CFG["seed"]
    scen = scenarios(quick)
    pd.DataFrame(scen).to_csv(out / "scenarios.csv", index=False)
    n_inst = 2 if quick else CFG["training_instances"]
    R = 8 if quick else CFG["scrambles_per_instance"]
    log2n = CFG["log2n"][:1] if quick else CFG["log2n"]
    spec, rows, chk, cost = [], [], [], []
    for sc in scen:
        f, p = objs(sc)
        dt = p.T / f.m
        L0 = bk.pca_factor(f.m, dt, p.rho)
        d = L0.shape[1]
        rng = make_generator([seed, 1, sc["sid"]])
        n_chk = 1 << (14 if quick else 19)
        X = rng.standard_normal((n_chk, d)) @ L0.T
        V = bk.smoothed(f, p, X, with_price=True)
        raw = bk.raw_payoff(f, p, X, rng.standard_normal(n_chk))
        scale = V[:, 1:].std(axis=0, ddof=1)
        chk.append(dict(sid=sc["sid"], price_smoothed=V[:, 0].mean(), price_raw=raw.mean(),
                        z=(V[:, 0].mean() - raw.mean()) / math.hypot(V[:, 0].std(), raw.std()) * math.sqrt(n_chk),
                        **{f"mc_mean_{n}": V[:, 1 + k].mean() for k, n in enumerate(NAMES)},
                        **{f"mc_se_{n}": scale[k] / math.sqrt(n_chk) for k, n in enumerate(NAMES)},
                        **{f"scale_{n}": scale[k] for k, n in enumerate(NAMES)}))
        t0 = time.perf_counter()
        bk.smoothed(f, p, rng.standard_normal((4096, d)) @ L0.T)
        sec_point = (time.perf_counter() - t0) / 4096
        rots = {}
        for i in range(n_inst):
            t0 = time.perf_counter()
            xi = make_generator([seed, 2, sc["sid"], i]).standard_normal((CFG["n_train"], d))
            g = bk.fd_gradients(f, p, L0, xi)
            Gs = {n: g[:, k, :].T @ g[:, k, :] / len(xi) for k, n in enumerate(NAMES)}
            t_grad = time.perf_counter() - t0
            tops = {n: top_vec(Gs[n]) for n in NAMES}
            row = dict(sid=sc["sid"], m=f.m, rho=p.rho, K=f.K, inst=i)
            for n in NAMES:
                er = effective_rank(Gs[n])
                ev = np.sort(np.linalg.eigvalsh(Gs[n]))[::-1]
                row.update({f"top_share_{n}": er["top_share"], f"top2_share_{n}": float(ev[:2].sum() / ev.sum()),
                            f"part_ratio_{n}": er["participation_ratio"], f"cos_pca1_{n}": abs(tops[n][0])})
            for a in range(len(NAMES)):
                for b in range(a + 1, len(NAMES)):
                    row[f"cos_{NAMES[a]}_{NAMES[b]}"] = abs(tops[NAMES[a]] @ tops[NAMES[b]])
            spec.append(row)
            for n in NAMES:
                rots[(f"spec-{n}", i)] = rotation(Gs, [n])
            t0 = time.perf_counter()
            for k, names in SETS.items():
                rots[(k, i)] = rotation(Gs, names)
            cost.append(dict(sid=sc["sid"], m=f.m, inst=i, grad_seconds=t_grad, eig_seconds=(time.perf_counter() - t0) / len(SETS),
                             sec_per_point=sec_point, dim=d, n_train=CFG["n_train"]))
        kinds = ["pca"] + list(SETS) + [f"spec-{n}" for n in NAMES]
        for lg in log2n:
            for k_i, kind in enumerate(kinds):
                for i in range(n_inst):
                    L = L0 if kind == "pca" else L0 @ rots[(kind, i)]
                    mu = scramble_means(f, p, L, 2**lg, R, make_generator([seed, 3, sc["sid"], lg, k_i, i]))
                    rows += [dict(sid=sc["sid"], m=f.m, rho=p.rho, K=f.K, log2n=lg, kind=kind, inst=i, scr=r,
                                  **dict(zip(["price"] + NAMES, mu[r]))) for r in range(R)]
        log(f"scenario {sc['sid']} (m={f.m}, rho={p.rho}, K={f.K}) done")
    pd.DataFrame(spec).to_csv(out / "gradient_spectra.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "scramble_means.csv.gz", index=False)
    pd.DataFrame(chk).to_csv(out / "checks.csv", index=False)
    pd.DataFrame(cost).to_csv(out / "training_costs.csv", index=False)
    (out / "meta.json").write_text(json.dumps({"provenance": provenance("configs/phase5_basket.json"), "quick": quick}, indent=2, default=str))


def analyse(out):
    sm = pd.read_csv(out / "scramble_means.csv.gz")
    chk = pd.read_csv(out / "checks.csv").set_index("sid")
    spec = pd.read_csv(out / "gradient_spectra.csv")
    cost = pd.read_csv(out / "training_costs.csv")
    rng = np.random.default_rng(CFG["seed"] + 99)

    def within_var(d, cols):
        X = d[cols].to_numpy() - d.groupby("inst")[cols].transform("mean").to_numpy()
        return (X**2).sum(axis=0) / (len(d) - d.inst.nunique())

    rows, zmax = [], 0.0
    for (sid, m, rho, K, lg), d in sm.groupby(["sid", "m", "rho", "K", "log2n"]):
        scale = chk.loc[sid, [f"scale_{n}" for n in NAMES]].to_numpy(float)
        vb = within_var(d[d.kind == "pca"], NAMES)
        for kind, x in d.groupby("kind"):
            v = within_var(x, NAMES)
            # unbiasedness check: grand mean against the independent plain-MC mean
            z = (x[NAMES].mean().to_numpy() - chk.loc[sid, [f"mc_mean_{n}" for n in NAMES]].to_numpy(float)) / np.sqrt(
                v / len(x) + chk.loc[sid, [f"mc_se_{n}" for n in NAMES]].to_numpy(float) ** 2)
            zmax = max(zmax, float(np.abs(z).max()))
            row = dict(sid=sid, m=m, rho=rho, K=K, log2n=lg, kind=kind, **{f"vr_{n}": v[k] / vb[k] for k, n in enumerate(NAMES)})
            for sname, names in SETS.items():
                idx = [NAMES.index(n) for n in names]
                row[f"loss_{sname}"] = float((v[idx] / scale[idx] ** 2).mean())
                row[f"loss_ratio_{sname}"] = row[f"loss_{sname}"] / float((vb[idx] / scale[idx] ** 2).mean())
            rows.append(row)
    fx = pd.DataFrame(rows)
    fx.to_csv(out / "fixed_n.csv", index=False)

    def gm(v):
        return float(np.exp(np.log(np.asarray(v, float)).mean()))

    agg = fx.groupby(["m", "rho", "log2n", "kind"]).agg(**{c: (c, gm) for c in fx.columns if c.startswith(("vr_", "loss_ratio_"))}).reset_index()
    agg.to_csv(out / "fixed_n_gm.csv", index=False)
    # own-Greek comparison: variance of Greek n under joint6 / under its own specific rotation
    own = []
    for (sid, lg), d in fx.groupby(["sid", "log2n"]):
        d = d.set_index("kind")
        for n in NAMES:
            own.append(dict(sid=sid, m=d.m.iloc[0], rho=d.rho.iloc[0], log2n=lg, greek=n,
                            joint6_over_specific=d.loc["joint6", f"vr_{n}"] / d.loc[f"spec-{n}", f"vr_{n}"],
                            joint3_over_specific=d.loc["joint3", f"vr_{n}"] / d.loc[f"spec-{n}", f"vr_{n}"],
                            specific_over_pca=d.loc[f"spec-{n}", f"vr_{n}"], joint6_over_pca=d.loc["joint6", f"vr_{n}"]))
    own = pd.DataFrame(own)
    own.to_csv(out / "joint_vs_specific.csv", index=False)
    owng = own.groupby(["m", "rho", "greek"])[["joint6_over_specific", "joint3_over_specific", "specific_over_pca", "joint6_over_pca"]].agg(gm).reset_index()
    owng.to_csv(out / "joint_vs_specific_gm.csv", index=False)
    cos_cols = [c for c in spec.columns if c.startswith("cos_") and not c.startswith("cos_pca1")]
    sg = spec.groupby(["m", "rho"])[[c for c in spec.columns if c.startswith(("top_share_", "top2_share_", "part_ratio_", "cos_"))]].median().reset_index()
    sg.to_csv(out / "gradient_spectra_summary.csv", index=False)

    # model-based break-even (alpha from config): path PCA needs rho^(-1/alpha) times the points to match
    alpha = CFG["alpha"]
    c = cost.groupby("m").agg(train=("grad_seconds", "median"), eig=("eig_seconds", "median"), spp=("sec_per_point", "median")).reset_index()
    be = []
    for r in agg[agg.kind == "joint6"].itertuples():
        cc = c[c.m == r.m].iloc[0]
        gain = r.loss_ratio_joint6 ** (-1.0 / alpha) - 1.0        # extra simulation time path PCA needs, as a fraction
        for B in CFG["break_even_budgets_seconds"]:
            be.append(dict(m=r.m, rho=r.rho, log2n=r.log2n, budget=B, loss_ratio_joint6=r.loss_ratio_joint6,
                           train_seconds=cc.train + cc.eig, break_even_M=(cc.train + cc.eig) / (B * gain) if gain > 0 else np.inf))
    pd.DataFrame(be).to_csv(out / "break_even_model.csv", index=False)

    e = lambda cols: float(spec[cols].min().min())
    headline = {
        "E1_min_top_share": e([f"top_share_{n}" for n in NAMES]),
        "E1_median_top_share_by_greek": {n: round(float(spec[f"top_share_{n}"].median()), 4) for n in NAMES},
        "E1_median_top2_share_by_greek": {n: round(float(spec[f"top2_share_{n}"].median()), 4) for n in NAMES},
        "E2_min_cos_same_asset": e(["cos_delta1_gamma11", "cos_delta1_vega1", "cos_gamma11_vega1", "cos_delta2_gamma22", "cos_delta2_vega2", "cos_gamma22_vega2"]),
        "E2_min_cos_cross_asset": e(["cos_delta1_delta2", "cos_gamma11_gamma22", "cos_vega1_vega2"]),
        "E2_median_cos_cross_asset_by_rho": {str(k): round(float(v), 4) for k, v in spec.groupby("rho")[["cos_delta1_delta2", "cos_gamma11_gamma22", "cos_vega1_vega2"]].median().min(axis=1).items()},
        "E3_gm_joint6_over_specific_range": [float(owng.joint6_over_specific.min()), float(owng.joint6_over_specific.max())],
        "E3_gm_joint6_loss_ratio_vs_pca": {f"m{r.m}_rho{r.rho}_N2^{r.log2n}": round(r.loss_ratio_joint6, 4) for r in agg[agg.kind == "joint6"].itertuples()},
        "E4_gm_specific_over_pca_range": [float(owng.specific_over_pca.min()), float(owng.specific_over_pca.max())],
        "max_abs_z_vs_plain_mc": zmax, "max_abs_z_price_raw_vs_smoothed": float(chk.z.abs().max()),
    }
    (out / "headline.json").write_text(json.dumps(headline, indent=2))
    log(json.dumps(headline))

    fd = out / "figures"
    fd.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    rhos = sorted(sg.rho.unique())
    for m, ls in zip(sorted(sg.m.unique()), ("-", "--")):
        d = sg[sg.m == m]
        for n, c in zip(NAMES, ("#2a78d6", "#7fb2ea", "#e34948", "#f0a0a0", "#1baf7a", "#8fd9bd")):
            axes[0].plot(d.rho, d[f"top_share_{n}"], marker="o", ms=3.5, ls=ls, color=c, label=f"{n}, m={m}")
        for cc, c in zip(("cos_delta1_delta2", "cos_gamma11_gamma22", "cos_vega1_vega2", "cos_delta1_gamma11", "cos_delta1_vega1"),
                         ("#2a78d6", "#e34948", "#1baf7a", "#9a9890", "#1f1f1e")):
            axes[1].plot(d.rho, d[cc], marker="o", ms=3.5, ls=ls, color=c, label=f"{cc[4:]}, m={m}")
    axes[0].set_xlabel("asset correlation rho"); axes[0].set_ylabel("share of gradient energy in the top eigen-direction"); axes[0].legend(fontsize=5.5, ncol=2)
    axes[1].set_xlabel("asset correlation rho"); axes[1].set_ylabel("|cos| between top gradient directions"); axes[1].legend(fontsize=5.5, ncol=2)
    lg = agg.log2n.max()
    kinds = ["joint6", "joint3"] + [f"spec-{n}" for n in NAMES]
    d = agg[(agg.log2n == lg) & (agg.m == agg.m.max())]
    for j, rho in enumerate(rhos):
        x = d[d.rho == rho].set_index("kind").loss_ratio_joint6
        axes[2].bar(np.arange(len(kinds)) + (j - (len(rhos) - 1) / 2) * 0.25, [x.get(k, np.nan) for k in kinds], 0.25, label=f"rho = {rho}")
    axes[2].axhline(1, color="#1f1f1e", lw=0.8); axes[2].set_yscale("log")
    axes[2].set_xticks(range(len(kinds)), kinds, rotation=35, ha="right", fontsize=7)
    axes[2].set_ylabel(f"six-Greek loss / loss with path PCA\n(m = {agg.m.max()}, N = 2^{lg})"); axes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fd / "fig5E1_basket.png", bbox_inches="tight"); fig.savefig(fd / "fig5E1_basket.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--analysis-only", action="store_true")
    a = ap.parse_args()
    out = RESULTS / "phase5_basket" / ("quick" if a.quick else "study")
    t0 = time.time()
    if not a.analysis_only:
        simulate(out, a.quick)
        meta = json.loads((out / "meta.json").read_text())
        meta["wall_seconds_total"] = time.time() - t0
        (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    analyse(out)

"""Phase 6B development check (NOT confirmatory): pick the primary training-free rotation.

    python scripts/phase6_tf_dev.py

Runs only on the Phase-5C contracts (configs/phase5_joint.json grid), which were already
used. Fixed N, SciPy LMS+shift scrambling, reference-free (variance of scramble means).
Output: results/phase6_training_free/dev/.
"""
import json
import math
import time

import numpy as np
import pandas as pd
from scipy import special
from scipy.stats import qmc

from mc_greeks import joint_greeks as jg
from mc_greeks import training_free as tf
from mc_greeks.asian import AsianDigital
from mc_greeks.asian_qmc import increments_from_residual
from mc_greeks.config import RESULTS, ROOT
from mc_greeks.models import BlackScholesParams
from mc_greeks.simulation import make_generator

EPS = np.finfo(float).eps
SEED = 20270201
G3 = jg.GREEKS


def means(f, p, L, n, R, rng):
    dt = p.T / f.m
    out = np.empty((R, 3))
    for r in range(R):
        xi = special.ndtri(np.clip(qmc.Sobol(d=f.m - 1, scramble=True, seed=rng).random(n), EPS, 1 - EPS))
        out[r] = jg.cpw_first_joint(f, p, increments_from_residual(xi @ L.T, dt)).mean(axis=0)
    return out


def main():
    g = json.loads((ROOT / "configs" / "phase5_joint.json").read_text())["grid"]
    out = RESULTS / "phase6_training_free" / "dev"
    out.mkdir(parents=True, exist_ok=True)
    rows, align = [], []
    sid = 0
    for m in g["m"]:
        for k in g["moneyness_K_over_S0"]:
            for s in g["sigma"]:
                f, p = AsianDigital(k * g["S0"], m), BlackScholesParams(g["S0"], g["r"], s, g["T"])
                dt = p.T / m
                scale = jg.risk_scales(f, p, make_generator([SEED, 0, sid]), 1 << 16)
                Ghi = jg.gradient_covariances(f, p, 1 << 14, make_generator([SEED, 1, sid]))
                Uhi, _ = jg.joint_rotation(Ghi)
                Gj = sum(Ghi[x] / np.trace(Ghi[x]) for x in G3) / 3
                lam = np.linalg.eigvalsh(Gj).max()
                kinds = {"pca": None, "trained-256": jg.train_rotation(f, p, 256, make_generator([SEED, 2, sid]))[0], "trained-16384": Uhi}
                for kd in tf.ALL_KINDS:
                    U, sec = tf.training_free_rotation(kd, f, p)
                    kinds[f"tf-{kd}"] = U
                    u = U[:, 0]
                    align.append(dict(sid=sid, m=m, kind=kd, cos_with_trained=abs(u @ Uhi[:, 0]),
                                      captured=float(u @ Gj @ u / lam), seconds=sec))
                for i, (name, U) in enumerate(kinds.items()):
                    mu = means(f, p, jg.rotated_factor(U, m, dt), 1024, 32, make_generator([SEED, 3, sid, i]))
                    v = mu.var(axis=0, ddof=1)
                    rows.append(dict(sid=sid, m=m, kind=name, loss=float(np.mean(v / scale**2)),
                                     **{f"var_{x}": v[j] / scale[j] ** 2 for j, x in enumerate(G3)}))
                print(time.strftime("%H:%M:%S"), "scenario", sid, flush=True)
                sid += 1
    d = pd.DataFrame(rows)
    base = d[d.kind == "trained-16384"].set_index("sid").loss
    d["loss_over_trained_hi"] = d.loss / d.sid.map(base)
    d.to_csv(out / "fixed_n.csv", index=False)
    a = pd.DataFrame(align)
    a.to_csv(out / "alignment.csv", index=False)
    gmv = d.groupby(["m", "kind"]).loss_over_trained_hi.apply(lambda v: math.exp(np.log(v).mean())).unstack()
    print(gmv.round(3).to_string())
    print(a.groupby(["m", "kind"])[["cos_with_trained", "captured"]].agg(["min", "median"]).round(5).to_string())
    gmv.to_csv(out / "summary.csv")


if __name__ == "__main__":
    main()

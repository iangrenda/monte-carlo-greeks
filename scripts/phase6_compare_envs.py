"""Phase 6A, hypothesis A6: compare the second-environment run with the local holdout.

    python scripts/phase6_compare_envs.py results/phase6_shift/portable-gha-ubuntu

1. Builds results/phase6_shift/holdout-portable-subset/: the local holdout restricted to the
   portable contracts (and to the first R replicates of the equal-N part), analysed with the
   same code.
2. Equal N: replicate means should agree to rounding, because the seeds are identical.
3. Equal time: compares the per-(m, budget) ratios of the two environments.
Writes <portable dir>/comparison.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from mc_greeks.config import RESULTS
import phase6_shift_study as s6


def main(portable: Path):
    local = RESULTS / "phase6_shift" / "holdout"
    sub = RESULTS / "phase6_shift" / "holdout-portable-subset"
    sub.mkdir(parents=True, exist_ok=True)
    sids = pd.read_csv(portable / "scenarios.csv").sid.tolist()
    R = int(pd.read_csv(portable / "equal_n.csv.gz").rep.max()) + 1
    for name in ("scenarios.csv", "references.csv", "calibration.csv"):
        d = pd.read_csv(local / name)
        (d[d.sid.isin(sids)] if "sid" in d else d).to_csv(sub / name, index=False)
    en = pd.read_csv(local / "equal_n.csv.gz")
    en[en.sid.isin(sids) & (en.rep < R)].to_csv(sub / "equal_n.csv.gz", index=False)
    tm = pd.read_csv(local / "timed.csv.gz")
    tm[tm.sid.isin(sids)].to_csv(sub / "timed.csv.gz", index=False)
    s6.analyse(sub)

    out = {}
    a = pd.read_csv(sub / "equal_n.csv.gz").sort_values(["sid", "log2n", "rand", "rep"]).reset_index(drop=True)
    b = pd.read_csv(portable / "equal_n.csv.gz").sort_values(["sid", "log2n", "rand", "rep"]).reset_index(drop=True)
    assert len(a) == len(b)
    rel = np.abs(a[s6.G3].to_numpy() - b[s6.G3].to_numpy()) / np.maximum(np.abs(a[s6.G3].to_numpy()), 1e-300)
    out["equal_n_rows"] = int(len(a))
    out["equal_n_max_rel_diff"] = float(rel.max())
    out["equal_n_identical_bits"] = bool((a[s6.G3].to_numpy() == b[s6.G3].to_numpy()).all())
    ra = pd.read_csv(sub / "references.csv").set_index("sid")
    rb = pd.read_csv(portable / "references.csv").set_index("sid")
    out["references_max_rel_diff"] = float((np.abs(ra.filter(like="ref_") - rb.filter(like="ref_")) / np.abs(ra.filter(like="ref_"))).max().max())

    ta = pd.read_csv(sub / "timed_ratios.csv")
    tb = pd.read_csv(portable / "timed_ratios.csv")
    j = ta.merge(tb, on=["m", "budget", "num", "den"], suffixes=("_local", "_gha"))
    j = j.dropna(subset=["gm_ratio_local", "gm_ratio_gha"])
    j["same_side_of_1"] = np.sign(np.log(j.gm_ratio_local)) == np.sign(np.log(j.gm_ratio_gha))
    j["local_sig"] = (j.ci_lo_local > 1) | (j.ci_hi_local < 1)
    j["gha_sig"] = (j.ci_lo_gha > 1) | (j.ci_hi_gha < 1)
    j["contradiction"] = j.local_sig & j.gha_sig & ~j.same_side_of_1
    j.to_csv(portable / "comparison_timed.csv", index=False)
    out["timed_cells_compared"] = int(len(j))
    out["timed_same_direction"] = int(j.same_side_of_1.sum())
    out["timed_significant_contradictions"] = int(j.contradiction.sum())
    for pair in (("shift", "owen"), ("lms", "shift"), ("lms", "owen"), ("shift", "MC")):
        x = j[(j.num == pair[0]) & (j.den == pair[1])]
        out[f"{pair[0]}_over_{pair[1]}"] = {f"m{r.m}_{r.budget * 1e3:.0f}ms": [round(r.gm_ratio_local, 3), round(r.gm_ratio_gha, 3)] for r in x.itertuples()}
    ca = pd.read_csv(sub / "calibration.csv")
    cb = pd.read_csv(portable / "calibration.csv")
    out["cost_local_vs_gha"] = {f"m{m}": {k: [round(ca[ca.m == m][k].iloc[0] * f, 4), round(cb[cb.m == m][k].iloc[0] * f, 4)]
                                         for k, f in (("owen_a", 1e3), ("lms_a", 1e3), ("shift_a", 1e3), ("owen_b", 1e6), ("lms_b", 1e6), ("shift_b", 1e6), ("mc_c", 1e6))}
                                for m in sorted(cb.m)}
    va, vb = json.loads((sub / "verdict.json").read_text()), json.loads((portable / "verdict.json").read_text())
    for k in ("A1_pass", "A2_pass", "A3_pass", "A4_pass", "A5_pass", "A4_gm_over_cells", "A5_min_pooled_coverage_rqmc", "A5_max_cell_miss_share", "infeasible_cells"):
        out[f"verdict_{k}"] = [va.get(k), vb.get(k)]
    (portable / "comparison.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main(Path(sys.argv[1]))

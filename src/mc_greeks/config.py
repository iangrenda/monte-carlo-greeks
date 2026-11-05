"""Experiment configurations (quick demo vs research grade) and seeds."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
DATA = RESULTS / "experiment_data"

#: One master seed per experiment. Seeds for sub-runs are derived as
#: SeedSequence([master, i, j, ...]) so every run is reproducible and the
#: bump-size tuning runs (C) never share random numbers with the runs that
#: use the tuned bumps (B, D-G).
SEEDS = {
    "A_paper": 1999_0391,
    "B_convergence": 2024_0001,
    "C_bump": 2024_0002,
    "D_volatility": 2024_0003,
    "E_moneyness": 2024_0004,
    "F_maturity": 2024_0005,
    "G_payoffs": 2024_0006,
}

#: Values reported in the table on p. 410 of Fournie et al. (1999), N = 10 000,
#: low-discrepancy sequences. (greek, payoff) -> (exact, MC finite difference, MC Malliavin)
PAPER_TABLE = {
    ("delta", "call"): (0.725747, 0.725639, 0.725660),
    ("gamma", "call"): (0.016660, 0.015330, 0.016634),
    ("vega", "call"): (33.320063, 33.250709, 33.267145),
    ("delta", "interval"): (-0.001335, -0.003167, -0.001335),
    ("gamma", "interval"): (-0.000389, +0.099532, -0.000389),
    ("vega", "interval"): (-0.777516, -0.542902, -0.778695),
}

#: FD bumps used before any tuning: 1% of S0 for Delta/Gamma, 0.01 vol for Vega.
DEFAULT_REL_BUMP = 0.01
DEFAULT_VOL_BUMP = 0.01

FD_METHODS = ("FD-CRN-fwd", "FD-CRN-ctr", "FD-IND-fwd", "FD-IND-ctr")
WEIGHT_METHODS = ("Malliavin", "Malliavin-loc", "Pathwise")
ALL_METHODS = WEIGHT_METHODS + FD_METHODS
SWEEP_METHODS = ("Malliavin", "Malliavin-loc", "Pathwise", "FD-CRN-ctr", "FD-IND-ctr")
GREEKS = ("delta", "gamma", "vega")


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    # B: convergence
    sample_sizes: tuple[int, ...]
    replications: int
    # C: bump size
    bump_n: int
    bump_reps: int
    rel_bumps: tuple[float, ...]
    vol_bumps: tuple[float, ...]
    # D-F sweeps and G equal-budget comparison
    sweep_n: int
    sweep_reps: int
    budget: int
    sigmas: tuple[float, ...] = (0.10, 0.15, 0.20, 0.30, 0.40, 0.60)
    moneyness: tuple[float, ...] = (0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20)
    maturities: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 1.00, 2.00)
    # A: paper replication
    paper_n: int = 10_000
    paper_reps: int = 50
    rqmc_log2n: tuple[int, ...] = (10, 11, 12, 13, 14, 15, 16, 17)
    loc_width: float = 10.0
    seeds: dict = field(default_factory=lambda: dict(SEEDS))

    def to_dict(self) -> dict:
        return asdict(self)


_REL = (1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3)
_VOL = (1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.02, 0.05, 0.1, 0.15)

QUICK = ExperimentConfig(
    name="quick",
    sample_sizes=(1000, 2500, 5000, 10_000, 25_000, 50_000),
    replications=30,
    bump_n=20_000, bump_reps=30, rel_bumps=_REL, vol_bumps=_VOL,
    sweep_n=20_000, sweep_reps=30, budget=60_000,
    paper_reps=20, rqmc_log2n=(10, 11, 12, 13, 14, 15),
)

RESEARCH = ExperimentConfig(
    name="research",
    sample_sizes=(1000, 2500, 5000, 10_000, 25_000, 50_000, 100_000, 200_000),
    replications=200,
    bump_n=50_000, bump_reps=200, rel_bumps=_REL, vol_bumps=_VOL,
    sweep_n=50_000, sweep_reps=200, budget=150_000,
    paper_reps=100,
)

CONFIGS = {"quick": QUICK, "research": RESEARCH}

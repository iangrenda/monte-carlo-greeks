"""Experiment A: replicate Section 4 of Fournie et al. (1999) (interval digital, N=10 000 table, running estimates; RQMC and localisation)."""
import _cli
from figures import figures_a
from mc_greeks.experiments import experiment_a

if __name__ == "__main__":
    cfg, args = _cli.parse(__doc__)
    if not args.figures_only:
        with _cli.Timer("experiment_a [" + cfg.name + "]"):
            experiment_a(cfg)
    figures_a()

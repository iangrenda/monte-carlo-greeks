"""Experiment C: finite-difference bump-size sensitivity, CRN vs independent random numbers; writes tuned bumps."""
import _cli
from figures import figures_c
from mc_greeks.experiments import experiment_c

if __name__ == "__main__":
    cfg, args = _cli.parse(__doc__)
    if not args.figures_only:
        with _cli.Timer("experiment_c [" + cfg.name + "]"):
            experiment_c(cfg)
    figures_c()

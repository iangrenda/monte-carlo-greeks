"""Experiment G: smooth vs discontinuous payoffs at an equal computational budget."""
import _cli
from figures import figures_sweeps
from mc_greeks.experiments import experiment_g

if __name__ == "__main__":
    cfg, args = _cli.parse(__doc__)
    if not args.figures_only:
        with _cli.Timer("experiment_g [" + cfg.name + "]"):
            experiment_g(cfg)
    figures_sweeps()

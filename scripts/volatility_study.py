"""Experiment D: estimator performance across volatility levels."""
import _cli
from figures import figures_sweeps
from mc_greeks.experiments import experiment_d

if __name__ == "__main__":
    cfg, args = _cli.parse(__doc__)
    if not args.figures_only:
        with _cli.Timer("experiment_d [" + cfg.name + "]"):
            experiment_d(cfg)
    figures_sweeps()

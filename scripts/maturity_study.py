"""Experiment F: estimator performance across maturities."""
import _cli
from figures import figures_sweeps
from mc_greeks.experiments import experiment_f

if __name__ == "__main__":
    cfg, args = _cli.parse(__doc__)
    if not args.figures_only:
        with _cli.Timer("experiment_f [" + cfg.name + "]"):
            experiment_f(cfg)
    figures_sweeps()

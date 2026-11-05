"""Experiment E: estimator performance across strike / moneyness (cash digital and call)."""
import _cli
from figures import figures_sweeps
from mc_greeks.experiments import experiment_e

if __name__ == "__main__":
    cfg, args = _cli.parse(__doc__)
    if not args.figures_only:
        with _cli.Timer("experiment_e [" + cfg.name + "]"):
            experiment_e(cfg)
    figures_sweeps()

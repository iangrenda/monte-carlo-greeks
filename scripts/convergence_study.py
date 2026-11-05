"""Experiment B: convergence of every estimator with sample size (run bump_size_study.py first to tune FD bumps)."""
import _cli
from figures import figures_b
from mc_greeks.experiments import experiment_b

if __name__ == "__main__":
    cfg, args = _cli.parse(__doc__)
    if not args.figures_only:
        with _cli.Timer("experiment_b [" + cfg.name + "]"):
            experiment_b(cfg)
    figures_b()

"""Run the whole study in dependency order (C tunes the FD bumps used by B, D-G), then all figures."""
import _cli
import figures
from mc_greeks import experiments as ex

if __name__ == "__main__":
    cfg, args = _cli.parse(__doc__)
    if not args.figures_only:
        for fn in (ex.experiment_c, ex.experiment_a, ex.experiment_b, ex.experiment_d, ex.experiment_e,
                   ex.experiment_f, ex.experiment_g):
            with _cli.Timer(f"{fn.__name__} [{cfg.name}]"):
                fn(cfg)
    figures.main()

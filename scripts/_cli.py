"""Shared command-line handling for the experiment scripts."""
import argparse
import time

from mc_greeks.config import CONFIGS


def parse(description: str):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", choices=sorted(CONFIGS), default="quick",
                    help="quick (minutes, demo) or research (larger N and replications)")
    ap.add_argument("--figures-only", action="store_true", help="re-plot from saved results without simulating")
    args = ap.parse_args()
    return CONFIGS[args.config], args


class Timer:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.t = time.perf_counter()
        print(f"[{self.label}] running ...", flush=True)

    def __exit__(self, *exc):
        print(f"[{self.label}] done in {time.perf_counter() - self.t:.1f} s", flush=True)

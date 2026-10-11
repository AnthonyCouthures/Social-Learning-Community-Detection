"""Helpers shared by the figure scripts of the paper (not part of the library)."""

import argparse


def parser(description: str, *, heavy: bool = False, seed: int = 42) -> argparse.ArgumentParser:
    """Command-line parser common to all figure scripts."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--quick", action="store_true", help="small, fast run (smoke test)")
    p.add_argument("--seed", type=int, default=seed, help=f"base random seed (default {seed})")
    if heavy:
        p.add_argument("--n-jobs", type=int, default=-1, help="parallel worker processes (-1: all cores)")
        p.add_argument("--plot-only", action="store_true", help="re-plot from the cached results CSV")
    return p

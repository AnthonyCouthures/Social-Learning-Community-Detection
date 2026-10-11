"""Fig. 3 -- Splitting thresholds of the whole graph vs. its ground-truth communities.

For LFR (a) and ABCD (b) graphs, plots K* of the whole graph and the minimum
and mean of K* over the subgraphs induced by the ground-truth communities.

By default K* is computed as in the published figure, i.e. as the inverse
spectral gap 1 / (1 - lambda_{N-1}(P)) of the normalized Laplacian. Use
``--definition paper`` for K* = 1 / lambda_{N-1}(P) as defined in Sec. IV.A.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from common import parser
from common.benchmarks import abcd_graph, lfr_graph, splitting_threshold
from common.plotting import save, set_style

RESULTS_DIR = Path(__file__).parent / "results"
N, N_REPS = 1000, 50
MIXING = np.round(np.linspace(0.05, 0.65, 15), 3)
CURVES = [
    ("full", "steelblue", r"$1/\lambda_{N-1}(G)$"),
    ("min_sub", "darkorange", r"$\min_c\,1/\lambda_{N-1}(G_c)$"),
    ("mean_sub", "forestgreen", r"$\mathrm{mean}_c\,1/\lambda_{N-1}(G_c)$"),
]


def thresholds(benchmark: str, mixing: float, rep: int, definition: str) -> dict | None:
    if benchmark == "lfr":
        out = lfr_graph(N, mixing, tau1=3, tau2=1.5, average_degree=20, min_community=100, seed=rep)
    else:
        out = abcd_graph(N, mixing, gamma=3, beta=1.5, min_degree=10, min_community_size=100, seed=rep)
    if out is None or not nx.is_connected(out[0]):
        return None
    G, communities = out
    sub = [splitting_threshold(G.subgraph(c), definition) for c in communities if len(c) > 1]
    sub = [v for v in sub if not np.isnan(v)]
    if not sub:
        return None
    return {
        "benchmark": benchmark,
        "mixing": mixing,
        "rep": rep,
        "full": splitting_threshold(G, definition),
        "min_sub": min(sub),
        "mean_sub": float(np.mean(sub)),
    }


def results_path(args) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    return RESULTS_DIR / f"fig03_thresholds_{args.definition}{'_quick' if args.quick else ''}.csv"


def compute(args) -> pd.DataFrame:
    mixing = MIXING[::3] if args.quick else MIXING
    n_reps = 3 if args.quick else N_REPS
    tasks = [(b, m, args.seed + r) for b in ("lfr", "abcd") for m in mixing for r in range(n_reps)]
    rows = Parallel(n_jobs=args.n_jobs, return_as="generator")(
        delayed(thresholds)(*task, args.definition) for task in tasks
    )
    df = pd.DataFrame([r for r in tqdm(rows, total=len(tasks), desc="spectral thresholds") if r is not None])
    df.to_csv(results_path(args), index=False)
    return df


def plot(df: pd.DataFrame, definition: str) -> None:
    set_style()
    stats = df.groupby(["benchmark", "mixing"])[[c for c, _, _ in CURVES]].agg(["mean", "std"])
    for benchmark, xlabel in (
        ("lfr", r"Mixing parameter LFR ($\mu$)"),
        ("abcd", r"Mixing parameter ABCD ($\xi$)"),
    ):
        s = stats.loc[benchmark]
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        for col, color, label in CURVES:
            mean, std = s[col]["mean"], s[col]["std"].fillna(0)
            ax.plot(s.index, mean, color=color, lw=1, label=label)
            ax.fill_between(s.index, mean - std, mean + std, color=color, alpha=0.18)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"Splitting Threshold ($K^*$)")
        ax.grid(True)
        if definition == "published":
            ax.set_yscale("symlog", linthresh=5.0)
            ax.set_ylim(0, 30)
            ax.set_yticks([0, 1, 2, 3, 4, 5, 10, 20])
            ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.legend()
        fig.tight_layout()
        suffix = "" if definition == "published" else "_paper_definition"
        save(fig, f"fig03_{benchmark}{suffix}", bbox_inches="tight")


def main() -> None:
    p = parser(__doc__, heavy=True, seed=0)
    p.add_argument("--definition", choices=["published", "paper"], default="published")
    args = p.parse_args()
    df = pd.read_csv(results_path(args)) if args.plot_only else compute(args)
    plot(df, args.definition)


if __name__ == "__main__":
    main()

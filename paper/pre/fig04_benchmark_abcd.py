"""Fig. 4 -- Benchmark of RNP, RNP-DC and SER against literature baselines on ABCD graphs.

(a) NMI, (b) relative segmentation error, (c) modularity, (d) average
conductance, as functions of the ABCD mixing parameter xi (Table I settings).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the paper/ folder, for ``common``


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

import slcd
from common import baselines, parser
from common.benchmarks import abcd_graph
from common.metrics import evaluate
from common.plotting import METHOD_STYLE, plot_fill, save, set_style

RESULTS_DIR = Path(__file__).parent / "results"

# Table I.
N, GAMMA, BETA, MIN_DEGREE, MIN_COMMUNITY = 1000, 2.5, 1.5, 10, 20
N_TRIALS = 20
XI_VALUES = np.round(np.r_[0.01, np.linspace(0.02, 0.6, 30)], 4)
DYNAMICS = {"K": 4.0, "h": 0.5, "T": 300.0}
METHODS = ["rnp", "rnp_dc", "ser", "louvain", "label_propagation", "spectral", "leiden"]


def run_trial(xi: float, trial: int, seed: int) -> list[dict]:
    G, truth = abcd_graph(
        N, xi, gamma=GAMMA, beta=BETA, min_degree=MIN_DEGREE, min_community_size=MIN_COMMUNITY, seed=seed
    )
    partitions = {
        "truth": truth,
        "rnp": slcd.rnp_communities(G, n_retry=50, seed=seed, **DYNAMICS),
        "rnp_dc": slcd.rnp_dc_communities(G, R=20.0, rho=0.99, n_retry=50, seed=seed, **DYNAMICS),
        "ser": slcd.ser_communities(G, n_runs=200, tau=1.0, epsilon=0.5, seed=seed, **DYNAMICS),
        "louvain": baselines.louvain(G, seed),
        "leiden": baselines.leiden(G, seed),
        "label_propagation": baselines.label_propagation(G, seed),
        "spectral": baselines.spectral(G, n_clusters=len(truth)),
    }
    return [{"xi": xi, "trial": trial, "method": m, **evaluate(G, truth, p)} for m, p in partitions.items()]


def results_path(args) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    return RESULTS_DIR / f"fig04_benchmark_abcd{'_quick' if args.quick else ''}.csv"


def compute(args) -> pd.DataFrame:
    xi_values = XI_VALUES[::5] if args.quick else XI_VALUES
    n_trials = 2 if args.quick else N_TRIALS
    tasks = [(xi, t, args.seed + 1000 * i + t) for i, xi in enumerate(xi_values) for t in range(n_trials)]
    rows = Parallel(n_jobs=args.n_jobs, return_as="generator")(delayed(run_trial)(*task) for task in tasks)
    df = pd.DataFrame([r for trial in tqdm(rows, total=len(tasks), desc="ABCD benchmark") for r in trial])
    df.to_csv(results_path(args), index=False)
    return df


def plot(df: pd.DataFrame) -> None:
    set_style()
    stats = df.groupby(["method", "xi"]).agg(["mean", "std"])
    panels = [
        ("nmi", "NMI Score", METHODS, (0, 1), {}),
        (
            "seg_error",
            r"$(M_{\mathrm{pred}} - M_{\mathrm{truth}}) / M_{\mathrm{truth}}$",
            METHODS,
            (None, None),
            {"ncol": 2, "loc": "upper left"},
        ),
        ("modularity", "Modularity (Q)", [*METHODS, "truth"], (0, 1), {}),
        ("conductance", "Conductance", [*METHODS, "truth"], (0, 1), {"ncol": 2, "loc": "upper left"}),
    ]
    for metric, ylabel, methods, (ymin, ymax), legend in panels:
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        for m in methods:
            if m not in stats.index.get_level_values("method"):
                continue
            color, style, name = METHOD_STYLE[m]
            s = stats.loc[m, metric]
            plot_fill(ax, s.index, s["mean"], s["std"], color, name, style, ymin, ymax)
        if metric == "seg_error":
            ax.axhline(0, color="k", linestyle="--", label="Truth Baseline")
        ax.set_xlabel(r"$\xi$")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(**legend)
        fig.tight_layout()
        save(fig, f"fig04_{metric}", bbox_inches="tight")


def main() -> None:
    args = parser(__doc__, heavy=True).parse_args()
    df = pd.read_csv(results_path(args)) if args.plot_only else compute(args)
    plot(df)


if __name__ == "__main__":
    main()

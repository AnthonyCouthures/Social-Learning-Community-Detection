"""Fig. 9 -- Sensitivity of RNP-DC to the interaction gain K and the confidence decay rho.

(a) NMI and (b) relative segmentation error on ABCD graphs for several (K, rho).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

import slcd
from common import parser
from common.benchmarks import abcd_graph
from common.metrics import evaluate
from common.plotting import plot_fill, save, set_style

RESULTS_DIR = Path(__file__).parent / "results"
N, N_TRIALS = 1000, 3
XI_VALUES = np.round(np.linspace(0.05, 0.55, 15), 3)
K_RHO = [(1.0, 0.75), (1.5, 0.9), (2.0, 0.95), (4.0, 0.99), (5.0, 0.995)]
R, N_RETRY, H, T = 5.0, 30, 0.5, 300.0


def run_trial(xi: float, seed: int) -> list[dict]:
    G, truth = abcd_graph(N, xi, gamma=2.5, beta=1.5, min_degree=10, min_community_size=20, seed=seed)
    rows = []
    for K, rho in K_RHO:
        communities = slcd.rnp_dc_communities(G, K=K, R=R, rho=rho, n_retry=N_RETRY, h=H, T=T, seed=seed)
        rows.append({"xi": xi, "seed": seed, "K": K, "rho": rho, **evaluate(G, truth, communities)})
    return rows


def results_path(args) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    return RESULTS_DIR / f"fig09_rnp_dc_sensitivity{'_quick' if args.quick else ''}.csv"


def compute(args) -> pd.DataFrame:
    xi_values = XI_VALUES[::4] if args.quick else XI_VALUES
    n_trials = 1 if args.quick else N_TRIALS
    tasks = [(xi, args.seed + 1000 * i + t) for i, xi in enumerate(xi_values) for t in range(n_trials)]
    rows = Parallel(n_jobs=args.n_jobs, return_as="generator")(delayed(run_trial)(*task) for task in tasks)
    df = pd.DataFrame([r for trial in tqdm(rows, total=len(tasks), desc="RNP-DC sensitivity") for r in trial])
    df.to_csv(results_path(args), index=False)
    return df


def plot(df: pd.DataFrame) -> None:
    set_style()
    stats = df.groupby(["K", "rho", "xi"])[["nmi", "seg_error"]].agg(["mean", "std"])
    for metric, ylabel in (
        ("nmi", "NMI Score"),
        ("seg_error", r"$(M_{\mathrm{pred}} - M_{\mathrm{truth}}) / M_{\mathrm{truth}}$"),
    ):
        fig, ax = plt.subplots(figsize=(3.6, 2.8))
        for i, (K, rho) in enumerate(K_RHO):
            s = stats.loc[(K, rho), metric]
            bounds = (0, 1) if metric == "nmi" else (None, None)
            label = rf"RNP-DC: K={K:g}, $\rho$={rho:g}"
            plot_fill(
                ax, s.index, s["mean"], s["std"], f"C{i}", label, "o-", *bounds, linewidth=1.2, markersize=3
            )
        if metric == "nmi":
            ax.set_ylim(-0.05, 1.05)
            ax.legend(loc="lower left")
        else:
            ax.axhline(0, color="k", linestyle="--", linewidth=1.5, alpha=0.8, label="Truth")
            ax.set_yscale("centered_hybrid")
            ax.set_ylim(-1.1, max(1.0, float(df[metric].max())))
            ax.set_yticks([-1, 0, 10, 20, 30, 40] if df[metric].max() > 10 else [-1, 0, 1])
            ax.legend(loc="upper left")
        ax.set_xlabel(r"$\xi$")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        save(fig, f"fig09_{metric}", bbox_inches="tight")


def main() -> None:
    args = parser(__doc__, heavy=True).parse_args()
    df = pd.read_csv(results_path(args)) if args.plot_only else compute(args)
    plot(df)


if __name__ == "__main__":
    main()

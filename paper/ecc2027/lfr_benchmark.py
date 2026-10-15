"""ECC 2027 paper, Sec. V -- second benchmark: RNP and the baselines on LFR graphs.

LFR graphs (``networkx`` implementation) with the degree and community size
ranges of the ABCD benchmark. The ``networkx`` generator realizes a mixing much
larger than its nominal parameter ``mu``, so the results are plotted against the
measured mean external degree ratio xi_bar; the RNP curve of the ABCD benchmark
(``abcd_benchmark.py``, also against its measured ratio) is shown in gray.
K*(G), min_c K*(G_c) and the fraction of disconnected communities are recorded.
As for ABCD, every randomized method is run ``abcd.N_RUNS`` times on each graph and
the curves show the mean with a 95% confidence band over the graphs.

Figure (written to ``paper/ecc2027/figures/``):
    ecc2027_lfr_nmi   NMI against the measured mean external degree ratio
The numbers quoted in the text are written to ``results/ecc2027_lfr_summary.txt``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the paper/ folder, for ``common``

import abcd_benchmark as abcd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from common import baselines, parser
from common.metrics import evaluate
from common.plotting import plot_fill, save, set_style

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
FIGURES_DIR = HERE / "figures"

# Graphs: as ABCD (N = 1000, degree exponent 2.5 in [10, 100], community size exponent 1.5 in [20, 250]).
MU_VALUES = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.425, 0.45, 0.475, 0.5, 0.55, 0.6, 0.65)
N_TRIALS = abcd.N_TRIALS
MAX_ITERS, N_ATTEMPTS = 2000, 20
METHODS = ["rnp", "dsb", "label_propagation", "louvain"]
FIGSIZE = (3.4, 2.0)  # one column of the IEEE two-column format


def lfr_graph(mu: float, seed: int) -> tuple[nx.Graph, list[set[int]]]:
    """LFR graph and its communities; generation is retried with new seeds when it fails."""
    for attempt in range(N_ATTEMPTS):
        try:
            G = nx.LFR_benchmark_graph(
                abcd.N,
                abcd.GAMMA,
                abcd.BETA,
                mu,
                min_degree=abcd.DEGREES[0],
                max_degree=abcd.DEGREES[1],
                min_community=abcd.SIZES[0],
                max_community=abcd.SIZES[1],
                seed=seed + 100_000 * attempt,
                max_iters=MAX_ITERS,
            )
            break
        except nx.ExceededMaxIterations:
            continue
    else:
        raise RuntimeError(f"LFR generation failed for mu={mu}, seed={seed}")
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    communities = [set(c) for c in {frozenset(G.nodes[v]["community"]) for v in G}]
    for v in G:
        del G.nodes[v]["community"]
    return nx.Graph(G), communities


def run_trial(mu: float, trial: int, seed: int) -> list[dict]:
    G, truth = lfr_graph(mu, seed)
    m = G.number_of_edges()
    spectral = abcd.spectral_quantities(G, truth)
    base = {"mu": mu, "trial": trial, "seed": seed, "n_edges": m, "n_true": len(truth), **spectral}
    rows = []

    def add(method: str, communities, messages=np.nan, rounds=np.nan, run=0) -> None:
        rows.append(
            {
                **base,
                "method": method,
                "run": run,
                **evaluate(G, truth, communities),
                "messages_per_edge": messages / m,
                "rounds": rounds,
            }
        )

    k = abcd.K_VALUES.index(abcd.K_MAIN)
    for run in range(abcd.N_RUNS):
        run_seed = seed * abcd.N_RUNS + run
        rng = np.random.default_rng([seed, k, run])
        rnp = abcd.rnp_checkpoints(G, K=abcd.K_MAIN, T_max=abcd.T_MAX, n_retry=(abcd.N_RETRY_MAIN,), rng=rng)
        rnp = rnp[abcd.N_RETRY_MAIN]
        add("rnp", rnp["communities"], rnp["messages"], rnp["rounds"], run=run)
        add("label_propagation", *baselines.async_label_propagation(G, run_seed), run=run)
        add("dsb", *baselines.distributed_spectral_bisection(G, run_seed, n_steps=abcd.DSB_STEPS), run=run)
        add("louvain", baselines.louvain(G, run_seed), run=run)
    return rows


def results_path(args) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    return RESULTS_DIR / f"ecc2027_lfr{'_quick' if args.quick else ''}.csv"


def compute(args) -> pd.DataFrame:
    mu_values = MU_VALUES[::4] if args.quick else MU_VALUES
    n_trials = 2 if args.quick else N_TRIALS
    tasks = [(mu, t, args.seed + 1000 * i + t) for i, mu in enumerate(mu_values) for t in range(n_trials)]
    rows = Parallel(n_jobs=args.n_jobs, return_as="generator")(delayed(run_trial)(*task) for task in tasks)
    df = pd.DataFrame([r for trial in tqdm(rows, total=len(tasks), desc="ECC LFR benchmark") for r in trial])
    df.to_csv(results_path(args), index=False)
    return df


def abcd_results(args) -> pd.DataFrame | None:
    path = abcd.RESULTS_DIR / f"ecc2027_abcd{'_quick' if args.quick else ''}.csv"
    return pd.read_csv(path) if path.exists() else None


def plot(df: pd.DataFrame, df_abcd: pd.DataFrame | None) -> None:
    set_style()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for m in METHODS:
        color, marker, linestyle, label = abcd.ECC_STYLE[m]
        d = df[df["method"] == m]
        s = abcd.mean_ci(d, "nmi", by="mu")
        x = d.groupby("mu")["xi_measured"].mean()
        plot_fill(ax, x, s["mean"], s["ci"], color, label, marker + linestyle, 0, 1)
    if df_abcd is not None:
        s = abcd.select(df_abcd, "rnp").groupby("xi")[["xi_measured", "nmi"]].mean()
        ax.plot(s["xi_measured"], s["nmi"], "-", color="gray", linewidth=0.8, alpha=0.8, label="RNP on ABCD")
    ax.axvline(abcd.threshold(abcd.K_MAIN), color="k", linestyle="-.", linewidth=0.8)
    ax.set_xlim(0, 0.7)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(r"Measured mean external degree ratio $\bar\xi$")
    ax.set_ylabel("NMI Score")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    save(fig, "ecc2027_lfr_nmi", directory=FIGURES_DIR, bbox_inches="tight")


def summary(df: pd.DataFrame, path: Path) -> None:
    pd.set_option("display.width", 200)
    blocks = []
    for metric in ("nmi", "seg_error", "messages_per_edge"):
        table = {m: df[df["method"] == m].groupby("mu")[metric].mean() for m in METHODS}
        table = {m: v for m, v in table.items() if not v.isna().all()}
        blocks.append(f"== mean {metric}\n{pd.DataFrame(table).round(3).to_string()}")
    graphs = df.drop_duplicates(["mu", "trial"]).assign(
        frac_disconnected=lambda d: d["n_disconnected"] / d["n_true"]
    )
    spec = graphs.groupby("mu")[["xi_measured", "k_star", "k_star_min_sub", "frac_disconnected"]].mean()
    blocks.append(f"== graphs (mean over trials)\n{spec.round(3).to_string()}")
    ci = {m: abcd.mean_ci(df[df["method"] == m], "nmi", by="mu")["ci"] for m in METHODS}
    blocks.append(f"== 95% CI half-width of the mean nmi\n{pd.DataFrame(ci).round(3).to_string()}")
    text = "\n\n".join(blocks)
    path.write_text(text + "\n", encoding="utf-8")
    print(text)


def main() -> None:
    args = parser(__doc__, heavy=True).parse_args()
    df = pd.read_csv(results_path(args)) if args.plot_only else compute(args)
    plot(df, abcd_results(args))
    summary(df, results_path(args).with_name(results_path(args).stem + "_summary.txt"))


if __name__ == "__main__":
    main()

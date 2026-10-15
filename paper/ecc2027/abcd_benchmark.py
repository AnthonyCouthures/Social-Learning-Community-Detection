"""ECC 2027 paper, Sec. V -- RNP against decentralized and centralized baselines on ABCD graphs.

For every graph, RNP (Algorithm 1: one realization per round, ``T_max`` steps,
stop after ``N_retry`` consecutive rounds without a cut) is run with
K = 2, 4, 8, together with RNP-light, asynchronous label propagation, the
distributed spectral bisection (DSB), Louvain and Leiden. The spectral
quantities K*(G) and min_c K*(G_c) of Sec. IV are also recorded.

Every randomized method is run ``N_RUNS`` times on each graph. The curves show
the mean over the graphs of the per-graph average, with a 95% confidence band
(normal approximation over the graphs).

For K = 4, a single run of RNP records the partition when the number of
consecutive unsuccessful rounds first reaches each value of ``N_RETRY``: this
is exactly the output of RNP with that ``N_retry`` (same random draws).

Figure (written to ``paper/ecc2027/figures/``):
    ecc2027_abcd   Fig. 2: (a) NMI of RNP, RNP-light and the baselines, (b) scalar messages
                   per edge, (c) NMI of RNP for K = 2, 4, 8 and the thresholds (K-1)/(2K)
The numbers quoted in the text (including the relative error on the number of
communities and the effect of N_retry) are written to ``results/ecc2027_abcd_summary.txt``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the paper/ folder, for ``common``

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

import slcd
from common import baselines, parser
from common.benchmarks import abcd_graph
from common.metrics import evaluate
from common.plotting import plot_fill, save, set_style
from slcd._core import apply_signal, component_labels, entry_rows, row_normalized_data, simulate
from slcd._graph import labels_to_communities, to_adjacency

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
FIGURES_DIR = HERE / "figures"

# Graphs (Sec. V): N = 1000, degree exponent 2.5 in [10, 100], community size exponent 1.5 in [20, 250].
N, GAMMA, BETA = 1000, 2.5, 1.5
DEGREES, SIZES = (10, 100), (20, 250)
N_TRIALS = 20
N_RUNS = 5  # runs of every randomized method on each graph
XI_VALUES = np.round(np.arange(0.05, 0.6501, 0.025), 3)

# RNP (Algorithm 1). T_MAX is a number of Euler steps.
H, T_MAX = 0.5, 300
K_VALUES = (2.0, 4.0, 8.0)
K_MAIN = 4.0
N_RETRY = (5, 10, 20, 50)
N_RETRY_MAIN = 50
LIGHT = {"T_max": 50, "n_retry": 10}
DSB_STEPS = 30

FIG2_SIZE = (7.1, 2.15)  # Fig. 2: three panels over the text width

# Colors, markers, line styles and names of the curves (palette of the ECC submission).
BLUE, GREEN, PURPLE, RED, ORANGE = "#006AA3", "#5B9A64", "#6D4C94", "#C1121F", "#FAA00F"
ECC_STYLE = {
    "rnp": (BLUE, "o", "-", "RNP"),
    "rnp_light": (GREEN, "^", "-", "RNP-light"),
    "dsb": (PURPLE, "s", "--", "Distr. spectral bisection"),
    "label_propagation": (RED, "D", "--", "Label propagation"),
    "louvain": (ORANGE, "v", ":", "Louvain (centralized)"),
}
GAIN_STYLE = {2.0: (RED, "D"), 4.0: (BLUE, "o"), 8.0: (GREEN, "^")}


def rnp_checkpoints(G, *, K: float, T_max: int, n_retry: tuple[int, ...], rng) -> dict[int, dict]:
    """Run sequential RNP until ``max(n_retry)`` consecutive failures, recording each checkpoint.

    Same loop as :func:`slcd._core.rnp_sequential_active`; the output for each
    ``n`` in ``n_retry`` is the partition, rounds and messages at the first
    time ``n`` consecutive rounds cut nothing.
    """
    nodes, A = to_adjacency(G, None)
    rows, cols = entry_rows(A), A.indices
    active = np.ones(A.nnz, dtype=bool)
    off_diagonal = rows != cols
    P = A.copy()
    targets = sorted(n_retry)
    out: dict[int, dict] = {}
    n_failed = n_rounds = n_messages = 0
    while targets:
        if not active.any():
            # Nothing left to cut: every remaining checkpoint is reached at once.
            n_failed = targets[-1]
        else:
            P.data = row_normalized_data(A, rows, active)
            x, _ = simulate(P, rng.uniform(-1.0, 1.0, size=len(nodes)), K=K, h=H, T=T_max * H, tol=-1.0)
            n_rounds += 1
            n_messages += T_max * int(np.count_nonzero(active & off_diagonal))
            s = apply_signal(x, K, None)
            idx = np.flatnonzero(active)
            cut = s[rows[idx]] * s[cols[idx]] < 0
            if cut.any():
                active[idx[cut]] = False
                n_failed = 0
            else:
                n_failed += 1
        while targets and n_failed >= targets[0]:
            labels = component_labels(len(nodes), rows, cols, active)
            out[targets.pop(0)] = {
                "communities": labels_to_communities(nodes, labels),
                "rounds": n_rounds,
                "messages": n_messages,
            }
    return out


def spectral_quantities(G, truth) -> dict:
    """K*(G), min_c K*(G_c) over the connected true communities, and the mean external degree ratio."""
    ext = [sum(1 for u in G[v] if u not in c) / G.degree(v) for c in truth for v in c if G.degree(v)]
    sub, disconnected = [], 0
    for c in truth:
        H_c = G.subgraph(c)
        if len(c) < 2 or not nx.is_connected(H_c):
            disconnected += 1
            continue
        sub.append(slcd.spectral_threshold(H_c, weight=None))
    return {
        "xi_measured": float(np.mean(ext)),
        "k_star": slcd.spectral_threshold(G, weight=None),
        "k_star_min_sub": float(min(sub)) if sub else np.nan,
        "n_disconnected": disconnected,
    }


def run_trial(xi: float, trial: int, seed: int) -> list[dict]:
    G, truth = abcd_graph(
        N,
        xi,
        gamma=GAMMA,
        beta=BETA,
        min_degree=DEGREES[0],
        max_degree=DEGREES[1],
        min_community_size=SIZES[0],
        max_community_size=SIZES[1],
        seed=seed,
    )
    m = G.number_of_edges()
    base = {"xi": xi, "trial": trial, "seed": seed, "n_edges": m, **spectral_quantities(G, truth)}
    rows = []

    def add(method: str, communities, messages=np.nan, rounds=np.nan, **extra) -> None:
        rows.append(
            {
                **base,
                "method": method,
                **extra,
                **evaluate(G, truth, communities),
                "messages_per_edge": messages / m,
                "rounds": rounds,
            }
        )

    for run in range(N_RUNS):
        run_seed = seed * N_RUNS + run
        for k, K in enumerate(K_VALUES):
            rng = np.random.default_rng([seed, k, run])
            retries = N_RETRY if K == K_MAIN else (N_RETRY_MAIN,)
            for n_retry, r in rnp_checkpoints(G, K=K, T_max=T_MAX, n_retry=retries, rng=rng).items():
                add(
                    "rnp",
                    r["communities"],
                    r["messages"],
                    r["rounds"],
                    K=K,
                    n_retry=n_retry,
                    T_max=T_MAX,
                    run=run,
                )
        light = rnp_checkpoints(
            G,
            K=K_MAIN,
            T_max=LIGHT["T_max"],
            n_retry=(LIGHT["n_retry"],),
            rng=np.random.default_rng([seed, 9, run]),
        )[LIGHT["n_retry"]]
        add("rnp_light", light["communities"], light["messages"], light["rounds"], K=K_MAIN, run=run, **LIGHT)
        add("label_propagation", *baselines.async_label_propagation(G, run_seed), run=run)
        add("dsb", *baselines.distributed_spectral_bisection(G, run_seed, n_steps=DSB_STEPS), run=run)
        add("louvain", baselines.louvain(G, run_seed), run=run)
        add("leiden", baselines.leiden(G, run_seed), run=run)
    return rows


def results_path(args) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    return RESULTS_DIR / f"ecc2027_abcd{'_quick' if args.quick else ''}.csv"


def compute(args) -> pd.DataFrame:
    xi_values = XI_VALUES[::4] if args.quick else XI_VALUES
    n_trials = 2 if args.quick else N_TRIALS
    tasks = [(xi, t, args.seed + 1000 * i + t) for i, xi in enumerate(xi_values) for t in range(n_trials)]
    rows = Parallel(n_jobs=args.n_jobs, return_as="generator")(delayed(run_trial)(*task) for task in tasks)
    df = pd.DataFrame([r for trial in tqdm(rows, total=len(tasks), desc="ECC ABCD benchmark") for r in trial])
    df.to_csv(results_path(args), index=False)
    return df


def select(df: pd.DataFrame, method: str, K: float = K_MAIN, n_retry: int = N_RETRY_MAIN) -> pd.DataFrame:
    """Rows of one method; for RNP, those of one gain and one ``N_retry``."""
    d = df[df["method"] == method]
    if method == "rnp":
        d = d[(d["K"] == K) & (d["n_retry"] == n_retry)]
    return d


def mean_ci(d: pd.DataFrame, metric: str, by: str = "xi") -> pd.DataFrame:
    """Mean over the graphs of the per-graph average of ``metric``, and the half-width of its 95% CI."""
    per_graph = d.groupby([by, "trial"])[metric].mean()
    stats = per_graph.groupby(level=0).agg(["mean", "std", "count"])
    stats["ci"] = 1.96 * stats["std"].fillna(0) / np.sqrt(stats["count"])
    return stats


def threshold(K: float) -> float:
    """Confinement threshold (K - 1) / (2K) of Proposition 3."""
    return (K - 1) / (2 * K)


def plot(df: pd.DataFrame) -> None:
    set_style()
    methods = ["rnp", "rnp_light", "dsb", "label_propagation", "louvain"]

    def curve(ax, d, metric, color, style, label, bounds=(None, None)):
        s = mean_ci(d, metric)
        plot_fill(ax, s.index, s["mean"], s["ci"], color, label, style, *bounds)

    def method_curve(ax, m, metric, bounds=(None, None)):
        color, marker, linestyle, label = ECC_STYLE[m]
        curve(ax, select(df, m), metric, color, marker + linestyle, label, bounds)

    def draw_nmi(ax):
        for m in methods:
            method_curve(ax, m, "nmi", (0, 1))
        ax.axvline(threshold(K_MAIN), color="k", linestyle="-.", linewidth=0.8)
        ax.set_ylim(-0.05, 1.05)

    def draw_gain(ax):
        for K in K_VALUES:
            color, marker = GAIN_STYLE[K]
            curve(ax, select(df, "rnp", K=K), "nmi", color, marker + "-", rf"RNP, $K={K:g}$", (0, 1))
            ax.axvline(threshold(K), color=color, linestyle="-.", linewidth=0.8)
        ax.set_ylim(-0.05, 1.05)

    def draw_messages(ax):
        for m in ["rnp", "rnp_light", "dsb", "label_propagation"]:
            method_curve(ax, m, "messages_per_edge")
        ax.set_yscale("log")

    handles = [
        plt.Line2D([], [], color=c, marker=mk, linestyle=ls, linewidth=1, markersize=2.5, label=name)
        for c, mk, ls, name in (ECC_STYLE[m] for m in methods)
    ]

    # Fig. 2 of the paper: (a) NMI and (b) messages of the methods, under their shared legend,
    # then (c) the gains of RNP with its own legend. Same size as the figure of the submission;
    # saved as is (no tight bounding box) to keep it.
    fig, axes = plt.subplots(1, 3, figsize=FIG2_SIZE)
    panels = [
        (draw_nmi, "NMI Score", None),
        (draw_messages, "Messages per edge", None),
        (draw_gain, "NMI Score", {"loc": "lower left"}),
    ]
    for ax, (draw, ylabel, legend), tag in zip(axes, panels, "abc", strict=True):
        draw(ax)
        ax.set_xlabel(r"$\xi$")
        ax.set_ylabel(ylabel)
        ax.set_title(f"({tag})", pad=2)
        ax.grid(True, alpha=0.3)
        if legend is not None:
            ax.legend(**legend)
    fig.tight_layout(pad=0.3, w_pad=0.8, rect=(0, 0, 1, 0.92))
    # Method legend on one row, centered over (a) and (b), shifted right if it would leave the figure.
    left = axes[0].get_tightbbox().transformed(fig.transFigure.inverted()).x0
    right = axes[1].get_position().x1
    center = (left + right) / 2
    kwargs = {"handles": handles, "loc": "upper center", "ncol": len(methods), "frameon": False}
    leg = fig.legend(bbox_to_anchor=(center, 1.0), **kwargs)
    fig.canvas.draw()
    overflow = 0.005 - leg.get_window_extent().transformed(fig.transFigure.inverted()).x0
    if overflow > 0:
        leg.remove()
        fig.legend(bbox_to_anchor=(center + overflow, 1.0), **kwargs)
    save(fig, "ecc2027_abcd", directory=FIGURES_DIR)


def summary(df: pd.DataFrame, path: Path) -> None:
    """Tables of the means quoted in the text."""
    pd.set_option("display.width", 200)
    blocks = []
    variants = [("rnp", K, n) for K in K_VALUES for n in ((N_RETRY_MAIN,) if K != K_MAIN else N_RETRY)]
    variants += [(m, None, None) for m in ["rnp_light", "dsb", "label_propagation", "louvain", "leiden"]]
    for metric in ("nmi", "seg_error", "messages_per_edge"):
        table = {}
        for m, K, n in variants:
            d = select(df, m, K=K, n_retry=n) if m == "rnp" else select(df, m)
            if metric == "messages_per_edge" and d[metric].isna().all():
                continue
            table[m if m != "rnp" else f"rnp K={K:g} Nr={n}"] = d.groupby("xi")[metric].mean()
        blocks.append(f"== mean {metric}\n{pd.DataFrame(table).round(3).to_string()}")
    spec = df.drop_duplicates(["xi", "trial"]).groupby("xi")[
        ["xi_measured", "k_star", "k_star_min_sub", "n_disconnected"]
    ]
    blocks.append(f"== graphs (mean over trials)\n{spec.mean().round(3).to_string()}")
    ci = {
        (m if m != "rnp" else f"rnp K={K:g} Nr={n}"): mean_ci(
            select(df, m, K=K, n_retry=n) if m == "rnp" else select(df, m), "nmi"
        )["ci"]
        for m, K, n in variants
    }
    blocks.append(f"== 95% CI half-width of the mean nmi\n{pd.DataFrame(ci).round(3).to_string()}")
    text = "\n\n".join(blocks)
    path.write_text(text + "\n", encoding="utf-8")
    print(text)


def main() -> None:
    args = parser(__doc__, heavy=True).parse_args()
    df = pd.read_csv(results_path(args)) if args.plot_only else compute(args)
    plot(df)
    summary(df, results_path(args).with_name(results_path(args).stem + "_summary.txt"))


if __name__ == "__main__":
    main()

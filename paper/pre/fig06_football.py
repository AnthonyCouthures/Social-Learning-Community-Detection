"""Fig. 6 -- Ground truth vs. SER on the US college football network.

(a) nominal conferences, (b) graph filtered by SER: only the edges kept by SER
are drawn; triangles are teams placed in a community whose majority
conference differs from theirs, squares are teams left isolated.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the paper/ folder, for ``common``

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D

from common import parser
from common.datasets import FOOTBALL_CONFERENCES, load_football
from common.plotting import save, set_style
from common.realworld import classify_nodes, shape_legend, threshold_communities
from slcd import ser_edge_reliability

# Fig. 6 caption: K = 6, h = 0.1, tau = 1, 100 realizations.
K, H, T, N_RUNS, TAU, EPSILON = 6.0, 0.1, 200.0, 100, 1.0, 0.25
NODE_STYLE = {"node_size": 25, "edgecolors": "black", "linewidths": 0.5}


def main() -> None:
    args = parser(__doc__).parse_args()
    n_runs = 20 if args.quick else N_RUNS
    set_style(legend_fontsize=6)
    G, conference = load_football()
    pos = nx.spring_layout(G, seed=30)
    tab20 = mpl.colormaps["tab20"]
    color = {v: tab20(conference[v] % 20) for v in G}
    handles = [
        Line2D([0], [0], marker="o", ls="None", ms=6, mfc=tab20(c % 20), mec="black")
        for c in FOOTBALL_CONFERENCES
    ]
    labels = list(FOOTBALL_CONFERENCES.values())

    fig, ax = plt.subplots(figsize=(3.7, 2.5))
    nx.draw_networkx_edges(G, pos, ax=ax, width=0.5, alpha=0.8)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=[color[v] for v in G], **NODE_STYLE)
    ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
    ax.axis("off")
    fig.tight_layout()
    save(fig, "fig06a_football_ground_truth", bbox_inches="tight")

    reliability = ser_edge_reliability(G, K=K, epsilon=EPSILON, n_runs=n_runs, h=H, T=T, seed=args.seed)
    kept = [e for e, p in reliability.items() if p >= TAU]
    communities = threshold_communities(G, kept)
    status = classify_nodes(communities, conference)
    print(f"SER: {len(communities)} communities, {sum(s != 'correct' for s in status.values())} marked teams")

    fig, ax = plt.subplots(figsize=(3.8, 2.5))
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=kept, width=0.5, alpha=0.8)
    for kind, shape, size in (("correct", "o", 25), ("wrong", "^", 30), ("disconnected", "s", 25)):
        nodes = [v for v in G if status[v] == kind]
        if nodes:
            style = {**NODE_STYLE, "node_size": size}
            nx.draw_networkx_nodes(
                G, pos, ax=ax, nodelist=nodes, node_shape=shape, node_color=[color[v] for v in nodes], **style
            )
    ax.add_artist(ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1, 1), frameon=False))
    shape_legend(
        ax,
        status,
        ["Correct Community", "Wrong Community", "Disconnected"],
        loc="lower left",
        bbox_to_anchor=(1, 0.1),
    )
    ax.axis("off")
    fig.tight_layout()
    save(fig, "fig06b_football_ser", bbox_inches="tight")


if __name__ == "__main__":
    main()

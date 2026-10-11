"""Fig. 2 -- SER edge reliability on Zachary's karate club.

(a) the network, (b) action vs. action reliability (undirected),
(c) opinion vs. action reliability (directed).
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx

from common import parser
from common.datasets import load_karate
from common.plotting import save, set_style
from slcd import ser_edge_reliability

# Parameters of the notebook that produced the published panels.
H, T, N_RUNS, EPSILON = 0.1, 300.0, 500, 0.2
K_ACTION, K_OPINION = 5.0, 8.0
NODE_STYLE = {"node_size": 25, "edgecolors": "black", "linewidths": 0.5}


def add_colorbar(fig, ax, cmap, norm):
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)


def main() -> None:
    args = parser(__doc__).parse_args()
    n_runs = 50 if args.quick else N_RUNS
    set_style()
    G = load_karate()
    pos = nx.spring_layout(G, seed=42)
    cmap, norm = mpl.colormaps["hot_r"], plt.Normalize(0, 1)

    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    nx.draw_networkx_edges(G, pos, ax=ax, width=1)
    nx.draw_networkx_nodes(G, pos, ax=ax, **NODE_STYLE)
    ax.axis("off")
    fig.tight_layout()
    save(fig, "fig02a_karate")

    action = ser_edge_reliability(
        G, K=K_ACTION, epsilon=EPSILON, n_runs=n_runs, h=H, T=T, mode="action", seed=args.seed
    )
    fig, ax = plt.subplots(figsize=(3, 2.5))
    edges = list(action)
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=edges, edge_color=[cmap(norm(action[e])) for e in edges])
    nx.draw_networkx_nodes(G, pos, ax=ax, **NODE_STYLE)
    ax.axis("off")
    add_colorbar(fig, ax, cmap, norm)
    fig.tight_layout()
    save(fig, "fig02b_karate_ser_action")

    opinion = ser_edge_reliability(
        G, K=K_OPINION, epsilon=EPSILON, n_runs=n_runs, h=H, T=T, mode="opinion", seed=args.seed
    )
    D = nx.DiGraph(list(opinion))
    fig, ax = plt.subplots(figsize=(3, 2.5))
    edges = list(D.edges())
    nx.draw_networkx_edges(
        D,
        pos,
        ax=ax,
        edgelist=edges,
        edge_color=[cmap(norm(opinion[e])) for e in edges],
        width=0.75,
        arrows=True,
        arrowstyle="->",
        arrowsize=5,
        node_size=25,
        min_source_margin=0,
        min_target_margin=0,
        connectionstyle="arc3,rad=0.1",
    )
    nx.draw_networkx_nodes(G, pos, ax=ax, **NODE_STYLE)
    ax.axis("off")
    add_colorbar(fig, ax, cmap, norm)
    fig.tight_layout()
    save(fig, "fig02c_karate_ser_opinion")


if __name__ == "__main__":
    main()

"""Fig. 5 -- Ground truth vs. SER on the Ngogo chimpanzee network (2015).

(a) the 2015 proximity network colored by the 2018 factions,
(b) SER edge reliability; nodes whose detected community disagrees with the
majority faction of that community are drawn as triangles.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the paper/ folder, for ``common``

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D

from common import parser
from common.datasets import CHIMP_FACTIONS, load_chimpanzees
from common.plotting import save, set_style
from common.realworld import classify_nodes, shape_legend, threshold_communities
from slcd import ser_edge_reliability

# Fig. 5 caption: K = 4, h = 0.5, tau = 1, 100 realizations.
K, H, T, N_RUNS, TAU, EPSILON = 4.0, 0.5, 300.0, 100, 1.0, 0.25
COLORS = {-1: "lightgray", 0: "tab:blue", 1: "tab:green"}
NODE_STYLE = {"node_size": 25, "edgecolors": "black", "linewidths": 0.5}


def faction_legend(ax, loc):
    handles = [
        Line2D([0], [0], marker="o", ls="None", ms=6, mfc=COLORS[g], mec="black") for g in CHIMP_FACTIONS
    ]
    return ax.legend(handles, list(CHIMP_FACTIONS.values()), loc=loc, frameon=False)


def main() -> None:
    args = parser(__doc__).parse_args()
    n_runs = 20 if args.quick else N_RUNS
    set_style()
    G, faction = load_chimpanzees(2015)
    pos = nx.spring_layout(G, seed=42)
    colors = {v: COLORS[faction[v]] for v in G}

    fig, ax = plt.subplots(figsize=(3, 2.5))
    nx.draw_networkx_edges(G, pos, ax=ax, width=0.5)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=[colors[v] for v in G], **NODE_STYLE)
    faction_legend(ax, "lower right")
    ax.axis("off")
    fig.tight_layout()
    save(fig, "fig05a_chimpanzees_ground_truth")

    params = {"K": K, "h": H, "T": T, "n_runs": n_runs, "epsilon": EPSILON, "seed": args.seed}
    reliability = ser_edge_reliability(G, **params)
    communities = threshold_communities(G, [e for e, p in reliability.items() if p >= TAU])
    status = classify_nodes(communities, faction, ignore=(-1,))
    sizes = sorted(map(len, communities), reverse=True)
    print(f"SER: {len(communities)} communities, sizes {sizes[:5]}...")

    cmap, norm = mpl.colormaps["hot_r"], plt.Normalize(0, 1)
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    edges = list(reliability)
    nx.draw_networkx_edges(
        G, pos, ax=ax, edgelist=edges, width=0.5, edge_color=[cmap(norm(reliability[e])) for e in edges]
    )
    for kind, shape in (("correct", "o"), ("wrong", "^"), ("disconnected", "s")):
        nodes = [v for v in G if status[v] == kind]
        if nodes:
            nx.draw_networkx_nodes(
                G,
                pos,
                ax=ax,
                nodelist=nodes,
                node_shape=shape,
                node_color=[colors[v] for v in nodes],
                **NODE_STYLE,
            )
    ax.add_artist(faction_legend(ax, "lower right"))
    shape_legend(ax, status, ["Correct / No label", "Wrong Community", "Disconnected"], loc="upper left")
    ax.axis("off")
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Agreement", rotation=270, labelpad=10)
    fig.tight_layout()
    save(fig, "fig05b_chimpanzees_ser")


if __name__ == "__main__":
    main()

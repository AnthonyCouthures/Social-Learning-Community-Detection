"""Fig. 7 -- The steps of Recursive Neighbor Pruning (RNP) on an SBM.

(a) random initial opinions, (b) polarized steady state and pruning of the
edges joining agents of opposite signs (dashed red), (c) final partition.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the paper/ folder, for ``common``

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from common import parser
from common.benchmarks import sbm_graph
from common.plotting import save, set_style
from slcd import opinion_dynamics, rnp_communities
from slcd.signals import tanh

# SBM of the notebook (4 blocks of 40, p_in = 0.30, p_out <= 0.05) and dynamics.
P_OUT = [[0, 0.05, 0.02, 0.03], [0.05, 0, 0.02, 0.02], [0.02, 0.02, 0, 0.02], [0.03, 0.02, 0.02, 0]]
K, H, T, N_RETRY = 5.0, 0.1, 50.0, 20
NODE_STYLE = {"node_size": 25, "edgecolors": "black", "linewidths": 0.5}
FIGSIZE = (2.3, 2.5)


def draw_states(G, pos, x, cut=()):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    nx.draw_networkx_edges(G, pos, ax=ax, width=0.5, alpha=0.2)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=x, cmap=plt.cm.coolwarm, vmin=-1, vmax=1, **NODE_STYLE)
    if cut:
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=cut, edge_color="red", style="dashed", width=0.5)
    ax.axis("off")
    fig.tight_layout()
    return fig


def main() -> None:
    args = parser(__doc__).parse_args()
    set_style()
    G, _ = sbm_graph([40] * 4, 0.30, P_OUT, seed=42)
    pos = nx.spring_layout(G, seed=42)
    nodes = list(G)

    # (a)-(b): one realization on the full graph (tanh signal, as in the original illustration).
    x0 = np.random.default_rng(args.seed).uniform(-1, 1, len(nodes))
    x = opinion_dynamics(G, K=K, h=H, T=T, x0=x0, signal=tanh)
    sign = dict(zip(nodes, np.sign(x), strict=True))
    cut = [(u, v) for u, v in G.edges() if sign[u] != sign[v]]
    save(draw_states(G, pos, x0), "fig07a_rnp_initialization")
    save(draw_states(G, pos, x, cut), "fig07b_rnp_pruning")

    # (c): full recursive procedure.
    communities = rnp_communities(G, K=K, h=H, T=T, n_retry=N_RETRY, signal=tanh, seed=args.seed)
    print(f"RNP: {len(communities)} communities")
    community_of = {v: i for i, c in enumerate(sorted(communities, key=min)) for v in c}
    inside = [(u, v) for u, v in G.edges() if community_of[u] == community_of[v]]
    outside = [(u, v) for u, v in G.edges() if community_of[u] != community_of[v]]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=inside, width=0.5, alpha=0.2)
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=outside, width=0.5, alpha=0.01)
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=[f"C{community_of[v] % 10}" for v in nodes], **NODE_STYLE
    )
    ax.axis("off")
    fig.tight_layout()
    save(fig, "fig07c_rnp_final")


if __name__ == "__main__":
    main()

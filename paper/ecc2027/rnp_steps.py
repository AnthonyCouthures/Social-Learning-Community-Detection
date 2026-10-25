"""ECC 2027 paper -- The rounds of Recursive Neighbor Pruning (Algorithm 1) on an SBM.

One run of sequential RNP (saturation, K = 4, h = 0.5, T_max = 300 steps) on a
stochastic block model with 4 blocks of 40 agents. Every round simulates one
realization on the current graph and removes the edges whose endpoints end
with public actions of opposite signs.

Figures (written to ``paper/ecc2027/figures/``), one per stage, graph only:
    ecc2027_rnp_a_initialization   random initial opinions
    ecc2027_rnp_b_round1           steady state of the first round that cuts edges, pruned edges dashed
    ecc2027_rnp_c_round2           same for the second round, on the pruned graph
    ecc2027_rnp_d_final            final partition
The nodes keep the same position in the four figures: a spring layout in which
the edges between the final communities are weakened, so that the communities
appear as separate clusters.
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
from slcd import opinion_dynamics

HERE = Path(__file__).parent
FIGURES_DIR = HERE / "figures"

# SBM (4 blocks of 40, p_in = 0.30, p_out <= 0.05) and RNP parameters of Sec. V.
P_OUT = [[0, 0.05, 0.02, 0.03], [0.05, 0, 0.02, 0.02], [0.02, 0.02, 0, 0.02], [0.03, 0.02, 0.02, 0]]
K, H, T_MAX, N_RETRY = 4.0, 0.5, 300, 50
FIG_SIZE = (1.75, 1.75)  # one stage
COMMUNITY_SCALE = 1.3  # drawn size of the communities relative to the spring layout

NODE_STYLE = {"node_size": 9, "edgecolors": "black", "linewidths": 0.3}
EDGE_STYLE = {"width": 0.3, "alpha": 0.25, "edge_color": "0.3"}
CUT_STYLE = {"width": 0.5, "edge_color": "red", "style": "dashed"}
OPINIONS = plt.cm.coolwarm
# Okabe-Ito colors (colour-blind safe) for the final communities.
COMMUNITY_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000"]


def rnp_rounds(G: nx.Graph, rng) -> tuple[list[tuple[nx.Graph, np.ndarray, list]], nx.Graph]:
    """Run sequential RNP on ``G``; return the rounds that cut edges and the final graph.

    Each round is ``(graph before the round, final opinions, removed edges)``.
    """
    nodes = list(G)
    H_graph = G.copy()
    rounds, n_failed = [], 0
    while n_failed < N_RETRY and H_graph.number_of_edges():
        x = opinion_dynamics(H_graph, K=K, h=H, T=T_MAX * H, seed=rng)
        sign = dict(zip(nodes, np.sign(x), strict=True))
        cut = [(u, v) for u, v in H_graph.edges() if sign[u] != sign[v]]
        if cut:
            rounds.append((H_graph.copy(), x, cut))
            H_graph.remove_edges_from(cut)
            n_failed = 0
        else:
            n_failed += 1
    return rounds, H_graph


def layout(G: nx.Graph, community_of: dict, seed: int) -> dict:
    """Spring layout with the edges between communities weakened, each community enlarged about its center."""
    weighted = nx.Graph()
    weighted.add_nodes_from(G)
    weighted.add_weighted_edges_from(
        (u, v, 1.0 if community_of[u] == community_of[v] else 0.1) for u, v in G.edges()
    )
    pos = nx.spring_layout(weighted, weight="weight", k=1.5 / np.sqrt(len(G)), iterations=200, seed=seed)
    centers = {
        c: np.mean([pos[v] for v in G if community_of[v] == c], axis=0) for c in set(community_of.values())
    }
    return {
        v: centers[community_of[v]] + COMMUNITY_SCALE * (p - centers[community_of[v]]) for v, p in pos.items()
    }


def main() -> None:
    args = parser(__doc__).parse_args()
    set_style()
    G, _ = sbm_graph([40] * 4, 0.30, P_OUT, seed=42)
    nodes = list(G)
    rng = np.random.default_rng(args.seed)

    x0 = rng.uniform(-1, 1, len(nodes))
    rounds, final = rnp_rounds(G, rng)
    communities = sorted(nx.connected_components(final), key=len, reverse=True)
    community_of = {v: i for i, c in enumerate(communities) for v in c}
    print(
        f"RNP: {len(rounds)} rounds with cuts, {len(communities)} communities of sizes "
        f"{[len(c) for c in communities]}"
    )
    pos = layout(G, community_of, args.seed)

    opinion_kwargs = {"cmap": OPINIONS, "vmin": -1, "vmax": 1, **NODE_STYLE}

    def stage(name, draw):
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        draw(ax)
        ax.set_aspect("equal")
        ax.axis("off")
        fig.tight_layout(pad=0.1)
        save(fig, name, directory=FIGURES_DIR)

    def initialization(ax):
        nx.draw_networkx_edges(G, pos, ax=ax, **EDGE_STYLE)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=x0, **opinion_kwargs)

    def pruning(graph, x, cut):
        def draw(ax):
            kept = [e for e in graph.edges() if e not in set(cut)]
            nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=kept, **EDGE_STYLE)
            nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=cut, **CUT_STYLE)
            nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=x, **opinion_kwargs)

        return draw

    def partition(ax):
        nx.draw_networkx_edges(final, pos, ax=ax, **EDGE_STYLE)
        colors = [COMMUNITY_COLORS[community_of[v] % len(COMMUNITY_COLORS)] for v in nodes]
        nx.draw_networkx_nodes(final, pos, ax=ax, node_color=colors, **NODE_STYLE)

    stage("ecc2027_rnp_a_initialization", initialization)
    for (graph, x, cut), name in zip(rounds[:2], ("b_round1", "c_round2"), strict=False):
        stage(f"ecc2027_rnp_{name}", pruning(graph, x, cut))
    stage("ecc2027_rnp_d_final", partition)


if __name__ == "__main__":
    main()

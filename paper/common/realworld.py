"""Shared logic of the real-world figures (Figs. 5 and 6)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Iterable

import networkx as nx
from matplotlib.lines import Line2D

SHAPES = {"correct": "o", "wrong": "^", "disconnected": "s"}


def threshold_communities(G: nx.Graph, kept_edges: Iterable[tuple]) -> list[set]:
    """Connected components of ``G`` restricted to ``kept_edges`` (as in ``ser_communities``)."""
    H = nx.Graph()
    H.add_nodes_from(G)
    H.add_edges_from(kept_edges)
    return [set(c) for c in nx.connected_components(H)]


def classify_nodes(
    communities: list[set[Hashable]], truth: dict[Hashable, int], ignore: tuple[int, ...] = ()
) -> dict[Hashable, str]:
    """Label each node ``"correct"``, ``"wrong"`` or ``"disconnected"``.

    Each detected community is aligned with the majority ground-truth label of
    its members (labels in ``ignore`` do not vote and are always "correct").
    Singleton communities are "disconnected".
    """
    status = {}
    for c in communities:
        if len(c) == 1:
            status.update(dict.fromkeys(c, "disconnected"))
            continue
        votes = Counter(truth[v] for v in c if truth[v] not in ignore)
        majority = votes.most_common(1)[0][0] if votes else None
        for v in c:
            ok = truth[v] in ignore or truth[v] == majority
            status[v] = "correct" if ok else "wrong"
    return status


def shape_legend(ax, status: dict, labels: list[str], **kwargs):
    """Legend of the node shapes, listing only the statuses that occur."""
    present = set(status.values())
    entries = [
        (shape, label) for (kind, shape), label in zip(SHAPES.items(), labels, strict=True) if kind in present
    ]
    handles = [Line2D([0], [0], marker=m, ls="None", ms=6, mfc="white", mec="black") for m, _ in entries]
    return ax.legend(handles, [label for _, label in entries], frameon=False, **kwargs)

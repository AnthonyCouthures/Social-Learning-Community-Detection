"""Conversion helpers between NetworkX graphs and the array-level core."""

from __future__ import annotations

from collections.abc import Hashable

import networkx as nx
import numpy as np
import scipy.sparse as sp

from slcd._core import validate_adjacency


def to_adjacency(G: nx.Graph, weight: str | None) -> tuple[list[Hashable], sp.csr_array]:
    """Return the node list and the validated CSR adjacency array of ``G``."""
    nodes = list(G)
    if not nodes:
        return nodes, sp.csr_array((0, 0), dtype=float)
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, weight=weight, dtype=float, format="csr")
    return nodes, validate_adjacency(A)


def labels_to_communities(nodes: list[Hashable], labels: np.ndarray) -> list[set[Hashable]]:
    """Group ``nodes`` by integer ``labels`` (communities ordered by first node)."""
    communities: dict[int, set[Hashable]] = {}
    for node, label in zip(nodes, labels.tolist(), strict=True):
        communities.setdefault(label, set()).add(node)
    return list(communities.values())

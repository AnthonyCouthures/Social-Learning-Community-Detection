"""Centralized baselines from the literature (Fig. 4)."""

from __future__ import annotations

import networkx as nx
import numpy as np


def louvain(G: nx.Graph, seed: int) -> list[set]:
    return list(nx.community.louvain_communities(G, seed=seed))


def leiden(G: nx.Graph, seed: int) -> list[set]:
    """Leiden (modularity) through ``leidenalg`` and ``igraph``."""
    import igraph as ig
    import leidenalg

    nodes = list(G)
    index = {v: i for i, v in enumerate(nodes)}
    g = ig.Graph(n=len(nodes), edges=[(index[u], index[v]) for u, v in G.edges()])
    membership = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, seed=seed).membership
    communities: dict[int, set] = {}
    for v, c in zip(nodes, membership, strict=True):
        communities.setdefault(c, set()).add(v)
    return list(communities.values())


def label_propagation(G: nx.Graph, seed: int) -> list[set]:
    # Semi-synchronous label propagation is deterministic given the node order.
    return list(nx.community.label_propagation_communities(G))


def spectral(G: nx.Graph, n_clusters: int, seed: int = 42) -> list[set]:
    """Spectral clustering given the true number of communities (oracle)."""
    from sklearn.cluster import SpectralClustering

    nodes = list(G)
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, format="csr", dtype=float)
    model = SpectralClustering(
        n_clusters=n_clusters, affinity="precomputed", random_state=seed, assign_labels="kmeans"
    )
    membership = model.fit_predict(A.toarray())
    return [{nodes[i] for i in np.flatnonzero(membership == c)} for c in np.unique(membership)]

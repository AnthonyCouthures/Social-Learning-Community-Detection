"""Baselines from the literature: centralized (Fig. 4) and decentralized (ECC 2027 paper)."""

from __future__ import annotations

import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


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


# ---------------------------------------------------------------------------
# Decentralized baselines with message counts (ECC 2027 paper)
# ---------------------------------------------------------------------------


def _edge_array(G: nx.Graph) -> tuple[list, np.ndarray]:
    """Node list and ``(m, 2)`` array of the undirected edges (no self-loops) of ``G``."""
    nodes = list(G)
    index = {v: i for i, v in enumerate(nodes)}
    edges = np.array([(index[u], index[v]) for u, v in G.edges() if u != v], dtype=int).reshape(-1, 2)
    return nodes, edges


def _adjacency(n: int, edges: np.ndarray) -> sp.csr_array:
    A = sp.coo_array((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    return sp.csr_array(A + A.T)


def _components(n: int, edges: np.ndarray) -> np.ndarray:
    return connected_components(_adjacency(n, edges), directed=False)[1]


def _to_communities(nodes: list, labels: np.ndarray) -> list[set]:
    return [{nodes[i] for i in np.flatnonzero(labels == c)} for c in np.unique(labels)]


def async_label_propagation(G: nx.Graph, seed: int, max_sweeps: int = 100) -> tuple[list[set], int]:
    """Asynchronous label propagation [Raghavan et al. 2007] and its number of messages.

    Nodes are updated one by one in a random order; each adopts a label held by
    the largest number of its neighbors (ties broken at random, keeping its own
    label if it is among the best). One sweep costs one label per edge
    direction, i.e. ``2 |E|`` messages. Stops after a sweep without change.
    """
    rng = np.random.default_rng(seed)
    nodes, edges = _edge_array(G)
    n = len(nodes)
    A = _adjacency(n, edges)
    neighbors = np.split(A.indices, A.indptr[1:-1])
    labels = np.arange(n)
    messages = 0
    for _ in range(max_sweeps):
        changed = False
        messages += 2 * len(edges)
        for i in rng.permutation(n):
            if len(neighbors[i]) == 0:
                continue
            values, counts = np.unique(labels[neighbors[i]], return_counts=True)
            best = values[counts == counts.max()]
            if labels[i] not in best:
                labels[i] = rng.choice(best)
                changed = True
        if not changed:
            break
    return _to_communities(nodes, labels), messages


def distributed_spectral_bisection(G: nx.Graph, seed: int, n_steps: int = 30) -> tuple[list[set], int]:
    """Recursive Fiedler-sign bisection computed by local averaging, and its number of messages.

    In the spirit of [Kempe & McSherry 2008; Lin & Cohen 2010]. On each
    component, ``n_steps`` steps of the lazy random walk ``y <- (y + P y) / 2``
    are run from a random vector; the local differences ``y(t) - y(t+1)``
    cancel the consensus mode and their signs approximate the Fiedler cut, whose
    edges are removed as in RNP. The split is kept only if it increases the
    modularity of the whole partition (a global test granted to the method);
    the resulting components are then processed in turn, and a component whose
    split is rejected is final. Each step costs one scalar per edge direction.
    """
    rng = np.random.default_rng(seed)
    nodes, edges = _edge_array(G)
    n = len(nodes)
    A0 = _adjacency(n, edges)  # modularity is measured on the original graph
    degree = np.bincount(edges.ravel(), minlength=n).astype(float)
    two_m = degree.sum()
    keep = np.ones(len(edges), dtype=bool)
    labels = _components(n, edges)
    queue = list(np.unique(labels))
    messages = 0
    while queue:
        members = labels == queue.pop()
        if members.sum() < 2:
            continue
        inside = keep & members[edges[:, 0]] & members[edges[:, 1]]
        A = _adjacency(n, edges[inside])
        d = np.asarray(A.sum(axis=1)).ravel()
        d[d == 0] = 1.0
        y = np.where(members, rng.uniform(-1.0, 1.0, n), 0.0)
        for _ in range(n_steps):
            y = 0.5 * (y + A @ y / d)
        u = y - 0.5 * (y + A @ y / d)
        messages += 2 * (n_steps + 1) * int(inside.sum())
        cut = inside & (u[edges[:, 0]] * u[edges[:, 1]] < 0)
        if not cut.any():
            continue
        new_labels = _components(n, edges[keep & ~cut])
        # Modularity change of replacing this component by its parts.
        parts = np.unique(new_labels[members])
        gain = sum(_modularity_term(A0, degree, two_m, new_labels == p) for p in parts)
        gain -= _modularity_term(A0, degree, two_m, members)
        if gain > 1e-12:
            keep &= ~cut
            labels = new_labels
            queue.extend(parts)
    return _to_communities(nodes, labels), messages


def _modularity_term(A: sp.csr_array, degree: np.ndarray, two_m: float, members: np.ndarray) -> float:
    """Contribution ``e_c / m - (d_c / 2m)^2`` of one community to the modularity."""
    idx = np.flatnonzero(members)
    internal = A[idx][:, idx].sum()
    return float(internal / two_m - (degree[idx].sum() / two_m) ** 2)

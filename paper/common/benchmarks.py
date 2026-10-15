"""Synthetic benchmark graphs (ABCD, LFR, SBM) and spectral quantities."""

from __future__ import annotations

import random

import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def abcd_graph(
    n: int,
    xi: float,
    *,
    gamma: float = 2.5,
    beta: float = 1.5,
    min_degree: int = 10,
    min_community_size: int = 20,
    max_degree: int | None = None,
    max_community_size: int | None = None,
    seed: int,
) -> tuple[nx.Graph, list[set[int]]]:
    """ABCD benchmark graph (Python ``abcd_graph`` package) and its communities.

    ``max_degree`` and ``max_community_size`` default to the bounds chosen by
    ``abcd_graph``.

    ``abcd_graph`` draws from the global ``random`` and ``numpy.random``
    generators, which are seeded here. Self-loops are removed.
    """
    from abcd_graph import ABCDGraph, ABCDParams

    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 -- abcd_graph uses the legacy global generator
    params = ABCDParams(
        vcount=n,
        gamma=gamma,
        beta=beta,
        xi=xi,
        min_degree=min_degree,
        max_degree=max_degree,
        min_community_size=min_community_size,
        max_community_size=max_community_size,
    )
    abcd = ABCDGraph(params).build()
    G = abcd.exporter.to_networkx() if hasattr(abcd, "exporter") else abcd.to_networkx()
    G = nx.Graph(G)
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    G.add_nodes_from(range(n))
    communities = [set(int(v) for v in c.vertices) for c in abcd.communities]
    return G, communities


def lfr_graph(
    n: int,
    mu: float,
    *,
    tau1: float = 3,
    tau2: float = 1.5,
    average_degree: float = 20,
    min_community: int = 100,
    seed: int,
    max_iters: int = 100,
) -> tuple[nx.Graph, list[set[int]]] | None:
    """LFR benchmark graph and its communities, or None if generation fails."""
    try:
        G = nx.LFR_benchmark_graph(
            n,
            tau1=tau1,
            tau2=tau2,
            mu=mu,
            average_degree=average_degree,
            min_community=min_community,
            seed=seed,
            max_iters=max_iters,
        )
    except nx.ExceededMaxIterations:
        return None
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    communities = list({frozenset(G.nodes[v]["community"]) for v in G})
    for v in G:
        del G.nodes[v]["community"]
    return G, [set(c) for c in communities]


def sbm_graph(sizes: list[int], p_in: float, p_out, *, seed: int) -> tuple[nx.Graph, list[set[int]]]:
    """Stochastic block model restricted to its largest connected component.

    ``p_out`` is either a scalar or a full matrix of block probabilities (its
    diagonal is overwritten by ``p_in``).
    """
    k = len(sizes)
    probs = np.full((k, k), p_out, dtype=float) if np.isscalar(p_out) else np.array(p_out, dtype=float)
    np.fill_diagonal(probs, p_in)
    G = nx.stochastic_block_model(sizes, probs.tolist(), seed=seed)
    blocks = [set(b) for b in G.graph["partition"]]
    G = nx.Graph(G.subgraph(max(nx.connected_components(G), key=len)))
    return G, [b & set(G) for b in blocks if b & set(G)]


def normalized_laplacian_gap(G: nx.Graph) -> float:
    """Second-smallest eigenvalue of the normalized Laplacian ``1 - lambda_{N-1}(P)``."""
    A = nx.to_scipy_sparse_array(G, format="csr", dtype=float)
    n = A.shape[0]
    d = np.maximum(np.asarray(A.sum(axis=1)).ravel(), 1e-12)
    d_inv_sqrt = sp.diags(1.0 / np.sqrt(d))
    S = d_inv_sqrt @ A @ d_inv_sqrt
    if n <= 500:
        eigs = np.linalg.eigvalsh(S.toarray())
    else:
        # The two largest eigenvalues of S = I - L_sym (much faster than which="SM" on L_sym).
        eigs = spla.eigsh(S, k=2, which="LA", return_eigenvectors=False, tol=1e-8)
    return float(1.0 - np.sort(eigs)[-2])


def splitting_threshold(G: nx.Graph, definition: str = "published") -> float:
    """Splitting threshold ``K*`` of a connected graph (``nan`` if undefined).

    ``definition="paper"`` is ``1 / lambda_{N-1}(P)`` as defined in Sec. IV.A of
    the paper. ``definition="published"`` is ``1 / (1 - lambda_{N-1}(P))``,
    i.e. the inverse spectral gap of the normalized Laplacian, which is the
    quantity actually plotted in the published Fig. 3.
    """
    if G.number_of_nodes() < 2 or G.number_of_edges() == 0:
        return np.nan
    gap = normalized_laplacian_gap(G)
    value = gap if definition == "published" else 1.0 - gap
    return 1.0 / value if value > 1e-10 else np.nan

"""Score-based Edge Reliability (SER) community detection."""

from __future__ import annotations

from collections.abc import Hashable

import networkx as nx
from networkx.utils import not_implemented_for

from slcd._core import Rng, Signal, check_random_state, component_labels, entry_rows, ser_scores
from slcd._graph import labels_to_communities, to_adjacency

__all__ = ["ser_communities", "ser_edge_reliability"]


def _ser(G, K, signal, h, T, n_runs, epsilon, mode, weight, seed):
    if K <= 0:
        raise ValueError("the interaction gain K must be positive")
    if n_runs < 1:
        raise ValueError("n_runs must be at least 1")
    nodes, A = to_adjacency(G, weight)
    rng = check_random_state(seed)
    scores = ser_scores(A, K=K, h=h, T=T, n_runs=n_runs, epsilon=epsilon, mode=mode, signal=signal, rng=rng)
    return nodes, A, scores


@not_implemented_for("directed")
@not_implemented_for("multigraph")
def ser_edge_reliability(
    G: nx.Graph,
    *,
    K: float = 4.0,
    epsilon: float = 0.5,
    n_runs: int = 200,
    mode: str = "action",
    h: float = 0.5,
    T: float = 300.0,
    signal: Signal | None = None,
    weight: str | None = "weight",
    seed: int | Rng | None = None,
) -> dict[tuple[Hashable, Hashable], float]:
    r"""Reliability score of every edge under the social learning dynamics.

    Every node is an agent with a private opinion :math:`x_i \in [-1, 1]` that
    it updates from the public actions :math:`s(x_j) = f(K x_j)` of its
    neighbors (Eq. (2) of [1]_)

    .. math::

        x_i(t+1) = x_i(t) + h \Big(\frac{1}{d_i} \sum_{j} a_{ij} s(x_j(t)) - x_i(t)\Big).

    The dynamics is run ``n_runs`` times from independent uniform initial
    opinions. The reliability :math:`P_{ij}` of an edge is the fraction of runs
    in which its steady-state tension is below ``epsilon``.

    Parameters
    ----------
    G : NetworkX graph
        Undirected graph. Edge weights must be non-negative.
    K : float, default 4.0
        Interaction gain :math:`K = s'(0)`. Communities can only be separated
        when :math:`K \lambda_{N-1} > 1`, see :func:`spectral_threshold`.
    epsilon : float, default 0.5
        Tension threshold below which two agents agree.
    n_runs : int, default 200
        Number :math:`M` of independent realizations.
    mode : {"action", "opinion"}, default "action"
        Tension measured on edge :math:`(i, j)`:

        * ``"action"`` -- action vs. action, :math:`|s(x_i) - s(x_j)|`;
          the score is symmetric.
        * ``"opinion"`` -- opinion vs. action, :math:`|x_i - s(x_j)|`, the
          disagreement felt by :math:`i` with the action of :math:`j`; the
          score is directed.
    h : float, default 0.5
        Euler step size, in :math:`(0, 1]`.
    T : float, default 300.0
        Time horizon; at most ``round(T / h)`` steps are simulated per run
        (integration stops earlier once the states have converged).
    signal : callable or None, default None
        Unit signal function :math:`f`, see :mod:`slcd.signals`. ``None``
        uses the saturation :math:`f(z) = \min(1, \max(-1, z))`.
    weight : str or None, default "weight"
        Edge attribute used as weight. If None, every edge has weight 1.
    seed : int, numpy.random.Generator or None, default None
        Seed or generator for the random initial opinions.

    Returns
    -------
    dict
        For ``mode="action"``, maps each edge ``(u, v)`` of ``G.edges`` to its
        score in :math:`[0, 1]`. For ``mode="opinion"``, maps both ``(u, v)``
        and ``(v, u)`` to the directed scores :math:`P_{u \to v}` and
        :math:`P_{v \to u}`.

    Raises
    ------
    NetworkXNotImplemented
        If ``G`` is directed or a multigraph.
    ValueError
        If a weight is negative or a parameter is out of range.

    See Also
    --------
    ser_communities

    References
    ----------
    .. [1] A. Couthures, A. V. Jayakumar, V. S. Varma, I.-C. Morarescu,
       S. Lasaulce and A. Girard, "Social learning community detection with
       nonlinear interaction".

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.barbell_graph(5, 0)
    >>> scores = ser_edge_reliability(G, K=4, n_runs=50, seed=0)
    >>> scores[(0, 1)]  # edge inside a clique
    1.0
    >>> scores[(4, 5)] < 1  # the bridge
    True
    """
    nodes, A, scores = _ser(G, K, signal, h, T, n_runs, epsilon, mode, weight, seed)
    rows, cols = entry_rows(A), A.indices
    score_of = {(nodes[i], nodes[j]): float(p) for i, j, p in zip(rows, cols, scores, strict=True)}
    if mode == "opinion":
        return score_of
    return {(u, v): score_of[(u, v)] for u, v in G.edges()}


@not_implemented_for("directed")
@not_implemented_for("multigraph")
def ser_communities(
    G: nx.Graph,
    *,
    K: float = 4.0,
    tau: float = 1.0,
    epsilon: float = 0.5,
    n_runs: int = 200,
    mode: str = "action",
    h: float = 0.5,
    T: float = 300.0,
    signal: Signal | None = None,
    weight: str | None = "weight",
    seed: int | Rng | None = None,
) -> list[set[Hashable]]:
    r"""Find communities with Score-based Edge Reliability (SER).

    SER (Algorithm 3 of [1]_) computes the reliability score :math:`P_{ij}`
    of every edge (see :func:`ser_edge_reliability`), removes the edges with
    :math:`P_{ij} < \tau` and returns the connected components of the
    remaining graph. With ``mode="opinion"`` the scores are directed; an edge
    is kept if it is reliable in at least one direction (weakly connected
    components).

    The algorithm is decentralized: each agent only needs the public actions
    of its neighbors, and never reveals its private opinion.

    Parameters
    ----------
    G : NetworkX graph
        Undirected graph. Edge weights must be non-negative.
    K : float, default 4.0
        Interaction gain.
    tau : float, default 1.0
        Cut threshold: edges with reliability below ``tau`` are removed.
        ``tau=1`` keeps only the edges that agreed in every run.
    epsilon : float, default 0.5
        Tension threshold.
    n_runs : int, default 200
        Number of independent realizations.
    mode : {"action", "opinion"}, default "action"
        Tension measure, see :func:`ser_edge_reliability`.
    h : float, default 0.5
        Euler step size.
    T : float, default 300.0
        Time horizon of each realization.
    signal : callable or None, default None
        Unit signal function; ``None`` is the saturation.
    weight : str or None, default "weight"
        Edge attribute used as weight. If None, every edge has weight 1.
    seed : int, numpy.random.Generator or None, default None
        Seed or generator for the random initial opinions.

    Returns
    -------
    list of sets
        A partition of the nodes of ``G``.

    Raises
    ------
    NetworkXNotImplemented
        If ``G`` is directed or a multigraph.

    See Also
    --------
    ser_edge_reliability, rnp_communities, rnp_dc_communities

    Notes
    -----
    The cost is :math:`O(M \cdot T/h \cdot |E|)` for :math:`M` runs, all runs
    being simulated simultaneously as sparse matrix products.

    References
    ----------
    .. [1] A. Couthures, A. V. Jayakumar, V. S. Varma, I.-C. Morarescu,
       S. Lasaulce and A. Girard, "Social learning community detection with
       nonlinear interaction".

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.barbell_graph(5, 0)
    >>> sorted(map(sorted, ser_communities(G, n_runs=50, seed=0)))
    [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    """
    nodes, A, scores = _ser(G, K, signal, h, T, n_runs, epsilon, mode, weight, seed)
    keep = scores >= tau
    labels = component_labels(len(nodes), entry_rows(A), A.indices, keep)
    return labels_to_communities(nodes, labels)

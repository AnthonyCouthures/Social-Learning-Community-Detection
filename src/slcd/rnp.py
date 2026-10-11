"""Recursive Neighbor Pruning (RNP) and its Decaying Confidence variant (RNP-DC)."""

from __future__ import annotations

from collections.abc import Hashable

import networkx as nx
from networkx.utils import not_implemented_for

from slcd._core import (
    Rng,
    Signal,
    check_random_state,
    component_labels,
    entry_rows,
    rnp_active,
    rnp_dc_active,
)
from slcd._graph import labels_to_communities, to_adjacency

__all__ = ["rnp_communities", "rnp_dc_communities"]


def _check(K: float, n_retry: int) -> None:
    if K <= 0:
        raise ValueError("the interaction gain K must be positive")
    if n_retry < 1:
        raise ValueError("n_retry must be at least 1")


@not_implemented_for("directed")
@not_implemented_for("multigraph")
def rnp_communities(
    G: nx.Graph,
    *,
    K: float = 4.0,
    n_retry: int = 50,
    h: float = 0.5,
    T: float = 300.0,
    signal: Signal | None = None,
    weight: str | None = "weight",
    seed: int | Rng | None = None,
) -> list[set[Hashable]]:
    r"""Find communities with Recursive Neighbor Pruning (RNP).

    RNP (Algorithm 1 of [1]_) runs the social learning dynamics from random
    initial opinions until the agents polarize, then every agent severs its
    ties with the neighbors whose public action has the opposite sign
    (:math:`s(x_i) s(x_j) < 0`). The procedure is repeated on the pruned
    graph until ``n_retry`` independent realizations no longer cut any edge.
    The communities are the connected components of the final graph.

    Parameters
    ----------
    G : NetworkX graph
        Undirected graph. Edge weights must be non-negative.
    K : float, default 4.0
        Interaction gain :math:`K = s'(0)`.
    n_retry : int, default 50
        Number of independent realizations tried at each pruning round
        (:math:`N_{\mathrm{retry}}`); they are simulated simultaneously.
    h : float, default 0.5
        Euler step size.
    T : float, default 300.0
        Time horizon of each realization.
    signal : callable or None, default None
        Unit signal function, see :mod:`slcd.signals`; ``None`` is the
        saturation.
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
    rnp_dc_communities, ser_communities

    Notes
    -----
    RNP assumes that the dynamics polarizes, i.e. :math:`K` is above the
    splitting threshold of the graph (see :func:`slcd.spectral_threshold`).
    Below it, all states decay towards 0, the signs compared by RNP are
    numerical noise, and the output is an arbitrary fragmentation.

    With a large step ``h``, two very strongly coupled agents (e.g. joined by
    an edge carrying most of their weight) can lock into a period-2 orbit of
    the Euler map and be separated; use a smaller ``h`` in that case.

    References
    ----------
    .. [1] A. Couthures, A. V. Jayakumar, V. S. Varma, I.-C. Morarescu,
       S. Lasaulce and A. Girard, "Social learning community detection with
       nonlinear interaction".

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.barbell_graph(5, 0)
    >>> sorted(map(sorted, rnp_communities(G, seed=0)))
    [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    """
    _check(K, n_retry)
    nodes, A = to_adjacency(G, weight)
    active = rnp_active(A, K=K, h=h, T=T, n_retry=n_retry, signal=signal, rng=check_random_state(seed))
    return labels_to_communities(nodes, component_labels(len(nodes), entry_rows(A), A.indices, active))


@not_implemented_for("directed")
@not_implemented_for("multigraph")
def rnp_dc_communities(
    G: nx.Graph,
    *,
    K: float = 4.0,
    R: float = 20.0,
    rho: float = 0.99,
    n_retry: int = 50,
    h: float = 0.5,
    T: float = 300.0,
    signal: Signal | None = None,
    weight: str | None = "weight",
    seed: int | Rng | None = None,
) -> list[set[Hashable]]:
    r"""Find communities with RNP with Decaying Confidence (RNP-DC).

    RNP-DC (Algorithm 2 of [1]_) prunes edges *during* the transient phase of
    the dynamics: at step :math:`t` an agent severs its tie with a neighbor
    whenever their public actions differ by more than the shrinking
    confidence radius :math:`r_t = R \rho^t`, and keeps averaging over its
    remaining neighbors only (Eq. (B1)). Rounds with fresh initial opinions
    are repeated until a whole round cuts no edge. The communities are the
    connected components of the final graph.

    Because communities decouple early and synchronize exactly, RNP-DC can be
    used with multi-plateau signals such as :func:`slcd.signals.staircase`.

    Parameters
    ----------
    G : NetworkX graph
        Undirected graph. Edge weights must be non-negative.
    K : float, default 4.0
        Interaction gain.
    R : float, default 20.0
        Initial confidence radius.
    rho : float, default 0.99
        Decay rate of the confidence radius, in :math:`(0, 1)`.
    n_retry : int, default 50
        Number of independent realizations simulated simultaneously in each
        round; an edge is cut as soon as one of them exceeds the radius.
    h : float, default 0.5
        Euler step size.
    T : float, default 300.0
        Time horizon of each round.
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
    rnp_communities, ser_communities

    Notes
    -----
    The balance between ``K`` (speed of local consensus) and ``rho`` (speed of
    the threshold decay) matters: a fast decay with a weak gain cuts edges
    before communities have synchronized, leading to over-segmentation.

    References
    ----------
    .. [1] A. Couthures, A. V. Jayakumar, V. S. Varma, I.-C. Morarescu,
       S. Lasaulce and A. Girard, "Social learning community detection with
       nonlinear interaction".

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.barbell_graph(5, 0)
    >>> sorted(map(sorted, rnp_dc_communities(G, seed=0)))
    [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    """
    _check(K, n_retry)
    if not 0 < rho < 1:
        raise ValueError("rho must lie in (0, 1)")
    nodes, A = to_adjacency(G, weight)
    active, _, _ = rnp_dc_active(
        A, K=K, h=h, T=T, R=R, rho=rho, n_retry=n_retry, signal=signal, rng=check_random_state(seed)
    )
    return labels_to_communities(nodes, component_labels(len(nodes), entry_rows(A), A.indices, active))

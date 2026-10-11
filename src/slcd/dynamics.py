"""The social learning opinion dynamics and its spectral splitting threshold."""

from __future__ import annotations

import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from networkx.utils import not_implemented_for

from slcd._core import Rng, Signal, check_random_state, rnp_dc_active, simulate, transition_matrix
from slcd._graph import to_adjacency

__all__ = ["opinion_dynamics", "spectral_threshold"]


@not_implemented_for("directed")
@not_implemented_for("multigraph")
def opinion_dynamics(
    G: nx.Graph,
    *,
    K: float = 4.0,
    x0: np.ndarray | None = None,
    n_runs: int = 1,
    h: float = 0.5,
    T: float = 300.0,
    signal: Signal | None = None,
    weight: str | None = "weight",
    seed: int | Rng | None = None,
    record: bool = False,
    confidence: tuple[float, float] | None = None,
) -> np.ndarray:
    r"""Simulate the social learning dynamics on ``G``.

    Integrates :math:`x(t+1) = x(t) + h (P s(x(t)) - x(t))` with
    :math:`P = D^{-1} A` and :math:`s(x) = f(K x)`.

    Parameters
    ----------
    G : NetworkX graph
        Undirected graph. Rows of the result follow the order of ``list(G)``.
    K : float, default 4.0
        Interaction gain.
    x0 : array_like or None, default None
        Initial opinions, shape ``(n,)`` or ``(n, n_runs)``. If None, they are
        drawn uniformly in :math:`[-1, 1]` for ``n_runs`` realizations.
    n_runs : int, default 1
        Number of realizations when ``x0`` is None.
    h : float, default 0.5
        Euler step size.
    T : float, default 300.0
        Time horizon (``round(T / h)`` steps).
    signal : callable or None, default None
        Unit signal function; ``None`` is the saturation.
    weight : str or None, default "weight"
        Edge attribute used as weight.
    seed : int, numpy.random.Generator or None, default None
        Seed or generator for the random initial opinions.
    record : bool, default False
        If True, return the whole trajectory instead of the final state.
        Integration then never stops early.
    confidence : (R, rho) or None, default None
        If given, simulate the decaying-confidence dynamics of RNP-DC
        (Eq. (B1)): at step :math:`t`, edges whose tension
        :math:`|s(x_i) - s(x_j)|` exceeds :math:`R \rho^t` (in any of the
        realizations) are removed for good.

    Returns
    -------
    ndarray
        Final opinions of shape ``(n,)`` (or ``(n, n_runs)`` when
        ``n_runs > 1`` or a 2-D ``x0`` is given). With ``record=True``, a
        trailing time axis of length ``round(T / h) + 1`` is added.

    Examples
    --------
    >>> import networkx as nx
    >>> x = opinion_dynamics(nx.complete_graph(4), K=4, seed=0)
    >>> bool(abs(x).min() > 0.99)  # consensus on a saturated state
    True
    """
    nodes, A = to_adjacency(G, weight)
    if x0 is None:
        rng = check_random_state(seed)
        shape = (len(nodes),) if n_runs == 1 else (len(nodes), n_runs)
        x0 = rng.uniform(-1.0, 1.0, size=shape)
    x0 = np.asarray(x0, dtype=float)
    if x0.shape[0] != len(nodes):
        raise ValueError(f"x0 must have {len(nodes)} rows, got shape {x0.shape}")
    if confidence is None:
        x, traj = simulate(transition_matrix(A), x0, K=K, h=h, T=T, signal=signal, record=record)
    else:
        R, rho = confidence
        _, x, traj = rnp_dc_active(
            A,
            K=K,
            h=h,
            T=T,
            R=R,
            rho=rho,
            n_retry=1,
            signal=signal,
            rng=check_random_state(seed),
            x0=x0,
            record=record,
        )
    return traj if traj is not None else x


@not_implemented_for("directed")
@not_implemented_for("multigraph")
def spectral_threshold(G: nx.Graph, *, weight: str | None = "weight") -> float:
    r"""Critical gain :math:`K^* = 1 / \lambda_{N-1}` above which ``G`` can split.

    :math:`\lambda_{N-1}` is the second-largest eigenvalue of the random-walk
    matrix :math:`P = D^{-1} A`. If :math:`K \lambda_{N-1} < 1` every
    trajectory converges to consensus, so community detection requires
    :math:`K > K^*`. Isolated nodes are ignored.

    Parameters
    ----------
    G : NetworkX graph
        Undirected graph with at least two non-isolated nodes.
    weight : str or None, default "weight"
        Edge attribute used as weight.

    Returns
    -------
    float
        :math:`K^*`; ``inf`` if :math:`\lambda_{N-1} \le 0` (e.g. complete
        graphs), meaning that the graph never splits.

    Examples
    --------
    >>> import networkx as nx
    >>> spectral_threshold(nx.complete_graph(5))
    inf
    >>> round(spectral_threshold(nx.cycle_graph(6)), 3)  # lambda = cos(pi/3)
    2.0
    """
    H = G.subgraph(n for n in G if G.degree(n) > 0)
    if H.number_of_nodes() < 2:
        raise ValueError("G must have at least two non-isolated nodes")
    _, A = to_adjacency(H, weight)
    d_inv_sqrt = 1.0 / np.sqrt(A.sum(axis=1))
    S = sp.diags(d_inv_sqrt) @ A @ sp.diags(d_inv_sqrt)  # similar to P, symmetric
    n = S.shape[0]
    if n <= 500:
        eigs = np.linalg.eigvalsh(S.toarray())
    else:
        eigs = spla.eigsh(S, k=2, which="LA", return_eigenvectors=False)
    lam = float(np.sort(eigs)[-2])
    return 1.0 / lam if lam > 1e-12 else float("inf")

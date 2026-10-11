"""Array-level numerical core shared by all algorithms.

Everything in this module works on square SciPy CSR arrays indexed by
``0..n-1``. Graph-library front-ends (NetworkX in :mod:`slcd.ser` and
:mod:`slcd.rnp`) only convert their input to such an array and map the integer
labels back to nodes, so other front-ends (igraph, cdlib, ...) can reuse this
module unchanged.

The opinion dynamics is the forward-Euler discretization (Eq. (2) of the paper)

    x(t+1) = x(t) + h * (P s(x(t)) - x(t)),        P = D^{-1} A,

simulated for many independent initial conditions at once: ``x`` has shape
``(n, batch)`` and each column is one realization.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

Signal = Callable[[np.ndarray], np.ndarray]
Rng = np.random.Generator | np.random.RandomState

#: Convergence is tested every ``_CHECK_EVERY`` Euler steps.
_CHECK_EVERY = 10
#: Upper bound on the number of floats in an ``(n_entries, batch)`` work array.
_MAX_WORK = 20_000_000


def check_random_state(seed: int | Rng | None) -> Rng:
    """Turn ``seed`` into a NumPy random generator."""
    if seed is None or isinstance(seed, int | np.integer):
        return np.random.default_rng(seed)
    if isinstance(seed, np.random.Generator | np.random.RandomState):
        return seed
    raise TypeError(f"seed must be None, an int or a NumPy random generator, got {seed!r}")


def validate_adjacency(A: sp.sparray | sp.spmatrix) -> sp.csr_array:
    """Return ``A`` as a float CSR array, checking it is square and non-negative."""
    A = sp.csr_array(A, dtype=float)
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"adjacency matrix must be square, got shape {A.shape}")
    A.sum_duplicates()
    if A.nnz and A.data.min() < 0:
        raise ValueError("edge weights must be non-negative")
    return A


def entry_rows(A: sp.csr_array) -> np.ndarray:
    """Row index of every stored entry of the CSR array ``A``."""
    return np.repeat(np.arange(A.shape[0]), np.diff(A.indptr))


def row_normalized_data(A: sp.csr_array, rows: np.ndarray, active: np.ndarray | None = None) -> np.ndarray:
    """Data array of ``P = D^{-1} A`` restricted to the ``active`` entries.

    Rows without any active entry (isolated agents) are left at zero, so the
    corresponding agents simply relax towards ``x = 0``.
    """
    w = A.data if active is None else np.where(active, A.data, 0.0)
    deg = np.bincount(rows, weights=w, minlength=A.shape[0])
    deg[deg == 0] = 1.0
    return w / deg[rows]


def transition_matrix(A: sp.csr_array) -> sp.csr_array:
    """Row-stochastic matrix ``P = D^{-1} A`` (zero rows for isolated nodes)."""
    P = A.copy()
    P.data = row_normalized_data(A, entry_rows(A))
    return P


def apply_signal(x: np.ndarray, K: float, signal: Signal | None, out: np.ndarray | None = None) -> np.ndarray:
    """Public actions ``s(x) = f(K x)``; ``signal=None`` is the saturation."""
    if signal is None:
        out = np.multiply(x, K, out=out)
        return np.clip(out, -1.0, 1.0, out=out)
    return np.asarray(signal(K * x), dtype=float)


def simulate(
    P: sp.csr_array,
    x0: np.ndarray,
    *,
    K: float,
    h: float,
    T: float,
    signal: Signal | None = None,
    tol: float = 1e-4,
    record: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Integrate the opinion dynamics from ``x0`` for ``round(T / h)`` steps.

    Parameters
    ----------
    P : csr_array
        Row-stochastic interaction matrix.
    x0 : ndarray, shape (n,) or (n, batch)
        Initial opinions (each column is an independent realization).
    K, h, T : float
        Interaction gain, step size and time horizon.
    signal : callable or None
        Unit signal function, see :mod:`slcd.signals`.
    tol : float
        Integration stops early once ``max |P s(x) - x| < tol`` (checked every
        10 steps). Ignored when ``record`` is True.
    record : bool
        If True, also return the full trajectory, shape ``x0.shape + (steps+1,)``.

    Returns
    -------
    x : ndarray
        Final opinions.
    trajectory : ndarray or None
        The trajectory if ``record`` is True.
    """
    x = np.array(x0, dtype=float, copy=True)
    n_steps = int(round(T / h))
    sx = np.empty_like(x)
    traj: list[np.ndarray] = [x.copy()]
    for k in range(n_steps):
        sx = apply_signal(x, K, signal, out=sx)
        drift = P @ sx
        drift -= x
        x += h * drift
        if record:
            traj.append(x.copy())
        elif k % _CHECK_EVERY == 0 and np.max(np.abs(drift), initial=0.0) < tol:
            break
    return x, (np.stack(traj, axis=-1) if record else None)


def component_labels(n: int, rows: np.ndarray, cols: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Weakly connected component label of each node, using kept entries only."""
    M = sp.csr_array((np.ones(int(keep.sum())), (rows[keep], cols[keep])), shape=(n, n))
    _, labels = connected_components(M, directed=True, connection="weak")
    return labels


def _chunk_size(n_entries: int, n_runs: int) -> int:
    return max(1, min(n_runs, _MAX_WORK // max(n_entries, 1)))


# ---------------------------------------------------------------------------
# Score-based Edge Reliability (SER) -- Algorithm 3
# ---------------------------------------------------------------------------


def ser_scores(
    A: sp.csr_array,
    *,
    K: float,
    h: float,
    T: float,
    n_runs: int,
    epsilon: float,
    mode: str,
    signal: Signal | None,
    rng: Rng,
    tol: float = 1e-4,
) -> np.ndarray:
    """Reliability score of every stored entry ``(i, j)`` of ``A``.

    The score is the fraction of the ``n_runs`` realizations in which the
    steady-state tension on the entry is below ``epsilon``:

    * ``mode="action"``:  ``|s(x_i) - s(x_j)|`` (symmetric);
    * ``mode="opinion"``: ``|x_i - s(x_j)|``, the tension felt by ``i`` from
      the public action of ``j`` (asymmetric).

    Returns an array aligned with ``A.data``.
    """
    if mode not in ("action", "opinion"):
        raise ValueError(f"mode must be 'action' or 'opinion', got {mode!r}")
    n = A.shape[0]
    rows, cols = entry_rows(A), A.indices
    P = transition_matrix(A)
    counts = np.zeros(A.nnz)
    chunk = _chunk_size(A.nnz, n_runs)
    done = 0
    while done < n_runs:
        b = min(chunk, n_runs - done)
        x, _ = simulate(P, rng.uniform(-1.0, 1.0, size=(n, b)), K=K, h=h, T=T, signal=signal, tol=tol)
        s = apply_signal(x, K, signal)
        left = s if mode == "action" else x
        counts += np.count_nonzero(np.abs(left[rows] - s[cols]) < epsilon, axis=1)
        done += b
    return counts / n_runs


# ---------------------------------------------------------------------------
# Recursive Neighbor Pruning (RNP) -- Algorithm 1
# ---------------------------------------------------------------------------


def rnp_active(
    A: sp.csr_array,
    *,
    K: float,
    h: float,
    T: float,
    n_retry: int,
    signal: Signal | None,
    rng: Rng,
    tol: float = 1e-4,
) -> np.ndarray:
    """Boolean mask of the entries of ``A`` that survive RNP.

    Each round simulates ``n_retry`` independent realizations on the current
    (pruned) graph and removes every edge whose endpoints end with public
    actions of opposite signs in at least one realization. Rounds are repeated
    until a round prunes nothing; since every split graph is simulated as a
    whole, this is equivalent to recursing on each connected component.
    """
    n = A.shape[0]
    rows, cols = entry_rows(A), A.indices
    active = np.ones(A.nnz, dtype=bool)
    P = A.copy()
    while active.any():
        P.data = row_normalized_data(A, rows, active)
        x, _ = simulate(P, rng.uniform(-1.0, 1.0, size=(n, n_retry)), K=K, h=h, T=T, signal=signal, tol=tol)
        s = apply_signal(x, K, signal)
        idx = np.flatnonzero(active)
        cut = np.any(s[rows[idx]] * s[cols[idx]] < 0, axis=1)
        if not cut.any():
            break
        active[idx[cut]] = False
    return active


# ---------------------------------------------------------------------------
# RNP with Decaying Confidence (RNP-DC) -- Algorithm 2
# ---------------------------------------------------------------------------


def rnp_dc_active(
    A: sp.csr_array,
    *,
    K: float,
    h: float,
    T: float,
    R: float,
    rho: float,
    n_retry: int,
    signal: Signal | None,
    rng: Rng,
    tol: float = 1e-4,
    x0: np.ndarray | None = None,
    record: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Boolean mask of the entries of ``A`` that survive RNP-DC.

    During the integration, at step ``t`` every edge whose tension
    ``|s(x_i) - s(x_j)|`` exceeds the confidence radius ``R * rho**t`` in at
    least one of the ``n_retry`` realizations is removed, and ``P`` is
    renormalized on the fly (Eq. (B1)). Rounds with fresh initial conditions
    are repeated until a whole round prunes nothing.

    If ``x0`` is given, only one round is run from it (used to simulate the
    decaying-confidence dynamics itself).

    Returns the mask, the final opinions of the last round and, if ``record``
    is True, the trajectory of the last round.
    """
    n = A.shape[0]
    rows, cols = entry_rows(A), A.indices
    active = np.ones(A.nnz, dtype=bool)
    P = A.copy()
    P.data = row_normalized_data(A, rows, active)
    n_steps = int(round(T / h))
    traj: list[np.ndarray] = []
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)
    while active.any():
        x = rng.uniform(-1.0, 1.0, size=(n, n_retry)) if x0 is None else np.array(x0, dtype=float)
        traj = [x.copy()]
        sx = np.empty_like(x)
        radius = R
        pruned = False
        for k in range(n_steps):
            sx = apply_signal(x, K, signal, out=sx)
            # |s_i - s_j| <= 2 max|s|: no edge can be cut while the radius is larger.
            if radius < 2.0 * np.max(np.abs(sx), initial=0.0):
                idx = np.flatnonzero(active)
                cut = np.abs(sx[rows[idx]] - sx[cols[idx]]) > radius
                cut = cut.any(axis=1) if cut.ndim > 1 else cut
                if cut.any():
                    active[idx[cut]] = False
                    P.data = row_normalized_data(A, rows, active)
                    pruned = True
            drift = P @ sx
            drift -= x
            x += h * drift
            if record:
                traj.append(x.copy())
            radius *= rho
            if (
                not record
                and radius < 1e-3
                and k % _CHECK_EVERY == 0
                and np.max(np.abs(drift), initial=0.0) < tol
            ):
                break
        if not pruned or x0 is not None:
            break
    return active, x, (np.stack(traj, axis=-1) if record and traj else None)

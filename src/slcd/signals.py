"""Signal (communication) functions.

In the social learning model every agent broadcasts the public action
``s(x_i) = f(K * x_i)`` where ``x_i`` is its private opinion, ``K > 0`` is the
interaction gain and ``f`` is a non-decreasing function mapping onto
``[-1, 1]`` with unit slope at the origin. This module provides such unit
functions ``f``; the gain ``K`` is always passed separately to the algorithms.

Any callable ``f(z: np.ndarray) -> np.ndarray`` with these properties can be
used as the ``signal`` argument of the algorithms.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

__all__ = ["saturation", "tanh", "staircase"]

Signal = Callable[[np.ndarray], np.ndarray]


def saturation(z: np.ndarray) -> np.ndarray:
    """Piecewise-linear saturation ``f(z) = min(1, max(-1, z))``.

    This is the default signal of every algorithm.

    Examples
    --------
    >>> import numpy as np
    >>> saturation(np.array([-3.0, 0.5, 2.0]))
    array([-1. ,  0.5,  1. ])
    """
    return np.clip(z, -1.0, 1.0)


def tanh(z: np.ndarray) -> np.ndarray:
    """Smooth sigmoidal signal ``f(z) = tanh(z)``.

    Examples
    --------
    >>> import numpy as np
    >>> float(tanh(np.array(0.0)))
    0.0
    """
    return np.tanh(z)


def staircase(n_levels: int = 3, delta: float = 0.01) -> Signal:
    """Multi-plateau signal with ``n_levels`` stable fixed points.

    The interval ``[-1, 1]`` is split into ``n_levels`` bins of width
    ``w = 2 / n_levels``; ``f`` maps each bin to its midpoint. Points within
    ``delta`` of a bin boundary are mapped to the midpoint of the lower bin.
    This signal is used with RNP-DC to let communities synchronize on
    different plateaus.

    Parameters
    ----------
    n_levels : int, default 3
        Number of plateaus (stable fixed points).
    delta : float, default 0.01
        Half-width of the transition region around each boundary.

    Returns
    -------
    callable
        The signal function ``f``.

    Examples
    --------
    >>> import numpy as np
    >>> f = staircase(3)
    >>> np.round(f(np.array([-1.0, 0.0, 1.0])), 3)
    array([-0.667,  0.   ,  0.667])
    """
    if n_levels < 1:
        raise ValueError("n_levels must be a positive integer")
    width = 2.0 / n_levels
    delta = min(delta, width / 2)

    def f(z: np.ndarray) -> np.ndarray:
        zc = np.clip(z, -1.0, 1.0)
        idx = np.clip(np.floor((zc + 1.0) / width), 0, n_levels - 1)
        out = -1.0 + (idx + 0.5) * width
        for k in range(1, n_levels):
            boundary = -1.0 + k * width
            out = np.where(np.abs(zc - boundary) < delta, -1.0 + (k - 0.5) * width, out)
        return out

    return f

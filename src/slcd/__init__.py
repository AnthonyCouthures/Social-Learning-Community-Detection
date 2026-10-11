"""Social learning community detection.

Decentralized community detection through nonlinear opinion dynamics:
agents exchange saturated public actions, the network polarizes along its
sparsest cuts, and communities are read off the steady states.

Algorithms
----------
ser_communities
    Score-based Edge Reliability (SER): statistical edge filtering over many
    independent realizations. Recommended default.
rnp_communities
    Recursive Neighbor Pruning (RNP).
rnp_dc_communities
    RNP with Decaying Confidence (RNP-DC).

Tools
-----
ser_edge_reliability
    Reliability score of every edge.
opinion_dynamics
    Simulate the underlying dynamics.
spectral_threshold
    Critical interaction gain K* above which a graph can split.
"""

from slcd import signals
from slcd.dynamics import opinion_dynamics, spectral_threshold
from slcd.rnp import rnp_communities, rnp_dc_communities
from slcd.ser import ser_communities, ser_edge_reliability

__version__ = "0.1.0"

__all__ = [
    "ser_communities",
    "ser_edge_reliability",
    "rnp_communities",
    "rnp_dc_communities",
    "opinion_dynamics",
    "spectral_threshold",
    "signals",
    "__version__",
]

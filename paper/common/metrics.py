"""Partition quality metrics used in the benchmark figures."""

from __future__ import annotations

import contextlib
from collections.abc import Hashable, Iterable

import networkx as nx
import numpy as np
from sklearn.metrics import normalized_mutual_info_score


def labels(nodes: list[Hashable], communities: Iterable[Iterable[Hashable]]) -> np.ndarray:
    """Integer community label of each node of ``nodes``."""
    label_of = {v: i for i, c in enumerate(communities) for v in c}
    return np.array([label_of[v] for v in nodes])


def nmi(G: nx.Graph, truth, predicted) -> float:
    """Normalized mutual information between two partitions of ``G``."""
    nodes = list(G)
    return float(normalized_mutual_info_score(labels(nodes, truth), labels(nodes, predicted)))


def relative_segmentation_error(truth, predicted) -> float:
    """``(M_pred - M_truth) / M_truth`` where ``M`` is the number of communities."""
    return (len(predicted) - len(truth)) / len(truth)


def average_conductance(G: nx.Graph, communities) -> float:
    """Mean conductance over the communities (the whole graph is skipped)."""
    values = []
    for c in communities:
        if 0 < len(c) < len(G):
            with contextlib.suppress(ZeroDivisionError):
                values.append(nx.conductance(G, c))
    return float(np.mean(values)) if values else 0.0


def evaluate(G: nx.Graph, truth, predicted) -> dict[str, float]:
    """All the metrics of Fig. 4 for one partition."""
    return {
        "nmi": nmi(G, truth, predicted),
        "seg_error": relative_segmentation_error(truth, predicted),
        "modularity": nx.community.modularity(G, predicted),
        "conductance": average_conductance(G, predicted),
        "n_communities": len(predicted),
    }

import networkx as nx
import numpy as np
import pytest

import slcd
from slcd import rnp_communities, rnp_dc_communities, ser_communities, ser_edge_reliability
from slcd._core import rnp_sequential_active
from slcd.signals import staircase

ALGORITHMS = [
    pytest.param(lambda G, **kw: ser_communities(G, n_runs=50, **kw), id="ser"),
    pytest.param(rnp_communities, id="rnp"),
    pytest.param(lambda G, **kw: rnp_communities(G, mode="sequential", **kw), id="rnp_seq"),
    pytest.param(rnp_dc_communities, id="rnp_dc"),
]


def as_partition(communities):
    return {frozenset(c) for c in communities}


def planted_sbm(seed=1):
    sizes = [40, 40, 40]
    p = [[0.5, 0.01, 0.01], [0.01, 0.5, 0.01], [0.01, 0.01, 0.5]]
    G = nx.stochastic_block_model(sizes, p, seed=seed)
    truth = {frozenset(b) for b in G.graph["partition"]}
    return G, truth


@pytest.mark.parametrize("algo", ALGORITHMS)
def test_barbell_split(algo):
    G = nx.barbell_graph(6, 0)
    assert as_partition(algo(G, seed=0)) == {frozenset(range(6)), frozenset(range(6, 12))}


@pytest.mark.parametrize("algo", ALGORITHMS)
def test_planted_partition_recovered(algo):
    G, truth = planted_sbm()
    assert as_partition(algo(G, seed=0)) == truth


@pytest.mark.parametrize("algo", ALGORITHMS)
def test_seed_reproducibility(algo):
    G = nx.karate_club_graph()
    assert as_partition(algo(G, seed=3)) == as_partition(algo(G, seed=3))
    rng1, rng2 = np.random.default_rng(7), np.random.default_rng(7)
    assert as_partition(algo(G, seed=rng1)) == as_partition(algo(G, seed=rng2))


@pytest.mark.parametrize("algo", ALGORITHMS)
def test_result_is_partition(algo):
    G = nx.karate_club_graph()
    G.add_node("isolated")
    communities = algo(G, seed=0)
    assert nx.community.is_partition(G, communities)
    assert {"isolated"} in communities


@pytest.mark.parametrize("algo", ALGORITHMS)
def test_string_labels(algo):
    G = nx.relabel_nodes(nx.barbell_graph(5, 0), lambda i: f"n{i}")
    assert as_partition(algo(G, seed=0)) == {
        frozenset(f"n{i}" for i in range(5)),
        frozenset(f"n{i}" for i in range(5, 10)),
    }


@pytest.mark.parametrize("algo", [p for p in ALGORITHMS if p.id not in ("rnp", "rnp_seq")])
def test_below_spectral_threshold_gives_consensus(algo):
    # RNP is excluded: below the threshold all states decay to 0 and the sign
    # test of RNP is meaningless (documented in its Notes).
    G, _ = planted_sbm()
    K_star = slcd.spectral_threshold(G)
    assert K_star > 1
    assert len(algo(G, K=0.5 * K_star, seed=0)) == 1


@pytest.mark.parametrize("algo", ALGORITHMS)
def test_empty_graph(algo):
    assert algo(nx.Graph(), seed=0) == []
    assert as_partition(algo(nx.empty_graph(3), seed=0)) == {frozenset({i}) for i in range(3)}


@pytest.mark.parametrize("algo", ALGORITHMS)
@pytest.mark.parametrize("graph", [nx.DiGraph([(0, 1)]), nx.MultiGraph([(0, 1)])])
def test_not_implemented(algo, graph):
    with pytest.raises(nx.NetworkXNotImplemented):
        algo(graph, seed=0)


@pytest.mark.parametrize("algo", ALGORITHMS)
def test_negative_weight_rejected(algo):
    G = nx.Graph()
    G.add_edge(0, 1, weight=-1.0)
    with pytest.raises(ValueError, match="non-negative"):
        algo(G, seed=0)


@pytest.mark.parametrize("algo", ALGORITHMS)
def test_weights_are_used(algo):
    # Two cliques joined by a bridge: unweighted they split, but a very heavy
    # bridge glues its endpoints together. (h=0.1: with a large step, a pair of
    # strongly coupled agents can lock into a period-2 orbit of the Euler map.)
    G = nx.barbell_graph(6, 0)
    nx.set_edge_attributes(G, 1.0, "weight")
    G[5][6]["weight"] = 1000.0
    assert not any({5, 6} <= c for c in algo(G, weight=None, h=0.1, seed=0))
    assert any({5, 6} <= c for c in algo(G, h=0.1, seed=0))


def test_ser_reliability_modes():
    G = nx.barbell_graph(5, 0)
    action = ser_edge_reliability(G, n_runs=40, seed=0)
    assert set(action) == set(G.edges())
    assert all(0.0 <= p <= 1.0 for p in action.values())
    opinion = ser_edge_reliability(G, n_runs=40, mode="opinion", seed=0)
    assert set(opinion) == set(G.edges()) | {(v, u) for u, v in G.edges()}


def test_ser_tau_monotone():
    G = nx.karate_club_graph()
    n_strict = len(ser_communities(G, K=5, tau=1.0, epsilon=0.2, n_runs=60, seed=0))
    n_loose = len(ser_communities(G, K=5, tau=0.5, epsilon=0.2, n_runs=60, seed=0))
    assert n_loose <= n_strict


def test_invalid_parameters():
    G = nx.karate_club_graph()
    with pytest.raises(ValueError):
        ser_communities(G, mode="bogus")
    with pytest.raises(ValueError):
        rnp_communities(G, K=-1)
    with pytest.raises(ValueError, match="mode"):
        rnp_communities(G, mode="bogus")
    with pytest.raises(ValueError):
        rnp_dc_communities(G, rho=1.5)


def test_rnp_dc_with_staircase_signal():
    G, truth = planted_sbm()
    communities = rnp_dc_communities(G, K=2.5, R=4.0, rho=0.98, h=0.1, signal=staircase(3), seed=0)
    assert as_partition(communities) == truth


@pytest.mark.parametrize(
    "graph",
    [nx.barbell_graph(6, 0), planted_sbm()[0], planted_sbm(seed=4)[0]],
    ids=["barbell", "sbm1", "sbm4"],
)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_rnp_modes_agree(graph, seed):
    batch = rnp_communities(graph, mode="batch", seed=seed)
    sequential = rnp_communities(graph, mode="sequential", seed=seed)
    assert as_partition(batch) == as_partition(sequential)


def test_rnp_sequential_counts_without_cut():
    # A complete graph never splits: exactly n_retry rounds of round(T / h)
    # steps, one message per step and per directed edge.
    G = nx.complete_graph(6)
    A = nx.to_scipy_sparse_array(G, format="csr", dtype=float)
    rng = np.random.default_rng(0)
    active, n_rounds, n_messages = rnp_sequential_active(
        A, K=4.0, h=0.5, T=20.0, n_retry=7, signal=None, rng=rng
    )
    assert active.all()
    assert n_rounds == 7
    assert n_messages == 7 * 40 * A.nnz


def test_rnp_sequential_counts_with_cut():
    # Only the bridge is cut, in a single round. Rounds up to that one run on
    # the whole graph; the last n_retry rounds run without the bridge.
    G = nx.barbell_graph(6, 0)
    A = nx.to_scipy_sparse_array(G, format="csr", dtype=float)
    n_retry, steps = 5, 100
    active, n_rounds, n_messages = rnp_sequential_active(
        A, K=4.0, h=0.5, T=50.0, n_retry=n_retry, signal=None, rng=np.random.default_rng(0)
    )
    assert active.sum() == A.nnz - 2
    assert n_rounds > n_retry
    assert n_messages == steps * ((n_rounds - n_retry) * A.nnz + n_retry * (A.nnz - 2))

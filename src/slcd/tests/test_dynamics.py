import networkx as nx
import numpy as np
import pytest

from slcd import opinion_dynamics, spectral_threshold
from slcd.signals import saturation, staircase, tanh


def test_trajectory_shape_and_start():
    G = nx.path_graph(5)
    x0 = np.linspace(-1, 1, 5)
    traj = opinion_dynamics(G, x0=x0, h=0.1, T=2.0, record=True)
    assert traj.shape == (5, 21)
    np.testing.assert_allclose(traj[:, 0], x0)


def test_euler_step_matches_definition():
    G = nx.Graph([(0, 1), (1, 2)])
    x0 = np.array([0.1, -0.2, 0.05])
    K, h = 3.0, 0.5
    x1 = opinion_dynamics(G, x0=x0, K=K, h=h, T=h)
    s = np.clip(K * x0, -1, 1)
    P = np.array([[0, 1, 0], [0.5, 0, 0.5], [0, 1, 0]])
    np.testing.assert_allclose(x1, x0 + h * (P @ s - x0))


def test_multiple_runs_shape():
    x = opinion_dynamics(nx.karate_club_graph(), n_runs=7, seed=0)
    assert x.shape == (34, 7)


def test_spectral_threshold_matches_dense_eigenvalue():
    G = nx.karate_club_graph()
    A = nx.to_numpy_array(G)
    P = A / A.sum(axis=1, keepdims=True)
    lam = np.sort(np.linalg.eigvals(P).real)[-2]
    assert spectral_threshold(G) == pytest.approx(1 / lam)


def test_spectral_threshold_large_graph_uses_sparse_solver():
    G = nx.random_regular_graph(4, 600, seed=0)
    A = nx.to_numpy_array(G)
    lam = np.sort(np.linalg.eigvalsh(A / 4))[-2]
    assert spectral_threshold(G) == pytest.approx(1 / lam, rel=1e-6)


def test_signals_are_odd_and_bounded():
    z = np.linspace(-3, 3, 61)
    for f in (saturation, tanh, staircase(3), staircase(4)):
        out = f(z)
        assert np.all(np.abs(out) <= 1)
        assert np.all(np.diff(out) >= -1e-12)
    np.testing.assert_allclose(saturation(z), -saturation(-z))

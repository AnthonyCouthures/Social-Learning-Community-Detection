"""Loaders for the real-world networks of the paper (see ``paper/data/README.md``)."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

FOOTBALL_CONFERENCES = {
    0: "Atlantic Coast",
    1: "Big East",
    2: "Big Ten",
    3: "Big Twelve",
    4: "Conference USA",
    5: "Independents",
    6: "Mid-American",
    7: "Mountain West",
    8: "Pacific Ten",
    9: "Southeastern",
    10: "Sun Belt",
    11: "Western Athletic",
}

#: Ngogo faction labels of ``combined_membership_labels.csv``.
CHIMP_FACTIONS = {-1: "No label", 0: "Central group", 1: "Western group"}


def load_karate() -> nx.Graph:
    """Zachary's karate club (unweighted, as bundled with NetworkX)."""
    G = nx.karate_club_graph()
    nx.set_edge_attributes(G, 1.0, "weight")
    return G


def load_football() -> tuple[nx.Graph, dict[str, int]]:
    """US college football 2000 (Girvan & Newman), nodes are team names.

    Returns the graph and the conference id of each team.
    """
    raw = nx.read_gml(DATA_DIR / "football" / "football.gml", label="id")
    names = {i: attrs["label"] for i, attrs in raw.nodes(data=True)}
    conference = {names[i]: int(attrs["value"]) for i, attrs in raw.nodes(data=True)}
    G = nx.Graph(nx.relabel_nodes(raw, names))
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    G.remove_nodes_from(list(nx.isolates(G)))
    for v in G:
        G.nodes[v].clear()
    return G, {v: conference[v] for v in G}


def load_chimpanzees(year: int = 2015) -> tuple[nx.Graph, dict[str, int]]:
    """Ngogo chimpanzee proximity network of ``year`` (weighted).

    Nodes are chimpanzee identifiers; weights are the entries of the published
    adjacency matrix. Self-loops and inactive individuals are removed.
    Returns the graph and the 2018 faction label of each individual
    (see :data:`CHIMP_FACTIONS`).
    """
    adjacency = np.loadtxt(DATA_DIR / "chimpanzees" / f"A_{year}.csv", delimiter=",")
    members = pd.read_csv(DATA_DIR / "chimpanzees" / "combined_membership_labels.csv")
    members.columns = ["index", "name", "label"]
    members["name"] = members["name"].astype(str).str.strip().str.lower()
    names = dict(zip(members["index"].astype(int), members["name"], strict=True))
    G = nx.relabel_nodes(nx.from_numpy_array(adjacency), names)
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    G.remove_nodes_from(list(nx.isolates(G)))
    faction = dict(zip(members["name"], members["label"].astype(int), strict=True))
    return G, {v: faction.get(v, -1) for v in G}

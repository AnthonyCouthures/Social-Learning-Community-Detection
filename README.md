# Social Learning Community Detection

[![CI](https://github.com/AnthonyCouthures/Social-Learning-Community-Detection/actions/workflows/ci.yml/badge.svg)](https://github.com/AnthonyCouthures/Social-Learning-Community-Detection/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23317344.svg)](https://doi.org/10.5281/zenodo.23317344)

**Decentralized community detection with nonlinear opinion dynamics.** Every node is an agent with a private opinion.
It only observes the saturated *public actions* of its neighbors. Above a spectral threshold, the network breaks
symmetry and polarizes along its sparsest cuts, and the communities can be read from the steady states. No agent
needs a global view of the graph or has to reveal its private opinion.

This repository has two parts:

1. **`slcd`**, a small Python library with a NetworkX-style API. Its only dependencies are NumPy, SciPy and NetworkX.
2. **`paper/`**, scripts that reproduce the figures of the papers. `paper/pre/` covers every figure of the companion
   paper below; `paper/ecc2027/` covers the numerical section of a conference paper on RNP.

> A. Couthures, A. V. Jayakumar, V. S. Varma, I.-C. Morărescu, S. Lasaulce and A. Girard,
> *Social learning community detection with nonlinear interaction*.

```python
import networkx as nx
import slcd

G = nx.karate_club_graph()
communities = slcd.ser_communities(G, K=5, seed=0)   # list of sets of nodes, like nx.community.*
print(nx.community.modularity(G, communities))
```

---

## 1. Scientific context

Agent $i$ holds an opinion $x_i \in [-1, 1]$ and broadcasts the action $s(x_i)$, where $s$ is a non-decreasing,
saturating signal with gain $K = s'(0)$, for instance $s(x) = \min(1, \max(-1, Kx))$. Opinions follow the
discrete-time social learning dynamics

$$
\Large x(t+1) = x(t) + h\big(P\, s(x(t)) - x(t)\big), \qquad P = D^{-1}A,
$$

that is, each agent moves towards the average action of its neighbors.

If $K\lambda_{N-1}(P) < 1$, every trajectory reaches consensus. Above this threshold the origin becomes unstable and
the network splits along the Fiedler direction. Dense cores then lock into forward-invariant hypercubes. Three
algorithms turn this symmetry breaking into a partition:

| Algorithm | Function | Principle |
|---|---|---|
| **SER**: Score-based Edge Reliability | `ser_communities` | Run the dynamics $M$ times from random opinions. An edge is reliable if its steady-state tension is below $\varepsilon$ in a fraction $\geq \tau$ of the runs. Communities are the connected components of the reliable edges. *Recommended.* |
| **RNP**: Recursive Neighbor Pruning | `rnp_communities` | After polarization, cut every edge whose endpoints have actions of opposite signs. Repeat until nothing is cut. |
| **RNP-DC**: RNP with Decaying Confidence | `rnp_dc_communities` | Cut edges *during* the transient when the tension exceeds a shrinking radius $R\rho^t$. Supports multi-plateau signals. |

## 2. Installation

```bash
git clone https://github.com/AnthonyCouthures/Social-Learning-Community-Detection.git
cd Social-Learning-Community-Detection
pip install -e .
```

The library needs Python ≥ 3.10, NumPy, SciPy and NetworkX. For development, use `pip install -e .[dev]` (pytest,
ruff, mypy). The full environment used for the paper is described in [`environment.yml`](environment.yml).

## 3. API overview

All community functions take an undirected NetworkX graph `G` followed by keyword-only parameters. They return a
list of node sets, as `nx.community.louvain_communities` does. Edge weights are read from `weight="weight"`, and
`seed` accepts an int or a `numpy.random.Generator`.

| Function | Returns |
|---|---|
| `ser_communities(G, *, K=4, tau=1, epsilon=0.5, n_runs=200, mode="action", h=0.5, T=300, ...)` | partition (SER) |
| `ser_edge_reliability(G, *, K=4, epsilon=0.5, n_runs=200, mode="action", ...)` | `{(u, v): P_uv}`; with `mode="opinion"`, directed scores |
| `rnp_communities(G, *, K=4, n_retry=50, mode="batch", h=0.5, T=300, ...)` | partition (RNP); `mode="sequential"` runs one realization per round |
| `rnp_dc_communities(G, *, K=4, R=20, rho=0.99, n_retry=50, ...)` | partition (RNP-DC) |
| `opinion_dynamics(G, *, K=4, x0=None, n_runs=1, record=False, confidence=None, ...)` | final opinions or trajectories |
| `spectral_threshold(G)` | critical gain $K^* = 1/\lambda_{N-1}(P)$ |
| `signals.saturation`, `signals.tanh`, `signals.staircase(n_levels)` | signal functions (`signal=` argument) |

The defaults are the settings of Table I of the paper. Choose `K` above `spectral_threshold(G)`. For graphs with
clear community structure, `K = 4` works well with the saturated signal. Every function is documented with numpydoc
docstrings and runnable examples, for instance `help(slcd.ser_communities)`.

### Example: where does the karate club split?

```python
import networkx as nx
import slcd

G = nx.karate_club_graph()
P = slcd.ser_edge_reliability(G, K=5, epsilon=0.2, n_runs=500, h=0.1, seed=0)
weakest = sorted(P, key=P.get)[:5]   # least reliable ties: the fracture lines of the club
```

A short tour is available in [`notebooks/quickstart.ipynb`](notebooks/quickstart.ipynb).

## 4. Reproducing the paper

All figures of the paper are in [`paper/pre/figures/`](paper/pre/figures/). To regenerate them, open the notebook
[`paper/pre/reproduce_figures.ipynb`](paper/pre/reproduce_figures.ipynb), which shows every figure with a short
explanation, or run the scripts from a terminal:

```bash
pip install -e . -r paper/requirements.txt
python paper/pre/make_all.py            # all figures -> paper/pre/figures/
python paper/pre/make_all.py --quick    # fast smoke test
```

[`paper/pre/README.md`](paper/pre/README.md) maps each figure to its script, parameters and running time.
[`paper/ecc2027/README.md`](paper/ecc2027/README.md) does the same for the conference paper on RNP.

## 5. Repository structure

```
Social-Learning-Community-Detection/
├── src/slcd/                 # the library
│   ├── ser.py                #   SER (Alg. 3)
│   ├── rnp.py                #   RNP (Alg. 1) and RNP-DC (Alg. 2)
│   ├── dynamics.py           #   opinion dynamics, spectral threshold
│   ├── signals.py            #   signal functions
│   ├── _core.py              #   array-level numerics (graph-library independent)
│   └── tests/
├── paper/                    # reproduction of the paper figures
│   ├── common/               #   benchmarks (ABCD, LFR, SBM), baselines, metrics, plotting
│   ├── data/                 #   football and chimpanzee networks, with sources and licenses
│   ├── pre/                  #   companion paper
│   │   ├── reproduce_figures.ipynb
│   │   ├── fig01_*.py ... fig09_*.py, make_all.py
│   │   ├── figures/          #     the figures (PDF)
│   │   └── results/          #     raw results of the parameter sweeps
│   └── ecc2027/              #   conference paper on RNP (same layout)
├── notebooks/quickstart.ipynb
├── pyproject.toml, environment.yml, CITATION.cff, CHANGELOG.md, LICENSE
└── .github/workflows/ci.yml
```

## 6. Roadmap and contributing

The goal is to make these algorithms available where network scientists already work. The library is designed with
that integration in mind:

- **NetworkX.** The public functions follow the conventions of `networkx.algorithms.community`: graph first,
  keyword-only parameters, `weight` and `seed` arguments, a list of node sets as output, and numpydoc docstrings with
  doctests. They are meant to be proposed upstream, or exposed as a NetworkX backend.
- **Other libraries.** All the numerics live in `slcd._core` and work on SciPy sparse arrays only. A front-end for
  igraph or cdlib therefore only needs to convert its graph type.

To contribute, install with `pip install -e .[dev]`. `pytest` runs the unit tests and docstring examples, and
`ruff check`, `ruff format --check` and `mypy` must pass. CI runs all of them, plus a quick run of every figure
script. Issues and pull requests are welcome.

## 7. How to cite

If you use this code, please cite the paper and the archived software (see also [`CITATION.cff`](CITATION.cff)):

```bibtex
@article{couthures2026social,
  title  = {Social learning community detection with nonlinear interaction},
  author = {Couthures, Anthony and Jayakumar, Athira Varma and Varma, Vineeth Satheeskumar and
            Mor{\u{a}}rescu, Irinel-Constantin and Lasaulce, Samson and Girard, Antoine},
  note   = {Submitted},
  year   = {2026}
}
```

```bibtex
@software{couthures2026slcd,
  title     = {Social Learning Community Detection},
  author    = {Couthures, Anthony},
  year      = {2026},
  version   = {0.1.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23317344},
  url       = {https://doi.org/10.5281/zenodo.23317344}
}
```

The nonlinear consensus model behind it was studied in

```bibtex
@article{couthures2025global,
  title   = {Global Synchronization of Multi-Agent Systems With Nonlinear Interactions},
  author  = {Couthures, Anthony and Varma, Vineeth S. and Lasaulce, Samson and
             Mor{\u{a}}rescu, Irinel-Constantin},
  journal = {IEEE Control Systems Letters},
  volume  = {9},
  pages   = {354--359},
  year    = {2025},
  doi     = {10.1109/LCSYS.2025.3573563}
}
```

If you use the bundled datasets or the benchmark generators, please also cite their original sources, listed with
BibTeX entries in [`paper/data/README.md`](paper/data/README.md#how-to-cite).

## 8. Acknowledgements

The algorithms and experiments were first developed by the authors in research notebooks. The code was then
refactored into this package and the figure-reproduction scripts with the help of Claude (Anthropic), an AI coding
assistant. The refactored algorithms were checked against the original implementation, which they reproduce exactly
on identical random draws.

## 9. License

MIT. See [LICENSE](LICENSE). The bundled datasets keep their original terms; see
[`paper/data/README.md`](paper/data/README.md).

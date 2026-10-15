# Reproducing the paper

This folder reproduces every figure of *Social learning community detection with nonlinear interaction*. It is not
part of the installed library: it uses the benchmark generators, baselines, metrics and plotting of
[`../common/`](../common/) on top of `slcd`.

## Setup

From the repository root:

```bash
conda create -n slcd python=3.11
conda activate slcd
pip install -e .
pip install -r paper/requirements.txt
```

## Figures

The easiest way to browse the figures is the notebook [`reproduce_figures.ipynb`](reproduce_figures.ipynb). It runs
each script in turn and shows the result inline. The PDFs of all figures are also stored in
[`paper/pre/figures/`](figures/).

From a terminal, run everything with `python paper/pre/make_all.py`, or a subset with `python paper/pre/make_all.py --only 2 5`.
Every script accepts `--quick` (a small, fast run). The three parameter sweeps also accept `--n-jobs` (number of
worker processes) and `--plot-only`, which re-plots from the results stored in `paper/pre/results/`.

| Paper | Script | Content | Time* |
|---|---|---|---|
| Fig. 1 | `fig01_signal_functions.py` | Signal functions `tanh(Kx)` and `min(1, max(-1, Kx))` | seconds |
| Fig. 2 | `fig02_karate_ser.py` | SER reliability on the karate club, action vs. opinion tension | < 1 min |
| Fig. 3 | `fig03_spectral_thresholds.py` | Splitting thresholds K\* on LFR and ABCD (2 × 15 mixing values × 50 graphs) | ~ 5 min |
| Fig. 4 | `fig04_benchmark_abcd.py` | RNP, RNP-DC, SER vs. Louvain, Leiden, label propagation, spectral clustering on ABCD (31 ξ × 20 graphs) | ~ 20 min |
| Fig. 5 | `fig05_chimpanzees.py` | SER on the Ngogo chimpanzee network (2015) | < 1 min |
| Fig. 6 | `fig06_football.py` | SER on the college football network | < 1 min |
| Fig. 7 | `fig07_rnp_steps.py` | The steps of RNP on an SBM | seconds |
| Fig. 8 | `fig08_rnp_dc_multiplateau.py` | RNP-DC trajectories with a three-plateau signal | seconds |
| Fig. 9 | `fig09_rnp_dc_sensitivity.py` | RNP-DC sensitivity to (K, ρ) on ABCD | ~ 5 min |

\* On a 14-core laptop.

## Parameters

The parameters follow Table I and the figure captions. Each script lists them as constants at the top of the file.

- **Fig. 2.** `K = 5` (action) and `K = 8` (opinion), `h = 0.1`, `ε = 0.2`, 500 runs. The panels show the
  reliability `P_ij` of every edge.
- **Fig. 3.** `K*` is computed from the spectral gap of the normalized Laplacian, `1 / (1 - λ_{N-1}(P))`. The option
  `--definition paper` plots `1 / λ_{N-1}(P)` instead, which is the quantity returned by `slcd.spectral_threshold`.
- **Fig. 5.** `K = 4`, `h = 0.5`, `τ = 1`, `ε = 0.25`, 100 runs, on the weighted proximity network.
- **Fig. 7.** The dynamics uses the signal `s(x) = tanh(Kx)`.
- **Fig. 8.** Panel (b) shows the unit staircase `f`; the dynamics uses `s(x) = f(Kx)`.

## Options

LaTeX (with the `newtxtext` package) is used for text rendering when a `latex` executable is found. Set
`SLCD_USETEX=0` to use Matplotlib's built-in mathtext instead, and `SLCD_PNG=1` to also write PNG previews.

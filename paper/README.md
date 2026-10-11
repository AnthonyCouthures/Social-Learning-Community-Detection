# Reproducing the paper

This folder reproduces every figure of *Social learning community detection with nonlinear interaction*. It is not
part of the installed library: it adds benchmark generators, baselines, metrics and plotting on top of `slcd`.

## Setup

From the repository root:

```bash
conda create -n slcd python=3.11
conda activate slcd
pip install -e .
pip install -r paper/requirements.txt
```

LaTeX (with the `newtxtext` package) is used for text rendering when a `latex` executable is found. Set
`SLCD_USETEX=0` to force Matplotlib's built-in mathtext instead. Set `SLCD_PNG=1` to also write PNG previews.

## Figures

All outputs go to `paper/figures/`. Every script accepts `--quick` (small smoke-test run) and `--seed`. The heavy
sweeps also accept `--n-jobs` and `--plot-only`, which re-plots from the cached results in `paper/results/`.

| Paper | Script | Content | Time* |
|---|---|---|---|
| Fig. 1 | `fig01_signal_functions.py` | Signal functions `tanh(Kx)` and `min(1, max(-1, Kx))` | seconds |
| Fig. 2 | `fig02_karate_ser.py` | SER reliability on the karate club, action vs. opinion tension | < 1 min |
| Fig. 3 | `fig03_spectral_thresholds.py` | Splitting thresholds K\* on LFR and ABCD (2 × 15 mixing values × 50 graphs) | ~ 15 min |
| Fig. 4 | `fig04_benchmark_abcd.py` | RNP, RNP-DC, SER vs. Louvain, Leiden, label propagation, spectral clustering on ABCD (31 ξ × 20 graphs) | ~ 25 min |
| Fig. 5 | `fig05_chimpanzees.py` | SER on the Ngogo chimpanzee network (2015) | < 1 min |
| Fig. 6 | `fig06_football.py` | SER on the college football network | < 1 min |
| Fig. 7 | `fig07_rnp_steps.py` | The steps of RNP on an SBM | seconds |
| Fig. 8 | `fig08_rnp_dc_multiplateau.py` | RNP-DC trajectories with a three-plateau signal | seconds |
| Fig. 9 | `fig09_rnp_dc_sensitivity.py` | RNP-DC sensitivity to (K, ρ) on ABCD | ~ 20 min |

\* On a 14-core laptop. Run everything with `python paper/make_all.py`, or a subset with
`python paper/make_all.py --only 2 5`.

## Parameters

The parameters follow Table I and the figure captions. Each script states its parameters as constants at the top.
Every random draw is seeded: each ABCD/LFR graph uses its own seed, which is also passed to the algorithms.

## Differences with the original notebooks

The figures were first produced with exploratory notebooks. These scripts reimplement them, and the following points
differ from the published PDFs.

- **Seeds.** The original ABCD graphs and RNP-DC runs were not seeded. Figs. 4 and 9 therefore reproduce the
  published curves statistically, not point by point. Spring layouts may also differ slightly across NetworkX
  versions.
- **Fig. 3, definition of K\*.** The published figure plots the inverse spectral gap `1 / (1 - λ_{N-1}(P))` of the
  normalized Laplacian, although its legend and Sec. IV.A define `K* = 1 / λ_{N-1}(P)`. The script reproduces the
  published figure by default. `--definition paper` plots `1 / λ_{N-1}(P)`, which is also what
  `slcd.spectral_threshold` returns.
- **Fig. 2.** The panels were produced with `K = 5` (action) and `K = 8` (opinion), `h = 0.1`, `ε = 0.2` and 500
  runs. The panels show the reliability `P_ij` of every edge, so they do not depend on the threshold `τ` quoted in
  the caption.
- **Fig. 5.** The script uses the caption values (`K = 4`, `h = 0.5`, `τ = 1`, 100 runs) with `ε = 0.25`. SER runs
  on the weighted proximity network, as the original code did. The text describes the network as unweighted.
- **Fig. 7.** As in the original illustration, the dynamics uses `s(x) = tanh(Kx)`.
- **Fig. 8.** The time axis is the physical time `t = k h`; the original plot used the step index `k`. Panel (b)
  shows the unit staircase `f`, while the dynamics uses `s(x) = f(Kx)`.

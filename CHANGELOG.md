# Changelog

## 0.1.0 (unreleased)

First public version, accompanying the paper *Social learning community detection with nonlinear interaction*.

- `ser_communities`, `ser_edge_reliability`: Score-based Edge Reliability (action and opinion tension).
- `rnp_communities`: Recursive Neighbor Pruning. `mode="sequential"` runs one realization per round and stops after `n_retry` consecutive rounds without a cut, as the agents of a decentralized implementation would; the default `mode="batch"` is unchanged.
- `rnp_dc_communities`: RNP with Decaying Confidence, with pluggable (multi-plateau) signals.
- `opinion_dynamics`, `spectral_threshold`, `slcd.signals`.
- Scripts and a notebook (`paper/pre/reproduce_figures.ipynb`) reproducing every figure of the paper, with the figures in `paper/pre/figures/`.
- `paper/ecc2027/`: numerical results of the ECC 2027 paper on RNP (ABCD and LFR benchmarks against label propagation, a distributed spectral bisection and Louvain, with message counts).

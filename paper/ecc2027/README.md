# ECC 2027 paper: Recursive Neighbor Pruning

Numerical results of *Recursive Neighbor Pruning: Decentralized Community Detection with Certified Behavior from
Saturated Opinion Dynamics* (Sec. V). Setup as in [`paper/README.md`](../README.md).

RNP is Algorithm 1 of that paper: one realization per round, `T_max` Euler steps, stop after `N_retry` consecutive
rounds without a cut, i.e. `slcd.rnp_communities(G, mode="sequential", T=T_max * h, ...)`.

| Script | Content | Time* |
|---|---|---|
| `abcd_benchmark.py` | RNP (K = 2, 4, 8; N_retry = 5, 10, 20, 50), RNP-light, distributed spectral bisection, asynchronous label propagation, Louvain and Leiden on ABCD graphs (25 ξ × 20 graphs × 5 runs); K\*(G) and min_c K\*(G_c) | ~ 22 min |
| `lfr_benchmark.py` | RNP, distributed spectral bisection, asynchronous label propagation and Louvain on LFR graphs (`networkx`, same degree and community size ranges; 15 μ × 20 graphs × 5 runs), against the measured mean external ratio; K\*(G), min_c K\*(G_c), disconnected communities. Run after `abcd_benchmark.py` (gray ABCD curve) | ~ 8 min |

\* On a 14-core laptop.

```bash
python paper/ecc2027/abcd_benchmark.py            # full run, then figures and summary
python paper/ecc2027/abcd_benchmark.py --quick    # smoke test
python paper/ecc2027/abcd_benchmark.py --plot-only
```

Figures go to `paper/ecc2027/figures/`, results to `paper/ecc2027/results/`. The means quoted in the text are in
`results/ecc2027_abcd_summary.txt`.

## Parameters

- **Runs and bands.** Every randomized method is run 5 times on each graph. Curves show the mean over the graphs of
  the per-graph average, with a 95% confidence band (1.96 standard errors over the graphs).
- **Graphs.** ABCD, N = 1000, degree exponent 2.5 with degrees in [10, 100], community size exponent 1.5 with sizes
  in [20, 250]. Graph `t` of the `i`-th value of ξ uses the seed `42 + 1000 i + t`.
- **RNP.** Saturation `s(x) = sat(Kx)`, `h = 0.5`, `T_max = 300` steps, `N_retry = 50`, `K = 4` unless stated
  otherwise. RNP-light: `T_max = 50`, `N_retry = 10`.
- **Distributed spectral bisection.** 30 steps of the lazy walk per component; a split is kept only if it increases
  the modularity.
- **Messages.** One scalar per step and per edge direction (RNP, DSB), one label per sweep and per edge direction
  (label propagation), divided by the number of edges.

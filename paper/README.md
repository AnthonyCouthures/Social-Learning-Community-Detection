# Reproducing the papers

Scripts that reproduce the figures of the papers built on `slcd`. They are not part of the installed library.

| Folder | Paper |
|---|---|
| [`pre/`](pre/README.md) | *Social learning community detection with nonlinear interaction* (every figure) |
| [`ecc2027/`](ecc2027/README.md) | *Recursive Neighbor Pruning: Decentralized Community Detection with Certified Behavior from Saturated Opinion Dynamics* (Sec. V) |

Shared by both:

- [`common/`](common/): benchmark generators (ABCD, LFR, SBM), baselines, metrics and the plotting style;
- [`data/`](data/README.md): the real-world networks, with their sources and licenses;
- [`requirements.txt`](requirements.txt): the exact versions used for the figures.

Setup, from the repository root:

```bash
conda create -n slcd python=3.11
conda activate slcd
pip install -e .
pip install -r paper/requirements.txt
```

# Real-world datasets

Zachary's karate club is loaded from NetworkX (`nx.karate_club_graph()`). The two other networks are bundled here
unchanged; please cite the original sources when using them.

| Dataset | Files | Source | License / terms |
|---|---|---|---|
| US college football, regular season 2000 | `football/football.gml`, `football/football.txt` | M. Girvan and M. E. J. Newman, *Community structure in social and biological networks*, PNAS 99, 7821–7826 (2002); distributed by M. Newman (<https://websites.umich.edu/~mejn/netdata/>) | Free for research use; cite the paper above (see `football.txt`) |
| Ngogo chimpanzee proximity network, 2015 | `chimpanzees/A_2015.csv`, `chimpanzees/combined_membership_labels.csv` | Dryad, [doi:10.5061/dryad.sf7m0cgkg](https://doi.org/10.5061/dryad.sf7m0cgkg) — refs. [55, 56] of the paper | CC0 1.0 (Dryad) |

## Formats

**Football.** The GML `value` attribute of each team is its conference id (mapping in `football.txt` and in
`paper/common/datasets.py`). Two duplicated edges were removed upstream on 21 Sep 2014.

**Chimpanzees.**

- `A_2015.csv` is the 219 × 219 symmetric, non-negative, weighted adjacency matrix for 2015, one of the yearly
  matrices of the Dryad deposit (`learned_graphs_csv/A_<year>.csv`). Rows and columns follow the `Index` column of
  the labels file. Individuals absent that year have an all-zero row; the diagonal is ignored.
- `combined_membership_labels.csv` gives, for each `Index`, the identifier (`Name`) of the individual and its
  faction after the 2018 fission: `0` = Central group, `1` = Western group, `-1` = no label (not present in 2018).

To use another year, download the full deposit from Dryad and copy the corresponding `A_<year>.csv` next to
`A_2015.csv`; then call `load_chimpanzees(year)`.

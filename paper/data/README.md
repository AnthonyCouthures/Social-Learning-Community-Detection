# Real-world datasets

The three real-world networks of the paper come from the publications below. Please cite them if you use these data.
Zachary's karate club is loaded from NetworkX (`nx.karate_club_graph()`). The two other networks are bundled here,
unchanged.

| Dataset | Files | Reference | License / terms |
|---|---|---|---|
| Zachary's karate club | none (`networkx`) | Zachary (1977) | Public, widely redistributed |
| US college football, regular season 2000 | `football/football.gml`, `football/football.txt` | Girvan & Newman (2002), file distributed by M. Newman (<https://websites.umich.edu/~mejn/netdata/>) | Free for research use, with citation (see `football.txt`) |
| Ngogo chimpanzee proximity network, 2015 | `chimpanzees/A_2015.csv`, `chimpanzees/combined_membership_labels.csv` | Sandel et al. (2026), dataset and article; Dryad [doi:10.5061/dryad.sf7m0cgkg](https://doi.org/10.5061/dryad.sf7m0cgkg) | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) |

## How to cite

```bibtex
@article{zachary1977information,
  title   = {An Information Flow Model for Conflict and Fission in Small Groups},
  author  = {Zachary, Wayne W.},
  journal = {Journal of Anthropological Research},
  volume  = {33},
  number  = {4},
  pages   = {452--473},
  year    = {1977},
  doi     = {10.1086/jar.33.4.3629752}
}

@article{girvan2002community,
  title   = {Community Structure in Social and Biological Networks},
  author  = {Girvan, M. and Newman, M. E. J.},
  journal = {Proceedings of the National Academy of Sciences},
  volume  = {99},
  number  = {12},
  pages   = {7821--7826},
  year    = {2002},
  doi     = {10.1073/pnas.122653799}
}

@article{sandel2026lethal,
  title   = {Lethal Conflict after Group Fission in Wild Chimpanzees},
  author  = {Sandel, Aaron A. and He, Yixuan and Ren, Junpeng and Kei, Yik Lun and Lee, Kevin C. and
             Clark, Isabelle R. and Reddy, Rachna B. and Negrey, Jacob D. and Birungi, Charles and
             Apamaku, Blessing A. and Kanweri, Diana and Kalunga, Davis and Aliganyira, Christopher and
             Ram{\'i}rez-Amaya, Sebasti{\'a}n and Nakayima, Phionah and Katumba, Raymond and
             Kamugyisha, Brian and Acosta-Florez, Daniela and van Boekholt, Bas and Mbabazi, Godfrey and
             Akamumpa, Erone and Namaganda, Sharifah and Tumusiime, Alfred and Angedakin, Samuel and
             Reinert, Gesine and Madrid-Padilla, Oscar and Cucuringu, Mihai and Wipf, David and
             Langergraber, Kevin E. and Watts, David P. and Mitani, John C.},
  journal = {Science},
  volume  = {392},
  number  = {6794},
  pages   = {216--220},
  year    = {2026},
  doi     = {10.1126/science.adz4944}
}

@misc{sandel2026data,
  title        = {Data from: Lethal Conflict after Group Fission in Wild Chimpanzees},
  author       = {Sandel, Aaron and Mitani, John and Langergraber, Kevin and Clark, Isabelle R. and Lee, Kevin and
                  Ren, Junpeng and Kei, Yik Lun and He, Yixuan and Reddy, Rachna B. and Negrey, Jacob and
                  Birungi, Charles and Apamaku, Blessing A. and Kanweri, Diana and Wipf, David and Kalunga, Davis and
                  Nakayima, Phionah and Katumba, Raymond and Kamugyisha, Brian and Acosta-Florez, Daniela and
                  van Boekholt, Bas and Mbabazi, Godfrey and Akamumpa, Erone and Namaganda, Sharifah and
                  Aliganyira, Christopher and Tumusiime, Alfred and Angedakin, Samuel and Madrid-Padilla, Oscar and
                  Reinert, Gesine and Cucuringu, Mihai and Watts, David},
  publisher    = {Dryad},
  year         = {2026},
  doi          = {10.5061/dryad.sf7m0cgkg},
  note         = {Ngogo Chimpanzee Project; CC0 1.0}
}
```

The synthetic benchmarks are generated with ABCD (Kamiński, Prałat & Théberge, *Network Science* 9(2), 2021,
[doi:10.1017/nws.2020.45](https://doi.org/10.1017/nws.2020.45)), through the
[`abcd-graph`](https://pypi.org/project/abcd-graph/) package, and with LFR (Lancichinetti, Fortunato & Radicchi,
*Phys. Rev. E* 78, 046110, 2008, [doi:10.1103/PhysRevE.78.046110](https://doi.org/10.1103/PhysRevE.78.046110)),
through NetworkX.

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

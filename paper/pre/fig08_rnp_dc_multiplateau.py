"""Fig. 8 -- RNP-DC dynamics with a multi-plateau signal function.

(a) opinions x_i(t) of the decaying-confidence dynamics (B1) on an SBM,
(b) the three-plateau signal function.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the paper/ folder, for ``common``

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from common import parser
from common.benchmarks import sbm_graph
from common.plotting import save, set_style
from slcd import opinion_dynamics
from slcd.signals import staircase

# SBM of the notebook (4 blocks of 40, p_in = 0.80, p_out <= 0.02) and dynamics.
P_OUT = [[0, 0.02, 0.01, 0.01], [0.02, 0, 0.02, 0.01], [0.01, 0.02, 0, 0.02], [0.01, 0.01, 0.02, 0]]
K, H, T, R, RHO = 2.5, 0.1, 300.0, 4.0, 0.98
SIGNAL = staircase(n_levels=3, delta=0.01)


def main() -> None:
    args = parser(__doc__, seed=5).parse_args()  # seed of the published run
    set_style()
    G, blocks = sbm_graph([40] * 4, 0.80, P_OUT, seed=42)
    block_of = {v: b for b, nodes in enumerate(blocks) for v in nodes}

    traj = opinion_dynamics(G, K=K, h=H, T=T, signal=SIGNAL, confidence=(R, RHO), seed=args.seed, record=True)
    t = H * np.arange(traj.shape[1])
    fig, ax = plt.subplots(figsize=(5, 2.5))
    for row, v in zip(traj, G, strict=True):
        ax.plot(t[1:], row[1:], color=f"C{block_of[v]}", alpha=0.9, linewidth=1)
    ax.set_xscale("log")
    ax.set_xlim(t[1], T)
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"State $x_i(t)$")
    ax.grid(True)
    ax.legend(
        [Line2D([0], [0], color=f"C{b}", lw=1) for b in range(len(blocks))],
        [f"Community ${b + 1}$" for b in range(len(blocks))],
    )
    fig.tight_layout()
    save(fig, "fig08a_rnp_dc_trajectories")

    # (b) as in the paper, the unit staircase f is shown (the dynamics uses s(x) = f(Kx)).
    x = np.linspace(-1, 1, 2001)
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.plot(x, x, "--", color="grey", lw=0.8, label="$y = x$")
    ax.plot(x, SIGNAL(x), color="C0", lw=1, label="$s(x)$")
    for side in ("left", "bottom"):
        ax.spines[side].set_position("zero")
    for side in ("right", "top"):
        ax.spines[side].set_visible(False)
    ax.set_xticks([-1, 1])
    ax.set_yticks([-1, 1])
    ax.grid(True, alpha=0.5)
    ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, "fig08b_staircase_signal")


if __name__ == "__main__":
    main()

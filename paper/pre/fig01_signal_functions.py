"""Fig. 1 -- Examples of nonlinear interaction (signal) functions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the paper/ folder, for ``common``

import matplotlib.pyplot as plt
import numpy as np

from common import parser
from common.plotting import save, set_style


def main() -> None:
    parser(__doc__).parse_args()
    set_style()
    x = np.linspace(-1, 1, 1001)

    fig, ax = plt.subplots(figsize=(2.6, 2.4))
    ax.plot(x, x, "--", color="grey", lw=0.8, label="$y = x$")
    ax.plot(x, np.clip(4 * x, -1, 1), color="tab:blue", lw=1, label=r"$s(x) = \min(1, \max(-1, Kx))$")
    ax.plot(x, np.tanh(2.5 * x), color="tab:red", lw=1, label=r"$s(x) = \tanh(Kx)$")
    ax.plot([-1, 1], [-1, 1], "o", color="k", ms=4, zorder=5)
    ax.plot([0], [0], "o", mfc="white", mec="k", ms=4, zorder=5)
    ax.annotate("Stable", (1, 1), (0.55, 0.82), fontsize=6)
    ax.annotate("Unstable", (0, 0), (-0.55, 0.06), fontsize=6)

    for side in ("left", "bottom"):
        ax.spines[side].set_position("zero")
    for side in ("right", "top"):
        ax.spines[side].set_visible(False)
    ax.set_xticks([-1, 1])
    ax.set_yticks([-1, 1])
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.5)
    ax.text(1.02, 0.04, "$x$", fontsize=7)
    ax.text(0.04, 1.02, "$s(x)$", fontsize=7)
    ax.legend(loc="lower right", fontsize=5, framealpha=1)
    fig.tight_layout()
    save(fig, "fig01_signal_functions")


if __name__ == "__main__":
    main()

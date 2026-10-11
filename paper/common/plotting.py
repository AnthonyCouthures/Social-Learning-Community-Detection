"""Matplotlib style and helpers shared by the figure scripts."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.scale import ScaleBase, register_scale
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import Transform

FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"
DPI = 300


def use_latex() -> bool:
    """LaTeX rendering is used when available, unless ``SLCD_USETEX=0``."""
    flag = os.environ.get("SLCD_USETEX")
    if flag is not None:
        return flag not in ("0", "false", "False", "")
    return shutil.which("latex") is not None


def set_style(legend_fontsize: float = 7) -> None:
    """Apply the paper style (serif, small fonts, dashed grid)."""
    usetex = use_latex()
    mpl.rcParams.update(
        {
            "text.usetex": usetex,
            "text.latex.preamble": r"\usepackage{newtxtext}" if usetex else "",
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": legend_fontsize,
            "xtick.major.size": 3.5,
            "xtick.minor.size": 2.0,
            "xtick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "grid.linewidth": 0.8,
            "grid.color": "lightgrey",
            "grid.linestyle": (0, (6, 2)),
            "axes.linewidth": 1,
            "axes.labelpad": 4.0,
            "axes.axisbelow": True,
            "axes.xmargin": 0,
        }
    )


def save(fig: plt.Figure, name: str, **kwargs) -> Path:
    """Save ``fig`` as ``paper/figures/<name>.pdf``.

    In a Jupyter notebook the figure is also displayed; otherwise it is closed.
    Figures of ``--quick`` runs go to ``paper/figures/quick/``.
    A PNG preview is also written when the environment variable ``SLCD_PNG=1``.
    """
    # Smoke-test runs (--quick) must not overwrite the reference figures.
    out_dir = FIGURES_DIR / "quick" if "--quick" in sys.argv else FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.pdf"
    kwargs.setdefault("transparent", True)
    fig.savefig(path, dpi=DPI, **kwargs)
    if os.environ.get("SLCD_PNG") == "1":
        fig.savefig(path.with_suffix(".png"), dpi=150, **{**kwargs, "transparent": False})
    print(f"saved {path.relative_to(FIGURES_DIR.parents[1])}")
    if "inline" in mpl.get_backend():
        plt.show()
    else:
        plt.close(fig)
    return path


def plot_fill(ax, x, mean, std, color, label, style="o-", ymin=None, ymax=None, **kwargs):
    """Line for ``mean`` with a shaded ``mean +/- std`` band (clipped to [ymin, ymax])."""
    mean, std = np.asarray(mean, dtype=float), np.nan_to_num(np.asarray(std, dtype=float))
    lo, hi = mean - std, mean + std
    if ymin is not None:
        lo = np.clip(lo, ymin, None)
    if ymax is not None:
        hi = np.clip(hi, None, ymax)
    kwargs.setdefault("linewidth", 1)
    kwargs.setdefault("markersize", 2.5)
    ax.plot(x, mean, style, color=color, label=label, **kwargs)
    ax.fill_between(x, lo, hi, alpha=0.1, color=color)


# ---------------------------------------------------------------------------
# Hybrid y-scale: linear below 0, log1p(y)/2 above 0 (relative error plots)
# ---------------------------------------------------------------------------

_SOFTNESS = 2.0


def _forward(y):
    y = np.asarray(y, dtype=float)
    return np.where(y < 0, y, np.log1p(np.maximum(y, 0)) / _SOFTNESS)


def _inverse(v):
    v = np.asarray(v, dtype=float)
    return np.where(v < 0, v, np.expm1(np.maximum(v, 0) * _SOFTNESS))


class _HybridTransform(Transform):
    input_dims = output_dims = 1

    def __init__(self, inverse: bool = False):
        super().__init__()
        self._inv = inverse

    def transform_non_affine(self, values):
        return _inverse(values) if self._inv else _forward(values)

    def inverted(self):
        return _HybridTransform(not self._inv)


class CenteredHybridScale(ScaleBase):
    """Linear on the negative side, compressed logarithmic on the positive side."""

    name = "centered_hybrid"

    def __init__(self, axis, **kwargs):
        super().__init__(axis)

    def get_transform(self):
        return _HybridTransform()

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.2f}"))


register_scale(CenteredHybridScale)

# Colors / markers / names of the methods in the benchmark figures.
METHOD_STYLE = {
    "rnp": ("#1f77b4", "o-", "RNP"),
    "rnp_dc": ("#2ca02c", "^-", "RNP-DC"),
    "ser": ("#9467bd", "v-", "SER"),
    "louvain": ("#ff7f0e", "s--", "Louvain"),
    "label_propagation": ("#e377c2", "D--", "Label Propagation"),
    "spectral": ("#8c564b", "P--", "Spectral Clustering"),
    "leiden": ("#17becf", "X--", "Leiden"),
    "truth": ("k", "--", "Ground Truth"),
}

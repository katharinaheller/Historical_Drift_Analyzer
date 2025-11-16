"""
Unified visualization style for scientific evaluation plots.
Applies consistent layout, fonts, and color palette across all figures.
Follows principles from Tufte (1983), IEEE Vis Guidelines, and RSS Data Viz Guide (2023).
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def apply_scientific_style() -> None:
    """Configure Matplotlib for publication-grade scientific figures."""
    mpl.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.format": "svg",
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": mpl.cycler(
                color=[
                    "#1b9e77",
                    "#d95f02",
                    "#7570b3",
                    "#e7298a",
                    "#66a61e",
                    "#e6ab02",
                ]
            ),
            "figure.figsize": (6, 4),
            "figure.autolayout": True,
            "savefig.transparent": False,
        }
    )


def annotate_sample_info(
    ax: plt.Axes,
    n: int | None = None,
    k: int | None = None,
    bootstrap_iters: int | None = None,
    show_conf_int: tuple[float, float] | None = None,
) -> None:
    """Add standardized annotation text (sample info, parameters) inside a figure."""
    txt_parts = []
    if n is not None:
        txt_parts.append(f"n={n}")
    if k is not None:
        txt_parts.append(f"k={k}")
    if bootstrap_iters is not None:
        txt_parts.append(f"boot={bootstrap_iters}")
    if show_conf_int is not None:
        lo, hi = show_conf_int
        txt_parts.append(f"95% CI=[{lo:.3f}, {hi:.3f}]")

    if not txt_parts:
        return

    ax.text(
        0.98,
        0.02,
        ", ".join(txt_parts),
        ha="right",
        va="bottom",
        fontsize=8,
        color="gray",
        alpha=0.85,
        transform=ax.transAxes,
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.55,
            boxstyle="round,pad=0.2",
        ),
    )


def add_violin_overlay(ax: plt.Axes, data: np.ndarray, color: str = "#1b9e77") -> None:
    """
    Add a violin-style density overlay (no seaborn dependency).
    Uses kernel density estimation for smooth distribution visualization.
    """
    from scipy.stats import gaussian_kde

    if len(data) < 5:
        return
    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]
    if data.size == 0:
        return

    dmin, dmax = float(np.min(data)), float(np.max(data))
    if dmax - dmin < 1e-6:
        dmin -= 1e-3
        dmax += 1e-3

    kde = gaussian_kde(data)
    xs = np.linspace(dmin, dmax, 200)
    ys = kde(xs)
    ys = ys / ys.max() * 0.25

    ax.fill_betweenx(xs, -ys, ys, facecolor=color, alpha=0.18, linewidth=0)
    ax.plot(ys, xs, color=color, alpha=0.5, linewidth=0.6)
    ax.plot(-ys, xs, color=color, alpha=0.5, linewidth=0.6)

    cur_xlim = ax.get_xlim()
    pad = (cur_xlim[1] - cur_xlim[0]) * 0.05
    ax.set_xlim(cur_xlim[0] - pad, cur_xlim[1] + pad)

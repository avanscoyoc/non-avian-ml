"""Shared figure settings and helpers: model colours/markers (same in every figure), chart chrome, class display
names, saving, panel grids and the proportion matrix used by the cross-detection and confusion figures.

Model colours are from seaborn's "deep" palette, grouped by family: BirdNET blue/purple, Perch red/orange,
CNN14 green, ResNet brown. Each model always keeps its colour. Lines can cross, so every model also has its
own marker shape; identity never rests on colour alone, and figures always carry a legend.
"""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as mplcm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

MODELS = {  # plotting order -> (legend name, colour, marker)
    "birdnet": ("BirdNET 2.4", "#4c72b0", "o"),
    "birdnet3": ("BirdNET 3.0", "#8172b3", "s"),
    "perch": ("Perch v8", "#c44e52", "^"),
    "perch2": ("Perch 2.0", "#dd8452", "D"),
    "cnn14": ("CNN14 (AudioSet)", "#55a868", "v"),
    "resnet": ("ResNet-18 (ImageNet)", "#937860", "P"),
}

# Chart chrome (light surface)
INK = "#0b0b0b"          # titles, primary text
INK_SECONDARY = "#52514e"  # tick labels, secondary text (e.g. "low sample")
INK_MUTED = "#898781"    # reference lines
GRID = "#e1e0d9"         # hairline gridlines
AXIS = "#c3c2b7"         # spines / baseline

CLASS_COLORS = [mplcm.tab10(i) for i in range(10)] + [mplcm.tab20b(i) for i in range(0, 20, 4)]

PANEL_STYLE = {"font.family": "sans-serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
DIAGONAL_EDGE = "#e0475b"  # outline of the matrix diagonal (recall)
NO_DATA = "#eceff3"       # hatched matrix cells without test clips


def model_name(model: str) -> str:
    return MODELS.get(model, (model, None, None))[0]


def class_name(cls: str) -> str:
    return cls.replace("_", " ").capitalize()


def style_axes(ax):
    """Hairline grid, no top/right spines, muted ticks."""
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.tick_params(colors=INK_SECONDARY)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)


def panel_grid(n_panels: int, panel_w: float, panel_h: float):
    """Up to three panels per row (one per model); unused slots are hidden."""
    n_cols = min(3, n_panels)
    n_rows = math.ceil(n_panels / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(panel_w * n_cols, panel_h * n_rows), squeeze=False,
                             constrained_layout=True)
    axes = axes.flatten()
    for ax in axes[n_panels:]:
        ax.set_visible(False)
    return fig, axes


def save(fig, path, **savefig_kwargs):
    """Save as path.png and path.pdf (300 dpi, tight bounding box) and close the figure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"dpi": 300, "bbox_inches": "tight", **savefig_kwargs}
    fig.savefig(path.with_suffix(".png"), **kwargs)
    # No timestamp, so redrawing an unchanged figure leaves the file (and git) unchanged
    fig.savefig(path.with_suffix(".pdf"), metadata={"CreationDate": None}, **kwargs)
    plt.close(fig)
    print(f"Saved {path}.{{png,pdf}}")


def fraction(v: float) -> str:
    """0.92 -> '.92', 1.0 -> '1.00' (proportion, two decimals, no leading zero)."""
    return f"{v:.2f}".replace("0.", ".", 1) if v < 1 else f"{v:.2f}"


def draw_matrix(ax, values: np.ndarray, fontsize: float, diagonal_lw: float):
    """Square matrix of proportions (viridis, 0-1): two-decimal cell labels, hatched cells where there is no data
    (NaN) and the diagonal outlined. The caller sets ticks and labels. Returns the image for a colour bar."""
    n = len(values)
    im = ax.imshow(values, cmap="viridis", vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            if np.isnan(values[i, j]):
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=NO_DATA, edgecolor="#b8bec8",
                                       hatch="///", lw=0))
                continue
            ax.text(j, i, fraction(values[i, j]), ha="center", va="center", fontsize=fontsize,
                    color="#111111" if values[i, j] > 0.55 else "white")
        ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=DIAGONAL_EDGE, lw=diagonal_lw))
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#333333")
    return im

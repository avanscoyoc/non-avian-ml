"""
saturation_heatmap.py

One-figure summary of the saturation experiment (figs/saturation_summary.png/.pdf): classes x models,
cell colour and value = test AUC at the headline training size N = 40 (low-sample classes, which can't
reach 40, show their largest N, e.g. "@20"). Cells where the model has a built-in label for the class
("known") are outlined and show that label's zero-shot AUC ("zs 0.89"); the rest are new to the model.

Run scripts/saturation_summary.py first (it writes results/saturation_summary.csv).

Usage:
    pixi run python scripts/saturation_heatmap.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import FIGS, RESULTS, ROOT  # noqa: E402
from src.config import load_config  # noqa: E402
from src.plot_style import INK, INK_SECONDARY, MODELS, class_name, model_name  # noqa: E402

HEADLINE_N = 40
# Single-hue sequential ramp (validated reference palette, blue steps 100 -> 700): light = low AUC.
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
        "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
VMIN, VMAX = 0.5, 1.0


def cell_values(cfg) -> pd.DataFrame:
    """AUC at N = 40, or at the largest N a low-sample class reaches."""
    rows = []
    for c in cfg.species:
        f = cfg.results_path / f"results_{c}.csv"
        if not f.exists():
            continue
        r = pd.read_csv(f)
        for m in r.model.unique():
            t = r[(r.model == m) & (r.training_size > 0)]
            n = HEADLINE_N if HEADLINE_N in set(t.training_size) else int(t.training_size.max())
            rows.append({"class": c, "model": m, "n": n, "auc": float(t[t.training_size == n].test_auc_mean.iloc[0])})
    return pd.DataFrame(rows)


def main():
    cfg = load_config(ROOT / "src" / "config.yaml")
    summary = pd.read_csv(RESULTS / "saturation_summary.csv")
    cells = cell_values(cfg).merge(summary[["class", "model", "label_status", "zero_shot_auc", "low_sample"]],
                                   on=["class", "model"])
    models = [m for m in MODELS if m in set(cells.model)]
    low = set(cells[cells.low_sample]["class"])
    classes = [c for c in cfg.species if c in set(cells["class"]) and c not in low] + \
              [c for c in cfg.species if c in low]
    cmap = LinearSegmentedColormap.from_list("blue_ramp", RAMP)

    fig, ax = plt.subplots(figsize=(1.35 * len(models) + 2.6, 0.62 * len(classes) + 1.6))
    grid = np.full((len(classes), len(models)), np.nan)
    for _, row in cells.iterrows():
        grid[classes.index(row["class"]), models.index(row.model)] = row.auc
    im = ax.imshow(grid, cmap=cmap, vmin=VMIN, vmax=VMAX, aspect="auto")

    for _, row in cells.iterrows():
        i, j = classes.index(row["class"]), models.index(row.model)
        dark = (row.auc - VMIN) / (VMAX - VMIN) > 0.55
        ink = "white" if dark else INK
        label = f"{row.auc:.2f}" + (f" @{row.n}" if row.n != HEADLINE_N else "")
        known = row.label_status == "known"
        ax.text(j, i - (0.12 if known else 0), label, ha="center", va="center", fontsize=8.5, color=ink,
                fontweight="semibold")
        if known:
            ax.text(j, i + 0.24, f"zs {row.zero_shot_auc:.2f}", ha="center", va="center", fontsize=7, color=ink)
            ax.add_patch(Rectangle((j - 0.47, i - 0.47), 0.94, 0.94, fill=False, edgecolor=INK, linewidth=1.8))

    n_main = len(classes) - len(low)
    if low:
        ax.axhline(n_main - 0.5, color=INK, linewidth=1.5)
    ax.set_xticks(range(len(models)), [model_name(m).replace(" (", "\n(") for m in models], fontsize=8.5)
    ax.set_yticks(range(len(classes)), [class_name(c) + (" *" if c in low else "") for c in classes], fontsize=9)
    ax.tick_params(length=0)
    ax.xaxis.tick_top()
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label(f"Test ROC-AUC at N = {HEADLINE_N} clips per class", color=INK)
    cbar.outline.set_visible(False)
    fig.text(0.01, 0.01, f"Outlined = model has a built-in label for the class (zs = its zero-shot AUC); "
             f"* low sample (provisional), @N = largest N reached.", fontsize=8, color=INK_SECONDARY)
    FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"saturation_summary.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {FIGS / 'saturation_summary'}.{{png,pdf}}")


if __name__ == "__main__":
    main()

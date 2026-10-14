"""
figures.py

Figures for the multiclass experiment (experiment/multiclass_fit.py), one set per multiclass group in
src/config.yaml, with every model in its own panel:

  Figure 1  binary_vs_multiclass_auc_{group}.{png,pdf}
            Binary test AUC (training-size experiment at N = multiclass_train_size) vs multiclass
            one-vs-rest AUC, same test clips. Colour = class; dashed line y = x.
  Figure 2  confusion_matrices_{group}.{png,pdf}
            Rows = true class, columns = predicted class; each cell is the proportion of that class's test
            clips (counts pooled over the 10 draws), so rows sum to 1 and the outlined diagonal is recall.
            Same look as the binary cross-detection matrix (viridis, ".92" proportions). A class with no
            test clips gets a hatched row reading "no test clips".
  Figure 3  tsne_embeddings_{group}.{png,pdf}
            t-SNE of the test-clip embeddings. Stars = class centroids.

Prerequisites: main.py (binary results) and experiment/multiclass_fit.py.

Usage:
    pixi run python experiment/figures.py              # all figures
    pixi run python experiment/figures.py --fig 2      # one figure
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import FIGS, ROOT  # noqa: E402
from src.config import load_config  # noqa: E402
from src.plot_style import CLASS_COLORS, MODELS, class_name, model_name  # noqa: E402

STYLE = {"font.family": "sans-serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False}


def grid(n_panels: int, panel_w: float, panel_h: float):
    n_cols = min(3, n_panels)
    n_rows = math.ceil(n_panels / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(panel_w * n_cols, panel_h * n_rows), squeeze=False,
                             constrained_layout=True)
    axes = axes.flatten()
    for ax in axes[n_panels:]:
        ax.set_visible(False)
    return fig, axes


def save(fig, name: str):
    FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIGS / name}.{{png,pdf}}")


def available_models(exp: Path, group: str, prefix: str, suffix: str) -> list[str]:
    return [m for m in MODELS if (exp / f"{prefix}_{group}_{m}.{suffix}").exists()]


# ── Figure 1 ──────────────────────────────────────────────────────────────────

def fig_binary_vs_multiclass(cfg, exp: Path, group: str, classes: list[str]):
    n = cfg.multiclass_train_size
    binary = []
    for c in classes:
        path = cfg.results_path / f"results_{c}.csv"
        if path.exists():
            df = pd.read_csv(path)
            binary.append(df[df.training_size == n][["species", "model", "test_auc_mean"]])
    if not binary:
        print("  Figure 1: no binary results (run main.py first)")
        return
    binary = pd.concat(binary).rename(columns={"species": "class_name", "test_auc_mean": "binary_auc"})
    multi = pd.read_csv(exp / "multiclass_results.csv")
    multi = multi[multi.group == group][["model", "class_name", "ovr_auc_mean"]]
    both = binary.merge(multi, on=["model", "class_name"])
    models = [m for m in MODELS if m in set(both.model)]
    lo = math.floor(min(both.binary_auc.min(), both.ovr_auc_mean.min()) * 20) / 20

    with plt.rc_context(STYLE):
        fig, axes = grid(len(models), 3.4, 3.4)
        for ax, model in zip(axes, models):
            d = both[both.model == model]
            ax.plot([lo, 1], [lo, 1], "--", color="#bbbbbb", lw=0.9, zorder=0)
            for i, c in enumerate(classes):
                r = d[d.class_name == c]
                if len(r):
                    ax.scatter(r.binary_auc, r.ovr_auc_mean, color=CLASS_COLORS[i], s=40, zorder=3,
                               edgecolors="white", linewidths=0.5, label=class_name(c))
            ax.set_xlim(lo, 1.005); ax.set_ylim(lo, 1.005); ax.set_aspect("equal")
            ax.set_title(model_name(model), fontweight="bold")
            ax.set_xlabel(f"Binary AUC (N={n})"); ax.set_ylabel("Multiclass one-vs-rest AUC")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=min(len(labels), 4), frameon=False)
        save(fig, f"binary_vs_multiclass_auc_{group}")


# ── Figure 2 ──────────────────────────────────────────────────────────────────

DIAGONAL_EDGE = "#e0475b"  # outline for correct predictions (recall)
NO_DATA = "#eceff3"       # hatched cells without test clips


def _fraction(v: float) -> str:
    """0.92 -> '.92', 1.0 -> '1.00' (proportion, two decimals, no leading zero)."""
    return f"{v:.2f}".replace("0.", ".", 1) if v < 1 else f"{v:.2f}"


def _draw_cm(ax, cm: np.ndarray, classes: list[str], test_n: list[int], title: str, show_ylabels: bool):
    n = len(classes)
    rows = cm.sum(axis=1, keepdims=True)
    frac = np.divide(cm, rows, out=np.full(cm.shape, np.nan), where=rows > 0)
    im = ax.imshow(frac, cmap="viridis", vmin=0, vmax=1)
    for i in range(n):
        if rows[i, 0] == 0:  # no test clips: hatched row instead of a misleading row of zeros
            ax.add_patch(Rectangle((-0.5, i - 0.5), n, 1, facecolor=NO_DATA, edgecolor="#b8bec8", hatch="///", lw=0))
            ax.text((n - 1) / 2, i, "no test clips", ha="center", va="center", fontsize=7, color="#555555")
            continue
        for j in range(n):
            ax.text(j, i, _fraction(frac[i, j]), ha="center", va="center", fontsize=6.5,
                    color="#111111" if frac[i, j] > 0.55 else "white")
        ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=DIAGONAL_EDGE, lw=1.8))
    names = [c.replace("_", " ") for c in classes]
    ax.set_xticks(range(n), names, rotation=90, fontsize=7)
    ax.set_yticks(range(n), [f"{nm} (n={k})" for nm, k in zip(names, test_n)] if show_ylabels else [""] * n,
                  fontsize=7)
    ax.set_xlabel("Predicted class", fontsize=8)
    if show_ylabels:
        ax.set_ylabel("True class (test clips)", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#333333")
    return im


def fig_confusion_matrices(cfg, exp: Path, group: str, classes: list[str], test_n: list[int]):
    models = available_models(exp, group, "cm", "npy")
    with plt.rc_context(STYLE):
        fig, axes = grid(len(models), 3.6, 3.6)
        for k, (ax, model) in enumerate(zip(axes, models)):
            cm = np.load(exp / f"cm_{group}_{model}.npy").sum(axis=0)
            im = _draw_cm(ax, cm, classes, test_n, model_name(model), show_ylabels=k % 3 == 0)
        fig.colorbar(im, ax=axes[:len(models)].tolist(), shrink=0.6,
                     label="Proportion of a class's test clips (diagonal = recall), 10 draws pooled")
        save(fig, f"confusion_matrices_{group}")


# ── Figure 3 ──────────────────────────────────────────────────────────────────

def fig_tsne(cfg, exp: Path, group: str, classes: list[str], perplexity: int = 30, seed: int = 0):
    models = available_models(exp, group, "test_embeddings", "npz")
    with plt.rc_context(STYLE):
        fig, axes = grid(len(models), 3.4, 3.4)
        for ax, model in zip(axes, models):
            data = np.load(exp / f"test_embeddings_{group}_{model}.npz")
            embs, labels = data["embeddings"], data["labels"]
            print(f"  t-SNE {group}/{model} {embs.shape}")
            xy = TSNE(n_components=2, perplexity=min(perplexity, len(embs) - 1), random_state=seed,
                      max_iter=1000).fit_transform(embs)
            for i, c in enumerate(classes):
                pts = xy[labels == i]
                if len(pts):
                    ax.scatter(pts[:, 0], pts[:, 1], color=CLASS_COLORS[i], s=10, alpha=0.7, linewidths=0,
                               label=class_name(c))
                    ax.scatter(*pts.mean(axis=0), color=CLASS_COLORS[i], marker="*", s=140,
                               edgecolors="white", linewidths=0.8, zorder=5)
            ax.set_title(model_name(model), fontweight="bold")
            ax.set_xticks([]); ax.set_yticks([])
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=min(len(labels), 4), frameon=False,
                   markerscale=2, title="★ = class centroid")
        save(fig, f"tsne_embeddings_{group}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "src" / "config.yaml"))
    parser.add_argument("--fig", type=int, action="append", dest="figs", choices=[1, 2, 3],
                        help="Figure number(s) to generate (default: all)")
    args = parser.parse_args()
    figs = set(args.figs) if args.figs else {1, 2, 3}
    cfg = load_config(args.config)
    exp = cfg.results_path / "experiment"

    for group in cfg.multiclass_groups:
        labels_file = exp / f"labels_{group}.json"
        if not labels_file.exists():
            print(f"{group}: no multiclass results (run experiment/multiclass_fit.py first)")
            continue
        info = json.loads(labels_file.read_text())
        classes, test_n = info["classes"], info["test_n"]
        print(f"\n{group}: {len(classes)} classes")
        if 1 in figs:
            fig_binary_vs_multiclass(cfg, exp, group, classes)
        if 2 in figs:
            fig_confusion_matrices(cfg, exp, group, classes, test_n)
        if 3 in figs:
            fig_tsne(cfg, exp, group, classes)


if __name__ == "__main__":
    main()

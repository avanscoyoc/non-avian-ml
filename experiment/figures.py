"""
figures.py

Generates the three core figures from the binary-vs-multiclass experiment.

  Figure 1  binary_vs_multiclass_auc.{png,pdf}
            Scatter: binary AUC (x) vs multiclass one-vs-rest AUC (y) at N=80.
            Two subplots (anuran, nonbiotic). Axes auto-scaled per panel so the
            anuran cluster (near 1.0) is not compressed. Leader-line labels.

  Figure 2  confusion_matrices.{png,pdf}
            2×2 grid (group × model), counts summed over seeds, shared colourbar
            per row so axis labels don't collide.

  Figure 3  tsne_embeddings.{png,pdf}
            2×2 grid (group × model), t-SNE of test-set embeddings for both
            BirdNET and Perch. Stars = class centroids.

Prerequisites:
    python experiment/multiclass_fit.py      # Figures 1 & 2
    python experiment/extract_embeddings.py  # Figure 3

Usage:
    python experiment/figures.py [--fig 1] [--fig 2] [--fig 3]
    python experiment/figures.py                                # all three
"""

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import seaborn as sns
from sklearn.manifold import TSNE


ROOT      = Path(__file__).resolve().parent.parent
EXP_PATH  = ROOT / "results" / "experiment"
FIGS_PATH = ROOT / "figs"
BIN_PATH  = ROOT / "results"

ANURAN    = ["pacific_chorus_frog", "woodhouses_toad", "yellow_legged_frog", "american_bullfrog"]
NONBIOTIC = ["engine", "generator", "traffic", "device_static", "wind", "power_tools"]
GROUPS    = {"anuran": ANURAN, "nonbiotic": NONBIOTIC}
GROUP_TITLES = {"anuran": "Anuran group", "nonbiotic": "Non-biotic group"}

MODEL_COLOR  = {"birdnet": "#2E86AB", "perch": "#F18F01"}
MODEL_MARKER = {"birdnet": "o",       "perch": "s"}
MODEL_LABEL  = {"birdnet": "BirdNET", "perch": "Perch"}

DISPLAY = {
    "pacific_chorus_frog": "PCF",
    "woodhouses_toad":     "Woodhouse's toad",
    "yellow_legged_frog":  "Yellow-legged frog",
    "american_bullfrog":   "Bullfrog",
    "engine":              "Engine",
    "generator":           "Generator",
    "traffic":             "Traffic",
    "device_static":       "Device static",
    "wind":                "Wind",
    "power_tools":         "Power tools",
}

# tab10 gives 10 perceptually distinct colours (works for up to 6 classes)
CLASS_COLORS = [mplcm.tab10(i) for i in range(10)]

STYLE = {
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_binary_auc() -> pd.DataFrame:
    rows = []
    for sp in ANURAN + NONBIOTIC:
        path = BIN_PATH / f"results_{sp}.csv"
        if not path.exists():
            print(f"  [warn] missing binary results: {path.name}")
            continue
        df = pd.read_csv(path)
        for _, row in df[df.training_size == 80][["model", "test_auc_mean"]].iterrows():
            rows.append({"species": sp, "model": row["model"], "binary_auc": row["test_auc_mean"]})
    return pd.DataFrame(rows)


def load_multiclass_auc() -> pd.DataFrame:
    csvs = list(EXP_PATH.glob("run_multiclass_*.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"No multiclass run files in {EXP_PATH}. Run experiment/multiclass_fit.py first."
        )
    df = pd.concat(pd.read_csv(p) for p in csvs)
    return (
        df.groupby(["group", "model", "class_name"])["ovr_auc"]
        .agg(mean="mean", std="std")
        .reset_index()
        .rename(columns={"mean": "multi_auc_mean", "std": "multi_auc_std"})
    )


def load_confusion_matrices(group: str, model: str) -> tuple[np.ndarray, list[str]]:
    label_file = EXP_PATH / f"labels_{group}.json"
    if not label_file.exists():
        raise FileNotFoundError(f"Labels file not found: {label_file}")
    labels = json.loads(label_file.read_text())
    cms = [
        np.load(EXP_PATH / f"cm_multiclass_{group}_{model}_seed{s}.npy")
        for s in range(1, 11)
        if (EXP_PATH / f"cm_multiclass_{group}_{model}_seed{s}.npy").exists()
    ]
    if not cms:
        raise FileNotFoundError(f"No confusion matrices for {group}/{model}")
    return np.stack(cms).sum(axis=0), labels


# ── Figure 1 helpers ──────────────────────────────────────────────────────────

# Fixed, comparable axes for both panels
SCATTER_XLIM = (0.80, 1.003)
SCATTER_YLIM = (0.70, 1.003)

# ── Label positions — edit (text_x, text_y) in data coordinates to taste ──────
# x range: 0.80 – 1.00  |  y range: 0.70 – 1.00
LABEL_POS = {
    "anuran": {
        "pacific_chorus_frog": {"birdnet": (0.97, 0.92), "perch": (0.98, 0.97)},
        "woodhouses_toad":     {"birdnet": (0.99, 0.90), "perch": (0.95, 1.00)},
        "yellow_legged_frog":  {"birdnet": (0.94, 0.96), "perch": (0.93, 0.98)},
        "american_bullfrog":   {"birdnet": (0.97, 0.94), "perch": (0.98, 1.01)},
    },
    "nonbiotic": {
        "engine":              {"birdnet": (0.92, 0.75), "perch": (0.91, 0.96)},
        "generator":           {"birdnet": (0.96, 0.85), "perch": (1.00, 0.94)},
        "traffic":             {"birdnet": (0.92, 0.72), "perch": (0.88, 0.95)},
        "device_static":       {"birdnet": (0.90, 0.80), "perch": (0.93, 0.99)},
        "wind":                {"birdnet": (0.99, 0.90), "perch": (1.00, 0.97)},
        "power_tools":         {"birdnet": (0.98, 0.80), "perch": (0.97, 1.01)},
    },
}


# ── Figure 1 ──────────────────────────────────────────────────────────────────

def fig_scatter_auc():
    binary_df = load_binary_auc()
    multi_df  = load_multiclass_auc()
    models    = ["birdnet", "perch"]

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
        fig.subplots_adjust(wspace=0.38)

        for ax, (group, classes) in zip(axes, GROUPS.items()):
            ax.set_title(GROUP_TITLES[group], fontsize=10, fontweight="bold", pad=8)
            ax.set_xlabel("Binary AUC (N=80)", fontsize=9)
            ax.set_ylabel("Multiclass one-vs-rest AUC (N=80)", fontsize=9)

            # Fixed comparable axes
            xlim, ylim = SCATTER_XLIM, SCATTER_YLIM
            ax.set_xlim(xlim); ax.set_ylim(ylim)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))

            # Diagonal with label
            d_lo = max(xlim[0], ylim[0])
            d_hi = min(xlim[1], ylim[1])
            ax.plot([d_lo, d_hi], [d_lo, d_hi], "--", color="#bbbbbb", lw=0.9, zorder=0)
            mid = (d_lo + d_hi) / 2
            ax.text(mid, mid, "  y = x", fontsize=6.5, color="#999999",
                    rotation=45, rotation_mode="anchor", ha="left", va="bottom", zorder=1)

            # Scatter and label each dot
            any_data = False
            for model in models:
                for cls in classes:
                    br = binary_df[(binary_df.species == cls) & (binary_df.model == model)]
                    mr = multi_df[(multi_df.class_name == cls) & (multi_df.model == model)]
                    if br.empty or mr.empty:
                        continue
                    any_data = True
                    x = float(br.binary_auc.iloc[0])
                    y = float(mr.multi_auc_mean.iloc[0])

                    ax.scatter(x, y, color=MODEL_COLOR[model], marker=MODEL_MARKER[model],
                               s=55, zorder=4, linewidths=0.6, edgecolors="white")

                    tx, ty = LABEL_POS.get(group, {}).get(cls, {}).get(model, (x, y - 0.02))
                    ax.annotate(
                        DISPLAY.get(cls, cls), xy=(x, y), xytext=(tx, ty),
                        xycoords="data", textcoords="data",
                        fontsize=7, ha="center", va="center", color="#222222",
                        arrowprops=dict(arrowstyle="-", color="#bbbbbb", lw=0.5,
                                        shrinkA=0, shrinkB=3),
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.88),
                        zorder=6,
                    )

            if not any_data:
                ax.text(0.5, 0.5, "No data — run multiclass_fit.py first",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=8, color="gray")

            # Legend — boxed
            handles = [
                plt.Line2D([0], [0], marker=MODEL_MARKER[m], color=MODEL_COLOR[m],
                           linestyle="", markersize=6, label=MODEL_LABEL[m])
                for m in models
            ]
            ax.legend(handles=handles, fontsize=8, frameon=True,
                      edgecolor="#cccccc", facecolor="white", framealpha=0.9)

        FIGS_PATH.mkdir(parents=True, exist_ok=True)
        out = FIGS_PATH / "binary_vs_multiclass_auc"
        fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"Saved Figure 1 → {out}.{{png,pdf}}")


# ── Figure 2 ──────────────────────────────────────────────────────────────────

def _draw_cm(ax, cm: np.ndarray, label_names: list[str], title: str,
             show_cbar: bool = True) -> None:
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_pct   = cm / row_sums * 100
    display  = [DISPLAY.get(n, n.replace("_", " ").capitalize()) for n in label_names]

    annot = np.empty(cm.shape, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm_pct[i,j]:.0f}%\n({cm[i,j]})"

    sns.heatmap(
        cm_pct, annot=annot, fmt="", cmap="Blues", vmin=0, vmax=100,
        xticklabels=display, yticklabels=display,
        linewidths=0.4, linecolor="#e0e0e0",
        cbar=show_cbar,
        cbar_kws={"label": "Recall (%)", "shrink": 0.75, "pad": 0.02} if show_cbar else {},
        ax=ax, annot_kws={"size": 7, "linespacing": 1.2},
    )
    ax.set_title(title, fontsize=9, fontweight="bold", pad=6)
    ax.set_xlabel("Predicted", fontsize=8, labelpad=6)
    # Only draw y-label on left column (caller passes show_ylabel)
    ax.tick_params(axis="both", labelsize=7.5, length=0)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0)
    if show_cbar:
        ax.collections[0].colorbar.ax.tick_params(labelsize=7)


def fig_confusion_matrices():
    models = ["birdnet", "perch"]
    groups = list(GROUPS)

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(
            len(groups), len(models),
            figsize=(5.0 * len(models), 4.5 * len(groups)),
            constrained_layout=True,
        )

        for r, group in enumerate(groups):
            for c, model in enumerate(models):
                ax = axes[r][c]
                show_cbar = (c == len(models) - 1)   # colorbar only on right column
                try:
                    cm, labels = load_confusion_matrices(group, model)
                    _draw_cm(ax, cm, labels,
                             f"{GROUP_TITLES[group]} — {MODEL_LABEL[model]}",
                             show_cbar=show_cbar)
                    if c == 0:
                        ax.set_ylabel("True", fontsize=8, labelpad=8)
                    else:
                        ax.set_ylabel("")
                        ax.set_yticklabels([])
                except FileNotFoundError as e:
                    ax.text(0.5, 0.5, str(e), transform=ax.transAxes,
                            ha="center", va="center", fontsize=7, color="red")
                    ax.set_title(f"{GROUP_TITLES[group]} — {MODEL_LABEL[model]}", fontsize=9)

        FIGS_PATH.mkdir(parents=True, exist_ok=True)
        out = FIGS_PATH / "confusion_matrices"
        fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"Saved Figure 2 → {out}.{{png,pdf}}")


# ── Figure 3 ──────────────────────────────────────────────────────────────────

def _tsne_panel(ax, group: str, classes: list[str], model_name: str,
                perplexity: int, seed: int) -> None:
    npz_path = EXP_PATH / f"embeddings_{group}_{model_name}.npz"
    if not npz_path.exists():
        ax.text(0.5, 0.5,
                f"Missing:\n{npz_path.name}\nRun extract_embeddings.py",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=8, color="red")
        ax.set_title(f"{GROUP_TITLES[group]}\n{MODEL_LABEL[model_name]}")
        return

    data  = np.load(npz_path, allow_pickle=True)
    embs  = data["embeddings"]
    lbls  = data["labels"]
    names = list(data["class_names"])

    print(f"  t-SNE  {group}/{model_name}  {embs.shape} ...")
    coords = TSNE(
        n_components=2, perplexity=perplexity,
        random_state=seed, max_iter=1000,
    ).fit_transform(embs)

    for i, cls in enumerate(names):
        mask = lbls == i
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   color=CLASS_COLORS[i], label=DISPLAY.get(cls, cls),
                   s=18, alpha=0.72, linewidths=0)

    for i in range(len(names)):
        c = coords[lbls == i].mean(axis=0)
        ax.scatter(*c, color=CLASS_COLORS[i],
                   marker="*", s=160, edgecolors="white", linewidths=0.8, zorder=5)

    ax.set_title(f"{GROUP_TITLES[group]}\n{MODEL_LABEL[model_name]}",
                 fontsize=9, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=7.5, frameon=False, loc="best", markerscale=1.3,
              handletextpad=0.4, borderpad=0)

    # Centroid caption
    ax.text(0.98, 0.02, "★ = class centroid",
            transform=ax.transAxes, fontsize=6.5, color="#555555",
            ha="right", va="bottom")


def fig_tsne(perplexity: int = 30, seed: int = 0):
    models = ["birdnet", "perch"]
    groups = list(GROUPS)

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(
            len(groups), len(models),
            figsize=(5.5 * len(models), 5.0 * len(groups)),
            constrained_layout=True,
        )

        for r, (group, classes) in enumerate(GROUPS.items()):
            for c, model_name in enumerate(models):
                _tsne_panel(axes[r][c], group, classes, model_name, perplexity, seed)

        FIGS_PATH.mkdir(parents=True, exist_ok=True)
        out = FIGS_PATH / "tsne_embeddings"
        fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"Saved Figure 3 → {out}.{{png,pdf}}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fig", type=int, action="append", dest="figs",
                        choices=[1, 2, 3], metavar="N",
                        help="Figure number(s) to generate (default: all)")
    args = parser.parse_args()
    figs = set(args.figs) if args.figs else {1, 2, 3}

    if 1 in figs:
        print("\n── Figure 1: binary vs multiclass AUC scatter ──")
        fig_scatter_auc()
    if 2 in figs:
        print("\n── Figure 2: confusion matrices ──")
        fig_confusion_matrices()
    if 3 in figs:
        print("\n── Figure 3: t-SNE (BirdNET + Perch) ──")
        fig_tsne()

    print("\nAll done.")


if __name__ == "__main__":
    main()

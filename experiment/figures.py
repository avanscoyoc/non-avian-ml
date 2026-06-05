"""
figures.py

Generates the three core figures from the binary-vs-multiclass experiment.

  Figure 1  binary_vs_multiclass_auc.{png,pdf}
            Scatter: binary AUC (x) vs multiclass one-vs-rest AUC (y) at N=80.
            Two subplots (anuran, nonbiotic), two models (colour/marker).

  Figure 2  confusion_matrices.{png,pdf}
            2×2 grid of heatmaps (group × model), counts summed over seeds.

  Figure 3  tsne_perch.{png,pdf}
            t-SNE of Perch test-set embeddings, two subplots (anuran, nonbiotic).

Prerequisites (run first):
    python experiment/multiclass_fit.py      # required for Figures 1 & 2
    python experiment/extract_embeddings.py  # required for Figure 3

Usage:
    python experiment/figures.py [--fig 1] [--fig 2] [--fig 3]   # subset
    python experiment/figures.py                                   # all three
"""

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
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

# Nice display names for class labels in figures
DISPLAY = {
    "pacific_chorus_frog":  "PCF",
    "woodhouses_toad":      "Woodhouse's toad",
    "yellow_legged_frog":   "Yellow-legged frog",
    "american_bullfrog":    "Bullfrog",
    "engine":               "Engine",
    "generator":            "Generator",
    "traffic":              "Traffic",
    "device_static":        "Device static",
    "wind":                 "Wind",
    "power_tools":          "Power tools",
}

STYLE = {
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


# ── Data loading ─────────────────────────────────────────────────────────────

def load_binary_auc() -> pd.DataFrame:
    """Load binary (species-vs-not) AUC at N=80 for birdnet and perch."""
    rows = []
    all_classes = ANURAN + NONBIOTIC
    for sp in all_classes:
        path = BIN_PATH / f"results_{sp}.csv"
        if not path.exists():
            print(f"  [warn] missing binary results: {path.name}")
            continue
        df = pd.read_csv(path)
        df80 = df[df.training_size == 80][["model", "test_auc_mean"]]
        for _, row in df80.iterrows():
            rows.append({"species": sp, "model": row["model"], "binary_auc": row["test_auc_mean"]})
    return pd.DataFrame(rows)


def load_multiclass_auc() -> pd.DataFrame:
    """Load per-class OVR AUC averaged over seeds from experiment runs."""
    csvs = list(EXP_PATH.glob("run_multiclass_*.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"No multiclass run files found in {EXP_PATH}. "
            "Run experiment/multiclass_fit.py first."
        )
    df = pd.concat(pd.read_csv(p) for p in csvs)
    return (
        df.groupby(["group", "model", "class_name"])["ovr_auc"]
        .agg(mean="mean", std="std")
        .reset_index()
        .rename(columns={"mean": "multi_auc_mean", "std": "multi_auc_std"})
    )


def load_confusion_matrices(group: str, model: str) -> tuple[np.ndarray, list[str]]:
    """Sum raw confusion-matrix counts across all seeds."""
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


# ── Figure 1: scatter ─────────────────────────────────────────────────────────

def fig_scatter_auc():
    binary_df = load_binary_auc()
    multi_df  = load_multiclass_auc()
    models    = ["birdnet", "perch"]

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
        fig.subplots_adjust(wspace=0.35)

        for ax, (group, classes) in zip(axes, GROUPS.items()):
            ax.set_title(GROUP_TITLES[group], fontsize=10, fontweight="bold", pad=8)
            lim = [0.5, 1.01]
            ax.plot(lim, lim, "--", color="#aaaaaa", lw=0.9, zorder=0)
            ax.set_xlim(lim); ax.set_ylim(lim)
            ax.set_xlabel("Binary AUC (N=80)", fontsize=9)
            ax.set_ylabel("Multiclass one-vs-rest AUC (N=80)", fontsize=9)

            handles = []
            for model in models:
                for cls in classes:
                    bin_row  = binary_df[(binary_df.species == cls) & (binary_df.model == model)]
                    mult_row = multi_df[(multi_df.class_name == cls) & (multi_df.model == model)]
                    if bin_row.empty or mult_row.empty:
                        continue
                    x = float(bin_row.binary_auc.iloc[0])
                    y = float(mult_row.multi_auc_mean.iloc[0])
                    sc = ax.scatter(
                        x, y,
                        color=MODEL_COLOR[model],
                        marker=MODEL_MARKER[model],
                        s=55, zorder=3, linewidths=0.6, edgecolors="white",
                    )
                    label = DISPLAY.get(cls, cls)
                    ax.annotate(
                        label, (x, y),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=7, color="#333333",
                        path_effects=[pe.withStroke(linewidth=2, foreground="white")],
                    )
                # legend handle (one per model)
                handles.append(
                    plt.Line2D(
                        [0], [0],
                        marker=MODEL_MARKER[model], color=MODEL_COLOR[model],
                        linestyle="", markersize=6, label=MODEL_LABEL[model],
                    )
                )

            ax.legend(handles=handles[::len(classes)] if len(handles) > 2 else handles,
                      fontsize=8, frameon=False)

        FIGS_PATH.mkdir(parents=True, exist_ok=True)
        out = FIGS_PATH / "binary_vs_multiclass_auc"
        fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"Saved Figure 1 → {out}.{{png,pdf}}")


# ── Figure 2: confusion matrices ─────────────────────────────────────────────

def _draw_cm(ax, cm: np.ndarray, label_names: list[str], title: str):
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_pct   = cm / row_sums * 100
    display  = [DISPLAY.get(n, n.replace("_", " ").capitalize()) for n in label_names]

    annot = np.empty(cm.shape, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm_pct[i,j]:.0f}%\n(n={cm[i,j]})"

    sns.heatmap(
        cm_pct, annot=annot, fmt="", cmap="Blues", vmin=0, vmax=100,
        xticklabels=display, yticklabels=display,
        linewidths=0.4, linecolor="#e0e0e0",
        cbar_kws={"label": "Recall (%)", "shrink": 0.7},
        ax=ax, annot_kws={"size": 7, "linespacing": 1.2},
    )
    ax.set_title(title, fontsize=9, fontweight="bold", pad=6)
    ax.set_xlabel("Predicted", fontsize=8, labelpad=4)
    ax.set_ylabel("True", fontsize=8, labelpad=4)
    ax.tick_params(axis="both", labelsize=7.5, length=0)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0)
    ax.collections[0].colorbar.ax.tick_params(labelsize=7)


def fig_confusion_matrices():
    models = ["birdnet", "perch"]
    groups = list(GROUPS)

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(
            len(groups), len(models),
            figsize=(4.5 * len(models), 4.0 * len(groups)),
        )
        fig.subplots_adjust(hspace=0.45, wspace=0.4)

        for r, group in enumerate(groups):
            for c, model in enumerate(models):
                ax = axes[r][c]
                try:
                    cm, labels = load_confusion_matrices(group, model)
                    _draw_cm(ax, cm, labels, f"{GROUP_TITLES[group]}\n{MODEL_LABEL[model]}")
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


# ── Figure 3: t-SNE ──────────────────────────────────────────────────────────

# Qualitative palette (up to 5 classes)
CLASS_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]


def fig_tsne(perplexity: int = 30, seed: int = 0):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        fig.subplots_adjust(wspace=0.35)

        for ax, (group, classes) in zip(axes, GROUPS.items()):
            npz_path = EXP_PATH / f"embeddings_{group}_perch.npz"
            if not npz_path.exists():
                ax.text(0.5, 0.5,
                        f"Missing:\n{npz_path.name}\nRun extract_embeddings.py",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=8, color="red")
                ax.set_title(GROUP_TITLES[group])
                continue

            data  = np.load(npz_path, allow_pickle=True)
            embs  = data["embeddings"]
            lbls  = data["labels"]
            names = list(data["class_names"])

            print(f"  Running t-SNE for {group} ({embs.shape})...")
            coords = TSNE(
                n_components=2, perplexity=perplexity,
                random_state=seed, max_iter=1000,
            ).fit_transform(embs)

            for i, cls in enumerate(names):
                mask = lbls == i
                ax.scatter(
                    coords[mask, 0], coords[mask, 1],
                    color=CLASS_COLORS[i % len(CLASS_COLORS)],
                    label=DISPLAY.get(cls, cls),
                    s=20, alpha=0.75, linewidths=0,
                )

            # Overlay class centroids
            for i in range(len(names)):
                c = coords[lbls == i].mean(axis=0)
                ax.scatter(*c, color=CLASS_COLORS[i % len(CLASS_COLORS)],
                           marker="*", s=150, edgecolors="white", linewidths=0.8, zorder=5)

            ax.set_title(f"{GROUP_TITLES[group]}\n(Perch embeddings, t-SNE)",
                         fontsize=10, fontweight="bold")
            ax.set_xticks([]); ax.set_yticks([])
            ax.legend(fontsize=8, frameon=False, loc="best", markerscale=1.4)

        FIGS_PATH.mkdir(parents=True, exist_ok=True)
        out = FIGS_PATH / "tsne_perch"
        fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"Saved Figure 3 → {out}.{{png,pdf}}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fig", type=int, action="append", dest="figs",
        choices=[1, 2, 3], metavar="N",
        help="Figure number(s) to generate (default: all)",
    )
    args = parser.parse_args()
    figs = set(args.figs) if args.figs else {1, 2, 3}

    if 1 in figs:
        print("\n── Figure 1: binary vs multiclass AUC scatter ──")
        fig_scatter_auc()
    if 2 in figs:
        print("\n── Figure 2: confusion matrices ──")
        fig_confusion_matrices()
    if 3 in figs:
        print("\n── Figure 3: t-SNE (Perch embeddings) ──")
        fig_tsne()

    print("\nAll done.")


if __name__ == "__main__":
    main()

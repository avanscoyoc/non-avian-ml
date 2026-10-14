"""
training_size_plots.py

Saturation figures (test ROC-AUC vs training size, mean ± 95% CI over 10 random training draws):
  figs/saturation_main.png/.pdf    well-sampled classes (full design: 50 test clips, N up to 40-160)
  figs/saturation_low_sample.png/.pdf  low-sample classes (provisional: small test sets)
Zero-shot results (the model's own built-in label, no training) are open markers at N = 0; each panel
lists which models have a built-in label for the class ("known") - for the others the class is new.

Usage:
    pixi run python scripts/training_size_plots.py                 # both figures
    pixi run python scripts/training_size_plots.py --group main
    pixi run python scripts/training_size_plots.py --species engine wind --output figs/custom.png
"""
import argparse
import math
import textwrap
import sys
from math import log2
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import FIGS, RESULTS  # noqa: E402
from src.plot_style import AXIS, GRID, INK, INK_MUTED, INK_SECONDARY, MODELS, model_name  # noqa: E402
from src.zero_shot import load_label_map  # noqa: E402

BAND_ALPHA = 0.15  # 95% CI bands: up to six overlap per panel, so keep them light
# Training sizes double (5, 10, ..., 160), so x is plotted as log2(N): equal steps per doubling.
# Zero-shot (N = 0) has no log position; it sits one step left of the smallest size, past a separator.
ZS_GAP = 1.2

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 1.0,
    'axes.edgecolor': AXIS,
    'axes.labelcolor': INK,
    'xtick.color': INK_SECONDARY,
    'ytick.color': INK_SECONDARY,
    'grid.linewidth': 0.6,
    'grid.color': GRID,
    'lines.linewidth': 2,
    'lines.markersize': 6,
})

def class_groups(df) -> dict[str, list[str]]:
    """Well-sampled vs low-sample classes, in src/config.yaml order."""
    from src.config import load_config
    from src.paths import ROOT
    species = [c for c in load_config(ROOT / "src" / "config.yaml").species if c in set(df["species"])]
    low = set(df.loc[df["low_sample"].astype(bool), "species"]) if "low_sample" in df else set()
    return {"main": [c for c in species if c not in low], "low_sample": [c for c in species if c in low]}


def load_results(results_dir):
    files = sorted(Path(results_dir).glob("results_*.csv"))
    combined_df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()
    print(f"Loaded {len(combined_df)} rows from {len(files)} results files")
    return combined_df


def plot_species_models_publication(df, output_path, species_list=None, suptitle=None):
    """Create publication-quality plots with 95% confidence intervals."""
    label_map = load_label_map()
    if species_list is None:
        species_list = sorted(df["species"].unique())

    n_species = len(species_list)
    n_cols = 3 if n_species > 6 else 2
    n_rows = math.ceil(n_species / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes.flatten()

    for j in range(n_species, len(axes)):
        axes[j].set_visible(False)
        axes[j - n_cols].xaxis.set_tick_params(labelbottom=True)  # panel above an empty slot keeps its x labels

    handles, labels = {}, {}
    sizes = sorted(s for s in df["training_size"].unique() if s > 0)
    zs_x = log2(sizes[0]) - ZS_GAP

    for i, species in enumerate(species_list):
        ax = axes[i]
        species_data = df[df["species"] == species]
        models = [m for m in MODELS if m in set(species_data["model"])]

        for model in models:
            name, color, marker = MODELS[model]
            model_data = species_data[species_data["model"] == model].sort_values("training_size")
            trained = model_data[model_data["training_size"] > 0]
            x = [log2(n) for n in trained["training_size"]]
            y = trained["test_auc_mean"].to_numpy()
            ci = trained["test_auc_ci_95"].fillna(0).to_numpy()
            # 95% CI as a shaded band (a small box when a class has only one training size)
            band_x = x if len(x) > 1 else [x[0] - 0.2, x[0] + 0.2]
            band_lo, band_hi = (y - ci, y + ci) if len(x) > 1 else ([y[0] - ci[0]] * 2, [y[0] + ci[0]] * 2)
            ax.fill_between(band_x, band_lo, band_hi, color=color, alpha=BAND_ALPHA, linewidth=0, zorder=1)
            line, = ax.plot(x, y, color=color, marker=marker, markersize=6.5, linewidth=2,
                            markeredgecolor='white', markeredgewidth=1.0, label=name, zorder=3)
            handles.setdefault(model, line)
            zero = model_data[model_data["training_size"] == 0]
            if len(zero):  # zero-shot: open marker, same shape and colour
                ax.plot(zs_x, zero["test_auc_mean"].iloc[0], marker=marker, markersize=7.5, mfc='white',
                        mec=color, mew=1.8, linestyle='none', zorder=4)

        test_n = species_data["test_size_per_class"].iloc[0] if len(species_data) else None
        low = bool(species_data["low_sample"].iloc[0]) if "low_sample" in species_data and len(species_data) else False
        title = species.replace('_', ' ').title()
        if test_n is not None and test_n != 50:
            title += f" (test n={test_n})"
        knowers = [model_name(m).split(" (")[0] for m in MODELS if label_map.get(species, {}).get(m)]
        subtitle = ("built-in label: " + ", ".join(knowers)) if knowers else "no model has a built-in label"
        subtitle = textwrap.fill(subtitle, 44)
        ax.set_title(title, fontweight='semibold', pad=12 + 10 * subtitle.count("\n") + 10, color=INK)
        ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha='center', va='bottom', fontsize=8,
                color=INK_SECONDARY, linespacing=1.2)
        ax.grid(True)
        ax.set_axisbelow(True)
        ax.set_ylim(0.2, 1.02)
        ax.set_xlim(zs_x - 0.5, log2(sizes[-1]) + 0.4)
        ax.set_xticks([zs_x, *map(log2, sizes)], ["0", *map(str, sizes)])
        ax.axvline(zs_x + ZS_GAP / 2, color=AXIS, linewidth=1, zorder=0)  # separates zero-shot from the curve
        ax.axhline(y=0.5, color=INK_MUTED, linestyle=':', linewidth=1, zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.supxlabel("Training size (clips per class, log scale); shaded = 95% CI over 10 random draws; open markers = zero-shot",
                  fontweight='medium', fontsize=12, y=0.0)
    fig.supylabel("Test ROC-AUC", fontweight='medium', fontsize=12, x=0.02)

    order = [m for m in MODELS if m in handles]
    fig.legend([handles[m] for m in order], [MODELS[m][0] for m in order],
               loc='upper center',
               bbox_to_anchor=(0.5, -0.03),
               ncol=3,
               frameon=True,
               fancybox=False,
               edgecolor=AXIS,
               framealpha=0.95,
               fontsize=10)

    if suptitle:
        fig.suptitle(suptitle, fontweight="semibold", fontsize=13, y=0.95)
    plt.subplots_adjust(hspace=0.45, wspace=0.1, bottom=0.08 if n_rows > 1 else 0.14)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        plt.savefig(output_path.with_suffix(ext), dpi=300, bbox_inches='tight', pad_inches=0.3, facecolor='white')
    print(f"Plot saved to {output_path.with_suffix('')}.{{png,pdf}}")
    plt.close()


SUPTITLES = {"main": None,
             "low_sample": "Low-sample classes (provisional: test sets of 3–30 clips from 1–13 recordings)"}


def main():
    parser = argparse.ArgumentParser(description="Plot training-size saturation figures from results CSVs.")
    parser.add_argument("--group", choices=["main", "low_sample", "all"], default="all",
                        help="main = well-sampled classes, low_sample = provisional classes (default: both).")
    parser.add_argument("--species", nargs="+", help="Custom species list (overrides --group).")
    parser.add_argument("--output", help="Output path for a custom --species plot.")
    parser.add_argument("--results-dir", default=str(RESULTS), help="Directory containing results_*.csv files.")
    args = parser.parse_args()

    df = load_results(args.results_dir)
    if args.species:
        plot_species_models_publication(df, args.output or FIGS / "train_size_curves.png", species_list=args.species)
        return
    groups = class_groups(df)
    for group in (groups if args.group == "all" else [args.group]):
        if groups[group]:
            plot_species_models_publication(df, FIGS / f"saturation_{group}.png", species_list=groups[group],
                                            suptitle=SUPTITLES[group])


if __name__ == "__main__":
    main()

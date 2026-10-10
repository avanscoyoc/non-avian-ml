"""
training_size_plots.py

Saturation figures (test ROC-AUC vs training size, mean ± 95% CI over seeds) for the manuscript
groups. Zero-shot results (the model's own labels, no training) are open markers at N = 0.

Usage:
    pixi run python scripts/training_size_plots.py                 # all three groups
    pixi run python scripts/training_size_plots.py --group pulsed
    pixi run python scripts/training_size_plots.py --species engine wind --output figs/custom.png
"""
import argparse
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import FIGS, RESULTS  # noqa: E402

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
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.5,
    'lines.linewidth': 2,
    'lines.markersize': 6,
})

MODELS = {  # plotting order, legend name, colour
    'birdnet': ('BirdNET 2.4', '#2E86AB'),
    'birdnet3': ('BirdNET 3.0', '#1B4F72'),
    'perch': ('Perch v8', '#F18F01'),
    'perch2': ('Perch 2.0', '#C73E1D'),
    'cnn14': ('CNN14 (AudioSet)', '#A23B72'),
    'resnet': ('ResNet-18 (ImageNet)', '#06A77D'),
}

GROUPS = {  # manuscript figures
    'broadband': ["device_static", "engine", "generator", "power_tools", "traffic", "wind"],
    'pulsed': ["american_bullfrog", "pacific_chorus_frog", "woodhouses_toad", "yellow_legged_frog"],
    'variable': ["coyote", "nutria", "human_vocal"],
}


def load_results(results_dir):
    files = sorted(Path(results_dir).glob("results_*.csv"))
    combined_df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()
    print(f"Loaded {len(combined_df)} rows from {len(files)} results files")
    return combined_df


def plot_species_models_publication(df, output_path, species_list=None):
    """Create publication-quality plots with 95% confidence intervals."""
    if species_list is None:
        species_list = sorted(df["species"].unique())

    n_species = len(species_list)
    n_cols = 2
    n_rows = math.ceil(n_species / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes.flatten()

    for j in range(n_species, len(axes)):
        axes[j].set_visible(False)

    handles, labels = {}, {}

    for i, species in enumerate(species_list):
        ax = axes[i]
        species_data = df[df["species"] == species]
        models = [m for m in MODELS if m in set(species_data["model"])]

        for model in models:
            name, color = MODELS[model]
            model_data = species_data[species_data["model"] == model].sort_values("training_size")
            trained = model_data[model_data["training_size"] > 0]
            line = ax.errorbar(
                trained["training_size"],
                trained["test_auc_mean"],
                yerr=trained["test_auc_ci_95"],
                label=name,
                marker='o',
                color=color,
                capsize=4,
                capthick=1.5,
                elinewidth=1.5,
                alpha=0.9,
                markeredgewidth=0.5,
                markeredgecolor='white',
            )
            handles.setdefault(model, line)
            zero = model_data[model_data["training_size"] == 0]
            if len(zero):
                ax.plot(0, zero["test_auc_mean"].iloc[0], marker='o', markersize=7, mfc='none',
                        mec=color, mew=1.5, linestyle='none', alpha=0.9)

        test_n = species_data["test_size_per_class"].iloc[0] if len(species_data) else None
        title = species.replace('_', ' ').title()
        if test_n is not None and test_n != 50:
            title += f" (test n={test_n}/class)"
        ax.set_title(title, fontweight='semibold', pad=10)
        ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_ylim(0.2, 1.02)
        ax.set_xlim(-8, 175)
        ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5, zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.supxlabel("Training size (clips per class); open markers = zero-shot", fontweight='medium', fontsize=12, y=0.0)
    fig.supylabel("Test ROC-AUC", fontweight='medium', fontsize=12, x=0.02)

    order = [m for m in MODELS if m in handles]
    fig.legend([handles[m] for m in order], [MODELS[m][0] for m in order],
               loc='upper center',
               bbox_to_anchor=(0.5, -0.03),
               ncol=3,
               frameon=True,
               fancybox=False,
               edgecolor='gray',
               framealpha=0.95,
               fontsize=10)

    plt.subplots_adjust(hspace=0.3, wspace=0.1, bottom=0.08 if n_rows > 1 else 0.14)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.3, facecolor='white')
    print(f"Plot saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot training-size saturation figures from results CSVs.")
    parser.add_argument("--group", choices=[*GROUPS, "all"], default="all",
                        help="Manuscript species group to plot (default: all three).")
    parser.add_argument("--species", nargs="+", help="Custom species list (overrides --group).")
    parser.add_argument("--output", help="Output path for a custom --species plot.")
    parser.add_argument("--results-dir", default=str(RESULTS), help="Directory containing results_*.csv files.")
    args = parser.parse_args()

    df = load_results(args.results_dir)
    if args.species:
        plot_species_models_publication(df, args.output or FIGS / "train_size_curves.png", species_list=args.species)
        return
    for group in (GROUPS if args.group == "all" else [args.group]):
        plot_species_models_publication(df, FIGS / f"{group}.png", species_list=GROUPS[group])


if __name__ == "__main__":
    main()

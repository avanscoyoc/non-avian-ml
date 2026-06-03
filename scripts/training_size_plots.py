import argparse
import glob
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

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

COLORS = {
    'birdnet': '#2E86AB',
    'mobilenet': '#A23B72',
    'perch': '#F18F01',
    'resnet': '#06A77D',
    'vgg': '#C73E1D',
}


def load_results(results_dir):
    combined_df = pd.DataFrame()
    for file in glob.glob(f"{results_dir}/results_*.csv"):
        df = pd.read_csv(file)
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    print(f"Loaded {len(combined_df)} rows from results files")
    return combined_df


def plot_species_models_publication(df, output_path, species_list=None):
    """Create publication-quality plots with 95% confidence intervals."""
    if species_list is None:
        species_list = sorted(df["species"].unique())

    n_species = len(species_list)
    n_cols = 2
    n_rows = math.ceil(n_species / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows),
                             sharex=True, sharey=True)
    axes = axes.flatten()

    for j in range(n_species, len(axes)):
        axes[j].set_visible(False)

    handles, labels = None, None

    for i, species in enumerate(species_list):
        ax = axes[i]
        species_data = df[df["species"] == species]
        models = sorted(species_data["model"].unique())

        for model in models:
            model_data = species_data[species_data["model"] == model].sort_values("training_size")
            ax.errorbar(
                model_data["training_size"],
                model_data["test_auc_mean"],
                yerr=model_data["test_auc_ci_95"],
                label=model.upper(),
                marker='o',
                color=COLORS.get(model, '#333333'),
                capsize=4,
                capthick=1.5,
                elinewidth=1.5,
                alpha=0.9,
                markeredgewidth=0.5,
                markeredgecolor='white',
            )

        if i == 0:
            handles, labels = ax.get_legend_handles_labels()

        ax.set_title(species.replace('_', ' ').title(), fontweight='semibold', pad=10)
        ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_ylim(0.2, 1.02)
        ax.set_xlim(-5, 175)
        ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5, zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.supxlabel("Training Size (samples per class)", fontweight='medium', fontsize=12, y=0.13)
    fig.supylabel("Test ROC-AUC", fontweight='medium', fontsize=12, x=0.02)

    fig.legend(handles, labels,
               loc='lower center',
               ncol=len(labels) if labels else 5,
               frameon=True,
               fancybox=False,
               edgecolor='gray',
               framealpha=0.95,
               fontsize=11,
               bbox_to_anchor=(0.5, 0.04))

    plt.subplots_adjust(hspace=0.2, wspace=0.1, bottom=0.22)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.3, facecolor='white')
    print(f"Plot saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot model comparison graphs from results CSVs.")
    parser.add_argument(
        "--species", nargs="+",
        default=["engine", "traffic", "generator", "power_tools", "device_static", "wind"],
        help="Species to include in the plot (space-separated). Defaults to anthropogenic set.",
    )
    parser.add_argument(
        "--output", default="../figs/train_size_curves.png",
        help="Output file path for the saved figure.",
    )
    parser.add_argument(
        "--results-dir", default="../results",
        help="Directory containing results_*.csv files.",
    )
    args = parser.parse_args()

    df = load_results(args.results_dir)
    plot_species_models_publication(df, args.output, species_list=args.species)


if __name__ == "__main__":
    main()

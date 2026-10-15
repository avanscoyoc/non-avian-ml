from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_runs(rows, runs_dir, species, model):
    """Save all runs (every training size x seed, plus zero-shot) for one class x model."""
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    output_file = runs_dir / f"{species}_{model}.csv"
    pd.DataFrame(rows).to_csv(output_file, index=False)
    return output_file


def aggregate_results(runs_dir, species):
    """Load every model's run CSV for a species."""
    run_files = sorted(Path(runs_dir).glob(f"{species}_*.csv"))
    if not run_files:
        return None
    return pd.concat([pd.read_csv(f) for f in run_files], ignore_index=True)


def save_predictions(rows, preds_dir, species, model):
    """Every test clip's score for every training size x seed (and zero-shot), for later bootstraps."""
    preds_dir = Path(preds_dir)
    preds_dir.mkdir(parents=True, exist_ok=True)
    output_file = preds_dir / f"{species}_{model}.csv"
    pd.DataFrame(rows).to_csv(output_file, index=False)
    return output_file


def save_results(results, output_path):
    """Aggregate test AUC across the random training draws (seeds) and save to CSV."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    if "low_sample" not in df:
        df["low_sample"] = False
    agg_results = (
        df.groupby(["model", "species", "training_size", "n_epochs", "batch_size", "test_size_per_class", "low_sample"])
        ["test_auc"].agg(["mean", "std", "count"]).round(4)
    )
    agg_results.columns = ["test_auc_mean", "test_auc_std", "test_auc_n"]
    agg_results = agg_results.reset_index()

    # 95% CI across seeds: 1.96 * std / sqrt(n)
    agg_results["test_auc_ci_95"] = (
        1.96 * agg_results["test_auc_std"] / np.sqrt(agg_results["test_auc_n"])
    ).round(4)

    column_order = [
        "model", "species", "training_size", "n_epochs", "batch_size", "test_size_per_class", "low_sample",
        "test_auc_mean", "test_auc_std", "test_auc_ci_95", "test_auc_n",
    ]
    agg_results = agg_results[column_order]
    agg_results.to_csv(output_path, index=False)
    return agg_results


def plot_species_models(df, output_path):
    """Per-class learning curves: mean test AUC with a shaded 95% CI band; zero-shot (N=0) as open markers."""
    from src.plot_style import AXIS, GRID, INK, INK_MUTED, INK_SECONDARY, MODELS

    species_list = df["species"].unique()
    n_species = len(species_list)
    fig, axes = plt.subplots(1, n_species, figsize=(6 * n_species, 5), squeeze=False)
    axes = axes[0]

    for ax, species in zip(axes, species_list):
        species_data = df[df["species"] == species]
        for model in [m for m in MODELS if m in set(species_data["model"])]:
            name, color, marker = MODELS[model]
            d = species_data[species_data["model"] == model].sort_values("training_size")
            trained = d[d["training_size"] > 0]
            x, y = trained["training_size"].to_numpy(), trained["test_auc_mean"].to_numpy()
            ci = trained["test_auc_ci_95"].fillna(0).to_numpy()
            band_x = x if len(x) > 1 else [x[0] - 2.5, x[0] + 2.5]
            band_lo, band_hi = (y - ci, y + ci) if len(x) > 1 else ([y[0] - ci[0]] * 2, [y[0] + ci[0]] * 2)
            ax.fill_between(band_x, band_lo, band_hi, color=color, alpha=0.15, linewidth=0, zorder=1)
            ax.plot(x, y, color=color, marker=marker, markersize=6.5, linewidth=2, markeredgecolor="white",
                    markeredgewidth=1.0, label=name, zorder=3)
            zero = d[d["training_size"] == 0]
            if len(zero):
                ax.plot(0, zero["test_auc_mean"].iloc[0], marker=marker, markersize=7.5, mfc="white", mec=color,
                        mew=1.8, linestyle="none", zorder=4)

        ax.set_title(species.replace("_", " ").title(), color=INK)
        ax.set_xlabel("Training size (clips per class); shaded = 95% CI; open = zero-shot", color=INK)
        ax.set_ylabel("Test ROC-AUC", color=INK)
        ax.axhline(0.5, color=INK_MUTED, linestyle=":", linewidth=1, zorder=0)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK_SECONDARY)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.legend(frameon=False, fontsize=9)
        ax.set_ylim(0.2, 1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

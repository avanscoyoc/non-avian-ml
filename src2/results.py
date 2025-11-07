import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_species_models(df, output_path):
    """Plot learning curves showing test set performance."""
    species_list = df["species"].unique()
    n_species = len(species_list)
    fig, axes = plt.subplots(1, n_species, figsize=(6 * n_species, 6))
    if n_species == 1:
        axes = [axes]

    for i, species in enumerate(species_list):
        species_data = df[df["species"] == species]
        models = species_data["model"].unique()

        for model in models:
            model_data = species_data[species_data["model"] == model]

            axes[i].errorbar(
                model_data["training_size"],
                model_data["test_auc_mean"],
                yerr=model_data["test_auc_std"],
                label=model.upper(),
                marker="o",
                capsize=5,
            )

        axes[i].set_title(f"{species.replace('_', ' ').title()}")
        axes[i].set_xlabel("Training Size")
        axes[i].set_ylabel("Test ROC-AUC")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_results(results, output_path):
    """Aggregate results across random seeds and save."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)

    # Aggregate across random seeds
    agg_results = (
        df.groupby(["model", "species", "training_size"])
        .agg(
            {
                "cv_auc_mean": ["mean", "std"],
                "test_auc": ["mean", "std"],
            }
        )
        .round(4)
    )

    # Flatten column names
    agg_results.columns = [
        "cv_auc_mean",
        "cv_auc_std",
        "test_auc_mean",
        "test_auc_std",
    ]
    agg_results = agg_results.reset_index()
    agg_results.to_csv(output_path, index=False)

    return agg_results

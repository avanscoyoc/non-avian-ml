import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_species_models(df, output_path):
    species_list = df["species"].unique()
    fig, axes = plt.subplots(1, len(species_list), figsize=(6 * len(species_list), 6))
    if len(species_list) == 1:
        axes = [axes]

    for i, species in enumerate(species_list):
        species_data = df[df["species"] == species]
        models = species_data["model"].unique()

        for model in models:
            model_data = species_data[species_data["model"] == model]

            axes[i].errorbar(
                model_data["training_size"],
                model_data["mean_auc"],
                yerr=model_data["std_auc"],
                label=model.upper(),
                marker="o",
                capsize=5,
            )

        axes[i].set_title(f"{species.replace('_', ' ').title()}")
        axes[i].set_xlabel("Training Size")
        axes[i].set_ylabel("ROC-AUC")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_results(results, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    import numpy as np

    df = pd.DataFrame(results)

    # Calculate mean_auc from fold_scores if not present
    if "mean_auc" not in df.columns and "fold_scores" in df.columns:
        df["mean_auc"] = df["fold_scores"].apply(
            lambda x: np.mean(x) if isinstance(x, list) else x
        )

    # If we have multiple random_seeds, aggregate across them
    if "random_seed" in df.columns:
        agg_results = (
            df.groupby(["model", "species", "training_size"])
            .agg({"mean_auc": ["mean", "std"]})
            .round(4)
        )

        # Flatten column names
        agg_results.columns = ["mean_auc", "std_auc"]
        agg_results = agg_results.reset_index()
        agg_results.to_csv(output_path, index=False)
        return agg_results
    else:
        # Calculate std_auc from fold_scores if not present
        if "std_auc" not in df.columns and "fold_scores" in df.columns:
            df["std_auc"] = df["fold_scores"].apply(
                lambda x: np.std(x) if isinstance(x, list) else 0
            )

        df.to_csv(output_path, index=False)
        return df
    return df

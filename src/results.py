import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def save_run_result(result, results_path):
    """Save single run result immediately to CSV."""
    output_file = Path(results_path) / f"run_{result['species']}_{result['model']}_{result['training_size']}_{result['random_seed']}.csv"
    pd.DataFrame([result]).to_csv(output_file, index=False)


def aggregate_results(results_path, species):
    """Load all run CSVs for a species and aggregate statistics."""
    run_files = sorted(Path(results_path).glob(f"run_{species}_*.csv"))
    if not run_files:
        return None
    
    all_runs = pd.concat([pd.read_csv(f) for f in run_files], ignore_index=True)
    return all_runs


def save_results(results, output_path):
    """Aggregate results across random seeds and save to CSV."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)

    # Calculate aggregated statistics
    agg_results = df.groupby([
        "model", "species", "training_size", 
        "n_folds", "n_epochs", "batch_size", "test_size_per_class"
    ]).agg({
        "cv_auc_mean": ["mean", "std", "count"],
        "test_auc": ["mean", "std", "count"],
    }).round(4)

    # Flatten column names
    agg_results.columns = [
        "cv_auc_mean", "cv_auc_std", "cv_auc_n",
        "test_auc_mean", "test_auc_std", "test_auc_n",
    ]
    agg_results = agg_results.reset_index()
    
    # Calculate 95% CI: 1.96 * std / sqrt(n)
    agg_results["cv_auc_ci_95"] = (
        1.96 * agg_results["cv_auc_std"] / np.sqrt(agg_results["cv_auc_n"])
    ).round(4)
    agg_results["test_auc_ci_95"] = (
        1.96 * agg_results["test_auc_std"] / np.sqrt(agg_results["test_auc_n"])
    ).round(4)
    
    # Reorder columns for better readability
    column_order = [
        "model", "species", "training_size", 
        "n_folds", "n_epochs", "batch_size", "test_size_per_class",
        "cv_auc_mean", "cv_auc_std", "cv_auc_ci_95", "cv_auc_n",
        "test_auc_mean", "test_auc_std", "test_auc_ci_95", "test_auc_n",
    ]
    agg_results = agg_results[column_order]
    
    agg_results.to_csv(output_path, index=False)
    return agg_results


def plot_species_models(df, output_path):
    """Plot learning curves with 95% confidence intervals."""
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
                yerr=model_data["test_auc_ci_95"],
                label=model.upper(),
                marker="o",
                capsize=5,
                capthick=2,
            )

        axes[i].set_title(f"{species.replace('_', ' ').title()}")
        axes[i].set_xlabel("Training Size (samples per class)")
        axes[i].set_ylabel("Test ROC-AUC")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
